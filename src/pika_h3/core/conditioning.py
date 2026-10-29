# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Prompt + images/references -> the wrapper's ``conditional_dict`` (+ target geometry), for inference.

Uses the Diffusers pipeline primitives: Qwen3-VL hidden state 50 of the task presentation (the presentation helpers), keyframe /
reference latents through ``encode_vae_condition`` (posterior sample under seed 42, fp16 rounding, normalized),
soundtrack references through the audio VAE posterior mode. Components are loaded in stages so a single 80 GB GPU
can hold them one at a time: text encoder (63 GB bf16) -> VAEs (fp32, ~11 GB).

Keyframe geometry follows ``MiniMaxH3ResizeStep``: the first keyframe is stretched onto the canvas (whose aspect
it defines unless ``height``/``width`` are given), the last keyframe is cover-cropped. Bucket inference selects
``keyframe_fit="crop"`` to center-crop both source keyframes before resizing. Reference images follow
``MiniMaxH3Ref2VASetupStep`` (short edge 2048, upscaling, no cap) with ``reference_rule="pipeline"``; the tier rule
(``reference_rule="tiered"``, the default: the canvas tier's ``reference_short_edge`` from the packaged bucket rules — 2048 at
p768 and above, 1024 below — upscaling, no cap); or an explicit
``reference_short_edge``. The sizes actually used are returned on the condition dict as ``reference_images``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from . import geometry as G
from .packing import audio_storage_to_external, video_storage_to_external
from .pipeline_geometry import reference_canvas_size
from .presentation import build_presentation, vision_features

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
KEYFRAME_ENCODE_SEED = 42
TEXT_ENCODER_LAYER = 50
TASKS = ("t2va", "i2va", "fl2va", "l2va", "ref2va")


@dataclass
class Reference:
    kind: str  # image | video | audio
    image: Optional[Image.Image] = None
    audio: Optional[np.ndarray] = None  # [2, S] float32 at 32 kHz
    sample_rate: int = G.AUDIO_SAMPLE_RATE
    frames: Optional[np.ndarray] = None
    fps: float = G.FPS

    @property
    def has_audio(self):
        return self.audio is not None

    @staticmethod
    def from_spec(spec: str) -> "Reference":
        """Local image/video/audio paths; soundtracks are resampled to 32 kHz stereo."""
        kind, _, path = spec.partition(":")
        kind = kind.strip().lower()
        if kind == "image":
            return Reference(kind="image", image=Image.open(path).convert("RGB"))
        if kind == "audio":
            from .media import decode_audio_stereo

            wave = decode_audio_stereo(Path(path), sampling_rate=G.AUDIO_SAMPLE_RATE)
            if wave is None:
                raise ValueError(f"no audio stream in {path}")
            return Reference(kind="audio", audio=wave.astype(np.float32))
        if kind == "video":
            from .media import decode_video_reference
            frames, fps, wave = decode_video_reference(Path(path))
            return Reference(kind="video", frames=frames, fps=fps, audio=wave)
        raise ValueError(f"unsupported reference spec {spec!r} (image:<path> | video:<path> | audio:<path>)")


def normalize_references(req, geom):
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    from .buckets import DEFAULT_TIER_RULES, tier_for_canvas
    counts = {kind: sum(ref.kind == kind for ref in req.references) for kind in ("image", "video", "audio")}
    if len(req.references) > 12 or counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
        raise ValueError("Reference limits: 9 images, 3 videos, 3 audio; 12 total")
    video_seconds = sum(len(ref.frames) / ref.fps for ref in req.references if ref.kind == "video" and ref.frames is not None and np.isfinite(ref.fps) and ref.fps > 0)
    audio_seconds = sum(ref.audio.shape[-1] / ref.sample_rate for ref in req.references if ref.kind == "audio" and ref.audio is not None and ref.sample_rate > 0)
    if video_seconds > 15.0 + 1e-6 or audio_seconds > 15.0 + 1e-6:
        raise ValueError("Video-reference and standalone-audio totals must each be at most 15 seconds")
    for ref in req.references:
        if ref.kind == "video" and ref.frames is not None and np.isfinite(ref.fps) and ref.fps > 0:
            if not 2.0 <= len(ref.frames) / ref.fps <= 15.0 + 1e-6:
                raise ValueError("Each video reference must be 2–15 seconds")
        if ref.kind == "audio" and ref.audio is not None and ref.sample_rate > 0:
            if not 2.0 <= ref.audio.shape[-1] / ref.sample_rate <= 15.0 + 1e-6:
                raise ValueError("Each audio reference must be 2–15 seconds")
    normalized, sizes = [], []
    tier = DEFAULT_TIER_RULES[tier_for_canvas(geom.height, geom.width)]
    max_samples = round(geom.num_frames / G.FPS * G.AUDIO_SAMPLE_RATE)
    for ref in req.references:
        wave = None
        if ref.has_audio:
            if ref.sample_rate != G.AUDIO_SAMPLE_RATE:
                raise ValueError("In-memory audio references must be 32 kHz; file inputs are resampled automatically")
            wave = np.asarray(ref.audio, dtype=np.float32)
            if wave.ndim != 2 or wave.shape[0] not in (1, 2) or wave.shape[1] == 0 or not np.isfinite(wave).all():
                raise ValueError("Reference audio must be finite, nonempty mono/stereo samples")
            wave = np.repeat(wave, 2, axis=0) if wave.shape[0] == 1 else wave
            wave = np.ascontiguousarray(wave[:, :max_samples])
        if ref.kind == "image":
            image, size = fit_reference_image(ref.image, geom, req.reference_rule, req.reference_short_edge, req.reference_allow_upscale)
            normalized.append(Reference(kind="image", image=image))
            sizes.append(size)
        elif ref.kind == "audio":
            if wave is None:
                raise ValueError("Audio reference needs waveform samples")
            normalized.append(Reference(kind="audio", audio=wave))
        elif ref.kind == "video":
            if not np.isfinite(ref.fps) or ref.fps <= 0 or ref.frames is None or len(ref.frames) == 0:
                raise ValueError("Video reference needs frames and a finite positive FPS")
            frames = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
                ref.frames, ref.fps, geom.num_frames, G.CANVAS_MULTIPLE,
                tier.short_edge, tier.max_pixels, float(G.FPS))
            if len(frames) < 22:
                raise ValueError("Video reference must contain at least 22 frames at 24 fps")
            normalized.append(Reference(kind="video", frames=frames, fps=G.FPS, audio=wave))
        else:
            raise ValueError(f"Unsupported reference kind: {ref.kind}")
    return normalized, sizes


@dataclass
class Request:
    task: str
    prompt: str
    first_image: Optional[Image.Image] = None
    last_image: Optional[Image.Image] = None
    references: List[Reference] = field(default_factory=list)
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: int = 124
    short_edge: int = 768
    max_pixels: int = 768 * 1344
    reference_rule: str = "tiered"  # tiered | pipeline
    reference_short_edge: Optional[int] = None  # explicit image-reference short edge (overrides the rule)
    reference_allow_upscale: Optional[bool] = None  # with reference_short_edge; default: upscale (official behaviour)
    keyframe_fit: str = "pipeline"  # pipeline | crop (source center-crop, then resize)

    def keyframes(self) -> List[tuple[str, Image.Image]]:
        if self.task == "i2va":
            return [("first", self.first_image)]
        if self.task == "fl2va":
            return [("first", self.first_image), ("last", self.last_image)]
        if self.task == "l2va":
            return [("last", self.last_image)]
        return []


@dataclass
class Geometry:
    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int


def resolve_geometry(req: Request) -> Geometry:
    if (req.height is None) != (req.width is None):
        raise ValueError("height and width must be given together")
    if req.height is None:
        anchor = req.first_image or req.last_image
        if anchor is not None:
            h, w = G.resolve_canvas_size(anchor.width, anchor.height, short_edge=req.short_edge, max_pixels=req.max_pixels)
        else:
            h, w = G.resolve_canvas_size(16, 9, short_edge=req.short_edge, max_pixels=req.max_pixels)
    else:
        h, w = int(req.height), int(req.width)
    if h % G.CANVAS_MULTIPLE or w % G.CANVAS_MULTIPLE:
        raise ValueError(f"canvas must be a multiple of {G.CANVAS_MULTIPLE}, got {h}x{w}")
    n = G.align_num_frames(int(req.num_frames))
    H16, W16 = G.latent_hw(h, w)
    return Geometry(h, w, n, G.video_latent_num_frames(n), H16, W16, G.audio_latent_num_frames(n))


def center_crop_box(source_width: int, source_height: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Centered source-space crop in PIL box order."""
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError("source and target dimensions must be positive")
    aspect = width / height
    if source_width / source_height > aspect:
        crop_height = source_height
        crop_width = max(1, min(source_width, round(source_height * aspect)))
    else:
        crop_width = source_width
        crop_height = max(1, min(source_height, round(source_width / aspect)))
    left, top = (source_width - crop_width) // 2, (source_height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def fit_keyframe(image: Image.Image, height: int, width: int, mode: str) -> Image.Image:
    """Pipeline stretch/cover, or source center-crop then LANCZOS resize for bucket inference."""
    if mode == "crop":
        return image.crop(center_crop_box(image.width, image.height, width, height)).resize((width, height), Image.Resampling.LANCZOS)
    if mode == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)
    if mode != "cover":
        raise ValueError(f"unknown keyframe fit mode {mode!r}")
    scale = max(width / image.width, height / image.height)
    size = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    left = max(0, (size[0] - width) // 2)
    top = max(0, (size[1] - height) // 2)
    return resized.crop((left, top, left + width, top + height))


REFERENCE_RULES = ("tiered", "pipeline")


def resolve_reference_rule(geom: Geometry, rule: str, short_edge: Optional[int] = None, allow_upscale: Optional[bool] = None) -> Dict[str, Any]:
    """``{rule, short_edge, allow_upscale, max_pixels, tier}`` an image reference is encoded with for this canvas."""
    if short_edge is not None:
        if int(short_edge) <= 0:
            raise ValueError(f"reference_short_edge must be positive, got {short_edge}")
        return {"rule": "explicit", "short_edge": int(short_edge), "allow_upscale": True if allow_upscale is None else bool(allow_upscale),
                "max_pixels": None, "tier": None}
    if allow_upscale is not None:
        raise ValueError("reference_allow_upscale needs reference_short_edge; the named rules fix it")
    if rule == "tiered":
        from .buckets import reference_short_edge_for_canvas, tier_for_canvas

        tier = tier_for_canvas(geom.height, geom.width)
        return {"rule": rule, "short_edge": reference_short_edge_for_canvas(geom.height, geom.width), "allow_upscale": True, "max_pixels": None, "tier": tier}
    if rule == "pipeline":
        return {"rule": rule, "short_edge": 2048, "allow_upscale": True, "max_pixels": None, "tier": None}
    raise ValueError(f"reference_rule must be tiered|pipeline, got {rule!r}")


def fit_reference_image(image: Image.Image, geom: Geometry, rule: str, short_edge: Optional[int] = None,
                        allow_upscale: Optional[bool] = None) -> tuple[Image.Image, Dict[str, Any]]:
    """The resized reference and the resolved rule (with ``source`` and ``encoded`` sizes as ``[height, width]``)."""
    resolved = resolve_reference_rule(geom, rule, short_edge, allow_upscale)
    h, w = reference_canvas_size(image.width, image.height, short_edge=resolved["short_edge"], max_pixels=resolved["max_pixels"],
                                 allow_upscale=resolved["allow_upscale"])
    resolved = {**resolved, "source": [image.height, image.width], "encoded": [h, w]}
    return image.resize((w, h), Image.Resampling.LANCZOS), resolved


class H3ConditionEncoder:
    """Stage-loaded Qwen3-VL + VAEs. ``encode(request)`` returns CPU tensors ready for ``H3DiffusionWrapper``."""

    def __init__(self, checkpoint: str, device: torch.device | str = "cuda", keep_loaded: bool = False, text_encoder_checkpoint: str | None = None):
        self.checkpoint = Path(checkpoint)
        # Optionally load the tokenizer, processor and encoder from a separate complete snapshot.
        self.text_encoder_checkpoint = Path(text_encoder_checkpoint) if text_encoder_checkpoint else self.checkpoint
        self.device = torch.device(device)
        self.keep_loaded = keep_loaded
        self._te = None
        self._vae = None
        self._audio_vae = None

    # ---- component staging
    def _load_text_encoder(self):
        if self._te is None:
            from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

            t0 = time.time()
            root = self.text_encoder_checkpoint
            tok = AutoTokenizer.from_pretrained(root / "tokenizer")
            proc = AutoProcessor.from_pretrained(root / "processor")
            te = Qwen3VLForConditionalGeneration.from_pretrained(root / "text_encoder", dtype=torch.bfloat16).to(self.device).eval()
            self._te = (tok, proc, te)
            print(f"[cond] Qwen3-VL loaded in {time.time() - t0:.1f}s", flush=True)
        return self._te

    def _load_vaes(self):
        if self._vae is None:
            from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

            t0 = time.time()
            self._vae = AutoencoderKLMiniMaxH3.from_pretrained(self.checkpoint / "vae", torch_dtype=torch.float32).to(self.device).eval()
            self._audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(self.checkpoint / "audio_vae", torch_dtype=torch.float32).to(self.device).eval()
            print(f"[cond] VAEs loaded in {time.time() - t0:.1f}s", flush=True)
        return self._vae, self._audio_vae

    def free(self, what: str = "all"):
        if what in ("text", "all"):
            self._te = None
        if what in ("vae", "all"):
            self._vae = self._audio_vae = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- encoding
    @torch.no_grad()
    def encode_text(self, req: Request, images_in_presentation: Sequence[Image.Image], audio_ref_flags: Sequence[bool], references=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Presentation = [per reference/keyframe: label + vision block] + prompt -> (embeds [L,5120] bf16, tags [L])."""
        from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds

        tok, proc, te = self._load_text_encoder()
        if references is not None:
            from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
            step = MiniMaxH3Ref2VATextEncoderStep()
            vision_inputs, image_counts, video_counts, timestamps = step._gather_vision_features(proc, references, G.FPS)
            token_ids, token_tags = step._build_presentation(tok, req.prompt, references, image_counts, video_counts, timestamps,
                text_tag=G.TEXT_TAG, video_tag=G.VIDEO_TAG)
        else:
            vision_inputs, counts = vision_features(proc, list(images_in_presentation))
            token_ids, token_tags = build_presentation(tok, req.prompt, counts, text_tag=G.TEXT_TAG, video_tag=G.VIDEO_TAG)
        n_text = sum(1 for t in token_tags if t == G.TEXT_TAG)
        if n_text > G.MAX_TEXT_TOKENS:
            raise ValueError(f"prompt presentation has {n_text} text tokens > {G.MAX_TEXT_TOKENS}")
        embeds = get_qwen3vl_prompt_embeds(te, proc, token_ids, vision_inputs, text_encoder_layer=TEXT_ENCODER_LAYER, device=self.device, dtype=torch.bfloat16)
        return embeds[0].detach().to("cpu", torch.bfloat16).contiguous(), torch.tensor(token_tags, dtype=torch.long)

    @torch.no_grad()
    def encode_image_latent(self, image: Image.Image) -> torch.Tensor:
        """``[1, 1, 24, H16, W16]`` fp32 (external layout) via the pipeline's ``encode_vae_condition``."""
        from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition

        vae, _ = self._load_vaes()
        pixels = torch.from_numpy(np.array(image)).to(self.device).permute(2, 0, 1)[None, :, None]
        lat = encode_vae_condition(vae, pixels, PIXEL_MEAN, PIXEL_STD, KEYFRAME_ENCODE_SEED)  # [1,24,1,H,W] cpu fp32
        return video_storage_to_external(lat[0]).unsqueeze(0).contiguous()

    @torch.no_grad()
    def encode_video_latent(self, frames):
        from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
        vae, _ = self._load_vaes()
        count = G.align_num_frames_down(len(frames))
        pixels = torch.from_numpy(frames[:count].copy()).to(self.device).permute(3, 0, 1, 2)[None]
        lat = encode_vae_condition(vae, pixels, PIXEL_MEAN, PIXEL_STD, KEYFRAME_ENCODE_SEED)
        return video_storage_to_external(lat[0]).unsqueeze(0).contiguous()

    @torch.no_grad()
    def encode_audio_latent(self, wave: np.ndarray) -> torch.Tensor:
        """``[1, T, 64]`` fp32 (external layout), posterior mode, channels as batch items."""
        _, audio_vae = self._load_vaes()
        mean = torch.tensor(audio_vae.config.latents_mean).view(1, -1, 1)
        std = torch.tensor(audio_vae.config.latents_std).view(1, -1, 1)
        samples = torch.from_numpy(np.ascontiguousarray(wave)).to(self.device)[:, None]
        lat = audio_vae.encode(samples, return_dict=False)[0].mode().float().cpu()  # [2,32,T]
        lat = (lat - mean) / std
        return audio_storage_to_external(lat).unsqueeze(0).contiguous()

    def encode(self, req: Request) -> Dict[str, Any]:
        if req.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {req.task!r}")
        geom = resolve_geometry(req)
        if req.keyframe_fit not in {"pipeline", "crop"}:
            raise ValueError(f"keyframe_fit must be pipeline|crop, got {req.keyframe_fit!r}")
        keyframes = req.keyframes()
        fitted = []
        for i, (anchor, img) in enumerate(keyframes):
            if img is None:
                raise ValueError(f"task {req.task} needs a {anchor} image")
            mode = "crop" if req.keyframe_fit == "crop" else ("stretch" if i == 0 else "cover")
            fitted.append((anchor, fit_keyframe(img, geom.height, geom.width, mode)))
        normalized, ref_sizes = normalize_references(req, geom) if req.task == "ref2va" else ([], [])
        if req.task == "ref2va" and not any(r.kind in {"image", "video"} for r in normalized):
            raise ValueError("ref2va needs at least one image or video reference")
        presentation_images = [img for _, img in fitted]
        embeds, tags = self.encode_text(req, presentation_images, [], references=normalized if req.task == "ref2va" else None)
        if not self.keep_loaded:
            self.free("text")

        # stage 2: VAE latents
        cond: Dict[str, Any] = {"prompt_embeds": embeds.unsqueeze(0), "text_token_tags": tags, "keyframes": [], "references": []}
        for anchor, img in fitted:
            cond["keyframes"].append({"anchor": anchor, "latents": self.encode_image_latent(img)})
        if req.task == "ref2va":
            for ref in normalized:
                entry = {"kind": ref.kind}
                if ref.kind == "image":
                    entry["video_latents"] = self.encode_image_latent(ref.image)
                elif ref.kind == "video":
                    entry["video_latents"] = self.encode_video_latent(ref.frames)
                if ref.has_audio:
                    entry["audio_latents"] = self.encode_audio_latent(ref.audio)
                cond["references"].append(entry)
        if not self.keep_loaded:
            self.free("vae")
        cond["geometry"] = geom
        if req.task == "ref2va":
            cond["reference_images"] = ref_sizes  # the rule and size every image reference was encoded with
        cond["packed_tokens"] = G.packed_token_count(
            num_text_tokens=int(tags.numel()),
            num_latent_frames=geom.num_latent_frames,
            latent_height=geom.latent_height,
            latent_width=geom.latent_width,
            num_audio_latents=geom.num_audio_latents,
            num_keyframes=len(fitted),
            reference_video_rows=sum(int(r["video_latents"].shape[1] * (r["video_latents"].shape[3] // 2) * (r["video_latents"].shape[4] // 2)) for r in cond["references"] if "video_latents" in r),
            reference_audio_rows=sum(int(r["audio_latents"].shape[1] * G.AUDIO_CHANNELS) for r in cond["references"] if "audio_latents" in r),
        ).total
        return cond
