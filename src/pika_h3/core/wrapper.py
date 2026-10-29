# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Inference wrapper around ``MiniMaxH3Transformer3DModel``.

Input and output contract::

    video_x0, audio_x0 = wrapper(
        noisy_image_or_video,   # [B, F, 24, H16, W16] normalized VAE latents, or None
        conditional_dict,       # see below
        timestep,               # video sigma (SHIFTED, the one x_t was noised at): [B] or [B, F]
        noisy_audio=...,        # [B, T, 64] (channel-major feature blocks), or None
        audio_timestep=...,     # audio sigma (shifted): [B] or [B, T]
        frozen_video_latents=..., frozen_audio_latents=...,   # clean branch held at sigma 0 (v2a / a2v)
    )

``conditional_dict`` keys:
    prompt_embeds     [B, L, 5120]   Qwen3-VL hidden-state-50 rows of the task presentation
    text_token_tags   [B, L] or [L]  per-row modality tag (text 1, vision block rows 0)
    keyframes         optional list of {"anchor": "first"|"last", "latents": [B, 1, 24, H16, W16]}
    references        optional list of {"kind": "image"|"video"|"audio",
                                        "video_latents": [B, Fr, 24, Hr, Wr] (image/video),
                                        "audio_latents": [B, Tr, 64] (audio, or a video's soundtrack)}

Rules reproduced from the diffusers pipeline: keyframe / visual-reference rows are noised to ``t = 0.999`` with
fresh Gaussian noise and pinned there (their own AdaLN timestep entry, ``max(t_video, 0.999)``), audio reference
rows are clean at ``t = 1.0``, text rows inherit the video timestep, and every target row is denoised. The
transformer predicts ``v = x0 - eps``; this wrapper returns ``x0 = x_t + sigma * v`` for the target rows only.

"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from . import geometry as G
from .packing import (
    PackedLayout,
    ReferenceSpec,
    audio_external_to_rows,
    audio_rows_to_external,
    build_layout,
    row_timesteps,
    video_external_to_rows,
    video_rows_to_external,
)


class H3DiffusionWrapper(nn.Module):
    def __init__(self, model: nn.Module, *, keyframe_noise_aug: float = G.KEYFRAME_NOISE_AUG, layout_cache_size: int = 64):
        super().__init__()
        self.model = model
        self.keyframe_noise_aug = float(keyframe_noise_aug)
        self._layout_cache: Dict[Any, PackedLayout] = {}
        self._layout_cache_size = int(layout_cache_size)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _scalar_sigma(sigma: Optional[torch.Tensor], name: str) -> Optional[float]:
        if sigma is None:
            return None
        s = sigma.detach().float().reshape(sigma.shape[0], -1)
        if s.shape[0] != 1:
            raise ValueError(f"{name}: H3 packs one document per forward (batch_size must be 1), got {s.shape[0]}")
        if s.numel() > 1 and not bool(torch.all(s == s[0, 0])):
            raise ValueError(f"{name}: H3 uses one timestep per modality; per-frame sigmas must be equal")
        return float(s[0, 0])

    def _layout(self, key, builder) -> PackedLayout:
        layout = self._layout_cache.get(key)
        if layout is None:
            layout = builder()
            if len(self._layout_cache) >= self._layout_cache_size:
                self._layout_cache.pop(next(iter(self._layout_cache)))
            self._layout_cache[key] = layout
        return layout

    def _noise_condition_rows(self, entry: Dict[str, Any], key: str) -> torch.Tensor:
        """``x = t x0 + (1 - t) eps`` at ``t = keyframe_noise_aug``.

        Anchor noise is drawn once per request and cached on the condition entry,
        so every denoising step uses the same noised anchors.
        """
        cached = entry.get("_noised_rows")
        latents = entry[key]
        if cached is not None and cached.device == latents.device:
            return cached
        t = self.keyframe_noise_aug
        noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32)
        rows = video_external_to_rows(t * latents.float() + (1.0 - t) * noise)
        entry["_noised_rows"] = rows
        return rows

    def _prepare_document(self, video, audio, conditional_dict, sigma_v, sigma_a):
        """Assemble one document with the same reference layout as singleton inference."""
        device = video.device
        _, F, C, H, W = video.shape
        T = audio.shape[1]
        t_video, t_audio = 1.0 - sigma_v, 1.0 - sigma_a
        prompt_embeds = conditional_dict["prompt_embeds"]
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        text_tags = conditional_dict["text_token_tags"]
        text_tags = text_tags.reshape(-1)
        keyframes = list(conditional_dict.get("keyframes") or [])
        references = list(conditional_dict.get("references") or [])
        layout = conditional_dict.get("_packed_layout")

        # ---- condition rows (noised anchors first, in packed order) + layout
        cond_video_rows = []
        cond_audio_rows = []
        if references:
            specs, ref_shapes, ref_audio_rows = [], [], []
            for ref in references:
                kind = str(ref["kind"])
                has_audio = ref.get("audio_latents") is not None
                specs.append(ReferenceSpec(kind=kind, has_audio=has_audio if kind == "video" else (kind == "audio")))
                if kind in ("image", "video"):
                    lat = ref["video_latents"]
                    ref_shapes.append(tuple(int(x) for x in (lat.shape[1], lat.shape[3], lat.shape[4])))
                    cond_video_rows.append(self._noise_condition_rows(ref, "video_latents").to(device))
                if kind == "audio" or (kind == "video" and has_audio):
                    a = ref["audio_latents"].to(device)
                    rows = audio_external_to_rows(a.float())
                    ref_audio_rows.append(int(rows.shape[1]))
                    cond_audio_rows.append(rows)
            if layout is None:
                key = ("ref2va", tuple(text_tags.tolist()), F, H, W, T, tuple(s.kind for s in specs), tuple(s.has_audio for s in specs), tuple(ref_shapes), tuple(ref_audio_rows))
                layout = self._layout(
                    key,
                    lambda: build_layout(
                        text_tags,
                        num_latent_frames=F,
                        latent_height=H,
                        latent_width=W,
                        num_audio_latents=T,
                        references=specs,
                        reference_video_shapes=ref_shapes,
                        reference_audio_rows=ref_audio_rows,
                    ),
                )
        else:
            anchors = tuple(str(k["anchor"]) for k in keyframes)
            for k in keyframes:
                lat = k["latents"]
                if tuple(lat.shape[1:]) != (1, C, H, W):
                    raise ValueError(f"keyframe latents must be [B, 1, {C}, {H}, {W}], got {tuple(lat.shape)}")
                cond_video_rows.append(self._noise_condition_rows(k, "latents").to(device))
            if layout is None:
                key = ("fl2va", tuple(text_tags.tolist()), F, H, W, T, anchors)
                layout = self._layout(
                    key,
                    lambda: build_layout(
                        text_tags,
                        num_latent_frames=F,
                        latent_height=H,
                        latent_width=W,
                        num_audio_latents=T,
                        keyframe_anchors=anchors,
                    ),
                )
        layout_dev = conditional_dict.get("_packed_layout_device")
        if layout_dev is None or layout_dev.position_ids.device != device:
            layout_dev = layout.to(device)

        # ---- rows
        target_video_rows = video_external_to_rows(video.float())
        target_audio_rows = audio_external_to_rows(audio.float())
        video_rows = torch.cat(cond_video_rows + [target_video_rows], dim=1) if cond_video_rows else target_video_rows
        audio_rows = torch.cat(cond_audio_rows + [target_audio_rows], dim=1) if cond_audio_rows else target_audio_rows
        n_cond_v = layout.num_condition_video_rows
        n_cond_a = layout.num_condition_audio_rows
        if video_rows.shape[1] != layout.video_indices.numel() or audio_rows.shape[1] != layout.audio_indices.numel():
            raise ValueError(
                f"packed rows disagree with layout: video {video_rows.shape[1]} vs {layout.video_indices.numel()}, "
                f"audio {audio_rows.shape[1]} vs {layout.audio_indices.numel()}"
            )

        # ---- per-row timesteps (clean-time convention; conditioning rows pinned)
        unique_t, t_index = row_timesteps(
            layout,
            video_t=t_video,
            audio_t=t_audio,
            condition_video_t=max(t_video, self.keyframe_noise_aug),
            condition_audio_t=1.0,
        )
        unique_t = unique_t.to(device)
        t_index = t_index.to(device)

        model_inputs = dict(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=prompt_embeds.to(device),
            timestep=unique_t,
            timestep_indices=t_index,
            token_tags=layout_dev.token_tags,
            position_ids=layout_dev.position_ids,
            video_indices=layout_dev.video_indices,
            audio_indices=layout_dev.audio_indices,
            text_indices=layout_dev.text_indices,
            return_dict=False,
        )
        return model_inputs, target_video_rows, target_audio_rows, n_cond_v, n_cond_a

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        noisy_image_or_video: Optional[torch.Tensor],
        conditional_dict: Dict[str, Any],
        timestep: Optional[torch.Tensor],
        noisy_audio: Optional[torch.Tensor] = None,
        audio_timestep: Optional[torch.Tensor] = None,
        frozen_video_latents: Optional[torch.Tensor] = None,
        frozen_audio_latents: Optional[torch.Tensor] = None,
        output_type: str = "x0",
        head_coeffs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **_ignored,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if output_type != "x0" or head_coeffs is not None:
            raise ValueError("This inference wrapper supports x0 prediction only")
        if "documents" in conditional_dict:
            raise ValueError("Batched documents are not supported")
        freeze_video = frozen_video_latents is not None
        freeze_audio = frozen_audio_latents is not None
        video = frozen_video_latents if freeze_video else noisy_image_or_video
        audio = frozen_audio_latents if freeze_audio else noisy_audio
        if video is None or audio is None:
            raise ValueError("MiniMax-H3 always packs video AND audio rows; both branches are required")
        if video.shape[0] != 1 or audio.shape[0] != 1:
            raise ValueError("H3DiffusionWrapper supports batch_size=1 per forward (one packed document)")
        device = video.device
        B, F, C, H, W = video.shape
        T = audio.shape[1]
        if C != G.VIDEO_LATENT_CHANNELS:
            raise ValueError(f"video latents must have {G.VIDEO_LATENT_CHANNELS} channels, got {C}")

        sigma_v = 0.0 if freeze_video else self._scalar_sigma(timestep, "timestep")
        sigma_a = 0.0 if freeze_audio else self._scalar_sigma(audio_timestep, "audio_timestep")
        if sigma_v is None or sigma_a is None:
            raise ValueError("timestep and audio_timestep are required for non-frozen branches")
        t_video = 1.0 - sigma_v
        t_audio = 1.0 - sigma_a

        model_inputs, target_video_rows, target_audio_rows, n_cond_v, n_cond_a = self._prepare_document(
            video, audio, conditional_dict, sigma_v, sigma_a)
        v_video, v_audio = self.model(**model_inputs)
        v_video = v_video[:, n_cond_v:]
        v_audio = v_audio[:, n_cond_a:]

        # ---- x0 = x_t + sigma * v (fp32), target rows only
        if freeze_video:
            video_x0 = video
        else:
            x0_rows = target_video_rows + float(sigma_v) * v_video.float()
            video_x0 = video_rows_to_external(x0_rows, F, H, W).to(noisy_image_or_video.dtype)
        if freeze_audio:
            audio_x0 = audio
        else:
            a0_rows = target_audio_rows + float(sigma_a) * v_audio.float()
            audio_x0 = audio_rows_to_external(a0_rows, T).to(noisy_audio.dtype)
        return video_x0, audio_x0
