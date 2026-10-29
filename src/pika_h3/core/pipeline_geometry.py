# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""MiniMax-H3 geometry: the frame-count, canvas, audio and packed-row arithmetic.

Everything the released checkpoint fixes is re-exported from the diffusers modular pipeline rather than
re-derived here — `resolve_canvas_size`, `align_num_frames`, `video_latent_num_frames`,
`audio_latent_num_frames` are the pipeline's own functions. What this module adds is the preprocessing
side of the same arithmetic: snapping a decoded clip *down* onto the VAE's `17n + 5` grid (the pipeline
only ever snaps up, because it generates), the reference-image canvas rule, and the packed row counts the
bucket geometry requires.
"""

from __future__ import annotations

from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    MINIMAX_H3_FPS,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)

__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_LATENT_CHANNELS",
    "AUDIO_LATENTS_PER_SECOND",
    "AUDIO_SAMPLING_RATE",
    "CANVAS_MULTIPLE",
    "FPS",
    "LATENT_CHANNELS",
    "MAX_ASPECT_RATIO",
    "MIN_ASPECT_RATIO",
    "PATCH_SIZE",
    "SPATIAL_COMPRESSION",
    "TASK_CONDITION_KEYFRAMES",
    "VAE_FRAMES_PER_CHUNK",
    "VAE_LATENTS_PER_CHUNK",
    "align_num_frames",
    "audio_latent_num_frames",
    "audio_num_samples",
    "aspect_ratio_supported",
    "largest_aligned_num_frames",
    "latent_size",
    "packed_row_counts",
    "reference_canvas_size",
    "resolve_canvas_size",
    "video_latent_num_frames",
]

# --- released-checkpoint constants (mirrored from the `vae/`, `audio_vae/` and `transformer/` configs) ---
FPS = MINIMAX_H3_FPS  # 24
AUDIO_SAMPLING_RATE = 32000
AUDIO_LATENTS_PER_SECOND = MINIMAX_H3_AUDIO_LATENTS_PER_SECOND  # 40
AUDIO_CHANNELS = MINIMAX_H3_AUDIO_CHANNELS  # 2, packed channel-major
AUDIO_LATENT_CHANNELS = 32
LATENT_CHANNELS = 24
SPATIAL_COMPRESSION = 16
PATCH_SIZE = (1, 2, 2)
CANVAS_MULTIPLE = SPATIAL_COMPRESSION * PATCH_SIZE[2]  # 32
VAE_FRAMES_PER_CHUNK = 17  # vae.config.clip_length
VAE_LATENTS_PER_CHUNK = 5  # vae.tokens_chunk_size = ceil(17 / 4)
MIN_ASPECT_RATIO = MINIMAX_H3_MIN_ASPECT_RATIO
MAX_ASPECT_RATIO = MINIMAX_H3_MAX_ASPECT_RATIO

# How many keyframe conditioning blocks each task packs. `ref2va` carries a reference instead, whose row
# count depends on the reference's own canvas, so it is not a constant.
TASK_CONDITION_KEYFRAMES = {"t2va": 0, "i2va": 1, "fl2va": 2, "ref2va": 0}


def aspect_ratio_supported(width: int, height: int) -> bool:
    """Whether `resolve_canvas_size` would accept this source geometry (aspect ratios 1:4 .. 4:1)."""
    if width <= 0 or height <= 0:
        return False
    return MIN_ASPECT_RATIO <= width / height <= MAX_ASPECT_RATIO


def largest_aligned_num_frames(
    max_frames: int,
    frames_per_chunk: int = VAE_FRAMES_PER_CHUNK,
    latents_per_chunk: int = VAE_LATENTS_PER_CHUNK,
) -> int:
    """The largest `frames_per_chunk * n + latents_per_chunk` that fits in `max_frames`, or 0 if none does.

    The pipeline only ever rounds a *requested* frame count up (`align_num_frames`), because it generates.
    Preprocessing has the opposite problem: a clip has however many frames it has, and the aligned count has
    to fit inside them so encoding does not require repeated padding frames.
    """
    if max_frames < latents_per_chunk:
        return 0
    return (max_frames - latents_per_chunk) // frames_per_chunk * frames_per_chunk + latents_per_chunk


def latent_size(height: int, width: int, spatial_compression: int = SPATIAL_COMPRESSION) -> tuple[int, int]:
    """`(latent_height, latent_width)` of a canvas. Both axes must be whole latents."""
    if height % spatial_compression or width % spatial_compression:
        raise ValueError(
            f"A canvas must be a multiple of {spatial_compression} on both axes, got {height}x{width}."
        )
    return height // spatial_compression, width // spatial_compression


def audio_num_samples(
    num_frames: int, fps: float = FPS, sampling_rate: int = AUDIO_SAMPLING_RATE
) -> int:
    """Waveform samples that cover `num_frames` video frames, at the audio VAE's sample rate."""
    return int(round(num_frames / fps * sampling_rate))


def reference_canvas_size(
    width: int,
    height: int,
    canvas_multiple: int = CANVAS_MULTIPLE,
    short_edge: int = 2048,
    max_pixels: int | None = None,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """`(height, width)` a `ref2va` image reference is encoded at.

    `MiniMaxH3Ref2VASetupStep`'s own rule is `scale = reference_image_short_edge / min(width, height)`,
    rounded to `canvas_multiple` on both axes, **upscaling included and with no area cap**. Two deliberate
    departures from it here, both defaults, both switchable:

    * `allow_upscale=False` clamps the scale at 1. Upsampling a 544-short-edge source frame to 2048 adds
      no detail and costs 14x the reference rows.
    * `max_pixels` caps the area the way the generated canvas is capped. A 1080p reference at the pipeline's
      own rule is a 68x120 latent, i.e. 2040 packed rows *and* 2040 Qwen3-VL vision tokens — more text rows
      than the whole rest of the presentation, for a single frame. Pass `None` for the pipeline's behaviour.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {width}x{height}.")
    if not aspect_ratio_supported(width, height):
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")

    scale = short_edge / min(width, height)
    if not allow_upscale:
        scale = min(scale, 1.0)
    if max_pixels:
        area = (width * scale) * (height * scale)
        if area > max_pixels:
            scale *= (max_pixels / area) ** 0.5

    target_height = max(canvas_multiple, round(height * scale / canvas_multiple) * canvas_multiple)
    target_width = max(canvas_multiple, round(width * scale / canvas_multiple) * canvas_multiple)
    return target_height, target_width


def _visual_rows(num_latent_frames: int, latent_height: int, latent_width: int) -> int:
    """Packed rows a latent video occupies: one row per `patch_size` patch."""
    _, patch_h, patch_w = PATCH_SIZE
    return num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)


def packed_row_counts(
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    num_text_tokens: int,
    num_keyframes: int = 0,
    reference_latent_shapes: tuple[tuple[int, int, int], ...] = (),
) -> dict[str, int]:
    """Rows of one packed MiniMax-H3 sequence, by modality.

    Args:
        reference_latent_shapes: `(num_latent_frames, latent_height, latent_width)` per reference.

    Returns:
        `{"text", "condition", "audio", "video", "total"}`.
    """
    video = _visual_rows(num_latent_frames, latent_height, latent_width)
    keyframe_rows = _visual_rows(1, latent_height, latent_width)
    condition = num_keyframes * keyframe_rows + sum(
        _visual_rows(*shape) for shape in reference_latent_shapes
    )
    audio = AUDIO_CHANNELS * num_audio_latents
    return {
        "text": num_text_tokens,
        "condition": condition,
        "audio": audio,
        "video": video,
        "total": num_text_tokens + condition + audio + video,
    }
