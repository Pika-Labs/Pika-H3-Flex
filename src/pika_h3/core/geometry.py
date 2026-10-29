# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
# Portions adapted from Diffusers MiniMax H3: modular_pipeline.py.
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# The adapted upstream portions are licensed under Apache-2.0; see licenses/Apache-2.0.txt.
# Modified by Mellis, Inc. in 2026. Copyright © 2026 Mellis, Inc.
# Model Derivative of MiniMax H3, distributed under the MiniMax H3 Community License Agreement.
# Code remains licensed under Apache-2.0; the model terms above apply to the adapters.
# Changes: Adapted canvas-size resolution for standalone inference.
# Source revision: 3a2f35d4efa4c059c8bfb3bc0d6c906264895c81.

"""MiniMax-H3 latent geometry: the single source of truth for every shape formula.

Numbers come from the released checkpoint (``vae/config.json``: latent_channels 24, spatial /16, clip_length 17,
5 latents per chunk; ``audio_vae/config.json``: 32 channels, 32 kHz, 40 latents/s; transformer patch (1, 2, 2)).
"""

from __future__ import annotations

from dataclasses import dataclass

VIDEO_LATENT_CHANNELS = 24
VAE_SPATIAL_FACTOR = 16
VAE_FRAMES_PER_CHUNK = 17
VAE_LATENTS_PER_CHUNK = 5
PATCH_SIZE = (1, 2, 2)
CANVAS_MULTIPLE = VAE_SPATIAL_FACTOR * PATCH_SIZE[2]  # 32
FPS = 24
AUDIO_LATENT_CHANNELS = 32
AUDIO_CHANNELS = 2
AUDIO_SAMPLE_RATE = 32000
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_HOP = AUDIO_SAMPLE_RATE // AUDIO_LATENTS_PER_SECOND  # 800
# External audio layout: [T, AUDIO_CHANNELS * AUDIO_LATENT_CHANNELS], channel-major feature blocks.
AUDIO_EXTERNAL_DIM = AUDIO_CHANNELS * AUDIO_LATENT_CHANNELS  # 64
TEXT_DIM = 5120
MAX_TEXT_TOKENS = 2048  # maximum text rows in the Qwen3-VL presentation
KEYFRAME_NOISE_AUG = 0.999

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2


def align_num_frames(num_frames: int) -> int:
    """Snap up to the next ``17 n + 5`` the video VAE encodes without padding."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    while num_frames % VAE_FRAMES_PER_CHUNK != VAE_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def align_num_frames_down(num_frames: int) -> int:
    """Largest ``17 n + 5`` that is <= num_frames (at least 5)."""
    if num_frames < VAE_LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be >= {VAE_LATENTS_PER_CHUNK}, got {num_frames}")
    n = (num_frames - VAE_LATENTS_PER_CHUNK) // VAE_FRAMES_PER_CHUNK
    return n * VAE_FRAMES_PER_CHUNK + VAE_LATENTS_PER_CHUNK


def video_latent_num_frames(num_frames: int) -> int:
    if num_frames % VAE_FRAMES_PER_CHUNK != VAE_LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be 17n+5, got {num_frames}")
    return (num_frames - VAE_LATENTS_PER_CHUNK) // VAE_FRAMES_PER_CHUNK * VAE_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def latent_hw(height: int, width: int) -> tuple[int, int]:
    if height % CANVAS_MULTIPLE or width % CANVAS_MULTIPLE:
        raise ValueError(f"canvas must be a multiple of {CANVAS_MULTIPLE}, got {height}x{width}")
    return height // VAE_SPATIAL_FACTOR, width // VAE_SPATIAL_FACTOR


def rows_per_frame(latent_height: int, latent_width: int) -> int:
    return (latent_height // PATCH_SIZE[1]) * (latent_width // PATCH_SIZE[2])


@dataclass(frozen=True)
class PackedTokenCount:
    text: int
    condition_video: int
    condition_audio: int
    audio: int
    video: int

    @property
    def total(self) -> int:
        return self.text + self.condition_video + self.condition_audio + self.audio + self.video


def packed_token_count(
    *,
    num_text_tokens: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    num_keyframes: int = 0,
    reference_video_rows: int = 0,
    reference_audio_rows: int = 0,
) -> PackedTokenCount:
    rpf = rows_per_frame(latent_height, latent_width)
    return PackedTokenCount(
        text=int(num_text_tokens),
        condition_video=int(num_keyframes) * rpf + int(reference_video_rows),
        condition_audio=int(reference_audio_rows),
        audio=int(num_audio_latents) * AUDIO_CHANNELS,
        video=int(num_latent_frames) * rpf,
    )


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    short_edge: int = 768,
    max_pixels: int = 768 * 1344,
    canvas_multiple: int = CANVAS_MULTIPLE,
    min_aspect_ratio: float = 0.25,
    max_aspect_ratio: float = 4.0,
) -> tuple[int, int]:
    """(height, width) — same arithmetic as the diffusers pipeline's ``resolve_canvas_size``."""
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError("aspect must be positive")
    ratio = aspect_width / aspect_height
    if not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(f"aspect ratio {ratio:g} outside [{min_aspect_ratio}, {max_aspect_ratio}]")
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    m = canvas_multiple
    return max(m, round(height / m) * m), max(m, round(width / m) * m)
