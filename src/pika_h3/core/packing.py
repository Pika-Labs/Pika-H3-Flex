# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
# Portions adapted from Diffusers MiniMax H3: before_denoise.py and decoders.py.
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# The adapted upstream portions are licensed under Apache-2.0; see licenses/Apache-2.0.txt.
# Modified by Mellis, Inc. in 2026. Copyright © 2026 Mellis, Inc.
# Model Derivative of MiniMax H3, distributed under the MiniMax H3 Community License Agreement.
# Code remains licensed under Apache-2.0; the model terms above apply to the adapters.
# Changes: Adapted latent patchification and layout conversion for standalone inference.
# Source revision: 3a2f35d4efa4c059c8bfb3bc0d6c906264895c81.

"""Packed-sequence helpers using the Diffusers layout builders.

Convert video [B, F, C, H, W] and audio [B, T, 64] inputs to H3 rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from . import geometry as G

try:  # the reference builders
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3Ref2VAPrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )

    build_packed_sequence = MiniMaxH3PrepareLayoutStep.build_packed_sequence
    build_ref2va_packed_sequence = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence
    build_row_timesteps = MiniMaxH3SetTimestepsStep.build_row_timesteps
    HAVE_DIFFUSERS_H3 = True
except Exception as exc:  # pragma: no cover - only in envs without the H3 diffusers port
    HAVE_DIFFUSERS_H3 = False
    _IMPORT_ERROR = exc

    def _missing(*_args, **_kwargs):
        raise ImportError(
            "diffusers.modular_pipelines.minimax_h3 is required for H3 packing "
            f"(import failed: {_IMPORT_ERROR!r}); install the pinned requirements"
        )

    build_packed_sequence = build_ref2va_packed_sequence = build_row_timesteps = _missing

    def patchify_video_latents(latents: torch.Tensor, patch_size):
        pt, ph, pw = patch_size
        b, c, f, h, w = latents.shape
        x = latents.reshape(b, c, f // pt, pt, h // ph, ph, w // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(-1, c * pt * ph * pw).contiguous()


@dataclass
class ReferenceSpec:
    """Minimal stand-in for ``diffusers...MiniMaxH3Reference`` (only ``kind`` / ``has_audio`` are read)."""

    kind: str
    has_audio: bool = False


def video_external_to_rows(video: torch.Tensor) -> torch.Tensor:
    """``[B, F, C, H, W]`` -> ``[B, rows, C*ph*pw]`` (frame-major, row-major, as the pipeline packs)."""
    b, f, c, h, w = video.shape
    rows = patchify_video_latents(video.permute(0, 2, 1, 3, 4), G.PATCH_SIZE)
    return rows.view(b, -1, rows.shape[-1])


def video_rows_to_external(rows: torch.Tensor, num_frames: int, latent_height: int, latent_width: int) -> torch.Tensor:
    """Inverse of :func:`video_external_to_rows`."""
    pt, ph, pw = G.PATCH_SIZE
    b = rows.shape[0]
    c = rows.shape[-1] // (pt * ph * pw)
    x = rows.reshape(b, num_frames // pt, latent_height // ph, latent_width // pw, c, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(b, c, num_frames, latent_height, latent_width)
    return x.permute(0, 2, 1, 3, 4).contiguous()


def audio_external_to_rows(audio: torch.Tensor) -> torch.Tensor:
    """``[B, T, 2*32]`` -> ``[B, 2*T, 32]`` channel-major rows (all left rows, then all right rows)."""
    b, t, d = audio.shape
    if d != G.AUDIO_EXTERNAL_DIM:
        raise ValueError(f"audio external dim must be {G.AUDIO_EXTERNAL_DIM}, got {d}")
    return audio.view(b, t, G.AUDIO_CHANNELS, G.AUDIO_LATENT_CHANNELS).permute(0, 2, 1, 3).reshape(
        b, G.AUDIO_CHANNELS * t, G.AUDIO_LATENT_CHANNELS
    )


def audio_rows_to_external(rows: torch.Tensor, num_latents: int) -> torch.Tensor:
    b = rows.shape[0]
    return (
        rows.view(b, G.AUDIO_CHANNELS, num_latents, G.AUDIO_LATENT_CHANNELS)
        .permute(0, 2, 1, 3)
        .reshape(b, num_latents, G.AUDIO_EXTERNAL_DIM)
        .contiguous()
    )


def audio_storage_to_external(latents: torch.Tensor) -> torch.Tensor:
    """Artifact layout ``[2, 32, T]`` -> external ``[T, 64]``."""
    if latents.dim() != 3 or latents.shape[0] != G.AUDIO_CHANNELS or latents.shape[1] != G.AUDIO_LATENT_CHANNELS:
        raise ValueError(f"expected audio latents [2, 32, T], got {tuple(latents.shape)}")
    return latents.permute(2, 0, 1).reshape(latents.shape[-1], G.AUDIO_EXTERNAL_DIM).contiguous()


def audio_external_to_storage(latents: torch.Tensor) -> torch.Tensor:
    """External ``[T, 64]`` -> artifact layout ``[2, 32, T]``."""
    t = latents.shape[0]
    return latents.view(t, G.AUDIO_CHANNELS, G.AUDIO_LATENT_CHANNELS).permute(1, 2, 0).contiguous()


def video_storage_to_external(latents: torch.Tensor) -> torch.Tensor:
    """Artifact layout ``[24, F, H, W]`` -> external ``[F, 24, H, W]``."""
    if latents.dim() != 4 or latents.shape[0] != G.VIDEO_LATENT_CHANNELS:
        raise ValueError(f"expected video latents [24, F, H, W], got {tuple(latents.shape)}")
    return latents.permute(1, 0, 2, 3).contiguous()


def video_external_to_storage(latents: torch.Tensor) -> torch.Tensor:
    return latents.permute(1, 0, 2, 3).contiguous()


@dataclass
class PackedLayout:
    position_ids: torch.Tensor  # [S, 3] float64
    token_tags: torch.Tensor  # [S] long
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int
    num_text_tokens: int

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    def to(self, device) -> "PackedLayout":
        return PackedLayout(
            position_ids=self.position_ids.to(device),
            token_tags=self.token_tags.to(device),
            video_indices=self.video_indices.to(device),
            audio_indices=self.audio_indices.to(device),
            text_indices=self.text_indices.to(device),
            num_condition_video_rows=self.num_condition_video_rows,
            num_condition_audio_rows=self.num_condition_audio_rows,
            num_text_tokens=self.num_text_tokens,
        )


def build_layout(
    text_token_tags: torch.Tensor,
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    keyframe_anchors: Sequence[str] = (),
    references: Optional[Sequence[ReferenceSpec]] = None,
    reference_video_shapes: Optional[Sequence[tuple[int, int, int]]] = None,
    reference_audio_rows: Optional[Sequence[int]] = None,
) -> PackedLayout:
    """Build the packed layout on CPU (float64 rope grid) via the diffusers reference builders."""
    tags = text_token_tags.detach().to("cpu", torch.long).reshape(-1)
    if references:
        # The ref2va builder reads geometry off latent tensors; hand it shape-only stand-ins.
        cond_latents = [torch.empty(1, G.VIDEO_LATENT_CHANNELS, f, h, w, device="meta") for (f, h, w) in reference_video_shapes]
        audio_conds = [torch.empty(int(n), G.AUDIO_LATENT_CHANNELS, device="meta") for n in (reference_audio_rows or [])]
        out = build_ref2va_packed_sequence(
            tags,
            list(references),
            cond_latents,
            audio_conds,
            int(num_latent_frames),
            int(latent_height),
            int(latent_width),
            int(num_audio_latents),
            G.PATCH_SIZE,
            G.AUDIO_CHANNELS,
            G.AUDIO_TAG,
            G.VIDEO_TAG,
        )
    else:
        out = build_packed_sequence(
            tags,
            int(num_latent_frames),
            int(latent_height),
            int(latent_width),
            int(num_audio_latents),
            G.PATCH_SIZE,
            G.AUDIO_CHANNELS,
            G.AUDIO_TAG,
            G.VIDEO_TAG,
            tuple(keyframe_anchors),
        )
    position_ids, token_tags, video_indices, audio_indices, text_indices, n_cond_v, n_cond_a = out
    return PackedLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=int(n_cond_v),
        num_condition_audio_rows=int(n_cond_a),
        num_text_tokens=int(tags.numel()),
    )


def row_timesteps(
    layout: PackedLayout,
    *,
    video_t: float,
    audio_t: float,
    condition_video_t: float,
    condition_audio_t: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(unique timesteps [K], per-row index [S]) — built on CPU like the pipeline, then moved by the caller."""
    return build_row_timesteps(
        layout.video_indices.cpu(),
        layout.audio_indices.cpu(),
        layout.num_condition_video_rows,
        layout.num_condition_audio_rows,
        layout.num_text_tokens,
        float(video_t),
        float(audio_t),
        float(condition_video_t),
        float(condition_audio_t),
    )
