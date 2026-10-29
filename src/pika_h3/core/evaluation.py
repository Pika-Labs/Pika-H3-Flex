# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Replayable FP32 inference noise.

For seed S, initial video then audio noise use S, schedule re-noising uses S + 1,
and reference augmentation uses S + 7919. Caller RNG streams are restored even
on failure."""

from __future__ import annotations

from contextlib import contextmanager
import random
from typing import Any, Dict

import numpy as np
import torch

from . import geometry as G
from .sampling import few_step_sample

EVAL_NOISE_VERSION = "pika-flex-noise-v1"


@contextmanager
def evaluation_noise_scope(seed: int, device):
    """Isolate augmentation RNG, touching only CPU and the selected CUDA device.

    Unlike ``torch.manual_seed``, seeding the CPU default generator directly does
    not seed every visible CUDA device. This matters for one-process-per-GPU jobs.
    """
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    augmentation_seed = int(seed) + 7919
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed % (2**32))
            torch.random.default_generator.manual_seed(augmentation_seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(augmentation_seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def apply_condition_rows(conditional_dict: Dict[str, Any], rows: torch.Tensor) -> None:
    """Apply explicit packed fp32 visual anchors (keyframes, then visual refs).

    Audio references are not noise-augmented. Validate the full override before
    updating entries; callers must not rely on an old ``_noised_rows`` cache as
    an implicit override.
    """
    entries = [(k, "latents") for k in (conditional_dict.get("keyframes") or [])]
    entries += [(r, "video_latents") for r in (conditional_dict.get("references") or []) if "video_latents" in r]
    if not torch.is_tensor(rows) or rows.ndim != 3 or not rows.is_floating_point():
        raise ValueError("condition_rows must be a floating tensor [batch, rows, 96]")
    if rows.shape[2] != G.VIDEO_LATENT_CHANNELS * G.PATCH_SIZE[1] * G.PATCH_SIZE[2]:
        raise ValueError("condition_rows must have 96 packed video features")
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("condition_rows must be finite")
    counts = []
    for entry, key in entries:
        lat = entry[key]
        if lat.ndim != 5 or lat.shape[0] != rows.shape[0]:
            raise ValueError("condition_rows batch disagrees with the request")
        counts.append(int(lat.shape[1] * G.rows_per_frame(lat.shape[3], lat.shape[4])))
    if sum(counts) != int(rows.shape[1]):
        raise ValueError(f"condition rows disagree with the request: consumed {sum(counts)} of {int(rows.shape[1])}")
    offset = 0
    for (entry, key), count in zip(entries, counts):
        entry["_noised_rows"] = rows[:, offset:offset + count].to(device=entry[key].device, dtype=torch.float32)
        offset += count


def evaluation_conditions(conditional_dict: Dict[str, Any], *, condition_rows=None) -> Dict[str, Any]:
    """Copy mutable entry dictionaries, not latent tensors, for one fresh sample."""
    cond = dict(conditional_dict)
    for key in ("keyframes", "references"):
        cond[key] = [{k: v for k, v in entry.items() if k != "_noised_rows"} for entry in (conditional_dict.get(key) or [])]
    if condition_rows is not None:
        apply_condition_rows(cond, condition_rows)
    return cond


@torch.no_grad()
def sample_evaluation_ladder(
    wrapper,
    *,
    conditional_dict,
    video_shape,
    audio_shape,
    sigmas_v,
    sigmas_a,
    seed,
    device,
    noise=None,
    condition_rows=None,
    sampler="renoise",
    step_kwargs=None,
):
    """Sample the denoising schedule with isolated, replayable fp32 evaluation noise.

    Only the released re-noising sampler is supported.

    ``noise`` is an optional explicit (video, audio) initial-state pair; it never
    advances the independent re-noising or reference-augmentation generators.
    ``condition_rows`` is the only supported pre-noised visual-anchor override.
    """
    cond = evaluation_conditions(conditional_dict, condition_rows=condition_rows)
    device = torch.device(device)
    if noise is None:
        initial = torch.Generator(device=device).manual_seed(int(seed))
        video = torch.randn(video_shape, device=device, dtype=torch.float32, generator=initial)
        audio = torch.randn(audio_shape, device=device, dtype=torch.float32, generator=initial)
    else:
        if not isinstance(noise, (tuple, list)) or len(noise) != 2:
            raise ValueError("noise must be a (video, audio) tensor pair")
        states = []
        for name, value, shape in zip(("video", "audio"), noise, (video_shape, audio_shape)):
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
                raise ValueError(f"{name} noise must have shape {tuple(shape)}")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} noise must be a finite floating tensor")
            states.append(value.to(device=device, dtype=torch.float32))
        video, audio = states
    renoise = torch.Generator(device=device).manual_seed(int(seed) + 1)
    with evaluation_noise_scope(seed, device):
        xv, xa = few_step_sample(
            wrapper,
            video_noise=video,
            audio_noise=audio,
            conditional_dict=cond,
            sigmas_v=sigmas_v,
            sigmas_a=sigmas_a,
            noise_generator=renoise,
            sampler=sampler,
            step_kwargs=step_kwargs,
        )
    return xv.float(), xa.float()
