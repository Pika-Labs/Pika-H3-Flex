# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Six-step consistency re-noising with fresh noise and terminal clean prediction."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch

AddNoiseFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def add_noise(original: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """``x_t = (1 - sigma) x0 + sigma eps`` in fp32, returned in ``original.dtype``. sigma: [B] or [B, T]."""
    if sigma.dim() == 1:
        sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
    elif sigma.dim() == 2:
        sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
    sigma = sigma.to(dtype=torch.float32)
    return ((1 - sigma) * original.float() + sigma * noise.float()).to(dtype=original.dtype)


SAMPLERS = ("renoise",)


def _expand(sigma: torch.Tensor, batch: int, frames: int, device) -> torch.Tensor:
    return sigma.to(device).float().reshape(1, 1).expand(batch, frames).contiguous()


def num_eval_steps(sigmas: torch.Tensor) -> int:
    """K = number of generator evaluations for a K + 1 knot ladder that ends in a terminal 0."""
    if int(len(sigmas)) < 2 or float(sigmas[-1]) != 0.0:
        raise ValueError(
            f"ladder must hold K + 1 knots ending in 0 (LTX convention, e.g. denoising_step_list [1000, 750, 500, 250, 0]); "
            f"got {[round(float(s), 4) for s in sigmas]}"
        )
    return int(len(sigmas)) - 1


@torch.no_grad()
def few_step_trajectory(
    generator,
    *,
    video_noise: Optional[torch.Tensor],
    audio_noise: Optional[torch.Tensor],
    conditional_dict: Dict[str, Any],
    sigmas_v: torch.Tensor,
    sigmas_a: torch.Tensor,
    stop_index: Optional[int] = None,
    return_selected_only: bool = False,
    frozen_video_latents: Optional[torch.Tensor] = None,
    frozen_audio_latents: Optional[torch.Tensor] = None,
    noise_generator: Optional[torch.Generator] = None,
    sampler: str = "renoise",
    step_kwargs: Optional[Callable[[int], Dict[str, Any]]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Roll the student down the ladder (K + 1 knots, terminal 0; the student is never evaluated at the 0).

    Returns either the stacked trajectory of generator INPUTS ``[B, K', ...]`` (K' = stop_index + 1, default K, i.e.
    one input per evaluation sigma), or, with ``return_selected_only``, just the input at ``stop_index``. Frozen
    branches are passed through untouched.
    """
    if len(sigmas_v) != len(sigmas_a):
        raise ValueError("video and audio ladders must be index-aligned")
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {' | '.join(SAMPLERS)}, got {sampler!r}")
    K = num_eval_steps(sigmas_v)
    max_index = K - 1 if stop_index is None else int(stop_index)
    if max_index < 0 or max_index >= K:
        raise ValueError(f"stop_index must be within [0, {K - 1}] (K = {K} evaluation sigmas), got {stop_index}")
    has_video = video_noise is not None
    has_audio = audio_noise is not None
    ref = video_noise if has_video else audio_noise
    if ref is None:
        raise ValueError("at least one generated branch is required")
    B = ref.shape[0]
    device = ref.device
    F_v = video_noise.shape[1] if has_video else 0
    F_a = audio_noise.shape[1] if has_audio else 0

    noisy_video = video_noise
    noisy_audio = audio_noise
    traj_video = [noisy_video] if has_video else None
    traj_audio = [noisy_audio] if has_audio else None

    for i in range(max_index):
        sigma_v = _expand(sigmas_v[i], B, F_v, device) if has_video else None
        sigma_a = _expand(sigmas_a[i], B, F_a, device) if has_audio else None
        pred_video, pred_audio = generator(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=sigma_v,
            noisy_audio=noisy_audio,
            audio_timestep=sigma_a,
            frozen_video_latents=frozen_video_latents,
            frozen_audio_latents=frozen_audio_latents,
            **({} if step_kwargs is None else step_kwargs(i)),
        )
        next_v = sigmas_v[i + 1]
        next_a = sigmas_a[i + 1]
        if has_video:
            eps = torch.randn(pred_video.shape, device=device, dtype=torch.float32, generator=noise_generator)
            noisy_video = add_noise(pred_video, eps, _expand(next_v, B, F_v, device))
            traj_video.append(noisy_video)
        if has_audio:
            eps = torch.randn(pred_audio.shape, device=device, dtype=torch.float32, generator=noise_generator)
            noisy_audio = add_noise(pred_audio, eps, _expand(next_a, B, F_a, device))
            traj_audio.append(noisy_audio)

    if return_selected_only:
        return (noisy_video if has_video else None), (noisy_audio if has_audio else None)
    return (
        torch.stack(traj_video, dim=1) if has_video else None,
        torch.stack(traj_audio, dim=1) if has_audio else None,
    )


@torch.no_grad()
def few_step_sample(
    generator,
    *,
    video_noise: Optional[torch.Tensor],
    audio_noise: Optional[torch.Tensor],
    conditional_dict: Dict[str, Any],
    sigmas_v: torch.Tensor,
    sigmas_a: torch.Tensor,
    frozen_video_latents: Optional[torch.Tensor] = None,
    frozen_audio_latents: Optional[torch.Tensor] = None,
    noise_generator: Optional[torch.Generator] = None,
    sampler: str = "renoise",
    step_kwargs: Optional[Callable[[int], Dict[str, Any]]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Full K-step sample: run the trajectory to the last evaluation sigma (``sigmas[K-1]``) and denoise -> clean x0.

    Identical to iterating ``sigmas[:-1]``: K generator calls, the terminal 0 knot is never fed to the model.
    """
    K = num_eval_steps(sigmas_v)
    noisy_video, noisy_audio = few_step_trajectory(
        generator,
        video_noise=video_noise,
        audio_noise=audio_noise,
        conditional_dict=conditional_dict,
        sigmas_v=sigmas_v,
        sigmas_a=sigmas_a,
        stop_index=K - 1,
        return_selected_only=True,
        frozen_video_latents=frozen_video_latents,
        frozen_audio_latents=frozen_audio_latents,
        noise_generator=noise_generator,
        sampler=sampler,
        step_kwargs=step_kwargs,
    )
    has_video = noisy_video is not None
    has_audio = noisy_audio is not None
    ref = noisy_video if has_video else noisy_audio
    B, device = ref.shape[0], ref.device
    sigma_v = _expand(sigmas_v[K - 1], B, noisy_video.shape[1], device) if has_video else None
    sigma_a = _expand(sigmas_a[K - 1], B, noisy_audio.shape[1], device) if has_audio else None
    return generator(
        noisy_image_or_video=noisy_video,
        conditional_dict=conditional_dict,
        timestep=sigma_v,
        noisy_audio=noisy_audio,
        audio_timestep=sigma_a,
        frozen_video_latents=frozen_video_latents,
        frozen_audio_latents=frozen_audio_latents,
        **({} if step_kwargs is None else step_kwargs(K - 1)),
    )
