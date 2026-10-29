# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Six-forward consistency re-noising, matching the released students."""
import torch
from . import geometry as G
from .evaluation import sample_evaluation_ladder
from .schedules import H3SigmaSchedule

DEFAULT_KNOTS = (1000, 750, 500, 250, 100, 40, 0)

@torch.no_grad()
def sample_ladder(wrapper, cond, *, seed=0, device="cuda", knots=DEFAULT_KNOTS,
                  video_shift=12.0, audio_shift=3.0, noise=None, condition_rows=None):
    geom = cond["geometry"]
    schedule = H3SigmaSchedule({"denoising_step_list": list(knots),
        "video_score_sigma_shift": video_shift, "audio_score_sigma_shift": audio_shift}, device=device)
    return sample_evaluation_ladder(wrapper, conditional_dict=cond,
        video_shape=(1, geom.num_latent_frames, G.VIDEO_LATENT_CHANNELS, geom.latent_height, geom.latent_width),
        audio_shape=(1, geom.num_audio_latents, G.AUDIO_EXTERNAL_DIM),
        sigmas_v=schedule.video_ladder, sigmas_a=schedule.audio_ladder,
        seed=seed, device=device, noise=noise, condition_rows=condition_rows, sampler="renoise")
