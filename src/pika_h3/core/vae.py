# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
# Portions adapted from Diffusers MiniMax H3: decoders.py.
# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# The adapted upstream portions are licensed under Apache-2.0; see licenses/Apache-2.0.txt.
# Modified by Mellis, Inc. in 2026. Copyright © 2026 Mellis, Inc.
# Model Derivative of MiniMax H3, distributed under the MiniMax H3 Community License Agreement.
# Code remains licensed under Apache-2.0; the model terms above apply to the adapters.
# Changes: Adapted normalization and layout handling for standalone VAE decoding.
# Source revision: 3a2f35d4efa4c059c8bfb3bc0d6c906264895c81.

"""H3 VAE decode helpers for validation clips (latents are stored/normalized the pipeline's way)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from . import geometry as G


class H3Decoders:
    def __init__(self, checkpoint_dir: str, *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        # The released VAEs keep fp32 biases/norms; the pipeline runs them in fp32 (10 GB video VAE) — keep that.
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=dtype).to(device).eval()
        self.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(checkpoint_dir, subfolder="audio_vae", torch_dtype=dtype).to(device).eval()
        self.device = torch.device(device)
        self.dtype = dtype
        self.video_mean = torch.tensor(self.vae.config.latents_mean, device=self.device).view(1, -1, 1, 1, 1)
        self.video_std = torch.tensor(self.vae.config.latents_std, device=self.device).view(1, -1, 1, 1, 1)
        self.audio_mean = torch.tensor(self.audio_vae.config.latents_mean, device=self.device).view(1, -1, 1)
        self.audio_std = torch.tensor(self.audio_vae.config.latents_std, device=self.device).view(1, -1, 1)

    def to(self, device) -> "H3Decoders":
        self.device = torch.device(device)
        self.vae.to(device)
        self.audio_vae.to(device)
        self.video_mean = self.video_mean.to(device)
        self.video_std = self.video_std.to(device)
        self.audio_mean = self.audio_mean.to(device)
        self.audio_std = self.audio_std.to(device)
        return self

    @torch.no_grad()
    def decode_video(self, latents_external: torch.Tensor) -> np.ndarray:
        """``[F, 24, H, W]`` normalized -> uint8 frames ``[N, H*16, W*16, 3]``."""
        lat = latents_external.to(self.device, torch.float32).permute(1, 0, 2, 3).unsqueeze(0)
        lat = lat * self.video_std + self.video_mean
        video = self.vae.decode(lat.to(self.dtype), return_dict=False)[0]  # [1, 3, N, H, W] in pixel_mean/std space?
        video = video[0].float()
        # The pipeline's decode step reverts the VAE's own pixel normalization; mirror it if the VAE exposes it.
        mean = torch.tensor((0.485, 0.456, 0.406), device=video.device).view(3, 1, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=video.device).view(3, 1, 1, 1)
        # Match the official pipeline unconditionally; bright decoded tensors
        # may contain no value below -0.5 and still use ImageNet normalization.
        video = video * std + mean
        video = video.clamp(0, 1).permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).cpu().numpy()
        return video

    @torch.no_grad()
    def decode_audio(self, latents_external: torch.Tensor) -> np.ndarray:
        """``[T, 64]`` normalized -> float32 waveform ``[2, S]`` at 32 kHz."""
        from .packing import audio_external_to_storage

        lat = audio_external_to_storage(latents_external.to(self.device, torch.float32))  # [2, 32, T]
        lat = lat * self.audio_std + self.audio_mean
        audio = self.audio_vae.decode(lat.to(self.dtype), return_dict=False)[0]
        audio = audio.float().reshape(G.AUDIO_CHANNELS, -1).clamp(-1, 1).cpu().numpy()
        return audio


def write_mp4(path: str, frames: np.ndarray, fps: int = G.FPS, audio: Optional[np.ndarray] = None, sample_rate: int = G.AUDIO_SAMPLE_RATE) -> None:
    """Write frames (+ optional stereo audio) with PyAV."""
    import av

    container = av.open(path, mode="w")
    vstream = container.add_stream("libx264", rate=fps)
    vstream.width = int(frames.shape[2])
    vstream.height = int(frames.shape[1])
    vstream.pix_fmt = "yuv420p"
    astream = None
    if audio is not None:
        astream = container.add_stream("aac", rate=sample_rate)
        astream.layout = "stereo"
    for frame in frames:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        for packet in vstream.encode(vf):
            container.mux(packet)
    for packet in vstream.encode():
        container.mux(packet)
    if astream is not None:
        pcm = np.ascontiguousarray(audio.astype(np.float32))
        chunk = 1024
        for start in range(0, pcm.shape[1], chunk):
            af = av.AudioFrame.from_ndarray(pcm[:, start:start + chunk], format="fltp", layout="stereo")
            af.sample_rate = sample_rate
            af.pts = start
            for packet in astream.encode(af):
                container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)
    container.close()
