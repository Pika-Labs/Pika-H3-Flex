# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Explicit conditioning precision and backend settings for inference.

FP32 conditioning preserves floating latent inputs in float32; BF16 conditioning
casts conditioning tensors to the prompt dtype. Both inference policies disable
TF32 matrix multiplication and enable cuDNN TF32."""
from __future__ import annotations

import torch

FP32_CONDITIONING = "fp32_conditioning_v1"
BF16_CONDITIONING = "bf16_conditioning_v1"
NUMERICS_POLICIES = (FP32_CONDITIONING, BF16_CONDITIONING)
LATENT_FIELDS = frozenset({"latents", "video_latents", "audio_latents"})


def validate_numerics_policy(policy: str) -> str:
    if policy not in NUMERICS_POLICIES:
        raise ValueError(f"unknown numerics_policy {policy!r}; expected one of {NUMERICS_POLICIES}")
    return policy


def numerics_backend_settings(policy: str, *, runtime: str) -> dict[str, bool]:
    """Return the fixed backend flags for a policy and execution entrypoint."""
    validate_numerics_policy(policy)
    if runtime != "inference":
        raise ValueError("numerics runtime must be 'inference'")
    return {"matmul_allow_tf32": False,
            "cudnn_allow_tf32": True}


def configure_numerics_backend(policy: str, *, runtime: str) -> dict[str, bool]:
    """Set process-global precision before model execution, including on CPU.

    Reset both flags explicitly before execution. These flags do not change
    parameter/state dtypes or the model's explicit BF16 module boundaries.
    """
    settings = numerics_backend_settings(policy, runtime=runtime)
    torch.backends.cuda.matmul.allow_tf32 = settings["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = settings["cudnn_allow_tf32"]
    return settings


def _condition_entry_to(entry, device, prompt_dtype, policy):
    result = {}
    for key, value in entry.items():
        if key == "_noised_rows":
            continue
        if torch.is_tensor(value):
            if policy == BF16_CONDITIONING:
                dtype = prompt_dtype
            else:
                dtype = torch.float32 if key in LATENT_FIELDS and value.is_floating_point() else value.dtype
            value = value.to(device=device, dtype=dtype)
        result[key] = value
    return result


def prepare_conditioning(cond, device, prompt_dtype=torch.bfloat16, *, policy=FP32_CONDITIONING):
    """Move a new request to its device and discard request-local augmentation caches.

    FP32 conditioning preserves integral tags, masks and non-latent metadata.
    """
    validate_numerics_policy(policy)
    result = dict(cond)
    result["prompt_embeds"] = cond["prompt_embeds"].to(device=device, dtype=prompt_dtype)
    result["text_token_tags"] = cond["text_token_tags"].to(device=device)
    for key in ("keyframes", "references"):
        result[key] = [_condition_entry_to(entry, device, prompt_dtype, policy) for entry in cond.get(key, [])]
    return result
