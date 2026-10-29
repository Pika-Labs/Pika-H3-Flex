# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Inference-only PEFT loading. Public manifests are validated by the release loader."""
from __future__ import annotations
from typing import Optional, Sequence
import torch
import torch.nn as nn

DEFAULT_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]

def build_transformer(path, *, dtype=torch.bfloat16, device="cpu", subfolder=None):
    from diffusers import MiniMaxH3Transformer3DModel
    kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if subfolder:
        kwargs["subfolder"] = subfolder
    model = MiniMaxH3Transformer3DModel.from_pretrained(str(path), **kwargs)
    model.requires_grad_(False)
    return model.to(device) if str(device) != "cpu" else model

def apply_lora(
    transformer: nn.Module,
    *,
    rank: int,
    alpha: Optional[int] = None,
    dropout: float = 0.0,
    target_modules: Optional[Sequence[str]] = None,
    adapter_name: str = "default",
    master_dtype: torch.dtype = torch.bfloat16,
    init_lora_weights: bool | str = True,
) -> nn.Module:
    from peft import LoraConfig

    alpha = int(rank if alpha is None else alpha)
    targets = list(target_modules or DEFAULT_LORA_TARGETS)
    transformer.requires_grad_(False)
    config = LoraConfig(r=int(rank), lora_alpha=alpha, lora_dropout=float(dropout), target_modules=targets, init_lora_weights=init_lora_weights)
    transformer.add_adapter(config, adapter_name=adapter_name)
    n = 0
    for name, param in transformer.named_parameters():
        if "lora_" in name:
            param.data = param.data.to(master_dtype)
            param.requires_grad_(False)
            n += 1
        else:
            param.requires_grad_(False)
    if n == 0:
        raise RuntimeError(f"apply_lora injected no adapter weights for targets {targets}")
    transformer._pika_lora = {"rank": int(rank), "alpha": alpha, "targets": targets, "adapter": adapter_name}
    return transformer


def load_lora_weights(transformer: nn.Module, path: str, *, strict: bool = True) -> None:
    from safetensors.torch import load_file

    state = load_file(path)
    own = dict(transformer.named_parameters())
    missing = [k for k in own if "lora_" in k and k not in state]
    unexpected = [k for k in state if k not in own]
    if strict and (missing or unexpected):
        raise RuntimeError(f"LoRA load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    with torch.no_grad():
        for k, v in state.items():
            if k in own:
                own[k].copy_(v.to(own[k].dtype))


