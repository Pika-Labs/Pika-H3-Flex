# SPDX-License-Identifier: Apache-2.0
"""Dilated Conv1d of the audio VAE decoder as an explicit fp32 im2col GEMM.

Under ``cudnn.deterministic`` MIOpen runs every dilated Conv1d of the audio decoder (kernel 7/11, dilation 3/5) with its naive
direct-convolution kernel (naive_conv_ab_nonpacked_fwd: 7-10 ms per call even at 8 channels; 218 of 294 ms of a 5 s audio decode).
This computes the same convolution in fp32 (zero padding, strided unfold, fp32 GEMM with TF32 off): deterministic for fixed shapes,
fp32 accumulation over Cin*k <= 5632 terms (the naive kernel accumulates in fp64; both are far inside the 40 dB decode gate)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _conv1d_gemm(m: torch.nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
    w = m.weight  # weight_norm pre-hooks have already recomputed it for this call
    B, Cin, L = x.shape
    k, d, p = m.kernel_size[0], m.dilation[0], m.padding[0]
    xp = F.pad(x, (p, p))
    Lp = L + 2 * p
    Lout = Lp - d * (k - 1)
    cols = xp.as_strided((B, Cin, k, Lout), (Cin * Lp, Lp, d, 1)).reshape(B, Cin * k, Lout)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = torch.matmul(w.reshape(w.shape[0], Cin * k), cols)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    if m.bias is not None:
        out = out + m.bias.view(1, -1, 1)
    return out


def patch_dilated_conv1d(module: torch.nn.Module) -> int:
    """Route every fp32 Conv1d with dilation > 1 (stride 1, zero padding, groups 1) through _conv1d_gemm. Returns the count."""
    n = 0
    for m in module.modules():
        if (isinstance(m, torch.nn.Conv1d) and m.dilation[0] > 1 and m.stride[0] == 1 and m.groups == 1
                and m.padding_mode == "zeros" and isinstance(m.padding, tuple)):
            m.forward = (lambda mm: (lambda x: _conv1d_gemm(mm, x) if x.dtype == torch.float32 else torch.nn.Conv1d.forward(mm, x)))(m)
            n += 1
    return n
