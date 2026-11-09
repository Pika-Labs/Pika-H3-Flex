# SPDX-License-Identifier: Apache-2.0
"""LOSSY compressed communication (lossy key "comm_fp8", off by default): the Ulysses all-to-alls of the world-8 SDMA paths
carry FP8 e4m3 rows with one fp32 scale per (row, 128-element head segment) instead of bf16 (payload + scales = 0.52x the bytes).
The receiver dequantizes into a bf16 buffer, so the attention / out-projection code is unchanged.

Layout inside an SDMA receive buffer sized for bf16 [R, C]: fp8 payload [R, C] at byte 0, fp32 scales [R, C / 128] at byte
off_scales(R, C) (256-byte aligned).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

F8 = torch.float8_e4m3fn
SEG = 128


@triton.jit
def _q_kernel(x_ptr, q_ptr, s_ptr, R, x_rs, G: tl.constexpr, BR: tl.constexpr, SEGC: tl.constexpr):
    pid = tl.program_id(0)
    g = tl.program_id(1)
    rows = pid * BR + tl.arange(0, BR)
    cols = g * SEGC + tl.arange(0, SEGC)
    m = rows[:, None] < R
    x = tl.load(x_ptr + rows[:, None] * x_rs + cols[None, :], mask=m, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    sc = tl.maximum(amax, 1e-12) / 448.0
    y = x / sc[:, None]
    tl.store(q_ptr + rows[:, None] * (G * SEGC) + cols[None, :], y.to(tl.float8e4nv), mask=m)
    tl.store(s_ptr + rows * G + g, sc, mask=rows < R)


@triton.jit
def _dq_kernel(q_ptr, s_ptr, y_ptr, R, y_rs, G: tl.constexpr, BR: tl.constexpr, SEGC: tl.constexpr):
    pid = tl.program_id(0)
    g = tl.program_id(1)
    rows = pid * BR + tl.arange(0, BR)
    cols = g * SEGC + tl.arange(0, SEGC)
    m = rows[:, None] < R
    q = tl.load(q_ptr + rows[:, None] * (G * SEGC) + cols[None, :], mask=m, other=0.0).to(tl.float32)
    sc = tl.load(s_ptr + rows * G + g, mask=rows < R, other=0.0)
    tl.store(y_ptr + rows[:, None] * y_rs + cols[None, :], (q * sc[:, None]).to(tl.bfloat16), mask=m)


def off_scales(R: int, C: int) -> int:
    return (R * C + 255) // 256 * 256


def quant_into(x, q, s):
    """x bf16 [R, C] (last dim contiguous, any row stride) -> q fp8 [R, C] contiguous, s fp32 [R, C/128]."""
    R, C = x.shape
    G = C // SEG
    if R:
        _q_kernel[(triton.cdiv(R, 16), G)](x, q, s, R, x.stride(0), G=G, BR=16, SEGC=SEG, num_warps=4)


def dequant_into(q, s, y):
    R, C = y.shape
    G = C // SEG
    if R:
        _dq_kernel[(triton.cdiv(R, 16), G)](q, s, y, R, y.stride(0), G=G, BR=16, SEGC=SEG, num_warps=4)


def views(cm, bi, R, C):
    """fp8 payload [R, C] and fp32 scales [R, C/128] views of SDMA buffer bi."""
    buf = cm.bufs[bi]
    pay = buf.view(torch.uint8)[:R * C].view(F8).view(R, C)
    o = off_scales(R, C)
    sc = buf.view(torch.uint8)[o:o + R * (C // SEG) * 4].view(torch.float32).view(R, C // SEG)
    return pay, sc
