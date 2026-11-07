# SPDX-License-Identifier: Apache-2.0
"""Split-KV flash attention with fp32 partial outputs, for the few "tail" query rows that do not fill a whole wave of the
main asm launch (sp_model.tail_plan).

Why: the default tail path runs the bf16 flash kernel on KV chunks and merges the chunk outputs by their LSE. The chunk
outputs are rounded to bf16 before the merge, so merged rows carry a second bf16 rounding of partial accumulators
(measured on real Q/K/V: 53.5-53.7 dB vs 55.4-55.5 dB for unsplit rows, SNR against fp64). Here the partial
accumulators stay fp32 until the single final rounding, as in an unsplit kernel: S = QK^T in fp32 (bf16 inputs), online
softmax in fp32, P rounded to bf16 for PV (as the flash kernels do), PV accumulated in fp32.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _split_kv_fwd(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, Nt, Nk, cs, qk_scale,
                  sq_r, sq_h, sk_r, sk_h, sv_r, sv_h, so_c, so_h, so_r, sl_c, sl_h,
                  D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    mb = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.program_id(2)
    rows = mb * BM + tl.arange(0, BM)
    rmask = rows < Nt
    d = tl.arange(0, D)
    q = tl.load(q_ptr + rows[:, None] * sq_r + h * sq_h + d[None, :], mask=rmask[:, None], other=0.0)
    k0 = c * cs
    k1 = tl.minimum(k0 + cs, Nk)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for start in range(k0, k1, BN):
        cols = start + tl.arange(0, BN)
        cmask = cols < k1
        kt = tl.load(k_ptr + cols[None, :] * sk_r + h * sk_h + d[:, None], mask=cmask[None, :], other=0.0)  # [D, BN]
        s = tl.dot(q, kt) * qk_scale  # log2 domain (qk_scale = scale * log2 e)
        s = tl.where(cmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        vt = tl.load(v_ptr + cols[:, None] * sv_r + h * sv_h + d[None, :], mask=cmask[:, None], other=0.0)  # [BN, D]
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + c * so_c + h * so_h + rows[:, None] * so_r + d[None, :], o, mask=rmask[:, None])
    lse = (m_i + tl.math.log2(l_i)) * 0.6931471805599453  # natural log, like the flash kernels' LSE
    tl.store(lse_ptr + c * sl_c + h * sl_h + rows, lse, mask=rmask)


def split_kv_attention_fp32(q, k, v, chunks: int, BM: int = 64, BN: int = 64):
    """q [Nt, h, D], k/v [Nk, h, D] bf16 views (last dim contiguous). Returns o [C, h, Nt, D] fp32 (each chunk's normalised
    partial output) and lse [C, h, Nt] fp32 (natural log), for K.lse_merge_into."""
    Nt, h, D = q.shape
    Nk = k.shape[0]
    cs = triton.cdiv(triton.cdiv(Nk, chunks), BN) * BN
    C = triton.cdiv(Nk, cs)
    o = torch.empty((C, h, Nt, D), dtype=torch.float32, device=q.device)
    lse = torch.empty((C, h, Nt), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(Nt, BM), h, C)
    _split_kv_fwd[grid](q, k, v, o, lse, Nt, Nk, cs, D ** -0.5 * 1.4426950408889634,
                        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                        o.stride(0), o.stride(1), o.stride(2), lse.stride(0), lse.stride(1),
                        D=D, BM=BM, BN=BN, num_warps=4, num_stages=2)
    return o, lse
