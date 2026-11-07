# SPDX-License-Identifier: Apache-2.0
"""Triton producers for the lower-precision attention (lowprec.py; LOSSY track, off by default).

quant_qkv_fp8(q4, k4, v4): q/k/v [1, N, h, 128] strided views (last dim contiguous) -> contiguous FP8 e4m3 [1, N, h, 128] each
plus fp32 per-tensor descales [1].  Q and K are rotated by the orthogonal Sylvester-Hadamard matrix H/sqrt(128) (q.k unchanged),
which spreads channel outliers.  Scales: pass 1 takes max_row ||q_row||_2 (a bound on every rotated entry) and amax|v| with
atomics; pass 2 rotates (tl.dot, fp32 acc), scales and casts.  Two reads of bf16 Q/K/V, one write of fp8.
"""
import torch
import triton
import triton.language as tl

F8 = torch.float8_e4m3fn
_H = {}


def hadamard_pm1(device):
    """Sylvester Hadamard 128x128 with +-1 entries (exact in bf16); the 1/sqrt(128) normalisation goes into the fp32 scale."""
    if device not in _H:
        h = torch.ones((1, 1), dtype=torch.float32)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H[device] = h.to(device=device, dtype=torch.bfloat16).contiguous()
    return _H[device]


@triton.jit
def _amax_kernel(q_ptr, k_ptr, v_ptr, sn, sh, N, h, acc_ptr, BM: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    hh = tl.program_id(1)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, D)
    m = rows[:, None] < N
    off = rows[:, None] * sn + hh * sh + cols[None, :]
    q = tl.load(q_ptr + off, mask=m, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + off, mask=m, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + off, mask=m, other=0.0).to(tl.float32)
    qn = tl.max(tl.sqrt(tl.sum(q * q, axis=1)), axis=0)
    kn = tl.max(tl.sqrt(tl.sum(k * k, axis=1)), axis=0)
    va = tl.max(tl.max(tl.abs(v), axis=1), axis=0)
    tl.atomic_max(acc_ptr + 0, qn)
    tl.atomic_max(acc_ptr + 1, kn)
    tl.atomic_max(acc_ptr + 2, va)


@triton.jit
def _quant_kernel(q_ptr, k_ptr, v_ptr, sn, sh, N, h, acc_ptr, hd_ptr, qo, ko, vo, ds_ptr,
                  BM: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr):
    pid = tl.program_id(0)
    hh = tl.program_id(1)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, D)
    m = rows[:, None] < N
    off = rows[:, None] * sn + hh * sh + cols[None, :]
    oo = rows[:, None] * (h * D) + hh * D + cols[None, :]
    qa = tl.maximum(tl.load(acc_ptr + 0), 1e-20)
    ka = tl.maximum(tl.load(acc_ptr + 1), 1e-20)
    va = tl.maximum(tl.load(acc_ptr + 2), 1e-20)
    # descale = bound / 448; with ROT the +-1 Hadamard output is sqrt(128) x the orthonormal rotation
    RS: tl.constexpr = 0.08838834764831845 if ROT else 1.0  # 1/sqrt(128)
    qd = qa / 448.0
    kd = ka / 448.0
    vd = va / 448.0
    if pid == 0 and hh == 0:
        tl.store(ds_ptr + 0, qd)
        tl.store(ds_ptr + 1, kd)
        tl.store(ds_ptr + 2, vd)
    q = tl.load(q_ptr + off, mask=m, other=0.0)
    k = tl.load(k_ptr + off, mask=m, other=0.0)
    v = tl.load(v_ptr + off, mask=m, other=0.0).to(tl.float32)
    if ROT:
        hm = tl.load(hd_ptr + cols[:, None] * D + cols[None, :])
        qr = tl.dot(q, hm) * (RS / qd)
        kr = tl.dot(k, hm) * (RS / kd)
    else:
        qr = q.to(tl.float32) / qd
        kr = k.to(tl.float32) / kd
    qr = tl.minimum(tl.maximum(qr, -448.0), 448.0)
    kr = tl.minimum(tl.maximum(kr, -448.0), 448.0)
    vr = tl.minimum(tl.maximum(v / vd, -448.0), 448.0)
    tl.store(qo + oo, qr.to(qo.dtype.element_ty), mask=m)
    tl.store(ko + oo, kr.to(ko.dtype.element_ty), mask=m)
    tl.store(vo + oo, vr.to(vo.dtype.element_ty), mask=m)


def quant_qkv_fp8(q4, k4, v4, rot=True, bm=64):
    """q4/k4/v4: [1, N, h, 128] views with identical strides (one packed receive buffer). Returns (qf, kf, vf, descale[3])."""
    _, N, h, D = q4.shape
    assert D == 128 and q4.stride() == k4.stride() == v4.stride() and q4.stride(3) == 1
    assert k4.shape[1] == N
    sn, sh = q4.stride(1), q4.stride(2)
    dev = q4.device
    acc = torch.zeros((3,), dtype=torch.float32, device=dev)
    ds = torch.empty((3,), dtype=torch.float32, device=dev)
    qf = torch.empty((1, N, h, D), dtype=F8, device=dev)
    kf = torch.empty_like(qf)
    vf = torch.empty_like(qf)
    grid = (triton.cdiv(N, bm), h)
    _amax_kernel[grid](q4, k4, v4, sn, sh, N, h, acc, BM=bm, D=D, num_warps=4)
    _quant_kernel[grid](q4, k4, v4, sn, sh, N, h, acc, hadamard_pm1(dev), qf, kf, vf, ds, BM=bm, D=D, ROT=rot, num_warps=4)
    return qf, kf, vf, ds


@triton.jit
def _rowq_kernel(x_ptr, sx, y_ptr, s_ptr, K, BK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BK)
    m = c < K
    x = tl.load(x_ptr + r * sx + c, mask=m, other=0.0).to(tl.float32)
    a = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
    d = a / 448.0
    tl.store(s_ptr + r, d)
    y = tl.minimum(tl.maximum(x / d, -448.0), 448.0)
    tl.store(y_ptr + r * K + c, y.to(y_ptr.dtype.element_ty), mask=m)


def quant_rows_fp8(x):
    """x [M, K] bf16 (row stride any, last dim contiguous) -> (fp8 e4m3 [M, K] contiguous, fp32 descale [M, 1])."""
    M, K = x.shape
    assert x.stride(1) == 1
    y = torch.empty((M, K), dtype=F8, device=x.device)
    s = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    BK = triton.next_power_of_2(K)
    _rowq_kernel[(M,)](x, x.stride(0), y, s, K, BK=BK, num_warps=8 if BK <= 8192 else 16)
    return y, s


@triton.jit
def _swiglu_q8_kernel(up_ptr, out_ptr, q_ptr, s_ptr, FF, stride_up, stride_out, BLOCK: tl.constexpr):
    """LOSSY producer: swiglu row (bf16, as kernels._swiglu_kernel) + its FP8 e4m3 per-row quantization in one pass (row in registers)."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < FF
    a = tl.load(up_ptr + row * stride_up + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(up_ptr + row * stride_up + FF + cols, mask=mask, other=0.0).to(tl.float32)
    sg = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    y = (a * sg).to(tl.bfloat16)
    if out_ptr is not None:
        tl.store(out_ptr + row * stride_out + cols, y, mask=mask)
    yf = y.to(tl.float32)
    d = tl.maximum(tl.max(tl.abs(yf), axis=0), 1e-12) / 448.0
    tl.store(s_ptr + row, d)
    tl.store(q_ptr + row * FF + cols, tl.minimum(tl.maximum(yf / d, -448.0), 448.0).to(q_ptr.dtype.element_ty), mask=mask)


def swiglu_q8(up, s, pad, keep_bf16=True):
    """up [Mp, 2FF] -> (bf16 out [pad, FF] or None, fp8 [pad, FF], descale [pad, 1]); rows >= s of the fp8 output are zero-scaled."""
    FF = up.shape[1] // 2
    out = torch.empty((pad, FF), dtype=torch.bfloat16, device=up.device) if keep_bf16 else None
    q = torch.empty((pad, FF), dtype=F8, device=up.device)
    sc = torch.zeros((pad, 1), dtype=torch.float32, device=up.device)
    if pad > s:
        q[s:].zero_()
    _swiglu_q8_kernel[(s,)](up, out, q, sc, FF, up.stride(0), FF, BLOCK=triton.next_power_of_2(FF), num_warps=16)
    return out, q, sc
