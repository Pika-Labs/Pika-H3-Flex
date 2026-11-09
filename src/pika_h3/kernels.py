# SPDX-License-Identifier: Apache-2.0
"""Fused Triton kernels for the sequence-parallel MiniMax-H3 block.

Every kernel reproduces the *rounding points* of the reference bf16 op sequence (each intermediate tensor of the
reference is rounded to bf16, arithmetic inside an op is fp32): this is what keeps the output as close to the reference
as the stock kernel variants are (a numerically "better" fused version with fewer roundings is measurably farther).
"""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_rows(xf, w, H, eps, BLOCK: tl.constexpr):
    var = tl.sum(xf * xf, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    return (xf * rstd * w).to(tl.bfloat16)


# ------------------------------------------------------------------------------------------------------------------
# y = bf16(bf16(rms(x) * bf16(1 + scale[i])) ... ) exactly: n = bf16(rms*w); m = bf16(n * bf16(1+sc)); o = bf16(m + sh)
# optional residual update first: x_new = bf16(x + bf16(g_res[i] * y)); returns (x_new, o)
# ------------------------------------------------------------------------------------------------------------------
@triton.jit
def _mod_norm_kernel(x_ptr, y_ptr, gres_ptr, w_ptr, sc_ptr, sh_ptr, idx_ptr, xout_ptr, out_ptr,
                     H, eps, stride_row, stride_tab, stride_y, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    i = tl.load(idx_ptr + row).to(tl.int64)
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    if HAS_RES:
        y = tl.load(y_ptr + row * stride_y + cols, mask=mask, other=0.0)
        g = tl.load(gres_ptr + i * stride_tab + cols, mask=mask, other=0.0)
        gy = (g.to(tl.float32) * y.to(tl.float32)).to(tl.bfloat16)
        x = (x.to(tl.float32) + gy.to(tl.float32)).to(tl.bfloat16)
        tl.store(xout_ptr + row * stride_row + cols, x, mask=mask)
    xf = x.to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    n = _rms_rows(xf, w, H, eps, BLOCK)
    sc = tl.load(sc_ptr + i * stride_tab + cols, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(sh_ptr + i * stride_tab + cols, mask=mask, other=0.0).to(tl.float32)
    s1 = (1.0 + sc).to(tl.bfloat16)
    m = (n.to(tl.float32) * s1.to(tl.float32)).to(tl.bfloat16)
    o = (m.to(tl.float32) + sh).to(tl.bfloat16)
    tl.store(out_ptr + row * stride_row + cols, o, mask=mask)


def mod_norm(x, w, scale_tab, shift_tab, idx, eps, y=None, gate_tab=None, pad=None):
    """x [s,H] bf16; tabs [R,H] bf16; idx [s] int. With y/gate_tab: first x = x + gate[idx]*y (returned too).
    ``pad``: allocate the normalised output with that many rows (rows >= s are left unwritten; GEMM row padding)."""
    s, H = x.shape
    out = torch.empty((pad or s, H), dtype=x.dtype, device=x.device)
    has_res = y is not None
    xout = torch.empty_like(x) if has_res else x
    BLOCK = triton.next_power_of_2(H)
    _mod_norm_kernel[(s,)](x, y if has_res else x, gate_tab if has_res else scale_tab, w, scale_tab, shift_tab, idx, xout, out,
                           H, eps, x.stride(0), scale_tab.stride(0), y.stride(0) if has_res else x.stride(0),
                           HAS_RES=has_res, BLOCK=BLOCK, num_warps=4)
    return (xout, out) if has_res else out


@triton.jit
def _resid_kernel(x_ptr, y_ptr, gres_ptr, idx_ptr, out_ptr, H, stride_row, stride_tab, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    i = tl.load(idx_ptr + row).to(tl.int64)
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    y = tl.load(y_ptr + row * stride_y + cols, mask=mask, other=0.0)
    g = tl.load(gres_ptr + i * stride_tab + cols, mask=mask, other=0.0)
    gy = (g.to(tl.float32) * y.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + row * stride_row + cols, (x.to(tl.float32) + gy.to(tl.float32)).to(tl.bfloat16), mask=mask)


def resid(x, y, gate_tab, idx):
    s, H = x.shape
    out = torch.empty_like(x)
    _resid_kernel[(s,)](x, y, gate_tab, idx, out, H, x.stride(0), gate_tab.stride(0), y.stride(0), BLOCK=triton.next_power_of_2(H), num_warps=4)
    return out


# ------------------------------------------------------------------------------------------------------------------
# q/k: bf16 rms over head_dim (weights bf16), then rotary on the first RD dims; v copied. Output is written in the
# all-to-all send layout [W, s, 3, HPR*128] (destination rank g owns heads g*HPR .. g*HPR+HPR-1).
# ------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["s"])
def _qk_rope_kernel(qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, out_ptr, s, eps, s0, stride_in,
                    NH: tl.constexpr, HPR: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr, SPLIT: tl.constexpr = False):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, HD)
    g = head // HPR
    hl = head % HPR
    base_in = row * stride_in + head * HD
    # destination: ((g * s + row) * 3 + which) * (HPR*HD) + hl*HD
    if SPLIT:  # two row micro-batches [rows < s0 | rows >= s0], each destination-major (PIKA_SPLIT overlap)
        NG: tl.constexpr = NH // HPR
        drow = tl.where(row < s0, g * s0 + row, NG * s0 + g * (s - s0) + (row - s0))
        dst0 = (drow * 3) * (HPR * HD) + hl * HD
    else:
        dst0 = ((g * s + row) * 3) * (HPR * HD) + hl * HD
    HALF: tl.constexpr = RD // 2
    for which in tl.static_range(2):
        xin = tl.load(qkv_ptr + base_in + which * NH * HD + d).to(tl.float32)
        if which == 0:
            wgt = tl.load(qn_ptr + d).to(tl.float32)
        else:
            wgt = tl.load(kn_ptr + d).to(tl.float32)
        var = tl.sum(xin * xin, axis=0) / HD
        rstd = 1.0 / tl.sqrt(var + eps)
        n = (xin * rstd * wgt).to(tl.bfloat16)  # [HD] bf16 (rounded like nn.RMSNorm)
        # rotary: out[d] = n[d]*cos[d] + rot[d]*sin[d], rot = (-n[d+HALF], n[d-HALF]) for d < RD
        dpart = tl.where(d < HALF, d + HALF, d - HALF)
        npart = tl.load(qkv_ptr + base_in + which * NH * HD + dpart).to(tl.float32)
        # recompute the normalized value of the partner element (rms is over the whole head: reuse var/rstd)
        if which == 0:
            wpart = tl.load(qn_ptr + dpart).to(tl.float32)
        else:
            wpart = tl.load(kn_ptr + dpart).to(tl.float32)
        npart = (npart * rstd * wpart).to(tl.bfloat16).to(tl.float32)
        rot = tl.where(d < HALF, -npart, npart)
        inrd = d < RD
        c = tl.load(cos_ptr + row * RD + d, mask=inrd, other=1.0).to(tl.float32)
        sn = tl.load(sin_ptr + row * RD + d, mask=inrd, other=0.0).to(tl.float32)
        a = (n.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
        b = (rot * sn).to(tl.bfloat16).to(tl.float32)
        res = tl.where(inrd, (a + b).to(tl.bfloat16).to(tl.float32), n.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_ptr + dst0 + which * (HPR * HD) + d, res)
    v = tl.load(qkv_ptr + base_in + 2 * NH * HD + d)
    tl.store(out_ptr + dst0 + 2 * (HPR * HD) + d, v)


@triton.jit(do_not_specialize=["s"])
def _qk_rope_rows_kernel(qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, out_ptr, s, eps, s0, stride_in,
                         NH: tl.constexpr, HPR: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr, BR: tl.constexpr,
                         SPLIT: tl.constexpr = False):
    """Same per-element math as _qk_rope_kernel, BR rows per program (the one-row version is launch/latency bound)."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < s
    head = tl.program_id(1)
    d = tl.arange(0, HD)
    g = head // HPR
    hl = head % HPR
    base_in = rows[:, None] * stride_in + head * HD
    if SPLIT:
        NG: tl.constexpr = NH // HPR
        drow = tl.where(rows < s0, g * s0 + rows, NG * s0 + g * (s - s0) + (rows - s0))
        dst0 = (drow[:, None] * 3) * (HPR * HD) + hl * HD
    else:
        dst0 = ((g * s + rows[:, None]) * 3) * (HPR * HD) + hl * HD
    HALF: tl.constexpr = RD // 2
    m2 = rm[:, None]
    dpart = tl.where(d < HALF, d + HALF, d - HALF)
    inrd = d < RD
    c = tl.load(cos_ptr + rows[:, None] * RD + d[None, :], mask=m2 & inrd[None, :], other=1.0).to(tl.float32)
    sn = tl.load(sin_ptr + rows[:, None] * RD + d[None, :], mask=m2 & inrd[None, :], other=0.0).to(tl.float32)
    for which in tl.static_range(2):
        xin = tl.load(qkv_ptr + base_in + which * NH * HD + d[None, :], mask=m2, other=0.0).to(tl.float32)
        if which == 0:
            wgt = tl.load(qn_ptr + d).to(tl.float32)
            wpart = tl.load(qn_ptr + dpart).to(tl.float32)
        else:
            wgt = tl.load(kn_ptr + d).to(tl.float32)
            wpart = tl.load(kn_ptr + dpart).to(tl.float32)
        var = tl.sum(xin * xin, axis=1) / HD
        rstd = 1.0 / tl.sqrt(var + eps)
        n = (xin * rstd[:, None] * wgt[None, :]).to(tl.bfloat16)
        npart = tl.load(qkv_ptr + base_in + which * NH * HD + dpart[None, :], mask=m2, other=0.0).to(tl.float32)
        npart = (npart * rstd[:, None] * wpart[None, :]).to(tl.bfloat16).to(tl.float32)
        rot = tl.where(d[None, :] < HALF, -npart, npart)
        a = (n.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
        b = (rot * sn).to(tl.bfloat16).to(tl.float32)
        res = tl.where(inrd[None, :], (a + b).to(tl.bfloat16).to(tl.float32), n.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_ptr + dst0 + which * (HPR * HD) + d[None, :], res, mask=m2)
    vv = tl.load(qkv_ptr + base_in + 2 * NH * HD + d[None, :], mask=m2)
    tl.store(out_ptr + dst0 + 2 * (HPR * HD) + d[None, :], vv, mask=m2)


ROPE_ROWS = int(os.environ.get("PIKA_ROPE_ROWS", "8"))  # default 8 since 4-GPU eval 6.382; rows per program of qk_rope_pack; 0 = original one-row kernel


def qk_rope_pack(qkv, qn, kn, cos, sin, world, eps, NH=56, HD=128, s=None, s0=None):
    """s0 (optional): split the destination layout into row micro-batches [0, s0) and [s0, s), each destination-major."""
    s = qkv.shape[0] if s is None else s
    hpr = NH // world
    out = torch.empty((world * s, 3 * hpr * HD), dtype=qkv.dtype, device=qkv.device)
    if ROPE_ROWS > 0:
        _qk_rope_rows_kernel[(triton.cdiv(s, ROPE_ROWS), NH)](
            qkv, qn, kn, cos, sin, out, s, eps, 0 if s0 is None else s0, qkv.stride(0), NH=NH, HPR=hpr, HD=HD, RD=cos.shape[1],
            BR=ROPE_ROWS, SPLIT=s0 is not None, num_warps=4)
        return out
    if s0 is None:
        _qk_rope_kernel[(s, NH)](qkv, qn, kn, cos, sin, out, s, eps, 0, qkv.stride(0), NH=NH, HPR=hpr, HD=HD, RD=cos.shape[1],
                                 num_warps=1)
    else:
        _qk_rope_kernel[(s, NH)](qkv, qn, kn, cos, sin, out, s, eps, s0, qkv.stride(0), NH=NH, HPR=hpr, HD=HD, RD=cos.shape[1],
                                 SPLIT=True, num_warps=1)
    return out


# ------------------------------------------------------------------------------------------------------------------
# swiglu: a * bf16(silu(g)) with a = up[:, :F], g = up[:, F:]
# ------------------------------------------------------------------------------------------------------------------
@triton.jit
def _swiglu_kernel(up_ptr, out_ptr, FF, stride_up, stride_out, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < FF
    a = tl.load(up_ptr + row * stride_up + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(up_ptr + row * stride_up + FF + cols, mask=mask, other=0.0).to(tl.float32)
    sg = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * stride_out + cols, (a * sg).to(tl.bfloat16), mask=mask)


def swiglu(up, s=None, pad=None):
    """up [Mp, 2*FF]; processes the first s rows; output has ``pad`` rows (default s)."""
    F2 = up.shape[1]
    s = up.shape[0] if s is None else s
    FF = F2 // 2
    out = torch.empty((pad or s, FF), dtype=up.dtype, device=up.device)
    BLOCK = 4096
    _swiglu_kernel[(s, triton.cdiv(FF, BLOCK))](up, out, FF, up.stride(0), out.stride(0), BLOCK=BLOCK, num_warps=4)
    return out


# ------------------------------------------------------------------------------------------------------------------
# LoRA up-projection fused with the add:  y <- bf16(y + bf16(t @ B^T))  (in place).  t: [s, P*R] (P parts),
# B: [N, R] stacked rows; output column n uses part n // PART_N of t.
# ------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["M"])
def _lora_add_kernel(t_ptr, b_ptr, y_ptr, M, N, PART_N, stride_t, stride_y,
                     R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, R)
    part = (pid_n * BN) // PART_N
    t = tl.load(t_ptr + rm[:, None] * stride_t + part * R + rk[None, :], mask=rm[:, None] < M, other=0.0)
    b = tl.load(b_ptr + rn[None, :] * R + rk[:, None], mask=rn[None, :] < N, other=0.0)  # [R, BN]
    acc = tl.dot(t, b, out_dtype=tl.float32)
    lo = acc.to(tl.bfloat16).to(tl.float32)
    ptrs = y_ptr + rm[:, None] * stride_y + rn[None, :]
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    y = tl.load(ptrs, mask=msk, other=0.0).to(tl.float32)
    tl.store(ptrs, (y + lo).to(tl.bfloat16), mask=msk)


def lora_add_(y, t, B, part_n=None, R=128, M=None):
    N = y.shape[1]
    M = y.shape[0] if M is None else M
    part_n = part_n or N
    BM, BN = 128, 128
    _lora_add_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](t, B, y, M, N, part_n, t.stride(0), y.stride(0),
                                                                R=R, BM=BM, BN=BN, num_warps=8)
    return y


# ------------------------------------------------------------------------------------------------------------------
# Same as _qk_rope_kernel for an uneven head split: destination rank g owns hc[g] heads starting at hs[g]; the send buffer is
# the concatenation over g of [s, 3, hc[g]*HD] blocks (flat).
# ------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["s"])
def _qk_rope_var_kernel(qkv_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, out_ptr, hdest_ptr, hloc_ptr, hs_ptr, hc_ptr, s, eps,
                        NH: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, HD)
    g = tl.load(hdest_ptr + head).to(tl.int64)
    hl = tl.load(hloc_ptr + head).to(tl.int64)
    hs = tl.load(hs_ptr + g).to(tl.int64)
    hc = tl.load(hc_ptr + g).to(tl.int64)
    row64 = row.to(tl.int64)
    s64 = s.to(tl.int64)
    base_in = row64 * (3 * NH * HD) + head * HD
    dst0 = s64 * 3 * HD * hs + (row64 * 3) * (hc * HD) + hl * HD
    HALF: tl.constexpr = RD // 2
    for which in tl.static_range(2):
        xin = tl.load(qkv_ptr + base_in + which * NH * HD + d).to(tl.float32)
        if which == 0:
            wgt = tl.load(qn_ptr + d).to(tl.float32)
        else:
            wgt = tl.load(kn_ptr + d).to(tl.float32)
        var = tl.sum(xin * xin, axis=0) / HD
        rstd = 1.0 / tl.sqrt(var + eps)
        n = (xin * rstd * wgt).to(tl.bfloat16)
        dpart = tl.where(d < HALF, d + HALF, d - HALF)
        npart = tl.load(qkv_ptr + base_in + which * NH * HD + dpart).to(tl.float32)
        if which == 0:
            wpart = tl.load(qn_ptr + dpart).to(tl.float32)
        else:
            wpart = tl.load(kn_ptr + dpart).to(tl.float32)
        npart = (npart * rstd * wpart).to(tl.bfloat16).to(tl.float32)
        rot = tl.where(d < HALF, -npart, npart)
        inrd = d < RD
        c = tl.load(cos_ptr + row * RD + d, mask=inrd, other=1.0).to(tl.float32)
        sn = tl.load(sin_ptr + row * RD + d, mask=inrd, other=0.0).to(tl.float32)
        a = (n.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
        b = (rot * sn).to(tl.bfloat16).to(tl.float32)
        res = tl.where(inrd, (a + b).to(tl.bfloat16).to(tl.float32), n.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_ptr + dst0 + which * (hc * HD) + d, res)
    v = tl.load(qkv_ptr + base_in + 2 * NH * HD + d)
    tl.store(out_ptr + dst0 + 2 * (hc * HD) + d, v)


def qk_rope_pack_var(qkv, qn, kn, cos, sin, tables, eps, NH=56, HD=128, s=None):
    """tables = (head_dest, head_local, dest_hstart, dest_hcount) int32 device tensors. Returns flat send buffer."""
    s = qkv.shape[0] if s is None else s
    out = torch.empty((s * 3 * NH * HD,), dtype=qkv.dtype, device=qkv.device)
    hdest, hloc, hs, hc = tables
    _qk_rope_var_kernel[(s, NH)](qkv, qn, kn, cos, sin, out, hdest, hloc, hs, hc, s, eps, NH=NH, HD=HD, RD=cos.shape[1], num_warps=1)
    return out


# ------------------------------------------------------------------------------------------------------------------
# Exact flash-decoding style merge of partial attention results (KV split into chunks) for the tail query rows:
#   out = sum_c exp(lse_c - m) * o_c / sum_c exp(lse_c - m),  m = max_c lse_c   (fp32 math, one bf16 rounding)
# o1: [C1, H, Nt, D] bf16 (strides given), lse1: [C1, H, Nt(+pad)] fp32 ; o2/lse2: optional single extra chunk.
# Output rows are written to out[row0 + r, h, :] with out of layout [N, H, D] (contiguous).
# ------------------------------------------------------------------------------------------------------------------
@triton.jit
def _lse_merge_kernel(o1_ptr, l1_ptr, o2_ptr, l2_ptr, out_ptr, row0, C1,
                      so1_c, so1_h, so1_r, sl1_c, sl1_h, so2_h, so2_r, sl2_h,
                      H: tl.constexpr, D: tl.constexpr, HAS2: tl.constexpr, CBLOCK: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    c = tl.arange(0, CBLOCK)
    cm = c < C1
    l1 = tl.load(l1_ptr + c * sl1_c + h * sl1_h + r, mask=cm, other=float("-inf"))
    m = tl.max(l1, axis=0)
    if HAS2:
        l2 = tl.load(l2_ptr + h * sl2_h + r)
        m = tl.maximum(m, l2)
    w1 = tl.where(cm, tl.exp(l1 - m), 0.0)
    o1 = tl.load(o1_ptr + c[:, None] * so1_c + h * so1_h + r * so1_r + d[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
    num = tl.sum(o1 * w1[:, None], axis=0)
    den = tl.sum(w1, axis=0)
    if HAS2:
        w2 = tl.exp(l2 - m)
        o2 = tl.load(o2_ptr + h * so2_h + r * so2_r + d).to(tl.float32)
        num += o2 * w2
        den += w2
    tl.store(out_ptr + ((row0 + r) * H + h) * D + d, (num / den).to(tl.bfloat16))


@triton.jit
def _lse_merge_rows_kernel(o1_ptr, l1_ptr, o2_ptr, l2_ptr, out_ptr, row0, C1, Nt,
                           so1_c, so1_h, so1_r, sl1_c, sl1_h, so2_h, so2_r, sl2_h,
                           H: tl.constexpr, D: tl.constexpr, HAS2: tl.constexpr, CBLOCK: tl.constexpr, BR: tl.constexpr):
    """Same math and per-element operation order as _lse_merge_kernel, BR rows per program (the one-row version runs at
    ~1/4 of the memory roofline for the PIKA_SPLIT m0 merge: 15k rows x heads programs)."""
    rb = tl.program_id(0)
    h = tl.program_id(1)
    r = rb * BR + tl.arange(0, BR)
    rm = r < Nt
    d = tl.arange(0, D)
    c = tl.arange(0, CBLOCK)
    cm = c < C1
    l1 = tl.load(l1_ptr + c[:, None] * sl1_c + h * sl1_h + r[None, :], mask=cm[:, None] & rm[None, :], other=float("-inf"))
    m = tl.max(l1, axis=0)  # [BR]
    if HAS2:
        l2 = tl.load(l2_ptr + h * sl2_h + r, mask=rm, other=0.0)
        m = tl.maximum(m, l2)
    w1 = tl.where(cm[:, None], tl.exp(l1 - m[None, :]), 0.0)  # [CBLOCK, BR]
    o1 = tl.load(o1_ptr + c[:, None, None] * so1_c + h * so1_h + r[None, :, None] * so1_r + d[None, None, :],
                 mask=cm[:, None, None] & rm[None, :, None], other=0.0).to(tl.float32)
    num = tl.sum(o1 * w1[:, :, None], axis=0)  # [BR, D]
    den = tl.sum(w1, axis=0)
    if HAS2:
        w2 = tl.exp(l2 - m)
        o2 = tl.load(o2_ptr + h * so2_h + r[:, None] * so2_r + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        num += o2 * w2[:, None]
        den += w2
    tl.store(out_ptr + ((row0 + r[:, None]) * H + h) * D + d[None, :], (num / den[:, None]).to(tl.bfloat16), mask=rm[:, None])


MERGE_ROWS = int(os.environ.get("PIKA_MERGE_ROWS", "16"))  # default 16 (4-GPU eval 6.347, gate pass); rows per program of the LSE merge; 0 = original one-row kernel


def lse_merge_into(out, row0, o1, lse1, o2=None, lse2=None):
    """out [N, H, D] bf16 contiguous; o1 [C1, H, Nt, D] (any strides), lse1 [C1, H, >=Nt] fp32."""
    C1, H, Nt, D = o1.shape
    has2 = o2 is not None
    if has2:   # o2 [H, Nt, D], lse2 [H, Nt(+pad)]
        a = (o2, lse2, o2.stride(0), o2.stride(1), lse2.stride(0))
    else:
        a = (o1, lse1, 0, 0, 0)
    if MERGE_ROWS > 0:
        _lse_merge_rows_kernel[(triton.cdiv(Nt, MERGE_ROWS), H)](
            o1, lse1, a[0], a[1], out, row0, C1, Nt, o1.stride(0), o1.stride(1), o1.stride(2), lse1.stride(0), lse1.stride(1),
            a[2], a[3], a[4], H=H, D=D, HAS2=has2, CBLOCK=triton.next_power_of_2(C1), BR=MERGE_ROWS, num_warps=4)
        return out
    _lse_merge_kernel[(Nt, H)](o1, lse1, a[0], a[1], out, row0, C1,
                               o1.stride(0), o1.stride(1), o1.stride(2), lse1.stride(0), lse1.stride(1),
                               a[2], a[3], a[4],
                               H=H, D=D, HAS2=has2, CBLOCK=triton.next_power_of_2(C1))
    return out


# ------------------------------------------------------------------------------------------------------------------
# PIKA_FUSE_LORA: the LoRA-B up-projection + add of the qkv / up linears folded into their
# consumers, removing the read-modify-write pass over y.  Rounding points are the reference ones:
#   lo = bf16(t_part @ B^T) ; y' = bf16(y + lo) ; then the consumer's own math on y' (identical to lora_add_ + consumer).
# ------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["M"])
def _swiglu_lora_kernel(up_ptr, t_ptr, b_ptr, out_ptr, M, FF, stride_up, stride_t, stride_out,
                        R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, FASTDIV: tl.constexpr = False):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, R)
    mm = rm[:, None] < M
    t = tl.load(t_ptr + rm[:, None] * stride_t + rk[None, :], mask=mm, other=0.0)
    ba = tl.load(b_ptr + rn[None, :] * R + rk[:, None])            # [R, BN] rows of B for the 'a' half
    bg = tl.load(b_ptr + (FF + rn[None, :]) * R + rk[:, None])     # gate half
    la = tl.dot(t, ba, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
    lg = tl.dot(t, bg, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
    pa = up_ptr + rm[:, None] * stride_up + rn[None, :]
    a = (tl.load(pa, mask=mm, other=0.0).to(tl.float32) + la).to(tl.bfloat16).to(tl.float32)
    g = (tl.load(pa + FF, mask=mm, other=0.0).to(tl.float32) + lg).to(tl.bfloat16).to(tl.float32)
    if FASTDIV:
        sg = tl.fdiv(g, 1.0 + tl.exp(-g), ieee_rounding=False).to(tl.bfloat16).to(tl.float32)
    else:
        sg = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], (a * sg).to(tl.bfloat16), mask=mm)


SWIGLU_FASTDIV = os.environ.get("PIKA_SWIGLU_FASTDIV", "0") == "1"


def swiglu_lora(up, t, B, s=None, pad=None, BM=128, BN=64, num_warps=4):
    """== swiglu(lora_add_(up, t, B, M=s), s, pad) without writing up back. up [Mp, 2FF] base GEMM output, t [Mp, R], B [2FF, R]."""
    F2 = up.shape[1]
    s = up.shape[0] if s is None else s
    FF = F2 // 2
    R = B.shape[1]
    assert FF % BN == 0 and B.is_contiguous()
    out = torch.empty((pad or s, FF), dtype=up.dtype, device=up.device)
    _swiglu_lora_kernel[(triton.cdiv(s, BM), FF // BN)](up, t, B, out, s, FF, up.stride(0), t.stride(0), out.stride(0),
                                                         R=R, BM=BM, BN=BN, FASTDIV=SWIGLU_FASTDIV, num_warps=num_warps)
    return out


@triton.jit
def _lora_cols(t, b_ptr, qkv_ptr, row_in, mm, col0, c, cm, R: tl.constexpr):
    """y'[:, col0 + c] = bf16(y + bf16(t @ B[col0 + c]^T)) for a (masked) column group c, fp32 tensor out."""
    rk = tl.arange(0, R)
    bt = tl.load(b_ptr + (col0 + c[None, :]) * R + rk[:, None], mask=cm[None, :], other=0.0)
    lo = tl.dot(t, bt, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
    y = tl.load(qkv_ptr + row_in + col0 + c[None, :], mask=mm & cm[None, :], other=0.0).to(tl.float32)
    return (y + lo).to(tl.bfloat16).to(tl.float32)


@triton.jit(do_not_specialize=["s", "row0", "row1"])
def _qk_rope_lora_kernel(qkv_ptr, t_ptr, b_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, out_ptr, s, eps, stride_t, stride_in, s0,
                         row0, row1,
                         NH: tl.constexpr, HPR: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr, R: tl.constexpr,
                         BM: tl.constexpr, SPLIT: tl.constexpr = False):
    # head dims split in groups A = [0, HALF), B = [HALF, RD) (rotated pair, HALF = 48 -> 64-wide masked tiles) and
    # C = [RD, HD) (pass-through, 32 wide): rot(A) = -n(B), rot(B) = n(A) elementwise, no partner recomputation.
    # Rows [row0, row1) of the s local rows; qkv / t hold rows from row0 on (row-chunked GEMMs, PIKA_SDMA_PIPE).
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    rm = row0 + pid_m * BM + tl.arange(0, BM)
    mm = rm[:, None] < row1
    rm64 = rm.to(tl.int64)
    ri64 = rm64 - row0
    HALF: tl.constexpr = RD // 2
    GA: tl.constexpr = 64
    GC: tl.constexpr = HD - RD
    ca = tl.arange(0, GA)
    cam = ca < HALF
    cc = RD + tl.arange(0, GC)
    ccm = cc < HD
    rk = tl.arange(0, R)
    g = head // HPR
    hl = head % HPR
    if SPLIT:  # PIKA_SPLIT micro-batch layout, as _qk_rope_kernel
        NG: tl.constexpr = NH // HPR
        drow = tl.where(rm64 < s0, g * s0 + rm64, NG * s0 + g * (s - s0) + (rm64 - s0))
        dst = (drow[:, None] * 3) * (HPR * HD) + hl * HD
    else:
        dst = ((g * s + rm64[:, None]) * 3) * (HPR * HD) + hl * HD
    row_in = ri64[:, None] * stride_in
    for which in tl.static_range(3):
        col0 = which * NH * HD + head * HD
        t = tl.load(t_ptr + ri64[:, None] * stride_t + which * R + rk[None, :], mask=mm, other=0.0)
        ya = _lora_cols(t, b_ptr, qkv_ptr, row_in, mm, col0, ca, cam, R)
        yb = _lora_cols(t, b_ptr, qkv_ptr, row_in, mm, col0, ca + HALF, cam, R)
        yc = _lora_cols(t, b_ptr, qkv_ptr, row_in, mm, col0, cc, ccm, R)
        o_ptr = out_ptr + dst + which * (HPR * HD)
        if which == 2:
            tl.store(o_ptr + ca[None, :], ya.to(tl.bfloat16), mask=mm & cam[None, :])
            tl.store(o_ptr + HALF + ca[None, :], yb.to(tl.bfloat16), mask=mm & cam[None, :])
            tl.store(o_ptr + cc[None, :], yc.to(tl.bfloat16), mask=mm & ccm[None, :])
        else:
            w_ptr = qn_ptr if which == 0 else kn_ptr
            wa = tl.load(w_ptr + ca, mask=cam, other=0.0).to(tl.float32)
            wb = tl.load(w_ptr + HALF + ca, mask=cam, other=0.0).to(tl.float32)
            wc = tl.load(w_ptr + cc, mask=ccm, other=0.0).to(tl.float32)
            var = (tl.sum(ya * ya, axis=1) + tl.sum(yb * yb, axis=1) + tl.sum(yc * yc, axis=1)) / HD
            rstd = 1.0 / tl.sqrt(var + eps)
            na = (ya * rstd[:, None] * wa[None, :]).to(tl.bfloat16).to(tl.float32)
            nb = (yb * rstd[:, None] * wb[None, :]).to(tl.bfloat16).to(tl.float32)
            nc = (yc * rstd[:, None] * wc[None, :]).to(tl.bfloat16)
            cm2 = mm & cam[None, :]
            c_a = tl.load(cos_ptr + rm64[:, None] * RD + ca[None, :], mask=cm2, other=1.0).to(tl.float32)
            s_a = tl.load(sin_ptr + rm64[:, None] * RD + ca[None, :], mask=cm2, other=0.0).to(tl.float32)
            c_b = tl.load(cos_ptr + rm64[:, None] * RD + HALF + ca[None, :], mask=cm2, other=1.0).to(tl.float32)
            s_b = tl.load(sin_ptr + rm64[:, None] * RD + HALF + ca[None, :], mask=cm2, other=0.0).to(tl.float32)
            ra = ((na * c_a).to(tl.bfloat16).to(tl.float32) + (-nb * s_a).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
            rb = ((nb * c_b).to(tl.bfloat16).to(tl.float32) + (na * s_b).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
            tl.store(o_ptr + ca[None, :], ra, mask=cm2)
            tl.store(o_ptr + HALF + ca[None, :], rb, mask=cm2)
            tl.store(o_ptr + cc[None, :], nc, mask=mm & ccm[None, :])


def qk_rope_pack_lora(qkv, t, B, qn, kn, cos, sin, world, eps, NH=56, HD=128, s=None, BM=64, num_warps=4, s0=None, rows=None,
                      out=None):
    """== qk_rope_pack(lora_add_(qkv, t, B, part_n=NH*HD, M=s), ...) without the write-back of qkv.
    rows=(r0, r1): only local rows [r0, r1) (qkv / t start at row r0), written into ``out`` (the full send layout)."""
    if ROPE_PAIR_HIP:
        from . import hipk
        return hipk.rope_pair(qkv, t, B, qn, kn, cos, sin, world, eps, NH=NH, HD=HD, s=s, s0=s0, rows=rows, out=out)
    if ROPE_PAIR:
        return qk_rope_pack_lora_pair(qkv, t, B, qn, kn, cos, sin, world, eps, NH=NH, HD=HD, s=s, s0=s0, rows=rows, out=out)
    s = qkv.shape[0] if s is None else s
    hpr = NH // world
    R = B.shape[1]
    assert B.is_contiguous() and cos.is_contiguous() and sin.is_contiguous() and qkv.stride(1) == 1
    if out is None:
        out = torch.empty((world * s, 3 * hpr * HD), dtype=qkv.dtype, device=qkv.device)
    r0, r1 = rows if rows is not None else (0, s)
    _qk_rope_lora_kernel[(triton.cdiv(r1 - r0, BM), NH)](qkv, t, B, qn, kn, cos, sin, out, s, eps, t.stride(0), qkv.stride(0),
                                                         0 if s0 is None else s0, r0, r1, NH=NH, HPR=hpr, HD=HD, RD=cos.shape[1],
                                                         R=R, BM=BM, SPLIT=s0 is not None, num_warps=num_warps)
    return out


# ------------------------------------------------------------------------------------------------------------------
# PIKA_FUSE_LORA2: the LoRA-B add of the o / down linears folded into the following mod_norm,
# removing lora_add_'s read-modify-write pass over y.  Same rounding points as lora_add_ + mod_norm:
#   lo = bf16(t @ B^T); y' = bf16(y + lo); gy = bf16(g * y'); x' = bf16(x + gy); h = bf16(bf16(bf16(rms(x') * w) * bf16(1+sc)) + sh)
# Row tile of BM rows, two passes over H in BN-column chunks (pass 1: LoRA + residual, x' stored, sum of squares; pass 2: norm).
# The row sum of squares is accumulated chunk-wise (mod_norm: one tree sum) -> rstd may differ in the last bit.
# ------------------------------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["M"])
def _mod_norm_lora_kernel(x_ptr, y_ptr, t_ptr, b_ptr, gres_ptr, w_ptr, sc_ptr, sh_ptr, idx_ptr, xout_ptr, out_ptr,
                          M, eps, stride_row, stride_tab, stride_y, stride_t,
                          H: tl.constexpr, R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    mm = rm < M
    rk = tl.arange(0, R)
    i = tl.load(idx_ptr + rm, mask=mm, other=0).to(tl.int64)
    t = tl.load(t_ptr + rm[:, None] * stride_t + rk[None, :], mask=mm[:, None], other=0.0)
    ss = tl.zeros((BM,), dtype=tl.float32)
    for c0 in range(0, H, BN):
        rn = c0 + tl.arange(0, BN)
        b = tl.load(b_ptr + rn[None, :] * R + rk[:, None])                      # [R, BN]
        lo = tl.dot(t, b, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
        y = tl.load(y_ptr + rm[:, None] * stride_y + rn[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
        y = (y + lo).to(tl.bfloat16).to(tl.float32)
        g = tl.load(gres_ptr + i[:, None] * stride_tab + rn[None, :]).to(tl.float32)
        gy = (g * y).to(tl.bfloat16).to(tl.float32)
        x = tl.load(x_ptr + rm[:, None] * stride_row + rn[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
        xn = (x + gy).to(tl.bfloat16)
        tl.store(xout_ptr + rm[:, None] * stride_row + rn[None, :], xn, mask=mm[:, None])
        xf = xn.to(tl.float32)
        ss += tl.sum(xf * xf, axis=1)
    rstd = 1.0 / tl.sqrt(ss / H + eps)
    for c0 in range(0, H, BN):
        rn = c0 + tl.arange(0, BN)
        xf = tl.load(xout_ptr + rm[:, None] * stride_row + rn[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + rn).to(tl.float32)
        n = (xf * rstd[:, None] * w[None, :]).to(tl.bfloat16)
        sc = tl.load(sc_ptr + i[:, None] * stride_tab + rn[None, :]).to(tl.float32)
        sh = tl.load(sh_ptr + i[:, None] * stride_tab + rn[None, :]).to(tl.float32)
        s1 = (1.0 + sc).to(tl.bfloat16)
        m = (n.to(tl.float32) * s1.to(tl.float32)).to(tl.bfloat16)
        o = (m.to(tl.float32) + sh).to(tl.bfloat16)
        tl.store(out_ptr + rm[:, None] * stride_row + rn[None, :], o, mask=mm[:, None])


def mod_norm_lora(x, w, scale_tab, shift_tab, idx, eps, y, t, B, gate_tab, pad=None, BM=64, BN=256, num_warps=8):
    """== mod_norm(x, ..., y=lora_add_(y, t, B, M=s), gate_tab=gate_tab, pad=pad) without writing y back.
    x [s,H] bf16; y [>=s, H] base GEMM output; t [>=s, R] bf16(x A^T); B [H, R]."""
    s, H = x.shape
    R = B.shape[1]
    assert H % BN == 0 and B.is_contiguous() and x.stride(0) == H
    out = torch.empty((pad or s, H), dtype=x.dtype, device=x.device)
    xout = torch.empty_like(x)
    _mod_norm_lora_kernel[(triton.cdiv(s, BM),)](x, y, t, B, gate_tab, w, scale_tab, shift_tab, idx, xout, out, s, eps,
                                                 x.stride(0), scale_tab.stride(0), y.stride(0), t.stride(0),
                                                 H=H, R=R, BM=BM, BN=BN, num_warps=num_warps)
    return xout, out



# swiglu_lora with NC column blocks per program (t [BM, R] loaded once per program instead of once per BN columns:
# the one-block version re-reads t FF/BN = 224 times per row tile from L2). Same per-element math as _swiglu_lora_kernel.
@triton.jit(do_not_specialize=["M"])
def _swiglu_lora_nc_kernel(up_ptr, t_ptr, b_ptr, out_ptr, M, FF, stride_up, stride_t, stride_out,
                           R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NC: tl.constexpr, DOT: tl.constexpr = True,
                           BT: tl.constexpr = False):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rk = tl.arange(0, R)
    mm = rm[:, None] < M
    t = tl.load(t_ptr + rm[:, None] * stride_t + rk[None, :], mask=mm, other=0.0)
    for j in range(NC):
        rn = (pid_n * NC + j) * BN + tl.arange(0, BN)
        if BT:  # B^T [R, 2FF] (pre-transposed weight): loads contiguous along the N columns
            ba = tl.load(b_ptr + rk[:, None] * (2 * FF) + rn[None, :])
            bg = tl.load(b_ptr + rk[:, None] * (2 * FF) + FF + rn[None, :])
        else:
            ba = tl.load(b_ptr + rn[None, :] * R + rk[:, None])
            bg = tl.load(b_ptr + (FF + rn[None, :]) * R + rk[:, None])
        if DOT:
            la = tl.dot(t, ba, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
            lg = tl.dot(t, bg, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
        else:
            la = tl.zeros((BM, BN), dtype=tl.float32); lg = la
        pa = up_ptr + rm[:, None] * stride_up + rn[None, :]
        a = (tl.load(pa, mask=mm, other=0.0).to(tl.float32) + la).to(tl.bfloat16).to(tl.float32)
        g = (tl.load(pa + FF, mask=mm, other=0.0).to(tl.float32) + lg).to(tl.bfloat16).to(tl.float32)
        sg = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + rm[:, None] * stride_out + rn[None, :], (a * sg).to(tl.bfloat16), mask=mm)


def swiglu_lora_nc(up, t, B, s=None, pad=None, BM=128, BN=64, NC=8, num_warps=4, DOT=True, num_stages=1, wpe=0, Bt=None):
    """Bt: optional B^T [R, 2FF] contiguous (same values; coalesced loads)."""
    F2 = up.shape[1]
    s = up.shape[0] if s is None else s
    FF = F2 // 2
    R = B.shape[1]
    assert FF % (BN * NC) == 0 and B.is_contiguous()
    out = torch.empty((pad or s, FF), dtype=up.dtype, device=up.device)
    _swiglu_lora_nc_kernel[(triton.cdiv(s, BM), FF // (BN * NC))](up, t, B if Bt is None else Bt, out, s, FF, up.stride(0), t.stride(0), out.stride(0),
                                                                   R=R, BM=BM, BN=BN, NC=NC, DOT=DOT, BT=Bt is not None, num_warps=num_warps,
                                                                   num_stages=num_stages, **({'waves_per_eu': wpe} if wpe else {}))
    return out


# ------------------------------------------------------------------------------------------------------------------
# PIKA_ROPE_PAIR: q/k head dims stored in pair-interleaved order (a consistent permutation of the
# q and k head dims, applied to the W / LoRA-B rows and the q/k norm weights at load time; attention scores are invariant to
# it): new column 2p = dim p, 2p+1 = dim HALF+p (p < HALF), the pass-through dims follow in order.  The RoPE partner is then
# the other element of a column pair: a reshape/split, no gather, one full-width [BM, HD] LoRA dot per part.
# Same rounding points as qk_rope_pack_lora; the RMS sum is a tree over the 128 dims.
# ------------------------------------------------------------------------------------------------------------------
def pair_perm(HD=128, RD=96):
    """perm[j] = original head dim stored at new column j."""
    half = RD // 2
    p = []
    for i in range(HD // 2):
        if i < half:
            p += [i, half + i]
        else:
            p += [RD + 2 * (i - half), RD + 2 * (i - half) + 1]
    return p


@triton.jit(do_not_specialize=["s", "row0", "row1"])
def _qk_rope_pair_kernel(qkv_ptr, t_ptr, b_ptr, qn_ptr, kn_ptr, cos_ptr, sin_ptr, out_ptr, s, eps, stride_t, stride_in, s0,
                         row0, row1,
                         NH: tl.constexpr, HPR: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr, R: tl.constexpr,
                         BM: tl.constexpr, SPLIT: tl.constexpr = False, DOT: tl.constexpr = True, MATH: tl.constexpr = True,
                         BT: tl.constexpr = False, NB_: tl.constexpr = 0):
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    rm = row0 + pid_m * BM + tl.arange(0, BM)
    m2 = rm[:, None] < row1
    rm64 = rm.to(tl.int64)
    ri64 = rm64 - row0
    HALF: tl.constexpr = RD // 2
    P: tl.constexpr = HD // 2
    d = tl.arange(0, HD)
    pp = tl.arange(0, P)
    inr = pp < HALF
    rk = tl.arange(0, R)
    g = head // HPR
    hl = head % HPR
    if SPLIT:
        NG: tl.constexpr = NH // HPR
        drow = tl.where(rm64 < s0, g * s0 + rm64, NG * s0 + g * (s - s0) + (rm64 - s0))
        dst = (drow[:, None] * 3) * (HPR * HD) + hl * HD
    else:
        dst = ((g * s + rm64[:, None]) * 3) * (HPR * HD) + hl * HD
    row_in = ri64[:, None] * stride_in
    mp = m2 & inr[None, :]
    for which in tl.static_range(3):
        col0 = which * NH * HD + head * HD
        t = tl.load(t_ptr + ri64[:, None] * stride_t + which * R + rk[None, :], mask=m2, other=0.0)
        y = tl.load(qkv_ptr + row_in + col0 + d[None, :], mask=m2, other=0.0).to(tl.float32)
        if DOT:
            if BT:  # B^T [R, NB_] contiguous along the output columns
                bt = tl.load(b_ptr + rk[:, None] * NB_ + col0 + d[None, :])
            else:
                bt = tl.load(b_ptr + (col0 + d[None, :]) * R + rk[:, None])
            lo = tl.dot(t, bt, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
            y = (y + lo).to(tl.bfloat16).to(tl.float32)
        o_ptr = out_ptr + dst + which * (HPR * HD) + d[None, :]
        if which == 2 or not MATH:
            tl.store(o_ptr, y.to(tl.bfloat16), mask=m2)
        else:
            w_ptr = qn_ptr if which == 0 else kn_ptr
            wgt = tl.load(w_ptr + d).to(tl.float32)
            var = tl.sum(y * y, axis=1) / HD
            rstd = 1.0 / tl.sqrt(var + eps)
            n = (y * rstd[:, None] * wgt[None, :]).to(tl.bfloat16).to(tl.float32)
            na, nb = tl.split(tl.reshape(n, (BM, P, 2)))
            c_a = tl.load(cos_ptr + rm64[:, None] * RD + pp[None, :], mask=mp, other=1.0).to(tl.float32)
            s_a = tl.load(sin_ptr + rm64[:, None] * RD + pp[None, :], mask=mp, other=0.0).to(tl.float32)
            c_b = tl.load(cos_ptr + rm64[:, None] * RD + HALF + pp[None, :], mask=mp, other=1.0).to(tl.float32)
            s_b = tl.load(sin_ptr + rm64[:, None] * RD + HALF + pp[None, :], mask=mp, other=0.0).to(tl.float32)
            ra = ((na * c_a).to(tl.bfloat16).to(tl.float32) + (-nb * s_a).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            rb = ((nb * c_b).to(tl.bfloat16).to(tl.float32) + (na * s_b).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            ra = tl.where(inr[None, :], ra, na)
            rb = tl.where(inr[None, :], rb, nb)
            res = tl.reshape(tl.join(ra, rb), (BM, HD)).to(tl.bfloat16)
            tl.store(o_ptr, res, mask=m2)


# default ON: 8-GPU 12.4826 / 4-GPU 6.4166 gate pass, margins no worse (exact q/k head-dim permutation)
ROPE_PAIR = os.environ.get("PIKA_ROPE_PAIR", "1") in ("1", "hip")
ROPE_PAIR_HIP = os.environ.get("PIKA_ROPE_PAIR", "1") == "hip"  # experimental: same layout, hand-written HIP kernel (hipk.rope_pair; slower)
ROPE_PAIR_BM = int(os.environ.get("PIKA_ROPE_PAIR_BM", "32"))
ROPE_PAIR_W = int(os.environ.get("PIKA_ROPE_PAIR_W", "4"))


def qk_rope_pack_lora_pair(qkv, t, B, qn, kn, cos, sin, world, eps, NH=56, HD=128, s=None, BM=None, num_warps=None, s0=None,
                           rows=None, out=None, **kw):
    """qk_rope_pack_lora for pair-interleaved q/k head dims (qkv, B, qn, kn already permuted; output in the same order)."""
    BM = BM or ROPE_PAIR_BM
    num_warps = num_warps or ROPE_PAIR_W
    s = qkv.shape[0] if s is None else s
    hpr = NH // world
    R = B.shape[1]
    assert B.is_contiguous() and cos.is_contiguous() and sin.is_contiguous() and qkv.stride(1) == 1
    if out is None:
        out = torch.empty((world * s, 3 * hpr * HD), dtype=qkv.dtype, device=qkv.device)
    if kw.get('BT'):
        R = B.shape[0]; kw['NB_'] = B.shape[1]
    r0, r1 = rows if rows is not None else (0, s)
    _qk_rope_pair_kernel[(triton.cdiv(r1 - r0, BM), NH)](qkv, t, B, qn, kn, cos, sin, out, s, eps, t.stride(0), qkv.stride(0),
                                                         0 if s0 is None else s0, r0, r1, NH=NH, HPR=hpr, HD=HD, RD=cos.shape[1],
                                                         R=R, BM=BM, SPLIT=s0 is not None, num_warps=num_warps, **kw)
    return out

