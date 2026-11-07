# SPDX-License-Identifier: Apache-2.0
"""bf16-GEMM / fp32-residual implementation of ``MiniMaxH3VideoViTDecoder3d`` (VAE decoder tile).

Same math as the released module; the 36 transformer blocks use bf16 GEMMs and bf16 flash attention with the residual
stream, norms, q/k RMS and rotary kept in fp32. Weights are cast once.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from . import f16x3
from . import kernels as K
from . import gemm_sel as G


@triton.jit
def _rms_cast_kernel(x_ptr, w_ptr, out_ptr, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    var = tl.sum(x * x, axis=0) / D
    y = x * (1.0 / tl.sqrt(var + eps)) * w
    tl.store(out_ptr + row * D + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _resid_scale_kernel(x_ptr, y_ptr, sc_ptr, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
    y = tl.load(y_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sc_ptr + cols, mask=mask, other=0.0)
    tl.store(x_ptr + row * D + cols, x + y * s, mask=mask)


@triton.jit(do_not_specialize=["NTOK"])
def _qk_kernel(qkv_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr, v_ptr, NTOK, eps,
               NH: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr, PD: tl.constexpr = 0):
    # qkv [R, 3*NH*HD] (bf16) -> q,k,v [R, NH, HD] bf16 ; row = b*NTOK + tok ; cos/sin [NTOK, RD] fp32
    row = tl.program_id(0)
    head = tl.program_id(1)
    tok = row % NTOK
    d = tl.arange(0, HD)
    HALF: tl.constexpr = RD // 2
    base = row * (3 * NH * HD) + head * HD
    OD: tl.constexpr = HD if PD == 0 else PD  # output head dim (PD = 128: zero-padded, see _sdpa_chunked)
    out0 = (((row // NTOK) * NH + head) * NTOK + tok) * OD  # [B, NH, NTOK, OD] contiguous
    dpart = tl.where(d < HALF, d + HALF, d - HALF)
    inrd = d < RD
    c = tl.load(cos_ptr + tok * RD + d, mask=inrd, other=1.0)
    s = tl.load(sin_ptr + tok * RD + d, mask=inrd, other=0.0)
    for which in tl.static_range(2):
        x = tl.load(qkv_ptr + base + which * NH * HD + d).to(tl.float32)
        xp = tl.load(qkv_ptr + base + which * NH * HD + dpart).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / HD + eps)
        n = x * rstd
        npart = xp * rstd
        rot = tl.where(d < HALF, -npart, npart)
        res = tl.where(inrd, n * c + rot * s, n)
        if which == 0:
            tl.store(q_ptr + out0 + d, res.to(tl.bfloat16))
        else:
            tl.store(k_ptr + out0 + d, res.to(tl.bfloat16))
    v = tl.load(qkv_ptr + base + 2 * NH * HD + d)
    tl.store(v_ptr + out0 + d, v)
    if PD > 0:
        z = tl.zeros([HD], dtype=tl.bfloat16)
        tl.store(q_ptr + out0 + HD + d, z)
        tl.store(k_ptr + out0 + HD + d, z)
        tl.store(v_ptr + out0 + HD + d, z)


# torch SDPA (flash and mem-efficient backends alike) at head dim 64 returns WRONG results on some calls (nondeterministic, most likely a
# race in the kernel; 32 heads x 1796 tokens, SNR vs fp32 per call, normal = 53.0 dB):
#   contiguous [B,H,N,hd], B <= 5: always correct; B = 6: 2 of 30 calls bad (36-39 dB); B >= 7: 4-8 of 30 bad (28-32 dB);
#   strided [B,N,H,hd] views (an earlier layout, B = 4): 1 of 11 calls bad (34 dB) -> the decoder was not deterministic.
# Fix: q/k/v are written contiguous [B,H,N,hd] and attention runs in chunks of <= 4 tiles (12 of 12 calls identical, 53.0 dB).
# Even then 1 of 19 repeated chunked calls differed (49 dB). Default (PIKA_VAE_ATTN=pad128): q/k/v zero-padded to head dim 128 and run by
# the aiter hd128 kernel with the hd64 softmax scale - exact (zero dims add nothing to q.k; padded v columns are dropped), bit-deterministic
# (29/29 identical, 53.0 dB) and faster (B=4: 0.246 vs 0.296 ms). PIKA_VAE_ATTN=sdpa keeps the hd64 SDPA (contiguous, chunks of 4).
_ATTN_CHUNK = int(__import__("os").environ.get("PIKA_VAE_ATTN_CHUNK", "4"))
# Default tri64 (8-GPU eval 12.0119, gate pass, decode -0.01/-0.045/-0.05 s, detD 0.00); pad128 = previous default.
_VATT = __import__("os").environ.get("PIKA_VAE_ATTN", "tri64")
PAD128 = _VATT == "pad128"
TRI64 = _VATT == "tri64"   # aiter Triton flash attention at the native head dim 64 (BSHD, no padding)
_fa64 = None


def _tri64_attn(q, k, v):
    """q, k [B, N, H, 64] contiguous, v [B, N, H, 64] view (row stride 3*H*64) -> o [B, N, H, 64]."""
    global _fa64
    if _fa64 is None:
        import sys
        from .sp_model import AITER_DIR
        if AITER_DIR not in sys.path:
            sys.path.insert(0, AITER_DIR)
        from aiter.ops.triton.attention.mha import flash_attn_func
        _fa64 = flash_attn_func
    return _fa64(q, k, v, softmax_scale=q.shape[-1] ** -0.5, config=dict(FA64_CFG) if FA64_CFG else None)


# PIKA_VAE_FA_BM256=1 (default ON: 38/38 trace tensors bit-identical): Triton config of the tri64 attention measured faster in the
# tile (-1.3%), output bit-identical to the default config.
FA64_CFG = (dict(BLOCK_M=256, BLOCK_N=64, PRELOAD_V=True, waves_per_eu=0, num_warps=4, num_ctas=1, num_stages=3)
            if __import__("os").environ.get("PIKA_VAE_FA_BM256", "1") == "1" else None)


def _sdpa_chunked(qq, kk, vv, hd=None):
    """qq,kk,vv [B, H, N, D] contiguous (D = hd, or 128 zero-padded when hd < D) -> o [B, N, H, hd] contiguous."""
    B, H, N, D = qq.shape
    hd = hd or D
    o = torch.empty((B, N, H, hd), dtype=qq.dtype, device=qq.device)
    for b0 in range(0, B, _ATTN_CHUNK):
        b1 = min(B, b0 + _ATTN_CHUNK)
        r = F.scaled_dot_product_attention(qq[b0:b1], kk[b0:b1], vv[b0:b1], scale=hd ** -0.5)
        o[b0:b1].copy_(r[..., :hd].transpose(1, 2))
    return o


@triton.jit(do_not_specialize=["NTOK", "R"])
def _qk2d_kernel(qkv_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr, v_ptr, NTOK, R, eps,
                 NH: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr, BM: tl.constexpr, PD: tl.constexpr = 0,
                 BSHD: tl.constexpr = False):
    # == _qk_kernel on a [BM rows, HD] tile per program (PIKA_VAE_QK2D): fp32 RMS over the head + rotary, one bf16 rounding
    # BSHD (PIKA_VAE_ATTN=tri64): q, k written [R, NH, HD] (= [B, NTOK, NH, HD]); v is not written (a strided view of qkv is used)
    pid = tl.program_id(0)
    head = tl.program_id(1)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows[:, None] < R
    rows64 = rows.to(tl.int64)
    tok = rows64 % NTOK
    d = tl.arange(0, HD)
    HALF: tl.constexpr = RD // 2
    base = rows64[:, None] * (3 * NH * HD) + head * HD
    OD: tl.constexpr = HD if PD == 0 else PD   # output layout [B, NH, NTOK, OD] contiguous, as _qk_kernel
    if BSHD:
        out0 = (rows64[:, None] * NH + head) * HD + d[None, :]
    else:
        out0 = (((rows64[:, None] // NTOK) * NH + head) * NTOK + tok[:, None]) * OD + d[None, :]
    dpart = tl.where(d < HALF, d + HALF, d - HALF)
    inrd = d < RD
    cm = rmask & inrd[None, :]
    c = tl.load(cos_ptr + tok[:, None] * RD + d[None, :], mask=cm, other=1.0)
    sn = tl.load(sin_ptr + tok[:, None] * RD + d[None, :], mask=cm, other=0.0)
    for which in tl.static_range(2):
        x = tl.load(qkv_ptr + base + which * NH * HD + d[None, :], mask=rmask, other=0.0).to(tl.float32)
        xp = tl.load(qkv_ptr + base + which * NH * HD + dpart[None, :], mask=rmask, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / HD + eps)
        n = x * rstd[:, None]
        npart = xp * rstd[:, None]
        rot = tl.where(d[None, :] < HALF, -npart, npart)
        res = tl.where(inrd[None, :], n * c + rot * sn, n)
        if which == 0:
            tl.store(q_ptr + out0, res.to(tl.bfloat16), mask=rmask)
        else:
            tl.store(k_ptr + out0, res.to(tl.bfloat16), mask=rmask)
    if not BSHD:
        v = tl.load(qkv_ptr + base + 2 * NH * HD + d[None, :], mask=rmask, other=0.0)
        tl.store(v_ptr + out0, v, mask=rmask)
    if PD > 0:
        z = tl.zeros([BM, HD], dtype=tl.bfloat16)
        tl.store(q_ptr + out0 + HD, z, mask=rmask)
        tl.store(k_ptr + out0 + HD, z, mask=rmask)
        tl.store(v_ptr + out0 + HD, z, mask=rmask)


@triton.jit
def _resid_rms_kernel(x_ptr, y_ptr, sc_ptr, w_ptr, out_ptr, D, eps, BLOCK: tl.constexpr):
    # == _resid_scale_kernel then _rms_cast_kernel (PIKA_VAE_RESRMS): x += y * s (fp32, written back), h = bf16(rms(x) * w)
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
    y = tl.load(y_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sc_ptr + cols, mask=mask, other=0.0)
    x = x + y * s
    tl.store(x_ptr + row * D + cols, x, mask=mask)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    var = tl.sum(x * x, axis=0) / D
    h = x * (1.0 / tl.sqrt(var + eps)) * w
    tl.store(out_ptr + row * D + cols, h.to(tl.bfloat16), mask=mask)


import os as _os
_GE = _os.environ.get("PIKA_GEMM_ENG", "1")  # umbrella flag, default ON since 4-GPU eval 6.382
QK2D = _os.environ.get("PIKA_VAE_QK2D", _GE) == "1"
RESRMS = _os.environ.get("PIKA_VAE_RESRMS", _GE) == "1"


class FastViTDecoder:
    def __init__(self, decoder):
        dec = decoder
        self.D = dec.proj_in.out_features
        self.patch, self.patch_t = dec.patch_size, dec.patch_size_t
        self.nreg = dec.num_register_tokens
        self.out_channels = dec.out_channels
        self.dev = dec.proj_in.weight.device
        self.dec = dec  # keeps the fp32 head modules (proj_in / norm_out / proj_out)
        self.eps = dec.transformer_blocks[0].norm1.eps
        self.qk_eps = dec.transformer_blocks[0].attn.norm_q.eps
        self.heads = dec.transformer_blocks[0].attn.heads
        self.hd = dec.transformer_blocks[0].attn.dim_head
        self.rd = int(dec.rope.inv_freq.numel() * 6)
        self.blocks = []
        bf = torch.bfloat16
        for b in dec.transformer_blocks:
            a = b.attn
            self.blocks.append(dict(
                n1=b.norm1.weight.detach().float().clone(), n2=b.norm2.weight.detach().float().clone(),
                wqkv=torch.cat([a.to_q.weight, a.to_k.weight, a.to_v.weight], 0).to(bf).contiguous(),
                bqkv=torch.cat([a.to_q.bias, a.to_k.bias, a.to_v.bias], 0).to(bf).contiguous(),
                wo=a.to_out[0].weight.to(bf).contiguous(), bo=a.to_out[0].bias.to(bf).contiguous(),
                w1=b.ff.net[0].proj.weight.to(bf).contiguous(), b1=b.ff.net[0].proj.bias.to(bf).contiguous(),
                w2=b.ff.net[2].weight.to(bf).contiguous(), b2=b.ff.net[2].bias.to(bf).contiguous(),
                s1=b.scale1.detach().float().clone(), s2=b.scale2.detach().float().clone()))
        self._rope_cache = {}

    def _rope(self, nf, nh, nw, device):
        key = (nf, nh, nw)
        if key not in self._rope_cache:
            dec = self.dec
            grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=device) / size) - 1.0 for size in (nf, nh, nw)]
            pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
            suffix = pos.new_zeros((self.nreg + 1, 3))
            pos = torch.cat([pos, suffix], 0).unsqueeze(0)
            cos, sin = dec.rope(pos)  # [1, N, 1, RD]
            self._rope_cache[key] = (cos[0, :, 0].contiguous().float(), sin[0, :, 0].contiguous().float())
        return self._rope_cache[key]

    @torch.no_grad()
    def __call__(self, z):
        """z [B, 24, F, H, W] (fp32) -> [B, 3, F*pt, H*p, W*p] fp32."""
        dec = self.dec
        B, C, nf, nh, nw = z.shape
        D = self.D
        from . import lossy as LS
        vf = LS.get("vae_fp8") if LS.ACTIVE else None  # LOSSY (off by default): FP8 row-wise GEMMs in the tile ViT
        if vf:
            ops = {"qkv", "o", "up", "down"} if vf is True or vf == 1 or vf == "all" else set(str(vf).split(","))
            from . import lowprec as LP
            from .lp_kernels import quant_rows_fp8

            def lin(x_, w_, b_, op):
                if op not in ops:
                    return G.linear(x_, w_, b_)
                wq, ws = LP._wq(w_)
                xq, xs = quant_rows_fp8(x_)
                return torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16).add_(b_)
        else:
            def lin(x_, w_, b_, op):
                return G.linear(x_, w_, b_)
        x = z.permute(0, 2, 3, 4, 1).reshape(B, nf * nh * nw, C)
        x = dec.proj_in(x.float())
        npatch = x.shape[1]
        reg = dec.register_tokens.expand(B, -1, -1)
        cls = torch.zeros_like(x[:, :1])
        x = torch.cat([x, reg, cls], dim=1).contiguous()  # [B, NTOK, D] fp32
        NTOK = x.shape[1]
        cos, sin = self._rope(nf, nh, nw, x.device)
        R = B * NTOK
        x = x.view(R, D)
        H, hd = self.heads, self.hd
        BLK = triton.next_power_of_2(D)
        h = torch.empty((R, D), dtype=torch.bfloat16, device=x.device)
        _rms_cast_kernel[(R,)](x, self.blocks[0]["n1"], h, D, self.eps, BLOCK=BLK, num_warps=4)
        nblk = len(self.blocks)
        for bi, b in enumerate(self.blocks):
            if bi > 0 and not RESRMS:
                h = torch.empty((R, D), dtype=torch.bfloat16, device=x.device)
                _rms_cast_kernel[(R,)](x, b["n1"], h, D, self.eps, BLOCK=BLK, num_warps=4)
            qkv = lin(h, b["wqkv"], b["bqkv"], "qkv")
            pd = 128 if (PAD128 and hd < 128) else hd
            q = torch.empty((R, H, pd), dtype=torch.bfloat16, device=x.device)
            k = torch.empty_like(q)
            if TRI64:
                _qk2d_kernel[(triton.cdiv(R, 32), H)](qkv, cos, sin, q, k, q, NTOK, R, self.qk_eps, NH=H, HD=hd, RD=self.rd,
                                                       BM=32, BSHD=True, num_warps=4)
                vv = qkv[:, 2 * H * hd:].view(B, NTOK, H, hd)
                o = _tri64_attn(q.view(B, NTOK, H, hd), k.view(B, NTOK, H, hd), vv).reshape(R, H * hd)
            v = q if TRI64 else torch.empty_like(q)
            if TRI64:
                pass
            elif QK2D:
                _qk2d_kernel[(triton.cdiv(R, 32), H)](qkv, cos, sin, q, k, v, NTOK, R, self.qk_eps, NH=H, HD=hd, RD=self.rd,
                                                       PD=(pd if pd != hd else 0), BM=32, num_warps=4)
            else:
                _qk_kernel[(R, H)](qkv, cos, sin, q, k, v, NTOK, self.qk_eps, NH=H, HD=hd, RD=self.rd, PD=(pd if pd != hd else 0),
                                   num_warps=1)
            if not TRI64:
                qq = q.view(B, H, NTOK, pd)  # contiguous BHND (written so by _qk_kernel / _qk2d_kernel)
                kk = k.view(B, H, NTOK, pd)
                vv = v.view(B, H, NTOK, pd)
                o = _sdpa_chunked(qq, kk, vv, hd).reshape(R, H * hd)
            y = lin(o, b["wo"], b["bo"], "o")
            if RESRMS:
                _resid_rms_kernel[(R,)](x, y, b["s1"], b["n2"], h, D, self.eps, BLOCK=BLK, num_warps=4)
            else:
                _resid_scale_kernel[(R,)](x, y, b["s1"], D, BLOCK=BLK, num_warps=4)
                _rms_cast_kernel[(R,)](x, b["n2"], h, D, self.eps, BLOCK=BLK, num_warps=4)
            up = lin(h, b["w1"], b["b1"], "up")
            a_ = K.swiglu(up)
            y = lin(a_, b["w2"], b["b2"], "down")
            if RESRMS and bi + 1 < nblk:   # residual + the next block's norm1 (h is free: its last reader, the up GEMM, is done)
                h = torch.empty((R, D), dtype=torch.bfloat16, device=x.device)
                _resid_rms_kernel[(R,)](x, y, b["s2"], self.blocks[bi + 1]["n1"], h, D, self.eps, BLOCK=BLK, num_warps=4)
            else:
                _resid_scale_kernel[(R,)](x, y, b["s2"], D, BLOCK=BLK, num_warps=4)
        x = x.view(B, NTOK, D)
        x = dec.norm_out(x)
        x = dec.proj_out(x)  # fp32 head
        x = x[:, :npatch, :]
        p, pt = self.patch, self.patch_t
        x = x.view(B, nf, nh, nw, self.out_channels, pt, p, p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(B, self.out_channels, nf * pt, nh * p, nw * p)


# ======================================================================================================================
# Reference-precision (fp32) decoder tile.  Every activation, weight and accumulation is fp32 as in the released module;
# only the op fusion differs.  Two GEMM modes:
#   "fp32": F.linear on fp32 tensors (TF32 off) -- the reference GEMMs.
#   "x6"  : each fp32 GEMM as ONE bf16 GEMM with fp32 accumulation over a 6x longer K (operands split into bf16 hi/mid/lo parts:
#           hh + hm + mh + mm + hl + lh).  Measured error vs an fp64 GEMM (2.8e-7) is below that of the fp32 GEMM itself (8.1e-7),
#           i.e. normal fp32 variation, not a precision reduction.
# ======================================================================================================================
@triton.jit
def _rms_f32_kernel(x_ptr, w_ptr, out_ptr, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    var = tl.sum(x * x, axis=0) / D
    tl.store(out_ptr + row * D + cols, x * (1.0 / tl.sqrt(var + eps)) * w, mask=mask)


@triton.jit(do_not_specialize=["NTOK"])
def _qk_f32_kernel(qkv_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr, v_ptr, NTOK, eps,
                   NH: tl.constexpr, HD: tl.constexpr, RD: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    tok = row % NTOK
    d = tl.arange(0, HD)
    HALF: tl.constexpr = RD // 2
    base = row * (3 * NH * HD) + head * HD
    out0 = (row * NH + head) * HD
    dpart = tl.where(d < HALF, d + HALF, d - HALF)
    inrd = d < RD
    c = tl.load(cos_ptr + tok * RD + d, mask=inrd, other=1.0)
    s = tl.load(sin_ptr + tok * RD + d, mask=inrd, other=0.0)
    for which in tl.static_range(2):
        x = tl.load(qkv_ptr + base + which * NH * HD + d)
        xp = tl.load(qkv_ptr + base + which * NH * HD + dpart)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / HD + eps)
        n = x * rstd
        npart = xp * rstd
        rot = tl.where(d < HALF, -npart, npart)
        res = tl.where(inrd, n * c + rot * s, n)
        if which == 0:
            tl.store(q_ptr + out0 + d, res)
        else:
            tl.store(k_ptr + out0 + d, res)
    tl.store(v_ptr + out0 + d, tl.load(qkv_ptr + base + 2 * NH * HD + d))


@triton.jit
def _swiglu_f32_kernel(up_ptr, out_ptr, FF, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < FF
    a = tl.load(up_ptr + row * 2 * FF + cols, mask=mask, other=0.0)
    g = tl.load(up_ptr + row * 2 * FF + FF + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * FF + cols, a * (g / (1.0 + tl.exp(-g))), mask=mask)


@triton.jit
def _split6_kernel(x_ptr, out_ptr, K, BLOCK: tl.constexpr):
    # x [R, K] fp32 -> out [R, 6K] bf16 = [h | h | m | m | h | l]  (pairs with weights [h | m | h | m | l | h])
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(x_ptr + row * K + cols, mask=mask, other=0.0)
    h = x.to(tl.bfloat16)
    r = x - h.to(tl.float32)
    m = r.to(tl.bfloat16)
    l = (r - m.to(tl.float32)).to(tl.bfloat16)
    o = out_ptr + row * 6 * K + cols
    tl.store(o, h, mask=mask)
    tl.store(o + K, h, mask=mask)
    tl.store(o + 2 * K, m, mask=mask)
    tl.store(o + 3 * K, m, mask=mask)
    tl.store(o + 4 * K, h, mask=mask)
    tl.store(o + 5 * K, l, mask=mask)


def _split6_weight(w):
    """w [N, K] fp32 -> [N, 6K] bf16 = [h | m | h | m | l | h]."""
    bf = torch.bfloat16
    h = w.to(bf)
    r = w - h.float()
    m = r.to(bf)
    l = (r - m.float()).to(bf)
    return torch.cat([h, m, h, m, l, h], 1).contiguous()


class ExactViTDecoder(FastViTDecoder):
    def __init__(self, decoder, mode="fp32"):
        dec = decoder
        self.D = dec.proj_in.out_features
        self.patch, self.patch_t = dec.patch_size, dec.patch_size_t
        self.nreg = dec.num_register_tokens
        self.out_channels = dec.out_channels
        self.dev = dec.proj_in.weight.device
        self.dec = dec
        self.eps = dec.transformer_blocks[0].norm1.eps
        self.qk_eps = dec.transformer_blocks[0].attn.norm_q.eps
        self.heads = dec.transformer_blocks[0].attn.heads
        self.hd = dec.transformer_blocks[0].attn.dim_head
        self.rd = int(dec.rope.inv_freq.numel() * 6)
        self._rope_cache = {}
        self.mode = mode
        self.blocks = []
        f = torch.float32
        for b in decoder.transformer_blocks:
            a = b.attn
            lin = dict(
                wqkv=torch.cat([a.to_q.weight, a.to_k.weight, a.to_v.weight], 0), bqkv=torch.cat([a.to_q.bias, a.to_k.bias, a.to_v.bias], 0),
                wo=a.to_out[0].weight, bo=a.to_out[0].bias, w1=b.ff.net[0].proj.weight, b1=b.ff.net[0].proj.bias,
                w2=b.ff.net[2].weight, b2=b.ff.net[2].bias)
            e = dict(n1=b.norm1.weight.detach().float().clone(), n2=b.norm2.weight.detach().float().clone(),
                     s1=b.scale1.detach().float().clone(), s2=b.scale2.detach().float().clone())
            for k, v in lin.items():
                v = v.detach().to(f)
                if mode == "x6" and k[0] == "w":
                    e[k] = _split6_weight(v)
                elif mode == "f16x3" and k[0] == "w":
                    e[k] = f16x3.prepare_weight(v)   # (W3 [N, 3K] fp16, column scale [N] fp32)
                else:
                    e[k] = v.contiguous()
            self.blocks.append(e)

    def _linear(self, x, w, b):
        if self.mode == "fp32":
            return F.linear(x, w, b)
        if self.mode == "f16x3":
            return f16x3.linear(x, w[0], w[1], b)
        R, K = x.shape
        xs = torch.empty((R, 6 * K), dtype=torch.bfloat16, device=x.device)
        _split6_kernel[(R, triton.cdiv(K, 1024))](x, xs, K, BLOCK=1024, num_warps=4)
        return torch.mm(xs, w.t(), out_dtype=torch.float32) + b

    @torch.no_grad()
    def __call__(self, z):
        dec = self.dec
        B, C, nf, nh, nw = z.shape
        D = self.D
        x = z.permute(0, 2, 3, 4, 1).reshape(B, nf * nh * nw, C)
        x = dec.proj_in(x.float())
        npatch = x.shape[1]
        reg = dec.register_tokens.expand(B, -1, -1)
        cls = torch.zeros_like(x[:, :1])
        x = torch.cat([x, reg, cls], dim=1).contiguous()
        NTOK = x.shape[1]
        cos, sin = self._rope(nf, nh, nw, x.device)
        R = B * NTOK
        x = x.view(R, D)
        H, hd = self.heads, self.hd
        BLK = triton.next_power_of_2(D)
        for b in self.blocks:
            h = torch.empty((R, D), dtype=torch.float32, device=x.device)
            _rms_f32_kernel[(R,)](x, b["n1"], h, D, self.eps, BLOCK=BLK, num_warps=4)
            qkv = self._linear(h, b["wqkv"], b["bqkv"])
            q = torch.empty((R, H, hd), dtype=torch.float32, device=x.device)
            k = torch.empty_like(q)
            v = torch.empty_like(q)
            _qk_f32_kernel[(R, H)](qkv, cos, sin, q, k, v, NTOK, self.qk_eps, NH=H, HD=hd, RD=self.rd, num_warps=1)
            qq = q.view(B, NTOK, H, hd).transpose(1, 2).contiguous()
            kk = k.view(B, NTOK, H, hd).transpose(1, 2).contiguous()
            vv = v.view(B, NTOK, H, hd).transpose(1, 2).contiguous()
            o = _sdpa_chunked(qq, kk, vv).reshape(R, H * hd)
            y = self._linear(o, b["wo"], b["bo"])
            _resid_scale_kernel[(R,)](x, y, b["s1"], D, BLOCK=BLK, num_warps=4)
            _rms_f32_kernel[(R,)](x, b["n2"], h, D, self.eps, BLOCK=BLK, num_warps=4)
            up = self._linear(h, b["w1"], b["b1"])
            a_ = torch.empty((R, up.shape[1] // 2), dtype=torch.float32, device=x.device)
            _swiglu_f32_kernel[(R, triton.cdiv(a_.shape[1], 2048))](up, a_, a_.shape[1], BLOCK=2048, num_warps=4)
            y = self._linear(a_, b["w2"], b["b2"])
            _resid_scale_kernel[(R,)](x, y, b["s2"], D, BLOCK=BLK, num_warps=4)
        x = x.view(B, NTOK, D)
        x = dec.norm_out(x)
        x = dec.proj_out(x)
        x = x[:, :npatch, :]
        p, pt = self.patch, self.patch_t
        x = x.view(B, nf, nh, nw, self.out_channels, pt, p, p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(B, self.out_channels, nf * pt, nh * p, nw * p)
