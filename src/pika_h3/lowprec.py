# SPDX-License-Identifier: Apache-2.0
"""LOSSY track, lower precision. OFF by default: nothing here runs unless the lossy config
(env PIKA_LOSSY JSON or the request field "lossy", see lossy.py) has one of the keys below.

Keys (each value: one entry for all six generator evaluations, or a list of 6 per-step entries):
  attn   attention recipe: "bf16" (= lossless kernel) | "fp8" (Hadamard-rotated per-tensor FP8 Q/K, K mean removed, FP8 V)
         | "mxfp8" (MXFP8 Q/K 1x32 blocks, FP8 V) | "int8" (INT8 Q/K, FP8 V) | "bf16fp8" (BF16 Q/K, FP8 V)
         | "mxfp6" | "mxfp6fp8" (MXFP6 Q/K, FP8 V) | "fp8mxfp6" | "mxfp4"
  gemm   comma list of base GEMMs in FP8 (e4m3, per-row activation scale x per-output-channel weight scale, bf16 out):
         subset of qkv,o,up,down, e.g. "qkv,o,up,down"; "" = none.  LoRA (x A^T, B) stays bf16.
  gemm_layers  optional [lo, hi) layer range for the FP8 GEMMs (default all 48)
"""
from __future__ import annotations

import os
import sys

import torch

from . import lossy as LS

PREFIX_TOKENS = 0  # global text+audio tokens (L + Na) of the request
PREFIX_ROWS = 0  # local text+audio rows of this rank (set by SPTransformer.forward when LS.ACTIVE)
LAYER = 0  # current transformer block (set by SPTransformer.forward via set_layer when LS.ACTIVE)


def set_layer(li):
    global LAYER, GEN
    LAYER = li
    GEN += 1
_SHAPE_OP = {(21504, 5376): "qkv", (5376, 7168): "o", (28672, 5376): "up", (5376, 14336): "down"}
_WQ = {}
F8 = torch.float8_e4m3fn


def _per_step(val, step, default):
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return val[step] if step < len(val) else val[-1]
    return val


def attn_recipe():
    return LS.attn_recipe() if LS.ACTIVE else "bf16"


def _rule_hit(r):
    st = r.get("steps")
    if st is not None and LS.STEP not in st:
        return False
    lr = r.get("layers")
    return not (lr and not (lr[0] <= LAYER < lr[1]))


def gemm_rule(op):
    """fine-grained map (key 'gemm_map': list of rules {steps, ops, layers [lo, hi), rows 'all'|'video'|'prefix', fmt}); first
    matching rule wins. fmt: fp8 (real rowwise x channel scaled GEMM) | mxfp8 | mxfp6 | mxfp4 (1x32 E8M0 block fake-quant of
    x and W, bf16 GEMM on the dequantized values = exact MX product up to summation order) | legit (fp32 GEMM rounded to bf16:
    a legitimate kernel variant, the chaos-floor reference)."""
    for r in LS.get("gemm_map") or ():
        ops = r.get("ops", "qkv,o,up,down")
        ops = ops.split(",") if isinstance(ops, str) else ops
        if op in ops and _rule_hit(r):
            return r
    return None


def gemm_ops():
    if not LS.ACTIVE:
        return ()
    if LS.get("gemm_map") is not None:
        return ("map",)
    g = _per_step(LS.get("gemm"), LS.STEP, "")
    if not g:
        return ()
    lr = LS.get("gemm_layers")
    if lr and not (lr[0] <= LAYER < lr[1]):
        return ()
    return tuple(g.split(",")) if isinstance(g, str) else tuple(g)


# ----------------------------------------------------------------------------------------------------- attention
_AF = None


def _aiter():
    global _AF
    if _AF is None:
        from .sp_model import AITER_DIR
        if AITER_DIR not in sys.path:
            sys.path.insert(0, AITER_DIR)
        from aiter.ops import mha_v4 as M
        _AF = M
    return _AF


def recipe_args(name):
    M = _aiter()
    AF, SM = M.AttentionFormat, M.AttentionScaleMode
    fmt = {"fp8": (AF.FP8, AF.FP8), "mxfp8": (AF.FP8, AF.FP8), "int8": (AF.INT8, AF.FP8), "bf16fp8": (AF.BF16, AF.FP8),
           "mxfp6": (AF.MXFP6, AF.MXFP6), "mxfp6fp8": (AF.MXFP6, AF.FP8), "fp8mxfp6": (AF.FP8, AF.MXFP6),
           "mxfp4": (AF.MXFP4, AF.MXFP4)}[name]
    kw = {}
    if name == "mxfp8":
        kw = dict(q_scale_mode=SM.E8M0_PER_1X32, k_scale_mode=SM.E8M0_PER_1X32, v_scale_mode=SM.F32_PER_TENSOR)
    return fmt[0], fmt[0], fmt[1], kw


GEN = 0          # bumped per transformer block (LAYER set) -> K/V quantized once per block even with several query launches
_KV = None       # (GEN, recipe, k ptr, Nk, quantized K/V operands)
FAST = os.environ.get("PIKA_LP_FAST", "1") == "1"


def _quant_kv(k, v, recipe):
    """K/V producers of the recipe, cached for the current block (all query launches of a block share K/V)."""
    global _KV
    key = (GEN, recipe, k.data_ptr(), k.shape[1], LS.STEP)
    if _KV is not None and _KV[0] == key:
        return _KV[1]
    M = _aiter()
    AF = M.AttentionFormat
    kc, vc = k.contiguous(), v.contiguous()
    if recipe == "mxfp6":
        _, qk_, qv_ = M.mha_v4_mx_producers(AF.MXFP6, 128)
        km = M._k_mean(kc, M._RawRecipeKind.MXFP6)
        r = (qk_(kc, km), qv_(vc))
    elif recipe == "mxfp8":
        km = M._k_mean(kc, M._RawRecipeKind.MXFP8)
        r = (M.quantize_mxfp8_k(kc, km), M.quantize_fp8(vc))
    else:
        raise KeyError(recipe)
    _KV = (key, r, kc, vc)
    return r


def _kv_bf16():
    return _KV[2], _KV[3]


def _prefix_rows_bf16(q, k, out):
    """attn_prefix_bf16: the text+audio query rows (global rows [0, PREFIX_TOKENS), at KEY_PREFIX_OFF in the receive buffer) are
    recomputed with the exact bf16 kernel (keys split 16 ways + LSE merge inside aiter: ~0.09 ms/layer at w8 5 s)."""
    off, P = LS.KEY_PREFIX_OFF, PREFIX_TOKENS
    if off is None or off < 0 or P <= 0:
        return
    h, D = q.shape[2], q.shape[3]
    sn = q.stride(1)
    if k.stride(1) != sn or k.shape[1] < off + P:
        return
    d = q.data_ptr() - (k.data_ptr() - h * D * q.element_size())  # q row 0 sits one q/k/v block before k row 0 (packed recv)
    if d % (sn * q.element_size()):
        return
    r0 = d // (sn * q.element_size())
    a, b = max(off, r0), min(off + P, r0 + q.shape[1])
    if b <= a:
        return
    M = _aiter()
    AF = M.AttentionFormat
    kc, vc = _kv_bf16()
    o = M.mha_v4(q[:, a - r0:b - r0].contiguous(), kc, vc, AF.BF16, AF.BF16, AF.BF16, kv_splits=16)
    out[:, a - r0:b - r0].copy_(o)


def attn_into(q, k, v, out, recipe):
    """Lossy replacement of sp_model._v4_into. q,k,v [1, n|Nk, h, D] views, out [1, n, h, D] bf16."""
    M = _aiter()
    if not FAST or recipe not in ("mxfp6", "mxfp8"):
        fq, fk, fv, kw = recipe_args(recipe)
        o = M.mha_v4(q.contiguous(), k.contiguous(), v.contiguous(), fq, fk, fv, **kw)
        out.copy_(o)
        return out
    AF, SM = M.AttentionFormat, M.AttentionScaleMode
    (kq, kd), (vq, vd) = _quant_kv(k, v, recipe)
    sc = 128 ** -0.5
    qc = q.contiguous()
    dst = out if out.is_contiguous() else torch.empty(out.shape, dtype=out.dtype, device=out.device)
    if recipe == "mxfp6":
        quant_q, _, _ = M.mha_v4_mx_producers(AF.MXFP6, 128)
        qq, qd = quant_q(qc, M.mha_v4_q_multiplier(sc))
        M._launch_mxfp6(qq, qd, kq, kd, vq, vd, dst, k.shape[1], k.shape[2], int(AF.MXFP6), int(M.AttentionPack.V_FOR_FP6_P),
                        sc, None, None)
    else:
        qq, qd = M.quantize_mxfp8_q(qc, M.mha_v4_q_multiplier(sc))
        M.mha_v4_packed(qq, kq, vq, qd, kd, vd, AF.FP8, AF.FP8, AF.FP8, SM.E8M0_PER_1X32, SM.E8M0_PER_1X32, SM.F32_PER_TENSOR,
                        softmax_scale=sc, out=dst, v_pack=M.AttentionPack.DEFAULT)
    if dst is not out:
        out.copy_(dst)
    if LS.get("attn_prefix_bf16"):
        _prefix_rows_bf16(q, k, out)
    return out


# ----------------------------------------------------------------------------------------------------- GEMM
def _quant_rows(x):
    xf = x.float()
    s = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    return (xf / s).to(F8), s


def _wq(W):
    key = W.data_ptr()
    if key not in _WQ:
        wq, ws = _quant_rows(W)
        _WQ[key] = (wq, ws.view(1, -1).contiguous())
    return _WQ[key]


def gemm(x, W):
    """bf16(x W^T) in FP8 if this op is selected for the current step/layer, else None."""
    ops = gemm_ops()
    if not ops:
        return None
    op = _SHAPE_OP.get(tuple(W.shape))
    if ops == ("map",):
        r = gemm_rule(op) if (op is not None and x.dim() == 2) else None
        return None if r is None else _gemm_map(x, W, r)
    if op is None or op not in ops or x.dim() != 2:
        return None
    fmt = _per_step(LS.get("gemm_fmt"), LS.STEP, "fp8")  # fp8 (rowwise x channel) | a6w6 (aiter MXFP6 asm)
    return _gemm_map(x, W, {"fmt": fmt})


def _fp8_mm(x, wq, ws):
    if FAST:
        from .lp_kernels import quant_rows_fp8
        xq, xs = quant_rows_fp8(x)
    else:
        xq, xs = _quant_rows(x)
    return torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)


def sparse_quant(recipe):
    """quant callable for sparse2.sparse_into (combination contract): (q4, k4, v4) -> (q, k, v, qd, kd, vd, fmts, scale modes).
    Only recipes whose operands can be sliced along keys and that export an LSE (sparse2 merges the partial last key block):
    'bf16fp8' (BF16 Q/K, per-tensor FP8 V) and 'fp8' (rotated per-tensor FP8 Q/K, FP8 V). MX recipes: use whole blocks only."""
    M = _aiter()
    AF, SM = M.AttentionFormat, M.AttentionScaleMode

    def q_bf16fp8(q4, k4, v4):
        vq, vd = M.quantize_fp8(v4.contiguous())
        return q4, k4, vq, q4, k4, vd, (AF.BF16, AF.BF16, AF.FP8), (SM.NONE, SM.NONE, SM.F32_PER_TENSOR)

    def q_fp8(q4, k4, v4):
        qq, qd = M.quantize_fp8_rotated(q4.contiguous())
        kq, kd = M.quantize_fp8_rotated(k4.contiguous())
        vq, vd = M.quantize_fp8(v4.contiguous())
        return qq, kq, vq, qd, kd, vd, (AF.FP8, AF.FP8, AF.FP8), (SM.F32_PER_TENSOR,) * 3

    return {"bf16fp8": q_bf16fp8, "fp8": q_fp8}[recipe]


# ----------------------------------------------------------------------------------------------------- fine-grained map (dev)
_MX = {"mxfp8": (3, -6, 8, 448.0), "mxfp6": (3, 0, 2, 7.5), "mxfp4": (1, 0, 2, 6.0)}  # mantissa bits, emin, emax, max (e4m3/e2m3/e2m1)
_WF = {}


def _round_minifloat(v, mb, emin, vmax):
    a = v.abs()
    e = torch.floor(torch.log2(a.clamp_min(1e-30))).clamp_min(emin)
    q = torch.exp2(e - mb)
    return (torch.round(v / q) * q).clamp(-vmax, vmax)


def fake_mx(x, fmt):
    """1x32 (along the last dim) E8M0-scaled block quantize/dequantize (OCP MX), returned in x.dtype (exact in bf16)."""
    mb, emin, emax, vmax = _MX[fmt]
    M, K = x.shape
    out = torch.empty_like(x)
    for i in range(0, M, 4096):
        xb = x[i:i + 4096].float().view(-1, K // 32, 32)
        am = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        sc = torch.exp2(torch.floor(torch.log2(am)) - emax)
        out[i:i + 4096] = (_round_minifloat(xb / sc, mb, emin, vmax) * sc).view(-1, K).to(x.dtype)
    return out


def _wfake(W, fmt):
    key = (W.data_ptr(), fmt)
    if key not in _WF:
        if len(_WF) and next(iter(_WF))[1] != fmt:
            _WF.clear()
        _WF[key] = fake_mx(W, fmt)
    return _WF[key]


_A6 = None
_W6 = {}
_PS = None


def _a6():
    global _A6
    if _A6 is None:
        _aiter()  # puts AITER_DIR on sys.path
        from aiter.ops import gemm_op_a6w6 as A6
        _A6 = A6
    return _A6


def a6w6_mm(x, W, out=None):
    """aiter gfx950 MXFP6 (E2M3, 1x32 E8M0, 32-block Hadamard along K in both quantizers) asm GEMM: bf16 x [M, K] W^T."""
    A6 = _a6()
    key = W.data_ptr()
    if key not in _W6:
        _W6[key] = A6.quant_mxfp6_gemm(W)
    wp, ws = _W6[key]
    M, K = x.shape
    N = W.shape[0]
    # aiter's a6w6 asm GEMM uses int32 byte offsets: rows past 2^31 bytes of output (or input) come out as
    # garbage (measured: up GEMM N=28672 at M=37784 -> last rows 0 dB; world 1 at any duration, world 2 at 15 s). Chunk M below it.
    mmax = ((2 ** 31 - 1) // (2 * max(N, K))) // 256 * 256
    if M > mmax:
        y = out if out is not None else torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        for a in range(0, M, mmax):
            y[a:a + mmax].copy_(a6w6_mm(x[a:a + mmax], W))
        return y
    xp, xs = A6.quant_mxfp6_gemm(x if x.is_contiguous() else x.contiguous())
    if out is not None and M % 256 == 0 and N % 256 == 0 and out.is_contiguous():
        kn = A6._select_gemm_a6w6_kernel(M, N, K, None, device=x.device)
        A6.gemm_a6w6_asm(xp, wp, xs, ws, out, A6._ceil(K, 128), kn, 1.0)
        return out
    y = A6.gemm_a6w6(xp, wp, xs, ws, M, N, K)
    if out is not None:
        out.copy_(y)
        return out
    return y if y.is_contiguous() else y.contiguous()


def _gemm_rows(x, W, fmt):
    if fmt == "a6w6":
        return a6w6_mm(x, W)
    if fmt == "fp8":
        wq, ws = _wq(W)
        return _fp8_mm(x, wq, ws)
    if fmt == "legit":
        return (x.float() @ W.float().t()).to(torch.bfloat16)
    return torch.nn.functional.linear(fake_mx(x, fmt), _wfake(W, fmt))


def _gemm_map(x, W, r):
    fmt = r.get("fmt", "fp8")
    rows = r.get("rows", "video" if LS.get("gemm_prefix_bf16") else "all")
    pr = min(PREFIX_ROWS, x.shape[0])
    if rows == "all" or (rows == "video" and pr == 0):
        return _gemm_rows(x, W, fmt)
    if rows == "prefix" and pr == 0:
        return None
    if rows == "video" and fmt in ("fp8", "a6w6") and LS.get("pfx_stream"):  # prefix rows' bf16 GEMM on a side stream
        global _PS
        if _PS is None:
            _PS = torch.cuda.Stream(device=x.device)
        cur = torch.cuda.current_stream(x.device)
        _PS.wait_stream(cur)
        with torch.cuda.stream(_PS):
            yp = torch.matmul(x[:pr], W.t())
        y = _gemm_rows(x, W, fmt)
        cur.wait_stream(_PS)
        yp.record_stream(cur)
        y[:pr].copy_(yp)
        return y
    if rows == "video" and fmt in ("fp8", "a6w6"):  # fast form: low precision for all rows, prefix rows recomputed exactly
        y = _gemm_rows(x, W, fmt)
        torch.matmul(x[:pr], W.t(), out=y[:pr])
        return y
    y = torch.nn.functional.linear(x, W)  # bf16 rows (exact)
    if rows == "video":
        y[pr:] = _gemm_rows(x[pr:], W, fmt)
    else:
        y[:pr] = _gemm_rows(x[:pr].contiguous(), W, fmt)
    return y
