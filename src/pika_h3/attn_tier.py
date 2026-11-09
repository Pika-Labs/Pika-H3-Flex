# SPDX-License-Identifier: Apache-2.0
"""LOSSY: tiered-precision block attention with compensation of dropped blocks, under one
error budget. OFF by default: only runs when the lossy config (env PIKA_LOSSY / request field "lossy") has the key "tier".

  "tier": spec | [spec0..spec5]   (per generator evaluation; ""/None = this step uses the other lossy keys / lossless)
  spec  = comma list, tiers in importance order, then optional flags:
          "<fmt>:<x>"  fmt in bf16 | mxfp8 | mxfp6 | mxfp4;  x = cumulative block fraction (0.35) or cumulative estimated
                       attention mass "m0.9" (per head and query tile: top-p over the estimated block masses; ragged LUT)
          "comp"       dropped blocks (after the last tier) enter as a log-normal pooled correction: per row r, block b
                       logit s q.kbar_b + ln128 + s^2/2 sum_d q_d^2 var_bd, value vbar_b (keeps the softmax normalisation)
          "est=<e>"    block estimate for this spec (rows16k4 | rows8k4 | mean); default lossy key "est" or rows16k4
          "layers=a-b" the spec applies to DiT blocks [a, b) only (other blocks: the step's "attn" recipe)
          "min<k>"     at least k blocks in the tiers (mass mode), "max<f>" at most fraction f (mass mode)
  e.g. "mxfp6:1" (= dense MXFP6), "mxfp6:0.35,comp", "bf16:m0.5,mxfp6:m0.97,comp", "mxfp6:0.3,mxfp4:1"
  other keys used: "est" (block score: rows16k4 (default) | mean), "attn_prefix_bf16" (text+audio query rows exact bf16).

Per (head, 256-row query tile) the 128-key blocks are ranked once by the estimate (text/audio prefix key blocks forced to the top);
tier i takes ranks [c_{i-1}, c_i) as one sorted-sparse aiter mha_v4 launch in its format (LUT = slices of one sorted index buffer,
no mask), each with LSE export; the remainder keys (Nk % 128) are one small dense bf16 launch; the parts (+ compensation) are
merged by LSE in one Triton pass into out. MX operands are quantized once per layer (cached). LSEs of MX launches get the K-mean
shift restored (aiter's K smoothing), so parts of different formats merge consistently.
"""
from __future__ import annotations

import math
import os
import re
import sys

import torch
import triton
import triton.language as tl

from . import lossy as LS

_M = None


def _aiter():
    global _M
    if _M is None:
        from .sp_model import AITER_DIR
        if AITER_DIR not in sys.path:
            sys.path.insert(0, AITER_DIR)
        from aiter.ops import mha_v4 as M
        _M = M
    return _M


# ------------------------------------------------------------------------------------------------ spec parsing
_SPECS = {}


def parse(spec):
    """-> (tiers [(fmt, kind 'f'|'m', value)], comp bool, kmin int, fmax float)"""
    if spec in _SPECS:
        return _SPECS[spec]
    tiers, comp, kmin, fmax, est, layers = [], False, 1, 1.0, None, None
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "comp":
            comp = True
        elif tok.startswith("est="):
            est = tok[4:]
        elif tok.startswith("layers="):
            lo, hi = tok[7:].split("-")
            layers = (int(lo), int(hi))
        elif tok.startswith("min"):
            kmin = int(tok[3:])
        elif tok.startswith("max"):
            fmax = float(tok[3:])
        else:
            f, x = tok.split(":")
            tiers.append((f, "m", float(x[1:])) if x.startswith("m") else (f, "f", float(x)))
    r = (tiers, comp, kmin, fmax, est, layers)
    _SPECS[spec] = r
    return r


def spec_now():
    """tier spec of the current step and layer, or None ("layers=lo-hi" limits a spec to DiT blocks [lo, hi))."""
    t = LS.CFG.get("tier")
    if not t:
        return None
    s = LS.per_step(t)
    if not s:
        return None
    lr = parse(s)[5]
    if lr is not None and not (lr[0] <= LS.LAYER < lr[1]):
        return None
    return s


# ------------------------------------------------------------------------------------------------ operands (cached per layer)
_OPS = {}
GEN = [0]


def _layer_key(k4):
    return (LS.STEP, LS.LAYER, k4.data_ptr(), k4.shape[1])


def operands(fmt, q4, k4, v4, nk):
    """(q, k, v, qd, kd, vd, fq, fk, fv, sq, sk, sv, v_pack, km) for mha_v4_packed over keys [0, nk)."""
    M = _aiter()
    AF, SM = M.AttentionFormat, M.AttentionScaleMode
    key = (fmt,) + _layer_key(k4) + (q4.data_ptr(), q4.shape[1])
    hit = _OPS.get(key)
    if hit is not None:
        return hit
    if len(_OPS) > 24:
        _OPS.clear()
    sc = 128 ** -0.5
    if fmt == "bf16":
        k_, v_ = k4[:, :nk], v4[:, :nk]
        r = (q4, k_, v_, q4, k_, v_, AF.BF16, AF.BF16, AF.BF16, SM.NONE, SM.NONE, SM.NONE, M.AttentionPack.DEFAULT, None, None)
        _OPS[key] = r
        return r
    kv = _kv_ops(fmt, k4, v4, nk)  # K/V producers once per layer (shared by all query launches of the layer)
    qc = _q_contig(q4)
    fmt = _base(fmt)
    if fmt == "mxfp6":
        quant_q = M.mha_v4_mx_producers(AF.MXFP6, 128)[0]
        qq, qd = quant_q(qc, M.mha_v4_q_multiplier(sc))
    elif fmt == "mxfp4":
        quant_q = M.mha_v4_mx_producers(AF.MXFP4, 128)[0]
        qq, qd = quant_q(qc, M.mha_v4_q_multiplier(sc))
    elif fmt == "mxfp8":
        qq, qd = M.quantize_mxfp8_q(qc, M.mha_v4_q_multiplier(sc))
    else:
        raise KeyError(fmt)
    kq, vq, kd, vd, fmts, km, vbar = kv
    r = (qq, kq, vq, qd, kd, vd, *fmts, km, vbar)
    _OPS[key] = r
    return r


def _kv_ops(fmt, k4, v4, nk):
    M = _aiter()
    AF, SM = M.AttentionFormat, M.AttentionScaleMode
    key = ("kvop", fmt) + _layer_key(k4) + (nk,)
    hit = _OPS.get(key)
    if hit is not None:
        return hit
    kc, vc = _kv_contig(k4, v4, nk)
    flags = fmt[len(_base(fmt)):]
    fmt = _base(fmt)
    vbar = None
    if "c" in flags:  # V centred per head (exact: the mean is added back to the output); quantization sees only the deviations
        st = max(1, nk // 4096)
        vbar = vc[:, ::st].float().mean(1, keepdim=True)  # [1, 1, h, D]
        vc = (vc.float() - vbar).to(vc.dtype)
    if "k" in flags:  # K centred per head (exact for the softmax; LSE shift restored below)
        st = max(1, nk // 4096)
        _KM_FORCE[0] = kc[:, ::st].float().mean(1).contiguous()  # [1, h, D]
    else:
        _KM_FORCE[0] = None
    if fmt == "mxfp6":
        _, quant_k, quant_v = M.mha_v4_mx_producers(AF.MXFP6, 128)
        km = _kmean(kc, M._RawRecipeKind.MXFP6)
        kraw, kdraw = quant_k(kc, km)
        kq, kd = M.mxfp6_k_view(kraw, kdraw, 1, nk, kc.shape[2])
        vq, vd = quant_v(vc)
        plan = M._resolve_raw_recipe(AF.MXFP6, AF.MXFP6, AF.MXFP6, None, None, None, sparse=True)
        r = (kq, vq, kd, vd, (AF.MXFP6, AF.MXFP6, AF.MXFP6, *plan.scale_modes, plan.v_pack), km, vbar)
    elif fmt == "mxfp4":
        _, quant_k, quant_v = M.mha_v4_mx_producers(AF.MXFP4, 128)
        km = _kmean(kc, M._RawRecipeKind.MXFP4)
        kq0, kd = quant_k(kc, km)
        vq0, vd = quant_v(vc)
        plan = M._resolve_raw_recipe(AF.MXFP4, AF.MXFP4, AF.MXFP4, None, None, None, sparse=True)
        r = (M.mxfp4_k_view(kq0, kd), M.mxfp4_v_view(vq0, vd, nk), kd, vd,
             (AF.MXFP4, AF.MXFP4, AF.MXFP4, *plan.scale_modes, plan.v_pack), km, vbar)
    elif fmt == "mxfp8":
        km = _kmean(kc, M._RawRecipeKind.MXFP8)
        kq, kd = M.quantize_mxfp8_k(kc, km)
        vq, vd = M.quantize_fp8(vc)
        r = (kq, vq, kd, vd, (AF.FP8, AF.FP8, AF.FP8, SM.E8M0_PER_1X32, SM.E8M0_PER_1X32, SM.F32_PER_TENSOR,
                              M.AttentionPack.DEFAULT), km, vbar)
    else:
        raise KeyError(fmt)
    _OPS[key] = r
    return r


KSMOOTH = os.environ.get("PIKA_TIER_KSMOOTH", "0") == "1"


_KM_FORCE = [None]


def _base(fmt):
    m = re.match(r"(bf16|mxfp[468])", fmt)
    return m.group(1) if m else fmt


def _kmean(kc, kind):
    """aiter's K smoothing (shared K component removed before quantization) is gated off on every real layer measured (common mode
    0.12-0.76 < 0.85), so by default it is skipped: no k-mean pass, no LSE shift (PIKA_TIER_KSMOOTH=1 restores aiter's behaviour)."""
    if _KM_FORCE[0] is not None:
        return _KM_FORCE[0]
    return _aiter()._k_mean(kc, kind) if KSMOOTH else None


_FULL = {}


def _full(n, val, dev):
    key = (n, val, str(dev))
    t = _FULL.get(key)
    if t is None:
        t = torch.full((n,), val, dtype=torch.int32, device=dev)
        _FULL[key] = t
    return t


_ADD = {}


def _starts_off(h, nq, kk, c0, dev):
    key = (h, nq, kk, c0, str(dev))
    t = _ADD.get(key)
    if t is None:
        t = _starts(h, nq, kk, dev) + c0
        _ADD[key] = t
    return t


def _kv_contig(k4, v4, nk):
    """contiguous copies of the WHOLE K/V (shared with the exact prefix-row recompute), sliced to the first nk keys."""
    kf, vf = _kv_full(k4, v4)
    return kf[:, :nk], vf[:, :nk]


def _kv_full(k4, v4):
    key = ("kvfull",) + _layer_key(k4)
    hit = _OPS.get(key)
    if hit is None:
        hit = (k4.contiguous(), v4.contiguous())
        _OPS[key] = hit
    return hit


def _q_contig(q4):
    key = ("q", q4.data_ptr(), q4.shape[1], LS.STEP, LS.LAYER)
    hit = _OPS.get(key)
    if hit is None:
        hit = q4.contiguous()
        _OPS[key] = hit
    return hit


# ------------------------------------------------------------------------------------------------ block ranking
def block_scores(q4, k4, nq, nkb, est):
    """[h, nq, nkb] fp32 estimated attention mass per (query tile, key block), not normalised."""
    from . import sparse2 as S2
    if est.startswith("rows") and "k" in est:
        r_, ks_ = est[4:].split("k")
        return S2.rows_score_ks(q4, k4, nq, nkb, int(r_), int(ks_))
    if est.startswith("rows"):
        return S2.rows_score(q4, k4, nq, nkb, int(est[4:] or 16))
    n = q4.shape[1]
    qm = S2._pool_mean(q4[0], n, 256)
    km = S2._pool_mean(k4[0], nkb * 128, 128)
    s = torch.einsum("qhd,khd->hqk", qm, km) * (q4.shape[3] ** -0.5)
    return torch.softmax(s, -1)


_LAY = {}


def _starts(h, nq, kk, dev):
    key = (h, nq, kk, str(dev))
    v = _LAY.get(key)
    if v is None:
        v = torch.arange(h * nq, dtype=torch.int32, device=dev) * kk
        _LAY[key] = v
    return v


def rank_blocks(s, prefix_blocks, kk):
    """s [h, nq, nkb] -> int32 [h*nq*kk] block ids sorted by decreasing score (prefix blocks first)."""
    if prefix_blocks is not None:
        b0, b1 = prefix_blocks
        s[..., b0:b1] = torch.finfo(torch.float32).max
    if kk >= s.shape[-1]:
        idx = s.argsort(-1, descending=True)
    else:
        idx = s.sort(-1, descending=True, stable=True).indices[..., :kk]  # deterministic among ties
    return idx.to(torch.int32).reshape(-1)


# ------------------------------------------------------------------------------------------------ Triton: compensation, merge
@triton.jit(do_not_specialize=["n", "nkb", "nq"])
def _comp_kernel(q_ptr, sq, sqh, kb_ptr, kv_ptr, vb_ptr, keep_ptr, o_ptr, l_ptr, n, nkb, nq, scale,
                 D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """Per (64-row block, head): softmax over the DROPPED key blocks of the row's query tile with the log-normal block logit.
    kb/kv/vb: [h, nkbp, D] fp32 (kv pre-scaled by s^2/2; kb already includes nothing else), keep: int8 [h, nq, nkbp].
    o: [h, n, D] fp32 (normalised), l: [h, n] fp32 (natural-log mass, -inf if nothing dropped)."""
    pm = tl.program_id(0)
    hh = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    mr = rows < n
    rd = tl.arange(0, D)
    q = tl.load(q_ptr + rows[:, None] * sq + hh * sqh + rd[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
    q2 = q * q
    tile = (pm * BM) // 256
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    LN128: tl.constexpr = 4.852030263919617
    for b0 in range(0, nkb, BN):
        bi = b0 + tl.arange(0, BN)
        mb = bi < nkb
        kp = (hh * nkb + bi)[:, None] * D + rd[None, :]
        kb = tl.load(kb_ptr + kp, mask=mb[:, None], other=0.0)
        kv = tl.load(kv_ptr + kp, mask=mb[:, None], other=0.0)
        keep = tl.load(keep_ptr + (hh * nq + tile) * nkb + bi, mask=mb, other=1)
        s = tl.dot(q, tl.trans(kb)) * scale + tl.dot(q2, tl.trans(kv)) + LN128
        s = tl.where((keep == 0)[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        vb = tl.load(vb_ptr + kp, mask=mb[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, vb)
        m_i = m_new
    has = l_i > 0
    o = acc / tl.where(has, l_i, 1.0)[:, None]
    lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), float("-inf"))
    tl.store(o_ptr + (hh * n + rows)[:, None] * D + rd[None, :], o, mask=mr[:, None])
    tl.store(l_ptr + hh * n + rows, lse, mask=mr)


def compensation(q4, k4, v4, nkb, keep_i8):
    """-> (o [h, n, D] fp32, lse [h, n] fp32) over the dropped blocks (keep_i8 [h, nq, nkb] int8, 0 = dropped)."""
    _, n, h, D = q4.shape
    nq = keep_i8.shape[1]
    kb_, vb_ = k4[0, :nkb * 128].unflatten(0, (nkb, 128)), v4[0, :nkb * 128].unflatten(0, (nkb, 128))
    kbar = kb_.mean(1, dtype=torch.float32)  # [nkb, h, D]
    k2 = (kb_.float().pow_(2)).mean(1)
    sc = D ** -0.5
    kvar = (k2 - kbar * kbar).clamp_min_(0).mul_(0.5 * sc * sc)
    vbar = vb_.mean(1, dtype=torch.float32)
    kbar, kvar, vbar = (x.transpose(0, 1).contiguous() for x in (kbar, kvar, vbar))  # [h, nkb, D]
    o = torch.empty((h, n, D), dtype=torch.float32, device=q4.device)
    l = torch.empty((h, n), dtype=torch.float32, device=q4.device)
    BM = 64
    _comp_kernel[(triton.cdiv(n, BM), h)](q4, q4.stride(1), q4.stride(2), kbar, kvar, vbar, keep_i8, o, l, n, nkb, nq, sc,
                                          D=D, BM=BM, BN=32, num_warps=4)
    return o, l


@triton.jit(do_not_specialize=['so_n', 'so_h', 's0n', 's0h', 's1n', 's1h', 's2n', 's2h', 's3n', 's3h', 's4n', 's4h', 's5n', 's5h', 'lh0', 'lh1', 'lh2', 'lh3', 'lh4', 'lh5', 'n'])  # no re-JIT per request length
def _merge_kernel(out_ptr, so_n, so_h, o0, l0, o1, l1, o2, l2, o3, l3, o4, l4, o5, l5,
                  s0n, s0h, s1n, s1h, s2n, s2h, s3n, s3h, s4n, s4h, s5n, s5h, lh0, lh1, lh2, lh3, lh4, lh5, n,
                  NP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr):
    """out[r, h] = sum_i exp(l_i - L) o_i, L = logsumexp_i l_i. o_i element (r, h, d) at o_i + r*sin + h*sih + d; l_i at h*lhi + r."""
    pm = tl.program_id(0)
    hh = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    mr = rows < n
    rd = tl.arange(0, D)
    L = tl.full((BM,), float("-inf"), tl.float32)
    la = tl.load(l0 + hh * lh0 + rows, mask=mr, other=float("-inf"))
    L = tl.maximum(L, la)
    if NP > 1:
        lb = tl.load(l1 + hh * lh1 + rows, mask=mr, other=float("-inf"))
        L = tl.maximum(L, lb)
    if NP > 2:
        lc = tl.load(l2 + hh * lh2 + rows, mask=mr, other=float("-inf"))
        L = tl.maximum(L, lc)
    if NP > 3:
        ld = tl.load(l3 + hh * lh3 + rows, mask=mr, other=float("-inf"))
        L = tl.maximum(L, ld)
    if NP > 4:
        le = tl.load(l4 + hh * lh4 + rows, mask=mr, other=float("-inf"))
        L = tl.maximum(L, le)
    if NP > 5:
        lf = tl.load(l5 + hh * lh5 + rows, mask=mr, other=float("-inf"))
        L = tl.maximum(L, lf)
    L = tl.where(L == float("-inf"), 0.0, L)
    w = tl.exp(la - L)
    den = w
    acc = w[:, None] * tl.load(o0 + rows[:, None] * s0n + hh * s0h + rd[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
    if NP > 1:
        w = tl.exp(lb - L)
        den += w
        acc += w[:, None] * tl.load(o1 + rows[:, None] * s1n + hh * s1h + rd[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
    if NP > 2:
        w = tl.exp(lc - L)
        den += w
        acc += w[:, None] * tl.load(o2 + rows[:, None] * s2n + hh * s2h + rd[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
    if NP > 3:
        w = tl.exp(ld - L)
        den += w
        acc += w[:, None] * tl.load(o3 + rows[:, None] * s3n + hh * s3h + rd[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
    if NP > 4:
        w = tl.exp(le - L)
        den += w
        acc += w[:, None] * tl.load(o4 + rows[:, None] * s4n + hh * s4h + rd[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
    if NP > 5:
        w = tl.exp(lf - L)
        den += w
        acc += w[:, None] * tl.load(o5 + rows[:, None] * s5n + hh * s5h + rd[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
    acc = acc / tl.where(den > 0, den, 1.0)[:, None]
    tl.store(out_ptr + rows[:, None] * so_n + hh * so_h + rd[None, :], acc.to(out_ptr.dtype.element_ty), mask=mr[:, None])


def merge_into(out, parts):
    """out [1, n, h, D] bf16 view; parts: list of (o, lse, (sn, sh), lse_head_stride) with o element (r, h, d) at r*sn + h*sh + d."""
    _, n, h, D = out.shape
    NP = len(parts)
    assert 1 <= NP <= 6
    pad = parts + [parts[0]] * (6 - NP)
    args = []
    for o, l, _, _ in pad:
        args += [o, l]
    st = []
    for _, _, (sn, sh), _ in pad:
        st += [sn, sh]
    lh = [p[3] for p in pad]
    BM = 32
    _merge_kernel[(triton.cdiv(n, BM), h)](out, out.stride(1), out.stride(2), *args, *st, *lh, n, NP=NP, D=D, BM=BM, num_warps=4)
    return out


# ------------------------------------------------------------------------------------------------ main entry
TIME_ENV = os.environ.get("PIKA_TIER_TIME") == "1"
_ev = []


def summary():
    if not _ev:
        return
    torch.cuda.synchronize()
    per = {}
    for st, a, b, c in _ev:
        d = per.setdefault(st, [0, 0.0, 0.0])
        d[0] += 1
        d[1] += a.elapsed_time(b)
        d[2] += b.elapsed_time(c)
    print("[tiertime] " + " | ".join(f"step{s}: {n} calls rank {r:.1f} ms attn {k:.1f} ms" for s, (n, r, k) in sorted(per.items())),
          file=sys.stderr, flush=True)
    _ev.clear()


FASTLAUNCH = os.environ.get("PIKA_TIER_FASTLAUNCH", "1") == "1"


def _launch(ops, out, lut, q_rows=None):
    """one sorted-sparse launch with LSE. FASTLAUNCH: straight to aiter's launch op (operands were validated by the first
    mha_v4_packed call of the format; saves ~50-100 us of Python checks per call)."""
    M = _aiter()
    (qq, kq, vq, qd, kd, vd, fq, fk, fv, sq, sk, sv, vpack, km) = ops[:14]
    if not FASTLAUNCH or ("v", fq) not in _OK:
        r = M.mha_v4_packed(qq, kq, vq, qd, kd, vd, fq, fk, fv, sq, sk, sv, v_pack=vpack, softmax_scale=128 ** -0.5, out=out,
                            return_lse=True, kv_block_indices=lut[0], lut_start=lut[1], lut_count=lut[2])
        _OK.add(("v", fq))
        return r
    b, n, h = qq.shape[0], qq.shape[1], qq.shape[2]
    if out is None:
        out = torch.empty((b, n, h, 128), dtype=torch.bfloat16, device=qq.device)
    lse = torch.empty((b, h, n), dtype=torch.float32, device=qq.device)
    M._mha_v4_fwd_sparse_launch(qq, kq, vq, qd, kd, vd, out, int(fq), int(fk), int(fv), int(vpack), int(sq), int(sk), int(sv),
                                128 ** -0.5, lut[0], lut[1], lut[2], lse)
    return out, lse


_OK = set()


def tier_into(q4, k4, v4, out, spec):
    """q4 [1, n, h, D], k4/v4 [1, Nk, h, D] strided views; out [1, n, h, D] bf16 (written)."""
    M = _aiter()
    AF, SM = M.AttentionFormat, M.AttentionScaleMode
    tiers, comp, kmin, fmax, est, _ = parse(spec)
    if len(tiers) == 1 and tiers[0][1] == "f" and tiers[0][2] >= 1.0 and tiers[0][0] in ("mxfp6", "mxfp8"):
        from . import lowprec  # dense single format: the lowprec path (whole K incl. remainder, no merge)
        return lowprec.attn_into(q4, k4, v4, out, tiers[0][0])
    _, n, h, D = q4.shape
    Nk = k4.shape[1]
    nkb = Nk // 128
    nk = nkb * 128
    nq = -(-n // 256)
    dev = q4.device
    TIME = TIME_ENV or LS.CFG.get("tiertime")
    if TIME:
        e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        e[0].record()
    pref = None
    if LS.KEY_PREFIX_OFF is not None and LS.KEY_PREFIX_OFF >= 0 and LS.SEQ_PREFIX > 0:
        p0 = LS.KEY_PREFIX_OFF
        pref = (p0 // 128, min(nkb, -(-(p0 + LS.SEQ_PREFIX) // 128)))
    full = len(tiers) == 1 and tiers[0][1] == "f" and tiers[0][2] >= 1.0
    mass_mode = any(t[1] == "m" for t in tiers)
    cuts = None
    if full:
        kk = nkb
        idx = None
    else:
        s = block_scores(q4, k4, nq, nkb, est or LS.CFG.get("est", "rows16k4"))
        if mass_mode:
            kk = nkb
            s0 = s
            if pref is not None:
                s = s.clone()
                s[..., pref[0]:pref[1]] = torch.finfo(torch.float32).max  # prefix blocks first
            si = s.argsort(-1, descending=True)
            idx = si.to(torch.int32).reshape(-1)
            cm = s0.gather(-1, si).cumsum(-1)  # cumulative ESTIMATED mass in rank order
            cm = cm / cm[..., -1:].clamp_min(1e-30)
            kcap = max(kmin, math.ceil(fmax * nkb))
            cuts = []
            prev = None
            for f, kind, x in tiers:
                if kind == "m":
                    c = (cm < x).sum(-1, dtype=torch.int32) + 1  # blocks needed to reach mass x
                else:
                    c = torch.full((h, nq), max(1, min(nkb, math.ceil(x * nkb))), dtype=torch.int32, device=dev)
                c = c.clamp_(min(kmin, nkb), min(kcap, nkb))
                if prev is not None:
                    c = torch.maximum(c, prev)
                cuts.append(c)
                prev = c
        else:
            fr = [t[2] for t in tiers]
            kk = max(1, min(nkb, math.ceil(fr[-1] * nkb)))
            idx = rank_blocks(s, pref, kk)
    if TIME:
        e[1].record()
    parts = []
    base = None
    prev_c = None
    for i, (f, kind, x) in enumerate(tiers):
        ops = operands(f, q4, k4, v4, nk)
        if full:
            if f == "bf16":
                o, l = M.mha_v4_packed(q4, k4[:, :nk], v4[:, :nk], q4, k4, v4, AF.BF16, AF.BF16, AF.BF16, SM.NONE, SM.NONE, SM.NONE,
                                       out=out, return_lse=True)
            else:
                (qq, kq, vq, qd, kd, vd, fq, fk, fv, sq, sk, sv, vpack, km) = ops[:14]
                o, l = M.mha_v4_packed(qq, kq, vq, qd, kd, vd, fq, fk, fv, sq, sk, sv, v_pack=vpack, softmax_scale=128 ** -0.5,
                                       out=out if out.is_contiguous() else None, return_lse=True)
        else:
            if cuts is not None:
                c = cuts[i]
                st = _starts(h, nq, nkb, dev).view(h, nq) + (0 if prev_c is None else prev_c)
                cnt = c - (0 if prev_c is None else prev_c)
                prev_c = c
                lut = (idx, st.reshape(-1).to(torch.int32), cnt.reshape(-1).to(torch.int32))
            else:
                c0 = 0 if i == 0 else max(1, min(nkb, math.ceil(tiers[i - 1][2] * nkb)))
                c1 = kk if i == len(tiers) - 1 else max(1, min(nkb, math.ceil(x * nkb)))
                lut = (idx, _starts_off(h, nq, kk, c0, dev), _full(h * nq, c1 - c0, dev))
            o, l = _launch(ops, out if (i == 0 and out.is_contiguous()) else None, lut)
        if ops[14] is not None:  # centred V: add the mean back (bf16 output, in place)
            o.add_(ops[14].to(o.dtype))
        km = ops[13]
        if km is not None:  # aiter K smoothing shifted the scores by q.k_mean (all keys): restore for cross-format merging
            l = l + torch.einsum("bshd,bhd->bhs", q4, km.to(q4.dtype)).float() * (128 ** -0.5)
        parts.append((o, l, (o.stride(1), o.stride(2)), l.stride(1)))
    if nk < Nk:  # remainder keys: dense bf16
        o2, l2 = M.mha_v4_packed(q4, k4[:, nk:], v4[:, nk:], q4, k4, v4, AF.BF16, AF.BF16, AF.BF16, SM.NONE, SM.NONE, SM.NONE,
                                 return_lse=True)
        parts.append((o2, l2, (o2.stride(1), o2.stride(2)), l2.stride(1)))
    if comp and not full:
        keep = torch.zeros((h, nq, nkb), dtype=torch.int8, device=dev)
        if cuts is not None:
            ar = torch.arange(nkb, device=dev, dtype=torch.int32)
            kept = (ar[None, None, :] < cuts[-1][..., None]).to(torch.int8)  # rank < cut
            keep.scatter_(-1, idx.view(h, nq, nkb).long(), kept)
        else:
            keep.scatter_(-1, idx.view(h, nq, kk).long(), 1)
        oc, lc = compensation(q4, k4, v4, nkb, keep)
        parts.append((oc.unsqueeze(0).transpose(1, 2), lc.unsqueeze(0), (oc.stride(1), oc.stride(0)), lc.stride(0)))
    if len(parts) == 1 and parts[0][0].data_ptr() == out.data_ptr():
        pass
    else:
        merge_into(out, parts)
    if LS.get("attn_prefix_bf16"):
        _prefix_rows(q4, k4, v4, out, nk)
    if TIME:
        e[2].record()
        _ev.append((LS.STEP, *e))
    return out


def _prefix_rows(q4, k4, v4, out, nk):
    """text+audio query rows (global rows [0, SEQ_PREFIX) at KEY_PREFIX_OFF of the key buffer) recomputed exactly (bf16, all keys,
    keys split 16 ways)."""
    off, P = LS.KEY_PREFIX_OFF, LS.SEQ_PREFIX
    if off is None or off < 0 or P <= 0:
        return
    r0 = LS._q_offset(q4, k4)
    a, b = max(off, r0), min(off + P, r0 + q4.shape[1])
    if b <= a:
        return
    M = _aiter()
    AF = M.AttentionFormat
    kc, vc = _kv_full(k4, v4)
    o = M.mha_v4(q4[:, a - r0:b - r0].contiguous(), kc, vc, AF.BF16, AF.BF16, AF.BF16, kv_splits=16)
    out[:, a - r0:b - r0].copy_(o)
