# SPDX-License-Identifier: Apache-2.0
"""LOSSY: block-sparse attention, lean implementation. Only runs when the lossy config selects it.

Same selection rule as lossy.sparse_v4_into (per (head, 256-row query tile): top ceil(keep * nk) 128-key blocks by pooled-Q x
pooled-K score, plus the global text+audio prefix keys), but:
  * the ragged LUT is built straight from the top-k indices (row stride nk, lut_start = row * nk): no bool mask, no mask->LUT kernel;
  * Q/K/V are passed as the strided all-to-all views (no .contiguous() copies) and out is written in place;
  * the partial last key block (Nk % 128) is a regular LUT entry, always kept (the kernel masks keys >= Nk): no second launch,
    no LSE merge;
  * optional dense query tiles: tiles that contain text/audio query rows (``dense_q``) attend to every key block;
  * pooled scores in fp32 via sum(dtype=float32) on the bf16 views (no fp32 copy of Q/K).
Selection options (lossy config keys):
  dense_q   1 = query tiles holding global rows [0, prefix) are dense (text + audio queries)
  est       'mean' (default): mean-query x mean-key; 'lse': per-block logsumexp of the tile-mean query against sampled keys
"""
from __future__ import annotations

import math

import torch

_mod = {}


def _v4p():
    if "p" not in _mod:
        import sys
        from .sp_model import AITER_DIR
        if AITER_DIR not in sys.path:
            sys.path.insert(0, AITER_DIR)
        from aiter.ops.mha_v4 import AttentionFormat as AF, AttentionScaleMode as SM, mha_v4_packed
        _mod["p"] = (AF, SM, mha_v4_packed)
    return _mod["p"]


def _pool_mean(x, n, tile):
    """x [n, h, D] bf16 view -> [ceil(n/tile), h, D] fp32 row means per tile (no host syncs)."""
    nf = n // tile
    parts = []
    if nf:
        parts.append(torch.sum(x[:nf * tile].unflatten(0, (nf, tile)), dim=1, dtype=torch.float32).div_(tile))
    if n > nf * tile:
        parts.append(torch.sum(x[nf * tile:n], dim=0, keepdim=True, dtype=torch.float32).div_(n - nf * tile))
    return torch.cat(parts, 0) if len(parts) > 1 else parts[0]


def build_lut(q4, k4, keep: float, key_prefix=(0, 0), q_dense=None, est="mean", local=None):
    """q4 [1, n, h, D], k4 [1, Nk, h, D] views. key_prefix = (first key row, rows) always kept. q_dense = (first query row, rows)
    whose tiles are dense, or None. Returns (kv_block_indices int32 [h*nq*nkb], lut_start int32 [h*nq], lut_count int32 [h*nq])."""
    _, n, h, D = q4.shape
    Nk = k4.shape[1]
    nq = -(-n // 256)
    nkb = Nk // 128  # whole key blocks (the sparse kernel needs Nk % 128 == 0; the remainder keys are attended densely)
    dev = q4.device
    if est == "sub":  # 64-row query sub-tiles x 32-key sub-blocks; block score = logsumexp over the 4 x 4 sub-pairs (scaled)
        qm = _pool_mean(q4[0], n, 64)
        km = _pool_mean(k4[0], nkb * 128, 32)
        ss = torch.einsum("qhd,khd->hqk", qm, km) * (D ** -0.5)  # [h, nq64, nk32]
        pq, pk = 4 * nq - ss.shape[1], 4 * nkb - ss.shape[2]
        if pq or pk:
            ss = torch.nn.functional.pad(ss, (0, pk, 0, pq), value=float("-inf"))
        s = ss.view(h, nq, 4, nkb, 4).permute(0, 1, 3, 2, 4).reshape(h, nq, nkb, 16).logsumexp(-1)
    elif est.startswith("rows") and "k" in est:  # e.g. 'rows16k4': 16 sampled rows per tile, every 4th key of each block
        r_, ks_ = est[4:].split("k")
        s = rows_score_ks(q4, k4, nq, nkb, int(r_), int(ks_))
    elif est.startswith("rows"):  # e.g. 'rows16'
        s = rows_score(q4, k4, nq, nkb, int(est[4:] or 16))
    else:
        qm = _pool_mean(q4[0], n, 256)
        km = _pool_mean(k4[0], nkb * 128, 128)
        s = torch.einsum("qhd,khd->hqk", qm, km)  # [h, nq, nkb]
    big = torch.finfo(torch.float32).max
    p0, pn = key_prefix
    if pn > 0 and p0 >= 0:
        b0, b1 = p0 // 128, min(nkb, -(-(p0 + pn) // 128))
        if b1 > b0:
            s[..., b0:b1] = big
    if local is not None:  # always-kept spatio-temporal window around each video query tile
        li = _local_blocks(n, nq, nkb, key_prefix[0], local, dev)
        if li is not None:
            s.scatter_(-1, li.unsqueeze(0).expand(h, -1, -1), big)
    kk = max(1, min(nkb, math.ceil(keep * nkb)))
    t0 = t1 = 0
    if q_dense is not None and q_dense[1] > 0:
        t0 = max(0, q_dense[0] // 256)
        t1 = max(t0, min(nq, -(-(q_dense[0] + q_dense[1]) // 256)))
    buf, start, cnt = _lut_layout(h, nq, nkb, kk, t0, t1, dev)
    # top-k block ids into the cached LUT buffer, in ASCENDING block order: topk(sorted=False) returns the same set in a different
    # order on every call, and the kernel's online-softmax accumulation order follows the LUT -> non-deterministic output.
    # Dense tiles keep their own entries in the tail of the buffer (written once per layout).
    buf[:h * nq * kk].view(h, nq, kk).copy_(_top_set(s, kk))
    return buf, start, cnt


_LOCAL = {}


def _local_blocks(n, nq, nkb, kpo, local, dev):
    """[nq, m] key-block indices of the window of each query tile: global rows of tile t = (t*256 + qg) mod N; window = same
    frame +-rows latent rows and the same region in frames +-1..frames (clipped to the video); key position = (g + kpo) mod N.
    Non-video tiles get the prefix block (kept anyway)."""
    qg, N, (v0, Nv, ft, rt), rows, frames = local
    key = (n, nq, nkb, kpo, qg, N, v0, Nv, ft, rt, rows, frames, str(dev))
    v = _LOCAL.get(key)
    if v is None:
        import numpy as np
        pad = rows * rt
        span = 256 + 2 * pad
        npts = -(-span // 128) + 1
        g0 = (np.arange(nq) * 256 + qg) % N  # [nq] first global row of each tile
        out = np.full((nq, (2 * frames + 1) * npts), max(0, kpo) // 128, dtype=np.int64)
        for t in range(nq):
            if g0[t] < v0 or g0[t] >= v0 + Nv:
                continue
            col = 0
            for df in range(-frames, frames + 1):
                a = max(v0, g0[t] - pad + df * ft)
                b = min(v0 + Nv - 1, g0[t] + 255 + pad + df * ft)
                for j in range(npts):
                    if a <= b:
                        g = min(a + j * 128, b)
                        blk = ((g + kpo) % N) // 128
                        if blk < nkb:
                            out[t, col] = blk
                    col += 1
        v = torch.from_numpy(out).to(dev)
        if len(_LOCAL) > 256:
            _LOCAL.clear()
        _LOCAL[key] = v
    return v


SORT_IDX = False


def _top_set(s, kk):
    """[h, nq, kk] ids of the kk best blocks, ascending. Deterministic: on real maps many far blocks' estimated mass underflows to
    exactly 0, and topk picks an arbitrary subset among such ties (different on every call -> non-deterministic output);
    a stable descending sort breaks ties by block index."""
    return s.sort(dim=-1, descending=True, stable=True).indices[..., :kk].sort(dim=-1).values
_LAYOUT = {}


def _lut_layout(h, nq, nkb, kk, t0, t1, dev):
    """Cached per shape: LUT buffer [h*nq*kk sparse entries | h*(t1-t0)*nkb dense-tile entries], lut_start, lut_count.
    The buffer is reused by every call with this layout (same stream: the previous kernel has consumed it)."""
    key = (h, nq, nkb, kk, t0, t1, str(dev))
    v = _LAYOUT.get(key)
    if v is None:
        nd = t1 - t0
        A = h * nq * kk
        buf = torch.empty(A + h * nd * nkb, dtype=torch.int32, device=dev)
        start = (torch.arange(h * nq, dtype=torch.int64, device=dev) * kk).view(h, nq)
        cnt = torch.full((h, nq), kk, dtype=torch.int32, device=dev)
        if nd:
            buf[A:] = torch.arange(nkb, dtype=torch.int32, device=dev).repeat(h * nd)
            start[:, t0:t1] = A + (torch.arange(h * nd, dtype=torch.int64, device=dev) * nkb).view(h, nd)
            cnt[:, t0:t1] = nkb
        v = (buf, start.to(torch.int32).view(-1), cnt.view(-1))
        if len(_LAYOUT) > 64:
            _LAYOUT.clear()
        _LAYOUT[key] = v
    return v


# dev timing (PIKA_S2_TIME=1): CUDA events around LUT build and kernel per call, summarised per request on stderr (no syncs in the
# hot path; the summary syncs once when the next request starts).
import os
import sys
TIME_ENV = os.environ.get("PIKA_S2_TIME") == "1"
_ev = []
_last_step = [10 ** 9]


def _summary():
    if not _ev:
        return
    torch.cuda.synchronize()
    per = {}
    for st, a, b, c in _ev:
        d = per.setdefault(st, [0, 0.0, 0.0])
        d[0] += 1
        d[1] += a.elapsed_time(b)
        d[2] += b.elapsed_time(c)
    print("[s2time] " + " | ".join(f"step{st}: {n} calls lut {l:.1f} ms kernel {k:.1f} ms" for st, (n, l, k) in sorted(per.items())),
          file=sys.stderr, flush=True)
    _ev.clear()


def sparse_into(q4, k4, v4, out, keep: float, key_prefix=(0, 0), q_dense=None, est="mean", step=0, timing=False, quant=None,
                local=None):
    """out [1, n, h, D] bf16 (written in place). All inputs strided views (last dim contiguous).
    quant (combination with the low-precision formats): None = BF16 operands; else a callable quant(q4, k4, v4) -> (q, k, v, q_descale,
    k_descale, v_descale, (q_fmt, k_fmt, v_fmt), (q_sm, k_sm, v_sm)) in mha_v4_packed's argument convention (BSHD, keys along dim 1).
    The block selection always uses the BF16 q4/k4."""
    AF, SM, mha_v4_packed = _v4p()
    TIME = timing or TIME_ENV
    if TIME:
        if step < _last_step[0]:
            _summary()
        _last_step[0] = step
        e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        e[0].record()
    idx, start, cnt = build_lut(q4, k4, keep, key_prefix, q_dense, est, local)
    if TIME:
        e[1].record()
    Nk = k4.shape[1]
    nk = Nk // 128 * 128
    if quant is None:
        ops = (q4, k4, v4, q4, k4, v4, AF.BF16, AF.BF16, AF.BF16, SM.NONE, SM.NONE, SM.NONE)
    else:
        qq, kq, vq, qd, kd, vd, fm, sm = quant(q4, k4, v4)
        ops = (qq, kq, vq, qd, kd, vd, *fm, *sm)
    if nk == Nk:
        mha_v4_packed(*ops, out=out, kv_block_indices=idx, lut_start=start, lut_count=cnt)
    else:  # sparse launch on whole blocks (written into out), remainder keys dense, merged in place by LSE
        from . import kernels as K
        qq, kq, vq = ops[:3]
        _, l1 = mha_v4_packed(qq, kq[:, :nk], vq[:, :nk], *ops[3:], out=out, return_lse=True, kv_block_indices=idx, lut_start=start,
                              lut_count=cnt)
        o2, l2 = mha_v4_packed(qq, kq[:, nk:], vq[:, nk:], *ops[3:], return_lse=True)
        dst = out[0]
        assert dst.is_contiguous()
        K.lse_merge_into(dst, 0, out.transpose(1, 2), l1, o2[0].transpose(0, 1), l2[0])  # in place: each row read before write
    if TIME:
        e[2].record()
        _ev.append((step, *e))
    return out


# ---- est='rows': exact per-block attention mass of R sampled query rows per tile (Triton), summed over the tile's rows ----
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["nr", "nkb", "so_h", "so_r", "n"])  # request-length ints: no re-JIT per prompt length
def _blse_kernel(q_ptr, k_ptr, o_ptr, nr, nkb, sq, sqh, sk, skh, so_h, so_r, scale, R: tl.constexpr, STEP: tl.constexpr,
                 OFF: tl.constexpr, n, D: tl.constexpr, BM: tl.constexpr, BPP: tl.constexpr):
    """Row i of the sample = query row (i // R) * 256 + (i % R) * STEP + OFF (clamped to n - 1); out[h, i, b] = logsumexp over the
    128 keys of block b of the scaled scores."""
    pm = tl.program_id(0)
    hh = tl.program_id(1)
    pb = tl.program_id(2)
    ri = pm * BM + tl.arange(0, BM)
    mi = ri < nr
    qrow = tl.minimum((ri // R) * 256 + (ri % R) * STEP + OFF, n - 1)
    rd = tl.arange(0, D)
    q = tl.load(q_ptr + qrow[:, None] * sq + hh * sqh + rd[None, :], mask=mi[:, None], other=0.0)
    rk = tl.arange(0, 128)
    for bb in range(0, BPP):
        b = tl.minimum(pb * BPP + bb, nkb - 1)
        k = tl.load(k_ptr + (b * 128 + rk)[:, None] * sk + hh * skh + rd[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        m = tl.max(s, 1)
        l = m + tl.log(tl.sum(tl.exp(s - m[:, None]), 1))
        tl.store(o_ptr + hh * so_h + ri * so_r + b, l, mask=mi)


BM_, BPP_, NW_ = 128, 8, 4


def rows_score(q4, k4, nq, nkb, R=16):
    """[h, nq, nkb] fp32: sum over R sampled rows of each query tile of the exact attention mass of each key block (whole blocks)."""
    _, n, h, D = q4.shape
    nr = nq * R
    step = 256 // R
    bl = torch.empty((h, nr, nkb), dtype=torch.float32, device=q4.device)
    BM, BPP = BM_, BPP_
    _blse_kernel[(triton.cdiv(nr, BM), h, triton.cdiv(nkb, BPP))](q4, k4, bl, nr, nkb, q4.stride(1), q4.stride(2), k4.stride(1), k4.stride(2),
                                          bl.stride(0), bl.stride(1), D ** -0.5, R=R, STEP=step, OFF=step // 2, n=n, D=D, BM=BM,
                                          BPP=BPP, num_warps=NW_)
    lse = torch.logsumexp(bl, -1, keepdim=True)
    return torch.exp(bl - lse).view(h, nq, R, nkb).sum(2)


@triton.jit(do_not_specialize=["nr", "nkb", "so_h", "so_r", "n"])
def _blse_ks_kernel(q_ptr, k_ptr, o_ptr, nr, nkb, sq, sqh, sk, skh, so_h, so_r, scale, R: tl.constexpr, STEP: tl.constexpr,
                    OFF: tl.constexpr, n, D: tl.constexpr, BM: tl.constexpr, KS: tl.constexpr, BPP: tl.constexpr):
    """As _blse_kernel but each block is represented by its keys 0, KS, 2*KS, ... (128 // KS keys); one 128-wide dot covers KS blocks.
    out[h, i, b] = logsumexp over the block's sampled keys."""
    pm = tl.program_id(0)
    hh = tl.program_id(1)
    pb = tl.program_id(2)
    ri = pm * BM + tl.arange(0, BM)
    mi = ri < nr
    qrow = tl.minimum((ri // R) * 256 + (ri % R) * STEP + OFF, n - 1)
    rd = tl.arange(0, D)
    q = tl.load(q_ptr + qrow[:, None] * sq + hh * sqh + rd[None, :], mask=mi[:, None], other=0.0)
    c = tl.arange(0, 128)
    NPB: tl.constexpr = 128 // KS  # sampled keys per block
    for g in range(0, BPP):
        b0 = (pb * BPP + g) * KS
        bi = b0 + c // NPB
        mb = bi < nkb
        krow = bi * 128 + (c % NPB) * KS
        k = tl.load(k_ptr + krow[:, None] * sk + hh * skh + rd[None, :], mask=mb[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s3 = tl.reshape(s, (BM, KS, NPB))
        m = tl.max(s3, 2)
        l = m + tl.log(tl.sum(tl.exp(s3 - m[:, :, None]), 2))
        jb = b0 + tl.arange(0, KS)
        tl.store(o_ptr + hh * so_h + ri[:, None] * so_r + jb[None, :], l, mask=mi[:, None] & (jb[None, :] < nkb))


def rows_score_ks(q4, k4, nq, nkb, R=16, KS=4):
    _, n, h, D = q4.shape
    nr = nq * R
    step = 256 // R
    bl = torch.empty((h, nr, nkb), dtype=torch.float32, device=q4.device)
    BM, BPP = 128, 2
    ngrp = triton.cdiv(nkb, KS)
    _blse_ks_kernel[(triton.cdiv(nr, BM), h, triton.cdiv(ngrp, BPP))](
        q4, k4, bl, nr, nkb, q4.stride(1), q4.stride(2), k4.stride(1), k4.stride(2), bl.stride(0), bl.stride(1), D ** -0.5,
        R=R, STEP=step, OFF=step // 2, n=n, D=D, BM=BM, KS=KS, BPP=BPP, num_warps=4)
    lse = torch.logsumexp(bl, -1, keepdim=True)
    return torch.exp(bl - lse).view(h, nq, R, nkb).sum(2)
