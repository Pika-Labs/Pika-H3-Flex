# SPDX-License-Identifier: Apache-2.0
"""DEV: per-(step, layer, head) attention-structure statistics from the real maps, collected online while
the attention itself stays exact. Only runs with the lossy key "headstats": "<dir>" (no effect on outputs; ~1 ms per layer).

Per call: rows16k4 block-mass estimate s [h, nq, nkb] (16 sampled query rows per 256-row tile, every 4th key), normalised per tile.
Per head (mean over the video query tiles):
  cov{5,10,20,35}  estimated mass in the top 5/10/20/35 % key blocks          (concentration)
  n90, n95          fraction of blocks needed for 90 / 95 % of the mass
  pfx               mass on the text+audio prefix key blocks
  f0, f1, f2p       mass in key blocks of the same latent frame / +-1 frame / >= 2 frames away   (spatial vs temporal)
  col               mass in other frames within +-1.5 latent rows of the query tile's position (temporal 'column' heads)
  reuse35           mass captured at this step by the PREVIOUS step's top-35 % block set (pattern reuse across steps)
  fresh35           = cov35 (the same, with this step's own selection), for comparison
Written per rank to <dir>/stats_r<rank>.jsonl, one line per (step, layer)."""
from __future__ import annotations

import json
import math
import os

import torch

from . import lossy as LS

_PREV = {}
_FH = {}


def _rank():
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        return 0


def collect(q4, k4, v4):
    d = LS.CFG.get("headstats")
    if not d or q4.shape[1] != k4.shape[1]:  # only whole-sequence calls (the 5 s serial path); partial query launches skipped
        return
    from . import sparse2 as S2
    _, n, h, D = q4.shape
    Nk = k4.shape[1]
    nkb = Nk // 128
    nq = -(-n // 256)
    s = S2.rows_score_ks(q4, k4, nq, nkb, 16, 4)
    s = s / s.sum(-1, keepdim=True).clamp_min(1e-30)  # [h, nq, nkb]
    dev = s.device
    kpo = LS.KEY_PREFIX_OFF if (LS.KEY_PREFIX_OFF is not None and LS.KEY_PREFIX_OFF >= 0) else 0
    N = n
    P = LS.SEQ_PREFIX
    # global rows of query tiles / key blocks (key j <-> global (j - kpo) mod N; query r <-> global r - kpo... same buffer)
    qg = (torch.arange(nq, device=dev) * 256 + 128 - kpo) % N
    kg = (torch.arange(nkb, device=dev) * 128 + 64 - kpo) % N
    out = {"step": LS.STEP, "layer": LS.LAYER, "h": h}
    ss, si = s.sort(-1, descending=True)
    cs = ss.cumsum(-1)
    vid = torch.ones(nq, dtype=torch.bool, device=dev)
    if LS.VGEOM is not None:
        v0, Nv, ft, rt = LS.VGEOM
        vid = (qg >= v0) & (qg < v0 + Nv)
        fq = (qg - v0).clamp(0, Nv - 1) // ft
        fk = (kg - v0).clamp(0, Nv - 1) // ft
        kv = (kg >= v0) & (kg < v0 + Nv)
        df = (fq[:, None] - fk[None, :]).abs()  # [nq, nkb]
        pq = (qg - v0).clamp(0, Nv - 1) % ft
        pk = (kg - v0).clamp(0, Nv - 1) % ft
        dp = (pq[:, None] - pk[None, :]).abs()
        dp = torch.minimum(dp, ft - dp)
        m0 = (df == 0) & kv[None]
        m1 = (df == 1) & kv[None]
        m2 = (df >= 2) & kv[None]
        mc = (df >= 1) & (dp <= int(1.5 * rt) + 128) & kv[None]
        for nm, m in (("f0", m0), ("f1", m1), ("f2p", m2), ("col", mc)):
            out[nm] = _vmean((s * m[None]).sum(-1), vid)
    pb = (kg < P)
    out["pfx"] = _vmean((s * pb[None, None]).sum(-1), vid)
    for f in (5, 10, 20, 35):
        kk = max(1, math.ceil(f / 100 * nkb))
        out[f"cov{f}"] = _vmean(cs[..., kk - 1], vid)
    for t in (90, 95):
        out[f"n{t}"] = _vmean(((cs < t / 100).sum(-1) + 1).float() / nkb, vid)
    kk = max(1, math.ceil(0.35 * nkb))
    top = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, si[..., :kk], True)
    key = (LS.LAYER, n, nkb)
    prev = _PREV.get(key)
    if prev is not None and prev[0] == LS.STEP - 1:
        out["reuse35"] = _vmean((s * prev[1]).sum(-1), vid)
    out["fresh35"] = out["cov35"]
    _PREV[key] = (LS.STEP, top)
    fh = _FH.get(d)
    if fh is None:
        os.makedirs(d, exist_ok=True)
        fh = open(os.path.join(d, f"stats_r{_rank()}.jsonl"), "a")
        _FH[d] = fh
    fh.write(json.dumps(out) + "\n")
    fh.flush()


def _vmean(x, vid):
    """x [h, nq] -> per-head mean over video tiles (list of h floats, 4 decimals)."""
    m = vid.float()
    r = (x * m[None]).sum(-1) / m.sum().clamp_min(1)
    return [round(v, 4) for v in r.tolist()]
