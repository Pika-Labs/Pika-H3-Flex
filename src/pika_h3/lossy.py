# SPDX-License-Identifier: Apache-2.0
"""LOSSY track (main line). OFF by default: with no PIKA_LOSSY env and no request ``lossy`` field nothing here runs.

Configuration (a preset name or a dict, JSON; presets and validation in lossy_presets.py, user docs in docs/PIKA_HANDOVER.md):
  * process default: env ``PIKA_LOSSY`` (preset name or JSON object), applied at import;
  * per request: the request field ``"lossy": "near_lossless" | "balanced" | "fast" | {...}`` replaces the default for that
    request (``"lossy": {}`` = lossless). Unknown presets / keys / values raise LossyConfigError (no silent fallback).
    Every rank sees the same request (collective), so the state is set identically on all ranks (engine.run_request).

Known keys (every key must be registered in lossy_presets.SUPPORTED_KEYS / EXPERIMENTAL_KEYS, else it is rejected; read them with
``get(name, default)``):
  sparse   list of keep fractions per generator evaluation 0..5 (1 = dense), e.g. [1, 1, .35, .35, .35, .35] (= late35)
  prefix   tokens at the start of the global packed sequence (text, then audio) always attended (default 512); "auto" = L + Na
  q_tile   (fixed by the gfx950 kernel: 256-row query tiles x 128-key blocks)
  impl     "s2" = sparse2.py's lean block-sparse path (ragged LUT, strided operands); est = its block estimator (e.g. "rows16k4")
  dense_q  1 = the text + audio query rows attend to every key (s2 path: dense query tiles, or see s2_prefix_bf16)
  s2_prefix_bf16  true (with dense_q, s2 path): the text + audio query rows are recomputed afterwards with the exact bf16 kernel
           (keys split 16 ways) instead of running dense query tiles inside the sparse launch: exact for those rows even when the
           step's attention format is low precision, and no long-pole dense tiles (used by the fastest tier)

Block-sparse attention (sparse v4): per (head, 256-row query tile) the top
ceil(keep * nk) 128-key blocks by mean-query x mean-key score + the prefix blocks; skipped blocks never load (aiter mha_v4 BF16
sorted-sparse asm kernel). Keys past the last whole 128-block are attended densely and LSE-merged.
Integration with the fast lossless infrastructure: the hook is inside sp_model._v4_into, so every v4 path keeps its SDMA transfers,
GEMM padding and pipelining. The key order differs per path (serial SDMA: global order; PIPE 4/5: receive order d+1, ..., d) ->
the path sets KEY_PREFIX_OFF to where global row 0 sits in its key buffer. PIKA_SPLIT (world 4 only) is turned off on sparse steps.
"""
from __future__ import annotations

import json
import math
import os

import torch

CFG: dict = {}
SPARSE: list = []
PREFIX = 512
PREFIX_AUTO = False
ACTIVE = False          # any lossy option set (cheap guard for the hot path)
STEP = 0                # current generator evaluation (set by SPTransformer.forward)
VGEOM = None            # (video start row, video rows, tokens per latent frame, tokens per latent row) of the request
LAYER = 0               # current DiT block (set by SPTransformer.forward when ACTIVE)
KEY_PREFIX_OFF = 0      # offset of global row 0 in the current key buffer (set by the attention path)
SEQ_PREFIX = 512        # resolved prefix tokens of the current request
STATS = {}


def _parse(cfg):
    """preset name | dict | JSON string | None -> validated raw spec (lossy_presets.expand; raises LossyConfigError)."""
    from .lossy_presets import expand
    return expand(cfg)


def apply(cfg) -> None:
    """Set the lossy state (all ranks, before the request runs)."""
    global CFG, SPARSE, PREFIX, PREFIX_AUTO, ACTIVE
    CFG = _parse(cfg)
    sp = CFG.get("sparse") or []
    if isinstance(sp, str):
        sp = [float(x) for x in sp.split(",") if x]
    SPARSE = [float(x) for x in sp]
    if SPARSE and all(x >= 1.0 for x in SPARSE):
        SPARSE = []
    p = CFG.get("prefix", 512)
    PREFIX_AUTO = p == "auto"
    PREFIX = 512 if PREFIX_AUTO else int(p)
    ACTIVE = bool(CFG)


DEFAULT = os.environ.get("PIKA_LOSSY", "")
apply(DEFAULT)
DEFAULT_CFG = dict(CFG)


def for_request(req) -> None:
    import sys as _sys
    s2 = _sys.modules.get("pika_h3.sparse2")
    at = _sys.modules.get("pika_h3.attn_tier")
    if at is not None and at._ev:
        at.summary()
    if s2 is not None and s2._ev:  # dev timing of the previous request (lossy key s2time)
        s2._summary()
    apply(req["lossy"] if "lossy" in req else DEFAULT_CFG)


def get(name, default=None):
    return CFG.get(name, default)


def keep_for_step(step: int) -> float:
    if not SPARSE:
        return 1.0
    return SPARSE[step] if step < len(SPARSE) else SPARSE[-1]


def sparse_now() -> float:
    return keep_for_step(STEP) if SPARSE else 1.0


def per_step(val, step=None, default=None):
    """'x' | [x0..x5] (last entry repeats) | None -> value for generator evaluation `step` (default: current STEP)."""
    step = STEP if step is None else step
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return val[step] if step < len(val) else (val[-1] if val else default)
    return val


def attn_recipe() -> str:
    """attention format for the current step ('bf16' = exact kernel); 'tier' when tiered
    spec (key "tier", attn_tier.py) is set for this step (the dense-wave tail paths are then bypassed, as for MX steps)."""
    if CFG.get("tier") and per_step(CFG.get("tier")):
        from . import attn_tier
        if attn_tier.spec_now():
            return "tier"
    if CFG.get("headstats"):
        return "headstats"
    r = per_step(CFG.get("attn"), default="bf16") or "bf16"
    lr = CFG.get("attn_layers")  # dev map: [lo, hi) or a per-step list of [lo, hi] / None (= all layers)
    if r != "bf16" and lr is not None:
        lr = per_step(lr) if (lr and isinstance(lr[0], (list, tuple)) or (lr and lr[0] is None)) else lr
        if lr and not (lr[0] <= LAYER < lr[1]):
            return "bf16"
    return r


def attn_into(q, k, v, out):
    """Combination dispatcher called by sp_model._v4_into when ACTIVE. Returns out, or None for the lossless kernel.
    sparse keep < 1 -> block-sparse kernel in the step's attention format; else non-bf16 format -> lowprec dense; else None."""
    rec = attn_recipe()
    if rec == "headstats":  # dev: attention-structure statistics, attention itself exact (one launch)
        from . import attn_stats
        attn_stats.collect(q, k, v)
        return None
    if rec == "tier":  # tiered precision + compensation (attn_tier.py)
        from . import attn_tier
        return attn_tier.tier_into(q, k, v, out, attn_tier.spec_now())
    keep = sparse_now()
    if keep < 1.0:
        if CFG.get("impl") == "s2" and rec in ("bf16", "bf16fp8", "fp8"):  # lean path (+ sliceable formats)
            quant = None
            if rec != "bf16":
                from . import lowprec
                quant = lowprec.sparse_quant(rec)
            return _s2_into(q, k, v, out, keep, quant)
        sparse_v4_into(q, k, v, out, keep, rec)  # default path; MX formats (whole key blocks + bf16 remainder)
        if CFG.get("dense_q"):  # text + audio query rows: exact dense bf16 attention (as sparse2's dense_q)
            qd = _dense_q_rows(q, k)
            if qd is not None:
                a0, a1 = qd
                from . import sparse2 as S2
                AF, SM, mp = S2._v4p()
                qs = q[:, a0:a1]
                mp(qs, k, v, qs, k, v, AF.BF16, AF.BF16, AF.BF16, SM.NONE, SM.NONE, SM.NONE, out=out[:, a0:a1])
        return out
    if rec != "bf16":
        from . import lowprec
        return lowprec.attn_into(q, k, v, out, rec)
    return None


def set_request_geometry(L: int, Na: int) -> None:
    global SEQ_PREFIX
    SEQ_PREFIX = (L + Na) if PREFIX_AUTO else PREFIX


def _v4():
    import sys
    from .sp_model import AITER_DIR
    if AITER_DIR not in sys.path:
        sys.path.insert(0, AITER_DIR)
    from aiter.ops.mha_v4 import AttentionFormat as AF, mha_v4, mha_v4_packed, AttentionScaleMode as SM
    return AF, mha_v4, mha_v4_packed, SM


def block_mask(q4, k4, keep: float, prefix_off: int, prefix_tokens: int):
    """q4 [1, n, h, D], k4 [1, Nk, h, D] views. Returns bool [1, h, nq, nk128] (nk128 = Nk // 128 whole key blocks).
    Key rows [prefix_off, prefix_off + prefix_tokens) (the global text + audio prefix) are always kept."""
    _, N, h, D = q4.shape
    Nk = k4.shape[1]
    nq, nk = -(-N // 256), Nk // 128
    nf = N // 256  # whole query tiles; reductions read bf16 and accumulate in fp32 (no fp32 copy of Q/K)
    qm = torch.empty((nq, h, D), dtype=torch.float32, device=q4.device)
    if nf:
        torch.sum(q4[0, :nf * 256].unflatten(0, (nf, 256)), dim=1, dtype=torch.float32, out=qm[:nf])
        qm[:nf] /= 256.0
    if nq > nf:
        qm[nf] = q4[0, nf * 256:].sum(0, dtype=torch.float32) / (N - nf * 256)
    km = k4[0, :nk * 128].unflatten(0, (nk, 128)).sum(1, dtype=torch.float32) / 128.0   # [nk, h, D]
    s = torch.einsum("qhd,khd->hqk", qm, km)                          # [h, nq, nk]
    kk = max(1, min(nk, math.ceil(keep * nk)))
    idx = s.sort(dim=-1, descending=True, stable=True).indices[..., :kk]  # deterministic among tied (zero) estimates
    m = torch.zeros((h, nq, nk), dtype=torch.bool, device=q4.device)
    m.scatter_(-1, idx, True)
    if prefix_tokens > 0 and prefix_off >= 0:
        b0, b1 = prefix_off // 128, min(nk, -(-(prefix_off + prefix_tokens) // 128))
        if b1 > b0:
            m[..., b0:b1] = True
    return m.unsqueeze(0)


def sparse_v4_into(q4, k4, v4, out, keep: float, rec: str = "bf16"):
    """Lossy replacement of sp_model._v4_into: q4/k4/v4 [1, n|Nk, h, D] views, out [1, n, h, D] bf16."""
    from . import kernels as K
    AF, mha_v4, mha_v4_packed, SM = _v4()
    _, N, h, D = q4.shape
    Nk = k4.shape[1]
    nk = Nk // 128
    bm = block_mask(q4, k4, keep, KEY_PREFIX_OFF, SEQ_PREFIX)
    if rec != "bf16":  # sparse x low precision (recipe formats; quantizes inside mha_v4)
        from . import lowprec
        fq, fk, fv, kw = lowprec.recipe_args(rec)
        if Nk == nk * 128:
            out.copy_(mha_v4(q4.contiguous(), k4.contiguous(), v4.contiguous(), fq, fk, fv, block_mask=bm, **kw))
            return out
        o1, l1 = mha_v4(q4.contiguous(), k4[:, :nk * 128].contiguous(), v4[:, :nk * 128].contiguous(), fq, fk, fv, block_mask=bm,
                        return_lse=True, **kw)
        o2, l2 = mha_v4_packed(q4, k4[:, nk * 128:], v4[:, nk * 128:], q4, k4, v4, AF.BF16, AF.BF16, AF.BF16,
                               SM.NONE, SM.NONE, SM.NONE, return_lse=True)
        return _merge(out, o1, l1, o2, l2)
    qc = q4.contiguous()
    kc = k4[:, :nk * 128].contiguous()
    vc = v4[:, :nk * 128].contiguous()
    if Nk == nk * 128:
        o1 = mha_v4(qc, kc, vc, AF.BF16, AF.BF16, AF.BF16, block_mask=bm)
        out.copy_(o1)
        return out
    o1, l1 = mha_v4(qc, kc, vc, AF.BF16, AF.BF16, AF.BF16, block_mask=bm, return_lse=True)
    o2, l2 = mha_v4_packed(q4, k4[:, nk * 128:], v4[:, nk * 128:], q4, k4, v4, AF.BF16, AF.BF16, AF.BF16,
                           SM.NONE, SM.NONE, SM.NONE, return_lse=True)
    return _merge(out, o1, l1, o2, l2)


def _merge(out, o1, l1, o2, l2):
    from . import kernels as K
    _, N, h, D = out.shape
    dst = out[0]
    if dst.is_contiguous():
        K.lse_merge_into(dst, 0, o1.transpose(1, 2), l1, o2[0].transpose(0, 1), l2[0])
    else:
        tmp = torch.empty((N, h, D), dtype=out.dtype, device=out.device)
        K.lse_merge_into(tmp, 0, o1.transpose(1, 2), l1, o2[0].transpose(0, 1), l2[0])
        dst.copy_(tmp)
    return out


def _layer_dense() -> bool:
    """restrict sparsity to a layer range: 'layers01' [lo, hi) on steps 0-1, 'layers' [lo, hi) on the others."""
    lr = CFG.get("layers01") if STEP <= 1 else None
    lr = lr if lr is not None else CFG.get("layers")
    return lr is not None and not (lr[0] <= LAYER < lr[1])


def _q_offset(q4, k4):
    """row offset of q inside the receive buffer that k views (0 if q is not a row slice of it)."""
    if q4.stride(1) == k4.stride(1) and q4.data_ptr() != k4.data_ptr():
        return round((q4.data_ptr() - k4.data_ptr()) / (q4.stride(1) * q4.element_size()))
    return 0


def _dense_q_rows(q4, k4):
    """[a0, a1) rows of q that are global prefix rows (text + audio), or None."""
    if KEY_PREFIX_OFF < 0:
        return None
    a0 = KEY_PREFIX_OFF - _q_offset(q4, k4)
    a1 = a0 + SEQ_PREFIX
    a0, a1 = max(0, a0), min(q4.shape[1], a1)
    return (a0, a1) if a1 > a0 else None


def _v4m():
    from . import sparse2 as S2
    S2._v4p()
    from aiter.ops import mha_v4 as M
    return M


def _s2_into(q4, k4, v4, out, keep, quant=None):
    from . import sparse2 as S2
    if _layer_dense():  # exactly the default dense call of sp_model._v4_into (aiter mha_v4_packed, no LUT)
        AF, SM, mp = S2._v4p()
        mp(q4, k4, v4, q4, k4, v4, AF.BF16, AF.BF16, AF.BF16, SM.NONE, SM.NONE, SM.NONE, out=out)
        return out
    qoff = 0
    if q4.stride(1) == k4.stride(1) and q4.data_ptr() != k4.data_ptr():  # q is a row slice of the same receive buffer as k
        qoff = round((q4.data_ptr() - k4.data_ptr()) / (q4.stride(1) * q4.element_size()))
    kp = (KEY_PREFIX_OFF, SEQ_PREFIX) if KEY_PREFIX_OFF >= 0 else (0, 0)
    qd = None
    if CFG.get("dense_q") and KEY_PREFIX_OFF >= 0:
        qd = (KEY_PREFIX_OFF - qoff, SEQ_PREFIX)
        if qd[0] + qd[1] <= 0 or qd[0] >= q4.shape[1]:
            qd = None
        elif qd[0] < 0:
            qd = (0, qd[1] + qd[0])
    loc = None
    lw = CFG.get("local")  # always-kept spatio-temporal window [rows, frames] around each query tile
    if lw and VGEOM is not None and KEY_PREFIX_OFF >= 0:
        lw = per_step(lw) if isinstance(lw[0], (list, tuple)) else lw
        if lw and lw[0] >= 0:
            loc = (qoff - KEY_PREFIX_OFF, k4.shape[1], VGEOM, int(lw[0]), int(lw[1]))
    exact = bool(CFG.get("s2_prefix_bf16")) and qd is not None
    S2.sparse_into(q4, k4, v4, out, keep, kp, None if exact else qd, CFG.get("est", "mean"), STEP, bool(CFG.get("s2time")),
                   quant=quant, local=loc)
    if exact:
        # Key "s2_prefix_bf16": the text + audio query rows are recomputed with the exact bf16 kernel over all
        # keys (keys split 16 ways inside aiter, so the few prefix tiles are not a long pole), overwriting the sparse / low-precision
        # result of those rows; the sparse launch then needs no dense query tiles.
        a0, a1 = qd[0], min(q4.shape[1], qd[0] + qd[1])
        if a1 > a0:
            AF = S2._v4p()[0]
            M = _v4m()
            o = M.mha_v4(q4[:, a0:a1].contiguous(), k4.contiguous(), v4.contiguous(), AF.BF16, AF.BF16, AF.BF16, kv_splits=16)
            out[:, a0:a1].copy_(o)
    return out
