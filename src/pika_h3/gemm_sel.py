# SPDX-License-Identifier: Apache-2.0
"""Per-shape hipBLASLt solution selection for the DiT GEMMs.

PIKA_GEMM_SOLS=<json> maps "MxNxK" (x [M,K] @ W[N,K]^T) to a hipBLASLt solution index found by a *sustained* sweep at the
power cap (sustained rotated-weight benchmark); those shapes run through aiter's hipb_mm (same hipBLASLt library, bf16 in / fp32
accumulate / bf16 out: a different tile choice of the same GEMM).  Every other shape, and everything when the flag is unset,
uses F.linear (the default heuristic).  Off by default."""
import json
import os
import sys

import torch
import torch.nn.functional as F

_TABLE = None
_mm = None
_BYNK = {}
_CACHE = {}
# a solution tuned at M is used for M' within TOL * M (the row count depends on the prompt's text length: no exact-M keys)
TOL = float(os.environ.get("PIKA_GEMM_SOLS_TOL", "0.04"))


def _init():
    global _TABLE, _mm
    if _TABLE is not None:
        return
    path = os.environ.get("PIKA_GEMM_SOLS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gemm_sols.json") if os.environ.get("PIKA_GEMM_ENG", "1") == "1" else "")
    _TABLE = {}
    if not path or path == "off":
        return
    explicit = "PIKA_GEMM_SOLS" in os.environ
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        if explicit:
            raise FileNotFoundError(f"PIKA_GEMM_SOLS={os.environ['PIKA_GEMM_SOLS']!r}: tuned GEMM table not found at {path}")
        # the package's table (pika_h3/data/gemm_sols.json) is missing, e.g. an install without the package data: the
        # default hipBLASLt heuristic is used (legitimate numerics, a few % slower); say so loudly instead of failing
        print(f"[gemm_sel] WARNING: tuned GEMM table {os.path.normpath(path)} not found; using the default hipBLASLt "
              "heuristic (set PIKA_GEMM_SOLS=<path> to the table, or PIKA_GEMM_SOLS=off to silence this)", file=sys.stderr,
              flush=True)
        return
    tab = json.load(open(path))
    from .paths import ensure_aiter
    ensure_aiter()
    from aiter.ops.gradlib import hipb_create_extension, hipb_mm
    hipb_create_extension()
    _mm = hipb_mm
    for k, v in tab.items():
        if k.startswith("_"):
            continue
        m, n, kk = (int(x) for x in k.split("x"))
        sol = int(v["sol"]) if isinstance(v, dict) else int(v)
        _TABLE[(m, n, kk)] = sol
        _BYNK.setdefault((n, kk), []).append((m, sol))


def lookup(m, n, k):
    key = (m, n, k)
    if key not in _CACHE:
        best = None
        for mt, sol in _BYNK.get((n, k), []):
            if abs(mt - m) <= TOL * mt and (best is None or abs(mt - m) < abs(best[0] - m)):
                best = (mt, sol)
        _CACHE[key] = None if best is None or best[1] == 0 else best[1]   # sol 0 = measured: default heuristic
    return _CACHE[key]


def linear(x, w, bias=None):
    """F.linear(x, w, bias) with the tuned hipBLASLt solution when the shape is in the table."""
    if _TABLE is None:
        _init()
    if _TABLE is not None and _BYNK:
        sol = lookup(x.shape[0], w.shape[0], w.shape[1])
        if sol is not None and x.dim() == 2 and x.stride(1) == 1:
            try:
                return _mm(x, w.t(), sol, bias=bias)
            except Exception:  # solution not applicable to this M: default heuristic from now on
                _CACHE[(x.shape[0], w.shape[0], w.shape[1])] = None
    return F.linear(x, w, bias)


def enabled():
    if _TABLE is None:
        _init()
    return bool(_TABLE)
