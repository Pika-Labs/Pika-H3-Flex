# SPDX-License-Identifier: Apache-2.0
"""Named lossy presets and validation of the request field ``"lossy"`` (pure Python: no torch, no GPU).

The default request (no ``lossy`` field) is the lossless path and never touches this module's output. A request opts in with

  "lossy": "near_lossless" | "balanced" | "fast"           a named preset (expands to the exact spec below)
  "lossy": {"preset": "fast", "vae_fp8": false}            a preset plus explicit overrides of single keys
  "lossy": {...raw keys...}                                the raw JSON spec (keys documented in docs/PIKA_HANDOVER.md)
  "lossy": {} | null | "lossless"                          lossless (same as no field)

Unknown preset names, unknown keys and malformed values raise ``LossyConfigError``; nothing falls back silently.
Presets are fixed dictionaries: they do not depend on the prompt, seed or duration of the request.
"""
from __future__ import annotations

import copy
import json
import re

N_STEPS = 6  # generator evaluations of the fixed six-step schedule


class LossyConfigError(ValueError):
    """Invalid ``lossy`` request field / PIKA_LOSSY value."""


_L2P = {"attn": ["bf16", "bf16", "mxfp6", "mxfp6", "mxfp6", "mxfp6"], "attn_prefix_bf16": True}
_A45 = {"gemm": ["", "", "", "", "qkv,o,up,down", "qkv,o,up,down"], "gemm_fmt": "a6w6", "gemm_prefix_bf16": True}
_A345 = {"gemm": ["", "", "", "qkv,o,up,down", "qkv,o,up,down", "qkv,o,up,down"], "gemm_fmt": "a6w6", "gemm_prefix_bf16": True}
_A2345 = {"gemm": ["", "", "qkv,o,up,down", "qkv,o,up,down", "qkv,o,up,down", "qkv,o,up,down"], "gemm_fmt": "a6w6",
          "gemm_prefix_bf16": True}
_DEC = {"vae_tile": [256, 256, 64, 32], "vae_fp8": "all"}

# Final tier specs. Internal names in brackets.
# All keep generator evaluations 0-1 exact (step 1 of "balanced": tiered attention only).
_BALANCED = {**_L2P, "tier": ["", "bf16:0.1,mxfp6:1", "", "", "", ""], **_A345, **_DEC}
# The former "fast" spec (not a preset any more; pass it as the raw "lossy" field). Audio fails the per-step check at 10 s.
F2C_A2345_D = {"impl": "s2", "dense_q": 1, "prefix": "auto", "est": "rows16k4", "sparse": [1, 1, 1, 0.35, 0.12, 0.08],
               "attn": ["bf16", "bf16", "mxfp6", "bf16", "bf16", "bf16"], "attn_prefix_bf16": True, "s2_prefix_bf16": True,
               **_A2345, **_DEC}
PRESETS: dict[str, dict] = {
    # [TQ2_a45_d] MXFP6 attention at steps 2-5 (text/audio query rows exact), step 2 tiered (top-10 % key blocks per head and
    # query tile in bf16, rest MXFP6), aiter MXFP6 (a6w6) GEMMs at steps 4-5 (text/audio rows exact), lighter VAE decode tiling
    # with FP8 tile GEMMs.
    "near_lossless": {**_L2P, "tier": ["", "", "bf16:0.1,mxfp6:1", "", "", ""], **_A45, **_DEC},
    # [T1_a345_d] MXFP6 attention at steps 2-5, tiered attention already at step 1, a6w6 GEMMs at steps 3-5 (late35 tier).
    "balanced": _BALANCED,
    # "fast" is an ALIAS of "balanced": at world 8 no spec was both faster than
    # "balanced" and inside the audio gate. The former fast spec F2c_a2345_d (s2 block-sparse bf16 attention at steps 3-5)
    # was 1-3 % faster but failed the per-step audio check at 10 s by up to 4.9 dB; the audio-safe sparse variants were 2-7 %
    # SLOWER than "balanced". F2c_a2345_d stays available as a raw spec: F2C_A2345_D above / docs/PIKA_HANDOVER.md.
    "fast": copy.deepcopy(_BALANCED),
}
PRESET_INTERNAL_NAMES = {"near_lossless": "TQ2_a45_d", "balanced": "T1_a345_d", "fast": "T1_a345_d (alias of balanced)"}
LOSSLESS_NAMES = ("lossless", "off", "none", "")

ATTN_RECIPES = ("bf16", "fp8", "mxfp8", "int8", "bf16fp8", "mxfp6", "mxfp6fp8", "fp8mxfp6", "mxfp4")
GEMM_OPS = ("qkv", "o", "up", "down")
GEMM_FMTS = ("fp8", "a6w6")
TIER_FMTS = ("bf16", "mxfp8", "mxfp6", "mxfp4")
_EST_RE = re.compile(r"^(mean|sub|rows\d*|rows\d+k\d+)$")


# ------------------------------------------------------------------------------------------------------------- validators
def _err(key, msg):
    raise LossyConfigError(f"lossy key {key!r}: {msg}")


def _per_step(key, val, check):
    """value for all steps, or a list of 1..6 per-step values (the last one repeats)."""
    if isinstance(val, (list, tuple)):
        if not 1 <= len(val) <= N_STEPS:
            _err(key, f"per-step list must have 1..{N_STEPS} entries, got {len(val)}")
        for v in val:
            check(key, v)
    else:
        check(key, val)


def _attn(key, v):
    if v not in ATTN_RECIPES:
        _err(key, f"unknown attention recipe {v!r}; expected one of {', '.join(ATTN_RECIPES)}")


def _gemm(key, v):
    if v in ("", None):
        return
    ops = v.split(",") if isinstance(v, str) else v
    if not isinstance(ops, (list, tuple)) or not ops or any(o not in GEMM_OPS for o in ops):
        _err(key, f"expected '' or a comma list of {', '.join(GEMM_OPS)}, got {v!r}")


def _gemm_fmt(key, v):
    if v not in GEMM_FMTS:
        _err(key, f"expected one of {', '.join(GEMM_FMTS)}, got {v!r}")


def _tier(key, v):
    if v in ("", None):
        return
    if not isinstance(v, str):
        _err(key, f"tier spec must be a string, got {v!r}")
    n = 0
    for tok in v.split(","):
        tok = tok.strip()
        if not tok or tok == "comp":
            continue
        if tok.startswith("est="):
            _est(key, tok[4:])
        elif tok.startswith("layers="):
            if not re.fullmatch(r"\d+-\d+", tok[7:]):
                _err(key, f"bad layer range {tok!r} (expected layers=lo-hi)")
        elif re.fullmatch(r"min\d+", tok) or re.fullmatch(r"max(\d*\.)?\d+", tok):
            pass
        else:
            m = re.fullmatch(r"([a-z0-9]+):(m?)((\d*\.)?\d+)", tok)
            if not m or re.match(r"(bf16|mxfp[468])", m.group(1)) is None:
                _err(key, f"bad tier {tok!r} in {v!r}; expected <fmt>:<fraction> or <fmt>:m<mass>, fmt in {', '.join(TIER_FMTS)}")
            if not 0.0 < float(m.group(3)) <= 1.0:
                _err(key, f"tier fraction must be in (0, 1], got {tok!r}")
            n += 1
    if n == 0:
        _err(key, f"tier spec {v!r} has no <fmt>:<fraction> tier")


def _est(key, v):
    if not isinstance(v, str) or not _EST_RE.match(v):
        _err(key, f"expected mean | sub | rows<r> | rows<r>k<s> (e.g. rows16k4), got {v!r}")


def _sparse(key, v):
    try:
        vals = [float(x) for x in v.split(",") if x] if isinstance(v, str) else v
    except ValueError:
        _err(key, f"expected a list of keep fractions, got {v!r}")
    if not isinstance(vals, (list, tuple)) or not 1 <= len(vals) <= N_STEPS:
        _err(key, f"expected a list of 1..{N_STEPS} keep fractions, got {v!r}")
    for x in vals:
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not 0.0 < float(x) <= 1.0:
            _err(key, f"keep fractions must be numbers in (0, 1], got {x!r}")


def _bool(key, v):
    if v not in (True, False, 0, 1):
        _err(key, f"expected true/false (or 0/1), got {v!r}")


def _prefix(key, v):
    if v != "auto" and (not isinstance(v, int) or isinstance(v, bool) or v < 0):
        _err(key, f"expected 'auto' or a non-negative integer, got {v!r}")


def _impl(key, v):
    if v != "s2":
        _err(key, f"the only alternative sparse implementation is 's2', got {v!r}")


def _vae_tile(key, v):
    if (not isinstance(v, (list, tuple)) or len(v) != 4 or
            any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 or x % 16 for x in v)):
        _err(key, f"expected [tile_h, tile_w, min_overlap_h, min_overlap_w], positive multiples of 16 px, got {v!r}")
    if v[2] >= v[0] or v[3] >= v[1]:
        _err(key, f"overlap must be smaller than the tile, got {v!r}")


def _vae_fp8(key, v):
    if v in (True, False, 0, 1, "all", None):
        return
    ops = str(v).split(",")
    if any(o not in GEMM_OPS for o in ops):
        _err(key, f"expected true/false, 'all' or a comma list of {', '.join(GEMM_OPS)}, got {v!r}")


def _any(key, v):
    pass


# Supported keys (used by the presets; documented with their trade-offs in docs/PIKA_HANDOVER.md).
SUPPORTED_KEYS = {
    "attn": lambda k, v: _per_step(k, v, _attn),
    "attn_prefix_bf16": _bool,
    "tier": lambda k, v: _per_step(k, v, _tier),
    "gemm": lambda k, v: _per_step(k, v, _gemm),
    "gemm_fmt": lambda k, v: _per_step(k, v, _gemm_fmt),
    "gemm_prefix_bf16": _bool,
    "impl": _impl,
    "dense_q": _bool,
    "s2_prefix_bf16": _bool,
    "prefix": _prefix,
    "est": _est,
    "sparse": _sparse,
    "vae_tile": _vae_tile,
    "vae_fp8": _vae_fp8,
}
# Experimental / developer keys: accepted (used by research configs), values not checked, not supported.
EXPERIMENTAL_KEYS = ("attn_layers", "comm_fp8", "gemm_layers", "gemm_map", "headstats", "layers", "layers01", "local",
                     "pfx_stream", "s2split", "s2time", "s2wave", "tiertime", "vae_clips")


# ------------------------------------------------------------------------------------------------------------------ API
def expand(cfg):
    """``lossy`` request value (preset name | dict | JSON string | None) -> validated raw spec dict ({} = lossless).
    Raises LossyConfigError on any unknown preset, key or value."""
    if cfg is None:
        return {}
    if isinstance(cfg, str):
        s = cfg.strip()
        if s.startswith("{"):
            try:
                cfg = json.loads(s)
            except json.JSONDecodeError as e:
                raise LossyConfigError(f"lossy: invalid JSON: {e}") from None
        else:
            cfg = {"preset": s}
    if not isinstance(cfg, dict):
        raise LossyConfigError(f"lossy: expected a preset name, an object or null, got {type(cfg).__name__}")
    cfg = dict(cfg)
    out = {}
    if "preset" in cfg:
        name = cfg.pop("preset")
        if not isinstance(name, str):
            raise LossyConfigError(f"lossy preset must be a string, got {name!r}")
        if name.strip().lower() in LOSSLESS_NAMES:
            if cfg:
                raise LossyConfigError(f"lossy preset {name!r} (lossless) cannot be combined with keys {sorted(cfg)}")
            return {}
        if name not in PRESETS:
            raise LossyConfigError(f"unknown lossy preset {name!r}; expected one of {', '.join(PRESETS)} (or 'lossless')")
        out = copy.deepcopy(PRESETS[name])
    for k, v in cfg.items():
        if k in SUPPORTED_KEYS:
            if v is None:  # explicit null removes a preset key
                out.pop(k, None)
                continue
            SUPPORTED_KEYS[k](k, v)
        elif k not in EXPERIMENTAL_KEYS:
            raise LossyConfigError(f"unknown lossy key {k!r}; supported: {', '.join(SUPPORTED_KEYS)} "
                                   f"(experimental: {', '.join(EXPERIMENTAL_KEYS)})")
        out[k] = copy.deepcopy(v)
    return out


def describe() -> str:
    return json.dumps({k: {"internal_name": PRESET_INTERNAL_NAMES[k], "spec": v} for k, v in PRESETS.items()}, indent=1)


if __name__ == "__main__":  # python -m pika_h3.lossy_presets : print the presets
    print(describe())
