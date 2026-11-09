# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel (Ulysses) MiniMax-H3 transformer for one packed document.

Every rank holds the full bf16 weights (LoRA merged, q/k/v fused). The packed sequence [text | audio | video] is split
into contiguous row ranges, one per rank; attention is computed per head group after an all-to-all (rank g owns heads
7g..7g+6), everything else is row-local. The outputs are all-gathered so every rank holds the identical full result.
"""
from __future__ import annotations

import gc
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import kernels as K
from . import gemm_sel as G
from . import lossy as LS  # LOSSY track: inert unless PIKA_LOSSY / request 'lossy' is set

from .paths import AITER_DIR  # PIKA_AITER_DIR, an installed aiter, or <repo>/third_party/aiter (paths.py)
_mha_v4 = None
_AF = None


def _load_v4():
    global _mha_v4, _AF
    if _mha_v4 is None:
        import sys
        if AITER_DIR not in sys.path:
            sys.path.insert(0, AITER_DIR)
        from aiter.ops.mha_v4 import AttentionFormat, mha_v4
        _mha_v4, _AF = mha_v4, AttentionFormat
    return _mha_v4, _AF


if os.environ.get("PIKA_FILLNAN") == "1":  # dev: fill torch.empty with NaN to expose reads of uninitialised memory
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.use_deterministic_algorithms(True, warn_only=True)
TAIL_SPLIT = os.environ.get("PIKA_TAIL_SPLIT", "1") == "1"
_CUS = int(os.environ.get("PIKA_NUM_CUS", "256"))


def _flash_lse(q, k, v):
    r = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False)
    return r[0], r[1]


def attention_tail_split(q, k, v, out_view, chunks=8, tile=256):
    """Exact softmax attention q,k,v: [1, H, N, D] (BHSD views) written into out_view [1, H, N, D].

    The flash kernel runs one block per CU per wave: H * ceil(N/tile) blocks over the CUs. When the last wave is almost
    empty (e.g. 7 heads x 149 tiles = 1043 blocks on 256 CUs = 4 waves + 19 blocks) the last query tile rows are
    computed separately with the keys split into ``chunks`` ranges (flash-decoding style, LSE merge in fp32)."""
    B, H, N, D = q.shape
    blocks = H * ((N + tile - 1) // tile)
    waves, rem = divmod(blocks, _CUS)
    if B != 1 or waves == 0 or rem == 0 or rem > 96:
        out_view.copy_(F.scaled_dot_product_attention(q, k, v))
        return
    Nm = ((waves * _CUS) // H) * tile
    out_view[:, :, :Nm].copy_(F.scaled_dot_product_attention(q[:, :, :Nm], k, v))
    qt = q[:, :, Nm:]
    Nt = N - Nm
    cs = N // chunks
    r = N - chunks * cs
    kc = k[:, :, :chunks * cs].unflatten(2, (chunks, cs)).movedim(2, 0)[:, 0]
    vc = v[:, :, :chunks * cs].unflatten(2, (chunks, cs)).movedim(2, 0)[:, 0]
    o, lse = _flash_lse(qt.expand(chunks, H, Nt, D), kc, vc)
    o = o.float()
    lse = lse.float()
    if r:
        o2, l2 = _flash_lse(qt, k[:, :, chunks * cs:], v[:, :, chunks * cs:])
        o = torch.cat([o, o2.float()], 0)
        lse = torch.cat([lse, l2.float()], 0)
    m = lse.max(0, keepdim=True).values
    w = torch.exp(lse - m)
    merged = (o * w.unsqueeze(-1)).sum(0) / w.sum(0).unsqueeze(-1)
    out_view[:, :, Nm:].copy_(merged.to(q.dtype).unsqueeze(0))


_NCU = None


def _num_cus():
    global _NCU
    if _NCU is None:
        _NCU = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _NCU


def tail_plan(N, h, tile=256, chunks=8, max_rem=96):
    """Wave-quantisation plan of the flash kernel (one 256-row query tile per CU per wave). Returns Nm (query rows handled by
    the main launch) or None when the plain launch is already efficient. The remaining tail rows are computed with the keys
    split in ``chunks`` ranges (all CUs busy) and merged exactly (LSE merge, fp32)."""
    cus = _num_cus()
    blocks = h * ((N + tile - 1) // tile)
    waves, rem = divmod(blocks, cus)
    if waves == 0 or rem == 0 or rem > max_rem:
        return None
    return ((waves * cus) // h) * tile


# PIKA_TAIL_FP32=1: tail rows via the fp32-partial Triton split-KV kernel (pika_h3.attn_tail) instead of bf16 chunk outputs.
# Default off (experimental candidate).
TAIL_FP32 = os.environ.get("PIKA_TAIL_FP32", "0") == "1"
# v4: the tail split only pays for launches of few waves (measured: W8 5 s 4.05 waves 5.17 -> 4.79 ms; W4 5 s 8.1 waves and
# W4 15 s 23.4 waves no gain: a nearly empty last wave runs at a higher clock under the power cap).
# Default 1000 = tail split whenever tail_plan says so (the evaluated default); candidate: 6.
V4_TAIL_MAX_WAVES = int(os.environ.get("PIKA_V4_TAIL_MAX_WAVES", "6"))  # default 6 (4-GPU eval 6.347, gate pass)


def _tail_fp32_into(o, Nm, qt, k, v):
    """Tail rows [Nm, N) of o [N, h, D]: Triton split-KV kernel with fp32 partials (pika_h3.attn_tail) + fp32 LSE merge.
    qt [1, h, Nt, D], k/v [1, h, Nk, D] BHSD views."""
    from .attn_tail import split_kv_attention_fp32
    Nt, h = qt.shape[2], qt.shape[1]
    progs = -(-Nt // 64) * h
    chunks = max(1, min(64, (4 * _num_cus()) // progs))
    o1, l1 = split_kv_attention_fp32(qt[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1), chunks)
    K.lse_merge_into(o, Nm, o1, l1)
    return o


def attention_tail_into(q, k, v, Nm, chunks=8):
    """Exact softmax attention. q,k,v: [1, h, N, D] views (transposed [N, h, D] storage). Returns o [N, h, D] contiguous."""
    _, h, N, D = q.shape
    o = torch.empty((N, h, D), dtype=q.dtype, device=q.device)
    o[:Nm].copy_(F.scaled_dot_product_attention(q[:, :, :Nm], k, v)[0].transpose(0, 1))
    qt = q[:, :, Nm:]
    if TAIL_FP32:
        return _tail_fp32_into(o, Nm, qt, k, v)
    Nt = N - Nm
    Nk = k.shape[2]  # key length (== N except for the PIKA_SPLIT micro-batch m1)
    cs = Nk // chunks
    kc = k[:, :, :chunks * cs].unflatten(2, (chunks, cs)).movedim(2, 0)[:, 0]
    vc = v[:, :, :chunks * cs].unflatten(2, (chunks, cs)).movedim(2, 0)[:, 0]
    o1, l1 = _flash_lse(qt.expand(chunks, h, Nt, D), kc, vc)
    if Nk > chunks * cs:
        o2, l2 = _flash_lse(qt, k[:, :, chunks * cs:], v[:, :, chunks * cs:])
        K.lse_merge_into(o, Nm, o1, l1, o2[0], l2[0])
    else:
        K.lse_merge_into(o, Nm, o1, l1)
    return o


# Exact bf16 attention kernel for the lossless path: 'sdpa' (torch SDPA -> aiter fmha_v3 asm, the default) or 'v4'
# (aiter mha_v4 BF16 asm kernel: same 256x64 tile, bf16 Q/K/V, fp32 softmax/accumulation; ~8% fewer cycles per tile).
ATTN_KERNEL = os.environ.get("PIKA_ATTN_KERNEL", "v4")  # default v4 (4-GPU 5.904 -> 6.313, gate pass)
_v4p = None
# kernel behind the v4 code paths: 'aiter' (mha_v4, default) | 'fa18' (line A's asm kernel, src/pika_h3/attn_fa18.py: same
# speed, exact running max -> above-v3 precision). Only _v4_into / _attn_lse are affected.
V4_IMPL = os.environ.get("PIKA_V4_IMPL", "aiter")


def _v4_into(q, k, v, out):
    """q,k,v: [1, n, h, D] views (last dim contiguous, any seq/head strides); out [1, n, h, D] bf16. No copies."""
    global _v4p
    if LS.ACTIVE:  # LOSSY (off by default): block-sparse and/or low-precision attention on this generator evaluation
        r = LS.attn_into(q, k, v, out)
        if r is not None:
            return r
    if V4_IMPL == "fa18":
        from .attn_fa18 import fa18_into
        return fa18_into(q, k, v, out)
    if _v4p is None:
        import sys
        if AITER_DIR not in sys.path:
            sys.path.insert(0, AITER_DIR)
        from aiter.ops.mha_v4 import AttentionFormat as AF, AttentionScaleMode as SM, mha_v4_packed
        _v4p = lambda q_, k_, v_, o_: mha_v4_packed(q_, k_, v_, q_, k_, v_, AF.BF16, AF.BF16, AF.BF16,
                                                   SM.NONE, SM.NONE, SM.NONE, out=o_)
    _v4p(q, k, v, out)
    return out


_v4lse = None


def _attn_lse(q, k, v):
    """Like _flash_lse (q,k,v: [1, h, n, D] BHSD views) -> (o [1, h, n, D] view, lse fp32 [1, h, n], natural log).
    PIKA_ATTN_KERNEL=v4: aiter mha_v4 BF16 with LSE export (strided views, no copies)."""
    if ATTN_KERNEL != "v4":
        return _flash_lse(q, k, v)
    if V4_IMPL == "fa18":
        from .attn_fa18 import fa18_lse
        return fa18_lse(q, k, v)
    global _v4lse
    if _v4lse is None:
        import sys
        if AITER_DIR not in sys.path:
            sys.path.insert(0, AITER_DIR)
        from aiter.ops.mha_v4 import AttentionFormat as AF, AttentionScaleMode as SM, mha_v4_packed
        _v4lse = lambda q_, k_, v_: mha_v4_packed(q_, k_, v_, q_, k_, v_, AF.BF16, AF.BF16, AF.BF16,
                                                  SM.NONE, SM.NONE, SM.NONE, return_lse=True)
    o, lse = _v4lse(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return o.transpose(1, 2), lse


def v4_tail_plan(N, h):
    Nm = tail_plan(N, h) if TAIL_SPLIT else None
    if Nm is not None and h * -(-N // 256) > V4_TAIL_MAX_WAVES * _num_cus():
        return None
    return Nm


def attention_v4_into(q4, k4, v4, Nm=None, chunks=8, out=None):
    """Exact softmax attention with the v4 kernel. q4,k4,v4: [1, N, h, D] views. Returns o [N, h, D] contiguous (= out if given).
    Nm (from tail_plan): main launch on rows [:Nm], tail rows with keys split in chunks + fp32 LSE merge (as attention_tail_into)."""
    _, N, h, D = q4.shape
    o = torch.empty((N, h, D), dtype=q4.dtype, device=q4.device) if out is None else out
    if Nm is not None and LS.ACTIVE and LS.attn_recipe() != "bf16" and LS.sparse_now() >= 1.0:
        Nm = None  # LOSSY low-precision dense step: the bf16 tail launch costs more than the MX kernel's partial wave
    if Nm is not None and LS.ACTIVE and LS.get("s2wave", 1) == 0 and LS.sparse_now() < 1.0:
        Nm = None  # LOSSY sparse step (opt-in): the sorted-sparse kernel balances its own waves; no dense tail rows
    if Nm is None:
        _v4_into(q4, k4, v4, o.unsqueeze(0))
        return o
    _v4_into(q4[:, :Nm], k4, v4, o[:Nm].unsqueeze(0))
    return _tail_rows_into(o, Nm, q4, k4, v4, h, chunks)


def _tail_rows_into(o, Nm, q4, k4, v4, h, chunks=8):
    """Rows [Nm, N) of o [N, h, D]: keys split in ``chunks`` ranges (flash-decoding style) + fp32 LSE merge."""
    N, D = q4.shape[1], q4.shape[-1]
    qq, kk, vv = (x.transpose(1, 2) for x in (q4, k4, v4))
    qt = qq[:, :, Nm:]
    if TAIL_FP32:
        return _tail_fp32_into(o, Nm, qt, kk, vv)
    Nt = N - Nm
    Nk = kk.shape[2]  # key length (!= N for the PIKA_SPLIT micro-batch m1)
    cs = Nk // chunks
    kc = kk[:, :, :chunks * cs].unflatten(2, (chunks, cs)).movedim(2, 0)[:, 0]
    vc = vv[:, :, :chunks * cs].unflatten(2, (chunks, cs)).movedim(2, 0)[:, 0]
    o1, l1 = _flash_lse(qt.expand(chunks, h, Nt, D), kc, vc)
    if Nk > chunks * cs:
        o2, l2 = _flash_lse(qt, kk[:, :, chunks * cs:], vv[:, :, chunks * cs:])
        K.lse_merge_into(o, Nm, o1, l1, o2[0], l2[0])
    else:
        K.lse_merge_into(o, Nm, o1, l1)
    return o


HEAD_ROT = int(os.environ.get("PIKA_HEAD_ROT", "0"))
SDMA_SELFLAST = os.environ.get("PIKA_SDMA_SELFLAST", "1") == "1"
SDMA_MIN_N = int(os.environ.get("PIKA_SDMA_MIN_N", "50000"))
SDMA_FA = float(os.environ.get("PIKA_SDMA_FA", "0.6"))  # unsplit SDMA path: fraction of rows produced and sent first
# DEV ONLY (wrong output, timing bound): ranks in DEV_SKIP_RANKS skip the last DEV_SKIP fraction of the query tiles of their main
# attention launch (rows zeroed). Upper bound of a sub-head attention rebalance away from the slow GPUs. Set per request.
DEV_SKIP = 0.0
DEV_SKIP_RANKS = set()
DEV_NOXFER = 0  # DEV ONLY (wrong output): 1 = serial SDMA path skips the qkv transfer, 2 = skips the return, 3 = both


def _dev_skip_rows(rank, n):
    if DEV_SKIP <= 0 or rank not in DEV_SKIP_RANKS:
        return n
    return max(256, n - int(round(DEV_SKIP * n / 256)) * 256)


# attention recipes per generator evaluation (index 0..5): 'sdpa' | 'bf16' | 'bf16fp8' | 'fp8' | 'int8fp8'
def attn_recipe(step):
    spec = os.environ.get("PIKA_ATTN", "sdpa")
    parts = spec.split(",")
    return parts[step] if len(parts) > step else parts[-1]

import os
LORA_MODE = os.environ.get('PIKA_LORA_MODE', 'exact')
H = 5376
NH = 56
HD = 128
INNER = NH * HD
ROPE_DIM = 96  # rotary dims per head (cos.shape[1]); asserted in forward when PIKA_ROPE_PAIR is on
FFN = 14336


def merge_lora_and_unwrap(model: nn.Module) -> None:
    """Merge every LoRA delta into its base weight (fp32 sum, one bf16 rounding) and drop the adapter layers."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, BaseTunerLayer)]
    for name, m in targets:
        base = m.base_layer
        with torch.no_grad():
            for adapter in list(m.lora_A.keys()):
                A = m.lora_A[adapter].weight.float()
                B = m.lora_B[adapter].weight.float()
                scale = float(m.scaling[adapter])
                w = base.weight.float() + (B @ A) * scale
                base.weight.copy_(w.to(base.weight.dtype))
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if child.isdigit():
            parent[int(child)] = base
        else:
            setattr(parent, child, base)


class _Block:
    """Fused weights of one transformer block. Each linear is (A [r_tot, K] or None, Wcat [out, K + r_tot])."""
    __slots__ = ("qkv", "o", "up", "down", "ada_w", "ada_b", "n1", "n2", "qn", "kn")


def _lora_parts(lin):
    """(W, A, B) of a PEFT-wrapped (or plain) Linear; the adapter scale must be exactly 1."""
    if hasattr(lin, "base_layer"):
        assert float(lin.scaling["default"]) == 1.0
        return lin.base_layer.weight, lin.lora_A["default"].weight, lin.lora_B["default"].weight
    return lin.weight, None, None


def fuse_linears(lins):
    """Concatenate output-wise: returns (A [P*r, K], W [N, K], B [N, r]) of the stacked linears."""
    parts = [_lora_parts(l) for l in lins]
    W = torch.cat([p[0] for p in parts], dim=0).contiguous()
    A = torch.cat([p[1] for p in parts], dim=0).contiguous()
    B = torch.cat([p[2] for p in parts], dim=0).contiguous()
    return A, W, B


_LC = os.environ.get("PIKA_LORA_CAT", "0")
LORA_CAT = {"qkv", "o", "up", "down"} if _LC == "1" else (set() if _LC in ("", "0") else set(_LC.split(",")))  # PIKA_LORA_CAT=1 or =qkv,up: LoRA-A rows appended to W


def cat_lora(lin):
    """(A, W, B) -> (A, W, B, Wc) with Wc = [W; A] and W, A views into it (no extra memory)."""
    A, W, B = lin
    Wc = torch.cat([W, A], dim=0).contiguous()
    n = W.shape[0]
    return (Wc[n:], Wc[:n], B, Wc)


def _pair_permute(qkv, qn, kn, nh=NH, hd=HD):
    """PIKA_ROPE_PAIR: permute the q / k head dims of the fused qkv linear (W and LoRA-B rows) and of the q / k norm weights
    into the pair-interleaved order of kernels.qk_rope_pack_lora_pair (exact: a row permutation of the weights; the
    attention scores q.k are invariant to a common permutation of the q and k head dims)."""
    A, W, B = qkv
    perm = torch.tensor(K.pair_perm(hd, ROPE_DIM), device=W.device)
    colp = torch.arange(3 * nh * hd, device=W.device).view(3, nh, hd).clone()
    colp[:2] = colp[:2][:, :, perm]
    colp = colp.view(-1)
    W.copy_(W.index_select(0, colp))
    B.copy_(B.index_select(0, colp))
    return (A, W, B), qn[perm].contiguous(), kn[perm].contiguous()


def build_blocks(model: nn.Module):
    blocks = []
    for blk in model.transformer_blocks:
        b = _Block()
        a = blk.attn
        b.qkv = fuse_linears([a.to_q, a.to_k, a.to_v])
        b.o = fuse_linears([a.to_out[0]])
        b.up = fuse_linears([blk.ff.net[0].proj])
        b.down = fuse_linears([blk.ff.net[2]])
        a.to_q = a.to_k = a.to_v = None
        a.to_out[0] = None
        blk.ff.net[0].proj = None
        blk.ff.net[2] = None
        b.ada_w = b.ada_b = None  # read on demand from the checkpoint (fixed-schedule tables are cached)
        b.n1, b.n2 = blk.norm1.weight, blk.norm2.weight
        b.qn, b.kn = a.norm_q.weight, a.norm_k.weight
        if K.ROPE_PAIR:
            b.qkv, b.qn, b.kn = _pair_permute(b.qkv, b.qn, b.kn)
        if LORA_CAT:  # after the ROPE_PAIR permutation (it permutes W rows only)
            b.qkv, b.o, b.up, b.down = (cat_lora(l) if n in LORA_CAT else l for n, l in zip(("qkv", "o", "up", "down"), (b.qkv, b.o, b.up, b.down)))
        blocks.append(b)
        if len(blocks) % 5 == 0:
            gc.collect()  # PEFT modules form reference cycles: release the unfused weights promptly
            torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    return blocks


FP8_EMU = set(filter(None, os.environ.get("PIKA_FP8_EMU", "").split(",")))
_wq_cache = {}


def _fq_rows(x):
    """per-row dynamic fp8 e4m3 quantize/dequantize round trip (emulation)."""
    xf = x.float()
    sc = xf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 448.0
    return ((xf / sc).to(torch.float8_e4m3fn).float() * sc).to(x.dtype)


def _fq_weight(W):
    key = id(W)
    if key not in _wq_cache:
        wf = W.float()
        sc = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 448.0
        _wq_cache[key] = ((wf / sc).to(torch.float8_e4m3fn).float() * sc).to(W.dtype)
    return _wq_cache[key]


def _lin_gemm(x, lin):
    """(bf16(x W^T), bf16(x A^T)): two GEMMs, or one GEMM over [W; A] (PIKA_LORA_CAT) returning column views."""
    if len(lin) == 4:
        yc = G.linear(x, lin[3])
        n = lin[1].shape[0]
        return yc[:, :n], yc[:, n:]
    if LS.ACTIVE:  # LOSSY: FP8 base GEMM (lowprec.py, key 'gemm'); LoRA-A stays bf16
        from . import lowprec as LP
        y = LP.gemm(x, lin[1])
        if y is not None:
            return y, G.linear(x, lin[0])
    return G.linear(x, lin[1]), G.linear(x, lin[0])


def lora_linear(x, lin, part_n=None, name="", s=None):
    """y = bf16(x W^T) + bf16(bf16(x A^T) B^T) with the reference rounding points (bf16 add fused into the B GEMM)."""
    if len(lin) == 4:
        y, t = _lin_gemm(x, lin)
        return K.lora_add_(y, t, lin[2], part_n=part_n or y.shape[1], M=s)
    A, W, B = lin
    if name in FP8_EMU:
        y = F.linear(_fq_rows(x), _fq_weight(W))
        t = F.linear(x, A)
        return K.lora_add_(y, t, B, part_n=part_n or y.shape[1], M=s)
    y = None
    if LS.ACTIVE:  # LOSSY: FP8 base GEMM (lowprec.py)
        from . import lowprec as LP
        y = LP.gemm(x, W)
    y = G.linear(x, W) if y is None else y
    t = G.linear(x, A)
    return K.lora_add_(y, t, B, part_n=part_n or y.shape[1], M=s)


_GE = os.environ.get("PIKA_GEMM_ENG", "1")  # umbrella flag of the GEMM features, default ON since 4-GPU eval 6.382
FUSE_LORA = os.environ.get("PIKA_FUSE_LORA", _GE) == "1"   # LoRA-B add folded into qk_rope_pack / swiglu (kernels.*_lora)
FUSE_LORA2 = os.environ.get("PIKA_FUSE_LORA2", "0") == "1"
HIP_SWIGLU = os.environ.get("PIKA_HIP_SWIGLU", "0") == "1"  # hand-written HIP swiglu_lora (pika_h3/hip/swiglu_lora.hip, v3)  # LoRA-B add of o / down folded into the next mod_norm (kernels.mod_norm_lora)


def lora_parts_(x, lin):
    """(bf16(x W^T), bf16(x A^T), B): the LoRA linear without its final add (done by a fused consumer kernel)."""
    y, t = _lin_gemm(x, lin)
    return y, t, lin[2]


GEMM_PAD = int(os.environ.get("PIKA_GEMM_PAD", "512"))
GEMM_PAD_MAX = int(os.environ.get("PIKA_GEMM_PAD_MAX", "6144"))
_TUNED_MS = set()


def load_gemm_table(path=None):
    """Enable PyTorch TunableOp in replay mode with the offline-tuned algorithm table (pika_h3/data/gemm_tunableop.csv): the
    per-rank GEMMs are padded to multiples of GEMM_PAD rows and the table has one entry per padded row count."""
    import csv
    path = path or os.environ.get("PIKA_GEMM_TABLE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gemm_tunableop.csv"))
    # measured sustained: the TunableOp table gives no gain over padding alone, so it is opt-in
    if "PIKA_GEMM_TABLE" not in os.environ or os.environ["PIKA_GEMM_TABLE"] == "off" or not os.path.exists(path):
        return
    torch.cuda.tunable.enable(True)
    torch.cuda.tunable.tuning_enable(False)
    torch.cuda.tunable.set_filename(path, insert_device_ordinal=False)
    torch.cuda.tunable.read_file(path)
    for row in csv.reader(open(path)):
        if len(row) > 1 and row[0].startswith("GemmTunableOp") and row[1].startswith("tn_"):
            _TUNED_MS.add(int(row[1].split("_")[2]))


def gemm_rows(s: int) -> int:
    """Rows of the GEMM operands: s padded to a multiple of GEMM_PAD when s <= GEMM_PAD_MAX (5 s: 4750 -> 5120, GEMM block time
    -4.6% sustained; at 10/15 s (9176/13636 rows) the default heuristic is already at 1.25-1.37 PF and padding gains nothing)."""
    if GEMM_PAD <= 1:
        return s
    mp = (s + GEMM_PAD - 1) // GEMM_PAD * GEMM_PAD
    return mp if (s <= GEMM_PAD_MAX or mp in _TUNED_MS) else s


def split_sizes(n: int, world: int):
    base, rem = divmod(n, world)
    return [base + (1 if r < rem else 0) for r in range(world)]


def split_sizes_weighted(n: int, weights):
    """Contiguous row counts proportional to ``weights`` (largest remainder), summing to n."""
    tot = float(sum(weights))
    ideal = [n * w / tot for w in weights]
    sizes = [int(x) for x in ideal]
    order = sorted(range(len(weights)), key=lambda i: -(ideal[i] - sizes[i]))
    for i in order[: n - sum(sizes)]:
        sizes[i] += 1
    return sizes


def compute_partition(speeds, nh=NH, tol=1.3):
    """(head counts per rank, row weights per rank). Uniform when the speeds are within ``tol`` of each other."""
    w = len(speeds)
    if max(speeds) / max(min(speeds), 1e-9) < tol and nh % w == 0:
        return [nh // w] * w, [1.0] * w
    tot = sum(speeds)
    ideal = [nh * x / tot for x in speeds]
    hc = [max(1, int(x)) for x in ideal]
    while sum(hc) > nh:
        i = max(range(w), key=lambda j: hc[j] - ideal[j])
        hc[i] -= 1
    while sum(hc) < nh:
        i = max(range(w), key=lambda j: ideal[j] - hc[j])
        hc[i] += 1
    return hc, [float(x) for x in speeds]


def rms_norm(x, w, eps):
    return F.rms_norm(x, (x.shape[-1],), w, eps)


def apply_rope(x, cos, sin):
    # x: [s, h, 128] ; cos/sin: [s, 96] (already x.dtype)
    rd = cos.shape[-1]
    xr = x[..., :rd]
    x1, x2 = xr.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    xr = xr * cos[:, None, :] + rot * sin[:, None, :]
    return torch.cat((xr, x[..., rd:]), dim=-1)


class _ReqState:
    """Everything that depends only on the request's prompt/geometry, computed once per request."""
    pass


class SPTransformer(nn.Module):
    """Sequence-parallel forward for the t2va packing [text | audio | video]."""

    def __init__(self, model, rank: int, world: int, group=None, ckpt_dir=None):
        super().__init__()
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.rank, self.world, self.group = rank, world, group
        self.set_partition([1.0] * world)
        if world == 8 and "PIKA_GEMM_PAD_MAX" not in os.environ:
            # world 8 also pads 10 s (9176 -> 9216 rows, the tuned 9176 solutions apply): bit-identical traces (38/38 at
            # 5/10/15 s), in-worker A/B 10 s sampling -1.4% (3 vs 3)
            # and 15 s (13636 -> 13824 rows, the tuned 13636 solutions apply within PIKA_GEMM_SOLS_TOL): 15 s trace 38/38
            # bit-identical, in-worker A/B 16.32 vs 16.43-16.49 s sampling (-0.7..-1.0%, 6 vs 7 requests)
            global GEMM_PAD_MAX
            GEMM_PAD_MAX = 20000
        self.blocks = build_blocks(model)
        load_gemm_table()
        # PIKA_SPLIT: row-microbatch overlap of the Ulysses all-to-alls with attention (second NCCL group = second stream)
        # Default "auto": on at world 4 only (4-GPU eval 5.794 -> 5.904, gate margins unchanged); world 8 is not measured yet.
        self.split_spec = os.environ.get("PIKA_SPLIT", "auto")
        if self.split_spec == "auto" and world != 4:
            self.split_spec = "0"
        self.g2 = dist.new_group(backend="nccl") if (self.split_spec != "0" and world > 1) else None
        # PIKA_SDMA=1: the PIKA_SPLIT all-to-alls run on the SDMA copy engines (pika_h3.sdma; no CUs, bit-identical data)
        # Default "auto": on at world 8 (serial PIPE=3 below 50k rows, bit-identical to the RCCL path; interleaved A/B at world 8:
        # 5 s -0.8..-2.0%, 10 s -1.8..-2.3%, 15 s -0.6..-1.1% sampling; PIPE=4 at >= 50k rows). World 4 keeps RCCL (PIPE=2 measured -3.8% / -1.7% at 10 / 15 s, +2% at 5 s:
        # not gate-evaluated at world 4 yet).
        spec = os.environ.get("PIKA_SDMA", "auto")
        if spec == "auto":
            spec = "1" if world == 8 else "0"
        self.sdma = None
        if spec == "1" and world > 1:
            ok = 1
            try:
                from .sdma import SdmaComm
                self.sdma = SdmaComm(rank, world, group=group)
            except Exception as e:  # e.g. no SDMA engine / IPC failure: every rank falls back to RCCL together
                print(f"[sp] rank {rank}: SDMA init failed ({e!r}); using RCCL", flush=True)
                ok = 0
            if ok:
                try:  # fixed capacity: PIKA_SDMA_MAX_ROWS sequence rows (15 s at 768p: ~110k incl. text)
                    cap = int(os.environ.get("PIKA_SDMA_MAX_ROWS", "120000"))
                    hpr0 = NH // world
                    self.sdma.allocate([cap * 3 * hpr0 * HD * 2, (cap + world) * hpr0 * HD * 2] * 2)
                except Exception as e:
                    print(f"[sp] rank {rank}: SDMA buffer allocation failed ({e!r}); using RCCL", flush=True)
                    ok = 0
            t = torch.tensor([ok], dtype=torch.int32, device=torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
            if int(t.item()) == 0:
                self.sdma = None
        self._xpar = 0
        self.sdma_on = True
        # PIKA_SDMA_PIPE=1 (needs PIKA_SDMA=1): QKV GEMM in two row chunks (a2a(m0) on the SDMA engines behind GEMM(m1)) and the
        # out-projection of the m0 rows behind return(m1)
        # =2: QKV chunking only (the row-chunked out-projection lands on poor default GEMM kernels: +0.55 ms/layer at 5 s)
        self.pipe = os.environ.get("PIKA_SDMA_PIPE", "auto") if self.sdma is not None else "0"
        if self.pipe == "auto":  # world 8: 4 = qkv a2a behind the row-chunked QKV GEMM at N >= SDMA_MIN_N, serial SDMA (3) below
            # (A/B vs serial: 10 s -2.5%, 15 s -2.2%; 8-GPU gate pass 12.16, min margins V 3.1 / A 4.0 dB)
            # 5 = PIPE=4 + wave-aligned query split, return behind the partial last wave (bit-identical to 4; 10 s -2.5%,
            # 15 s -1.3% in-worker A/B)
            self.pipe = "5" if self.g2 is None else "2"
        self.pipe = None if self.pipe == "0" else int(self.pipe)
        # schedule below SDMA_MIN_N rows: 3 = serial (bit-identical to RCCL), 6 = self-last order + return behind the tail rows
        self.pipe_short = int(os.environ.get("PIKA_SDMA_PIPE_SHORT", "3"))
        self.eps = model.config.norm_eps
        self.qk_eps = model.config.qk_norm_eps
        self._tt_cache = {}

    def set_partition(self, speeds):
        """Work split proportional to the measured GPU speeds (attention heads, sequence rows)."""
        hc, rw = compute_partition(speeds)
        self.hcnt = hc
        self.hstart = [sum(hc[:r]) for r in range(self.world)]
        self.row_weights = rw
        self.uniform = len(set(hc)) == 1 and all(x == rw[0] for x in rw)
        self.hpr = hc[self.rank]
        dev = torch.device("cuda", torch.cuda.current_device())
        hdest, hloc = [], []
        for g in range(self.world):
            hdest += [g] * hc[g]
            hloc += list(range(hc[g]))
        self._tables = (torch.tensor(hdest, dtype=torch.int32, device=dev), torch.tensor(hloc, dtype=torch.int32, device=dev),
                        torch.tensor(self.hstart, dtype=torch.int32, device=dev), torch.tensor(hc, dtype=torch.int32, device=dev))

    # ---------------------------------------------------------------- per request
    @torch.no_grad()
    def prepare(self, prompt_embeds, text_tags, geom):
        from .core.packing import build_layout
        from diffusers.models.modeling_utils import get_parameter_dtype
        m = self.model
        st = _ReqState()
        dev = prompt_embeds.device
        L = int(prompt_embeds.shape[1])
        layout = build_layout(text_tags, num_latent_frames=geom.num_latent_frames, latent_height=geom.latent_height,
                              latent_width=geom.latent_width, num_audio_latents=geom.num_audio_latents)
        N = layout.sequence_length
        Na = int(layout.audio_indices.numel())
        Nv = int(layout.video_indices.numel())
        assert layout.num_condition_video_rows == 0 and layout.num_condition_audio_rows == 0
        assert L + Na + Nv == N and int(layout.text_indices[-1]) == L - 1 and int(layout.audio_indices[0]) == L
        assert int(layout.video_indices[0]) == L + Na
        st.L, st.Na, st.Nv, st.N = L, Na, Nv, N
        if LS.ACTIVE:
            LS.set_request_geometry(L, Na)
            ft = Nv // max(1, geom.num_latent_frames)  # video tokens per latent frame; row tokens from the latent aspect
            LS.VGEOM = (L + Na, Nv, ft, int(round((ft * geom.latent_width / geom.latent_height) ** 0.5)))
        st.sizes = split_sizes(N, self.world) if self.uniform else split_sizes_weighted(N, self.row_weights)
        st.starts = [sum(st.sizes[:r]) for r in range(self.world)]
        a0 = st.starts[self.rank]
        a1 = a0 + st.sizes[self.rank]
        st.a0, st.a1 = a0, a1
        pos = layout.position_ids.to(dev)
        cos, sin = m.rope(pos[a0:a1])
        te = m.context_embedder(prompt_embeds.to(get_parameter_dtype(m.context_embedder)))
        te = m.token_refiner(te)[0]
        st.dtype = te.dtype
        st.cos, st.sin = cos.to(st.dtype), sin.to(st.dtype)
        st.te = te
        # segment id of each local row: 0 text, 1 audio, 2 video  ->  AdaLN table row = tsidx(seg) * 3 + tag(seg)
        seg = torch.empty(N, dtype=torch.long)
        seg[:L] = 0
        seg[L:L + Na] = 1
        seg[L + Na:] = 2
        st.seg = seg[a0:a1].to(dev)
        st.nv_list = [max(0, min(st.starts[r] + st.sizes[r], N) - max(st.starts[r], L + Na)) for r in range(self.world)]
        st.na_list = [max(0, min(st.starts[r] + st.sizes[r], L + Na) - max(st.starts[r], L)) for r in range(self.world)]
        return st

    # ---------------------------------------------------------------- attention with ulysses all-to-all
    def split_rows(self, N, s):
        """s0 = rows of micro-batch m0 per rank (PIKA_SPLIT overlap), or None. auto: f = 0.4 below 50k rows, else 0.25
        (world-4 microbenchmark: best f per sequence length)."""
        if self.g2 is None or not self.uniform or self.recipe != "sdpa" or getattr(self, "split_off", False):
            return None
        if LS.ACTIVE and (LS.sparse_now() < 1.0 or LS.attn_recipe() != "bf16"):  # LOSSY steps use the unsplit v4 paths
            return None
        spec = self.split_spec
        f = (0.4 if N < 50000 else 0.25) if spec in ("1", "auto") else float(spec)
        s0 = int(round(s * f))
        return s0 if 0 < s0 < s else None

    def _attn_split(self, send, sizes, s, s0, Mp=None):
        """Exact Ulysses attention with two row micro-batches m0 = local rows [0, s0), m1 = [s0, s) of every rank.
        a2a(m0) | a2a(m1) (async, group 2) || attn(Q0, KV0) | attn(Q0, KV1) + LSE merge | return(m0) (async) || attn(Q1, KV) |
        return(m1). send: [W*s, 3*hpr*128] in the split layout of qk_rope_pack(s0=...)."""
        w, hpr = self.world, self.hpr
        N = sum(sizes)  # rank sizes may differ by one row; s0 is common to all ranks
        n0 = w * s0
        C = 3 * hpr * HD
        in1 = [s - s0] * w
        out1 = [sz - s0 for sz in sizes]
        recv = torch.empty((N, C), dtype=send.dtype, device=send.device)
        dist.all_to_all_single(recv[:n0], send[:n0], group=self.group)
        w1 = dist.all_to_all_single(recv[n0:], send[n0:], output_split_sizes=out1, input_split_sizes=in1, group=self.g2,
                                    async_op=True)
        r = recv.view(N, 3, hpr * HD)
        qq = r[:, 0].view(1, N, hpr, HD).transpose(1, 2)
        kk = r[:, 1].view(1, N, hpr, HD).transpose(1, 2)
        vv = r[:, 2].view(1, N, hpr, HD).transpose(1, 2)
        oa, la = _attn_lse(qq[:, :, :n0], kk[:, :, :n0], vv[:, :, :n0])
        w1.wait()
        ob, lb = _attn_lse(qq[:, :, :n0], kk[:, :, n0:], vv[:, :, n0:])
        o = torch.empty((N, hpr, HD), dtype=send.dtype, device=send.device)
        K.lse_merge_into(o, 0, oa, la, ob[0], lb[0])
        back = torch.empty((w * s, hpr * HD), dtype=send.dtype, device=send.device)
        w2 = dist.all_to_all_single(back[:n0], o[:n0].view(n0, hpr * HD), group=self.g2, async_op=True)
        q1 = qq[:, :, n0:]
        Nm = tail_plan(N - n0, hpr) if TAIL_SPLIT else None
        if ATTN_KERNEL == "v4":  # written straight into o[n0:] (contiguous rows; the copy below becomes a no-op skip)
            o1 = attention_v4_into(q1.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), v4_tail_plan(N - n0, hpr),
                                   out=o[n0:])
        elif Nm is not None:
            o1 = attention_tail_into(q1, kk, vv, Nm)
        else:
            o1 = F.scaled_dot_product_attention(q1, kk, vv)[0].transpose(0, 1)
        if o1.data_ptr() != o[n0:].data_ptr():
            o[n0:].copy_(o1)
        dist.all_to_all_single(back[n0:], o[n0:].view(N - n0, hpr * HD), output_split_sizes=in1, input_split_sizes=out1,
                               group=self.group)
        w2.wait()
        res = torch.empty((Mp or s, INNER), dtype=send.dtype, device=send.device)
        res[:s0].view(s0, w, hpr * HD).copy_(back[:n0].view(w, s0, hpr * HD).permute(1, 0, 2))
        res[s0:s].view(s - s0, w, hpr * HD).copy_(back[n0:].view(w, s - s0, hpr * HD).permute(1, 0, 2))
        return res

    def _attn_split_sdma(self, send, sizes, s, s0, Mp=None, produce=None, outproj=None):
        """_attn_split with the four all-to-alls on the SDMA engines (same schedule, same kernels, same data).
        Buffers: [recv(even), back(even), recv(odd), back(odd)] by layer parity.
        PIKA_SDMA_PIPE: produce(rows, send) fills send for local rows (r0, r1) (row-chunked QKV GEMM); outproj(res_rows, n)
        returns the out-projection of n rows; the result is then (y_o [s, H]) instead of the attention output."""
        w, hpr, r = self.world, self.hpr, self.rank
        cm = self.sdma
        N = sum(sizes)
        n0 = w * s0
        C = 3 * hpr * HD
        RB = C * 2  # bytes per qkv row
        OB = hpr * HD * 2  # bytes per output row
        smax = max(sizes)
        cm.ensure([N * RB, w * smax * OB, N * RB, w * smax * OB])
        par = self._xpar
        self._xpar ^= 1
        bi_r, bi_b = 2 * par, 2 * par + 1
        if produce is not None:
            send = torch.empty((w * s, C), dtype=torch.bfloat16, device=torch.device("cuda", torch.cuda.current_device()))
            produce((0, s0), send)
        recv = cm.view(bi_r, send.dtype, (N, C))
        back = cm.view(bi_b, send.dtype, (w * s, hpr * HD))
        # m1 region of a receiver d: source blocks in the order d+1, d+2, ..., d (own rows last, PIKA_SDMA_SELFLAST=1) or 0..w-1
        def off1_at(d, src):
            order = [(d + 1 + i) % w for i in range(w)] if SDMA_SELFLAST else list(range(w))
            return n0 + sum(sizes[q] - s0 for q in order[:order.index(src)])
        off1 = [off1_at(r, p) for p in range(w)]  # row offset of source p's m1 block in my recv
        sb = send.data_ptr()
        # qkv m0 / m1: my destination-major blocks -> rows of the peer's recv
        t0 = cm.post({p: [(sb + p * s0 * RB, s0 * RB, r * s0 * RB)] for p in range(w) if p != r}, bi_r)
        if produce is not None:
            recv[r * s0:(r + 1) * s0].copy_(send[r * s0:(r + 1) * s0])
            produce((s0, s), send)
        t1 = cm.post({p: [(sb + (n0 + p * (s - s0)) * RB, (s - s0) * RB, off1_at(p, r) * RB)] for p in range(w) if p != r}, bi_r)
        if produce is None:
            recv[r * s0:(r + 1) * s0].copy_(send[r * s0:(r + 1) * s0])
        recv[off1[r]:off1[r] + s - s0].copy_(send[n0 + r * (s - s0):n0 + (r + 1) * (s - s0)])
        rv = recv.view(N, 3, hpr * HD)
        qq = rv[:, 0].view(1, N, hpr, HD).transpose(1, 2)
        kk = rv[:, 1].view(1, N, hpr, HD).transpose(1, 2)
        vv = rv[:, 2].view(1, N, hpr, HD).transpose(1, 2)
        cm.wait(t0)
        oa, la = _attn_lse(qq[:, :, :n0], kk[:, :, :n0], vv[:, :, :n0])
        cm.wait(t1)
        ob, lb = _attn_lse(qq[:, :, :n0], kk[:, :, n0:], vv[:, :, n0:])
        o = torch.empty((N, hpr, HD), dtype=send.dtype, device=send.device)
        K.lse_merge_into(o, 0, oa, la, ob[0], lb[0])
        ob_ = o.data_ptr()
        # return m0: rows of source p -> block r of p's back[:n0]
        t2 = cm.post({p: [(ob_ + p * s0 * OB, s0 * OB, r * s0 * OB)] for p in range(w) if p != r}, bi_b)
        back[r * s0:(r + 1) * s0].copy_(o[r * s0:(r + 1) * s0].view(s0, hpr * HD))
        q1 = qq[:, :, n0:]
        Nm = tail_plan(N - n0, hpr) if TAIL_SPLIT else None
        t3 = None
        if SDMA_SELFLAST and ATTN_KERNEL == "v4":
            # queries of the other ranks first, return(m1) behind the attention of my own rows (no transfer needed)
            ns = N - (s - s0)
            assert off1[r] == ns
            k4, v4_ = kk.transpose(1, 2), vv.transpose(1, 2)
            attention_v4_into(qq[:, :, n0:ns].transpose(1, 2), k4, v4_, v4_tail_plan(ns - n0, hpr), out=o[n0:ns])
            t3 = cm.post({p: [(ob_ + off1[p] * OB, (sizes[p] - s0) * OB, (n0 + r * (sizes[p] - s0)) * OB)] for p in range(w)
                          if p != r}, bi_b)
            o1 = attention_v4_into(qq[:, :, ns:].transpose(1, 2), k4, v4_, v4_tail_plan(N - ns, hpr), out=o[ns:])
            o1 = o[n0:]
        elif ATTN_KERNEL == "v4":
            o1 = attention_v4_into(q1.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), v4_tail_plan(N - n0, hpr),
                                   out=o[n0:])
        elif Nm is not None:
            o1 = attention_tail_into(q1, kk, vv, Nm)
        else:
            o1 = F.scaled_dot_product_attention(q1, kk, vv)[0].transpose(0, 1)
        if o1.data_ptr() != o[n0:].data_ptr():
            o[n0:].copy_(o1)
        # return m1: rows off1[p].. of source p -> p's back[n0 + r*(s_p - s0)]
        if t3 is None:
            t3 = cm.post({p: [(ob_ + off1[p] * OB, (sizes[p] - s0) * OB, (n0 + r * (sizes[p] - s0)) * OB)] for p in range(w)
                          if p != r}, bi_b)
        back[n0 + r * (s - s0):n0 + (r + 1) * (s - s0)].copy_(o[off1[r]:off1[r] + s - s0].view(s - s0, hpr * HD))
        cm.wait(t2)
        res = torch.empty((Mp or s, INNER), dtype=send.dtype, device=send.device)
        res[:s0].view(s0, w, hpr * HD).copy_(back[:n0].view(w, s0, hpr * HD).permute(1, 0, 2))
        if outproj is not None:
            y0 = outproj(res[:s0], s0)  # behind return(m1)
        cm.wait(t3)
        res[s0:s].view(s - s0, w, hpr * HD).copy_(back[n0:].view(w, s - s0, hpr * HD).permute(1, 0, 2))
        del send, o
        if outproj is not None:
            y1 = outproj(res[s0:], s - s0)
            return torch.cat([y0[:s0], y1[:s - s0]], 0)
        return res

    def _attn_packed_sdma_serial(self, send, sizes, s, Mp=None, selflast=False):
        """_attn_packed with the two all-to-alls on the SDMA engines, no overlap (PIKA_SDMA_PIPE=3): same layout and kernels as
        the RCCL path (bit-identical), only the transfer engine differs (world 8: 7 links x ~60 GB/s, no CUs)."""
        w, hpr, r = self.world, self.hpr, self.rank
        cm = self.sdma
        N = sum(sizes)
        C = 3 * hpr * HD
        RB, OB = C * 2, hpr * HD * 2
        cm.ensure([N * RB, w * max(sizes) * OB, N * RB, w * max(sizes) * OB])
        par = self._xpar
        self._xpar ^= 1
        bi_r, bi_b = 2 * par, 2 * par + 1
        st = [sum(sizes[:p]) for p in range(w)]
        recv = cm.view(bi_r, send.dtype, (N, C))
        back = cm.view(bi_b, send.dtype, (w * s, hpr * HD))
        sb = send.data_ptr()
        if LS.ACTIVE:
            LS.KEY_PREFIX_OFF = 0 if not HEAD_ROT else -1
        if selflast:
            return self._serial_selflast(send, sizes, s, Mp, recv, back, bi_r, bi_b)
        if LS.ACTIVE and LS.per_step(LS.CFG.get("comm_fp8"), default=0) and not HEAD_ROT and not DEV_NOXFER:
            return self._serial_fp8(send, sizes, s, Mp, bi_r, bi_b)  # LOSSY: FP8 all-to-alls (lossy_comm.py)
        # dev PIKA_HEAD_ROT=k: rank p computes head group (p - k) % w (send block g goes to rank (g + k) % w); exact, diagnostic
        k_ = HEAD_ROT % w
        if not DEV_NOXFER & 1:
            t0 = cm.post({p: [(sb + ((p - k_) % w) * s * RB, s * RB, st[r] * RB)] for p in range(w) if p != r}, bi_r)
        recv[st[r]:st[r] + s].copy_(send[((r - k_) % w) * s:((r - k_) % w + 1) * s])
        if not DEV_NOXFER & 1:
            cm.wait(t0)
        rv = recv.view(N, 3, hpr * HD)
        q4 = rv[:, 0].view(1, N, hpr, HD)
        k4 = rv[:, 1].view(1, N, hpr, HD)
        v4_ = rv[:, 2].view(1, N, hpr, HD)
        Ns = _dev_skip_rows(r, N)
        if Ns < N:
            o = torch.zeros((N, hpr, HD), dtype=send.dtype, device=send.device)
            attention_v4_into(q4[:, :Ns], k4, v4_, v4_tail_plan(Ns, hpr), out=o[:Ns])
            o = o.view(N, hpr * HD)
        else:
            o = attention_v4_into(q4, k4, v4_, v4_tail_plan(N, hpr)).view(N, hpr * HD)
        ob_ = o.data_ptr()
        if not DEV_NOXFER & 2:
            t1 = cm.post({p: [(ob_ + st[p] * OB, sizes[p] * OB, r * sizes[p] * OB)] for p in range(w) if p != r}, bi_b)
        back[r * s:(r + 1) * s].copy_(o[st[r]:st[r] + s])
        if not DEV_NOXFER & 2:
            cm.wait(t1)
        res = torch.empty((Mp or s, INNER), dtype=send.dtype, device=send.device)
        if k_:  # head group g came back from rank (g + k) % w
            idx = torch.tensor([(g + k_) % w for g in range(w)], device=back.device)
            res[:s].view(s, w, hpr * HD).copy_(back.view(w, s, hpr * HD)[idx].permute(1, 0, 2))
        else:
            res[:s].view(s, w, hpr * HD).copy_(back.view(w, s, hpr * HD).permute(1, 0, 2))
        del send, o
        return res

    def _serial_fp8(self, send, sizes, s, Mp, bi_r, bi_b):
        """LOSSY (lossy key comm_fp8): _attn_packed_sdma_serial with FP8 rows + per-(row, 128) fp32 scales on the wire."""
        from . import lossy_comm as LC
        w, hpr, r = self.world, self.hpr, self.rank
        cm = self.sdma
        N = sum(sizes)
        smax = max(sizes)
        C = 3 * hpr * HD
        G = C // LC.SEG
        st = [sum(sizes[:p]) for p in range(w)]
        pay_r, sc_r = LC.views(cm, bi_r, N, C)
        so = LC.off_scales(N, C)
        qs = torch.empty((w * s, C), dtype=LC.F8, device=send.device)
        ss = torch.empty((w * s, G), dtype=torch.float32, device=send.device)
        LC.quant_into(send, qs, ss)
        qp, sp = qs.data_ptr(), ss.data_ptr()
        t0 = cm.post({p: [(qp + p * s * C, s * C, st[r] * C), (sp + p * s * G * 4, s * G * 4, so + st[r] * G * 4)]
                      for p in range(w) if p != r}, bi_r)
        pay_r[st[r]:st[r] + s].copy_(qs[r * s:(r + 1) * s])
        sc_r[st[r]:st[r] + s].copy_(ss[r * s:(r + 1) * s])
        cm.wait(t0)
        recv = torch.empty((N, C), dtype=send.dtype, device=send.device)
        LC.dequant_into(pay_r, sc_r, recv)
        rv = recv.view(N, 3, hpr * HD)
        q4 = rv[:, 0].view(1, N, hpr, HD)
        k4 = rv[:, 1].view(1, N, hpr, HD)
        v4_ = rv[:, 2].view(1, N, hpr, HD)
        LS.KEY_PREFIX_OFF = 0
        o = attention_v4_into(q4, k4, v4_, v4_tail_plan(N, hpr)).view(N, hpr * HD)
        Co = hpr * HD
        Go = Co // LC.SEG
        pay_b, sc_b = LC.views(cm, bi_b, w * smax, Co)
        sob = LC.off_scales(w * smax, Co)
        oq = torch.empty((N, Co), dtype=LC.F8, device=send.device)
        osc = torch.empty((N, Go), dtype=torch.float32, device=send.device)
        LC.quant_into(o, oq, osc)
        op_, osp = oq.data_ptr(), osc.data_ptr()
        t1 = cm.post({p: [(op_ + st[p] * Co, sizes[p] * Co, r * sizes[p] * Co),
                          (osp + st[p] * Go * 4, sizes[p] * Go * 4, sob + r * sizes[p] * Go * 4)] for p in range(w) if p != r}, bi_b)
        pay_b[r * s:(r + 1) * s].copy_(oq[st[r]:st[r] + s])
        sc_b[r * s:(r + 1) * s].copy_(osc[st[r]:st[r] + s])
        cm.wait(t1)
        back = torch.empty((w * s, Co), dtype=send.dtype, device=send.device)
        LC.dequant_into(pay_b[:w * s], sc_b[:w * s], back)
        res = torch.empty((Mp or s, INNER), dtype=send.dtype, device=send.device)
        res[:s].view(s, w, hpr * HD).copy_(back.view(w, s, hpr * HD).permute(1, 0, 2))
        del send, o, qs, ss, oq, osc
        return res

    def _serial_selflast(self, send, sizes, s, Mp, recv, back, bi_r, bi_b):
        """PIPE=6 (short sequences): one qkv all-to-all (no GEMM chunking) into the receive order d+1, ..., d (own rows last); the
        attention main launch (full waves; all other ranks' rows) is followed by the return post, so the transfer overlaps the
        tail rows (own rows) of the wave-quantisation split. Same kernels as PIPE=3, key order differs (same precision)."""
        w, hpr, r = self.world, self.hpr, self.rank
        cm = self.sdma
        N = sum(sizes)
        C = 3 * hpr * HD
        RB, OB = C * 2, hpr * HD * 2

        def off_at(d, src):
            order = [(d + 1 + i) % w for i in range(w)]
            return sum(sizes[q] for q in order[:order.index(src)])
        sb = send.data_ptr()
        ns = N - s
        if LS.ACTIVE:
            LS.KEY_PREFIX_OFF = off_at(r, 0)
        t0 = cm.post({p: [(sb + p * s * RB, s * RB, off_at(p, r) * RB)] for p in range(w) if p != r}, bi_r)
        recv[ns:].copy_(send[r * s:(r + 1) * s])
        cm.wait(t0)
        rv = recv.view(N, 3, hpr * HD)
        q4 = rv[:, 0].view(1, N, hpr, HD)
        k4 = rv[:, 1].view(1, N, hpr, HD)
        v4_ = rv[:, 2].view(1, N, hpr, HD)
        o = torch.empty((N, hpr, HD), dtype=send.dtype, device=send.device)
        Nm = v4_tail_plan(N, hpr)
        if Nm is None or Nm < ns:  # no tail split (or the tail reaches into other ranks' rows): plain launch, serial return
            Nm = N
            _v4_into(q4, k4, v4_, o.unsqueeze(0))
        else:
            _v4_into(q4[:, :Nm], k4, v4_, o[:Nm].unsqueeze(0))
        ob_ = o.data_ptr()
        t1 = cm.post({p: [(ob_ + off_at(r, p) * OB, sizes[p] * OB, r * sizes[p] * OB)] for p in range(w) if p != r}, bi_b)
        if Nm < N:  # tail rows [Nm, N) (own rows): keys split in chunks + fp32 LSE merge, as attention_v4_into
            _tail_rows_into(o, Nm, q4, k4, v4_, hpr)
        back[r * s:(r + 1) * s].copy_(o[ns:].view(s, hpr * HD))
        cm.wait(t1)
        res = torch.empty((Mp or s, INNER), dtype=send.dtype, device=send.device)
        res[:s].view(s, w, hpr * HD).copy_(back.view(w, s, hpr * HD).permute(1, 0, 2))
        del send, o
        return res

    def _attn_packed_sdma(self, sizes, s, Mp, produce, fa, qsplit=True):
        """Unsplit Ulysses attention with SDMA transfers (PIKA_SDMA=1, PIKA_SPLIT off, e.g. world 8): QKV rows [0, sA) are produced
        and sent while rows [sA, s) are produced; the attention runs as two query launches (other ranks' rows, then my own rows) so
        the return of the other ranks' outputs is hidden behind the second launch. Receive order per receiver d: sources d+1, ..., d."""
        w, hpr, r = self.world, self.hpr, self.rank
        cm = self.sdma
        N = sum(sizes)
        C = 3 * hpr * HD
        RB, OB = C * 2, hpr * HD * 2
        smax = max(sizes)
        cm.ensure([N * RB, w * smax * OB, N * RB, w * smax * OB])
        par = self._xpar
        self._xpar ^= 1
        bi_r, bi_b = 2 * par, 2 * par + 1

        def off_at(d, src):
            order = [(d + 1 + i) % w for i in range(w)]
            return sum(sizes[q] for q in order[:order.index(src)])
        sA = max(1, min(s - 1, int(round(min(sizes) * fa))))
        if LS.ACTIVE:
            LS.KEY_PREFIX_OFF = off_at(r, 0)
        send = torch.empty((w * s, C), dtype=torch.bfloat16, device=torch.device("cuda", torch.cuda.current_device()))
        recv = cm.view(bi_r, send.dtype, (N, C))
        back = cm.view(bi_b, send.dtype, (w * s, hpr * HD))
        sb = send.data_ptr()
        produce((0, sA), send)
        tA = cm.post({p: [(sb + (p * s) * RB, sA * RB, off_at(p, r) * RB)] for p in range(w) if p != r}, bi_r)
        produce((sA, s), send)
        tB = cm.post({p: [(sb + (p * s + sA) * RB, (s - sA) * RB, (off_at(p, r) + sA) * RB)] for p in range(w) if p != r}, bi_r)
        ns = N - s
        recv[ns:].copy_(send[r * s:(r + 1) * s])
        cm.wait(tA)
        cm.wait(tB)
        rv = recv.view(N, 3, hpr * HD)
        q4 = rv[:, 0].view(1, N, hpr, HD)
        k4 = rv[:, 1].view(1, N, hpr, HD)
        v4_ = rv[:, 2].view(1, N, hpr, HD)
        o = torch.empty((N, hpr, HD), dtype=send.dtype, device=send.device)
        n1 = None
        if qsplit == "wave" and LS.ACTIVE and LS.get("s2wave", 1) == 0 and LS.sparse_now() < 1.0:
            qsplit = {"own": True, "none": False}.get(LS.get("s2split", "own"), "wave")  # LOSSY sparse step (opt-in)
        if qsplit == "wave":  # PIPE=5: launch 1 = the full waves (all other ranks' rows), return overlaps the partial last wave
            n1 = (hpr * -(-N // 256)) // _num_cus() * _num_cus() // hpr * 256
            if not (ns <= n1 < N):
                n1 = None
        if n1 is not None:
            n1s = _dev_skip_rows(r, n1)
            if n1s < n1:
                o[n1s:n1].zero_()
            attention_v4_into(q4[:, :n1s], k4, v4_, v4_tail_plan(n1s, hpr), out=o[:n1s])
            ob_ = o.data_ptr()
            t2 = cm.post({p: [(ob_ + off_at(r, p) * OB, sizes[p] * OB, r * sizes[p] * OB)] for p in range(w) if p != r}, bi_b)
            attention_v4_into(q4[:, n1:], k4, v4_, v4_tail_plan(N - n1, hpr), out=o[n1:])
        elif qsplit is True:
            attention_v4_into(q4[:, :ns], k4, v4_, v4_tail_plan(ns, hpr), out=o[:ns])
            ob_ = o.data_ptr()
            t2 = cm.post({p: [(ob_ + off_at(r, p) * OB, sizes[p] * OB, r * sizes[p] * OB)] for p in range(w) if p != r}, bi_b)
            attention_v4_into(q4[:, ns:], k4, v4_, v4_tail_plan(s, hpr), out=o[ns:])
        else:  # PIPE=4: one attention launch (no short self-rows launch), serial return
            attention_v4_into(q4, k4, v4_, v4_tail_plan(N, hpr), out=o)
            ob_ = o.data_ptr()
            t2 = cm.post({p: [(ob_ + off_at(r, p) * OB, sizes[p] * OB, r * sizes[p] * OB)] for p in range(w) if p != r}, bi_b)
        back[r * s:(r + 1) * s].copy_(o[ns:].view(s, hpr * HD))
        cm.wait(t2)
        res = torch.empty((Mp or s, INNER), dtype=send.dtype, device=send.device)
        res[:s].view(s, w, hpr * HD).copy_(back.view(w, s, hpr * HD).permute(1, 0, 2))
        del send, o
        return res

    def _attn_packed(self, send, sizes, s, Mp=None):
        """send: [W*s, 3*hpr*128] in destination-major layout. Returns attention output [s, 56*128]."""
        w, hpr = self.world, self.hpr
        N = sum(sizes)
        # single rank: the all-to-all is the identity. (NCCL's one-rank all_to_all_single leaves the output UNWRITTEN: the
        # attention output read uninitialised memory and the world-1 fallback produced non-finite latents.)
        recv = send if w == 1 else torch.empty((N, 3 * hpr * HD), dtype=send.dtype, device=send.device)
        if w == 1:
            pass
        elif self.uniform:
            dist.all_to_all_single(recv, send, output_split_sizes=sizes, input_split_sizes=[s] * w, group=self.group)
        else:
            dist.all_to_all_single(recv.view(-1), send.view(-1),
                                   output_split_sizes=[sz * 3 * hpr * HD for sz in sizes],
                                   input_split_sizes=[s * 3 * self.hcnt[r] * HD for r in range(w)], group=self.group)
        recv = recv.view(N, 3, hpr * HD)
        if LS.ACTIVE:
            LS.KEY_PREFIX_OFF = 0
        _dump = os.environ.get("PIKA_DUMP_QKV")
        if _dump and self.rank == 0 and f"{self._step}:{self._li}" in os.environ.get("PIKA_DUMP_AT", "0:0,2:24,5:47").split(","):
            os.makedirs(_dump, exist_ok=True)
            hd = min(hpr, 7)
            r3 = recv.view(N, 3, hpr, HD)[:, :, :hd]
            torch.save({k_: r3[:, i].contiguous().cpu() for i, k_ in enumerate("qkv")},
                       os.path.join(_dump, f"qkv_N{N}_s{self._step}_l{self._li}.pt"))
        qq = recv[:, 0].view(1, N, hpr, HD).transpose(1, 2)
        kk = recv[:, 1].view(1, N, hpr, HD).transpose(1, 2)
        vv = recv[:, 2].view(1, N, hpr, HD).transpose(1, 2)
        if self.recipe == "sdpa" and ATTN_KERNEL == "v4":
            Nm = v4_tail_plan(N, hpr)
            o = attention_v4_into(qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), Nm).view(N, hpr * HD)
        elif self.recipe == "sdpa":
            Nm = tail_plan(N, hpr) if TAIL_SPLIT else None
            if Nm is not None:
                o = attention_tail_into(qq, kk, vv, Nm).view(N, hpr * HD)
            else:
                o = F.scaled_dot_product_attention(qq, kk, vv)  # [1, hpr, N, 128]
                o = o[0].transpose(0, 1).contiguous().view(N, hpr * HD)
        else:
            mha, AF = _load_v4()
            fmt = {"bf16": (AF.BF16, AF.BF16, AF.BF16), "bf16fp8": (AF.BF16, AF.BF16, AF.FP8_E4M3),
                   "fp8": (AF.FP8_E4M3, AF.FP8_E4M3, AF.FP8_E4M3), "int8fp8": (AF.INT8, AF.INT8, AF.FP8_E4M3)}[self.recipe]
            q4 = recv[:, 0].reshape(1, N, hpr, HD).contiguous()
            k4 = recv[:, 1].reshape(1, N, hpr, HD).contiguous()
            v4 = recv[:, 2].reshape(1, N, hpr, HD).contiguous()
            o = mha(q4, k4, v4, *fmt).reshape(N, hpr * HD)
        if w == 1:
            res = torch.empty((Mp or s, INNER), dtype=send.dtype, device=send.device)
            res[:s].copy_(o)
            return res
        if self.uniform:
            back = torch.empty((w * s, hpr * HD), dtype=send.dtype, device=send.device)
            dist.all_to_all_single(back, o, output_split_sizes=[s] * w, input_split_sizes=sizes, group=self.group)
            res = torch.empty((Mp or s, INNER), dtype=send.dtype, device=send.device)
            res[:s].view(s, w, hpr * HD).copy_(back.view(w, s, hpr * HD).permute(1, 0, 2))
            return res
        back = torch.empty((s * INNER,), dtype=send.dtype, device=send.device)
        dist.all_to_all_single(back, o.reshape(-1), output_split_sizes=[s * self.hcnt[g] * HD for g in range(w)],
                               input_split_sizes=[sz * hpr * HD for sz in sizes], group=self.group)
        out = torch.empty((Mp or s, INNER), dtype=send.dtype, device=send.device)
        off = 0
        for g in range(w):
            n_g = s * self.hcnt[g] * HD
            out[:s, self.hstart[g] * HD:(self.hstart[g] + self.hcnt[g]) * HD] = back[off:off + n_g].view(s, self.hcnt[g] * HD)
            off += n_g
        return out

    def free_adaln(self, ckpt_dir):
        """Release the AdaLN projection weights (~26 GB per rank); tables for the fixed schedule are cached."""
        self.ckpt_dir = ckpt_dir
        torch.cuda.empty_cache()

    def _read_adaln(self, li, device):
        import glob
        import json
        from safetensors import safe_open
        idx = json.load(open(os.path.join(self.ckpt_dir, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
        out = []
        for nm in ("weight", "bias"):
            key = f"transformer_blocks.{li}.adaln_proj.linear.{nm}"
            with safe_open(os.path.join(self.ckpt_dir, idx[key]), framework="pt", device=str(device)) as f:
                out.append(f.get_tensor(key))
        return out[0], out[1]

    @torch.no_grad()
    def timestep_tables(self, tv: float, ta: float, dtype, device):
        """Everything that depends only on the (fixed-schedule) timestep values: per-layer AdaLN modulation tables, output
        norm shift/scale and the segment -> table-row maps. Cached by timestep value (no prompt/seed dependence)."""
        from diffusers.models.modeling_utils import get_parameter_dtype
        tv32 = float(torch.tensor(tv, dtype=torch.float32))
        ta32 = float(torch.tensor(ta, dtype=torch.float32))
        key = (tv32, ta32)
        tt = self._tt_cache.get(key)
        if tt is not None:
            return tt
        m = self.model
        # distinct timesteps exactly as torch.unique(sorted) over the row timesteps
        if tv32 == ta32:
            uniq, iv, ia = [tv32], 0, 0
        elif tv32 < ta32:
            uniq, iv, ia = [tv32, ta32], 0, 1
        else:
            uniq, iv, ia = [ta32, tv32], 1, 0
        timestep = torch.tensor(uniq, dtype=torch.float32, device=device)
        temb = m.time_proj(timestep)
        temb = m.time_embedder(temb.to(get_parameter_dtype(m.time_embedder)))
        silu_t = F.silu(temb.float()).to(dtype)
        mods = []
        for li, b in enumerate(self.blocks):
            ada_w, ada_b = b.ada_w, b.ada_b
            if ada_w is None:  # weights were released after the fixed-schedule tables were built: re-read this layer
                ada_w, ada_b = self._read_adaln(li, device)
            mods.append(F.linear(silu_t, ada_w, ada_b).view(-1, 6 * H).chunk(6, dim=-1))
        no = m.norm_out
        sh, sc = no.linear(F.silu(temb.float()).to(get_parameter_dtype(no.linear))).chunk(2, dim=-1)
        tt = dict(mods=mods, out_sh=sh.contiguous(), out_sc=sc.contiguous(),
                  seg_tab=torch.tensor([iv * 3 + 1, ia * 3 + 2, iv * 3 + 0], dtype=torch.long, device=device),
                  ti_tab=torch.tensor([iv, ia, iv], dtype=torch.long, device=device))
        self._tt_cache[key] = tt
        return tt

    @torch.no_grad()
    def forward(self, st, video_rows, audio_rows, tv: float, ta: float, step: int = 0):
        """video_rows [Nv, 96] fp32, audio_rows [Na, 32] fp32 (target rows, full). Returns full (video_out, audio_out)."""
        from diffusers.models.modeling_utils import get_parameter_dtype
        m = self.model
        rank, w = self.rank, self.world
        sizes, starts, a0, a1, N, L, Na = st.sizes, st.starts, st.a0, st.a1, st.N, st.L, st.Na
        s = sizes[rank]
        dtype = st.dtype
        pieces = []
        lo, hi = max(a0, 0), min(a1, L)
        if hi > lo:
            pieces.append(st.te[lo:hi])
        lo, hi = max(a0, L), min(a1, L + Na)
        if hi > lo:
            pieces.append(m.audio_proj_in(audio_rows[lo - L:hi - L].to(get_parameter_dtype(m.audio_proj_in))).to(dtype))
        lo, hi = max(a0, L + Na), min(a1, N)
        if hi > lo:
            pieces.append(m.proj_in(video_rows[lo - L - Na:hi - L - Na].to(get_parameter_dtype(m.proj_in))).to(dtype))
        x = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        tt = self.timestep_tables(tv, ta, dtype, x.device)
        adaln_idx = tt["seg_tab"][st.seg]
        cos, sin = st.cos.contiguous(), st.sin.contiguous()
        assert not K.ROPE_PAIR or cos.shape[1] == ROPE_DIM

        w_ = self.world
        hpr = self.hpr
        self.recipe = attn_recipe(step)
        ff = None
        g2_prev = None
        Mp = gemm_rows(s)  # GEMM row count (rows >= s are padding: never read, outputs ignored)
        if LS.ACTIVE:
            LS.STEP = step
        s0 = self.split_rows(N, min(sizes))
        self._step = step
        # SDMA only for long sequences: at 5 s (N 38k) the truly concurrent transfers cost more compute clock under the power cap
        # (warm profile: attention 538 vs 493 ms/step) than they hide; interleaved A/B 5 s +2%, 10 s -3.8%, 15 s -1.7%
        sdma_req = self.sdma_on
        fits = self.sdma is not None and self.sdma.fits([N * 3 * hpr * HD * 2, w_ * max(sizes) * hpr * HD * 2] * 2)
        pipe_req = self.pipe
        if self.pipe in (4, 5) and N < SDMA_MIN_N:  # short sequences: serial SDMA (overlap measured negative at 5 s)
            self.pipe = 3
        self.sdma_on = sdma_req and fits and (N >= SDMA_MIN_N or self.pipe == 3)
        if LS.ACTIVE:
            from . import lowprec as LP
            LP.PREFIX_ROWS = max(0, min(a1, L + Na) - a0)  # local text+audio rows (they lead the global sequence)
            LP.PREFIX_TOKENS = L + Na
        for li, b in enumerate(self.blocks):
            self._li = li
            if LS.ACTIVE:
                LP.set_layer(li)  # LOSSY: per-block GEMM selection + K/V quantization cache generation
                LS.LAYER = li
            sh1, sc1, g1, sh2, sc2, g2 = tt["mods"][li]
            if ff is None:
                h = K.mod_norm(x, b.n1, sc1, sh1, adaln_idx, self.eps, pad=Mp)
            elif FUSE_LORA2:
                x, h = K.mod_norm_lora(x, b.n1, sc1, sh1, adaln_idx, self.eps, *ff, g2_prev, pad=Mp)
            else:
                x, h = K.mod_norm(x, b.n1, sc1, sh1, adaln_idx, self.eps, y=ff, gate_tab=g2_prev, pad=Mp)
            if self.pipe == 3 and self.sdma_on and s0 is None and self.uniform and ATTN_KERNEL == "v4" and self.recipe == "sdpa":
                if FUSE_LORA:
                    y_, t_, B_ = lora_parts_(h, b.qkv)
                    send = K.qk_rope_pack_lora(y_, t_, B_, b.qn, b.kn, cos, sin, w_, self.qk_eps, s=s)
                else:
                    assert not K.ROPE_PAIR, 'PIKA_ROPE_PAIR needs the fused LoRA rope path (PIKA_FUSE_LORA=1, uniform heads)'
                    send = K.qk_rope_pack(lora_linear(h, b.qkv, part_n=INNER, name='qkv', s=s), b.qn, b.kn, cos, sin, w_,
                                          self.qk_eps, s=s)
                o = self._attn_packed_sdma_serial(send, sizes, s, Mp, selflast=self.pipe_short == 6)
                y_o = lora_linear(o, b.o, name='o', s=s)
                x, h = K.mod_norm(x, b.n2, sc2, sh2, adaln_idx, self.eps, y=y_o, gate_tab=g1, pad=Mp)
                if FUSE_LORA:
                    y_, t_, B_ = lora_parts_(h, b.up)
                    a_ = K.swiglu_lora(y_, t_, B_, s=s, pad=Mp)
                else:
                    a_ = K.swiglu(lora_linear(h, b.up, name='up', s=s), s=s, pad=Mp)
                ff = lora_linear(a_, b.down, name='down', s=s)
                g2_prev = g2
                continue
            if self.pipe in (2, 4, 5) and self.sdma_on and s0 is None and FUSE_LORA and self.uniform and ATTN_KERNEL == "v4" \
                    and self.recipe == "sdpa":
                def produce(rows, send, h=h, b=b):
                    r0, r1 = rows
                    hh = h[r0:r1] if r1 < s else h[r0:]
                    y_, t_, B_ = lora_parts_(hh, b.qkv)
                    K.qk_rope_pack_lora(y_, t_, B_, b.qn, b.kn, cos, sin, w_, self.qk_eps, s=s, rows=rows, out=send)
                qs = True if self.pipe == 2 else ("wave" if self.pipe == 5 else False)
                y_o = lora_linear(self._attn_packed_sdma(sizes, s, Mp, produce, SDMA_FA, qsplit=qs), b.o, name='o', s=s)
                x, h = K.mod_norm(x, b.n2, sc2, sh2, adaln_idx, self.eps, y=y_o, gate_tab=g1, pad=Mp)
                y_, t_, B_ = lora_parts_(h, b.up)
                a_ = K.swiglu_lora(y_, t_, B_, s=s, pad=Mp)
                ff = lora_linear(a_, b.down, name='down', s=s)
                g2_prev = g2
                continue
            if self.pipe and self.sdma_on and s0 is not None and FUSE_LORA and self.uniform:
                def produce(rows, send, h=h, b=b):
                    r0, r1 = rows
                    hh = h[r0:r1] if r1 < s else h[r0:]  # the last chunk keeps the GEMM row padding
                    y_, t_, B_ = lora_parts_(hh, b.qkv)
                    K.qk_rope_pack_lora(y_, t_, B_, b.qn, b.kn, cos, sin, w_, self.qk_eps, s=s, s0=s0, rows=rows, out=send)
                if self.pipe == 1:
                    y_o = self._attn_split_sdma(None, sizes, s, s0, Mp, produce=produce,
                                                outproj=lambda xr, n, b=b: lora_linear(xr, b.o, name='o', s=n))
                else:
                    y_o = lora_linear(self._attn_split_sdma(None, sizes, s, s0, Mp, produce=produce), b.o, name='o', s=s)
                x, h = K.mod_norm(x, b.n2, sc2, sh2, adaln_idx, self.eps, y=y_o, gate_tab=g1, pad=Mp)
                y_, t_, B_ = lora_parts_(h, b.up)
                a_ = K.swiglu_lora(y_, t_, B_, s=s, pad=Mp)
                ff = lora_linear(a_, b.down, name='down', s=s)
                g2_prev = g2
                continue
            if FUSE_LORA and self.uniform:
                y_, t_, B_ = lora_parts_(h, b.qkv)
                send = K.qk_rope_pack_lora(y_, t_, B_, b.qn, b.kn, cos, sin, w_, self.qk_eps, s=s, s0=s0)
            else:
                assert not K.ROPE_PAIR, 'PIKA_ROPE_PAIR needs the fused LoRA rope path (PIKA_FUSE_LORA=1, uniform heads)'
                qkv = lora_linear(h, b.qkv, part_n=INNER, name='qkv', s=s)
                if s0 is not None:
                    send = K.qk_rope_pack(qkv, b.qn, b.kn, cos, sin, w_, self.qk_eps, s=s, s0=s0)
                elif self.uniform:
                    send = K.qk_rope_pack(qkv, b.qn, b.kn, cos, sin, w_, self.qk_eps, s=s)
                else:
                    send = K.qk_rope_pack_var(qkv.contiguous() if qkv.stride(0) != qkv.shape[1] else qkv, b.qn, b.kn, cos, sin,
                                              self._tables, self.qk_eps, s=s)
            if s0 is not None and self.sdma is not None and self.sdma_on:
                o = self._attn_split_sdma(send, sizes, s, s0, Mp)
            elif s0 is not None:
                o = self._attn_split(send, sizes, s, s0, Mp)
            else:
                o = self._attn_packed(send, sizes, s, Mp)
            if FUSE_LORA2:
                x, h = K.mod_norm_lora(x, b.n2, sc2, sh2, adaln_idx, self.eps, *lora_parts_(o, b.o), g1, pad=Mp)
            else:
                y_o = lora_linear(o, b.o, name='o', s=s)
                x, h = K.mod_norm(x, b.n2, sc2, sh2, adaln_idx, self.eps, y=y_o, gate_tab=g1, pad=Mp)
            if FUSE_LORA:
                y_, t_, B_ = lora_parts_(h, b.up)
                if HIP_SWIGLU:
                    from . import hipk
                    a_ = hipk.swiglu_lora_v3(y_, t_, B_, s=s, pad=Mp, cw=512, rs=2)
                else:
                    a_ = K.swiglu_lora(y_, t_, B_, s=s, pad=Mp)
            else:
                up = lora_linear(h, b.up, name='up', s=s)
                a_ = K.swiglu(up, s=s, pad=Mp)
            ff = lora_parts_(a_, b.down) if FUSE_LORA2 else lora_linear(a_, b.down, name='down', s=s)
            g2_prev = g2
        if FUSE_LORA2:
            ff = K.lora_add_(*ff, M=s)
        x = K.resid(x, ff, g2_prev, adaln_idx)
        self.sdma_on = sdma_req
        self.pipe = pipe_req

        no = m.norm_out
        sh, sc = tt["out_sh"], tt["out_sc"]
        ti = tt["ti_tab"][st.seg]
        hN = no.norm(x)
        hN = hN * (1.0 + sc.index_select(0, ti)) + sh.index_select(0, ti)
        hN = hN.to(get_parameter_dtype(m.proj_out))
        lo_a, hi_a = max(a0, L), min(a1, L + Na)
        lo_v, hi_v = max(a0, L + Na), min(a1, N)
        vo = m.proj_out(hN[lo_v - a0:hi_v - a0]) if hi_v > lo_v else hN.new_zeros((0, m.proj_out.out_features))
        ao = m.audio_proj_out(hN[lo_a - a0:hi_a - a0]) if hi_a > lo_a else hN.new_zeros((0, m.audio_proj_out.out_features))
        return self._gather_variable(vo, st.nv_list), self._gather_variable(ao, st.na_list)

    def _gather_variable(self, x, counts):
        mx = max(max(counts), 1)
        pad = x.new_zeros((mx,) + tuple(x.shape[1:]))
        pad[: x.shape[0]] = x
        out = [torch.empty_like(pad) for _ in range(self.world)]
        dist.all_gather(out, pad, group=self.group)
        return torch.cat([o[:c] for o, c in zip(out, counts)], dim=0)


_STEPLOG = os.environ.get("PIKA_STEPLOG") == "1"
GC_LOG = []  # dev (PIKA_STEPLOG=1): (generation, seconds) of every Python GC run in this process since the last report
if _STEPLOG:
    import gc as _gc
    import time as _gtm
    _gc_t0 = [0.0]

    def _gc_cb(phase, info):
        if phase == "start":
            _gc_t0[0] = _gtm.perf_counter()
        else:
            GC_LOG.append((info.get("generation"), _gtm.perf_counter() - _gc_t0[0], info.get("collected")))
    _gc.callbacks.append(_gc_cb)


def steplog_report(sl):
    """dev (PIKA_STEPLOG=1): per-step host/GPU times of one request on this rank, after the caller synchronized."""
    if not sl or sl[0][0] != "base":
        return None
    _, hb, eb = sl[0]
    out = []
    for step, N, h0, h1, e0, e1 in sl[1:]:
        out.append({"i": step, "N": N, "host_start_ms": round((h0 - hb) * 1e3, 2), "host_ms": round((h1 - h0) * 1e3, 2),
                    "gpu_start_ms": round(eb.elapsed_time(e0), 2), "gpu_ms": round(e0.elapsed_time(e1), 2)})
    return out


class SPWrapper(nn.Module):
    """t2va generator: same contract as H3DiffusionWrapper.forward (x0 prediction), sequence-parallel inside."""

    def __init__(self, sp: SPTransformer):
        super().__init__()
        self.sp = sp

    @torch.no_grad()
    def forward(self, noisy_image_or_video, conditional_dict, timestep, noisy_audio=None, audio_timestep=None, **_):
        from .core import geometry as G
        from .core.packing import (audio_external_to_rows, audio_rows_to_external, video_external_to_rows,
                                   video_rows_to_external)
        video, audio = noisy_image_or_video, noisy_audio
        B, F_, C, Hh, Ww = video.shape
        T = audio.shape[1]
        sigma_v = float(timestep.detach().float().reshape(-1)[0])
        sigma_a = float(audio_timestep.detach().float().reshape(-1)[0])
        _sl = _STEPLOG and conditional_dict.get("_steplog")  # dev: per-step event timing (non-synchronizing), default off
        if _sl is not None and _sl is not False and not _sl:
            import time as _tm
            _sl.append(("base", _tm.perf_counter(), torch.cuda.Event(enable_timing=True)))
            _sl[-1][2].record()
        st = conditional_dict.get("_sp_state")
        if st is None:
            st = self.sp.prepare(conditional_dict["prompt_embeds"], conditional_dict["text_token_tags"].reshape(-1),
                                 conditional_dict["geometry"])
            conditional_dict["_sp_state"] = st
        tvr = video_external_to_rows(video.float())
        tar = audio_external_to_rows(audio.float())
        step = conditional_dict.get("_tf_step", getattr(st, 'step', 0))  # _tf_step: dev teacher-forced eval only
        st.step = step + 1
        prof = None
        occ = getattr(self, "_occ", {})
        self._occ = occ
        if step == 0:
            occ[st.N] = occ.get(st.N, -1) + 1
        want = os.environ.get("PIKA_PROF_OCC")  # dev: profile these occurrences (0-based) of each sequence length
        if want is not None:
            ok_occ = str(occ.get(st.N, 0)) in want.split(",")
            key = (st.N, occ.get(st.N, 0))
        else:
            ok_occ, key = True, st.N
        if os.environ.get("PIKA_PROF_STEP") == str(step) and ok_occ and key not in getattr(self, "_prof_done", set()):
            from torch.profiler import profile, ProfilerActivity
            torch.cuda.synchronize()
            prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])
            prof.__enter__()
        tdev = os.environ.get("PIKA_TIMING") == "2"  # dev: per-forward wall time (synchronizing)
        if tdev:
            import time as _time
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()
        if _sl:
            import time as _tm
            _sh0 = _tm.perf_counter()
            _se0 = torch.cuda.Event(enable_timing=True)
            _se0.record()
        v_video, v_audio = self.sp(st, tvr[0], tar[0], 1.0 - sigma_v, 1.0 - sigma_a, step=step)
        if _sl:
            _se1 = torch.cuda.Event(enable_timing=True)
            _se1.record()
            _sl.append((step, st.N, _sh0, _tm.perf_counter(), _se0, _se1))
        if tdev:
            torch.cuda.synchronize()
            if self.sp.rank == 0:
                print(f"[timing] step {step} N {st.N} forward {(_time.perf_counter() - _t0) * 1e3:.1f} ms", flush=True)
        if prof is not None:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            self._prof_done = getattr(self, "_prof_done", set()) | {key}
            pd = f"{os.environ.get('PIKA_PROF_DIR') or __import__('tempfile').gettempdir()}/N{st.N}" + (f"_{key[1]}" if want is not None else "")
            os.makedirs(pd, exist_ok=True)
            prof.export_chrome_trace(f"{pd}/prof_rank{self.sp.rank}.json")
        x0 = tvr + float(sigma_v) * v_video.unsqueeze(0).float()
        video_x0 = video_rows_to_external(x0, F_, Hh, Ww).to(video.dtype)
        a0 = tar + float(sigma_a) * v_audio.unsqueeze(0).float()
        audio_x0 = audio_rows_to_external(a0, T).to(audio.dtype)
        return video_x0, audio_x0
