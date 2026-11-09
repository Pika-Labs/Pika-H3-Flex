# SPDX-License-Identifier: Apache-2.0
"""Hand-written HIP kernels, built once with hipcc into a cache dir and called through ctypes on
torch's current stream. Sources: pika_h3/hip/*.hip."""
import ctypes
import hashlib
import os
import subprocess

import torch

_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hip")
from .paths import cache_dir as _cache_dir

_CACHE = _cache_dir("hipk", env="PIKA_HIP_CACHE")
_LIBS = {}


def _lib(name):
    if name in _LIBS:
        return _LIBS[name]
    src = os.path.join(_HERE, name + ".hip")
    h = hashlib.sha1(open(src, "rb").read() + os.environ.get("PIKA_HIPCC_FLAGS", "").encode()).hexdigest()[:12]
    os.makedirs(_CACHE, exist_ok=True)
    so = os.path.join(_CACHE, f"{name}_{h}.so")
    if not os.path.exists(so):
        tmp = so + f".tmp{os.getpid()}"
        subprocess.run([os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc"), "-O3"] + os.environ.get("PIKA_HIPCC_FLAGS", "").split() + ["--offload-arch=gfx950", "-shared", "-fPIC", "-o", tmp, src], check=True)
        os.replace(tmp, so)
    _LIBS[name] = ctypes.CDLL(so)
    return _LIBS[name]


def swiglu_lora(up, t, B, s=None, pad=None, cw=1024):
    """== kernels.swiglu_lora (same rounding points). up [Mp, 2FF] bf16, t [Mp, 128], B [2FF, 128] contiguous."""
    F2 = up.shape[1]
    FF = F2 // 2
    s = up.shape[0] if s is None else s
    assert B.is_contiguous() and B.shape[1] == 128 and up.stride(1) == 1 and t.stride(1) == 1 and FF % cw == 0
    out = torch.empty((pad or s, FF), dtype=up.dtype, device=up.device)
    L = _lib("swiglu_lora")
    r = L.swiglu_lora_launch(ctypes.c_void_p(up.data_ptr()), ctypes.c_void_p(t.data_ptr()), ctypes.c_void_p(B.data_ptr()),
                             ctypes.c_void_p(out.data_ptr()), s, FF, up.stride(0), t.stride(0), out.stride(0), cw,
                             ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"swiglu_lora_launch failed: {r}")
    return out


def swiglu_lora_v2(up, t, B, s=None, pad=None, cw=1024, rs=2):
    """v2 (transposed MFMA, 32-byte row accesses); same contract as swiglu_lora. Needs 32-byte aligned rows."""
    F2 = up.shape[1]
    FF = F2 // 2
    s = up.shape[0] if s is None else s
    assert B.is_contiguous() and B.shape[1] == 128 and up.stride(1) == 1 and t.stride(1) == 1 and FF % cw == 0
    out = torch.empty((pad or s, FF), dtype=up.dtype, device=up.device)
    assert up.data_ptr() % 32 == 0 and (up.stride(0) * 2) % 32 == 0 and out.data_ptr() % 32 == 0 and (out.stride(0) * 2) % 32 == 0
    L = _lib("swiglu_lora")
    r = L.swiglu_lora_v2_launch(ctypes.c_void_p(up.data_ptr()), ctypes.c_void_p(t.data_ptr()), ctypes.c_void_p(B.data_ptr()),
                                ctypes.c_void_p(out.data_ptr()), s, FF, up.stride(0), t.stride(0), out.stride(0), cw, rs,
                                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"swiglu_lora_v2_launch failed: {r}")
    return out


def swiglu_lora_v3(up, t, B, s=None, pad=None, cw=1024, rs=2):
    """v3 (v2 + LDS-staged LoRA-B tiles); same contract as swiglu_lora."""
    F2 = up.shape[1]
    FF = F2 // 2
    s = up.shape[0] if s is None else s
    assert B.is_contiguous() and B.shape[1] == 128 and up.stride(1) == 1 and t.stride(1) == 1 and FF % cw == 0
    out = torch.empty((pad or s, FF), dtype=up.dtype, device=up.device)
    assert up.data_ptr() % 32 == 0 and (up.stride(0) * 2) % 32 == 0 and out.data_ptr() % 32 == 0 and (out.stride(0) * 2) % 32 == 0
    r = _lib("swiglu_lora").swiglu_lora_v3_launch(ctypes.c_void_p(up.data_ptr()), ctypes.c_void_p(t.data_ptr()),
                                                  ctypes.c_void_p(B.data_ptr()), ctypes.c_void_p(out.data_ptr()), s, FF, up.stride(0),
                                                  t.stride(0), out.stride(0), cw, rs, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"swiglu_lora_v3_launch failed: {r}")
    return out


def rope_pair(qkv, t, B, qn, kn, cos, sin, world, eps, NH=56, HD=128, s=None, s0=None, rows=None, out=None, rs=None):
    """HIP version of kernels.qk_rope_pack_lora_pair (pika_h3/hip/rope_pair.hip); same contract."""
    rs = rs or int(os.environ.get("PIKA_ROPE_HIP_RS", "2"))
    s = qkv.shape[0] if s is None else s
    hpr = NH // world
    assert HD == 128 and B.is_contiguous() and B.shape[1] == 128 and cos.is_contiguous() and sin.is_contiguous()
    assert qkv.stride(1) == 1 and t.stride(1) == 1 and qkv.stride(0) % 4 == 0 and t.stride(0) % 8 == 0 and cos.shape[1] % 2 == 0
    if out is None:
        out = torch.empty((world * s, 3 * hpr * HD), dtype=qkv.dtype, device=qkv.device)
    r0, r1 = rows if rows is not None else (0, s)
    r = _lib("rope_pair").rope_pair_launch(
        ctypes.c_void_p(qkv.data_ptr()), ctypes.c_void_p(t.data_ptr()), ctypes.c_void_p(B.data_ptr()), ctypes.c_void_p(qn.data_ptr()),
        ctypes.c_void_p(kn.data_ptr()), ctypes.c_void_p(cos.data_ptr()), ctypes.c_void_p(sin.data_ptr()), ctypes.c_void_p(out.data_ptr()),
        s, ctypes.c_float(eps), t.stride(0), qkv.stride(0), 0 if s0 is None else s0, int(s0 is not None), r0, r1, NH, hpr, cos.shape[1], rs,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if r != 0:
        raise RuntimeError(f"rope_pair_launch failed: {r}")
    return out
