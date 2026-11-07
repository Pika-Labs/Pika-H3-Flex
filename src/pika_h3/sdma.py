# SPDX-License-Identifier: Apache-2.0
"""CU-free peer transfers on the SDMA copy engines (PIKA_SDMA=1).

RCCL's all-to-all runs as kernels on the compute units (~32 CTAs for link bandwidth), so a concurrent all-to-all slows the
attention it is meant to hide behind. Here the bytes move on the GPU's SDMA engines (hsa_amd_memory_async_copy_on_engine,
one engine per peer link) straight into IPC-mapped receive buffers of the peer ranks; no kernel is involved.

Ordering for one exchange (sequence number q, the same on every rank because all ranks call exchange() in the same order):
  sender stream:   ... producer kernels ... | writeValue(ready := 0)
  sender SDMA:     poll(ready == 0) -> data copies -> (data done) -> 8-byte copy of q into peer.flags[sender]
  receiver stream: waitValue(flags[p] >= q) for every peer p; waitValue(own data copies done) (source buffers reusable)
Flags are monotonic, so nothing is ever reset. Data are moved bit-exactly (pure copies).
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

import torch
import torch.distributed as dist

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "sdma_shim.cpp")
_L = None
WAIT_KERNEL = os.environ.get("PIKA_SDMA_WAITK", "1") == "1"


def _build():
    src = open(_SRC, "rb").read()
    tag = hashlib.sha1(src).hexdigest()[:12]
    from .paths import cache_dir
    d = cache_dir("sdma", env="PIKA_SDMA_BUILD")
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f"libpika_sdma_{tag}.so")
    if not os.path.exists(so):
        tmp = f"{so}.{os.getpid()}.tmp"
        hipcc = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc")
        subprocess.check_call([hipcc, "-O2", "-shared", "-fPIC", _SRC, "-lhsa-runtime64", "-o", tmp],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(tmp, so)
    return so


def lib():
    global _L
    if _L is None:
        L = ctypes.CDLL(_build())
        u64, vp, i64 = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int64
        L.sd_agent.restype = u64
        L.sd_agent.argtypes = [vp]
        L.sd_engines.restype = ctypes.c_uint32
        L.sd_engines.argtypes = [u64, u64]
        L.sd_preferred.restype = ctypes.c_uint32
        L.sd_preferred.argtypes = [u64, u64]
        L.sd_signal.restype = u64
        L.sd_signal_set.argtypes = [u64, i64]
        L.sd_signal_load.restype = i64
        L.sd_signal_load.argtypes = [u64]
        L.sd_signal_addr.restype = vp
        L.sd_signal_addr.argtypes = [u64]
        L.sd_wait.restype = i64
        L.sd_wait.argtypes = [u64]
        L.sd_copy.argtypes = [vp, u64, vp, u64, ctypes.c_size_t, u64, ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(u64)]
        L.sd_ipc_get.argtypes = [vp, vp]
        L.sd_ipc_open.restype = vp
        L.sd_ipc_open.argtypes = [vp]
        L.sd_ipc_close.argtypes = [vp]
        L.sd_stream_write.argtypes = [vp, vp, i64]
        L.sd_stream_wait_ge.argtypes = [vp, vp, i64]
        L.sd_stream_wait_eq0.argtypes = [vp, vp]
        L.sd_wait_kernel.argtypes = [vp, ctypes.c_int, ctypes.POINTER(u64), ctypes.POINTER(i64), ctypes.POINTER(ctypes.c_int32)]
        assert L.sd_init() == 0
        _L = L
    return _L


_HIP = None


def _hip():
    global _HIP
    if _HIP is None:
        _HIP = ctypes.CDLL("libamdhip64.so")
    return _HIP


class _Raw:
    def __init__(self, ptr, nb, dev):
        self.__cuda_array_interface__ = {"shape": (nb,), "typestr": "|u1", "data": (ptr, False), "version": 3, "strides": None}


def _wrap(ptr, nb, dev):
    with torch.cuda.device(dev):
        t = torch.as_tensor(_Raw(ptr, nb, dev), device=dev)
    assert t.data_ptr() == ptr
    return t


def _alloc_base(ptr):
    hip = ctypes.CDLL("libamdhip64.so")
    base, sz = ctypes.c_void_p(), ctypes.c_size_t()
    assert hip.hipMemGetAddressRange(ctypes.byref(base), ctypes.byref(sz), ctypes.c_void_p(ptr)) == 0
    return base.value


class SdmaComm:
    RING = 512

    def __init__(self, rank, world, group=None):
        L = lib()
        self.rank, self.world, self.group = rank, world, group
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.flags = torch.zeros(world * 8, dtype=torch.int64, device=self.dev)  # flags[src * 8] (64 B apart)
        self.seqsrc = torch.arange(1 << 22, dtype=torch.int64, device=self.dev)  # source of the 8-byte flag copies
        self.local_agent = L.sd_agent(ctypes.c_void_p(self.flags.data_ptr()))
        self.bufs, self.cap, self._raw = [], 0, []
        self.peer_bufs, self.peer_flags, self.peer_agents, self.engine = {}, {}, {}, {}
        self._opened = []
        hs, offs = self._handles([self.flags])
        allh = [None] * world
        dist.all_gather_object(allh, (hs, offs), group=group)
        for p in range(world):
            if p == rank:
                continue
            ptr = self._open(allh[p][0][0], allh[p][1][0])
            self.peer_flags[p] = ptr
            self.peer_agents[p] = L.sd_agent(ctypes.c_void_p(ptr))
            m = L.sd_preferred(self.peer_agents[p], self.local_agent) or L.sd_engines(self.peer_agents[p], self.local_agent)
            assert m, f"no SDMA engine {rank}->{p}"
            self.engine[p] = m & -m
        self.seq = 0
        self.ready = [L.sd_signal() for _ in range(self.RING)]
        self.dsig = [[L.sd_signal() for _ in range(world)] for _ in range(self.RING)]
        self.fsig = [[L.sd_signal() for _ in range(world)] for _ in range(self.RING)]
        self.used = [False] * self.RING
        self.evs = [None] * self.RING  # event after the stream's waits of slot k (host must not re-arm its signals before)

    def _handles(self, ts):
        L = lib()
        hs, offs = [], []
        for t in ts:
            b = ctypes.create_string_buffer(64)
            assert L.sd_ipc_get(ctypes.c_void_p(t.data_ptr()), b) == 0
            hs.append(b.raw)
            offs.append(t.data_ptr() - _alloc_base(t.data_ptr()))
        return hs, offs

    def _open(self, h, off):
        base = lib().sd_ipc_open(ctypes.create_string_buffer(h, 64))
        assert base, "hipIpcOpenMemHandle failed"
        self._opened.append(base)
        return base + off

    def allocate(self, nbytes_list):
        """Collective, ONCE: the symmetric receive buffers (own hipMalloc allocations; re-allocating and re-exporting IPC handles
        failed on this stack: hipIpcGetMemHandle 'invalid argument' after hipFree + hipMalloc, so capacity is fixed up front)."""
        assert not self.bufs, "SDMA buffers are allocated once"
        for n in nbytes_list:
            nb = (int(n) + (1 << 20)) // 4096 * 4096
            ptr = ctypes.c_void_p()
            assert _hip().hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(nb)) == 0, "hipMalloc failed"
            self._raw.append(ptr.value)
            self.bufs.append(_wrap(ptr.value, nb, self.dev))
        hs, offs = self._handles(self.bufs)
        allh = [None] * self.world
        dist.all_gather_object(allh, (hs, offs), group=self.group)
        for p in self.peer_flags:
            self.peer_bufs[p] = [self._open(h, o) for h, o in zip(*allh[p])]
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def fits(self, nbytes_list):
        return len(self.bufs) == len(nbytes_list) and all(b.numel() >= n for b, n in zip(self.bufs, nbytes_list))

    def ensure(self, nbytes_list):
        assert self.fits(nbytes_list), "request exceeds the SDMA buffer capacity (callers check fits() first)"

    def view(self, i, dtype, shape):
        n = 1
        for x in shape:
            n *= x
        return self.bufs[i].view(dtype)[:n].view(shape)

    def post(self, sends, buf_idx, stream=None):
        """sends: {peer: [(src_ptr, nbytes, dst_byte_offset in the peer's buffer buf_idx), ...]} (every peer present,
        possibly with an empty list). Copies start once the work queued so far on `stream` is done. Returns a token for wait()."""
        L = lib()
        stream = stream or torch.cuda.current_stream()
        sp = ctypes.c_void_p(stream.cuda_stream)
        self.seq += 1
        seq = self.seq
        assert seq < self.seqsrc.numel()
        k = seq % self.RING
        if self.used[k]:
            if self.evs[k] is not None:
                self.evs[k].synchronize()
            for p in self.peer_flags:
                L.sd_wait(self.fsig[k][p])
        self.used[k] = True
        rdy = self.ready[k]
        L.sd_signal_set(rdy, 1)
        assert L.sd_stream_write(sp, ctypes.c_void_p(L.sd_signal_addr(rdy)), 0) == 0
        dep = (ctypes.c_uint64 * 1)(rdy)
        for p in self.peer_flags:
            lst = [x for x in sends.get(p, ()) if x[1] > 0]
            eng = self.engine[p]
            dst = self.peer_bufs[p][buf_idx]
            ds = self.dsig[k][p]
            L.sd_signal_set(ds, len(lst))
            for (src, n, off) in lst:
                assert L.sd_copy(ctypes.c_void_p(dst + off), self.peer_agents[p], ctypes.c_void_p(src), self.local_agent, n, ds,
                                 eng, 1, dep) == 0
            fs = self.fsig[k][p]
            L.sd_signal_set(fs, 1)
            d2 = (ctypes.c_uint64 * 2)(ds, rdy)
            assert L.sd_copy(ctypes.c_void_p(self.peer_flags[p] + self.rank * 64), self.peer_agents[p],
                             ctypes.c_void_p(self.seqsrc.data_ptr() + 8 * seq), self.local_agent, 8, fs, eng, 2, d2) == 0
        return (seq, k)

    def wait(self, tok, stream=None):
        """`stream` waits for the incoming data of every peer and for the completion of its own outgoing copies."""
        L = lib()
        stream = stream or torch.cuda.current_stream()
        sp = ctypes.c_void_p(stream.cuda_stream)
        seq, k = tok
        fbase = self.flags.data_ptr()
        if WAIT_KERNEL:  # one spin kernel for all conditions (HIP's wait-value packets are one blit kernel each)
            peers = list(self.peer_flags)
            n = 2 * len(peers)
            addr = (ctypes.c_uint64 * n)(*([fbase + p * 64 for p in peers] + [L.sd_signal_addr(self.dsig[k][p]) for p in peers]))
            val = (ctypes.c_int64 * n)(*([seq] * len(peers) + [0] * len(peers)))
            eq = (ctypes.c_int32 * n)(*([0] * len(peers) + [1] * len(peers)))
            assert L.sd_wait_kernel(sp, n, addr, val, eq) == 0
        else:
            for p in self.peer_flags:
                assert L.sd_stream_wait_ge(sp, ctypes.c_void_p(fbase + p * 64), seq) == 0
            for p in self.peer_flags:
                assert L.sd_stream_wait_eq0(sp, ctypes.c_void_p(L.sd_signal_addr(self.dsig[k][p]))) == 0
        ev = torch.cuda.Event()
        ev.record(stream)
        self.evs[k] = ev
