# SPDX-License-Identifier: Apache-2.0
"""Host-side latency hazards on ROCm (stability work): KFD GPU-queue evictions.

The latency spikes seen on this machine (stability runs) are periods in which the kernel driver (KFD) EVICTS the
worker processes' GPU queues: every kernel of that rank stops until the queues are restored, and because the eight ranks run in
lockstep (collectives every layer) the whole request stalls. The evictions coincide with host memory-management bursts (kcompactd
compaction / page migration; NUMA balancing adds a small steady rate): the kernel moves host pages that the GPU has mapped, the MMU
notifier invalidates them and KFD must quiesce the process's queues while it revalidates. Nothing in the model code triggers it;
lossless and lossy paths are hit alike.

This module:
  * ``host_report()``: the host settings that drive it (kernel.numa_balancing, vm.compaction_proactiveness, THP defrag); the worker
    prints a warning at start-up if they are in the risky state (fix on the service host: see docs/PIKA_HANDOVER.md "Host settings", needs root);
  * ``EvictionMeter``: per-request KFD evicted time of every rank process, read from /sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms
    (no GPU work, no collective; ~1 ms), reported in the request timings as ``kfd_evicted_ms`` so a slow request can be attributed;
  * ``apply_mempolicy()``: DEFAULT ON (env PIKA_HOST_MEMPOLICY=bind, the default; =off disables) MPOL_BIND over all NUMA nodes for this process and its children, which
    takes the process out of automatic NUMA balancing (no hinting faults / NUMA migrations of its pages) without changing where
    memory is allocated (local node first). It does not stop compaction.
"""
from __future__ import annotations

import ctypes
import glob
import os


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def host_report():
    """{setting: value} and a list of warnings for the settings that let the kernel evict GPU queues during inference."""
    r = {"kernel.numa_balancing": _read("/proc/sys/kernel/numa_balancing"),
         "vm.compaction_proactiveness": _read("/proc/sys/vm/compaction_proactiveness"),
         "transparent_hugepage.defrag": _read("/sys/kernel/mm/transparent_hugepage/defrag")}
    warn = []
    if r["kernel.numa_balancing"] not in (None, "0"):
        warn.append("kernel.numa_balancing=%s (AMD recommends 0 for MI300-class GPUs: NUMA hinting faults evict GPU queues)"
                    % r["kernel.numa_balancing"])
    if r["vm.compaction_proactiveness"] not in (None, "0"):
        warn.append("vm.compaction_proactiveness=%s (background compaction migrates GPU-mapped host pages -> GPU queue evictions; "
                    "set 0 on inference hosts)" % r["vm.compaction_proactiveness"])
    d = r["transparent_hugepage.defrag"] or ""
    if "[always]" in d:
        warn.append("transparent_hugepage/defrag=always (direct compaction on page faults; use madvise or never)")
    return r, warn


def _pid_evicted_ms(pid):
    tot = 0
    for f in glob.glob(f"/sys/class/kfd/kfd/proc/{pid}/stats_*/evicted_ms"):
        v = _read(f)
        if v:
            try:
                tot += int(v.split()[0])
            except ValueError:
                pass
    return tot


class EvictionMeter:
    """Cumulative KFD evicted ms per rank process; ``delta()`` returns {rank: ms since the previous call}."""

    def __init__(self, pids):
        self.pids = list(pids)  # index = rank
        self.ok = os.path.isdir("/sys/class/kfd/kfd/proc")
        self.last = [self._get(p) for p in self.pids] if self.ok else []

    def _get(self, pid):
        return _pid_evicted_ms(pid) if pid else 0

    def delta(self):
        if not self.ok:
            return None
        cur = [self._get(p) for p in self.pids]
        d = {r: c - l for r, (c, l) in enumerate(zip(cur, self.last))}
        self.last = cur
        return d


def settle_evictions(meter, quiet_s=0.5, max_s=10.0, poll_s=0.05):
    """Start-up only (serve.Worker.ready_barrier): wait until no rank's cumulative KFD evicted time has grown for quiet_s (at most
    max_s), then re-base the meter so the first request reports only its own evictions. Returns {settle_s, evicted_ms, timed_out}."""
    import time
    t0 = time.monotonic()
    first = meter.delta() or {}
    tot = dict(first)
    if not meter.ok:
        return {"settle_s": 0.0, "evicted_ms": None, "timed_out": False}
    quiet0 = time.monotonic()
    while True:
        time.sleep(poll_s)
        d = meter.delta() or {}
        for r, v in d.items():
            tot[r] = tot.get(r, 0) + v
        now = time.monotonic()
        if any(d.values()):
            quiet0 = now
        if now - quiet0 >= quiet_s or now - t0 >= max_s:
            return {"settle_s": round(now - t0, 3), "evicted_ms": tot, "timed_out": now - quiet0 < quiet_s}


MPOL_BIND = 2


def mempolicy_mode():
    """PIKA_HOST_MEMPOLICY: 'bind' (default) or anything else (e.g. 'off') = leave the process's memory policy alone."""
    return os.environ.get("PIKA_HOST_MEMPOLICY", "bind")


def thp_enabled():
    """PIKA_HOST_THP: on by default (measured: 6-50x less KFD eviction time and 2-5 % lower latency on the normal host,
    bit-identical output); PIKA_HOST_THP=0 disables."""
    return os.environ.get("PIKA_HOST_THP", "1") != "0"


def apply_mempolicy():
    """PIKA_HOST_MEMPOLICY=bind (default): set_mempolicy(MPOL_BIND, all online nodes) for this process (inherited by children)."""
    mode = mempolicy_mode()
    if mode != "bind":
        return None
    nodes = []
    for p in glob.glob("/sys/devices/system/node/node[0-9]*"):
        try:
            nodes.append(int(os.path.basename(p)[4:]))
        except ValueError:
            pass
    if not nodes:
        return None
    mask = 0
    for n in nodes:
        mask |= 1 << n
    libc = ctypes.CDLL(None, use_errno=True)
    m = ctypes.c_ulong(mask)
    rc = libc.syscall(238, MPOL_BIND, ctypes.byref(m), ctypes.c_ulong(max(nodes) + 2))  # SYS_set_mempolicy (x86_64)
    return {"mode": "bind", "nodes": sorted(nodes), "rc": rc, "errno": ctypes.get_errno() if rc else 0}


# ----------------------------------------------------------------------------------------------------------------------------------
# Default on (env PIKA_HOST_THP=0 disables): back the process's resident anonymous memory with transparent huge pages (madvise MADV_HUGEPAGE +
# MADV_COLLAPSE, Linux >= 6.1). The compaction migrate scanner skips compound (THP) pages, so host pages the GPU has mapped are no
# longer moved by kcompactd -> no MMU-notifier invalidation -> no queue eviction from compaction. Run once after loading (and,
# with PIKA_HOST_THP_INCR=1, the earlier behaviour, incrementally after each request: off by default because collapsing
# GPU-mapped pages right after a request evicts the queues at the start of the next one). Errors are counted, never raised.
MADV_HUGEPAGE = 14
MADV_COLLAPSE = 25
_HP = 2 << 20
_thp_seen = set()


def _anon_rw_vmas():
    out = []
    with open("/proc/self/maps") as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 5 or not parts[1].startswith("rw"):
                continue
            path = parts[5] if len(parts) > 5 else ""
            if path and path not in ("[heap]",) and not path.startswith("[anon"):
                continue  # file / device mappings (incl. the KFD doorbell mappings, shared tmpfs): leave alone
            a, b = (int(x, 16) for x in parts[0].split("-"))
            out.append((a, b))
    return out


def thp_incremental():
    """PIKA_HOST_THP_INCR: whether thp_pass() also runs after every request (on new mappings) or only once, after loading /
    warm-up, before EAT_READY. Measurements showed that the per-request pass is what triggers the remaining KFD evictions: collapsing
    pages that the GPU has mapped makes the driver invalidate them and evict the rank's queues, and because the pass runs right
    after a request it overlaps the start of the next one (every remaining eviction started 0.1-0.8 s into a request;
    0.5-1.6 s on the first requests of new shapes). Default: off (load-time pass only; A/B at world 8: bit-identical,
    equal latency, timed eviction 30/53/0 ms vs 80/113/166 ms mean at 5/10/15 s); PIKA_HOST_THP_INCR=1 restores the per-request pass."""
    return os.environ.get("PIKA_HOST_THP_INCR", INCR_DEFAULT) != "0"


INCR_DEFAULT = "0"


def thp_pass(min_resident=0.5):
    """Collapse the 2 MB chunks of new anonymous rw mappings that are already >= min_resident resident (mincore), so sparse
    reservations are never populated. Returns stats or None when disabled."""
    if not thp_enabled():
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    pg = os.sysconf("SC_PAGE_SIZE")
    per = _HP // pg
    st = {"vmas": 0, "chunks": 0, "bytes": 0, "collapse_err": 0}
    for a, b in _anon_rw_vmas():
        if (a, b) in _thp_seen:
            continue
        _thp_seen.add((a, b))
        s_, e_ = (a + _HP - 1) // _HP * _HP, b // _HP * _HP
        if e_ - s_ < _HP:
            continue
        n = (e_ - s_) // pg
        vec = (ctypes.c_ubyte * n)()
        if libc.mincore(ctypes.c_void_p(s_), e_ - s_, vec) != 0:
            continue
        st["vmas"] += 1
        ranges = []
        for c in range(n // per):
            res = sum(vec[c * per + j] & 1 for j in range(0, per, 8)) * 8  # sample every 8th page
            if res >= min_resident * per:
                if ranges and ranges[-1][1] == c:
                    ranges[-1][1] = c + 1
                else:
                    ranges.append([c, c + 1])
        for c0, c1 in ranges:
            addr, ln = s_ + c0 * _HP, (c1 - c0) * _HP
            libc.madvise(ctypes.c_void_p(addr), ln, MADV_HUGEPAGE)
            if libc.madvise(ctypes.c_void_p(addr), ln, MADV_COLLAPSE) != 0:
                st["collapse_err"] += 1
            st["chunks"] += c1 - c0
            st["bytes"] += ln
    return st
