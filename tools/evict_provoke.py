# SPDX-License-Identifier: Apache-2.0
"""Compaction provoker (bounded): repeatedly maps GB-sized anonymous regions with MADV_HUGEPAGE and touches them, which makes the
kernel compact memory (THP defrag=madvise -> direct compaction) and migrate other processes' pages. Usage: python tools/evict_provoke.py GB SECONDS
(GB of host RAM per cycle; make sure that much is free.)"""
import mmap, sys, time
GB, T = int(sys.argv[1]), float(sys.argv[2])
t_end = time.time() + T
n = 0
while time.time() < t_end:
    m = mmap.mmap(-1, GB << 30, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    m.madvise(mmap.MADV_HUGEPAGE)
    for off in range(0, GB << 30, 1 << 21):
        m[off] = 1
    m.close()
    n += 1
print(f"[provoke] {n} cycles of {GB} GB", flush=True)
