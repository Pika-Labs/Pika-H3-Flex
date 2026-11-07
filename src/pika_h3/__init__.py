# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Pika H3 Flex inference."""
__version__ = "0.1.0"

# ROCm 7.2 / torch 2.13: with rocprofiler-sdk registered (it is loaded into every torch process by kineto) the HSA runtime wraps the
# AQL queues in an InterceptQueue, and after some 10^5 kernel dispatches in a busy process the dispatch path segfaults in
# rocr::core::InterceptQueue::StoreRelaxed (reproduced with the tile decoder alone: world 1, 10 s decode dies at tile batch ~150-270,
# a rank death that hangs/kills the whole 8-rank request). Disabling the registration removes the intercept queue. Must be set before
# the HIP runtime is loaded, i.e. before ``import torch``; child ranks inherit it. Profiling runs (PIKA_PROF_STEP) keep it enabled.
import os as _os

if not _os.environ.get("PIKA_PROF_STEP"):
    _os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")

# Default (PIKA_HOST_MEMPOLICY=bind; =off disables): take this process and its rank children out of automatic NUMA balancing, whose hinting faults
# make the kernel driver evict GPU queues (see hosthealth.py). Must run before the large host allocations; children inherit it.
from . import hosthealth as _hh  # noqa: E402  (stdlib only)

_hh.apply_mempolicy()
# RCCL's default P2P channel count gives the Ulysses all-to-all (all_to_all_single = grouped send/recv to every peer) about one channel per
# peer at world 7/8: 9 ms per call at world 7 (qkv, N = 38k) vs 0.75 ms with >= 48 channels (measured; 12x). Must be set
# before the first NCCL communicator is created (children inherit). World 4 is unaffected (2.13 ms either way).
_os.environ.setdefault("NCCL_MIN_P2P_NCHANNELS", "64")
_os.environ.setdefault("NCCL_MAX_P2P_NCHANNELS", "64")
