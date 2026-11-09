# SPDX-License-Identifier: Apache-2.0
"""Follower-rank request loop."""
import os
import sys
import traceback

from . import hosthealth
from .faults import FATAL_EXIT_CODE, RequestRejected

_THP = hosthealth.thp_enabled() and hosthealth.thp_incremental()  # per-request THP pass: PIKA_HOST_THP + PIKA_HOST_THP_INCR (hosthealth.py)


def fatal_exit(rank):
    """Print the error and exit immediately (os._exit: no NCCL / atexit teardown that could block on the other ranks)."""
    try:
        traceback.print_exc()
        print(f"[serve] FATAL on rank {rank}: exiting (code {FATAL_EXIT_CODE}); the ranks are out of step", file=sys.stderr,
              flush=True)
    finally:
        os._exit(FATAL_EXIT_CODE)


def follower_loop(eng):
    import torch
    torch.cuda.synchronize()  # load-time copies done before the pass touches host memory
    st = hosthealth.thp_pass()
    if st:
        print(f"[serve] rank {eng.rank}: host THP pass {st}", flush=True)
    while True:
        msg = eng.bcast_obj(None)
        if msg is None or msg.get("cmd") == "exit":
            return
        if msg.get("cmd") == "sync":  # start-up ready barrier (serve.Worker.ready_barrier): this rank's pass is done
            eng.barrier()
            continue
        eng.phase = "collective"
        try:
            if msg["cmd"] == "tf":
                eng.teacher_forced(msg["req"], msg["path"])
                continue
            eng.run_request(msg["req"])
            if _THP:
                hosthealth.thp_pass()  # PIKA_HOST_THP (default on): new host mappings -> THP (between requests; see hosthealth.py)
        except RequestRejected:  # all ranks agreed on this failure at the same point: still in step
            traceback.print_exc()
        except BaseException:  # anything else: the ranks are out of step -> exit now; rank 0's watchdog sees it (faults.py)
            if getattr(eng, "phase", "collective") == "local":  # after the request's last collective: still in step
                traceback.print_exc()
                continue
            fatal_exit(eng.rank)
