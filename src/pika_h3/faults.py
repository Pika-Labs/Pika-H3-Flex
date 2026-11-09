# SPDX-License-Identifier: Apache-2.0
"""Failure semantics of the multi-GPU worker, and a default-off fault-injection hook for testing them.

A request that fails is either
  * REJECTED: every rank agrees on the failure at the same point of the request (validation on rank 0 before the
    broadcast, or a conditioning failure on rank 0 that is broadcast to all ranks as ``RequestRejected``). The ranks stay
    in step; the worker answers ``ok: false`` and keeps serving; or
  * FATAL: anything else that fails while the ranks are inside the collective part of a request (an exception or out of
    memory on one rank, a helper rank that dies, a failed collective). The ranks can no longer be trusted to be in step,
    so the worker answers ``ok: false, "fatal": true`` for the request in flight (if any), prints
    ``EAT_FATAL {...}``, kills its helper ranks and exits with code ``FATAL_EXIT_CODE`` within seconds (instead of
    hanging until the NCCL timeout). A process manager restarts it.
  Failures on rank 0 after the last collective of a request (MP4 mux, trace write) are local: ok: false, keep serving.

Fault injection (testing only; unset = no code path changes):
  PIKA_FAULT="<rank>:<point>[:<n>]"   raise ``InjectedFault`` on rank <rank> in the <n>-th request (default 1) at <point>:
     step0 .. step5   before generator evaluation i      decode   before the video tile decode
     post             rank 0 only, after the MP4 is written (a local failure such as the trace write: keep serving)
     oom              like step2, but raises torch.OutOfMemoryError
"""
import os

FATAL_EXIT_CODE = 3


class RequestRejected(RuntimeError):
    """A failure that every rank agreed on at the same point of the request (the ranks stay in step)."""


class InjectedFault(RuntimeError):
    pass


def _parse(spec):
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"PIKA_FAULT={spec!r}: expected <rank>:<point>[:<n>]")
    rank, point, n = int(parts[0]), parts[1], int(parts[2]) if len(parts) == 3 else 1
    ok = {f"step{i}" for i in range(6)} | {"decode", "post", "oom"}
    if point not in ok:
        raise ValueError(f"PIKA_FAULT={spec!r}: point must be one of {sorted(ok)}")
    return rank, point, n


FAULT = _parse(os.environ.get("PIKA_FAULT"))
_count = {"n": 0}


def new_request():
    """Called once per request on every rank (only when PIKA_FAULT is set)."""
    _count["n"] += 1


def armed(rank, point):
    return FAULT is not None and FAULT[0] == rank and FAULT[2] == _count["n"] and FAULT[1] == point


def fire(rank, point):
    if armed(rank, point):
        msg = f"injected fault (PIKA_FAULT) on rank {rank} at {point} of request {_count['n']}"
        if point == "oom":
            import torch
            raise torch.OutOfMemoryError(msg)
        raise InjectedFault(msg)


class FaultyGenerator:
    """Generator wrapper that raises at the chosen evaluation (installed only when PIKA_FAULT targets this rank)."""

    def __init__(self, gen, rank):
        self.gen, self.rank, self.i = gen, rank, 0

    def __call__(self, **kw):
        fire(self.rank, f"step{self.i}")
        if self.i == 2:
            fire(self.rank, "oom")
        self.i += 1
        return self.gen(**kw)
