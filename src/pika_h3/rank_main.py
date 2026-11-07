# SPDX-License-Identifier: Apache-2.0
"""Helper rank process: ``python -m pika_h3.rank_main --rank R --world W --base-path P --adapter-path A``."""
import argparse
import os
import sys
import traceback


def main():
    import ctypes, signal
    ctypes.CDLL('libc.so.6').prctl(1, signal.SIGKILL)  # die with the parent
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--base-path", required=True)
    p.add_argument("--adapter-path", required=True)
    a = p.parse_args()
    from .engine import Engine, init_dist
    from .serve_loop import follower_loop
    ctl = init_dist(a.rank, a.world, a.device)
    eng = Engine(a.rank, a.world, a.base_path, a.adapter_path, ctl, a.device)
    try:
        follower_loop(eng)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
