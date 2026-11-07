# SPDX-License-Identifier: Apache-2.0
"""Compare two PIKA_TEST_OUT directories of tests/test_world2.py (e.g. this checkout vs a reference checkout):
every traced request bit-identical (all trace tensors) and every MP4 byte-identical (sha256).

    python -m tests.compare_runs <out_a> <out_b>      exit code 0 = identical
"""
import json
import sys
from pathlib import Path

from tests.worker_client import traces_equal


def main(a, b):
    a, b = Path(a), Path(b)
    sa, sb = (json.loads((d / "summary.json").read_text())["requests"] for d in (a, b))
    bad = 0
    for k in sorted(set(sa) & set(sb)):
        line = f"{k:14s} sha256 {'same' if sa[k].get('sha256') == sb[k].get('sha256') else 'DIFF'}"
        if sa[k].get("sha256") != sb[k].get("sha256"):
            bad += 1
        ta, tb = a / f"{k}_trace" / "trace.safetensors", b / f"{k}_trace" / "trace.safetensors"
        if ta.exists() and tb.exists():
            eq, diff = traces_equal(ta, tb)
            line += f"  trace {'bit-identical' if eq else 'DIFF at ' + str(diff)}"
            bad += 0 if eq else 1
        print(line)
    print("IDENTICAL" if bad == 0 else f"{bad} DIFFERENCES")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
