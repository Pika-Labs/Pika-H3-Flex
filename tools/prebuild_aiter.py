# SPDX-License-Identifier: Apache-2.0
"""Pre-build the ROCm aiter JIT modules the worker uses, without a GPU (e.g. while building a deployment image).

Without this step aiter builds each module the first time it is used, i.e. during the first worker start (default path) or
the first request of a lossy preset. Building ahead makes the first start fast and keeps compilers out of the serving path.

    PIKA_AITER_DIR=/path/to/aiter python tools/prebuild_aiter.py            # default-path modules
    PIKA_AITER_DIR=/path/to/aiter python tools/prebuild_aiter.py --lossy    # + the lossy presets' modules
    python tools/prebuild_aiter.py --list                                   # what is built already

Builds target gfx950 (GPU_ARCHS, default gfx950) into <aiter>/aiter/jit/. Needs ROCm's hipcc, ninja and pybind11.
"""
import argparse
import os
import sys
import time

DEFAULT_MODULES = ["module_aiter_core", "module_fmha_v4_fwd", "module_hipbsolgemm"]
LOSSY_MODULES = ["module_gemm_a6w6_asm", "module_quant", "module_mha_v4_quant"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lossy", action="store_true", help="also build the modules of the lossy presets")
    ap.add_argument("--list", action="store_true", help="only list the built modules")
    ap.add_argument("modules", nargs="*", help="explicit module names (default: the worker's modules)")
    a = ap.parse_args()
    os.environ.setdefault("GPU_ARCHS", "gfx950")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from pika_h3.paths import AITER_DIR, AITER_SOURCE, ensure_aiter
    ensure_aiter()
    from aiter.jit import core
    jit_dir = core.get_user_jit_dir()
    print(f"aiter: {AITER_DIR} ({AITER_SOURCE}); JIT dir {jit_dir}; GPU_ARCHS={os.environ['GPU_ARCHS']}")
    mods = a.modules or DEFAULT_MODULES + (LOSSY_MODULES if a.lossy else [])
    if a.list:
        for m in mods:
            print(f"  {m:28s} {'built' if os.path.exists(os.path.join(jit_dir, m + '.so')) else 'missing'}")
        return 0
    for m in mods:
        so = os.path.join(jit_dir, m + ".so")
        if os.path.exists(so):
            print(f"  {m:28s} already built")
            continue
        t0 = time.time()
        d = core.get_args_of_build(m)
        core.build_module(m, d["srcs"], d["flags_extra_cc"], d["flags_extra_hip"], d["blob_gen_cmd"], d["extra_include"],
                          d["extra_ldflags"], d["verbose"], d["is_python_module"], d["is_standalone"], d["torch_exclude"],
                          d.get("third_party", []), flags_extra_hip_per_source=d.get("flags_extra_hip_per_source", {}))
        print(f"  {m:28s} built in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
