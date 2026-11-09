# SPDX-License-Identifier: Apache-2.0
"""Machine-independent locations: the aiter checkout and the build/cache directories of the JIT-built kernels.

Nothing here depends on the machine the code was developed on. Every location can be set by an environment variable;
the defaults are inside the repository or the user's cache directory.

  PIKA_AITER_DIR   ROCm/aiter source checkout (the directory that contains the ``aiter`` package). Default: an
                   installed ``aiter`` package if Python can import it, else ``<repo>/third_party/aiter``.
  PIKA_CACHE_DIR   root of the kernel build caches (SDMA shim, HIP kernels, fa18). Default:
                   ``$XDG_CACHE_HOME/pika_h3`` or ``~/.cache/pika_h3``. Per-cache overrides: PIKA_SDMA_BUILD,
                   PIKA_HIP_CACHE, PIKA_FA18_DIR.
  PIKA_LOAD_LOCK_DIR  directory of the two host-RAM load-slot lock files (at most two ranks load weights at a time).
                   Default: PIKA_CACHE_DIR. Processes that should share the two slots must use the same directory.
"""
from __future__ import annotations

import importlib.util
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _resolve_aiter_dir() -> tuple[str, str]:
    env = os.environ.get("PIKA_AITER_DIR")
    if env:
        return env, "PIKA_AITER_DIR"
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        pkg = list(spec.submodule_search_locations)[0]
        return os.path.dirname(os.path.abspath(pkg)), "installed package"
    return os.path.join(REPO_ROOT, "third_party", "aiter"), "repository default"


AITER_DIR, AITER_SOURCE = _resolve_aiter_dir()


def check_aiter() -> None:
    """Start-up check (does not touch sys.path): raise a clear error if the aiter package cannot be found."""
    if AITER_SOURCE == "installed package" or os.path.isfile(os.path.join(AITER_DIR, "aiter", "__init__.py")):
        return
    raise RuntimeError(
        f"ROCm aiter not found (looked in {AITER_DIR!r}, from {AITER_SOURCE}). The default path needs it (mha_v4 "
        "attention, hipBLASLt solution GEMMs). Install it as described in docs/PIKA_HANDOVER.md and set "
        "PIKA_AITER_DIR=<aiter checkout> (or pip-install it into the environment).")


def ensure_aiter() -> None:
    """Put AITER_DIR on sys.path; raise a clear error at start-up if aiter cannot be imported."""
    if AITER_DIR not in sys.path:
        sys.path.insert(0, AITER_DIR)
    check_aiter()


def cache_dir(sub: str = "", env: str | None = None) -> str:
    """Build cache directory ``<PIKA_CACHE_DIR>/<sub>``; ``env`` names a per-cache override variable."""
    if env and os.environ.get(env):
        return os.environ[env]
    root = os.environ.get("PIKA_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "pika_h3")
    return os.path.join(root, sub) if sub else root


def load_lock_dir() -> str:
    d = os.environ.get("PIKA_LOAD_LOCK_DIR") or cache_dir()
    os.makedirs(d, exist_ok=True)
    return d


def tmp_dir() -> str:
    import tempfile
    return tempfile.gettempdir()
