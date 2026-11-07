# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the machine-independent locations (src/pika_h3/paths.py) and of src/ being free of machine paths."""
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def _run(code, **env):
    e = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp"), "PYTHONPATH": str(SRC)}
    e.update(env)
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True)


class TestPaths(unittest.TestCase):
    def test_cache_dir_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            code = "from pika_h3 import paths as p; print(p.cache_dir('sdma', env='PIKA_SDMA_BUILD'))"
            self.assertEqual(_run(code, XDG_CACHE_HOME=d).stdout.strip(), f"{d}/pika_h3/sdma")
            self.assertEqual(_run(code, XDG_CACHE_HOME=d, PIKA_CACHE_DIR=f"{d}/c").stdout.strip(), f"{d}/c/sdma")
            self.assertEqual(_run(code, PIKA_CACHE_DIR=f"{d}/c", PIKA_SDMA_BUILD=f"{d}/s").stdout.strip(), f"{d}/s")

    def test_aiter_env_wins_and_missing_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            r = _run("from pika_h3 import paths as p; print(p.AITER_DIR, p.AITER_SOURCE); p.check_aiter()", PIKA_AITER_DIR=d)
            self.assertTrue(r.stdout.startswith(f"{d} PIKA_AITER_DIR"), r.stdout + r.stderr)
            self.assertIn("ROCm aiter not found", r.stderr)

    def test_no_machine_paths_in_src(self):
        bad = re.compile(r"(?<![\w.])/(?:data|home|work|scratch)/[a-z]|/shm/")
        hits = []
        for f in SRC.rglob("*"):
            if f.is_file() and f.suffix in (".py", ".cpp", ".hip", ".inc", ".yaml", ".h"):
                for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                    if bad.search(line):
                        hits.append(f"{f.relative_to(SRC)}:{i}")
        self.assertFalse(hits, f"machine-specific paths in src/: {hits}")


if __name__ == "__main__":
    unittest.main()
