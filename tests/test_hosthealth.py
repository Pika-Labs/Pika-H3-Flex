# SPDX-License-Identifier: Apache-2.0
"""CPU-only: host mitigation defaults (PIKA_HOST_THP / PIKA_HOST_MEMPOLICY on by default, env switches them off)."""
import os
import subprocess
import sys
import unittest
from unittest import mock

from pika_h3 import hosthealth


class TestHostDefaults(unittest.TestCase):
    def test_thp_default_on_and_off_switch(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PIKA_HOST_THP", None)
            self.assertTrue(hosthealth.thp_enabled())
            os.environ["PIKA_HOST_THP"] = "0"
            self.assertFalse(hosthealth.thp_enabled())
            self.assertIsNone(hosthealth.thp_pass())
            os.environ["PIKA_HOST_THP"] = "1"
            self.assertTrue(hosthealth.thp_enabled())

    def test_mempolicy_default_bind_and_off_switch(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PIKA_HOST_MEMPOLICY", None)
            self.assertEqual(hosthealth.mempolicy_mode(), "bind")
            os.environ["PIKA_HOST_MEMPOLICY"] = "off"
            self.assertIsNone(hosthealth.apply_mempolicy())

    def test_mempolicy_unknown_arch_skipped(self):
        with mock.patch.dict(os.environ, {"PIKA_HOST_MEMPOLICY": "bind"}), mock.patch("platform.machine", return_value="riscv64"):
            self.assertEqual(hosthealth.apply_mempolicy()["mode"], "skipped")
        self.assertEqual(hosthealth.SYS_SET_MEMPOLICY["x86_64"], 238)
        self.assertEqual(hosthealth.SYS_SET_MEMPOLICY["aarch64"], 237)

    def test_package_import_has_no_side_effects(self):
        # importing pika_h3 must not change the environment or the memory policy (the worker sets them in worker_env.apply)
        env = {k: v for k, v in os.environ.items()
               if k not in ("ROCPROFILER_REGISTER_ENABLED", "NCCL_MIN_P2P_NCHANNELS", "NCCL_MAX_P2P_NCHANNELS")}
        code = ("import os, sys; before = dict(os.environ);"
                " import pika_h3, pika_h3.hosthealth, pika_h3.lossy_presets; sys.exit(0 if dict(os.environ) == before else 1)")
        r = subprocess.run([sys.executable, "-c", code], env=env, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.assertEqual(r.returncode, 0)

    def test_worker_env_defaults(self):
        from pika_h3 import worker_env
        with mock.patch.dict(os.environ, {"PIKA_HOST_MEMPOLICY": "off", "NCCL_MIN_P2P_NCHANNELS": "8"}):
            os.environ.pop("NCCL_MAX_P2P_NCHANNELS", None)
            worker_env.apply()
            self.assertEqual(os.environ["NCCL_MIN_P2P_NCHANNELS"], "8")  # explicit value wins
            self.assertEqual(os.environ["NCCL_MAX_P2P_NCHANNELS"], "64")

    def test_incremental_pass_switch(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PIKA_HOST_THP_INCR", None)
            self.assertEqual(hosthealth.thp_incremental(), hosthealth.INCR_DEFAULT != "0")
            os.environ["PIKA_HOST_THP_INCR"] = "0"
            self.assertFalse(hosthealth.thp_incremental())
            os.environ["PIKA_HOST_THP_INCR"] = "1"
            self.assertTrue(hosthealth.thp_incremental())

    def test_settle_evictions(self):
        class FakeMeter:  # eviction grows for 3 polls, then stays flat
            ok = True
            def __init__(self):
                self.seq = [{0: 100, 1: 50}, {0: 20, 1: 0}, {0: 0, 1: 30}, {0: 5, 1: 0}]
            def delta(self):
                return self.seq.pop(0) if self.seq else {0: 0, 1: 0}
        st = hosthealth.settle_evictions(FakeMeter(), quiet_s=0.05, max_s=2.0, poll_s=0.01)
        self.assertEqual(st["evicted_ms"], {0: 125, 1: 80})
        self.assertFalse(st["timed_out"])


if __name__ == "__main__":
    unittest.main()
