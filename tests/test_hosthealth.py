# SPDX-License-Identifier: Apache-2.0
"""CPU-only: host mitigation defaults (PIKA_HOST_THP / PIKA_HOST_MEMPOLICY on by default, env switches them off)."""
import os
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
