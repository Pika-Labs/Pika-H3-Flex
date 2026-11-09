# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the fault-injection hook (src/pika_h3/faults.py); the GPU semantics are in tests/test_world2.py."""
import unittest

from pika_h3 import faults as F


class TestFaultSpec(unittest.TestCase):
    def test_default_off(self):
        self.assertIsNone(F._parse(None))
        self.assertIsNone(F._parse(""))

    def test_parse(self):
        self.assertEqual(F._parse("1:step3"), (1, "step3", 1))
        self.assertEqual(F._parse("0:post:2"), (0, "post", 2))
        for bad in ("1", "1:step9", "x:step1", "1:decode:2:3"):
            with self.assertRaises(ValueError, msg=bad):
                F._parse(bad)

    def test_fires_only_on_target(self):
        old, n = F.FAULT, F._count["n"]
        try:
            F.FAULT, F._count["n"] = (1, "step2", 1), 0
            F.new_request()
            calls = []
            g = F.FaultyGenerator(lambda **kw: calls.append(kw) or (0, 0), rank=1)
            g(x=0)
            g(x=1)
            with self.assertRaises(F.InjectedFault):
                g(x=2)
            self.assertEqual(len(calls), 2)
            F.fire(0, "step2")  # other rank: no fault
            F.new_request()     # request 2: not armed any more
            F.FaultyGenerator(lambda **kw: (0, 0), rank=1)(x=0)
            self.assertFalse(F.armed(1, "step2"))
        finally:
            F.FAULT, F._count["n"] = old, n


if __name__ == "__main__":
    unittest.main()
