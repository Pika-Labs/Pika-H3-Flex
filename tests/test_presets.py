# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the lossy presets and request validation (no GPU, no model, no eval or dev tools).

    PYTHONPATH=src python -m unittest tests.test_presets -v
"""
import json
import os
import unittest

from pika_h3 import lossy_presets as P

# The final specs the presets must expand to, verbatim (plus the decode keys).
_DEC = {"vae_tile": [256, 256, 64, 32], "vae_fp8": "all"}
_G = "qkv,o,up,down"
EXPECTED = {
    "near_lossless": {"attn": ["bf16", "bf16", "mxfp6", "mxfp6", "mxfp6", "mxfp6"], "attn_prefix_bf16": True,
                      "tier": ["", "", "bf16:0.1,mxfp6:1", "", "", ""], "gemm": ["", "", "", "", _G, _G],
                      "gemm_fmt": "a6w6", "gemm_prefix_bf16": True, **_DEC},
    "balanced": {"attn": ["bf16", "bf16", "mxfp6", "mxfp6", "mxfp6", "mxfp6"], "attn_prefix_bf16": True,
                 "tier": ["", "bf16:0.1,mxfp6:1", "", "", "", ""], "gemm": ["", "", "", _G, _G, _G],
                 "gemm_fmt": "a6w6", "gemm_prefix_bf16": True, **_DEC},
}
EXPECTED["fast"] = EXPECTED["balanced"]  # alias
F2C = {"impl": "s2", "dense_q": 1, "prefix": "auto", "est": "rows16k4", "sparse": [1, 1, 1, 0.35, 0.12, 0.08],
       "attn": ["bf16", "bf16", "mxfp6", "bf16", "bf16", "bf16"], "attn_prefix_bf16": True, "s2_prefix_bf16": True,
       "gemm": ["", "", _G, _G, _G, _G], "gemm_fmt": "a6w6", "gemm_prefix_bf16": True, **_DEC}


class TestPresets(unittest.TestCase):
    def test_names(self):
        self.assertEqual(set(P.PRESETS), set(EXPECTED))

    def test_expand_exact(self):
        for name, spec in EXPECTED.items():
            for form in (name, {"preset": name}, json.dumps({"preset": name})):
                self.assertEqual(P.expand(form), spec, (name, form))

    def test_expand_returns_copy(self):
        a = P.expand("fast")
        a["attn"][3] = "bf16"
        self.assertEqual(P.expand("fast")["attn"][3], "mxfp6")
        self.assertEqual(P.expand("balanced")["attn"][3], "mxfp6")

    def test_old_fast_raw_spec(self):  # the former fast spec is still valid as a raw spec
        self.assertEqual(P.expand(P.F2C_A2345_D), F2C)
        self.assertEqual(P.expand(json.dumps(F2C)), F2C)

    def test_steps_0_1_exact(self):  # every preset keeps the first two generator evaluations exact
        for name, spec in P.PRESETS.items():
            self.assertEqual(spec["attn"][:2], ["bf16", "bf16"], name)
            self.assertEqual(spec["gemm"][:2], ["", ""], name)
            self.assertTrue(all(x >= 1 for x in spec.get("sparse", [1, 1])[:2]), name)
            self.assertFalse(spec.get("tier", ["", ""])[0], name)

    def test_override(self):
        s = P.expand({"preset": "near_lossless", "vae_fp8": False})
        self.assertIs(s["vae_fp8"], False)
        s = P.expand({"preset": "near_lossless", "vae_tile": None, "vae_fp8": None})
        self.assertNotIn("vae_tile", s)
        self.assertNotIn("vae_fp8", s)

    def test_lossless_forms(self):
        for form in (None, {}, "", "lossless", "off", {"preset": "lossless"}, "{}"):
            self.assertEqual(P.expand(form), {}, form)

    def test_raw_spec_passthrough(self):
        raw = {"attn": "mxfp6", "attn_prefix_bf16": True, "sparse": [1, 1, 0.5], "est": "rows8k4"}
        self.assertEqual(P.expand(raw), raw)
        self.assertEqual(P.expand({"headstats": "/tmp/x"}), {"headstats": "/tmp/x"})  # experimental keys pass

    def test_unknown_and_bad(self):
        bad = ["fastest", "Near_Lossless", {"preset": "turbo"}, {"preset": 3}, {"foo": 1}, {"preset": "fast", "spars": [1]},
               {"preset": "lossless", "attn": "mxfp6"}, {"attn": "fp7"}, {"attn": ["bf16"] * 7}, {"gemm": "qkv,ffn"},
               {"gemm_fmt": "int4"}, {"sparse": [1, 1, 0]}, {"sparse": [1, 1, 1.5]}, {"sparse": "x"},
               {"tier": "bf16:0.1,fp3:1"}, {"tier": "bf16:2"}, {"tier": "comp"}, {"tier": 5}, {"est": "magic"},
               {"prefix": -1}, {"prefix": "all"}, {"impl": "s3"}, {"vae_tile": [256, 256, 64]},
               {"vae_tile": [256, 256, 300, 32]}, {"vae_tile": [250, 256, 64, 32]}, {"vae_fp8": "qkv,mlp"},
               {"attn_prefix_bf16": "yes"}, "{not json", 7, ["fast"]]
        for b in bad:
            with self.assertRaises(P.LossyConfigError, msg=repr(b)):
                P.expand(b)

    def test_error_is_value_error(self):
        self.assertTrue(issubclass(P.LossyConfigError, ValueError))

    def test_every_key_read_by_the_code_is_known(self):
        """Each lossy key the source reads (LS.get / CFG.get) must be registered, else valid configs would be rejected."""
        import re
        src = os.path.join(os.path.dirname(P.__file__))
        keys = set()
        for f in os.listdir(src):
            if f.endswith(".py"):
                t = open(os.path.join(src, f)).read()
                keys |= set(re.findall(r"(?:LS|_LS)\.(?:CFG\.)?get\(\"([a-z_0-9]+)\"", t))
                keys |= set(re.findall(r"\bCFG\.get\(\"([a-z_0-9]+)\"", t))
        known = set(P.SUPPORTED_KEYS) | set(P.EXPERIMENTAL_KEYS)
        self.assertFalse(keys - known, f"lossy keys read by the code but not registered in lossy_presets: {keys - known}")


class TestRequestValidation(unittest.TestCase):
    def setUp(self):
        from pika_h3.serve import validate_request
        self.v = validate_request
        self.req = {"id": "a", "task": "t2va", "prompt": "a red fox", "seed": 1, "resolution": "768p", "aspect": "16:9",
                    "duration": 5, "output": "/tmp/x.mp4"}

    def test_valid_unchanged(self):
        self.assertEqual(self.v(dict(self.req)), self.req)  # no lossy field: the request passes through unchanged

    def test_preset_expanded(self):
        r = self.v(dict(self.req, lossy="balanced"))
        self.assertEqual(r["lossy"], EXPECTED["balanced"])

    def test_bad_requests(self):
        for k, v in [("task", "i2v"), ("prompt", ""), ("prompt", None), ("seed", "1"), ("seed", 1.5), ("seed", True),
                     ("resolution", "1080p"), ("aspect", "16:10"), ("duration", 16), ("duration", 1), ("duration", "5"),
                     ("output", ""), ("lossy", "turbo"), ("lossy", {"bogus": 1})]:
            with self.assertRaises(ValueError, msg=f"{k}={v!r}"):
                self.v(dict(self.req, **{k: v}))


if __name__ == "__main__":
    unittest.main()
