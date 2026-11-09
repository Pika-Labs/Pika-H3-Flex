# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU test of the worker at world 2 (two visible GPUs; no eval or dev tools needed).

    export PIKA_BASE_PATH=<MiniMax-H3 snapshot> PIKA_ADAPTER_PATH=<fl2va adapter dir>
    HIP_VISIBLE_DEVICES=0,1 PYTHONPATH=src python -m unittest tests.test_world2 -v

One long-lived worker serves, in this order (all 768p 16:9, PIKA_TEST_DURATION s, default 5; one fixed test prompt/seed):
  L1 lossless (no "lossy" field, traced) | L2 "lossy": {} (traced) | three invalid lossy requests (must fail cleanly, worker keeps
  serving) | near_lossless | balanced | fast | the three presets again | L3 lossless again (traced).
Checks: L1 == L2 == L3 bit-identical (trace tensors and MP4 sha256: flag-off is the lossless path, and a lossy request leaves no
state behind); each preset deterministic (same MP4 bytes twice); every MP4 valid (1344x768, 17n+5 frame count, 24 fps, 32 kHz stereo, finite non-silent audio).
PIKA_TEST_OUT=<dir> keeps the outputs (summary.json with sha256s), e.g. to compare two checkouts with tests/compare_runs.py.
Reference mode (to run the same requests on an older checkout without presets): PIKA_TEST_SRC=<other repo>/src and
PIKA_TEST_RAW_PRESETS=1 (presets sent as their expanded raw dicts; the invalid-request cases are skipped).
TestFaultsWorld2: failure semantics with the default-off PIKA_FAULT hook (three short-lived workers; PIKA_TEST_FAULTS=0 skips).
Skipped unless PIKA_BASE_PATH / PIKA_ADAPTER_PATH are set and >= 2 GPUs are visible (PIKA_TEST_WORLD to change).
"""
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from tests.worker_client import WorkerClient, probe_mp4, sha256, traces_equal

BASE, ADAPTER = os.environ.get("PIKA_BASE_PATH"), os.environ.get("PIKA_ADAPTER_PATH")
WORLD = int(os.environ.get("PIKA_TEST_WORLD", "2"))
DURATION = int(os.environ.get("PIKA_TEST_DURATION", "5"))
PROMPT = ("A cozy bakery at dawn: a baker in a white apron slides a tray of croissants onto a wooden counter, steam rising, "
          "a hand-painted sign reading 'FRESH BREAD' in the window. Warm light, soft jazz playing on a radio.")
SEED = 20261107
PRESETS = ("near_lossless", "balanced", "fast")
SRC = os.environ.get("PIKA_TEST_SRC")
RAW = os.environ.get("PIKA_TEST_RAW_PRESETS") == "1"
BAD = [{"lossy": "turbo"}, {"lossy": {"attn": "fp7"}}, {"lossy": {"preset": "fast", "unknown_key": 1}}]


def _frames(duration):
    from pika_h3.core.geometry import align_num_frames  # the 17n+5 grid
    return align_num_frames(duration * 24)


def _gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


@unittest.skipUnless(BASE and ADAPTER, "set PIKA_BASE_PATH and PIKA_ADAPTER_PATH")
class TestWorld2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _gpu_count() < WORLD:
            raise unittest.SkipTest(f"needs {WORLD} visible GPUs")
        keep = os.environ.get("PIKA_TEST_OUT")
        cls.out = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="pika_test_w2_"))
        cls.out.mkdir(parents=True, exist_ok=True)
        cls.res = {}
        env = {"PIKA_WORLD": str(WORLD)}
        t0 = time.monotonic()
        with WorkerClient(BASE, ADAPTER, env=env, log_path=cls.out / "worker.log", src=SRC) as w:
            cls.ready = w.ready
            base = {"task": "t2va", "prompt": PROMPT, "seed": SEED, "resolution": "768p", "aspect": "16:9", "duration": DURATION}

            def run(name, trace=False, **extra):
                req = dict(base, id=name, output=str(cls.out / f"{name}.mp4"), **extra)
                if trace:
                    req["trace_dir"] = str(cls.out / f"{name}_trace")
                cls.res[name] = w.request(req)

            run("L1", trace=True)
            run("L2", trace=True, lossy={})
            for i, b in enumerate([] if RAW else BAD):
                run(f"BAD{i}", **b)
            for rep in ("", "_again"):  # each preset twice: valid output, and the same bytes again (determinism)
                for p in PRESETS:
                    if RAW:
                        from pika_h3.lossy_presets import PRESETS as SPECS
                        run(p + rep, lossy=SPECS[p])
                    else:
                        run(p + rep, lossy=p)
            run("L3", trace=True)
        cls.total_s = time.monotonic() - t0
        summ = {k: {"ok": v.get("ok"), "latency_s": round(v["latency_s"], 3), "error": v.get("error")} for k, v in cls.res.items()}
        for k in cls.res:
            mp4 = cls.out / f"{k}.mp4"
            if mp4.exists():
                summ[k]["sha256"] = sha256(mp4)
        (cls.out / "summary.json").write_text(json.dumps({"world": WORLD, "duration": DURATION, "ready": cls.ready,
                                                          "total_s": round(cls.total_s, 1), "requests": summ}, indent=1))
        print(f"\n[test_world2] outputs in {cls.out}; summary: {json.dumps(summ)}", flush=True)

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("PIKA_TEST_OUT") and getattr(cls, "out", None):
            shutil.rmtree(cls.out, ignore_errors=True)

    def _ok(self, name):
        r = self.res[name]
        self.assertTrue(r.get("ok"), f"{name}: {r.get('error')}")

    def _valid_mp4(self, name):
        info = probe_mp4(self.out / f"{name}.mp4")
        self.assertEqual((info["width"], info["height"]), (1344, 768), name)
        self.assertEqual(info["frames"], _frames(DURATION), name)
        self.assertAlmostEqual(info["fps"], 24.0, places=3, msg=name)
        self.assertEqual((info["audio_rate"], info["audio_channels"]), (32000, 2), name)
        self.assertTrue(info["audio_finite"], name)
        self.assertGreater(info["audio_rms"], 1e-4, f"{name}: silent audio")
        self.assertGreaterEqual(info["audio_samples"], int(32000 * (_frames(DURATION) / 24.0)) - 2048, name)

    def test_lossless_valid(self):
        for n in ("L1", "L2", "L3"):
            self._ok(n)
            self._valid_mp4(n)
            self.assertTrue((self.out / f"{n}_trace" / "trace.safetensors").exists(), n)

    def test_flag_off_bit_identical(self):
        """no field == "lossy": {} (trace tensors and MP4 bytes)."""
        eq, diff = traces_equal(self.out / "L1_trace" / "trace.safetensors", self.out / "L2_trace" / "trace.safetensors")
        self.assertTrue(eq, f"trace differs at {diff}")
        self.assertEqual(sha256(self.out / "L1.mp4"), sha256(self.out / "L2.mp4"))

    def test_no_state_leak_after_lossy(self):
        """lossless after three lossy presets == the first lossless request."""
        eq, diff = traces_equal(self.out / "L1_trace" / "trace.safetensors", self.out / "L3_trace" / "trace.safetensors")
        self.assertTrue(eq, f"trace differs at {diff}")
        self.assertEqual(sha256(self.out / "L1.mp4"), sha256(self.out / "L3.mp4"))

    def test_presets_end_to_end(self):
        for p in PRESETS:
            self._ok(p)
            self._valid_mp4(p)
            self.assertNotEqual(sha256(self.out / f"{p}.mp4"), sha256(self.out / "L1.mp4"), f"{p} produced the lossless output")

    def test_presets_deterministic(self):
        """the same lossy request twice in one worker -> the same MP4 bytes (a service needs reproducible outputs)."""
        bad = [p for p in PRESETS if sha256(self.out / f"{p}.mp4") != sha256(self.out / f"{p}_again.mp4")]
        self.assertFalse(bad, f"non-deterministic presets: {bad}")

    @unittest.skipIf(RAW, "reference mode")
    def test_invalid_lossy_rejected(self):
        for i in range(len(BAD)):
            r = self.res[f"BAD{i}"]
            self.assertFalse(r.get("ok"), f"BAD{i} was accepted")
            self.assertIn("lossy", r.get("error", ""))
            self.assertFalse((self.out / f"BAD{i}.mp4").exists())


def _children(pid):
    out = set()
    for t in Path(f"/proc/{pid}/task").glob("*/children"):
        out |= {int(x) for x in t.read_text().split()}
    return out


def _alive(pid):
    try:
        st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return st not in ("Z", "X")
    except OSError:
        return False


FATAL_EXIT_S = float(os.environ.get("PIKA_TEST_FATAL_EXIT_S", "30"))


@unittest.skipUnless(BASE and ADAPTER, "set PIKA_BASE_PATH and PIKA_ADAPTER_PATH")
@unittest.skipIf(SRC or os.environ.get("PIKA_TEST_FAULTS") == "0", "fault tests off (reference mode or PIKA_TEST_FAULTS=0)")
class TestFaultsWorld2(unittest.TestCase):
    """Failure semantics (src/pika_h3/faults.py) with the default-off PIKA_FAULT hook, one worker per case:
      helper-rank exception mid-sampling and rank-0 out-of-memory -> ok=false/fatal, EAT_FATAL, exit code 3 within
      PIKA_TEST_FATAL_EXIT_S s (default 30), no helper rank left; rank-0 failure after the MP4 (local) and a prompt rejected
      on rank 0 during conditioning -> ok=false, and the next request succeeds with the same MP4 bytes."""

    @classmethod
    def setUpClass(cls):
        if _gpu_count() < WORLD:
            raise unittest.SkipTest(f"needs {WORLD} visible GPUs")
        cls.out = Path(tempfile.mkdtemp(prefix="pika_test_faults_"))
        cls.base = {"task": "t2va", "prompt": PROMPT, "seed": SEED, "resolution": "768p", "aspect": "16:9", "duration": DURATION}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def _fatal_case(self, fault):
        name = fault.replace(":", "_")
        w = WorkerClient(BASE, ADAPTER, env={"PIKA_WORLD": str(WORLD), "PIKA_FAULT": fault}, log_path=self.out / f"{name}.log")
        try:
            kids = _children(w.proc.pid)
            self.assertEqual(len(kids), WORLD - 1, f"helper ranks {kids}")
            res = w.request(dict(self.base, id=name, output=str(self.out / f"{name}.mp4")), timeout=300)
            t_res = time.monotonic()
            self.assertFalse(res.get("ok"), res)
            self.assertTrue(res.get("fatal"), res)
            code = w.proc.wait(timeout=FATAL_EXIT_S)
            dt = time.monotonic() - t_res
            self.assertEqual(code, 3, f"exit code {code}")
            left = [k for k in kids if _alive(k)]
            self.assertFalse(left, f"helper ranks still alive: {left}")
            print(f"\n[test_faults] {fault}: result after {res['latency_s']:.1f} s ({res.get('error')}); "
                  f"exit {code} {dt:.1f} s after the result", flush=True)
        finally:
            w.close()

    def test_helper_rank_exception_is_fatal(self):
        self._fatal_case("1:step3")

    def test_rank0_oom_is_fatal(self):
        self._fatal_case("0:oom")

    def test_local_and_rejected_failures_keep_serving(self):
        w = WorkerClient(BASE, ADAPTER, env={"PIKA_WORLD": str(WORLD), "PIKA_FAULT": "0:post:1"}, log_path=self.out / "post.log")
        try:
            r1 = w.request(dict(self.base, id="post", output=str(self.out / "post.mp4")))
            self.assertFalse(r1.get("ok"), r1)
            self.assertFalse(r1.get("fatal"), r1)
            self.assertIn("injected fault", r1.get("error", ""))
            long_prompt = " ".join(["a red balloon drifting over a quiet town"] * 600)  # > MAX_TEXT_TOKENS: rejected on rank 0
            r2 = w.request(dict(self.base, id="long", prompt=long_prompt, output=str(self.out / "long.mp4")))
            self.assertFalse(r2.get("ok"), r2)
            self.assertFalse(r2.get("fatal"), r2)
            r3 = w.request(dict(self.base, id="next", output=str(self.out / "next.mp4")))
            self.assertTrue(r3.get("ok"), r3)
            self.assertEqual(sha256(self.out / "post.mp4"), sha256(self.out / "next.mp4"))  # same request, same bytes
            self.assertIsNone(w.proc.poll())
            print(f"\n[test_faults] post/long/next: {r1.get('error')} | {r2.get('error')[:120]} | ok {r3['latency_s']:.1f} s, "
                  f"sha {sha256(self.out / 'next.mp4')[:8]}", flush=True)
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
