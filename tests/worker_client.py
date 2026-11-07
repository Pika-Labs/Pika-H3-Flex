# SPDX-License-Identifier: Apache-2.0
"""Minimal client for the stdin/stdout worker ``python -m pika_h3.serve`` (no dependency on any eval or dev tool).

    with WorkerClient(base_path, adapter_path) as w:
        res = w.request({"task": "t2va", "prompt": "...", "seed": 1, "resolution": "768p", "aspect": "16:9",
                         "duration": 5, "output": "/abs/out.mp4"})
        res["ok"], res["latency_s"], res["timings"]
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class WorkerClient:
    def __init__(self, base_path, adapter_path, *, env=None, log_path=None, ready_timeout=1800, src=None):
        e = dict(os.environ)
        e.update(env or {})
        src = str(Path(src or REPO / "src").resolve())
        e["PYTHONPATH"] = src + (":" + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
        e["PYTHONUNBUFFERED"] = "1"
        self.log = open(log_path, "a") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen([sys.executable, "-m", "pika_h3.serve", "--base-path", str(base_path),
                                      "--adapter-path", str(adapter_path)], cwd=str(Path(src).parent), env=e,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                                     start_new_session=True)
        self.q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.ready = self._wait("EAT_READY", ready_timeout)
        self._n = 0

    def _read(self):
        for line in self.proc.stdout:
            if self.log is not subprocess.DEVNULL:
                self.log.write(line)
                self.log.flush()
            self.q.put(line)
        self.q.put(None)

    def _wait(self, tag, timeout):
        t_end = time.monotonic() + timeout
        while True:
            left = t_end - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no {tag} line within {timeout} s")
            try:
                line = self.q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(f"worker exited (code {self.proc.wait()}) before {tag}")
            if line.startswith(tag + " "):
                return json.loads(line[len(tag) + 1:])

    def request(self, req, timeout=900):
        req = dict(req)
        self._n += 1
        req.setdefault("id", f"r{self._n}")
        t0 = time.monotonic()
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        res = self._wait("EAT_RESULT", timeout)
        res["latency_s"] = time.monotonic() - t0
        if res.get("id") != req["id"]:
            raise RuntimeError(f"result id {res.get('id')!r} != request id {req['id']!r}")
        return res

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # EOF: the worker shuts its helper ranks down and exits
                self.proc.wait(timeout=120)
            except Exception:
                self.proc.kill()
                self.proc.wait()
        if self.proc.stdout:
            self.proc.stdout.close()
        if self.log is not subprocess.DEVNULL:
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def probe_mp4(path):
    """-> dict(width, height, frames, fps, audio_rate, audio_channels, audio_finite, audio_rms) of an MP4 (PyAV)."""
    import av
    import numpy as np
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        fps = float(vs.average_rate)
        n = sum(1 for _ in c.decode(video=0))
    with av.open(str(path)) as c:
        a = c.streams.audio[0]
        rate, ch = a.rate, a.channels
        chunks = [f.to_ndarray().astype(np.float64) for f in c.decode(audio=0)]
    wav = np.concatenate([x.reshape(ch, -1) if x.shape[0] != ch else x for x in chunks], axis=1) if chunks else np.zeros((ch, 0))
    return {"width": vs.codec_context.width, "height": vs.codec_context.height, "frames": n, "fps": fps,
            "audio_rate": rate, "audio_channels": ch, "audio_samples": wav.shape[1],
            "audio_finite": bool(np.isfinite(wav).all()), "audio_rms": float(np.sqrt((wav ** 2).mean())) if wav.size else 0.0}


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def traces_equal(a, b):
    """(bit-identical?, first differing tensor or None) for two trace.safetensors files."""
    import torch
    from safetensors.torch import load_file
    ta, tb = load_file(str(a)), load_file(str(b))
    if set(ta) != set(tb):
        return False, f"keys differ: {sorted(set(ta) ^ set(tb))[:5]}"
    for k in sorted(ta):
        if ta[k].shape != tb[k].shape or ta[k].dtype != tb[k].dtype or not torch.equal(ta[k], tb[k]):
            return False, k
    return True, None
