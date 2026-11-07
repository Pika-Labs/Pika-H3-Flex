# SPDX-License-Identifier: Apache-2.0
"""Warm request worker: load every model once, then serve t2va requests read from stdin.

Protocol (one JSON object per line):
  stdin  request : {"id": str, "task": "t2va", "prompt": str, "seed": int, "resolution": "480p"|"768p",
                    "aspect": "16:9", "duration": int, "output": "/abs/path.mp4", "trace_dir": "/abs/dir"?,
                    "lossy": "near_lossless"|"balanced"|"fast"|{...}?}   (lossy: opt-in, see docs/PIKA_HANDOVER.md)
                   Requests are validated before they run (validate_request); an invalid one gets ok=false.
                   With "trace_dir" (verification requests, untimed) the worker also writes trace_dir/trace.safetensors:
                   for each of the six generator evaluations i, step{i}.video_in / audio_in (x_t, external layout),
                   step{i}.sigma_v / sigma_a and step{i}.video_pred / audio_pred (x0), all fp32, plus the decoded
                   output before encoding: decoded.pixels (uint8 [N, H, W, 3]) and decoded.wave (fp32 [2, S]).
  stdout ready   : EAT_READY {"load_seconds": float, ...}            (once, after all models are loaded)
  stdout result  : EAT_RESULT {"id": str, "ok": bool, "timings": {...}, "error": str?, "fatal": true?}
                   (written after the MP4 at "output" is complete and closed)
  stdout fatal   : EAT_FATAL {"error": str, "rank": int?, "exit_code": 3}  (then the process exits with code 3 within
                   seconds; a process manager restarts it. Failure semantics: faults.py and docs/PIKA_HANDOVER.md)
Every other line on stdout/stderr is free-form log output.

The request path is the same as ``pipeline.generate`` (conditioning, six-step sampling, decode, MP4 encode and
validation), with the text encoder, transformer and decoders kept resident on one GPU.
"""
import argparse
import os
import json
import sys
import threading
import time
from pathlib import Path

_EMIT_LOCK = threading.RLock()


def emit(tag, payload):
    with _EMIT_LOCK:
        sys.stdout.write(f"{tag} {json.dumps(payload)}\n")
        sys.stdout.flush()


class TracingGenerator:
    """Records every generator evaluation: inputs x_t, sigmas and x0 predictions (fp32, CPU)."""

    def __init__(self, wrapper, trace):
        self.wrapper, self.trace = wrapper, trace

    def __call__(self, *, noisy_image_or_video, noisy_audio, timestep, audio_timestep, **kw):
        pv, pa = self.wrapper(noisy_image_or_video=noisy_image_or_video, noisy_audio=noisy_audio,
                              timestep=timestep, audio_timestep=audio_timestep, **kw)
        self.trace.append({"video_in": noisy_image_or_video, "audio_in": noisy_audio, "sigma_v": timestep,
                           "sigma_a": audio_timestep, "video_pred": pv, "audio_pred": pa})
        return pv, pa


def write_trace(trace_dir, trace, *, pixels, wave):
    """trace_dir/trace.safetensors: step{i}.{video_in,audio_in,sigma_v,sigma_a,video_pred,audio_pred}, i = 0..5, and the
    decoded output before MP4 encoding: decoded.pixels (uint8 [N, H, W, 3]) and decoded.wave (fp32 [2, S])."""
    import torch
    from safetensors.torch import save_file
    trace_dir.mkdir(parents=True, exist_ok=True)
    tensors = {f"step{i}.{k}": v.detach().float().cpu().contiguous() for i, s in enumerate(trace) for k, v in s.items()}
    tensors["decoded.pixels"] = torch.from_numpy(pixels).contiguous()
    tensors["decoded.wave"] = torch.from_numpy(wave.astype("float32")).contiguous()
    save_file(tensors, str(trace_dir / "trace.safetensors"), metadata={"steps": str(len(trace))})


def choose_devices(world=None):
    """Pick the GPUs to run on. All visible GPUs when they have room; otherwise the largest group whose size divides the
    56 attention heads (8, 7, 4, 2, 1) among the GPUs with enough free memory (other users share this machine)."""
    import os
    import subprocess
    forced = os.environ.get("PIKA_DEVICES")
    if forced:
        return [int(x) for x in forced.split(",")]
    out = subprocess.run([sys.executable, "-c",
                          "import torch;print(' '.join(str(torch.cuda.mem_get_info(i)[0]>>20) for i in range(torch.cuda.device_count())))"],
                         capture_output=True, text=True, check=True).stdout.split()
    free = [int(x) for x in out]  # MiB per visible device
    n_vis = len(free)
    need_rank0, need_other = 135 * 1024, 66 * 1024
    want = int(world or os.environ.get("PIKA_WORLD") or n_vis)
    if all(f >= (need_rank0 if i == 0 else need_other) for i, f in enumerate(free[:want])) and want <= n_vis:
        return list(range(want))
    # rank 0 needs the text encoder: take the GPU with the most free memory (>= need_rank0)
    order = sorted(range(n_vis), key=lambda i: -free[i])
    r0 = order[0]
    if free[r0] < need_rank0:
        raise RuntimeError(f"no GPU has {need_rank0 >> 10} GB free for the text encoder: free MiB {free}")
    others = [i for i in range(n_vis) if i != r0 and free[i] >= need_other]
    n = 1 + len(others)
    for k in (8, 7, 4, 2, 1):
        if k <= min(n, want):
            print(f"[serve] WARNING: free MiB per GPU {free}; running on {k} GPUs", flush=True)
            return [r0] + others[:k - 1]
    return [r0]


_RESOLUTIONS = ("480p", "768p")


def validate_request(req):
    """Checks a request on rank 0 BEFORE it is broadcast to the helper ranks (a request that fails inside the collective
    path could leave the ranks out of step). Returns the request with ``lossy`` expanded to its validated raw spec
    (a preset name becomes the preset's dict). Raises ValueError with a clear message."""
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")
    if "tf_trace" in req:  # dev teacher-forced evaluation: own fields
        return req
    if req.get("task", "t2va") != "t2va":
        raise ValueError(f"task {req.get('task')!r} is not supported by this worker (only 't2va')")
    if not isinstance(req.get("prompt"), str) or not req["prompt"].strip():
        raise ValueError("'prompt' must be a non-empty string")
    if not isinstance(req.get("seed"), int) or isinstance(req.get("seed"), bool):
        raise ValueError(f"'seed' must be an integer, got {req.get('seed')!r}")
    if req.get("resolution") not in _RESOLUTIONS:
        raise ValueError(f"'resolution' must be one of {_RESOLUTIONS}, got {req.get('resolution')!r}")
    from .engine import _bucket_table
    if req.get("aspect") not in _bucket_table()["aspects"]:
        raise ValueError(f"'aspect' must be one of {_bucket_table()['aspects']}, got {req.get('aspect')!r}")
    d = req.get("duration")
    if not isinstance(d, int) or isinstance(d, bool) or not 2 <= d <= 15:
        raise ValueError(f"'duration' must be an integer number of seconds in 2..15, got {d!r}")
    if not isinstance(req.get("output"), str) or not req["output"]:
        raise ValueError(f"'output' must be a file path (the MP4 to write), got {req.get('output')!r}")
    if req.get("trace_dir") is not None and not isinstance(req["trace_dir"], str):
        raise ValueError(f"'trace_dir' must be a directory path, got {req.get('trace_dir')!r}")
    if "lossy" in req:
        from .lossy_presets import expand
        req = dict(req, lossy=expand(req["lossy"]))  # LossyConfigError is a ValueError
    return req


class Worker:
    """Rank-0 serving object: starts the helper ranks and runs every request collectively on all GPUs."""

    def __init__(self, base_path, adapter_path, world=None):
        import os
        import socket
        import subprocess
        import torch
        from .engine import Engine, init_dist

        if not torch.cuda.is_available():
            raise RuntimeError("a GPU is required for model generation")
        devices = choose_devices(world)
        world = len(devices)
        self.world = world
        self.devices = devices
        print(f"[serve] using {world} GPUs: {devices}", flush=True)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        self.procs = []
        child_env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parent.parent)
        child_env["PYTHONPATH"] = src_root + (":" + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else "")
        for r in range(1, world):
            self.procs.append(subprocess.Popen(
                [sys.executable, "-m", "pika_h3.rank_main", "--rank", str(r), "--world", str(world), "--device", str(devices[r]),
                 "--base-path", str(base_path), "--adapter-path", str(adapter_path)], env=child_env))
        # failure semantics (faults.py): a helper rank that exits, or a request over PIKA_REQUEST_TIMEOUT s, is fatal
        self._lock = threading.RLock()
        self._cur = self._cur_t0 = None
        self._closing = False
        self._req_timeout = float(os.environ.get("PIKA_REQUEST_TIMEOUT", "0") or 0)
        threading.Thread(target=self._watchdog, name="pika-watchdog", daemon=True).start()
        t0 = time.monotonic()
        ctl = init_dist(0, world, devices[0])
        self.engine = Engine(0, world, base_path, adapter_path, ctl, devices[0])
        self.load_seconds = time.monotonic() - t0
        from . import hosthealth
        hr, hw = hosthealth.host_report()
        for w_ in hw:  # host settings that let the kernel driver evict our GPU queues mid-request (latency spikes; hosthealth.py)
            print(f"[serve] WARNING host config: {w_}", file=sys.stderr, flush=True)
        self.evm = hosthealth.EvictionMeter([os.getpid()] + [pr.pid for pr in self.procs])

    def _watchdog(self):
        """Thread on rank 0 (sleeps 0.5 s between checks; no GPU work, no synchronisation with the request path)."""
        while not self._closing:
            time.sleep(0.5)
            if self._closing:
                return
            for r, pr in enumerate(self.procs, start=1):
                code = pr.poll()
                if code is not None and not self._closing:
                    self.fatal(f"helper rank {r} exited with code {code}", rank=r)
            t0 = self._cur_t0
            if self._req_timeout > 0 and t0 is not None and time.monotonic() - t0 > self._req_timeout:
                self.fatal(f"request {self._cur!r} exceeded PIKA_REQUEST_TIMEOUT={self._req_timeout:g} s", rank=None)

    def begin(self, rid):
        with self._lock:
            self._cur, self._cur_t0 = rid, time.monotonic()
        eng = getattr(self, "engine", None)
        if eng is not None:
            eng.phase = "idle"

    def finish(self, payload):
        """Answer the request in flight (unless a fatal error already answered it)."""
        with self._lock:
            if self._cur_t0 is None:
                return
            emit("EAT_RESULT", payload)
            self._cur = self._cur_t0 = None

    def is_fatal(self, exc):
        """Rank-0 failure: fatal while the ranks are inside the collective part of a request (faults.py)."""
        from .faults import RequestRejected
        return not isinstance(exc, RequestRejected) and getattr(self.engine, "phase", "idle") == "collective"

    def fatal(self, error, rank=0):
        """Answer the request in flight with ok=false/fatal, print EAT_FATAL, kill our helper ranks, exit(FATAL_EXIT_CODE).
        Callable from any thread; never returns."""
        from .faults import FATAL_EXIT_CODE
        try:
            with self._lock:
                self._closing = True
                if self._cur_t0 is not None:
                    emit("EAT_RESULT", {"id": self._cur, "ok": False, "fatal": True, "error": f"fatal: {error}"})
                    self._cur = self._cur_t0 = None
                emit("EAT_FATAL", {"error": str(error), "rank": rank, "exit_code": FATAL_EXIT_CODE})
                print(f"[serve] FATAL: {error}; stopping the helper ranks and exiting (code {FATAL_EXIT_CODE})",
                      file=sys.stderr, flush=True)
                for pr in self.procs:
                    if pr.poll() is None:
                        pr.kill()
                for pr in self.procs:
                    try:
                        pr.wait(timeout=20)
                    except Exception:
                        pass
        finally:
            sys.stderr.flush()
            os._exit(FATAL_EXIT_CODE)

    def generate(self, req):
        import torch
        req = validate_request(req)  # before the broadcast: an invalid request never reaches the helper ranks
        self.engine.phase = "collective"  # until run_request passes its last collective (faults.py)
        if "tf_trace" in req:  # dev (lossy track): teacher-forced evaluation of lossy configs on a trace's step inputs
            return self._teacher_forced(req)
        _t0 = time.monotonic()
        self.engine.bcast_obj({"cmd": "run", "req": req})
        _t1 = time.monotonic()
        timings = self.engine.run_request(req)
        if os.environ.get("PIKA_TIMING") == "4":  # dev: request overheads outside the timed phases
            print(f"[t4] bcast {(_t1 - _t0) * 1e3:.1f} ms run_request {(time.monotonic() - _t1) * 1e3:.1f} ms "
                  f"marks {getattr(self.engine, '_t4', None)}", file=sys.stderr, flush=True)
        ev = self.evm.delta()  # KFD GPU-queue eviction time of every rank during this request (sysfs; no GPU work)
        if ev is not None:
            timings["kfd_evicted_ms"] = ev
            if max(ev.values(), default=0) >= 100:
                print(f"[serve] WARNING request {req.get('id')}: GPU queues evicted by the kernel driver during the request "
                      f"(ms per rank {ev}); latency is affected (host memory compaction / NUMA balancing, see hosthealth.py)",
                      file=sys.stderr, flush=True)
        timings["sampling_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        timings["gpu_allocated_bytes"] = torch.cuda.memory_allocated()
        return timings

    def warmup(self, durations, presets=()):
        """Opt-in start-up warm-up (before EAT_READY): one synthetic request per duration, lossless and per lossy preset, so
        that kernels which load / compile on first use (aiter modules, Triton specialisations, the presets' kernels) do it
        here and not in the first real requests. The synthetic prompt and seed are fixed (nothing request-dependent);
        outputs go to a temporary directory that is deleted afterwards."""
        import shutil
        import tempfile
        d = tempfile.mkdtemp(prefix="pika_warmup_")
        try:
            for lossy in (None,) + tuple(presets):
                for dur in durations:
                    req = {"id": f"warmup_{lossy or 'lossless'}_{dur}s", "task": "t2va", "seed": 0, "resolution": "768p",
                           "aspect": "16:9", "duration": int(dur), "output": os.path.join(d, f"w_{lossy}_{dur}.mp4"),
                           "prompt": "A quiet harbour at sunrise: fishing boats rock gently on the water, gulls circle "
                                     "overhead, and waves lap against the stone pier."}
                    if lossy:
                        req["lossy"] = lossy
                    t0 = time.monotonic()
                    self.generate(req)
                    print(f"[serve] warm-up {req['id']}: {time.monotonic() - t0:.2f} s", flush=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def ready_barrier(self):
        """Start-up only (before EAT_READY, not timed). The helper ranks run their load-time THP pass (hosthealth.thp_pass) before
        they enter the request loop, and rank 0 does not otherwise wait for it: the pass was still running on the helpers when
        the first request arrived, and collapsing their GPU-mapped host pages evicted their GPU queues (~1.2 s on every rank in the
        first request). Here every rank finishes its GPU work and its pass, meets in a barrier, and rank 0 then waits until no
        rank's KFD evicted time grows any more (hosthealth.settle_evictions; queue restores after the pass complete
        asynchronously), and re-bases the per-request eviction meter."""
        import torch
        t0 = time.monotonic()
        torch.cuda.synchronize()
        self.engine.bcast_obj({"cmd": "sync"})
        self.engine.barrier()
        t1 = time.monotonic()
        from . import hosthealth
        st = hosthealth.settle_evictions(self.evm)
        st.update(barrier_s=round(t1 - t0, 3))
        return st

    def _teacher_forced(self, req):
        """req: tf_trace (trace.safetensors), tf_cfgs {name: lossy cfg}, tf_out (safetensors written by rank 0)."""
        import torch
        from safetensors.torch import save_file
        res, timings = {}, {}
        for name, cfg in req["tf_cfgs"].items():
            r2 = dict(req, lossy=cfg)
            self.engine.bcast_obj({"cmd": "tf", "req": r2, "path": req["tf_trace"]})
            _, out = self.engine.teacher_forced(r2, req["tf_trace"])
            for i, (pv, pa, dt) in enumerate(out):
                res[f"{name}.step{i}.video_pred"] = pv.contiguous()
                res[f"{name}.step{i}.audio_pred"] = pa.contiguous()
            timings[name] = [round(o[2], 4) for o in out]
        save_file(res, req["tf_out"])
        return timings

    def close(self):
        self._closing = True
        try:
            self.engine.bcast_obj({"cmd": "exit"})
        except Exception:
            pass
        # helper ranks exit on the "exit" command; wait for them so the GPU memory is released before this process returns
        # (the verifier needs the memory), and kill our own children if one hangs.
        for pr in self.procs:
            try:
                pr.wait(timeout=30)
            except Exception:
                pr.kill()
                pr.wait()


def main():
    p = argparse.ArgumentParser(prog="python -m pika_h3.serve")
    p.add_argument("--base-path", required=True, help="local MiniMax-H3 snapshot")
    p.add_argument("--adapter-path", required=True, help="local fl2va adapter directory")
    args = p.parse_args()
    # opt-in warm-up before EAT_READY (default: none, i.e. the worker is ready right after loading):
    #   PIKA_WARMUP="5,10,15"            synthetic lossless requests at these durations
    #   PIKA_WARM_PRESETS="near_lossless,fast"   also each listed lossy preset (at PIKA_WARMUP's durations, default 5)
    from .lossy_presets import PRESETS
    warm_d = [int(x) for x in os.environ.get("PIKA_WARMUP", "").replace(" ", "").split(",") if x]
    warm_p = [x for x in os.environ.get("PIKA_WARM_PRESETS", "").replace(" ", "").split(",") if x]
    for x in warm_p:
        if x not in PRESETS:
            raise SystemExit(f"PIKA_WARM_PRESETS: unknown preset {x!r}; expected some of {', '.join(PRESETS)}")
    if warm_p and not warm_d:
        warm_d = [5]
    worker = Worker(args.base_path, args.adapter_path)
    if warm_d:
        worker.warmup(warm_d, warm_p)
    import torch
    from . import hosthealth
    _thp = hosthealth.thp_pass()  # PIKA_HOST_THP, default on (hosthealth.py); None when PIKA_HOST_THP=0
    if _thp:
        print(f"[serve] rank 0: host THP pass {_thp}", file=sys.stderr, flush=True)
    st = worker.ready_barrier()  # every rank's load-time host work (THP pass) done and evictions settled before EAT_READY
    print(f"[serve] ready barrier {st}", file=sys.stderr, flush=True)
    emit("EAT_READY", {"load_seconds": worker.load_seconds, "devices": torch.cuda.device_count(),
                       "gpu_allocated_bytes": torch.cuda.memory_allocated()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as exc:
            emit("EAT_RESULT", {"id": None, "ok": False, "error": f"request is not valid JSON: {exc}"})
            continue
        rid = req.get("id") if isinstance(req, dict) else None
        worker.begin(rid)
        try:
            timings = worker.generate(req)
            worker.finish({"id": rid, "ok": True, "timings": timings})
            if _thp is not None and hosthealth.thp_incremental():
                hosthealth.thp_pass()  # after the result line: new mappings of this request (not timed)
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            if worker.is_fatal(exc) or not isinstance(exc, Exception):  # ranks out of step: exit now (faults.py)
                worker.fatal(f"rank 0: {type(exc).__name__}: {exc}", rank=0)
            worker.finish({"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"})  # rejected / local: keep serving
    worker.close()


if __name__ == "__main__":
    main()
