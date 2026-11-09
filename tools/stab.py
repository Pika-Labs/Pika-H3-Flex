#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Latency stability harness for the worker. One command:

  python tools/stab.py --gpus 0,1,2,3,4,5,6,7 --out /tmp/stab_x \
      --cfg lossless --cfg near_lossless=preset:near_lossless --cfg balanced=preset:balanced [--cfg name=@cfg.json] \
      --rounds 12 [--prompts tools/stab_prompts.json] [--seed 0] [--max-minutes 25] [--env K=V ...]
  python tools/stab.py --analyze /tmp/stab_x        # re-run the analysis only

Needs PIKA_BASE_PATH (MiniMax-H3 snapshot) and PIKA_ADAPTER_PATH (fl2va LoRA) in the environment; runs the worker with the
Python interpreter that runs this script.

What it does:
  1. builds a randomized schedule: `--rounds` rounds; each round = one (prompt, duration, seed) drawn from the prompt file
     (>= 10 prompts of 10-200 words; durations cycle 5/10/15 in shuffled order, seeds random) sent once per config in shuffled
     config order -> configs are interleaved and see the same requests; prompts are new to the worker at their first use;
  2. ONE long-lived worker (`python -m pika_h3.serve` of this checkout, PIKA_WORLD = #gpus, HIP_VISIBLE_DEVICES = --gpus;
     wrapped in `gpu_lock <gpus> --` when a `gpu_lock` command is on PATH) with the timers PIKA_STEPLOG=1 (per-step, per-rank
     CUDA-event + host times, non-synchronizing) and PIKA_TIMING=5 (decode phases per rank); untimed warm-up = one fixed
     prompt (--warmup-prompt) at 5/10/15 s per config;
  3. timed requests one at a time (latency = request line -> EAT_RESULT); every request is appended to stab.jsonl with its
     stage timings, the per-step per-rank times and the worker's stderr lines;
  4. a monitor thread logs amd-smi metric (per-GPU power / gfx clock / hotspot temp / VRAM / activity, ~3 Hz) to monitor.jsonl,
     amd-smi process (foreign processes = not in the worker's process tree, on any GPU) to procs.jsonl, and `gpu_lock --status`
     (if available) to locks.jsonl;
  5. analysis -> report.md + summary.json: p50 / p95 / max per config and duration, outliers (> --outlier x p50 of the same
     config and duration) with stage breakdown (conditioning, each step's max-rank GPU ms and slowest host rank, decode, mp4),
     whether the prompt was new to the worker / config, and the interference seen during the request (min clock, max power,
     foreign processes, foreign locks).
--provoke GB:start_s:dur_s runs tools/evict_provoke.py (THP allocations -> host memory compaction) during the timed phase.
The worker is a child of this script and is stopped by PID. Stop: kill the driver PID it prints.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VENVPY = sys.executable
WARM_PROMPT_DEFAULT = ("A paper boat drifts down a rain-filled gutter past glowing shop windows at dusk; the camera follows it "
                       "at water level. Sound: rain, trickling water, distant traffic.")


def parse_cfg(s):
    if "=" not in s:
        return [s, None] if s == "lossless" else [s, {"preset": s}]
    n, v = s.split("=", 1)
    if v.startswith("preset:"):  # named preset of src/pika_h3/lossy_presets.py (request field "lossy": {"preset": X})
        return [n, {"preset": v[len("preset:"):]}]
    if v.startswith("@"):
        v = Path(v[1:]).read_text()
    return [n, json.loads(v)]


def build_schedule(a, cfgs):
    rng = random.Random(a.seed)
    prompts = json.loads(Path(a.prompts).read_text())
    order = list(range(len(prompts)))
    rng.shuffle(order)
    durs = []
    sched = []
    for r in range(a.rounds):
        if not durs:
            durs = [5, 10, 15]
            rng.shuffle(durs)
        d = durs.pop()
        p = prompts[order[r % len(order)]]
        seed = rng.randrange(1, 2 ** 31 - 1)
        co = [c[0] for c in cfgs]
        rng.shuffle(co)
        for name in co:
            sched.append({"round": r, "cfg": name, "d": d, "pname": p["name"], "prompt": p["prompt"], "seed": seed})
    return sched


# --------------------------------------------------------------------------------------------------------------- monitor
def _num(x):
    return x.get("value") if isinstance(x, dict) else None


def tree(root):
    """pids in the process tree of root (from /proc)."""
    kids = {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                ppid = int((p / "stat").read_text().rsplit(")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(p.name))
            except Exception:
                pass
    out, todo = set(), [root]
    while todo:
        x = todo.pop()
        out.add(x)
        todo += kids.get(x, [])
    return out


class Monitor(threading.Thread):
    def __init__(self, out, gpus, worker_pid_fn):
        super().__init__(daemon=True)
        self.out, self.gpus, self.wp = out, [int(g) for g in gpus.split(",")], worker_pid_fn
        self.stop = threading.Event()

    @staticmethod
    def _cpu_stat():
        f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return sum(f), f[3] + f[4]  # total, idle + iowait

    @staticmethod
    def _proc_cpu():
        o = {}
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                try:
                    st = (p / "stat").read_text()
                    rest = st.rsplit(")", 1)[1].split()
                    o[int(p.name)] = (int(rest[11]) + int(rest[12]), st[st.find("(") + 1:st.rfind(")")], p.stat().st_uid)
                except Exception:  # noqa: BLE001
                    pass
        return o

    def run(self):
        fm, fp, fl, fc = (open(self.out / n, "a") for n in ("monitor.jsonl", "procs.jsonl", "locks.jsonl", "cpu.jsonl"))
        k = 0
        hz = os.sysconf("SC_CLK_TCK")
        c_prev, pc_prev, tp_prev = self._cpu_stat(), self._proc_cpu(), time.time()
        me = os.getuid()
        while not self.stop.is_set():
            t = time.time()
            try:
                c = self._cpu_stat()
                busy = 1 - (c[1] - c_prev[1]) / max(1, c[0] - c_prev[0])
                c_prev = c
                rec = {"t": t, "busy": round(busy, 4), "load1": float(open("/proc/loadavg").read().split()[0])}
                if k % 10 == 0 and t - tp_prev > 1.0:
                    pc = self._proc_cpu()
                    dt = max(1e-3, t - tp_prev)
                    ours = tree(self.wp()) if self.wp() else set()
                    use = []
                    for pid, (tk, name, uid) in pc.items():
                        d0 = pc_prev.get(pid, (tk,))[0] if pid in pc_prev else 0
                        cores = (tk - d0) / hz / dt
                        if cores > 0.5:
                            use.append({"pid": pid, "name": name, "cores": round(cores, 2), "ours": pid in ours,
                                        "same_uid": uid == me})
                    use.sort(key=lambda x: -x["cores"])
                    rec["top"] = use[:12]
                    rec["foreign_cores"] = round(sum(u["cores"] for u in use if not u["ours"]), 2)
                    pc_prev, tp_prev = pc, t
                fc.write(json.dumps(rec) + "\n"); fc.flush()
            except Exception as e:  # noqa: BLE001
                fc.write(json.dumps({"t": t, "err": str(e)}) + "\n"); fc.flush()
            try:
                d = json.loads(subprocess.run(["amd-smi", "metric", "-p", "-c", "-t", "-u", "-m", "--json"], capture_output=True,
                                              text=True, timeout=10).stdout)
                rec = {"t": t, "g": {}}
                for g in d.get("gpu_data", []):
                    clks = [_num(v.get("clk", {})) for kk, v in g["clock"].items() if kk.startswith("gfx_")]
                    clks = [c for c in clks if isinstance(c, (int, float))]
                    rec["g"][g["gpu"]] = {"pw": _num(g["power"]["socket_power"]), "clk": min(clks) if clks else None,
                                          "clk_mean": sum(clks) / len(clks) if clks else None,
                                          "T": _num(g["temperature"].get("hotspot", {})),
                                          "used": _num(g["mem_usage"]["used_vram"]), "act": _num(g["usage"]["gfx_activity"]),
                                          "thr": g["power"].get("throttle_status")}
                fm.write(json.dumps(rec) + "\n"); fm.flush()
            except Exception as e:  # noqa: BLE001
                fm.write(json.dumps({"t": t, "err": str(e)}) + "\n"); fm.flush()
            if k % 10 == 0:
                try:
                    ours = tree(self.wp()) if self.wp() else set()
                    d = json.loads(subprocess.run(["amd-smi", "process", "--json"], capture_output=True, text=True, timeout=10).stdout)
                    fr = []
                    for g in d:
                        for pi in g.get("process_list", []):
                            pi = pi.get("process_info", pi)
                            if not isinstance(pi, dict) or "pid" not in pi:
                                continue
                            vr = _num(pi.get("memory_usage", {}).get("vram_mem", {})) or 0
                            if pi["pid"] not in ours:
                                fr.append({"gpu": g["gpu"], "pid": pi["pid"], "name": pi.get("name"), "vram_mb": round(vr / 2 ** 20)})
                    fp.write(json.dumps({"t": t, "foreign": fr, "ours": len(ours)}) + "\n"); fp.flush()
                    st = (subprocess.run(["gpu_lock", "--status"], capture_output=True, text=True, timeout=10).stdout
                          if shutil.which("gpu_lock") else "")
                    fl.write(json.dumps({"t": t, "status": st}) + "\n"); fl.flush()
                except Exception as e:  # noqa: BLE001
                    fp.write(json.dumps({"t": t, "err": str(e)}) + "\n"); fp.flush()
            k += 1
            self.stop.wait(max(0.0, 0.3 - (time.time() - t)))


# --------------------------------------------------------------------------------------------------------------- worker
def run_inner(a):
    out = Path(a.out)
    cfgs = json.loads((out / "cfgs.json").read_text())
    lossy = {n: lc for n, lc in cfgs}
    sched = json.loads((out / "schedule.json").read_text())
    G = os.environ.get("HIP_VISIBLE_DEVICES", "")
    env = dict(os.environ, PYTHONPATH=str(Path(a.repo) / "src"), PIKA_WORLD=str(len(G.split(","))), 
               PIKA_STEPLOG="1", PIKA_TIMING="5")
    for kv in a.env or []:
        k, v = kv.split("=", 1)
        env[k] = v
    cmd = [VENVPY, "-m", "pika_h3.serve", "--base-path", env["PIKA_BASE_PATH"], "--adapter-path", env["PIKA_ADAPTER_PATH"]]
    w = subprocess.Popen(cmd, cwd=a.repo, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, bufsize=1)
    (out / "worker.pid").write_text(str(w.pid))
    print(f"[stab] worker pid {w.pid} on GPUs {G} (driver pid {os.getpid()})", flush=True)
    mon = Monitor(out, a.gpus, lambda: w.pid)
    mon.start()
    log = open(out / "worker.log", "a")
    cur = {"bucket": None}
    lk = threading.Lock()

    def err_reader():
        for line in w.stderr:
            with lk:
                log.write(line); log.flush()
                if cur["bucket"] is not None and (line.startswith("[steplog]") or line.startswith("[timing]")):
                    cur["bucket"].append(line.rstrip("\n"))
    threading.Thread(target=err_reader, daemon=True).start()

    def readline_until(tag, tmax):
        t_end = time.monotonic() + tmax
        while True:
            line = w.stdout.readline()
            if not line:
                raise RuntimeError(f"worker exited (rc {w.poll()}) waiting for {tag}")
            with lk:
                log.write(line); log.flush()
                if cur["bucket"] is not None and (line.startswith("[steplog]") or line.startswith("[timing]")):
                    cur["bucket"].append(line.rstrip("\n"))
            if line.startswith(tag):
                return json.loads(line[len(tag) + 1:]), time.monotonic()
            if time.monotonic() > t_end:
                raise RuntimeError(f"timeout waiting for {tag}")

    tcd = Path(env.get("TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache")))
    seen_tc = {"s": set(os.listdir(tcd)) if tcd.is_dir() else set()}

    def new_kernels():
        """Triton kernels compiled (written to TRITON_CACHE_DIR) since the last call: names of the new cache entries."""
        if not tcd.is_dir():
            return []
        cur_s = set(os.listdir(tcd))
        new = cur_s - seen_tc["s"]
        seen_tc["s"] = cur_s
        names = []
        for d in new:
            try:
                names += [f[:-5] for f in os.listdir(tcd / d) if f.endswith(".json") and not f.startswith("__grp__")]
            except Exception:  # noqa: BLE001
                names.append(d)
        return sorted(names)

    def send(r, tmax=600):
        bucket = []
        with lk:
            cur["bucket"] = bucket
        tw = time.time()
        t0 = time.monotonic()
        w.stdin.write(json.dumps(r) + "\n"); w.stdin.flush()
        rr, t1 = readline_until("EAT_RESULT", tmax)
        time.sleep(0.05)  # let the stderr reader catch the helper ranks' lines
        with lk:
            cur["bucket"] = None
        if not rr.get("ok"):
            raise RuntimeError(f"request {r['id']} failed: {rr.get('error')}")
        tm_ = rr.get("timings", {})
        tm_["new_kernels"] = new_kernels()
        return t1 - t0, tw, time.time(), tm_, bucket

    def req(id_, prompt, seed, d, outp, lc):
        r = {"id": id_, "task": "t2va", "prompt": prompt, "seed": seed, "resolution": "768p", "aspect": "16:9", "duration": d,
             "output": str(outp)}
        if lc is not None:
            r["lossy"] = lc
        return r

    fo = open(out / "stab.jsonl", "a")
    mp = out / "mp4"
    mp.mkdir(exist_ok=True)
    try:
        ready, _ = readline_until("EAT_READY", 1800)
        print(f"[stab] ready {ready}; kernels compiled during load: {len(new_kernels())}", flush=True)
        t_dead = time.time() + a.max_minutes * 60
        wp = Path(a.warmup_prompt).read_text() if a.warmup_prompt else WARM_PROMPT_DEFAULT
        for name, lc in cfgs:
            for d in (5, 10, 15):
                lat, tw, te, tm, b = send(req(f"w_{name}_{d}", wp, 99, d, mp / f"warm_{name}_{d}.mp4", lc))
                fo.write(json.dumps({"kind": "warm", "cfg": name, "d": d, "latency": lat, "t0": tw, "t1": te, "timings": tm,
                                     "lines": b}) + "\n"); fo.flush()
                print(f"[stab] warm {name} {d}s {lat:.2f}s" + (f" new kernels {len(tm['new_kernels'])}" if tm["new_kernels"] else ""),
                      flush=True)
        prov = None
        t_timed0 = time.time()
        for i, s in enumerate(sched):
            if a.provoke and prov is None and time.time() - t_timed0 >= float(a.provoke.split(":")[1]):
                gb, _, dur = a.provoke.split(":")
                prov = subprocess.Popen([sys.executable, str(HERE / "evict_provoke.py"), gb, dur], stdout=open(out / "provoke.log", "a"))
                (out / "provoke.json").write_text(json.dumps({"pid": prov.pid, "t_start": time.time(), "gb": gb, "dur": dur}))
                print(f"[stab] compaction provoker pid {prov.pid}: {gb} GB cycles for {dur} s", flush=True)
            if time.time() > t_dead:
                print(f"[stab] --max-minutes reached after {i} requests", flush=True)
                break
            o = mp / f"r{s['round']:03d}_{s['cfg']}_{s['d']}s.mp4"
            lat, tw, te, tm, b = send(req(f"s{i}_{s['cfg']}_{s['d']}", s["prompt"], s["seed"], s["d"], o, lossy[s["cfg"]]))
            rec = dict(s, kind="timed", i=i, latency=lat, t0=tw, t1=te, timings=tm, lines=b)
            rec.pop("prompt")
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"[stab] {i + 1}/{len(sched)} r{s['round']} {s['cfg']} {s['d']}s {s['pname']} {lat:.3f}s "
                  f"(sampling {tm.get('sampling_seconds', 0):.3f})" + (f" NEW KERNELS {tm['new_kernels']}" if tm["new_kernels"] else ""),
                  flush=True)
            if not a.keep_mp4:
                o.unlink(missing_ok=True)
    finally:
        if a.provoke and prov is not None and prov.poll() is None:
            prov.kill()
        try:
            w.stdin.close()
            w.wait(timeout=90)
        except Exception:  # noqa: BLE001
            w.kill(); w.wait()
        mon.stop.set()
        print(f"[stab] worker {w.pid} exited rc {w.returncode}", flush=True)


# --------------------------------------------------------------------------------------------------------------- analysis
def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def parse_lines(lines):
    steps = {}  # rank -> [step dicts]
    dec = {}
    for ln in lines:
        if ln.startswith("[steplog] "):
            d = json.loads(ln[len("[steplog] "):])
            steps[d["rank"]] = d["steps"] or []
            if d.get("mem"):
                dec.setdefault("_mem", {})[d["rank"]] = d["mem"]
        elif ln.startswith("[timing] rank"):
            try:
                r = int(ln.split()[2].rstrip(":"))
                parts = ln.split(":", 1)[1].split()
                dec[r] = {parts[i]: float(parts[i + 1]) for i in range(0, len(parts) - 1, 2)}
            except Exception:  # noqa: BLE001
                pass
    return steps, dec


def step_summary(steps):
    """per step: max-rank GPU ms, min-rank GPU ms, max host ms and its rank, start offset of the latest rank."""
    out = []
    if not steps:
        return out
    n = min(len(v) for v in steps.values())
    for i in range(n):
        g = {r: v[i]["gpu_ms"] for r, v in steps.items()}
        h = {r: v[i]["host_ms"] for r, v in steps.items()}
        hs = {r: v[i]["host_start_ms"] for r, v in steps.items()}
        rh = max(h, key=h.get)
        out.append({"i": i, "gpu_max": max(g.values()), "gpu_min": min(g.values()), "host_max": h[rh], "host_max_rank": rh,
                    "host_start_max": max(hs.values()), "N": next(iter(steps.values()))[i]["N"]})
    return out


def window(recs, t0, t1, key="t"):
    return [r for r in recs if t0 - 0.2 <= r.get(key, 0) <= t1 + 0.2]


def analyze(out, outlier=1.15):
    out = Path(out)
    rows = [json.loads(x) for x in open(out / "stab.jsonl")]
    timed = [r for r in rows if r["kind"] == "timed"]
    mon = [json.loads(x) for x in open(out / "monitor.jsonl")] if (out / "monitor.jsonl").exists() else []
    mon = [m for m in mon if "g" in m]
    procs = [json.loads(x) for x in open(out / "procs.jsonl")] if (out / "procs.jsonl").exists() else []
    locks = [json.loads(x) for x in open(out / "locks.jsonl")] if (out / "locks.jsonl").exists() else []
    cpu = [json.loads(x) for x in open(out / "cpu.jsonl")] if (out / "cpu.jsonl").exists() else []
    seen_w, seen_c = set(), set()
    for r in timed:
        r["new_prompt_worker"] = r["pname"] not in seen_w
        r["new_prompt_cfg"] = (r["cfg"], r["pname"]) not in seen_c
        seen_w.add(r["pname"]); seen_c.add((r["cfg"], r["pname"]))
        steps, dec = parse_lines(r.get("lines", []))
        ev = r["timings"].get("kfd_evicted_ms")
        r["evicted_max_ms"] = max(ev.values()) if ev else None
        r["steps"] = step_summary(steps)
        mem = dec.pop("_mem", {})
        r["dec"] = dec
        if mem:
            r["gc"] = {"max_rank_s": max(m.get("gc_s", 0) for m in mem.values()), "sum_ranks_s": sum(m.get("gc_s", 0) for m in mem.values()),
                       "max_pause_s": max(m.get("gc_max_s", 0) for m in mem.values()),
                       "gen2_n": None, "alloc_retries": max(m.get("num_alloc_retries") or 0 for m in mem.values())}
        mw = window(mon, r["t0"], r["t1"])
        if mw:
            gs = sorted({g for m in mw for g in m["g"]})
            r["mon"] = {"n": len(mw),
                        "clk_min": {g: min((m["g"][g]["clk"] or 0) for m in mw) for g in gs},
                        "clk_med": {g: statistics.median([(m["g"][g]["clk_mean"] or 0) for m in mw]) for g in gs},
                        "pw_max": {g: max((m["g"][g]["pw"] or 0) for m in mw) for g in gs},
                        "T_max": {g: max((m["g"][g]["T"] or 0) for m in mw) for g in gs}}
        pw = [p for p in procs if r["t0"] - 3.5 <= p["t"] <= r["t1"] + 0.5 and "foreign" in p]
        fr = {}
        for p in pw:
            for f in p["foreign"]:
                if f["vram_mb"] >= 256:
                    fr[(f["gpu"], f["pid"])] = f
        r["foreign"] = list(fr.values())
        cw = [x for x in cpu if r["t0"] - 0.2 <= x["t"] <= r["t1"] + 3.5 and "busy" in x]
        if cw:
            fcs = [x["foreign_cores"] for x in cw if "foreign_cores" in x]
            r["cpu"] = {"busy_max": max(x["busy"] for x in cw), "load1_max": max(x["load1"] for x in cw),
                        "foreign_cores_max": max(fcs) if fcs else None,
                        "kcompactd_cores": max([sum(u["cores"] for u in x.get("top", []) if u["name"].startswith("kcompactd"))
                                                for x in cw] or [0]),
                        "top_foreign": sorted({(u["name"], u["pid"]) for x in cw for u in x.get("top", []) if not u["ours"]
                                               and u["cores"] >= 2})[:6]}
        lw = [x for x in locks if r["t0"] - 5.5 <= x["t"] <= r["t1"] + 0.5]
        r["lock_status"] = sorted({ln for x in lw for ln in x["status"].splitlines() if "HELD" in ln.upper() or "WAIT" in ln.upper()})
    groups = {}
    for r in timed:
        groups.setdefault((r["cfg"], r["d"]), []).append(r)
    summ = {"outlier_factor": outlier, "n_timed": len(timed), "groups": {}, "outliers": []}
    cfg_order = []
    for r in timed:
        if r["cfg"] not in cfg_order:
            cfg_order.append(r["cfg"])
    for (c, d), rs in sorted(groups.items(), key=lambda kv: (cfg_order.index(kv[0][0]), kv[0][1])):
        lat = [r["latency"] for r in rs]
        p50 = statistics.median(lat)
        outs = [r for r in rs if r["latency"] > outlier * p50]
        summ["groups"][f"{c}|{d}"] = {"cfg": c, "d": d, "n": len(rs), "p50": p50, "p95": pct(lat, 0.95), "max": max(lat),
                                     "min": min(lat), "n_out": len(outs),
                                     "samp_p50": statistics.median([r["timings"].get("sampling_seconds", 0) for r in rs]),
                                     "dec_p50": statistics.median([r["timings"].get("decode_seconds", 0) for r in rs]),
                                     "foreign_cores_max": max([(r.get("cpu") or {}).get("foreign_cores_max") or 0 for r in rs]),
                                     "evicted_max_ms": (max(r["evicted_max_ms"] for r in rs if r.get("evicted_max_ms") is not None)
                                                        if any(r.get("evicted_max_ms") is not None for r in rs) else "n/a"),
                                     "kcompactd_max": max([(r.get("cpu") or {}).get("kcompactd_cores") or 0 for r in rs])}
        for r in outs:
            base = {i: statistics.median([x["steps"][i]["gpu_max"] for x in rs if len(x["steps"]) > i]) for i in range(6)}
            summ["outliers"].append({"i": r["i"], "cfg": c, "d": d, "pname": r["pname"], "latency": r["latency"], "x_p50": r["latency"] / p50,
                                     "new_prompt_worker": r["new_prompt_worker"], "new_prompt_cfg": r["new_prompt_cfg"],
                                     "cond": r["timings"].get("conditioning_seconds"), "samp": r["timings"].get("sampling_seconds"),
                                     "dec": r["timings"].get("decode_seconds"), "mp4": r["timings"].get("mp4_seconds"),
                                     "steps": r["steps"], "step_p50": base, "mon": r.get("mon"), "foreign": r["foreign"],
                                     "lock_status": r["lock_status"], "cpu": r.get("cpu"),
                                     "new_kernels": r["timings"].get("new_kernels"), "gc": r.get("gc"),
                                     "kfd_evicted_ms": r["timings"].get("kfd_evicted_ms")})
    # interference over the whole run
    allfr = {}
    for p in procs:
        for f in p.get("foreign", []):
            if f["vram_mb"] >= 256:
                allfr.setdefault((f["gpu"], f["pid"], f["name"]), []).append(f["vram_mb"])
    summ["foreign_all"] = [{"gpu": g, "pid": p, "name": n, "vram_mb_max": max(v), "samples": len(v)} for (g, p, n), v in allfr.items()]
    if mon and timed:
        busy = [m for m in mon if any(r["t0"] <= m["t"] <= r["t1"] for r in timed)]
        gs = sorted({g for m in busy for g in m["g"]})
        summ["clock_busy"] = {g: {"p5": pct([m["g"][g]["clk_mean"] or 0 for m in busy], 0.05),
                                  "p50": pct([m["g"][g]["clk_mean"] or 0 for m in busy], 0.5),
                                  "pw_max": max(m["g"][g]["pw"] or 0 for m in busy),
                                  "T_max": max(m["g"][g]["T"] or 0 for m in busy)} for g in gs}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    (out / "requests_analyzed.json").write_text(json.dumps(timed, indent=0))
    L = [f"# Stability run {out}", "", f"{len(timed)} timed requests, outlier = latency > {outlier} x p50 of the same config and duration.",
         "", "| cfg | dur | n | p50 s | p95 s | max s | min s | outliers | sampling p50 | decode p50 | max foreign CPU cores | max kcompactd cores | max KFD evicted ms (any rank) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in summ["groups"].values():
        L.append(f"| {g['cfg']} | {g['d']} | {g['n']} | {g['p50']:.3f} | {g['p95']:.3f} | {g['max']:.3f} | {g['min']:.3f} | {g['n_out']} | "
                 f"{g['samp_p50']:.3f} | {g['dec_p50']:.3f} | {g['foreign_cores_max']} | {g['kcompactd_max']} | {g['evicted_max_ms']} |")
    L += ["", "## Outliers", ""]
    if not summ["outliers"]:
        L.append("none")
    for o in summ["outliers"]:
        st = " ".join(f"s{s['i']}={s['gpu_max']:.0f}({o['step_p50'].get(s['i'], 0):.0f})/h{s['host_max']:.0f}@r{s['host_max_rank']}"
                      for s in o["steps"])
        mon_s = ""
        if o["mon"]:
            mon_s = f" clk_min {min(o['mon']['clk_min'].values())} MHz, pw_max {max(o['mon']['pw_max'].values())} W"
        L.append(f"- #{o['i']} {o['cfg']} {o['d']} s {o['pname']}: {o['latency']:.2f} s ({o['x_p50']:.2f}x p50), KFD queue eviction ms per "
                 f"rank {o['kfd_evicted_ms']}, kcompactd cores {(o['cpu'] or {}).get('kcompactd_cores')}, new prompt "
                 f"worker/cfg {o['new_prompt_worker']}/{o['new_prompt_cfg']}; cond {o['cond']:.3f} samp {o['samp']:.3f} dec {o['dec']:.3f} "
                 f"mp4 {o['mp4']:.3f}; steps gpu-max ms (p50) / host-max ms @rank: {st};{mon_s}; foreign {o['foreign'] or 'none'}; "
                 f"locks {o['lock_status'] or 'n/a'}; cpu {o['cpu']}; python GC {o['gc']}; Triton kernels compiled in this request: {o['new_kernels'] or 'none'}")
    jit = [(r["i"], r["cfg"], r["d"], r["pname"], round(r["latency"], 3), r["timings"].get("new_kernels")) for r in timed
           if r["timings"].get("new_kernels")]
    summ["jit_requests"] = jit
    gcs = [r["gc"]["max_pause_s"] for r in timed if r.get("gc")]
    if gcs:
        summ["gc_pause_max_s"] = max(gcs)
        summ["gc_pause_p95_s"] = pct(gcs, 0.95)
    L += ["", "## Timed requests that compiled Triton kernels (new cache entries)", ""]
    L += [f"- #{i} {c} {d} s {p}: {lat} s, {k}" for i, c, d, p, lat, k in jit] or ["none"]
    if gcs:
        L += ["", f"Python GC: longest single pause on any rank per request p95 {summ['gc_pause_p95_s'] * 1e3:.1f} ms, max {summ['gc_pause_max_s'] * 1e3:.1f} ms"]
    L += ["", "## Interference over the run", "", f"foreign GPU processes (>= 256 MB, not in the worker tree): {summ['foreign_all'] or 'none'}"]
    if "clock_busy" in summ:
        L.append("")
        L.append("| GPU | gfx clk p5 / p50 MHz (during requests) | max power W | max hotspot C |")
        L.append("|---|---|---|---|")
        for g, v in summ["clock_busy"].items():
            L.append(f"| {g} | {v['p5']:.0f} / {v['p50']:.0f} | {v['pw_max']} | {v['T_max']} |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--out", help="output directory (created; with --analyze, pass the run directory to --analyze instead)")
    ap.add_argument("--cfg", action="append", help="lossless | name=preset:<preset name> | name=<json> | name=@file.json")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--prompts", default=str(HERE / "stab_prompts.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=25)
    ap.add_argument("--outlier", type=float, default=1.15)
    ap.add_argument("--wait", type=int, default=300)
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--env", action="append")
    ap.add_argument("--warmup-prompt", help="text file with the untimed warm-up prompt (default: a built-in prompt)")
    ap.add_argument("--keep-mp4", action="store_true")
    ap.add_argument("--provoke", help="GB:start_s:dur_s - run tools/evict_provoke.py (THP allocations -> host compaction) during the timed phase")
    ap.add_argument("--analyze")
    ap.add_argument("--inner", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze, a.outlier)
        return
    if a.inner:
        run_inner(a)
        return
    if not a.out:
        ap.error("--out is required (or --analyze <dir>)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfgs = [parse_cfg(c) for c in (a.cfg or ["lossless"])]
    (out / "cfgs.json").write_text(json.dumps(cfgs, indent=1))
    sched = build_schedule(a, cfgs)
    (out / "schedule.json").write_text(json.dumps(sched, indent=0))
    head = subprocess.run(["git", "-C", a.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", a.repo, "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip()
    (out / "meta.json").write_text(json.dumps({"repo": a.repo, "head": head, "src_dirty": bool(dirty), "argv": sys.argv,
                                               "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    print(f"[stab] {len(sched)} timed requests ({a.rounds} rounds x {len(cfgs)} cfgs), repo {a.repo} {head[:8]}"
          f"{' +dirty src' if dirty else ''}; driver pid {os.getpid()}", flush=True)
    lock = (["gpu_lock", *a.gpus.split(","), "--wait", str(a.wait), "--"] if shutil.which("gpu_lock")
            else ["env", f"HIP_VISIBLE_DEVICES={a.gpus}"])
    cmd = ["timeout", str(int(a.max_minutes * 60 + 1500)), *lock,
           VENVPY, str(Path(__file__).resolve()), "--inner", "--out", str(out), "--gpus", a.gpus, "--repo", a.repo,
           "--max-minutes", str(a.max_minutes)]
    for e in a.env or []:
        cmd += ["--env", e]
    if a.keep_mp4:
        cmd.append("--keep-mp4")
    if a.provoke:
        cmd += ["--provoke", a.provoke]
    if a.warmup_prompt:
        cmd += ["--warmup-prompt", a.warmup_prompt]
    rc = subprocess.call(cmd)
    print(f"[stab] worker phase rc {rc}", flush=True)
    if (out / "stab.jsonl").exists():
        analyze(out, a.outlier)


if __name__ == "__main__":
    main()
