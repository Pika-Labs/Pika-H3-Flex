# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU request engine: one process per GPU, SPMD over the six generator evaluations.

Rank 0 is the serving process (stdin/stdout, text encoder, MP4 writer); ranks 1.. are helper processes started by
rank 0 (``python -m pika_h3.rank_main``). All collectives run on the same ordered sequence of calls on every rank.
"""
from __future__ import annotations

import fcntl
import os
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist
from .paths import load_lock_dir

KNOTS = [1000, 750, 500, 250, 100, 40, 0]
REVISION = "0ddf55ad7067ca1e76812253bd7b4600df87f759"


_BUCKETS = None


def _bucket_table():
    """The static resolution/aspect bucket table (core/buckets.yaml), parsed once (was ~3 ms of yaml per request)."""
    global _BUCKETS
    if _BUCKETS is None:
        import yaml
        _BUCKETS = yaml.safe_load((Path(__file__).parent / "core" / "buckets.yaml").read_text())
    return _BUCKETS


class _LoadSlot:
    """At most two ranks load model weights at the same time (host RAM)."""

    def __enter__(self):
        while True:
            for i in range(2):
                f = open(os.path.join(load_lock_dir(), f".load_slot_{i}"), "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.f = f
                    return self
                except OSError:
                    f.close()
            time.sleep(0.5)

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def init_dist(rank, world, dev):
    torch.cuda.set_device(dev)
    import datetime
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", dev),
                            timeout=datetime.timedelta(seconds=int(os.environ.get("PIKA_NCCL_TIMEOUT", "600"))))
    ctl = dist.new_group(backend="gloo")
    return ctl


class Engine:
    def __init__(self, rank, world, base_path, adapter_path, ctl, dev):
        import argparse
        from .runtime import validate_runtime
        from .pipeline import resolve_assets
        from .core.numerics import configure_numerics_backend
        from .core.loader import apply_lora, load_lora_weights
        from .core.schedules import H3SigmaSchedule
        from .core.wrapper import H3DiffusionWrapper
        from .sp_model import SPTransformer, SPWrapper

        validate_runtime()
        from .paths import check_aiter
        check_aiter()  # clear error at start-up instead of an ImportError at the first attention call
        self.rank, self.world, self.ctl = rank, world, ctl
        torch.set_num_threads(8)
        self.dev = dev
        self.device = torch.device("cuda", dev)
        args = argparse.Namespace(task="t2va", adapter_path=Path(adapter_path), base_path=Path(base_path),
                                  model_repo="Pika-Labs/Pika-H3-Flex", revision=REVISION)
        t0 = time.monotonic()
        base, adapter, manifest = resolve_assets(args)
        self.manifest, self.policy = manifest, manifest["numerics_policy"]
        configure_numerics_backend(self.policy, runtime="inference")
        # MIOpen picks its conv solver from the workspace that happens to be available; deterministic mode makes the decoded
        # waveform bitwise reproducible (the AAC encoder's quality is chaotic in 1e-7 changes of its input).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.encoder = None
        if rank == 0:
            from .core.conditioning import H3ConditionEncoder
            self.encoder = H3ConditionEncoder(str(base), device=self.device, keep_loaded=True)
            self.encoder._load_text_encoder()
            self.encoder.keep_on_device = True  # prompt embeddings stay on the GPU (no device -> host -> device round trip)
            _bucket_table()
        with _LoadSlot():
            from diffusers import MiniMaxH3Transformer3DModel
            import diffusers.models.modeling_utils as _mu
            _mu._caching_allocator_warmup = lambda *a, **k: None  # would reserve the whole model size in one block
            model = MiniMaxH3Transformer3DModel.from_pretrained(
                str(base), subfolder=manifest["trunk"], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                device_map={"": dev, **{f"transformer_blocks.{i}.adaln_proj": "cpu" for i in range(50)}})
            model.requires_grad_(False)
            apply_lora(model, rank=manifest["rank"], alpha=manifest["alpha"], target_modules=manifest.get("target_modules"),
                       master_dtype=torch.bfloat16)
            load_lora_weights(model, str(adapter))
            model.requires_grad_(False).eval()
            for blk in model.transformer_blocks:  # AdaLN projections are only needed for the fixed-schedule tables
                blk.adaln_proj = None
            import gc
            gc.collect()
            model.to(self.device)
            torch.cuda.empty_cache()
        self.sp = SPTransformer(model, rank, world, ckpt_dir=str(Path(base) / manifest["trunk"]))
        self.wrapper = SPWrapper(self.sp).eval()
        self.schedule = H3SigmaSchedule({"denoising_step_list": KNOTS, "video_score_sigma_shift": 12, "audio_score_sigma_shift": 3},
                                        device=self.device)
        self.base = base
        # AdaLN tables for the six fixed schedule timesteps (independent of prompt and seed)
        for i in range(len(self.schedule.video_ladder) - 1):
            sv = float(self.schedule.video_ladder[i].float().reshape(-1)[0])
            sa = float(self.schedule.audio_ladder[i].float().reshape(-1)[0])
            self.sp.timestep_tables(1.0 - sv, 1.0 - sa, torch.bfloat16, self.device)
        self.sp.free_adaln(str(Path(base) / manifest["trunk"]))
        from .core.vae import H3Decoders
        from .pdecode import TileDecoder
        with _LoadSlot():
            self.decoders = H3Decoders(str(base), device=self.device)
        if os.environ.get("PIKA_AUDIO_GEMM", "1") == "1":  # dilated Conv1d as fp32 GEMM instead of MIOpen's naive kernel (294 -> 88 ms at 5 s)
            from .audio_conv import patch_dilated_conv1d
            patch_dilated_conv1d(self.decoders.audio_vae.decoder)
        self.tdec = TileDecoder(self.decoders, rank, world)
        self.audio_rank = world - 1
        self.speeds = [1.0] * world
        # Speed-proportional partitioning (PIKA_BALANCE=1) did not help on GPUs shared with other users' jobs (their load fluctuates
        # faster than the per-layer synchronisation): the default is the uniform split.
        if os.environ.get("PIKA_BALANCE", "0") == "1":
            self.apply_speeds(self.probe_speeds())
        if rank == 0:
            print(f"[engine] speeds {[round(x, 2) for x in self.speeds]} heads {self.sp.hcnt} audio rank {self.audio_rank}", flush=True)
        torch.cuda.synchronize()
        print(f"[engine] rank {rank} dev {dev}: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        dist.barrier()
        self.load_seconds = time.monotonic() - t0

    def probe_speeds(self, ms=200):
        """Relative throughput of every rank's GPU right now (other users' jobs time-share these GPUs)."""
        import math
        env = os.environ.get("PIKA_SPEEDS")
        if env:
            sp = [float(x) for x in env.split(",")]
            return sp
        a = torch.randn(4096, 5376, device=self.device, dtype=torch.bfloat16)
        b = torch.randn(5376, 5376, device=self.device, dtype=torch.bfloat16)
        for _ in range(4):
            torch.mm(a, b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < ms / 1000:
            for _ in range(8):
                torch.mm(a, b)
            torch.cuda.synchronize()
            n += 8
        rate = n / (time.perf_counter() - t0)
        gathered = [None] * self.world
        dist.all_gather_object(gathered, rate, group=self.ctl)
        best = max(gathered)
        return [g / best for g in gathered]

    def apply_speeds(self, speeds):
        self.speeds = speeds
        self.sp.set_partition(speeds)
        # audio VAE runs on the fastest rank
        self.audio_rank = max(range(self.world), key=lambda r: speeds[r])

    # ------------------------------------------------------------------ control channel
    def barrier(self):
        dist.barrier(group=self.ctl)

    def bcast_obj(self, obj):
        box = [obj]
        dist.broadcast_object_list(box, src=0, group=self.ctl)
        return box[0]

    # ------------------------------------------------------------------ one request (collective)
    def _condition(self, req):
        from .core.conditioning import Request
        from .core.numerics import prepare_conditioning
        from .core import geometry as G
        timings = {}
        self._mark("cond_entry")
        table = _bucket_table()
        tier = "p" + req["resolution"][:-1]
        width, height = table["tiers"][tier][table["aspects"].index(req["aspect"])]
        frames = G.align_num_frames(int(req["duration"]) * G.FPS)
        self._mark("yaml_done")
        t0 = time.monotonic()
        err = None
        if self.rank == 0:
            try:
                request = Request(task="t2va", prompt=req["prompt"], first_image=None, last_image=None, references=[],
                                  height=height, width=width, num_frames=frames, reference_rule="tiered", keyframe_fit="crop")
                cond = self.encoder.encode(request)
                meta = {"geom": cond["geometry"], "L": int(cond["prompt_embeds"].shape[1])}
            except Exception as e:  # noqa: BLE001  rank-0-only failure (e.g. a prompt over MAX_TEXT_TOKENS): every rank
                err = e                # must leave the request at the same collective, else the next request desyncs
                meta = {"error": f"{type(e).__name__}: {e}"}
        else:
            meta = None
        self._mark("encoded")
        meta = self.bcast_obj(meta)
        self._mark("meta_bcast")
        if "error" in meta:  # every rank leaves here: a REJECTED request (faults.py), the ranks stay in step
            from .faults import RequestRejected
            if err is not None:
                raise RequestRejected(meta["error"]) from err
            raise RequestRejected(f"request rejected on rank 0: {meta['error']}")
        geom, L = meta["geom"], meta["L"]
        if self.rank == 0:
            embeds = cond["prompt_embeds"].to(self.device, torch.bfloat16)
            tags = cond["text_token_tags"].to(self.device)
        else:
            embeds = torch.empty((1, L, 5120), dtype=torch.bfloat16, device=self.device)
            tags = torch.empty((L,), dtype=torch.long, device=self.device)
        dist.broadcast(embeds, src=0)
        dist.broadcast(tags, src=0)
        cond_dict = {"prompt_embeds": embeds, "text_token_tags": tags, "keyframes": [], "references": [],
                     "geometry": geom}
        timings["conditioning_seconds"] = time.monotonic() - t0
        self._mark("cond_done")
        return cond_dict, geom, timings, (frames, height, width)

    def sample(self, req, trace=None):
        """Returns (video [1,F,24,H,W] fp32, audio [1,T,64] fp32) on every rank, and timings (rank 0)."""
        from .core.evaluation import sample_evaluation_ladder
        from .core import geometry as G

        cond_dict, geom, timings, shape = self._condition(req)
        t0 = time.monotonic()
        generator = self.wrapper
        from . import sp_model as _SPM
        if _SPM._STEPLOG:  # dev: per-step event timing, printed per rank after the sampling synchronize
            cond_dict["_steplog"] = []
        from . import faults as _F
        if _F.FAULT is not None and _F.FAULT[0] == self.rank:  # testing only (PIKA_FAULT unset: not installed)
            generator = _F.FaultyGenerator(generator, self.rank)
        if trace is not None:
            from .serve import TracingGenerator
            generator = TracingGenerator(generator, trace)
        with torch.inference_mode():
            video, audio = sample_evaluation_ladder(
                generator, conditional_dict=cond_dict,
                video_shape=[1, geom.num_latent_frames, G.VIDEO_LATENT_CHANNELS, geom.latent_height, geom.latent_width],
                audio_shape=[1, geom.num_audio_latents, G.AUDIO_EXTERNAL_DIM],
                sigmas_v=self.schedule.video_ladder, sigmas_a=self.schedule.audio_ladder,
                seed=int(req["seed"]), device=self.device)
        torch.cuda.synchronize()
        timings["sampling_seconds"] = time.monotonic() - t0
        if _SPM._STEPLOG:
            import json as _json
            import sys as _sys
            _ms = torch.cuda.memory_stats()
            _mem = {k: _ms.get(k) for k in ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms",
                                            "reserved_bytes.all.current", "reserved_bytes.all.peak", "allocated_bytes.all.peak",
                                            "inactive_split_bytes.all.current")}
            _gcl = list(_SPM.GC_LOG)
            _SPM.GC_LOG.clear()
            _mem["gc_n"] = len(_gcl)
            _mem["gc_s"] = round(sum(x[1] for x in _gcl), 4)
            _mem["gc_gen2_s"] = round(sum(x[1] for x in _gcl if x[0] == 2), 4)
            _mem["gc_max_s"] = round(max([x[1] for x in _gcl], default=0.0), 4)
            print("[steplog] " + _json.dumps({"rid": str(req.get("id")), "rank": self.rank, "mem": _mem,
                                               "steps": _SPM.steplog_report(cond_dict.get("_steplog"))}), file=_sys.stderr, flush=True)
        self._mark("sampled")
        return video, audio, timings, shape

    def _mark(self, name):
        if self.rank == 0 and os.environ.get("PIKA_TIMING") == "4":
            t = time.monotonic()
            self._t4.append(f"{name}={(t - self._t4_0) * 1e3:.1f}")

    def run_request(self, req):
        """Collective. Rank 0 returns the timings dict after the MP4 is complete; other ranks return None."""
        self._t4_0, self._t4 = time.monotonic(), []
        self.phase = "collective"  # until the last collective of the request (faults.py: failures here are fatal)
        from . import faults as _F
        if _F.FAULT is not None:
            _F.new_request()
        import numpy as np
        from .core import geometry as G
        from .serve import write_trace
        from .mp4 import encode_audio, encode_segment, mux_segments

        from . import lossy as _LS
        if _LS.ACTIVE or "lossy" in req:  # LOSSY track (off by default): per-request config, same on every rank
            _LS.for_request(req)
        if "dev_sdma" in req and getattr(self.sp, "sdma", None) is not None:  # dev only: in-worker A/B of the SDMA path
            self.sp.sdma_on = bool(req["dev_sdma"])
        if "dev_pipe" in req and getattr(self.sp, "sdma", None) is not None:  # dev only: SDMA schedule per request
            self.sp.pipe = int(req["dev_pipe"])
        if "dev_pipe_short" in req and getattr(self.sp, "sdma", None) is not None:  # dev only: schedule below SDMA_MIN_N rows
            self.sp.pipe_short = int(req["dev_pipe_short"])
        if "dev_sdma_fa" in req:  # dev only
            from . import sp_model as _spm
            _spm.SDMA_FA = float(req["dev_sdma_fa"])
        if "dev_gemm_pad_max" in req:  # dev only
            from . import sp_model as _spm
            _spm.GEMM_PAD_MAX = int(req["dev_gemm_pad_max"])
        if "dev_sols_drop" in req:  # dev only: ignore the tuned GEMM solutions keyed at these row counts (in-worker A/B)
            from . import gemm_sel as _G
            _G._init()
            drop = {int(x) for x in str(req["dev_sols_drop"]).split("/") if x}
            if not hasattr(_G, "_BYNK_ALL"):
                _G._BYNK_ALL = {k: list(v) for k, v in _G._BYNK.items()}
            for k, v in _G._BYNK_ALL.items():
                _G._BYNK[k] = [x for x in v if x[0] not in drop]
            _G._CACHE.clear()
        if "dev_split" in req:  # dev only: PIKA_SPLIT on/off per request (needs PIKA_SPLIT != 0 at start)
            self.sp.split_off = not bool(req["dev_split"])
        if "dev_sdma_min_n" in req:
            from . import sp_model as _spm
            _spm.SDMA_MIN_N = int(req["dev_sdma_min_n"])
        if "dev_skip" in req:  # dev only (wrong output): "f/r1/r2..." attention query-tile skip on ranks r1.. (timing bound)
            from . import sp_model as _spm
            parts = str(req["dev_skip"]).split("/")
            _spm.DEV_SKIP = float(parts[0])
            _spm.DEV_SKIP_RANKS = {int(x) for x in parts[1:] if x}
        if "dev_noxfer" in req:  # dev only (wrong output)
            from . import sp_model as _spm
            _spm.DEV_NOXFER = int(req["dev_noxfer"])
        trace = [] if (self.rank == 0 and req.get("trace_dir")) else None
        want_trace = bool(req.get("trace_dir")) or bool(req.get("dump_pixels"))
        if req.get("load_latents"):  # dev only: decode saved final latents instead of sampling
            L = torch.load(req["load_latents"])
            video, audio = L["video"].to(self.device).unsqueeze(0), L["audio"].to(self.device).unsqueeze(0)
            timings = {}
            from .core import geometry as G0
            frames = G0.align_num_frames(int(req["duration"]) * G0.FPS)
            height, width = 768, 1344
        else:
            video, audio, timings, (frames, height, width) = self.sample(req, trace)
        if req.get("save_latents") and self.rank == 0:  # dev only: final latents for decode checks
            torch.save({"video": video[0].cpu(), "audio": audio[0].cpu()}, req["save_latents"])
        self._mark("dec_start")
        t0 = time.monotonic()
        w = self.world
        ar = self.audio_rank
        wave_np = None  # decoded waveform (audio rank), or received on rank 0 (float32 [2, n_samples] CPU)
        wreq = None
        aud_thread = aud_path = None
        n_samples = audio.shape[1] * G.AUDIO_HOP
        # Only rank 0 needs the waveform (MP4 mux): a point-to-point CPU transfer over the gloo control group replaces the former
        # NCCL broadcast to every rank (one NCCL collective less in the decode path, no GPU-stream coupling).
        if ar != 0 and self.rank == 0:
            wave_np = torch.empty((G.AUDIO_CHANNELS, n_samples), dtype=torch.float32)
            wreq = dist.irecv(wave_np, src=ar, group=self.ctl)
        with torch.inference_mode():
            weights = [float(x) for x in self.speeds]
            # audio rank's tile share: audio decode ~= 12-13% of a full rank's tile time with the GEMM convs (was ~40%: 0.5)
            # World >= 8: the audio decode costs c ~ 0.05 of the total tile work (0.09 / 0.25 / 0.34 s at 5 / 10 / 15 s), so the
            # audio rank's equal-time share is 1 - c*W/(1+c) = 0.62 at W=8 (0.85 was tuned at world 4; at world 8 clip owners waited
            # for the audio rank's tiles: in-worker A/B decode 5 s 0.40 -> 0.37, 10 s 0.745 -> 0.725, 15 s 1.10 -> 1.06 s).
            aw = "0.85" if os.environ.get("PIKA_AUDIO_GEMM", "1") == "1" else "0.5"
            if w >= 8 and aw == "0.85":
                c = float(os.environ.get("PIKA_AUDIO_COST", "0.05"))
                aw = str(1.0 - c * w / (1.0 + c))
            weights[ar] *= float(os.environ.get("PIKA_AUDIO_TILE_WEIGHT", aw))
            dw = req.get("dev_dec_weights") or os.environ.get("PIKA_DEC_WEIGHTS")  # dev: per-rank tile-decode weight multipliers
            if dw:
                weights = [x * float(m) for x, m in zip(weights, dw.split(","))]
            tm = {}
            tm["t_start"] = time.monotonic()
            if self.rank == ar:
                wave = self._decode_audio_checked(audio[0])
                wave_np = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))
                if tuple(wave_np.shape) != (G.AUDIO_CHANNELS, n_samples):
                    raise RuntimeError(f"decoded audio shape {tuple(wave_np.shape)} != {(G.AUDIO_CHANNELS, n_samples)}")
                if ar != 0:
                    wreq = dist.isend(wave_np, dst=0, group=self.ctl)
                # AAC-encode on a CPU thread while the tiles decode; rank 0 only copies the packets at the end
                aud_path = f"{req['output']}.aud.mp4"
                Path(req["output"]).parent.mkdir(parents=True, exist_ok=True)
                n_keep = round(frames / G.FPS * G.AUDIO_SAMPLE_RATE)
                aud_thread = threading.Thread(target=encode_audio, args=(aud_path, wave_np.numpy()[:, :n_keep], G.AUDIO_SAMPLE_RATE))
                aud_thread.start()
            tm["t_audio"] = time.monotonic()
            if _F.FAULT is not None:
                _F.fire(self.rank, "decode")
            pix = self.tdec.decode(video[0], weights=weights, gather=want_trace)
            torch.cuda.synchronize()
            tm["t_tiles"] = time.monotonic()
        output = Path(req["output"])
        seg_path = None
        seg = self.tdec.segment
        if seg is not None:
            seg_path = f"{output}.seg{self.rank}.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            nth = int(os.environ.get("PIKA_X264_THREADS", "8"))
            if os.environ.get("PIKA_GPU_YUV", "1") == "1":  # rgb -> yuv420p on the GPU, x264 gets planar frames
                from .mp4 import encode_segment_yuv, rgb_to_yuv420p
                planes = rgb_to_yuv420p(seg[1]).cpu().numpy()
                encode_segment_yuv(seg_path, planes, seg[1].shape[2], seg[1].shape[1], threads=nth)
            else:
                torch.cuda.synchronize()
                encode_segment(seg_path, seg[1].cpu().numpy(), threads=nth)
        tm["t_enc"] = time.monotonic()
        pre = None
        if self.rank == 0 and os.environ.get("PIKA_MUX_PREFETCH", "1") == "1" and getattr(self.tdec, "seg_plan", None):
            from .mp4 import MuxPrefetch  # opens the other ranks' segments as they appear
            pre = MuxPrefetch([(st0, f"{output}.seg{r}.mp4") for st0, r in self.tdec.seg_plan], f"{output}.aud.mp4")
        if os.environ.get("PIKA_TIMING"):
            print(f"[timing] rank {self.rank}: audio {tm['t_audio']-tm['t_start']:.3f} tiles {tm['t_tiles']-tm['t_audio']:.3f} enc {tm['t_enc']-tm['t_tiles']:.3f}", flush=True)
        starts = [(seg[0] if seg is not None else -1)]
        gathered = [None] * self.world
        if aud_thread is not None:
            aud_thread.join()
        dist.all_gather_object(gathered, starts[0], group=self.ctl)  # also the barrier: all segments are on disk
        if wreq is not None:
            wreq.wait()
        self.phase = "local"  # past the last collective: a failure below is local to this rank (rank 0: keep serving)
        if self.rank != 0:
            return None
        t1 = time.monotonic()
        timings["decode_seconds"] = t1 - t0
        wave = wave_np.numpy()
        if not np.isfinite(wave).all():
            raise RuntimeError("decoded audio is not finite")
        wave = wave[:, :round(frames / G.FPS * G.AUDIO_SAMPLE_RATE)]
        segs = sorted((g, f"{output}.seg{r}.mp4") for r, g in enumerate(gathered) if g >= 0)
        aud_path = f"{output}.aud.mp4"  # written by the audio rank (its thread was joined before the barrier above)
        if pre is not None and [(a, b) for a, b in segs] == [(st0, f"{output}.seg{r}.mp4") for st0, r in self.tdec.seg_plan]:
            try:
                pre.mux(str(output))
            except Exception as e:  # noqa: BLE001
                print(f"[engine] WARNING: mux prefetch failed ({e!r}); plain mux", flush=True)
                pre.close()
                mux_segments(str(output), segs, None, audio_file=aud_path)
        else:
            if pre is not None:
                print("[engine] WARNING: segment plan mismatch; plain mux", flush=True)
                pre.join()
                pre.close()
            mux_segments(str(output), segs, None, audio_file=aud_path)
        os.unlink(aud_path)
        for _, p in segs:
            os.unlink(p)
        timings["mp4_seconds"] = time.monotonic() - t1
        timings["decode_encode_seconds"] = time.monotonic() - t0
        self._mark("dec_done")
        if _F.FAULT is not None:
            _F.fire(self.rank, "post")  # testing: a local failure after the MP4 (e.g. the trace write)
        if req.get("dump_pixels"):
            np.save(req["dump_pixels"], pix.cpu().numpy())
        if trace is not None:
            pixels = pix.cpu().numpy()
            if pixels.shape[:3] != (frames, height, width):
                raise RuntimeError(f"decoded geometry mismatch: {pixels.shape}")
            write_trace(Path(req["trace_dir"]), trace, pixels=pixels, wave=wave)
        return timings

    def _decode_audio_checked(self, audio_lat):
        """Audio VAE decode with a safety net: a non-finite waveform (seen once next to a foreign job on a shared GPU, where the
        conv workspace allocation can fail) is retried after releasing cached memory, then decoded on the CPU."""
        import numpy as np
        if not bool(torch.isfinite(audio_lat).all()):
            raise RuntimeError("sampled audio latents are not finite")
        wave = self.decoders.decode_audio(audio_lat)
        if np.isfinite(wave).all():
            return wave
        print("[engine] WARNING: decoded audio not finite; retrying after empty_cache", flush=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        wave = self.decoders.decode_audio(audio_lat)
        if np.isfinite(wave).all():
            return wave
        print("[engine] WARNING: decoded audio still not finite; decoding on CPU", flush=True)
        dec = self.decoders
        dec.audio_vae.cpu()
        dev0, dec.device = dec.device, torch.device("cpu")
        dec.audio_mean, dec.audio_std = dec.audio_mean.cpu(), dec.audio_std.cpu()
        try:
            wave = dec.decode_audio(audio_lat.cpu())
        finally:
            dec.audio_vae.to(dev0)
            dec.device = dev0
            dec.audio_mean, dec.audio_std = dec.audio_mean.to(dev0), dec.audio_std.to(dev0)
        return wave

    def teacher_forced(self, req, trace_path):
        """Dev check (collective): evaluate the sharded wrapper on the trace's step inputs. Rank 0 returns preds."""
        from safetensors.torch import load_file
        from . import lossy as _LS
        if _LS.ACTIVE or "lossy" in req:
            _LS.for_request(req)
        tr = load_file(trace_path)
        cond_dict, geom, _, _ = self._condition(req)
        out = []
        with torch.inference_mode():
            for i in range(6):
                if "tf_steps" in req and i not in req["tf_steps"]:  # dev: skip this step (returns the trace's own prediction)
                    out.append((tr[f"step{i}.video_pred"].float(), tr[f"step{i}.audio_pred"].float(), 0.0))
                    continue
                cond_dict["_tf_step"] = i
                prof = None
                if os.environ.get("PIKA_PROF") and self.rank == 0 and i == 2:
                    from torch.profiler import profile, ProfilerActivity
                    prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])
                    prof.__enter__()
                t0 = time.monotonic()
                pv, pa = self.wrapper(noisy_image_or_video=tr[f"step{i}.video_in"].to(self.device), conditional_dict=cond_dict,
                                      timestep=tr[f"step{i}.sigma_v"].to(self.device), noisy_audio=tr[f"step{i}.audio_in"].to(self.device),
                                      audio_timestep=tr[f"step{i}.sigma_a"].to(self.device))
                torch.cuda.synchronize()
                dt = time.monotonic() - t0
                if prof is not None:
                    prof.__exit__(None, None, None)
                    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=22, max_name_column_width=70), flush=True)
                out.append((pv.float().cpu(), pa.float().cpu(), dt))
        return tr, out
