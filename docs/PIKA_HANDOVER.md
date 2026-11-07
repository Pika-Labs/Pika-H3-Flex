# Pika H3 Flex on 8x AMD MI350X

This branch extends the Pika H3 Flex package with a warm, multi-GPU worker for AMD Instinct MI350X (gfx950, ROCm 7.2). One
text-to-video+audio request (768p 16:9, 5 / 10 / 15 s, the released six-step schedule, MiniMax-H3 base + the `fl2va` LoRA) is
sharded over up to eight GPUs. The default path is lossless: its output is numerically equivalent to the unmodified single-GPU
pipeline, at the level of legitimate kernel variants, and is checked step by step against the reference transformer.

| warm latency, 768p 16:9, lossless default | 5 s | 10 s | 15 s |
|---|---|---|---|
| unmodified package, 1x MI350X | 46.2 s | 112.6 s | 209.7 s |
| this worker, 8x MI350X, final verification eval (2026-11-07) | **3.28 s** | **8.73 s** | **17.04 s** |
| this worker, 8x MI350X, stability run on unseen prompts, p50 / p95 / max | 3.29 / 3.30 / 3.30 s | 8.73 / 8.75 / 8.76 s | 16.88 / 16.91 / 16.91 s |

That is a 14x / 13x / 12x speed-up over one GPU (geometric mean 13.08). Latency is measured from writing the request line to the
complete MP4 on disk: text conditioning, six-step sampling, VAE / audio decode and MP4 encode. Model loading and shape-generic
warm-up happen before the worker reports ready and are not included.

Opt-in lossy presets are 1.3-1.7x faster again; see [Lossy presets](#lossy-presets).

Contents: [What the worker does](#what-the-worker-does) · [Requirements](#requirements) · [Install](#install) ·
[Start the worker](#start-the-worker) · [Request format](#request-format) · [Failure handling](#failure-handling) ·
[Lossy presets](#lossy-presets) · [Host settings](#host-settings) · [Latency stability](#latency-stability) ·
[Environment variables](#environment-variables) · [Limitations](#known-limitations) · [Tests](#tests) · [Code map](#code-map)

## What the worker does

- **Sequence-parallel DiT.** Ulysses all-to-all over the 56 attention heads; the all-to-all runs on the SDMA copy engines at
  world 8 (no compute units used), RCCL otherwise.
- **Attention.** aiter `mha_v4` assembly flash attention (bf16, fp32 softmax and accumulation), with a wave-aligned split of the
  query rows and an LSE merge of the partial last wave.
- **GEMMs.** Per-shape hipBLASLt solutions chosen by a sustained sweep on MI350X (`src/pika_h3/data/gemm_sols.json`), the LoRA
  up-projections folded into the neighbouring fused kernels, rotary embedding / modulation / norm / SwiGLU fused in Triton and HIP.
- **Text encoder** on rank 0; the AdaLN modulation tables of the fixed six-step schedule are cached, so the AdaLN
  projection weights (~26 GB per rank) are freed.
- **Decode.** The VAE decoder's 256x256 tiles and the audio decoder are split across all ranks; rgb -> yuv420p conversion on the
  GPU; each rank encodes its segment of the H.264 stream and rank 0 muxes them.
- **Host.** Huge-page backing and NUMA-policy binding of the worker's host memory against GPU-queue evictions ([host settings](#host-settings)).

## Requirements

- 1, 2, 4, 7 or 8 AMD Instinct MI350X / MI355X GPUs (gfx950). The kernels are built for `gfx950` only (hand-written HIP
  kernels, aiter assembly kernels, the tuned GEMM table). The world size must divide the 56 attention heads.
- GPU memory per request at world 8: about 135 GB on rank 0 (it also hosts the text encoder) and about 66 GB on every other rank.
- Host: about 70 GB RAM per rank while it loads weights. At most two ranks load at the same time (file lock, see `PIKA_LOAD_LOCK_DIR`).
- ROCm 7.2 (tested: 7.2.0, HIP 7.2.53211) at `/opt/rocm` or `$ROCM_PATH`, with `hipcc`, `clang` and `ld.lld`.
- Python 3.12. Local weights: the MiniMax-H3 snapshot (revision `42ed227ee7df40d41602854ae760620d6eb651fe`) and the Pika-H3-Flex
  `fl2va` adapter (revision `0ddf55ad7067ca1e76812253bd7b4600df87f759`, see `release.json`).

## Install

These steps were checked on 2026-11-07 from a fresh clone, a fresh venv built only from these files and a fresh aiter clone; the
CPU and world-2 tests passed and an 8-GPU run reproduced the reference MP4s bit for bit.

```bash
git clone <this repository> pika-h3 && cd pika-h3
python3.12 -m venv .venv && source .venv/bin/activate      # or: uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -r requirements-rocm-torch.txt    # torch 2.13.0+rocm7.2, torchvision, triton-rocm from the ROCm 7.2 index
pip install -r requirements-rocm.txt          # everything else, pinned to the tested environment
python tools/apply_diffusers_patches.py       # the two Diffusers patches of the original package
pip install --no-deps -e .
```

- A venv made with `uv venv` has no `pip`: use `uv pip install ...` for the three `pip install` lines.
- `TMPDIR` must point at a filesystem with a few GB free: the ROCm code-object manager (comgr) writes a 0.3-0.6 GB JIT directory
  per process there on a fresh host, and a full `/tmp` fails with `No space left on device`. A clean environment (`env -i`)
  drops `TMPDIR`, so set it again explicitly.
- `requirements.txt`, `environment.yml` and the `Dockerfile` are the original NVIDIA recipes (CUDA torch). Do not use them on ROCm.
- `requirements-extra.txt` lists the build helpers on top of the original requirements (already included in `requirements-rocm.txt`).

**aiter (required).** The default path uses ROCm aiter: the `mha_v4` attention kernels, the hipBLASLt solution GEMM
(`gradlib.hipb_mm`), and the Triton attention in the VAE decoder. Tested revision: ROCm/aiter `32b18c86984c59d6511a233d14b6c50e5c9e24d7`.

```bash
git clone https://github.com/ROCm/aiter.git third_party/aiter
git -C third_party/aiter checkout 32b18c86984c59d6511a233d14b6c50e5c9e24d7
git -C third_party/aiter submodule update --init --recursive
# or keep it anywhere and export PIKA_AITER_DIR=/path/to/aiter
```

The worker finds aiter in this order: `PIKA_AITER_DIR`, an installed `aiter` package, `<repo>/third_party/aiter`; if none has it,
the worker stops at start-up with an error. aiter JIT-builds its C++/HIP modules into `<aiter>/aiter/jit/` on first use, so the
directory must be writable. Pre-build them without a GPU (for example in an image build) with
`python tools/prebuild_aiter.py --lossy` (`--list` shows the modules): about 70 s for all six modules at `MAX_JOBS=32`.

**Other first-start builds** (automatic and cached): the SDMA copy-engine shim `src/pika_h3/csrc/sdma_shim.cpp` (`hipcc`,
about 2 s, world 8 only) in `$PIKA_CACHE_DIR/sdma`; the HIP kernels in `$PIKA_CACHE_DIR`; Triton kernels in `$TRITON_CACHE_DIR`.
If the SDMA shim cannot be built or initialised, every rank falls back to the RCCL all-to-all together (same data, slower).

**Tuned GEMM table.** `src/pika_h3/data/gemm_sols.json` (package data) maps GEMM shapes to hipBLASLt solution indices, found by a
sustained sweep on MI350X with the hipBLASLt of ROCm 7.2. The lossless reference output was produced with it; without it the
output is still valid (different, legitimate kernel choices) and a few % slower. If the file is missing the worker warns and uses
the hipBLASLt heuristic; a solution index the installed hipBLASLt rejects falls back to the heuristic per shape. On another
ROCm / hipBLASLt version, re-validate the table or disable it with `PIKA_GEMM_SOLS=off`.

## Start the worker

```bash
export PIKA_BASE_PATH=/models/MiniMax-H3            # snapshot directory (transformer/, text_encoder/, vae/, audio_vae/, ...)
export PIKA_ADAPTER_PATH=/models/Pika-H3-Flex/fl2va
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=src \
  python -m pika_h3.serve --base-path "$PIKA_BASE_PATH" --adapter-path "$PIKA_ADAPTER_PATH"
```

The worker starts one helper process per extra GPU, loads every model, builds the kernels, warms up shape-generically and prints
`EAT_READY {...}` (about 75 s with warm caches). It then reads one JSON request per line on stdin and writes `EAT_RESULT {...}` on
stdout once the MP4 is complete and closed. Requests run one at a time. Every other stdout / stderr line is log output. EOF on
stdin shuts the worker down.

**Warm up before taking traffic.** Some kernels load or compile lazily: the aiter modules on the first request, Triton
specialisations per duration, the lossy kernels on the first request of each preset. In a fresh world-8 worker the first lossless
5 s request took 7.7 s (later 3.7 s) and the first `near_lossless` request 4.4 s (later 2.6 s). Either send one untimed request
per duration and per lossy preset you serve, or let the worker do it before `EAT_READY` (off by default):

```bash
PIKA_WARMUP=5,10,15 PIKA_WARM_PRESETS=near_lossless,balanced python -m pika_h3.serve ...
```

This runs one synthetic request (fixed prompt, seed 0) per duration, lossless and per listed preset, and deletes the outputs. Keep
`TRITON_CACHE_DIR` on persistent storage so that restarts reuse compiled kernels; a prompt of very different length can still
trigger a one-off Triton compile for a new token-count size class.

**GPU count.** The worker uses every visible GPU when all have enough free memory (about 135 GB on the first, 66 GB on the
others); otherwise it picks the largest group of 8 / 7 / 4 / 2 / 1 GPUs that fits and prints a warning. `PIKA_WORLD=4` (or 2, 1)
caps the count; `PIKA_DEVICES=3,0,1,2` forces an explicit list of visible indices (the first one is rank 0).

| world | lossless latency | notes |
|---|---|---|
| 8 | 3.28 / 8.73 / 17.04 s (5 / 10 / 15 s) | the tuned and verified configuration; SDMA all-to-all |
| 7 | 3.9 s at 5 s (smoke test) | RCCL all-to-all; 10 / 15 s not measured |
| 4 | not re-timed with the final defaults | RCCL all-to-all with row-split overlap; has passed the per-step verification |
| 2 | 15.3 s at 5 s | RCCL; exercised by `tests/test_world2.py` |
| 1 | 23.6 s at 5 s (smoke test) | no communication; needs ~146 GB on one GPU; 10 / 15 s not measured |

`HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` select the GPUs; the code does not depend on particular indices. On machines where
HIP's device order differs from `amd-smi`, set `ROCR_VISIBLE_DEVICES` to the GPU UUIDs in the order you want.

## Request format

One JSON object per line:

```json
{"id": "job-17", "task": "t2va", "prompt": "A lighthouse at dusk ...", "seed": 1234, "resolution": "768p", "aspect": "16:9",
 "duration": 10, "output": "/abs/path/out.mp4", "lossy": "near_lossless"}
```

| field | required | values |
|---|---|---|
| `id` | no | any string, echoed in the result |
| `task` | no | `"t2va"` (the only task this worker serves) |
| `prompt` | yes | non-empty string (at most 2048 text tokens) |
| `seed` | yes | integer |
| `resolution` | yes | `"768p"` (tested) or `"480p"` (accepted, not tuned or tested) |
| `aspect` | yes | an aspect of `core/buckets.yaml`; only `"16:9"` is tuned and tested |
| `duration` | yes | integer seconds 2..15; 5, 10 and 15 are tested. Frames follow the 17n+5 grid (124 / 243 / 362 at 5 / 10 / 15 s) |
| `output` | yes | MP4 path (1344x768, 24 fps, H.264 + 32 kHz stereo AAC); parent directories are created |
| `trace_dir` | no | also write `trace.safetensors` (per-step inputs and x0 predictions, decoded pixels / wave) for verification |
| `lossy` | no | absent / `null` / `{}` / `"lossless"` = the lossless default; a preset name; or a spec object |

Result: `EAT_RESULT {"id": ..., "ok": true, "timings": {"conditioning_seconds", "sampling_seconds", "decode_encode_seconds",
"kfd_evicted_ms", ...}}`. Every request is validated on rank 0 before any GPU work; a malformed field (unknown preset or key,
non-integer seed, unsupported resolution, ...) gives `{"ok": false, "error": "..."}` and the worker keeps serving.

**Determinism.** Lossless requests and all presets are deterministic: the same request on the same world size, software and MIOpen
database gives the same MP4 bytes (tested). With a different MIOpen find database (for example a fresh host), the audio decoder
can pick other convolution kernels: in a clean-checkout test every sampling step and every video pixel was bit-identical and the
waveform differed at 122.7 dB SNR. Different world sizes are different legitimate kernel orders; because the six-step sampler is
chaotic, world-8 and world-2 outputs differ visibly in detail, just as two kernel variants of the unmodified pipeline do.

## Failure handling

Each failed request falls into one of three classes. Rank 0 decides the class; the healthy path has no extra synchronisation
(`src/pika_h3/faults.py`).

| class | examples | result | worker |
|---|---|---|---|
| rejected | invalid field; prompt over 2048 text tokens or another conditioning error on rank 0 (broadcast to every rank as `RequestRejected`) | `{"ok": false, "error": ...}` | keeps serving |
| local | rank 0 fails after the last collective of the request (MP4 mux, trace write) | `{"ok": false, "error": ...}` | keeps serving |
| fatal | any other exception or out-of-memory on any rank during sampling or decode; a helper rank that exits; a failed collective; a request over `PIKA_REQUEST_TIMEOUT` | `{"ok": false, "fatal": true, "error": "fatal: ..."}`, then `EAT_FATAL {"error", "rank", "exit_code": 3}` | kills its helper ranks and exits with code **3** |

A helper rank that raises exits at once; rank 0 runs a watchdog thread that checks its helpers every 0.5 s and exits even while
its main thread is blocked in a collective. **Run the worker under a process manager** (systemd `Restart=on-failure`,
Kubernetes, or a loop in your client) and treat exit code 3 (or any exit) as "restart and re-submit". Measured at world 2: the
fatal result arrives as soon as the fault happens and the process is gone 0.2-0.5 s later, with no helper left. A restart costs
the model load (about 75 s with warm caches, plus any warm-up).

`PIKA_REQUEST_TIMEOUT=<s>` (default off) makes a request that runs longer than that fatal, as a backstop against silent hangs; set
it to a few times the slowest expected request (for example 120 at world 8).

Fault injection, for testing only: `PIKA_FAULT=<rank>:<point>[:<n>]` raises on rank `<rank>` in the `<n>`-th request, where
`<point>` is `step0`..`step5`, `decode`, `oom` (a `torch.OutOfMemoryError` at step 2) or `post` (rank 0 after the MP4: local).
Unset, nothing is installed.

## Lossy presets

The lossless path is the default and is unchanged by everything in this section. A request opts in per request; presets are fixed
specs that do not depend on the prompt, seed or duration. Equivalent forms:

```json
"lossy": "balanced"
"lossy": {"preset": "balanced"}
"lossy": {"preset": "balanced", "vae_fp8": false}       // a preset with explicit key overrides (null removes a key)
"lossy": {"attn": ["bf16","bf16","mxfp6","mxfp6","mxfp6","mxfp6"], "attn_prefix_bf16": true}   // a raw spec
```

`PIKA_LOSSY` (a preset name or a JSON spec) sets a process-wide default for requests without a `lossy` field; leave it unset
unless every request should be lossy. An unknown preset, key or malformed value is an error (`LossyConfigError`), never a silent
fallback. `python -m pika_h3.lossy_presets` prints the presets.

### The presets

Generator evaluations 0 and 1 decide the composition, so all presets keep step 0 exact and approximate mainly steps 2-5:

- **`near_lossless`** (internal name TQ2_a45_d), the best fidelity: MXFP6 attention (aiter `mha_v4`) at steps 2-5 with text and
  audio query rows exact in bf16; at step 2 the top 10 % key blocks per head and query tile in bf16 and the rest in MXFP6, merged
  by LSE; aiter MXFP6 GEMMs (`a6w6`) for qkv / o / up / down at steps 4-5 with text and audio rows exact; a lighter VAE decode
  tiling (24 instead of 28 tiles per clip) with FP8 tile GEMMs. Keeps sign text and thin lines.
- **`balanced`** (T1_a345_d): the same, but the tiered attention runs at step 1 instead of step 2, and the a6w6 GEMMs also run at
  step 3. Lower PSNR, fine detail kept; lettering and object identity are re-generated in places (legible, no artifacts), as
  between two lossless runs. **Recommended default lossy choice.**
- **`fast`** is an **alias of `balanced`** (same spec, byte-identical output). No faster spec stayed inside the audio check (see
  below); the name stays valid for existing callers.

**Latency** (world 8, one long-lived worker with the final defaults, unseen prompts at all three durations, configurations
interleaved; 45 timed requests, 5 per preset and duration; 0 outliers; p95 within 0.6 % of p50):

| preset | p50 5 / 10 / 15 s | p95 5 / 10 / 15 s | max 5 / 10 / 15 s | speed-up vs lossless at p50 |
|---|---|---|---|---|
| lossless (default) | 3.29 / 8.73 / 16.88 | 3.30 / 8.75 / 16.91 | 3.30 / 8.76 / 16.91 | 1x |
| `near_lossless` | 2.60 / 6.25 / 11.64 | 2.60 / 6.28 / 11.69 | 2.60 / 6.28 / 11.69 | 1.27 / 1.40 / 1.45x |
| `balanced` (= `fast`) | 2.39 / 5.58 / 10.22 | 2.40 / 5.59 / 10.26 | 2.40 / 5.59 / 10.26 | 1.38 / 1.56 / 1.65x |

**Quality on unseen prompts.** Each MP4 compared with the lossless MP4 of the same request (mean over 9 prompts; audio LSD includes
the AAC encoder's noise):

| preset | PSNR dB 5/10/15 | detail* 5/10/15 | top-region detail** 5/10/15 | top-region SSIM | audio LSD (MP4) |
|---|---|---|---|---|---|
| `near_lossless` | 24.5 / 27.2 / 27.4 | .993 / .994 / .996 | .965 / .981 / .983 | .69-.77 | 5.3-5.6 dB |
| `balanced` | 19.8 / 22.3 / 21.8 | .990 / .992 / .989 | .922 / .952 / .947 | .48-.54 | 6.2-6.4 dB |
| lossless floor*** | 19.2-19.6 | .97-1.0 | .87-.97 | .33-.56 | |

\* mean |Laplacian| of the variant divided by that of lossless (every 8th frame); 1.0 = as sharp.
\*\* the same ratio inside the 5 % highest-detail 32x32 blocks of the lossless frame (sign text, thin lines); SSIM is luma SSIM there.
\*\*\* two lossless, verification-passing runs of the same request that differ only in world size (8 vs 2): the pipeline's own
sensitivity. Both presets are above it.

**Quality on three fixed reference requests** (768p 16:9, 5 s seed 42, 10 s seed 1234, 15 s seed 7), world 8, against the lossless
output of the same request; the last column is the per-step audio check of the lossless verifier (reference transformer on the
candidate's own inputs):

| preset | PSNR 5/10/15 | detail | top-region detail, SSIM 5/10/15 | audio LSD | per-step audio check |
|---|---|---|---|---|---|
| `near_lossless` | 30.5 / 25.7 / 28.3 | .993-.999 | .97,.86 / .99,.60 / 1.00,.81 | 1.6 / 2.0 / 2.0 | passes every step |
| `balanced` | 23.4 / 21.6 / 22.0 | .989-1.001 | .97,.68 / .94,.44 / .99,.47 | 2.4 / 2.5 / 2.7 | passes every step |

Decode PSNR against the default decode was 46 / 40 / 42 dB. No preset showed artifacts in close-up crops.

**Why `fast` = `balanced`.** In a world-8 stability run (120 requests, unseen prompts, 0 outliers, max within 4 % of p50), the
fastest block-sparse spec F2c_a2345_d (block-sparse bf16 attention at steps 3-5 with exact text/audio rows, MXFP6 attention at
step 2, a6w6 GEMMs at steps 2-5) was only 1-3 % faster than `balanced` (p50 2.43 / 5.65 / 10.37 s vs 2.50 / 5.74 / 10.51 s) and
its audio fails the per-step check on the 10 s reference request at steps 3-5 (by up to 4.9 dB). Audio-safe sparse variants were
slower than `balanced` (2.55-2.59 / 6.01-6.08 / 11.13-11.26 s). F2c_a2345_d stays available as a raw spec
(`pika_h3.lossy_presets.F2C_A2345_D`; deterministic, about 3 % softer fine detail).

### Lossy keys (raw spec)

Per-step values are one value for all six generator evaluations, or a list of up to 6 (the last entry repeats).

| key | values | effect |
|---|---|---|
| `attn` | per step: `bf16` (exact) \| `mxfp6` \| `mxfp6fp8` \| `fp8mxfp6` \| `mxfp8` \| `fp8` \| `int8` \| `bf16fp8` \| `mxfp4` | attention operand formats (Q/K, V) of the aiter `mha_v4` kernel |
| `attn_prefix_bf16` | bool | text + audio query rows of attention stay exact bf16 |
| `tier` | per step: `""` or `"<fmt>:<frac>,..."` (fmt bf16/mxfp8/mxfp6/mxfp4, cumulative key-block fraction or `m<mass>`; flags `comp`, `est=`, `layers=a-b`, `min<k>`, `max<f>`) | tiered-precision block attention |
| `gemm` | per step: `""` or comma list of `qkv,o,up,down` | base GEMMs of the DiT in low precision (the LoRA stays bf16) |
| `gemm_fmt` | per step: `fp8` (row-wise x channel FP8) \| `a6w6` (aiter MXFP6) | format of the `gemm` GEMMs |
| `gemm_prefix_bf16` | bool | text + audio rows of those GEMMs stay exact bf16 |
| `sparse` | list of 1..6 keep fractions in (0, 1] | block-sparse attention per step (1 = dense) |
| `impl` | `"s2"` | lean sparse implementation; supports `attn` `bf16` / `bf16fp8` / `fp8` on sparse steps |
| `dense_q` | bool | text + audio query rows dense on sparse steps |
| `s2_prefix_bf16` | bool | with `impl` `s2` and `dense_q`: text + audio query rows recomputed exactly over all keys |
| `prefix` | integer or `"auto"` (= text + audio length) | key tokens at the sequence start that sparse attention always keeps |
| `est` | `rows16k4` \| `rows<r>k<s>` \| `rows<r>` \| `mean` \| `sub` | block-importance estimate for `sparse` / `tier` |
| `vae_tile` | `[tile_h, tile_w, min_overlap_h, min_overlap_w]` px, multiples of 16 | VAE decode tiling (default 256 x 256 tiles with the package's overlap) |
| `vae_fp8` | bool \| `"all"` \| comma list of `qkv,o,up,down` | FP8 GEMMs in the tile decoder ViT |

Research keys are accepted but not validated or supported for service: `attn_layers`, `comm_fp8`, `gemm_layers`, `gemm_map`,
`headstats`, `layers`, `layers01`, `local`, `pfx_stream`, `s2split`, `s2time`, `s2wave`, `tiertime`, `vae_clips` (`headstats`
writes files; `s2time` / `tiertime` add synchronising timers).

## Host settings

**Why.** On ROCm, the kernel driver (KFD) *evicts* all GPU queues of a process while the host kernel moves or re-protects host
pages that the GPU has mapped: memory compaction (`kcompactd`), automatic NUMA balancing, or huge-page collapsing. One rank stops
until the driver has revalidated its mappings, and the other seven wait for it in the next collective. Signature: every denoising
step 2-10x slower at normal host launch time, GPU power at 400-550 W, `EVICTED_TIME` growing in `amd-smi process`. A bounded
compaction load next to an idle worker (`tools/evict_provoke.py`, 64 GB `MADV_HUGEPAGE` cycles) reproduces it on demand: lossless
5 s up to 15.7 s, 10 s up to 33.0 s.

For a service host (needs root):

| setting | recommended | why |
|---|---|---|
| `kernel.numa_balancing` | `0` | AMD's recommendation for MI300-class GPUs: NUMA hinting faults evict GPU queues |
| `vm.compaction_proactiveness` | `0` | proactive background compaction migrates GPU-mapped host pages |
| `/sys/kernel/mm/transparent_hugepage/defrag` | `madvise` or `never` (not `always`) | `always` compacts directly on page faults |

Also keep plenty of free host RAM and avoid large page-cache churn (for example copying weights) while serving. The measurements
below were taken on a shared host where these settings could not be changed, so their own effect is not measured here. The worker
checks them at start-up and prints `[serve] WARNING host config: ...` for each risky one; it reports `timings["kfd_evicted_ms"]`
= {rank: ms} for every request (a sysfs read, no GPU work) and warns when a request lost 100 ms or more.

**In-process mitigation (default on, no root needed; bit-identical output):**
- `PIKA_HOST_THP=1`: back the worker's resident anonymous host memory with transparent huge pages (`MADV_HUGEPAGE` +
  `MADV_COLLAPSE`, Linux >= 6.1), which compaction does not move. The pass runs once, after loading (`PIKA_HOST_THP_INCR=0`).
- `PIKA_HOST_MEMPOLICY=bind`: `MPOL_BIND` over all NUMA nodes for the worker and its ranks, which takes them out of automatic NUMA
  balancing without changing where memory is allocated.
- A ready barrier before `EAT_READY`: every rank finishes its GPU work and its huge-page pass, the ranks meet, and rank 0 waits
  until no rank's evicted time has grown for 0.5 s (at most 10 s). Cost: about 0.5-1.6 s of start-up; the first request after
  start-up then loses 0 ms on every rank (it lost ~1.2 s before).

A/B of the mitigation on a host with `kernel.numa_balancing=1` and no forced compaction (world 8, four long-lived workers in the
order off / on / off / on, 12 lossless requests per duration and configuration, unseen prompts plus the reference requests;
84 / 84 MP4s bit-identical across the workers):

| config | duration | p50 s | p95 s | max s | mean s | worst-rank KFD eviction per request, mean / p95 / max ms | requests > 500 ms evicted |
|---|---|---|---|---|---|---|---|
| mitigation off | 5 s | 3.465 | 3.586 | 3.643 | 3.471 | 567 / 878 / 880 | 6 / 12 |
| mitigation on | 5 s | 3.297 | 3.321 | 3.330 | 3.296 | 117 / 155 / 163 | 0 / 12 |
| mitigation off | 10 s | 9.018 | 9.313 | 9.457 | 9.081 | 896 / 1375 / 1407 | 11 / 12 |
| mitigation on | 10 s | 8.737 | 8.781 | 8.789 | 8.743 | 191 / 519 / 538 | 2 / 12 |
| mitigation off | 15 s | 17.412 | 17.645 | 17.657 | 17.438 | 1310 / 2154 / 2230 | 12 / 12 |
| mitigation on | 15 s | 16.860 | 17.069 | 17.099 | 16.889 | 104 / 202 / 263 | 0 / 12 |

Without compaction the evictions come from NUMA balancing, which `bind` opts out of; the saving is in the sampling stage
(fewer lockstep stalls), not in decode. Under forced host compaction the mitigation is only partial (about 6x less eviction time,
worst lossless 10 s request 33.0 -> 11.7 s), so the host settings above remain the real fix.

Load-time-only huge-page pass vs a pass after every request (world 8, arms A A C A C, 9 lossless requests on unseen prompts per arm
plus warm-ups and the reference requests; reference MP4s identical in all arms):

| | timed mean 5 / 10 / 15 s | timed p95 5 / 10 / 15 s | worst-rank eviction, timed mean (max) | untimed: 2nd warm-up / 10 s reference request |
|---|---|---|---|---|
| A, pass after every request | 3.313 / 8.743 / 16.845 s | 3.351 / 8.763 / 16.861 s | 80 / 113 / 166 ms (277) | 460 / 630 ms |
| C, load-time pass only (default) | 3.308 / 8.741 / 16.851 s | 3.353 / 8.759 / 16.867 s | 30 / 53 / 0 ms (177) | 0 / 68 ms |

Collapsing host pages that the GPU has mapped makes the driver evict the rank's queues for >= ~100 ms; a pass right after a request
put that eviction into the first second of the next one. `PIKA_HOST_THP_INCR=1` restores the per-request pass.

## Latency stability

Stability was measured with one long-lived world-8 worker serving many requests in a row on prompts it had not seen (42-423
characters), at all three durations, with configurations interleaved in random order. The harness is `tools/stab.py`
(`--help`; prompts in `tools/stab_prompts.json`); per configuration and duration it reports p50 / p95 / max and outliers (more
than 1.15x p50), and logs GPU clocks / power / foreign processes (`amd-smi`), host CPU, `kcompactd`, KFD evictions and
request-time kernel compiles with each request.

| run (world 8, one worker) | configurations | requests | outliers | spread |
|---|---|---|---|---|
| final defaults | lossless, `near_lossless`, `balanced` | 45 | 0 | p95 within 0.6 % of p50 |
| fast-tier selection | `balanced`, F2c_a2345_d, two audio-safe sparse variants | 120 | 0 | max within 4 % of p50 |
| before the host mitigation | lossless + the three presets | 60 | 0 | p95 within 3 % of p50 |

Example (run with the worker's Python environment; `PIKA_BASE_PATH` and `PIKA_ADAPTER_PATH` exported as in
[Start the worker](#start-the-worker); one worker loads once, so a short check such as `--cfg lossless --cfg balanced=preset:balanced
--rounds 4` (8 timed requests, all three durations) took 5 minutes on 8 GPUs, about half of it model loading; `--analyze <out>` re-runs only the report):

```bash
python tools/stab.py --gpus 0,1,2,3,4,5,6,7 --out /tmp/stab --cfg lossless --cfg near_lossless=preset:near_lossless \
    --cfg balanced=preset:balanced --rounds 15 --seed 15
python tools/stab.py ... --provoke 64:60:300     # optional: host compaction load (tools/evict_provoke.py) during the timed phase
```

**Trace requests** (`trace_dir`, a verification feature) can still lose up to ~0.8 s to evictions on rank 0: they gather the full
decoded video to rank 0 and write 1-2 GB of fp32 tensors from fresh host allocations. Normal requests do not take this path.

## Environment variables

Set automatically by the package (`pika_h3/__init__.py`, inherited by the helper ranks; an explicit value wins):

| variable | default | why |
|---|---|---|
| `NCCL_MIN_P2P_NCHANNELS`, `NCCL_MAX_P2P_NCHANNELS` | 64 | RCCL's default gives the Ulysses all-to-all about one channel per peer at world 7/8 (12x slower all-to-all) |
| `ROCPROFILER_REGISTER_ENABLED` | 0 | with rocprofiler-sdk registered, the HSA intercept queue segfaults after ~10^5 dispatches on ROCm 7.2 (profiling with `PIKA_PROF_STEP` keeps it enabled) |

Deployment settings:

| variable | default | meaning |
|---|---|---|
| `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` | all | which GPUs the worker may use |
| `PIKA_WORLD` | visible GPUs | cap on the GPU count |
| `PIKA_DEVICES` | auto | explicit device list (visible indices, rank 0 first) |
| `PIKA_AITER_DIR` | installed aiter, else `<repo>/third_party/aiter` | aiter checkout |
| `PIKA_CACHE_DIR` | `$XDG_CACHE_HOME/pika_h3` or `~/.cache/pika_h3` | kernel build caches (`PIKA_SDMA_BUILD`, `PIKA_HIP_CACHE`, `PIKA_FA18_DIR` override single caches) |
| `PIKA_LOAD_LOCK_DIR` | `PIKA_CACHE_DIR` | directory of the two host-RAM load-slot locks; share it between workers on one host |
| `PIKA_GEMM_SOLS` | `src/pika_h3/data/gemm_sols.json` | tuned hipBLASLt solutions (a path; relative paths are relative to the working directory; `off` = heuristic) |
| `PIKA_NCCL_TIMEOUT` | 600 | process-group timeout (s); a rank that raises or exits is detected within ~0.5 s anyway, this only bounds silent hangs |
| `PIKA_REQUEST_TIMEOUT` | 0 (off) | a request longer than this many seconds is fatal (exit code 3) |
| `PIKA_LOSSY` | unset | process-wide lossy default (keep unset) |
| `PIKA_WARMUP` | unset | durations (e.g. `5,10,15`) of synthetic warm-up requests before `EAT_READY` |
| `PIKA_WARM_PRESETS` | unset | lossy presets to warm up as well (at `PIKA_WARMUP`'s durations, default 5 s) |
| `PIKA_HOST_THP` | `1` | huge-page backing of host memory against GPU-queue evictions; `0` = off |
| `PIKA_HOST_THP_INCR` | `0` | `1` = also run the huge-page pass after every request (not recommended, see [host settings](#host-settings)) |
| `PIKA_HOST_MEMPOLICY` | `bind` | take the worker out of automatic NUMA balancing; `off` = leave the memory policy alone |
| `TRITON_CACHE_DIR` | `~/.triton` | Triton kernel cache; keep it persistent |
| `MIOPEN_USER_DB_PATH` | `~/.config/miopen` | MIOpen find database: decides the audio decoder's convolution kernels. Keep it persistent and identical across hosts for byte-identical audio |
| `TMPDIR` | `/tmp` | needs a few GB free (comgr JIT directories) |

Every other `PIKA_*` variable in the source is a development switch whose default is the tested path; do not set them in
production (this includes `PIKA_SDMA`, `PIKA_SPLIT`, `PIKA_ATTN_KERNEL`, `PIKA_V4_IMPL`, `PIKA_TIMING`, `PIKA_STEPLOG`,
`PIKA_PROF*`, `PIKA_DUMP*`, `PIKA_FILLNAN`). Request fields starting with `dev_`, and `load_latents`, `save_latents`,
`dump_pixels` and `tf_*`, are development hooks.

## Known limitations

- **Shapes.** Tuning (GEMM table, attention tiling, SDMA schedule, decode split) and all measurements cover 768p 16:9 at 5 / 10 /
  15 s (31-91k tokens). Other durations of 2-15 s, 480p and other aspects run the same code but are not tested or verified.
- **World size.** Must divide the 56 heads (1, 2, 4, 7, 8). World 8 is tuned and verified; the lossy presets were measured at world 8
  and run (tested, not timed) at world 2.
- **Hardware.** gfx950 only (`--offload-arch=gfx950`); the GEMM table is specific to ROCm 7.2's hipBLASLt. `PIKA_NUM_CUS`
  (default 256) is the CU count used to plan attention waves.
- **Lossy presets** are verified only at 768p 16:9 5 / 10 / 15 s; their audio passed the per-step check on the three reference
  requests, not on arbitrary prompts. Steps 0-1 stay (nearly) exact because every approximation there changed the content beyond
  the lossless floor. Lettering on signs is re-generated by the presets, as it is between world sizes in the lossless pipeline:
  judge text by legibility and the absence of artifacts, not by exact letters.
- **Failure handling** is fail-fast: a failure inside the multi-GPU part of a request ends the process (exit code 3); the ranks are
  not resynchronised in place. A pure hang without any exception ends only at `PIKA_NCCL_TIMEOUT` unless `PIKA_REQUEST_TIMEOUT` is set.
- **KFD evictions.** On a host with `kernel.numa_balancing=1` / `vm.compaction_proactiveness>0`, timed requests can still lose
  ~0.1-0.2 s, trace requests up to ~0.8 s on rank 0. Fix on the host: [host settings](#host-settings).
- **One request at a time** per worker; for concurrency, run several workers on disjoint GPU sets.
- **First requests** per duration and per preset in a fresh process are slower (lazy kernel loading; see the warm-up above).

## Tests

```bash
# CPU only, < 1 s: presets and validation, paths, fault hook, host defaults
PYTHONPATH=src python -m unittest tests.test_presets tests.test_paths tests.test_faults tests.test_hosthealth -v
# 2 GPUs, ~8 min (incl. the fault tests)
export PIKA_BASE_PATH=... PIKA_ADAPTER_PATH=...
HIP_VISIBLE_DEVICES=0,1 PYTHONPATH=src:. python -m unittest tests.test_world2 -v
```

`tests/test_world2.py` starts one world-2 worker and sends (5 s each): lossless with a trace; `"lossy": {}` with a trace; three
invalid lossy requests; the three presets, twice; lossless again with a trace. It checks that flag-off is bit-identical to lossless
(all trace tensors and the MP4 sha256); a lossy request leaves no state behind; each preset is deterministic and `fast` equals
`balanced`; invalid requests fail cleanly while the worker keeps serving; every MP4 is valid (1344x768, 124 frames, 24 fps,
32 kHz stereo, finite, non-silent audio). `TestFaultsWorld2` starts three short-lived workers with `PIKA_FAULT` and checks the
[failure handling](#failure-handling) (about 2 min; `PIKA_TEST_FAULTS=0` skips it). `PIKA_TEST_OUT=<dir>` keeps the outputs,
`python -m tests.compare_runs <dir_a> <dir_b>` compares two checkouts, `PIKA_TEST_WORLD=8` runs the suite at world 8.

Results on 2026-11-07 from a clean checkout of this branch (fresh venv, fresh aiter clone): CPU tests 24 / 24 pass; world-2 tests
9 / 9 pass; at world 8 the lossless outputs of the three reference requests were bit-identical to the reference MP4s, and every
step passed the per-step check against the reference transformer.

## Code map

| path | role |
|---|---|
| `src/pika_h3/serve.py` | the worker (stdin/stdout protocol, request validation, device choice, helper-rank start, ready barrier) |
| `src/pika_h3/engine.py`, `rank_main.py`, `serve_loop.py` | per-rank engine: model load, conditioning, sampling, decode, MP4 |
| `src/pika_h3/sp_model.py` | sequence-parallel (Ulysses) DiT with fused kernels, the aiter `mha_v4` attention, SDMA / RCCL all-to-all |
| `src/pika_h3/kernels.py`, `hipk.py`, `hip/`, `sdma.py`, `csrc/`, `gemm_sel.py` | Triton / HIP kernels, SDMA shim, per-shape GEMM selection |
| `src/pika_h3/data/` | tuned hipBLASLt solution table (`gemm_sols.json`, read at runtime) and an optional TunableOp table |
| `src/pika_h3/pdecode.py`, `fast_vit.py`, `audio_conv.py`, `mp4.py` | parallel VAE / audio decode, MP4 encode |
| `src/pika_h3/faults.py`, `hosthealth.py`, `paths.py` | failure semantics, host checks and mitigation, aiter / cache locations |
| `src/pika_h3/lossy_presets.py` | presets and validation of the `lossy` field |
| `src/pika_h3/lossy.py`, `lowprec.py`, `attn_tier.py`, `sparse2.py`, `lp_kernels.py`, `lossy_comm.py` | lossy implementations (inactive unless `lossy` is set) |
| `src/pika_h3/attn_fa18.py`, `fa18/` | optional hand-written gfx950 FMHA (off by default, `PIKA_V4_IMPL=fa18`) |
| `tests/` | CPU and world-2 tests |
| `tools/apply_diffusers_patches.py`, `tools/prebuild_aiter.py` | install helpers |
| `tools/stab.py`, `tools/stab_prompts.json`, `tools/evict_provoke.py` | latency stability harness and host-compaction provoker |
