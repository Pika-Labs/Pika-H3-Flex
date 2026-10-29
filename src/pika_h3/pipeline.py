# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""One staged native Diffusers inference path for both released adapters."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

BASE_REPO = "MiniMaxAI/MiniMax-H3"
BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
KNOTS = [1000, 750, 500, 250, 100, 40, 0]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_assets(args):
    from huggingface_hub import snapshot_download
    task = "ref2va" if args.task == "ref2va" else "fl2va"
    if args.adapter_path:
        adapter_root = args.adapter_path
    else:
        root = snapshot_download(args.model_repo, revision=args.revision,
                                 allow_patterns=[f"{task}/*", "LICENSE", "NOTICE", "licenses/*"])
        adapter_root = Path(root) / task
    manifest = json.loads((adapter_root / "manifest.json").read_text())
    if manifest.get("format_version") != 1 or manifest.get("rank") != 128 or manifest.get("alpha") != 128:
        raise ValueError("unsupported adapter format/rank/alpha")
    expected_policy = "fp32_conditioning_v1" if task == "ref2va" else "bf16_conditioning_v1"
    if manifest.get("numerics_policy") != expected_policy:
        raise ValueError("adapter numerical policy does not match this release")
    adapter = adapter_root / "adapter.safetensors"
    # Validate the exported release contract before loading any models.
    expected_trunk = "transformer_ref" if task == "ref2va" else "transformer"
    if manifest["trunk"] != expected_trunk:
        raise ValueError("adapter trunk does not match task")
    if manifest["sampling"]["knots"] != KNOTS or manifest["sampling"]["video_shift"] != 12 or manifest["sampling"]["audio_shift"] != 3 or manifest["sampling"]["method"] != "renoise":
        raise ValueError("unsupported adapter sampling contract")
    if file_sha256(adapter) != manifest["sha256"]:
        raise ValueError("adapter checksum mismatch")
    if manifest["base"] != {"repo_id": BASE_REPO, "revision": BASE_REVISION}:
        raise ValueError("adapter base identity mismatch")
    base = args.base_path or Path(snapshot_download(BASE_REPO, revision=BASE_REVISION,
        allow_patterns=[f"{expected_trunk}/*", "text_encoder/*", "tokenizer/*", "processor/*", "vae/*", "audio_vae/*", "*.json", "LICENSE", "NOTICE", "licenses/*"]))
    return Path(base), adapter, manifest


def generate(args):
    from .runtime import validate_runtime
    validate_runtime()
    import torch
    from PIL import Image
    from .core.conditioning import H3ConditionEncoder, Reference, Request
    from .core.buckets import load_bucket_config
    from .core.loader import build_transformer, apply_lora, load_lora_weights
    from .core.numerics import prepare_conditioning, configure_numerics_backend
    from .core.schedules import H3SigmaSchedule
    from .core.evaluation import sample_evaluation_ladder
    from .core.wrapper import H3DiffusionWrapper
    from .core.vae import H3Decoders, write_mp4
    from .core import geometry as G

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model generation")
    bucket_file = Path(__file__).parent / "core" / "buckets.yaml"
    import yaml
    table = yaml.safe_load(bucket_file.read_text())
    tier = "p" + args.resolution[:-1]
    width, height = table["tiers"][tier][table["aspects"].index(args.aspect)]
    frames = G.align_num_frames(args.duration * G.FPS)
    from .core.media import preflight_reference_specs
    preflight_reference_specs(args.ref)
    references = [Reference.from_spec(spec) for spec in args.ref]
    counts = {kind: sum(r.kind == kind for r in references) for kind in ("image", "video", "audio")}
    if len(references) > 12 or counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
        raise ValueError("reference limits: 9 images, 3 videos, 3 audio; 12 total")
    if args.task == "ref2va" and counts["image"] + counts["video"] == 0:
        raise ValueError("ref2va requires an image or video reference")
    request = Request(task=args.task, prompt=args.prompt,
        first_image=Image.open(args.first_image).convert("RGB") if args.first_image else None,
        last_image=Image.open(args.last_image).convert("RGB") if args.last_image else None,
        references=references, height=height, width=width, num_frames=frames,
        reference_rule="tiered", keyframe_fit="crop")
    start = time.monotonic()
    base, adapter, manifest = resolve_assets(args)
    timings = {"assets_seconds": time.monotonic() - start}
    policy = manifest["numerics_policy"]
    configure_numerics_backend(policy, runtime="inference")
    encoder = H3ConditionEncoder(str(base), device="cuda", text_encoder_checkpoint=str(args.text_encoder_path) if args.text_encoder_path else None)
    start = time.monotonic()
    cond = encoder.encode(request)
    encoder.free()
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    timings["conditioning_seconds"] = time.monotonic() - start
    start = time.monotonic()
    model = build_transformer(str(base), dtype=torch.bfloat16, device="cpu", subfolder=manifest["trunk"])
    apply_lora(model, rank=manifest["rank"], alpha=manifest["alpha"], target_modules=manifest.get("target_modules"), master_dtype=torch.bfloat16)
    load_lora_weights(model, str(adapter))
    model.requires_grad_(False).eval()
    if args.cpu_offload:
        model.enable_group_offload(onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"), offload_type="block_level",
            num_blocks_per_group=1, use_stream=True, record_stream=False)
        wrapper = H3DiffusionWrapper(model).eval()
    else:
        wrapper = H3DiffusionWrapper(model).eval().to("cuda")
    timings["transformer_load_seconds"] = time.monotonic() - start
    geom = cond["geometry"]
    schedule = H3SigmaSchedule({"denoising_step_list": KNOTS, "video_score_sigma_shift":12, "audio_score_sigma_shift":3}, device="cuda")
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        video, audio = sample_evaluation_ladder(wrapper,
            conditional_dict=prepare_conditioning(cond, "cuda", policy=policy),
            video_shape=[1,geom.num_latent_frames,G.VIDEO_LATENT_CHANNELS,geom.latent_height,geom.latent_width],
            audio_shape=[1,geom.num_audio_latents,G.AUDIO_EXTERNAL_DIM],
            sigmas_v=schedule.video_ladder, sigmas_a=schedule.audio_ladder,
            seed=args.seed, device="cuda")
    video, audio = video[0].cpu(), audio[0].cpu()
    torch.cuda.synchronize()
    timings["sampling_seconds"] = time.monotonic() - start
    peak = torch.cuda.max_memory_allocated()
    del wrapper, model, cond
    gc.collect()
    torch.cuda.empty_cache()
    start = time.monotonic()
    decoders = H3Decoders(str(base), device="cuda")
    with torch.inference_mode():
        pixels = decoders.decode_video(video)
        wave = decoders.decode_audio(audio)
    if pixels.shape[:3] != (frames, height, width):
        raise RuntimeError(f"decoded geometry mismatch: {pixels.shape}")
    import numpy as np
    if not np.isfinite(wave).all():
        raise RuntimeError("decoded audio is not finite")
    wave = wave[:, :round(frames / G.FPS * G.AUDIO_SAMPLE_RATE)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(str(args.output), pixels, audio=wave)
    import av
    with av.open(str(args.output)) as encoded:
        stream = encoded.streams.video[0]
        if (stream.width,stream.height) != (width,height) or not encoded.streams.audio:
            raise RuntimeError("encoded video/audio validation failed")
        audio_stream = encoded.streams.audio[0]
        if stream.average_rate != G.FPS or audio_stream.codec_context.sample_rate != G.AUDIO_SAMPLE_RATE or audio_stream.codec_context.channels != 2:
            raise RuntimeError("encoded frame rate or stereo audio format mismatch")
        if sum(1 for _ in encoded.decode(video=0)) != frames:
            raise RuntimeError("encoded frame count mismatch")
    timings["decode_encode_seconds"] = time.monotonic() - start
    metadata = {"task": args.task, "prompt": args.prompt, "seed": args.seed,
        "resolution": args.resolution, "aspect": args.aspect, "requested_seconds": args.duration,
        "geometry": asdict(geom), "adapter_sha256": manifest["sha256"],
        "adapter_revision": args.revision, "base": manifest["base"],
        "local_base_override": args.base_path is not None,
        "transformer_cpu_offload": args.cpu_offload,
        "local_conditioner_override": args.text_encoder_path is not None,
        "bucket_sha256": file_sha256(bucket_file), "noise_version": "pika-flex-noise-v1",
        "numerics_policy":policy, "sampling":manifest["sampling"],
        "timings":timings, "sampling_peak_allocated_bytes":peak,
        "torch":torch.__version__, "gpu":torch.cuda.get_device_name(),
        "reference_order": [r.kind for r in references]}
    import importlib.metadata
    metadata["dependencies"] = {name: importlib.metadata.version(name) for name in ("diffusers", "transformers", "peft", "av", "safetensors")}
    metadata["input_sha256"] = {
        "first_image": file_sha256(args.first_image) if args.first_image else None,
        "last_image": file_sha256(args.last_image) if args.last_image else None,
        "references": [file_sha256(spec.partition(":")[2]) for spec in args.ref],
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Saved {args.output}")
