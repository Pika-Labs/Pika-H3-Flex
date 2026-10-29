# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Resolve raw inference canvases from the shared H3 multires bucket table."""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import yaml

from .buckets import BucketConfig
from .conditioning import Request, center_crop_box, resolve_geometry

DEFAULT_MULTIRES_CONFIG = Path(__file__).resolve().parent / "buckets.yaml"


def resolution_tier(value):
    """Normalize p768/768p/768; return None for unrelated labels such as duration tier d5."""
    match = re.fullmatch(r"(?:p(\d+)|(\d+)p?)", str(value).strip().lower())
    return f"p{int(match.group(1) or match.group(2))}" if match else None


@lru_cache(maxsize=16)
def _bucket_snapshot(path: str):
    # Parse and fingerprint the same bytes; every job in this process uses this snapshot.
    data = Path(path).read_bytes()
    return BucketConfig.from_dict(yaml.safe_load(data), path=path), hashlib.sha256(data).hexdigest()


def _aspect(value):
    try:
        parts = str(value).split(":")
        ratio = float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(value)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(f"aspect ratio must be positive W:H or W/H, got {value!r}") from exc
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"aspect ratio must be finite and positive, got {value!r}")
    return ratio


def geometry_policy(job: dict, args) -> str:
    """An explicit CLI policy wins; all other omitted policies select buckets."""
    cli = getattr(args, "geometry_policy", None)
    if getattr(args, "geometry_policy_explicit", False):
        return cli
    return job.get("geometry_policy") or cli or "buckets"


def cached_canvas_provenance(height: int, width: int, row: dict, config=None) -> dict:
    """Validate an immutable cached canvas without changing any conditioning tensors."""
    path = str(Path(config or DEFAULT_MULTIRES_CONFIG).expanduser().resolve())
    buckets, digest = _bucket_snapshot(path)
    declared = row.get("resolution") or resolution_tier(row.get("tier"))
    tier = resolution_tier(declared) if declared is not None else None
    if declared is not None and tier not in buckets.tiers:
        raise ValueError(f"precomputed unknown resolution {declared!r}; re-encode conditioning for a configured tier")
    matches = [(name, index) for name, canvases in buckets.tiers.items()
               if tier is None or name == tier
               for index, canvas in enumerate(canvases) if canvas == (width, height)]
    if len(matches) != 1:
        raise ValueError(f"precomputed canvas {width}x{height} is not an unambiguous bucket for {declared or 'any tier'}; "
                         "re-encode conditioning using the multires config (cached tensors cannot be resized)")
    tier, index = matches[0]
    return {"policy": "buckets", "config": path, "config_sha256": digest, "tier": tier,
            "bucket_index": index, "bucket_aspect": buckets.family_name(index),
            "canvas": [height, width], "keyframe_fit": "precomputed"}


def verify_rendered_output(media: Path, job: dict, geom, provenance: dict) -> dict:
    """Reject stale existing outputs; inspect the media as well as its generation sidecar."""
    media = Path(media)
    reason = f"existing output {media}; use a fresh output directory or explicitly re-render it"
    try:
        side = json.loads(media.with_suffix(".json").read_text())
        expected = [geom.height, geom.width]
        if side.get("id") != job.get("id", "sample") or side.get("canvas") != expected or side.get("num_frames") != geom.num_frames:
            raise ValueError("sidecar ID/canvas/frame count does not match the resolved request")
        recorded = side.get("geometry_provenance") or {}
        if provenance["policy"] == "buckets":
            for key in ("policy", "config_sha256", "tier", "bucket_index", "canvas", "keyframe_fit"):
                if recorded.get(key) != provenance.get(key):
                    raise ValueError(f"sidecar bucket provenance differs at {key}")
        elif recorded and recorded.get("policy") != "pipeline":
            raise ValueError("sidecar geometry policy differs")
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
             "stream=width,height,nb_read_frames,avg_frame_rate", "-of", "json", str(media)],
            check=True, capture_output=True, text=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if len(streams) != 1:
            raise ValueError("expected one readable video stream")
        stream = streams[0]
        if [stream["height"], stream["width"]] != expected or int(stream.get("nb_read_frames", -1)) != geom.num_frames:
            raise ValueError("actual video dimensions/frame count differ from the resolved request")
        if Fraction(stream.get("avg_frame_rate", "0")) != 24:
            raise ValueError("actual video FPS differs from H3's 24 FPS")
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Cannot accept {reason}: {exc}") from exc
    return side


def configure_canvas(req: Request, job: dict, args) -> dict:
    """Set explicit raw request geometry and return JSON-safe selection/crop provenance.

    CLI resolution/aspect overrides per-job metadata. H/W are aspect hints in bucket mode;
    otherwise use the conditioning anchor's aspect, then 16:9. Programmatic inference wrappers
    also default to buckets; pipeline geometry requires an explicit diagnostic opt-in.
    """
    policy = geometry_policy(job, args)
    resolution = getattr(args, "resolution", None) or job.get("resolution")
    config = getattr(args, "multires_config", None) or job.get("multires_config")
    aspect = getattr(args, "aspect_ratio", None)
    if aspect is None:
        aspect = job.get("aspect_ratio")
    if (req.height is None) != (req.width is None):
        raise ValueError("height and width must be given together")
    if req.height is not None and (req.height <= 0 or req.width <= 0):
        raise ValueError("height and width must be positive")
    if policy == "pipeline":
        if resolution is not None or config is not None or aspect is not None:
            raise ValueError("resolution, multires_config and aspect_ratio require geometry_policy=buckets")
        geom = resolve_geometry(req)
        return {"policy": policy, "keyframe_fit": req.keyframe_fit, "canvas": [geom.height, geom.width]}
    if policy != "buckets":
        raise ValueError(f"geometry_policy must be buckets|pipeline, got {policy!r}")
    if any(getattr(args, key, None) is not None or key in job for key in ("short_edge", "max_pixels")):
        raise ValueError("short_edge/max_pixels require geometry_policy=pipeline; bucket mode uses multires_config")
    path = str(Path(config or DEFAULT_MULTIRES_CONFIG).expanduser().resolve())
    buckets, digest = _bucket_snapshot(path)
    if resolution is None:
        # Older benchmark `tier` may describe duration (d5/d10), not resolution.
        resolution = resolution_tier(job.get("tier")) or "p768"
    tier = resolution_tier(resolution)
    if tier not in buckets.tiers:
        raise ValueError(f"unknown resolution {resolution!r}; configured tiers: {', '.join(buckets.tiers)}")
    keyframes = req.keyframes()
    if aspect is not None:
        ratio, source = _aspect(aspect), "aspect_ratio"
    elif req.width is not None:
        ratio, source = _aspect(req.width / req.height), "width_height"
    elif keyframes and keyframes[0][1] is not None:
        anchor, image = keyframes[0]
        ratio, source = image.width / image.height, f"{anchor}_image"
    else:
        ratio, source = 16 / 9, "default_16:9"
    index = buckets.family_index(ratio, 1, tier=tier)
    req.width, req.height = buckets.canvas(tier, index)
    prepared = job.get("bucket_preparation")
    if prepared is not None and (prepared.get("config_sha256") != digest
                                 or prepared.get("tier") != tier
                                 or prepared.get("resolved_canvas") != [req.height, req.width]):
        raise ValueError("prepared benchmark geometry/config differs from this request; rebuild the job bundle")
    req.keyframe_fit = "crop"
    crops = []
    for anchor, image in keyframes:
        if image is None:
            raise ValueError(f"task {req.task} needs a {anchor} image")
        crops.append({"anchor": anchor, "source_size": [image.width, image.height],
                      "crop_box": list(center_crop_box(image.width, image.height, req.width, req.height))})
    return {"policy": policy, "config": path, "config_sha256": digest, "tier": tier,
            "bucket_index": index, "bucket_aspect": buckets.family_name(index),
            "requested_aspect": ratio, "aspect_source": source, "canvas": [req.height, req.width],
            "keyframe_fit": req.keyframe_fit, "keyframe_crops": crops}
