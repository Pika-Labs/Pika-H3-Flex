# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Resolution buckets derived from the model canvas rule.

Canvas sides lie on the 32-pixel lattice, with a fixed short edge and area cap.
Aspect-family indices stay aligned across tiers. Frame counts use the 17n + 5
grid. Reference images use a 1024-pixel short edge below 768p and 2048 pixels
at 768p and above, with upscaling allowed and no area cap."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import geometry as G

__all__ = [
    "DEFAULT_ASPECTS",
    "DEFAULT_FRAMES",
    "DEFAULT_TIER_RULES",
    "BucketConfig",
    "TierRule",
    "build_default_config",
    "derive_tier",
    "load_bucket_config",
    "reference_short_edge_for_canvas",
    "reference_short_edge_for_tier",
    "save_bucket_config",
    "select_frames",
    "tier_for_canvas",
]

# Landscape families first, then square, then the mirrored portraits — same 11 families as multires_v2.
DEFAULT_ASPECTS: Tuple[str, ...] = ("21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16", "9:21")

# 17n+5 from 2.3 s to 15.1 s at 24 fps (n = 3 .. 21); the official 5 / 10 / 15 s clips are 124 / 243 / 362.
DEFAULT_FRAMES: Tuple[int, ...] = tuple(17 * n + 5 for n in range(3, 22))


@dataclass(frozen=True)
class TierRule:
    short_edge: int
    max_pixels: int
    reference_short_edge: int = 2048  # Ref2VA image references (upscaled, uncapped, nearest 32)


DEFAULT_TIER_RULES: Dict[str, TierRule] = {
    "p480": TierRule(480, 832 * 480, reference_short_edge=1024),
    "p544": TierRule(544, 960 * 544, reference_short_edge=1024),
    "p768": TierRule(768, 1344 * 768, reference_short_edge=2048),
    "p1080": TierRule(1088, 1920 * 1088, reference_short_edge=2048),
    "p1440": TierRule(1440, 2560 * 1440, reference_short_edge=2048),
    "p2160": TierRule(2176, 3840 * 2176, reference_short_edge=2048),  # 4K UHD snapped to the 32-px VAE lattice
}


def tier_for_canvas(height: int, width: int, rules: Optional[Dict[str, TierRule]] = None) -> str:
    """The tier a canvas belongs to: the smallest tier whose short edge is >= the canvas short edge (960x416 is a p480
    21:9 bucket, 1536x672 a p768 one), the largest tier above every rule."""
    rules = rules or DEFAULT_TIER_RULES
    if not rules:
        raise ValueError("no tier rules")
    if height <= 0 or width <= 0:
        raise ValueError(f"canvas must be positive, got {height}x{width}")
    short = min(int(height), int(width))
    ordered = sorted(rules.items(), key=lambda item: item[1].short_edge)
    for name, rule in ordered:
        if rule.short_edge >= short:
            return name
    return ordered[-1][0]


def reference_short_edge_for_tier(tier: str, rules: Optional[Dict[str, TierRule]] = None) -> int:
    rules = rules or DEFAULT_TIER_RULES
    if tier not in rules:
        raise KeyError(f"no tier rule for {tier!r}; known: {sorted(rules)}")
    return int(rules[tier].reference_short_edge)


def reference_short_edge_for_canvas(height: int, width: int, rules: Optional[Dict[str, TierRule]] = None) -> int:
    """The image-reference short edge for an output canvas (832x480 -> 1024, 1344x768 -> 2048)."""
    return reference_short_edge_for_tier(tier_for_canvas(height, width, rules), rules)


def _parse_aspect(aspect: str) -> Tuple[int, int]:
    w, h = aspect.split(":")
    return int(w), int(h)


def derive_tier(rule: TierRule, aspects: Sequence[str] = DEFAULT_ASPECTS, canvas_multiple: int = G.CANVAS_MULTIPLE) -> Tuple[Tuple[int, int], ...]:
    """``((width, height), ...)`` of one tier, one bucket per aspect family, from the pipeline canvas rule."""
    out = []
    for aspect in aspects:
        aw, ah = _parse_aspect(aspect)
        height, width = G.resolve_canvas_size(aw, ah, short_edge=rule.short_edge, max_pixels=rule.max_pixels, canvas_multiple=canvas_multiple)
        out.append((int(width), int(height)))
    return tuple(out)


def select_frames(num_frames: int, frames: Sequence[int], policy: str = "truncate") -> int:
    """Pick the bucket frame count for a clip of ``num_frames`` decoded frames.

    ``truncate`` (default, what preprocessing wants): the largest entry <= num_frames, 0 if none fits.
    ``pad``: the smallest entry >= num_frames, the largest entry if the clip is longer than all of them (LTX's rule).
    """
    ordered = sorted(int(f) for f in frames)
    if policy == "truncate":
        fitting = [f for f in ordered if f <= num_frames]
        return fitting[-1] if fitting else 0
    if policy == "pad":
        fitting = [f for f in ordered if f >= num_frames]
        return fitting[0] if fitting else ordered[-1]
    raise ValueError(f"unknown frame policy {policy!r}")


@dataclass(frozen=True)
class BucketConfig:
    tiers: Dict[str, Tuple[Tuple[int, int], ...]]
    frames: Tuple[int, ...]
    reference_tier: str
    aspects: Tuple[str, ...] = DEFAULT_ASPECTS
    fps: int = G.FPS
    canvas_multiple: int = G.CANVAS_MULTIPLE
    tier_rules: Dict[str, TierRule] = field(default_factory=dict)
    path: Optional[str] = None

    # ---- validation -------------------------------------------------------------------------------------------
    def validate(self) -> "BucketConfig":
        if not self.tiers:
            raise ValueError("bucket config has no tiers")
        if self.reference_tier not in self.tiers:
            raise ValueError(f"reference_tier {self.reference_tier!r} is not one of {sorted(self.tiers)}")
        n_fam = {name: len(rows) for name, rows in self.tiers.items()}
        if len(set(n_fam.values())) != 1:
            raise ValueError(f"tiers disagree on the number of aspect families: {n_fam}")
        if self.aspects and len(self.aspects) != next(iter(n_fam.values())):
            raise ValueError(f"{len(self.aspects)} aspect names for {next(iter(n_fam.values()))} families")
        for name, rows in self.tiers.items():
            for w, h in rows:
                if w % self.canvas_multiple or h % self.canvas_multiple:
                    raise ValueError(f"{name}: {w}x{h} is not a multiple of {self.canvas_multiple}")
                if w <= 0 or h <= 0:
                    raise ValueError(f"{name}: non-positive bucket {w}x{h}")
        if not self.frames:
            raise ValueError("bucket config has no frames")
        for f in self.frames:
            if f % G.VAE_FRAMES_PER_CHUNK != G.VAE_LATENTS_PER_CHUNK:
                raise ValueError(f"frames must be 17n+5, got {f}")
        return self

    # ---- lookups ----------------------------------------------------------------------------------------------
    @property
    def tier_names(self) -> List[str]:
        return list(self.tiers)

    @property
    def num_families(self) -> int:
        return len(self.tiers[self.reference_tier])

    def family_index(self, width: int, height: int, *, tier: Optional[str] = None) -> int:
        """Nearest log-aspect in the requested (default: reference) tier; ties use the lower index."""
        if width <= 0 or height <= 0:
            raise ValueError(f"source size must be positive, got {width}x{height}")
        target = math.log(width / height)
        rows = self.tiers[self.reference_tier if tier is None else tier]
        return min(range(len(rows)), key=lambda i: (abs(target - math.log(rows[i][0] / rows[i][1])), i))

    def family_name(self, index: int) -> str:
        if self.aspects:
            return self.aspects[index]
        w, h = self.tiers[self.reference_tier][index]
        return f"{w}x{h}"

    def canvas(self, tier: str, family_index: int) -> Tuple[int, int]:
        """``(width, height)`` of one bucket."""
        return self.tiers[tier][family_index]

    def select_frames(self, num_frames: int, policy: str = "truncate") -> int:
        return select_frames(num_frames, self.frames, policy)

    def bucket_geometry(self, tier: str, family_index: int, num_frames: int) -> Dict[str, int]:
        """Geometry information for one (tier, family, frames) cell."""
        width, height = self.canvas(tier, family_index)
        lat_h, lat_w = G.latent_hw(height, width)
        return {
            "width": width,
            "height": height,
            "num_frames": int(num_frames),
            "num_latent_frames": G.video_latent_num_frames(int(num_frames)),
            "latent_height": lat_h,
            "latent_width": lat_w,
            "num_audio_latents": G.audio_latent_num_frames(int(num_frames)),
        }

    def packed_tokens(self, tier: str, family_index: int, num_frames: int, *, num_text_tokens: int = G.MAX_TEXT_TOKENS, num_keyframes: int = 0, reference_video_rows: int = 0) -> int:
        g = self.bucket_geometry(tier, family_index, num_frames)
        return G.packed_token_count(
            num_text_tokens=num_text_tokens,
            num_latent_frames=g["num_latent_frames"],
            latent_height=g["latent_height"],
            latent_width=g["latent_width"],
            num_audio_latents=g["num_audio_latents"],
            num_keyframes=num_keyframes,
            reference_video_rows=reference_video_rows,
        ).total

    # ---- serialisation ----------------------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "model": "minimax-h3",
            "fps": int(self.fps),
            "canvas_multiple": int(self.canvas_multiple),
            "reference_tier": self.reference_tier,
            "aspects": list(self.aspects),
        }
        if self.tier_rules:
            out["tier_rules"] = {name: {"short_edge": r.short_edge, "max_pixels": r.max_pixels, "reference_short_edge": r.reference_short_edge}
                                 for name, r in self.tier_rules.items()}
        out["frames"] = [int(f) for f in self.frames]
        out["tiers"] = {name: [[int(w), int(h)] for w, h in rows] for name, rows in self.tiers.items()}
        return out

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], path: Optional[str] = None) -> "BucketConfig":
        tiers_raw = raw.get("tiers") or {}
        tiers = {str(name): tuple((int(r[0]), int(r[1])) for r in rows) for name, rows in tiers_raw.items()}
        frames = tuple(int(f) for f in raw.get("frames") or [])
        reference_tier = str(raw.get("reference_tier") or ("p768" if "p768" in tiers else next(iter(tiers), "")))
        aspects = tuple(str(a) for a in raw.get("aspects") or ())
        rules = {}
        for n, r in (raw.get("tier_rules") or {}).items():
            default = DEFAULT_TIER_RULES.get(str(n))
            # When reference_short_edge is absent, use 1024 below 768 and 2048 otherwise.
            reference = r.get("reference_short_edge", default.reference_short_edge if default else (1024 if int(r["short_edge"]) < 768 else 2048))
            rules[str(n)] = TierRule(int(r["short_edge"]), int(r["max_pixels"]), int(reference))
        return cls(
            tiers=tiers,
            frames=frames,
            reference_tier=reference_tier,
            aspects=aspects,
            fps=int(raw.get("fps", G.FPS)),
            canvas_multiple=int(raw.get("canvas_multiple", G.CANVAS_MULTIPLE)),
            tier_rules=rules,
            path=path,
        ).validate()


def build_default_config(tier_names: Iterable[str] = ("p480", "p768", "p1080", "p1440"), *, frames: Sequence[int] = DEFAULT_FRAMES, aspects: Sequence[str] = DEFAULT_ASPECTS, reference_tier: str = "p768", rules: Optional[Dict[str, TierRule]] = None) -> BucketConfig:
    rules = dict(rules or DEFAULT_TIER_RULES)
    names = list(tier_names)
    for name in names:
        if name not in rules:
            raise KeyError(f"no tier rule for {name!r}; known: {sorted(rules)}")
    tiers = {name: derive_tier(rules[name], aspects) for name in names}
    ref = reference_tier if reference_tier in tiers else names[0]
    return BucketConfig(tiers=tiers, frames=tuple(int(f) for f in frames), reference_tier=ref, aspects=tuple(aspects), tier_rules={n: rules[n] for n in names}).validate()


def load_bucket_config(path: str | Path) -> BucketConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return BucketConfig.from_dict(raw, path=str(path))


def save_bucket_config(cfg: BucketConfig, path: str | Path, header: str = "") -> Path:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = cfg.to_dict()
    lines = [f"# {line}" if line else "#" for line in header.splitlines()] if header else []
    lines += [
        f"model: {d['model']}",
        f"fps: {d['fps']}",
        f"canvas_multiple: {d['canvas_multiple']}",
        f"reference_tier: {d['reference_tier']}",
        "aspects: [" + ", ".join(f'"{a}"' for a in d["aspects"]) + "]",
    ]
    if d.get("tier_rules"):
        lines.append("tier_rules:")
        for name, r in d["tier_rules"].items():
            lines.append(f"  {name}: {{short_edge: {r['short_edge']}, max_pixels: {r['max_pixels']}, reference_short_edge: {r['reference_short_edge']}}}")
    lines.append("frames: [" + ", ".join(str(f) for f in d["frames"]) + "]")
    lines.append("tiers:")
    for name, rows in d["tiers"].items():
        lines.append(f"  {name}:")
        for (w, h), aspect in zip(rows, d["aspects"] or [""] * len(rows)):
            lines.append(f"    - [{w}, {h}]" + (f"  # {aspect}" if aspect else ""))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # round-trip check: what we wrote is what we meant
    reloaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if reloaded["tiers"] != d["tiers"] or reloaded["frames"] != d["frames"] or reloaded.get("tier_rules") != d.get("tier_rules"):
        raise RuntimeError(f"bucket config round-trip mismatch writing {path}")
    return path
