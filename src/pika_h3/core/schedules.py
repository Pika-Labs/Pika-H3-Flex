# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""MiniMax-H3 rectified-flow sigma schedules.

The state is x_t = (1 - sigma) x0 + sigma eps. Per-modality shifts are
sigma' = s sigma / (1 + (s - 1) sigma), with s = 12 for video and 3 for audio.
The transformer consumes clean time 1 - sigma' and predicts velocity x0 - eps.
A denoising schedule has K + 1 knots, ending in zero; only the first K knots
evaluate the generator. Beta schedules use beta-distribution quantiles plus
terminal zero before applying the same modality shifts."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch


def shift_sigma(sigma: torch.Tensor | float, shift: float) -> torch.Tensor | float:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz, Numerical Recipes ``betacf``)."""
    max_iter, eps, fpmin = 400, 1e-15, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = fpmin if abs(d) < fpmin else d
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """``I_x(a, b)`` = CDF of Beta(a, b) at ``x`` (pure python, no scipy in the H3 image; 1e-13 vs scipy.stats.beta.cdf)."""
    if not (a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"beta parameters must be finite and positive, got alpha={a} beta={b}")
    x = float(x)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_ppf(u: float, a: float, b: float) -> float:
    """Quantile of Beta(a, b): bisection on ``regularized_incomplete_beta`` (monotone), ~1e-15 after 60 halvings."""
    u = float(u)
    if not 0.0 <= u <= 1.0:
        raise ValueError(f"quantile must lie in [0, 1], got {u}")
    if u == 0.0:
        return 0.0
    if u == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(mid, a, b) < u:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


def beta_ladder_knots(steps: int, alpha: float = 0.6, beta: float = 0.6) -> list[float]:
    """Base-grid knots (x1000, LTX convention: K + 1 values, first 1000, last 0) of the beta ladder: the Beta(alpha, beta)
    quantiles of ``1, 1 - 1/K, ..., 1/K`` (``1 - linspace(0, 1, K, endpoint=False)``), no
    10 000-entry table rounding. Push them through ``shift_sigma`` with the generator shift to get the sigmas."""
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"beta ladder needs steps >= 1, got {steps}")
    regularized_incomplete_beta(0.5, alpha, beta)  # parameter validation
    knots = [1000.0] + [1000.0 * beta_ppf(1.0 - i / steps, alpha, beta) for i in range(1, steps)] + [0.0]
    return knots


LADDER_SPEC_TYPES = ("beta",)


def resolve_ladder_spec(spec: dict) -> tuple[list[float], dict]:
    """``{type: beta, steps: K, alpha: a, beta: b}`` -> (base-grid knots x1000, normalized spec). Only ``beta`` exists."""
    kind = str(spec.get("type", spec.get("kind", ""))).lower()
    if kind not in LADDER_SPEC_TYPES:
        raise ValueError(f"denoising_step_list.type must be one of {LADDER_SPEC_TYPES}, got {kind!r}")
    allowed = {"type", "kind", "steps", "alpha", "beta"}
    extra = sorted(k for k in spec if k not in allowed)
    if extra:
        raise ValueError(f"denoising_step_list beta spec has unknown keys {extra}; allowed: steps, alpha, beta")
    if "steps" not in spec:
        raise ValueError("denoising_step_list beta spec needs 'steps' (the number of generator evaluations K)")
    steps = int(spec["steps"])
    alpha, beta = float(spec.get("alpha", 0.6)), float(spec.get("beta", 0.6))
    return beta_ladder_knots(steps, alpha, beta), {"type": "beta", "steps": steps, "alpha": alpha, "beta": beta}


def _as_float_list(values: Any, name: str) -> list[float]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(values):
            values = OmegaConf.to_container(values, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    if not isinstance(values, (list, tuple)) or len(values) < 1:
        raise ValueError(f"{name} must be a non-empty list, got {values!r}")
    return [float(v) for v in values]


DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250, 0]


def _validate_knots(steps: list[float], name: str) -> None:
    """LTX convention: K + 1 knots, first 1000, strictly decreasing, LAST == 0 (the terminal re-noising target)."""
    if len(steps) < 2:
        raise ValueError(f"{name} needs at least two knots (one evaluation sigma + the terminal 0), got {steps}")
    if steps[0] != 1000.0:
        raise ValueError(f"{name} must start at 1000, got {steps}")
    if steps[-1] != 0.0:
        raise ValueError(
            f"{name} must end with a terminal 0 (LTX convention [1000, ..., 0]: the last knot is the re-noising "
            f"target, not a generator timestep). Got {steps} -> append a trailing 0."
        )
    if any(a <= b for a, b in zip(steps[:-1], steps[1:])):
        raise ValueError(f"{name} must be strictly decreasing, got {steps}")


class H3SigmaSchedule:
    """Per-modality sigma tables + few-step ladders.

    Config keys (all optional):
      num_train_timestep (1000), video_score_sigma_shift (12.0), audio_score_sigma_shift (3.0),
      denoising_step_list ([1000, 750, 500, 250, 0] base-grid knots pushed through the generator shifts: first 1000,
      LAST MUST BE 0 -- the same meaning as LTX's plain-list form, which snaps base steps onto its shifted curve; OR
      the LTX mapping form {default: [...], audio: [...]} ('video' = alias of 'default') of EXPLICIT already-shifted
      sigmas x1000 with the same 1000 ... 0 shape; per-tier keys are refused, see ``ladders``; OR the generated form
      {type: beta, steps: K, alpha: 0.6, beta: 0.6} = ``beta_ladder_knots`` pushed through the generator shifts),
      generator_video_shift (defaults to video_score_sigma_shift), generator_audio_shift (defaults to
      audio_score_sigma_shift).

    ``video_ladder`` / ``audio_ladder`` are the K + 1 shifted knots (terminal 0 included); ``num_steps`` is K, the
    number of generator evaluations; ``eval_sigmas`` are the K non-zero knots.
    """

    def __init__(self, config: Any, device: torch.device | str = "cpu"):
        get = lambda key, default: _cfg_get(config, key, default)  # noqa: E731
        self.num_train_timestep = int(get("num_train_timestep", 1000))
        self.video_shift = float(get("video_score_sigma_shift", 12.0))
        self.audio_shift = float(get("audio_score_sigma_shift", 3.0))
        self.ladder_video_shift = float(get("generator_video_shift", self.video_shift))
        self.ladder_audio_shift = float(get("generator_audio_shift", self.audio_shift))
        base = torch.arange(self.num_train_timestep + 1, dtype=torch.float64) / self.num_train_timestep
        self.video_sigma_lookup = shift_sigma(base, self.video_shift).to(torch.float32).to(device)
        self.audio_sigma_lookup = shift_sigma(base, self.audio_shift).to(torch.float32).to(device)

        step_list = get("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(step_list):
                step_list = OmegaConf.to_container(step_list, resolve=True)
        except ImportError:  # pragma: no cover
            pass
        self.ladder_spec = None
        self.base_knots: Optional[list[float]] = None
        if isinstance(step_list, dict) and ("type" in step_list or "kind" in step_list):
            step_list, self.ladder_spec = resolve_ladder_spec(dict(step_list))
        if isinstance(step_list, dict):
            # LTX-style mapping: ``default`` (video) + ``audio`` of literal sigmas x1000. ``video`` is accepted as an
            # alias of ``default``. LTX tier keys (p360 ... p1440) are refused: H3 runs ONE ladder for every tier
            # (video shift 12 / audio shift 3, also for the 768p students).
            keys = set(step_list)
            if "video" in keys and "default" in keys:
                raise ValueError("denoising_step_list mapping: use either 'video' or its alias 'default', not both")
            tier_keys = sorted(k for k in keys if k not in {"video", "default", "audio"})
            if tier_keys:
                raise ValueError(
                    f"denoising_step_list mapping has per-tier keys {tier_keys}; H3 uses one ladder for every tier "
                    "(video shift 12 / audio shift 3, 768p included) -> keep only 'default' (or 'video') and 'audio'"
                )
            video_key = "video" if "video" in keys else "default"
            if video_key not in keys or "audio" not in keys:
                raise ValueError("explicit denoising_step_list mapping needs 'default' (alias 'video') and 'audio' entries")
            video_steps = _as_float_list(step_list[video_key], f"denoising_step_list.{video_key}")
            audio_steps = _as_float_list(step_list["audio"], "denoising_step_list.audio")
            _validate_knots(video_steps, "denoising_step_list.video")
            _validate_knots(audio_steps, "denoising_step_list.audio")
            if len(video_steps) != len(audio_steps):
                raise ValueError("video and audio ladders must have the same number of knots")
            video = torch.tensor(video_steps, dtype=torch.float64) / 1000.0
            audio = torch.tensor(audio_steps, dtype=torch.float64) / 1000.0
            self.base_grid = None
        else:
            steps = _as_float_list(step_list, "denoising_step_list")
            _validate_knots(steps, "denoising_step_list")
            q = torch.tensor(steps, dtype=torch.float64) / 1000.0
            self.base_grid = q.to(torch.float32)
            self.base_knots = steps
            video = shift_sigma(q, self.ladder_video_shift)
            audio = shift_sigma(q, self.ladder_audio_shift)
        self.video_ladder = video.to(torch.float32).to(device)
        self.audio_ladder = audio.to(torch.float32).to(device)
        if not bool(torch.all(self.video_ladder[:-1] > self.video_ladder[1:])):
            raise ValueError("video ladder must be strictly decreasing")
        if not bool(torch.all(self.audio_ladder[:-1] > self.audio_ladder[1:])):
            raise ValueError("audio ladder must be strictly decreasing")

    @property
    def num_steps(self) -> int:
        """Number of generator evaluations K (the ladder holds K + 1 knots, the last one being 0)."""
        return int(self.video_ladder.numel()) - 1

    @property
    def eval_sigmas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The K non-zero knots the student is evaluated at (video, audio)."""
        return self.video_ladder[:-1], self.audio_ladder[:-1]

    def to(self, device: torch.device | str) -> "H3SigmaSchedule":
        self.video_sigma_lookup = self.video_sigma_lookup.to(device)
        self.audio_sigma_lookup = self.audio_sigma_lookup.to(device)
        self.video_ladder = self.video_ladder.to(device)
        self.audio_ladder = self.audio_ladder.to(device)
        return self

    def video_timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.long().clamp(0, self.num_train_timestep)
        return self.video_sigma_lookup.to(t.device)[t]

    def audio_timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.long().clamp(0, self.num_train_timestep)
        return self.audio_sigma_lookup.to(t.device)[t]

    def ladders(self, tier: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One ladder pair for every tier: the H3 shifter values are resolution
        independent (video 12 / audio 3, including 768p), so ``tier`` is accepted and ignored."""
        return self.video_ladder, self.audio_ladder

    def describe(self) -> str:
        spec = "" if self.ladder_spec is None else f"ladder spec={self.ladder_spec} base_knots={[round(float(v), 2) for v in self.base_knots]}; "
        return (
            f"H3SigmaSchedule(score shift video={self.video_shift} audio={self.audio_shift}; "
            f"ladder shift video={self.ladder_video_shift} audio={self.ladder_audio_shift}; K={self.num_steps}; {spec}"
            f"video_ladder={[round(float(v), 4) for v in self.video_ladder]} "
            f"audio_ladder={[round(float(v), 4) for v in self.audio_ladder]})"
        )


def inference_sigmas(num_steps: int, shift: float) -> torch.Tensor:
    """The official pipeline's grid for ``num_steps`` model evaluations: shifted linspace(1, 0, N+1)."""
    base = torch.linspace(1.0, 0.0, int(num_steps) + 1, dtype=torch.float32)
    return shift_sigma(base, shift)


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            value = getter(key, default)
            return default if value is None else value
        except Exception:  # pragma: no cover
            pass
    value = getattr(config, key, default)
    return default if value is None else value
