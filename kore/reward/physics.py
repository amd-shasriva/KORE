"""Unified physics bridge for analysis, integrity, reward, and GRPO.

Integrity and shaping are intentionally separate:

* integrity uses conservative vendor upper bounds and may reject a physically
  impossible measurement without any empirical evidence;
* residual/counter shaping requires passing family-specific held-out evidence.

Unsupported operations, dtypes, calibration, or counters return unavailable
signals.  They never receive fabricated attainment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from kore.analysis.roofline import (
    ModelError,
    PhysicalModel,
    attainment,
    estimate_work,
    evaluate_roofline,
    make_physical_model,
)
from kore.config import CONFIG
from kore.reward.reward import (
    DEFAULT_ROOFLINE_TOL,
    Observation,
    RewardResult,
    _format_component,
    compute_reward,
    roofline_ceiling_violation,
    validate_reward_config,
)
from kore.reward.shaping import FamilyShapingEvidence, evidence_for_task

DEFAULT_PHYSICS_WEIGHT = 1.0


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _fraction(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if not _finite(value) or not 0.0 <= float(value) <= 1.0:
        raise ModelError(f"{name} must be finite and in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class PhysicsSignal:
    """One timed kernel under one fingerprinted physical model."""

    t_min_ms: float
    model_fingerprint: str
    measured_ms: Optional[float] = None
    family: Optional[str] = None
    stall_frac: Optional[float] = None
    occupancy: Optional[float] = None

    def __post_init__(self) -> None:
        if not _finite(self.t_min_ms) or float(self.t_min_ms) <= 0.0:
            raise ModelError("t_min_ms must be finite and positive")
        object.__setattr__(self, "t_min_ms", float(self.t_min_ms))
        if not self.model_fingerprint:
            raise ModelError("model_fingerprint is required")
        if self.measured_ms is not None:
            if not _finite(self.measured_ms) or float(self.measured_ms) <= 0.0:
                raise ModelError("measured_ms must be finite and positive")
            object.__setattr__(self, "measured_ms", float(self.measured_ms))
        object.__setattr__(self, "stall_frac", _fraction("stall_frac", self.stall_frac))
        object.__setattr__(self, "occupancy", _fraction("occupancy", self.occupancy))


def model_from_config(
    cfg=CONFIG, *, integrity: bool = False
) -> PhysicalModel:
    """Resolve the explicit SKU/calibration fields on a runtime config."""
    sku = str(getattr(cfg, "physics_sku", "") or "")
    if not sku:
        raise ModelError("physics_sku must be configured explicitly")
    calibration = getattr(cfg, "physics_calibration_path", None)
    expected = getattr(cfg, "physics_model_fingerprint", None)
    model = make_physical_model(
        sku,
        Path(calibration) if calibration else None,
        expected_fingerprint=expected,
    )
    return model.for_integrity() if integrity else model


def _evidence_matches(
    signal: PhysicsSignal, evidence: Optional[FamilyShapingEvidence]
) -> bool:
    return bool(
        evidence
        and evidence.passes()
        and evidence.model_fingerprint == signal.model_fingerprint
        and signal.family == evidence.family
        and signal.stall_frac is not None
        and signal.occupancy is not None
    )


def named_residual_ms(
    measured_ms: float,
    signal: PhysicsSignal,
    evidence: Optional[FamilyShapingEvidence] = None,
) -> Optional[float]:
    """Evidence-backed predicted residual; unavailable without passing evidence."""
    if not _finite(measured_ms) or float(measured_ms) <= 0.0:
        return None
    if not _evidence_matches(signal, evidence):
        return None
    gap = evidence.predict_gap_fraction(signal.stall_frac, signal.occupancy)
    full_residual = max(float(measured_ms) - signal.t_min_ms, 0.0)
    return min(gap * float(measured_ms), full_residual)


def residual_descent_frac(
    signal: PhysicsSignal,
    measured_ms: Optional[float] = None,
    evidence: Optional[FamilyShapingEvidence] = None,
) -> tuple[Optional[float], bool]:
    """Return bounded diagnostic eta or evidence-backed residual credit.

    ``pmc_used`` is true only when validated coefficients—not an unvalidated
    hand-written sum of overlapping counter fractions—produced the score.
    """
    measured = signal.measured_ms if signal.measured_ms is not None else measured_ms
    if not _finite(measured) or float(measured) <= 0.0:
        return None, False
    eta = max(0.0, min(1.0, signal.t_min_ms / float(measured)))
    if not _evidence_matches(signal, evidence):
        return eta, False
    gap = evidence.predict_gap_fraction(signal.stall_frac, signal.occupancy)
    return max(0.0, min(1.0, 1.0 - gap)), True


def compute_residual_reward(
    obs: Observation,
    physics: PhysicsSignal,
    source: str = "",
    dtype: str = "fp32",
    cfg=CONFIG,
    physics_weight: float = DEFAULT_PHYSICS_WEIGHT,
    snr_threshold: Optional[float] = None,
    response: Optional[str] = None,
    phase: Optional[str] = None,
    evidence: Optional[FamilyShapingEvidence] = None,
) -> RewardResult:
    """Family-evidence-gated residual reward.

    Without passing evidence this returns the ordinary verified speedup reward;
    eta remains diagnostic and cannot silently become a shaping surface.
    """
    validate_reward_config(cfg)
    if not _finite(physics_weight) or float(physics_weight) < 0.0:
        raise ValueError("physics_weight must be finite and non-negative")
    base = compute_reward(
        obs,
        source=source,
        dtype=dtype,
        cfg=cfg,
        snr_threshold=snr_threshold,
        response=response,
        phase=phase,
    )
    if not base.correct or not _evidence_matches(physics, evidence):
        if base.correct:
            base.flags.append("physics_shaping_disabled")
            if physics.stall_frac is None or physics.occupancy is None:
                base.flags.append("no_pmc")
            base.detail += " | no passing family-held-out physics evidence"
        return base

    measured = physics.measured_ms or obs.wall_ms
    if measured is None and obs.wall_by_shape:
        valid = [v for v in obs.wall_by_shape.values() if _finite(v) and float(v) > 0.0]
        measured = max(valid) if valid else None
    rho, used = residual_descent_frac(physics, measured, evidence)
    if rho is None or not used:
        base.flags.append("physics_shaping_disabled")
        return base
    fmt = _format_component(response, cfg)
    reward = float(cfg.correctness_weight) + float(physics_weight) * rho + fmt
    if not math.isfinite(reward):
        raise ValueError("residual reward became non-finite")
    return RewardResult(
        reward,
        True,
        base.speedup,
        "correct_residual",
        list(base.flags) + ["physics_evidence_passed"],
        f"held-out residual credit={rho:.3f}; evidence={evidence.report_fingerprint}",
    )


def observation_from_measure(measure, dtype: str = "bf16") -> Observation:
    return Observation(
        compiled=True,
        snr_db=getattr(measure, "snr_db", None),
        wall_ms=getattr(measure, "cand_ms", None),
        baseline_ms=getattr(measure, "vendor_ms", None),
        validation_passed=bool(getattr(measure, "correct", False)),
        dtype=dtype,
    )


def physics_from_measure(
    measure, model: Optional[PhysicalModel] = None
) -> Optional[PhysicsSignal]:
    try:
        from kore.eval.generalization import family_of

        return PhysicsSignal(
            t_min_ms=getattr(measure, "t_min_ms"),
            measured_ms=getattr(measure, "cand_ms", None),
            model_fingerprint=str(
                getattr(measure, "model_fingerprint", None)
                or (model.fingerprint if model is not None else "legacy-unfingerprinted")
            ),
            family=family_of(getattr(measure, "task_id", "")),
            stall_frac=getattr(measure, "stall_frac", None),
            occupancy=getattr(measure, "occupancy", None),
        )
    except (ModelError, TypeError, ValueError):
        return None


def _shape_walls(task, obs) -> list[tuple[str, float, object]]:
    walls = dict(getattr(obs, "wall_by_shape", None) or {})
    if not walls and _finite(getattr(obs, "wall_ms", None)) and obs.wall_ms > 0.0:
        primary = task.shape("primary") if hasattr(task, "shape") else None
        walls = {(primary.name if primary else "primary"): obs.wall_ms}
    out = []
    for name, wall in walls.items():
        if not _finite(wall) or float(wall) <= 0.0:
            continue
        shape = task.shape(name) if hasattr(task, "shape") else None
        if shape is None and name == "primary" and hasattr(task, "shape"):
            shape = task.shape("primary")
        out.append((str(name), float(wall), shape))
    return out


def physics_signal_from_obs(
    task,
    obs,
    model: Optional[PhysicalModel] = None,
    arch: Optional[str] = None,
) -> Optional[PhysicsSignal]:
    """Worst-shape timing signal under one explicit model."""
    if model is None:
        if arch not in {"gfx950", "gfx942"}:
            return None
        model = make_physical_model("mi350x" if arch == "gfx950" else "mi300x")
    worst: Optional[tuple[float, float, float]] = None
    for _, wall, shape in _shape_walls(task, obs):
        dims = getattr(shape, "dims", None)
        work = estimate_work(
            getattr(task, "operation", ""),
            dims or {},
            getattr(task, "dtype", ""),
        )
        result = evaluate_roofline(work, model) if work else None
        if result is None:
            continue
        eta = attainment(wall, result)
        if eta is not None and (worst is None or eta < worst[0]):
            worst = (eta, result.t_min_ms, wall)
    if worst is None:
        return None
    try:
        from kore.eval.generalization import family_of

        family = family_of(getattr(task, "task_id", ""))
    except Exception:
        family = None
    return PhysicsSignal(
        t_min_ms=worst[1],
        measured_ms=worst[2],
        model_fingerprint=model.fingerprint,
        family=family,
    )


def roofline_ceiling_violation_from_obs(
    task,
    obs,
    tol: float = DEFAULT_ROOFLINE_TOL,
    arch: Optional[str] = None,
    *,
    model: Optional[PhysicalModel] = None,
) -> tuple[bool, str]:
    """Integrity-only super-SOL check.

    Compute work is always mandatory.  The HBM floor is included only when the
    observation explicitly records verified cold-cache timing.
    """
    if model is None:
        if arch not in {"gfx950", "gfx942"}:
            return False, "physical model unavailable"
        model = make_physical_model("mi350x" if arch == "gfx950" else "mi300x")
    integrity_model = model.for_integrity()
    cold_cache = bool(getattr(obs, "cold_cache_verified", False))
    for name, wall, shape in _shape_walls(task, obs):
        work = estimate_work(
            getattr(task, "operation", ""),
            getattr(shape, "dims", {}) if shape is not None else {},
            getattr(task, "dtype", ""),
        )
        result = evaluate_roofline(work, integrity_model) if work else None
        if result is None:
            continue
        floor = (
            result.t_min_ms
            if cold_cache
            else result.t_compute_ms
        )
        if roofline_ceiling_violation(wall, floor, tol):
            basis = "compute+HBM cold-cache" if cold_cache else "mandatory compute"
            return True, (
                f"shape {name}: measured {wall:.6g} ms < integrity floor "
                f"{floor:.6g} ms ({basis}, tol={tol:.3f}, "
                f"model={integrity_model.fingerprint})"
            )
    return False, ""


def mask_reward_phase(
    reward_result: RewardResult, phase: str, correctness_weight: float
) -> RewardResult:
    if str(phase).lower() == "correctness" and reward_result.correct:
        if not _finite(correctness_weight):
            raise ValueError("correctness_weight must be finite")
        return replace(
            reward_result,
            reward=float(correctness_weight),
            speedup=None,
            tier="correct_masked",
        )
    return reward_result


def compute_kernel_reward(
    obs: Observation,
    source: str,
    task,
    *,
    mode: str = "speedup",
    dtype: str = "fp32",
    cfg=CONFIG,
    snr_threshold: Optional[float] = None,
    physics_weight: float = DEFAULT_PHYSICS_WEIGHT,
    response: Optional[str] = None,
    reward_phase: str = "all",
    roofline_gate: bool = False,
    roofline_tol: float = DEFAULT_ROOFLINE_TOL,
    arch: Optional[str] = None,
    model: Optional[PhysicalModel] = None,
    physics_config=None,
) -> RewardResult:
    """Single online dispatch using the same model as reports and P0."""
    validate_reward_config(cfg)
    physics_cfg = physics_config or cfg
    selected = model
    if selected is None:
        try:
            selected = model_from_config(physics_cfg)
        except (ModelError, OSError):
            if arch in {"gfx950", "gfx942"}:
                selected = make_physical_model("mi350x" if arch == "gfx950" else "mi300x")
    if roofline_gate and selected is not None:
        violated, reason = roofline_ceiling_violation_from_obs(
            task, obs, roofline_tol, model=selected
        )
        if violated:
            return RewardResult(
                cfg.reward_hack,
                False,
                None,
                "hack",
                ["hack", "roofline_ceiling"],
                reason,
            )

    evidence = (
        evidence_for_task(task, physics_cfg, selected.fingerprint)
        if selected is not None
        else None
    )
    if mode == "residual" and selected is not None and evidence is not None:
        signal = physics_signal_from_obs(task, obs, selected)
        if signal is not None:
            reward = compute_residual_reward(
                obs,
                signal,
                source=source,
                dtype=dtype,
                cfg=cfg,
                physics_weight=physics_weight,
                snr_threshold=snr_threshold,
                response=response,
                evidence=evidence,
            )
            return mask_reward_phase(reward, reward_phase, cfg.correctness_weight)
    reward = compute_reward(
        obs,
        source,
        dtype=dtype,
        cfg=cfg,
        snr_threshold=snr_threshold,
        response=response,
    )
    if mode == "residual" and reward.correct:
        reward.flags.append("physics_shaping_disabled")
    return mask_reward_phase(reward, reward_phase, cfg.correctness_weight)


__all__ = [
    "DEFAULT_PHYSICS_WEIGHT",
    "PhysicsSignal",
    "compute_kernel_reward",
    "compute_residual_reward",
    "mask_reward_phase",
    "model_from_config",
    "named_residual_ms",
    "observation_from_measure",
    "physics_from_measure",
    "physics_signal_from_obs",
    "residual_descent_frac",
    "roofline_ceiling_violation_from_obs",
]
