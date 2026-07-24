"""Counter extraction and evidence-gated white-box potentials."""

from __future__ import annotations

from typing import Mapping, Optional

from kore.analysis.roofline import (
    CounterUnit,
    PhysicalModel,
    WorkEstimate,
    counter_value,
    derived_percent,
    est_occupancy,
    make_physical_model,
)
from kore.reward import profile_reward as _profile
from kore.reward.physics import (
    PhysicsSignal,
    physics_signal_from_obs,
    residual_descent_frac,
)
from kore.reward.shaping import FamilyShapingEvidence


def _first(
    counters: Mapping[str, object], unit: CounterUnit, *names: str
) -> Optional[float]:
    for name in names:
        value = counter_value(counters or {}, name, unit)
        if value is not None:
            return value
    return None


def stall_frac_from_counters(counters: Mapping[str, object]) -> Optional[float]:
    """Derived memory-stall percentage, or unavailable.

    Raw wait quad-cycles are not divided by instruction counts.
    """
    return derived_percent(
        counters or {}, "MemUnitStalled", "MemUnitStalled_pct", "MEM_UNIT_STALLED"
    )


def occupancy_from_counters(
    counters: Mapping[str, object], model: Optional[PhysicalModel] = None
) -> Optional[float]:
    derived = derived_percent(
        counters or {}, "OccupancyPercent", "Occupancy", "OCCUPANCY_PERCENT"
    )
    if derived is not None:
        return derived
    vgpr = _first(counters or {}, CounterUnit.COUNT, "vgpr_count", "VGPR")
    if vgpr is None or model is None:
        return None
    lds = _first(counters or {}, CounterUnit.BYTES, "lds_bytes", "LDS_BYTES")
    warps = _first(counters or {}, CounterUnit.COUNT, "num_warps", "WARPS")
    try:
        return est_occupancy(
            int(vgpr),
            int(lds) if lds is not None else None,
            int(warps) if warps is not None else None,
            model=model,
        ).occupancy
    except (TypeError, ValueError):
        return None


def physics_signal_from_counters(
    task,
    obs,
    counters: Optional[Mapping[str, object]],
    arch: Optional[str] = None,
    *,
    model: Optional[PhysicalModel] = None,
) -> Optional[PhysicsSignal]:
    if model is None:
        if arch not in {"gfx950", "gfx942"}:
            return None
        model = make_physical_model("mi350x" if arch == "gfx950" else "mi300x")
    base = physics_signal_from_obs(task, obs, model)
    if base is None or not counters:
        return base
    stall = stall_frac_from_counters(counters)
    occupancy = occupancy_from_counters(counters, model)
    if stall is None or occupancy is None:
        # Both validated features are required by the preregistered model.
        return base
    return PhysicsSignal(
        t_min_ms=base.t_min_ms,
        measured_ms=base.measured_ms,
        model_fingerprint=base.model_fingerprint,
        family=base.family,
        stall_frac=stall,
        occupancy=occupancy,
    )


def whitebox_attainment(
    task,
    obs,
    counters: Optional[Mapping[str, object]] = None,
    arch: Optional[str] = None,
    *,
    model: Optional[PhysicalModel] = None,
    evidence: Optional[FamilyShapingEvidence] = None,
) -> tuple[Optional[float], bool]:
    """Diagnostic eta, upgraded only by matching passing evidence."""
    signal = physics_signal_from_counters(
        task, obs, counters, arch, model=model
    )
    if signal is None:
        return None, False
    return residual_descent_frac(signal, evidence=evidence)


def whitebox_structural_score(
    counters: Mapping[str, object],
    *,
    work: Optional[WorkEstimate] = None,
    model: Optional[PhysicalModel] = None,
    flops: Optional[float] = None,
    bytes: Optional[float] = None,
    measured_ms: Optional[float] = None,
    dtype: str = "bf16",
    ref: Optional[Mapping[str, object]] = None,
) -> Optional[float]:
    """Bounded diagnostic score; not a reward authorization."""
    return _profile.roofline_dense_score(
        counters or {},
        ref,
        work=work,
        model=model,
        flops=flops,
        bytes=bytes,
        measured_ms=measured_ms,
        dtype=dtype,
    )


def phi_potential(
    task,
    obs,
    counters: Optional[Mapping[str, object]] = None,
    arch: Optional[str] = None,
    *,
    model: Optional[PhysicalModel] = None,
    evidence: Optional[FamilyShapingEvidence] = None,
) -> Optional[float]:
    """Finite ``Phi`` only when family-held-out evidence passes.

    Timing-only eta and unvalidated counter heuristics remain diagnostics and do
    not become a GRPO potential.
    """
    if evidence is None or not evidence.passes():
        return None
    value, used = whitebox_attainment(
        task,
        obs,
        counters,
        arch,
        model=model,
        evidence=evidence,
    )
    if not used or value is None or not 0.0 <= value <= 1.0:
        return None
    return value


__all__ = [
    "occupancy_from_counters",
    "phi_potential",
    "physics_signal_from_counters",
    "stall_frac_from_counters",
    "whitebox_attainment",
    "whitebox_structural_score",
]
