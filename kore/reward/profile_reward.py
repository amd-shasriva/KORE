"""Dimensionally valid counter diagnostics.

These functions compute bounded *diagnostic* metrics.  Reward application is
separately evidence-gated in :mod:`kore.reward.shaping`; counter availability by
itself never authorizes a training signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

from kore.analysis.roofline import (
    CounterUnit,
    PhysicalModel,
    WorkEstimate,
    attained_fraction,
    counter_value,
    derived_percent,
    hbm_bytes,
    issued_instructions as _issued_instructions,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _exact(counters: Mapping[str, object], name: str, unit: CounterUnit) -> Optional[float]:
    return counter_value(counters or {}, name, unit)


def issued_instructions(counters: Mapping[str, object]) -> Optional[float]:
    """Issued instructions, excluding MOPS and MFMA double counting."""
    return _issued_instructions(counters or {})


def stall_fraction(counters: Mapping[str, object]) -> Optional[float]:
    """Memory-stall fraction from a percentage metric only.

    ``SQ_WAIT_INST_ANY`` is measured in quad-cycles while ``SQ_INSTS_*`` is an
    instruction count.  Their historical ratio was dimensionally invalid, so raw
    wait counters alone now yield ``None``.
    """
    return derived_percent(
        counters or {}, "MemUnitStalled", "MemUnitStalled_pct", "MEM_UNIT_STALLED"
    )


def issue_efficiency(counters: Mapping[str, object]) -> Optional[float]:
    fraction = stall_fraction(counters)
    return None if fraction is None else 1.0 - fraction


def _vmem_instructions(counters: Mapping[str, object]) -> Optional[float]:
    return _exact(counters or {}, "SQ_INSTS_VMEM", CounterUnit.INSTRUCTIONS)


@dataclass(frozen=True)
class ProfileMetrics:
    cand_stall_fraction: Optional[float]
    ref_stall_fraction: Optional[float]
    cand_issue_efficiency: Optional[float]
    ref_issue_efficiency: Optional[float]
    cand_vmem_instructions: Optional[float]
    ref_vmem_instructions: Optional[float]
    cand_hbm_bytes: Optional[float]
    ref_hbm_bytes: Optional[float]
    efficiency_score: Optional[float]


def profile_efficiency_score(
    cand: Mapping[str, object], ref: Mapping[str, object]
) -> Optional[float]:
    """Bounded same-unit candidate/reference diagnostics.

    Components are (a) derived percentage efficiency, (b) exact HBM bytes when
    transaction-size split counters exist, and otherwise (c) VMEM instruction
    count.  The latter is explicitly an instruction-pressure proxy, not bytes.
    """
    components: list[float] = []
    cand_eff, ref_eff = issue_efficiency(cand), issue_efficiency(ref)
    if cand_eff is not None and ref_eff is not None and ref_eff > 0.0:
        components.append(_clamp01(cand_eff / ref_eff))

    cand_bytes, ref_bytes = hbm_bytes(cand), hbm_bytes(ref)
    if cand_bytes is not None and ref_bytes is not None and cand_bytes > 0.0 and ref_bytes > 0.0:
        components.append(_clamp01(ref_bytes / cand_bytes))
    else:
        cand_vmem, ref_vmem = _vmem_instructions(cand), _vmem_instructions(ref)
        if cand_vmem is not None and ref_vmem is not None and cand_vmem > 0.0 and ref_vmem > 0.0:
            components.append(_clamp01(ref_vmem / cand_vmem))

    return sum(components) / len(components) if components else None


def roofline_dense_score(
    cand: Mapping[str, object],
    ref: Optional[Mapping[str, object]] = None,
    *,
    work: Optional[WorkEstimate] = None,
    model: Optional[PhysicalModel] = None,
    flops: Optional[float] = None,
    bytes: Optional[float] = None,
    measured_ms: Optional[float] = None,
    dtype: str = "bf16",
) -> Optional[float]:
    """Bounded diagnostic blend, never an implicit reward authorization.

    Roofline attainment is available only when an explicit fingerprinted model
    and typed work estimate (or validated scalar compatibility inputs) are given.
    """
    components: list[float] = []
    selected_work = work
    if selected_work is None and flops is not None and bytes is not None:
        try:
            selected_work = WorkEstimate(
                "compat-explicit-work", dtype, flops, bytes, "exact"
            )
        except ValueError:
            selected_work = None
    if (
        selected_work is not None
        and model is not None
        and measured_ms is not None
        and isinstance(measured_ms, (int, float))
        and not isinstance(measured_ms, bool)
        and math.isfinite(float(measured_ms))
        and float(measured_ms) > 0.0
    ):
        percentage = attained_fraction(
            float(measured_ms),
            selected_work.flops,
            selected_work.bytes,
            selected_work.dtype,
            model,
        )
        if percentage is not None:
            components.append(_clamp01(percentage / 100.0))

    efficiency = issue_efficiency(cand or {})
    if efficiency is not None:
        components.append(_clamp01(efficiency))

    if ref:
        relative = profile_efficiency_score(cand or {}, ref)
        if relative is not None:
            components.append(_clamp01(relative))

    return sum(components) / len(components) if components else None


def profile_metrics(
    cand: Mapping[str, object], ref: Mapping[str, object]
) -> ProfileMetrics:
    score = profile_efficiency_score(cand, ref)
    return ProfileMetrics(
        cand_stall_fraction=stall_fraction(cand),
        ref_stall_fraction=stall_fraction(ref),
        cand_issue_efficiency=issue_efficiency(cand),
        ref_issue_efficiency=issue_efficiency(ref),
        cand_vmem_instructions=_vmem_instructions(cand),
        ref_vmem_instructions=_vmem_instructions(ref),
        cand_hbm_bytes=hbm_bytes(cand),
        ref_hbm_bytes=hbm_bytes(ref),
        efficiency_score=score,
    )


__all__ = [
    "ProfileMetrics",
    "issue_efficiency",
    "issued_instructions",
    "profile_efficiency_score",
    "profile_metrics",
    "roofline_dense_score",
    "stall_fraction",
]
