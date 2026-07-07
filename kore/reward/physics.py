"""Physics residual-descent reward (KORE P0 -> reward, Phase 5).

Scores a CORRECT kernel by how much of the *named* runtime residual it has
removed relative to the roofline lower bound ``T_min`` -- an ABSOLUTE,
arch-normalized, physics-grounded signal, NOT a relative speedup versus a vendor
baseline. It is built directly on the validated check-(b) decomposition
(``kore.analysis.p0_sol``), which fits the residual to counter-derived stall /
occupancy-deficit time with R^2 ~ 0.99 on gfx950:

    T_meas = T_min + R ,   R = residual (removable, in principle, down to T_min)
    named residual  N = (stall_frac + occupancy_deficit) * T_meas
                      = MemUnitStalled/100 * T_meas + (1 - OccupancyPercent/100) * T_meas

The residual-descent credit is the physics floor as a fraction of the floor plus
the *named* residual that is still present:

    rho_phys = T_min / (T_min + N)   in (0, 1]      [PMC available]

``rho_phys -> 1`` as the kernel drives the named residual ``N -> 0`` (it
approaches the roofline). When PMC counters are unavailable we cannot attribute
the residual to named terms, so we degrade gracefully to the timing-only SOL
attainment

    eta = T_min / T_meas             in (0, 1]      [PMC-free fallback, flagged]

which uses the *full* residual ``R`` instead of the named part ``N`` (note
``eta <= rho_phys`` since ``N <= R``). The fallback is flagged ``no_pmc``.

Anti-hack lexicographic ordering is preserved ABSOLUTELY: every gate
(hack < compile_fail < incorrect) is delegated verbatim to
:func:`kore.reward.reward.compute_reward`; only the tier-3 (correct) speed term
is replaced with the physics credit. A kernel can therefore never trade
correctness for residual credit, and a reward hack is still the unique floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kore.config import CONFIG
from kore.reward.reward import (  # noqa: F401 - Observation re-exported for callers
    Observation,
    RewardResult,
    _format_component,
    compute_reward,
)

# Default weight on the physics credit (rho in (0,1]) for a correct kernel, so a
# correct kernel scores in [correctness_weight, correctness_weight + physics_weight].
# Kept O(1) like the speedup term it replaces; correctness_weight alone already
# dominates the shaped-incorrect ceiling (eps_shape + format_weight), so
# lexicographic dominance of the correct tier holds for any physics_weight >= 0.
DEFAULT_PHYSICS_WEIGHT = 1.0


@dataclass
class PhysicsSignal:
    """Physics inputs for one kernel at one shape.

    ``t_min_ms`` is the roofline lower bound (:func:`kore.analysis.rooflines.roofline`).
    ``stall_frac`` and ``occupancy`` are rocprofv3 gfx950 derived metrics normalized
    to ``[0, 1]`` (``MemUnitStalled/100`` and ``OccupancyPercent/100``); either may be
    None when PMC is unavailable, which triggers the eta fallback. ``measured_ms``
    overrides the candidate wall time (else taken from the Observation).
    """

    t_min_ms: float
    measured_ms: Optional[float] = None
    stall_frac: Optional[float] = None
    occupancy: Optional[float] = None


def named_residual_ms(measured_ms: float, sig: PhysicsSignal) -> Optional[float]:
    """Named residual time ``N = (stall + occupancy_deficit) * T_meas``.

    Returns None when NO PMC counter is present (neither stall nor occupancy), so
    the caller can fall back to the timing-only eta. The named residual is a PART
    of the total residual ``R = T_meas - T_min`` (raw derived counters overlap and
    can double-count a cycle as both stalled and low-occupancy), so it is clamped
    to ``[0, R]``. This guarantees ``eta <= rho_phys <= 1``: crediting the named
    residual is never harsher than the timing-only fallback.
    """
    if sig.stall_frac is None and sig.occupancy is None:
        return None
    stall = max(0.0, sig.stall_frac or 0.0)
    occ_deficit = max(0.0, 1.0 - (sig.occupancy if sig.occupancy is not None else 1.0))
    n_raw = (stall + occ_deficit) * measured_ms
    t_min = sig.t_min_ms
    residual = (max(measured_ms - t_min, 0.0) if (t_min is not None and t_min == t_min)
                else measured_ms)
    return min(max(n_raw, 0.0), residual)


def residual_descent_frac(sig: PhysicsSignal,
                          measured_ms: Optional[float] = None) -> tuple[Optional[float], bool]:
    """Return ``(rho, pmc_used)``: the residual-descent credit in ``(0, 1]``.

    PMC path: ``rho = T_min / (T_min + N)`` (rewards removing the NAMED residual).
    Fallback: ``eta = T_min / T_meas`` (uses the full residual; ``pmc_used=False``).
    Returns ``(None, False)`` when neither ``T_min`` nor a positive wall time exists.
    """
    t_meas = sig.measured_ms if sig.measured_ms is not None else measured_ms
    t_min = sig.t_min_ms
    if not t_meas or t_meas <= 0 or t_min is None or not (t_min > 0):
        return None, False
    n = named_residual_ms(t_meas, sig)
    if n is None:
        return min(t_min / t_meas, 1.0), False
    return min(t_min / (t_min + n), 1.0), True


def compute_residual_reward(obs: Observation, physics: PhysicsSignal, source: str = "",
                            dtype: str = "fp32", cfg=CONFIG,
                            physics_weight: float = DEFAULT_PHYSICS_WEIGHT,
                            snr_threshold: Optional[float] = None,
                            response: Optional[str] = None,
                            phase: Optional[str] = None) -> RewardResult:
    """Residual-descent reward with the exact :class:`RewardResult` ABI.

    Delegates hack / compile / incorrect gating verbatim to
    :func:`kore.reward.reward.compute_reward` (so anti-hack ordering is
    byte-identical) and, only for a CORRECT kernel, replaces the tier-3 relative
    speedup with the absolute physics credit ``physics_weight * rho``.
    """
    base = compute_reward(obs, source=source, dtype=dtype, cfg=cfg,
                          snr_threshold=snr_threshold, response=response, phase=phase)
    if not base.correct:
        # hack / compile_fail / incorrect (incl. shaping) tiers dominate, unchanged.
        return base

    measured = obs.wall_ms
    if measured is None and obs.wall_by_shape:
        # worst (largest) candidate time -> matches the reward's worst-shape discipline.
        measured = max(obs.wall_by_shape.values())
    rho, pmc_used = residual_descent_frac(physics, measured)
    flags = list(base.flags)
    fmt = _format_component(response, cfg)

    if rho is None:
        return RewardResult(cfg.correctness_weight + fmt, True, base.speedup,
                            "correct_no_physics", flags + ["no_physics"],
                            "correct; no roofline/timing for residual credit")
    if not pmc_used:
        flags.append("no_pmc")  # eta fallback -- clearly flagged
    reward = cfg.correctness_weight + physics_weight * max(rho, 0.0) + fmt
    kind = "named-residual" if pmc_used else "eta-fallback"
    return RewardResult(reward, True, base.speedup, "correct_residual", flags,
                        f"residual-descent rho={rho:.3f} ({kind}); "
                        f"credit={physics_weight * rho:.3f}")


# --------------------------------------------------------------------------- #
# Convenience: build reward inputs straight from a p0_sol KernelMeasure (the PMC
# pass already implemented there). Duck-typed (reads attributes) to avoid any
# import cycle with kore.analysis.p0_sol.
# --------------------------------------------------------------------------- #
def observation_from_measure(m, dtype: str = "bf16") -> Observation:
    """Build an :class:`Observation` from a ``KernelMeasure``-like object."""
    return Observation(
        compiled=True,
        snr_db=getattr(m, "snr_db", None),
        wall_ms=getattr(m, "cand_ms", None),
        baseline_ms=getattr(m, "vendor_ms", None),
        validation_passed=bool(getattr(m, "correct", False)),
        occupancy=getattr(m, "occupancy", None),
        dtype=dtype,
    )


def physics_from_measure(m) -> PhysicsSignal:
    """Build a :class:`PhysicsSignal` from a ``KernelMeasure``-like object."""
    return PhysicsSignal(
        t_min_ms=getattr(m, "t_min_ms", float("nan")),
        measured_ms=getattr(m, "cand_ms", None),
        stall_frac=getattr(m, "stall_frac", None),
        occupancy=getattr(m, "occupancy", None),
    )
