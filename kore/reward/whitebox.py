"""White-box, counter-grounded physics reward (P0 flagship).

The live campaign's residual reward degrades to the PMC-free ``eta = T_min/T_meas``
because :func:`kore.reward.physics.physics_signal_from_obs` never populates
``stall_frac``/``occupancy`` -- so the *named* residual ``rho_phys`` (the signal
with the R^2~=0.98 backing in ``docs/P0_RESULTS.md``) is dormant online, and the
live gradient is the low-contrast ``0.3 + eta``.

This module closes that gap WITHOUT touching the datagen pipeline or the existing
reward ABI:

  * :func:`physics_signal_from_counters` -- build a :class:`PhysicsSignal` with the
    named-residual terms populated from rocprofv3 counters, so
    :func:`kore.reward.physics.residual_descent_frac` uses ``rho = T_min/(T_min+N)``
    instead of the flat ``eta`` fallback.
  * :func:`whitebox_attainment` -- the ``(rho, pmc_used)`` credit for a kernel.
  * :func:`whitebox_structural_score` -- a hack-RESISTANT [0,1] score derived from
    *what the silicon actually did* (issue efficiency + roofline attainment +
    baseline-relative traffic). A memset / cache-reuse / "do-less" kernel attains
    ~0 useful work and therefore scores ~0 by construction -- this is the
    reward-hacking-immune surface the field (Kevin/CUDA-L1/Sakana) never had.
  * :func:`phi_potential` -- the scalar potential ``Phi(s) = rho`` used by the
    potential-based cross-turn shaping in :mod:`kore.reward.shaping` (Ng et al.
    policy-invariance), so the dense signal can be added with a *theorem* that it
    cannot change the optimal policy or introduce a hacking incentive.

Everything here is PURE and CPU-testable given a counter dict; the single GPU touch
(``KoreEnv.collect_counters``) is performed by the caller and injected as ``counters``.
Robust to the two PMC vocabularies in-tree: derived metrics (``OccupancyPercent`` /
``MemUnitStalled``, the ``p0_sol`` set) are preferred, with a raw-counter
(``SQ_*`` via :mod:`kore.reward.profile_reward`) fallback.
"""

from __future__ import annotations

from typing import Optional

from kore.reward import profile_reward as _pr
from kore.reward.physics import (
    PhysicsSignal,
    physics_signal_from_obs,
    residual_descent_frac,
)


def _clamp01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _lookup(counters: dict, *names: str) -> Optional[float]:
    """Case-insensitive exact lookup of the first present counter name."""
    if not counters:
        return None
    up = {str(k).upper(): v for k, v in counters.items()}
    for n in names:
        v = up.get(n.upper())
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def stall_frac_from_counters(counters: dict) -> Optional[float]:
    """Normalized memory-stall fraction in ``[0, 1]`` (lower is better).

    Prefers the gfx950 DERIVED metric ``MemUnitStalled`` (a percentage, as used by
    the offline ``p0_sol`` decomposition that carries the R^2~=0.98 evidence), and
    falls back to the RAW-counter estimate ``SQ_WAIT_INST_ANY / (issued + wait)``
    from :func:`kore.reward.profile_reward.stall_fraction`. None when neither is
    computable.
    """
    derived = _lookup(counters, "MemUnitStalled", "MemUnitStalled_pct", "MEM_UNIT_STALLED")
    if derived is not None:
        # derived metrics are percentages in [0, 100]
        return _clamp01(derived / 100.0)
    return _clamp01(_pr.stall_fraction(counters or {}))


def occupancy_from_counters(counters: dict) -> Optional[float]:
    """Normalized achieved occupancy in ``[0, 1]`` (higher is better).

    Prefers the DERIVED ``OccupancyPercent``; else estimates resource-limited
    occupancy from the captured VGPR/LDS/wavefront resource fields via
    :func:`kore.verifier.pmc.est_occupancy` (fully fail-safe -> None on any gap).
    """
    derived = _lookup(counters, "OccupancyPercent", "Occupancy", "OCCUPANCY_PERCENT")
    if derived is not None:
        return _clamp01(derived / 100.0)
    vgpr = _lookup(counters, "vgpr_count", "VGPR", "vgpr")
    lds = _lookup(counters, "lds_bytes", "LDS", "lds")
    warps = _lookup(counters, "num_warps", "WARPS", "num_warps_per_wg")
    if vgpr is None:
        return None
    try:  # pmc.est_occupancy is the resource-limited waves/SIMD model (CDNA4 defaults)
        from kore.verifier.pmc import est_occupancy

        occ = est_occupancy(int(vgpr), int(lds or 0), int(warps or 4))
        if occ is None:
            return None
        # est_occupancy returns a fraction in [0,1] (waves/SIMD over the max).
        return _clamp01(float(occ))
    except Exception:  # noqa: BLE001 - resource model unavailable -> no occupancy term
        return None


def physics_signal_from_counters(task, obs, counters: Optional[dict],
                                 arch: Optional[str] = None) -> Optional[PhysicsSignal]:
    """A worst-shape :class:`PhysicsSignal` with the NAMED-residual terms populated.

    Reuses :func:`kore.reward.physics.physics_signal_from_obs` for the roofline
    ``T_min`` + worst-shape measured wall (its worst-shape discipline is preserved),
    then attaches ``stall_frac``/``occupancy`` from ``counters`` so
    :func:`residual_descent_frac` takes the ``rho = T_min/(T_min+N)`` path. When no
    counters are available it returns the eta-based signal unchanged (graceful
    degrade, identical to today). Returns None iff the op is not roofline-modelable.
    """
    base = physics_signal_from_obs(task, obs, arch)
    if base is None:
        return None
    if not counters:
        return base
    sf = stall_frac_from_counters(counters)
    occ = occupancy_from_counters(counters)
    if sf is None and occ is None:
        return base  # counters present but unusable -> eta fallback (flagged upstream)
    return PhysicsSignal(t_min_ms=base.t_min_ms, measured_ms=base.measured_ms,
                         stall_frac=sf, occupancy=occ)


def whitebox_attainment(task, obs, counters: Optional[dict] = None,
                        arch: Optional[str] = None) -> tuple[Optional[float], bool]:
    """``(rho, pmc_used)`` roofline-attainment credit in ``(0, 1]`` for a kernel.

    ``pmc_used=True`` means the NAMED residual (stall+occupancy) was used; ``False``
    is the eta fallback. ``(None, False)`` when the op is not modelable / untimed.
    """
    sig = physics_signal_from_counters(task, obs, counters, arch)
    if sig is None:
        return None, False
    return residual_descent_frac(sig)


def whitebox_structural_score(counters: dict, *, flops: Optional[float] = None,
                              bytes: Optional[float] = None,
                              measured_ms: Optional[float] = None,
                              dtype: str = "bf16",
                              ref: Optional[dict] = None) -> Optional[float]:
    """Hack-RESISTANT structural performance score in ``[0, 1]``.

    Delegates to :func:`kore.reward.profile_reward.roofline_dense_score`, which
    blends (roofline attainment, issue efficiency, baseline-relative traffic) --
    all bounded and RELATIVE, and all derived from what the hardware executed. A
    kernel that cheats the wall-clock (memset the output, reuse a cached result, do
    less work) issues ~no useful instructions and attains ~0% of the roofline, so
    it scores ~0 here *by construction* -- unlike a wall-clock reward, this surface
    cannot be gamed by measurement tricks. Returns None when no counter is usable.
    """
    return _pr.roofline_dense_score(counters or {}, ref, flops=flops, bytes=bytes,
                                    measured_ms=measured_ms, dtype=dtype)


def phi_potential(task, obs, counters: Optional[dict] = None,
                  arch: Optional[str] = None) -> Optional[float]:
    """The scalar potential ``Phi(s) = rho`` for potential-based cross-turn shaping.

    Using the named-residual attainment as the potential means the shaping reward
    ``F = gamma*Phi(s') - Phi(s)`` (see :mod:`kore.reward.shaping`) densifies the
    gradient in the flat correct-but-slow valley while, by the Ng-Harada-Russell
    theorem, leaving the optimal policy unchanged -- so it is safe to add at any
    weight and cannot be reward-hacked. Returns None when no potential is defined
    (op not modelable / kernel not correct-and-timed), which the shaper treats as a
    zero-contribution boundary.
    """
    rho, _pmc = whitebox_attainment(task, obs, counters, arch)
    return rho
