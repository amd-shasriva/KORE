"""Measurement allocation for AlphaKernel: LCB statistics + Successive Halving.

The verifier is a *perfect but expensive* simulator: a single ``KoreEnv`` bench
returns one noisy speedup sample, and the timing variance (CV) can be large. The
search must decide *how many* measurements to spend on each candidate. Spending
the same budget on every candidate is wasteful - most candidates are obviously
worse and one or two measurements already reveal it, while the few real
contenders need several measurements to *tighten the lower confidence bound*
(LCB) that AlphaKernel selects and reports on.

This module provides the two primitives that realize that policy:

* :class:`Budget` - a hard global cap on verifier calls (compile/correctness +
  bench). Every ``env.step`` in the search is gated through it, so the whole
  search is anytime and never exceeds the caller's budget.
* :class:`MeasureStats` - streaming mean / variance / **pessimistic LCB** for a
  candidate's speedup samples. AlphaKernel *ranks and commits by the LCB*, not
  the mean, so a fast-but-noisy kernel never beats a slightly-slower-but-stable
  one it cannot reliably reproduce.
* :func:`successive_halving` - a Hyperband/SHA rung schedule: give every arm a
  cheap first look (``min_measures``), then repeatedly keep the top ``1/eta`` by
  LCB and re-invest the saved budget into the survivors to tighten *their* LCBs.
  It also honors the admissible roofline bound (an arm whose perf ceiling cannot
  beat the incumbent is dropped, never measured further).

Everything here is PURE (stdlib only) and deterministic given the arms' sample
streams, so it is unit-testable with scripted fake arms and carries no GPU / torch
dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

# One-sigma lower-confidence-bound multiplier. Deliberately conservative-but-cheap
# (~84% one-sided): pessimistic enough that a high-variance arm is visibly penalized
# with only a handful of samples, without demanding the many measurements a 95%
# bound (1.96) would need per candidate. Override via AlphaKernelConfig.lcb_z.
DEFAULT_LCB_Z: float = 1.0


class BudgetExhausted(Exception):
    """Raised by helpers that must not silently proceed once the cap is hit."""


@dataclass
class Budget:
    """A hard cap on the number of verifier (``env.step``) calls the search may make.

    ``spend`` is the single choke point every measurement/correctness call flows
    through, so the search is provably anytime: it can stop at any point and the
    total verifier calls never exceed ``total``.
    """

    total: int
    used: int = 0

    def __post_init__(self) -> None:
        self.total = max(0, int(self.total))

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)

    def can_afford(self, k: int = 1) -> bool:
        return self.used + k <= self.total

    def spend(self, k: int = 1) -> bool:
        """Reserve ``k`` calls; return True iff they fit under the cap.

        Never overspends: on a False return nothing is consumed, so the caller can
        cleanly abort the current step.
        """
        if self.used + k <= self.total:
            self.used += k
            return True
        return False


@dataclass
class MeasureStats:
    """Streaming speedup statistics with a pessimistic lower confidence bound.

    The LCB ``mean - z * std / sqrt(n)`` is the value AlphaKernel selects and
    commits on. With ``n < 2`` the sample std is undefined, so the LCB collapses to
    the mean (a single sample carries no variance evidence yet); as measurements
    accumulate the interval tightens toward the mean. A perfectly stable arm
    (std == 0) therefore has ``lcb == mean`` at any ``n``, while a noisy arm with
    the same mean is pushed strictly below it - exactly the ordering we want.
    """

    samples: list[float] = field(default_factory=list)
    z: float = DEFAULT_LCB_Z

    def add(self, x: float) -> None:
        self.samples.append(float(x))

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def var(self) -> float:
        # sample variance (ddof=1); 0 for n < 2 (no variance evidence yet).
        n = len(self.samples)
        if n < 2:
            return 0.0
        m = self.mean
        return sum((x - m) ** 2 for x in self.samples) / (n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    @property
    def sem(self) -> float:
        """Standard error of the mean; 0 when there is no variance evidence."""
        n = len(self.samples)
        return self.std / math.sqrt(n) if n >= 2 else 0.0

    @property
    def lcb(self) -> float:
        """Pessimistic speedup estimate = mean - z * SEM."""
        if not self.samples:
            return 0.0
        return self.mean - self.z * self.sem

    @property
    def ucb(self) -> float:
        """Optimistic speedup estimate = mean + z * SEM (diagnostic only)."""
        if not self.samples:
            return 0.0
        return self.mean + self.z * self.sem


@runtime_checkable
class Arm(Protocol):
    """One candidate under measurement (a correct kernel Node, in AlphaKernel).

    ``measure()`` draws exactly one fresh speedup sample and folds it into the
    arm's stats, returning the sample - or ``None`` if no sample could be produced
    (so the allocator stops on this arm). The global :class:`Budget` is reserved by
    the allocator *before* each ``measure()`` (one ``env.step`` per unit), so a
    measurement callback must NOT spend the budget itself. ``ceiling`` is the
    admissible roofline upper bound on the arm's achievable speedup (``+inf`` if
    unmodeled).
    """

    def measure(self) -> Optional[float]: ...
    @property
    def n(self) -> int: ...
    @property
    def mean(self) -> float: ...
    @property
    def lcb(self) -> float: ...
    @property
    def ceiling(self) -> float: ...


@dataclass
class CallbackArm:
    """Concrete :class:`Arm` wrapping a sampling callback + a :class:`MeasureStats`.

    Used both by AlphaKernel (the callback runs one ``env.step`` bench under the
    global budget) and by the unit tests (the callback pops a scripted sample). A
    ``None`` from the callback means "no budget / no sample" and stops allocation
    on this arm.
    """

    key: object
    sampler: Callable[[], Optional[float]]
    stats: MeasureStats = field(default_factory=MeasureStats)
    ceiling: float = float("inf")

    def measure(self) -> Optional[float]:
        x = self.sampler()
        if x is None:
            return None
        self.stats.add(x)
        return x

    @property
    def n(self) -> int:
        return self.stats.n

    @property
    def mean(self) -> float:
        return self.stats.mean

    @property
    def lcb(self) -> float:
        return self.stats.lcb


def _rank_value(arm: Arm, key: str) -> float:
    return arm.mean if key == "mean" else arm.lcb


def successive_halving(
    arms: list[Arm],
    budget: Budget,
    *,
    eta: int = 2,
    min_measures: int = 1,
    max_measures: int = 8,
    rank_key: str = "lcb",
    incumbent_lcb: float = float("-inf"),
) -> list[Arm]:
    """Allocate measurements across ``arms`` with a Successive-Halving rung schedule.

    Rung 0 gives every (admissible) arm ``min_measures`` cheap looks; each rung
    keeps the top ``ceil(k / eta)`` arms by ``rank_key`` (default the pessimistic
    LCB) and raises their measurement target by ``eta``, re-investing the budget
    the eliminated arms would have consumed into tightening the survivors' LCBs.
    Allocation stops as soon as the :class:`Budget` is exhausted, one arm remains,
    or the survivors reach ``max_measures``.

    Admissible branch-and-bound: an arm whose roofline ``ceiling`` cannot exceed
    ``incumbent_lcb`` is provably dominated and is dropped without spending another
    measurement on it (it still ranks, below the live arms).

    The allocator is the single budget authority for measurements: it reserves one
    :class:`Budget` unit *before* each ``measure()`` call, so the whole rung
    schedule is capped by ``budget`` and stops the instant it is exhausted.

    Returns ALL input arms ranked best-first by ``rank_key`` (survivors, with more
    measurements and tighter LCBs, naturally sort ahead of eliminated arms).
    """
    if eta < 2:
        eta = 2
    live = [a for a in arms if a.ceiling > incumbent_lcb]  # dominated arms dropped

    target = max(1, int(min_measures))
    while live and budget.can_afford(1):
        # Bring every survivor up to this rung's measurement target.
        for arm in live:
            while arm.n < target and budget.can_afford(1):
                if not budget.spend(1):       # reserve one env.step before measuring
                    break
                if arm.measure() is None:      # arm produced no sample -> stop it
                    break
        if len(live) <= 1 or target >= max_measures or not budget.can_afford(1):
            break
        # Keep the top 1/eta by pessimistic value; the rest are eliminated.
        live.sort(key=lambda a: _rank_value(a, rank_key), reverse=True)
        keep = max(1, math.ceil(len(live) / eta))
        live = live[:keep]
        target = min(max_measures, target * eta)

    ranked = list(arms)
    ranked.sort(key=lambda a: _rank_value(a, rank_key), reverse=True)
    return ranked
