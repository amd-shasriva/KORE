"""Rollout-time rejection sampling: MRS across turns, PRS on profiling signals.

Dr. Kernel (arXiv 2602.05885) pairs TRLOO with two filters. Multi-turn Rejection
Sampling (MRS) drops low-quality trajectories across turns; Profiling-based
Rejection Sampling (PRS) rejects candidates on profiling signals "to force
meaningful fusion". Both run INSIDE the RL loop on freshly rolled trajectories,
which is what distinguishes them from :mod:`kore.data.rejection` -- that module
selects finished ``WinRecord``s for offline RFT/SFT, and neither should be
confused for the other.

THE ONE THING THAT MAKES THIS DANGEROUS
---------------------------------------
Every advantage estimator here is GROUP-RELATIVE. It learns from the CONTRAST
between the trajectories in a group, so a filter that removes the failures also
removes the signal: a group where every survivor scored 0.8 teaches nothing, no
matter how good 0.8 is. StarPO-S already drops collapsed groups for this reason,
and dynamic sampling refills them.

So filtering is applied under a hard constraint: **if a filter would collapse a
group's reward variance, or leave too few trajectories to form a leave-one-out
baseline, it is not applied at all** and the group passes through untouched with
the reason recorded. A filter that improves average trajectory quality while
destroying the learning signal is a net loss, and it would show up as slow
progress rather than as an error.

WHY GEOMETRIC AGGREGATION ACROSS TURNS
--------------------------------------
Dr. Kernel report ``geometric`` as their rejection-sampling default, and a
geometric mean is the right aggregator for a multi-turn quality gate: it is
dominated by the WEAKEST turn and collapses to zero if any turn is zero, so one
strong turn cannot carry a trajectory that was broken for the rest of the
episode. An arithmetic mean lets exactly that happen. Both are available; the
default is geometric.

To be precise about provenance: the mechanism (aggregate per-turn quality, reject
low-quality trajectories) and the reported default name are from the paper; the
specific per-turn quality terms below are this repo's, chosen to match the
signals :class:`kore.agent.harness.AgentEpisode` actually records.

Pure arithmetic on per-turn traces. No torch, no GPU, no config object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from kore.reward.coverage import (
    LAZY_COVERAGE_THRESHOLD,
    MAX_PLAUSIBLE_SPEEDUP,
    implausible_speedup,
)

GEOMETRIC = "geometric"
ARITHMETIC = "arithmetic"
MINIMUM = "min"
AGGREGATES = (GEOMETRIC, ARITHMETIC, MINIMUM)

#: A leave-one-out baseline needs at least two trajectories at a turn, so a group
#: filtered below this cannot produce a TRLOO advantage for anything.
MIN_TRAJECTORIES = 2

#: Reward spread below this is a collapsed group: no contrast, nothing to learn.
MIN_REWARD_STD = 1e-3


def _finite(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


# --------------------------------------------------------------------------- #
# MRS: trajectory quality across turns
# --------------------------------------------------------------------------- #
def turn_quality(reward, correct: bool, *, reward_scale: float = 1.0) -> float:
    """Per-turn quality in [0, 1]: 0 for an incorrect turn, scaled reward otherwise.

    Deliberately hard-zero on an incorrect turn, because that is what makes the
    geometric aggregate collapse for a trajectory that was broken at any point.
    Note this is a REJECTION signal, not a training reward -- the training reward
    keeps its shaped credit for incorrect turns (``credit_incorrect_turns``), and
    conflating the two would undo that densification.
    """
    if not correct:
        return 0.0
    value = _finite(reward)
    if value is None or value <= 0.0:
        return 0.0
    scale = _finite(reward_scale) or 1.0
    if scale <= 0.0:
        scale = 1.0
    return max(0.0, min(1.0, value / scale))


def aggregate_quality(qualities: Sequence[float],
                      how: str = GEOMETRIC) -> Optional[float]:
    """Aggregate per-turn qualities into one trajectory score.

    ``geometric`` (default) is dominated by the weakest turn and is zero if any
    turn is zero. ``arithmetic`` averages. ``min`` is the strictest. Returns
    ``None`` for an empty trace -- a trajectory with no turns has no quality,
    which is different from having quality zero.
    """
    if how not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {list(AGGREGATES)}, got {how!r}")
    values = [max(0.0, min(1.0, float(q))) for q in qualities]
    if not values:
        return None
    if how == MINIMUM:
        return min(values)
    if how == ARITHMETIC:
        return sum(values) / len(values)
    if any(v <= 0.0 for v in values):
        return 0.0            # geometric mean with a zero factor
    return math.exp(sum(math.log(v) for v in values) / len(values))


@dataclass
class TrajectoryVerdict:
    """Why one trajectory was kept or rejected."""

    index: int
    quality: Optional[float]
    keep: bool
    reason: str = ""
    best_speedup: Optional[float] = None
    improved: bool = False


def trajectory_verdict(
    index: int,
    turn_rewards: Sequence,
    turn_correct: Sequence,
    *,
    turn_speedups: Optional[Sequence] = None,
    min_quality: float = 0.0,
    aggregate: str = GEOMETRIC,
    reward_scale: float = 1.0,
    require_improvement: bool = False,
    max_plausible_speedup: float = MAX_PLAUSIBLE_SPEEDUP,
) -> TrajectoryVerdict:
    """Judge one rolled trajectory.

    Rejection reasons, in the order they are tested:

    ``no_turns``
        nothing was recorded; there is nothing to train on either way.
    ``implausible_speedup``
        some turn claimed a speedup the hardware cannot deliver. This is a
        REWARD-HACKING rejection and it is checked first, because a trajectory
        built on a gamed measurement must not be able to pass on the strength of
        its other turns.
    ``never_correct``
        no turn produced a correct kernel.
    ``no_improvement``
        correct, but the measured speedup never moved -- the lazy-optimisation
        trajectory. Only applied when ``require_improvement`` is set, because a
        trajectory that arrives correct-and-fast on turn 1 has nothing left to
        improve and is still worth training on.
    ``low_quality``
        the aggregate is at or below ``min_quality``.
    """
    rewards = list(turn_rewards or [])
    corrects = list(turn_correct or [])
    n = min(len(rewards), len(corrects))
    speedups = list(turn_speedups or [])

    if n == 0:
        return TrajectoryVerdict(index, None, False, "no_turns")

    for su in speedups[:n]:
        if implausible_speedup(su, max_plausible_speedup):
            return TrajectoryVerdict(
                index, 0.0, False, "implausible_speedup",
                best_speedup=_finite(su))

    measured = [s for s in (_finite(x) for x in speedups[:n]) if s is not None]
    best_speedup = max(measured) if measured else None
    first_speedup = measured[0] if measured else None
    improved = bool(
        best_speedup is not None and first_speedup is not None
        and best_speedup > first_speedup)

    qualities = [turn_quality(rewards[t], bool(corrects[t]),
                              reward_scale=reward_scale) for t in range(n)]
    quality = aggregate_quality(qualities, aggregate)

    if not any(corrects[:n]):
        return TrajectoryVerdict(index, quality, False, "never_correct",
                                 best_speedup=best_speedup, improved=improved)
    if require_improvement and not improved and len(measured) > 1:
        return TrajectoryVerdict(index, quality, False, "no_improvement",
                                 best_speedup=best_speedup, improved=improved)
    if quality is not None and quality <= min_quality:
        return TrajectoryVerdict(index, quality, False, "low_quality",
                                 best_speedup=best_speedup, improved=improved)
    return TrajectoryVerdict(index, quality, True, "",
                             best_speedup=best_speedup, improved=improved)


@dataclass
class GroupFilterReport:
    """The group-level outcome, including a filter that declined to fire."""

    verdicts: list = field(default_factory=list)
    applied: bool = True
    skipped_reason: str = ""

    @property
    def keep_indices(self) -> list[int]:
        return [v.index for v in self.verdicts if v.keep]

    @property
    def rejected(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for v in self.verdicts:
            if not v.keep and v.reason:
                counts[v.reason] = counts.get(v.reason, 0) + 1
        return counts

    def as_dict(self) -> dict:
        return {
            "kept": len(self.keep_indices),
            "total": len(self.verdicts),
            "applied": self.applied,
            "skipped_reason": self.skipped_reason,
            "rejected": self.rejected,
        }


def multi_turn_rejection_sample(
    traj_rewards: Sequence[Sequence],
    traj_correct: Sequence[Sequence],
    *,
    traj_speedups: Optional[Sequence[Sequence]] = None,
    min_quality: float = 0.0,
    aggregate: str = GEOMETRIC,
    reward_scale: float = 1.0,
    require_improvement: bool = False,
    min_trajectories: int = MIN_TRAJECTORIES,
    min_reward_std: float = MIN_REWARD_STD,
    max_plausible_speedup: float = MAX_PLAUSIBLE_SPEEDUP,
) -> GroupFilterReport:
    """MRS over one rollout group, refusing to destroy the group's contrast.

    Judges each trajectory with :func:`trajectory_verdict`, then applies the
    survivors ONLY if the filtered group can still teach something:

      * at least ``min_trajectories`` survive (a leave-one-out baseline needs a
        peer, so a single survivor produces no comparison at all), and
      * the survivors' trajectory scores still have spread above
        ``min_reward_std``.

    If either check fails the filter DECLINES: every trajectory is kept and
    ``skipped_reason`` records why. That is deliberate. Removing the failures from
    a group raises its average quality and lowers its variance, and variance is
    the entire signal a group-relative estimator has -- so a filter that always
    fires would quietly convert a learning step into a no-op with a healthy-looking
    mean reward.

    One rejection is never waived: ``implausible_speedup``. A gamed measurement is
    not low-quality data to be traded off against group variance, it is a
    measurement that must not enter training at all, so those trajectories stay
    rejected even when the filter otherwise declines.
    """
    n_traj = min(len(traj_rewards), len(traj_correct))
    speedups = list(traj_speedups or [])
    verdicts = [
        trajectory_verdict(
            i, traj_rewards[i], traj_correct[i],
            turn_speedups=speedups[i] if i < len(speedups) else None,
            min_quality=min_quality, aggregate=aggregate,
            reward_scale=reward_scale, require_improvement=require_improvement,
            max_plausible_speedup=max_plausible_speedup)
        for i in range(n_traj)
    ]

    def _trajectory_score(v: TrajectoryVerdict) -> float:
        return v.quality if v.quality is not None else 0.0

    survivors = [v for v in verdicts if v.keep]
    if len(survivors) < max(1, int(min_trajectories)):
        return _decline(verdicts, "too_few_survivors")
    if _std([_trajectory_score(v) for v in survivors]) <= min_reward_std:
        return _decline(verdicts, "would_collapse_variance")
    return GroupFilterReport(verdicts=verdicts, applied=True)


def _decline(verdicts: list, reason: str) -> GroupFilterReport:
    """Keep everything except gamed measurements, and say why."""
    restored = []
    for v in verdicts:
        if v.reason == "implausible_speedup":
            restored.append(v)          # never waived
            continue
        restored.append(TrajectoryVerdict(
            v.index, v.quality, True, "", best_speedup=v.best_speedup,
            improved=v.improved))
    return GroupFilterReport(verdicts=restored, applied=False,
                             skipped_reason=reason)


# --------------------------------------------------------------------------- #
# PRS: profiling-based candidate rejection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CandidateVerdict:
    """Whether one candidate kernel may be treated as a win."""

    keep: bool
    reason: str = ""
    coverage: Optional[float] = None
    speedup: Optional[float] = None


def profiling_rejection_sample(
    *,
    correct: bool,
    speedup=None,
    coverage=None,
    min_coverage: float = LAZY_COVERAGE_THRESHOLD,
    max_plausible_speedup: float = MAX_PLAUSIBLE_SPEEDUP,
    require_profile: bool = False,
) -> CandidateVerdict:
    """PRS: reject a candidate on its profiling signals, to force real fusion.

    Reasons, in order:

    ``incorrect``
        correctness is not negotiable and is tested first.
    ``implausible_speedup``
        above the cap; a measurement exploit rather than a kernel.
    ``kernel_never_ran``
        coverage is exactly 0.0 on a good trace, i.e. the candidate's kernels
        never dispatched. The decoy-kernel hack, and the most important thing PRS
        catches: correctness can be satisfied by the reference path while the
        "optimised" kernel sits unused.
    ``lazy_optimisation``
        coverage is positive but below ``min_coverage``. The candidate tuned a
        sliver of the runtime, so its speedup cannot translate into an end-to-end
        win (see :func:`kore.reward.coverage.coverage_ceiling`).
    ``no_profile``
        coverage is unavailable. Only a rejection when ``require_profile`` is set.
        Default off, and that default is a deliberate integrity choice: the
        profiler is not available on every task or every node, and rejecting every
        unprofiled candidate would silently narrow training to the profilable
        subset while looking like a quality filter. A missing profile means PRS
        has no opinion, not that the candidate is bad.
    """
    if not correct:
        return CandidateVerdict(False, "incorrect", coverage=_finite(coverage),
                                speedup=_finite(speedup))
    if implausible_speedup(speedup, max_plausible_speedup):
        return CandidateVerdict(False, "implausible_speedup",
                                coverage=_finite(coverage),
                                speedup=_finite(speedup))
    covered = _finite(coverage)
    if covered is None:
        if require_profile:
            return CandidateVerdict(False, "no_profile",
                                    speedup=_finite(speedup))
        return CandidateVerdict(True, "", speedup=_finite(speedup))
    if not (0.0 <= covered <= 1.0):
        # A coverage outside [0, 1] is a broken trace, not a bad kernel.
        if require_profile:
            return CandidateVerdict(False, "no_profile", coverage=covered,
                                    speedup=_finite(speedup))
        return CandidateVerdict(True, "", coverage=covered,
                                speedup=_finite(speedup))
    if covered == 0.0:
        return CandidateVerdict(False, "kernel_never_ran", coverage=covered,
                                speedup=_finite(speedup))
    if covered < min_coverage:
        return CandidateVerdict(False, "lazy_optimisation", coverage=covered,
                                speedup=_finite(speedup))
    return CandidateVerdict(True, "", coverage=covered, speedup=_finite(speedup))


__all__ = [
    "AGGREGATES",
    "ARITHMETIC",
    "CandidateVerdict",
    "GEOMETRIC",
    "GroupFilterReport",
    "MINIMUM",
    "MIN_REWARD_STD",
    "MIN_TRAJECTORIES",
    "TrajectoryVerdict",
    "aggregate_quality",
    "multi_turn_rejection_sample",
    "profiling_rejection_sample",
    "trajectory_verdict",
    "turn_quality",
]
