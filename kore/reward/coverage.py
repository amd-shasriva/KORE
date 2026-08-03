"""Profiling-based rewards: how fast, and how much of the work.

Speedup alone cannot tell real optimisation from lazy optimisation. Dr. Kernel
(arXiv 2602.05885) motivates their profiling-based reward with a generated kernel
that accounted for **0.014%** of total GPU execution time -- it optimised
something irrelevant -- against **86.15%** for the same task solved with proper
fusion. Both can report a healthy local speedup. Only the second one matters, and
the reward has to be able to tell them apart.

COVERAGE
--------
:func:`kernel_coverage` is the fraction of the profiled region's GPU kernel time
that the candidate's own kernels account for. Both numerator and denominator are
sums of dispatch durations, so the result is a share of GPU BUSY time and is in
[0, 1] by construction even when dispatches overlap on concurrent streams. (Wall
time would need an interval union and would make "coverage" mean something
different depending on how much of the region was idle.)

Coverage 0.0 is a MEASUREMENT, not a gap: it means the candidate's kernels never
ran. That is the decoy-kernel reward hack -- ship a fast kernel that nothing
calls, let the reference path do the work, report the reference's latency as
yours. It must be distinguishable from "no profile available", so this module
returns ``None`` for the latter and never conflates the two.

WHY AMDAHL, NOT A WEIGHTED SUM
------------------------------
Given coverage ``C`` and a local speedup ``S`` on the covered kernels, the
end-to-end speedup is fixed by Amdahl's law:

    S_end_to_end = 1 / ((1 - C) + C / S)

This is the honest combination, and it reproduces the paper's example without any
tuning. At C = 0.00014 even an INFINITE local speedup yields 1.00014x end to end
-- a 0.014% ceiling, which is the whole point of the anecdote. At C = 0.8615 the
ceiling is 7.22x, and a 10x local speedup already delivers 4.45x. A weighted sum
of speedup and coverage would instead hand out most of the reward for a large
local speedup on a negligible slice, which is the behaviour being trained out.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Nothing in this module can promote a kernel to "correct" or relax a timing gate.
It shapes an already-verified reward, and it refuses to produce a number it
cannot ground: every function returns ``None`` when its inputs are missing,
non-finite, or self-contradictory, rather than a plausible default. A reward that
silently falls back to 0.0 on a profiler failure teaches the policy that
profiling failures are cheap.

Pure arithmetic + one CSV reader. No GPU, no torch.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from kore.data.agentic_filter import FilterPolicy

#: The paper's motivating pair, kept as data so the tests can assert against the
#: published numbers rather than against a paraphrase of them.
LAZY_COVERAGE = 0.00014      # 0.014% of total GPU time: optimised the wrong thing
FUSED_COVERAGE = 0.8615      # 86.15%: the same task with real fusion

#: Reward-hacking cap on a claimed speedup. Reuses the value
#: :class:`kore.data.agentic_filter.FilterPolicy` already derives from the
#: observed distribution (2x the measured p99, floored) instead of introducing a
#: third threshold. Real fused wins against a production AITER/hipBLASLt kernel
#: land in the low single digits; the 1541.94x that reached an early mixture from
#: third-party data was a measurement being gamed, not a kernel.
MAX_PLAUSIBLE_SPEEDUP = FilterPolicy.max_speedup

#: Below this share of GPU time, a speedup claim is about a sliver of the
#: workload. Not a hard reject on its own -- some tasks legitimately are a small
#: kernel in a larger trace -- but it is what :mod:`kore.policy.rejection` tests.
#:
#: This is the threshold coverage DESERVES once it is measured over a delimited
#: benched region, and it is deliberately still 0.10. What is not safe is
#: applying it to the coverage we can measure today: that number is the
#: candidate's share of a trace which still contains fixed harness work, so it
#: moves with the candidate's own speed rather than with the workload. On gfx950
#: ten CORRECT seed kernels landed between 0.036 and 0.587 while a deliberately
#: 46x-slowed kernel reached 0.739 -- at 0.10 that rejects correct gen_relu_fp32
#: and softmax_bf16 and keeps the slow one. See
#: docs/evidence/coverage_denominator.md.
#:
#: So the guard lives at the operational default instead:
#: ``GRPOConfig.prs_min_coverage`` is 0.0. Nothing in kore/ calls
#: :func:`kore.policy.rejection.profiling_rejection_sample` yet, and whatever
#: wires it up must pass that config value rather than take this default.
LAZY_COVERAGE_THRESHOLD = 0.10


def _finite(value) -> Optional[float]:
    """A usable float, or None. Rejects bools, NaN, inf and non-numerics."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# --------------------------------------------------------------------------- #
# identifying the candidate's kernels in a trace
# --------------------------------------------------------------------------- #
_TRITON_JIT_RE = re.compile(
    r"@\s*triton\.jit\s*(?:\([^)]*\))?\s*\n\s*def\s+([A-Za-z_]\w*)", re.MULTILINE)


def candidate_kernel_names(source: str) -> set[str]:
    """Names of the ``@triton.jit`` kernels defined in a candidate source.

    These are the symbols the compiler derives dispatch names from, so they are
    what a trace has to be matched against. An empty set means we could not tell
    which dispatches belong to the candidate, which makes coverage unknowable --
    the caller must then produce no reward rather than assume everything (or
    nothing) was the candidate's.
    """
    return set(_TRITON_JIT_RE.findall(source or ""))


def dispatch_matches(kernel_name: str, candidates: Iterable[str]) -> bool:
    """Does a traced dispatch name belong to one of ``candidates``?

    Triton compiles ``def add_kernel`` into a dispatch symbol that STARTS with
    the python name and may carry a specialisation suffix appended directly to it
    (``add_kernel_0d1d2d3e``), so the match is anchored on the left -- start of
    string or a non-identifier character -- and open on the right. Requiring a
    right-hand boundary too would miss every specialised launch and report a
    perfectly good kernel as never having run, which is a worse error than the
    one it would prevent.

    The residual risk is over-attribution: a candidate named ``add`` would also
    match an unrelated ``add_something``. That direction inflates coverage and so
    inflates the reward, which is why the names come from the candidate's OWN
    ``@triton.jit`` definitions -- a vendor kernel (AITER, hipBLASLt, ATen) is not
    named after the model's python function -- and why coverage is only ever a
    bounded shaping term on top of an already-verified correctness verdict.
    """
    name = kernel_name or ""
    for candidate in candidates:
        if not candidate:
            continue
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(candidate)}", name):
            return True
    return False


@dataclass(frozen=True)
class CoverageReport:
    """What the trace said, with enough detail to explain a rejection."""

    coverage: float
    candidate_ns: int
    total_ns: int
    n_candidate_dispatches: int
    n_dispatches: int

    @property
    def never_ran(self) -> bool:
        """No dispatch belonged to the candidate: the decoy-kernel signature."""
        return self.n_candidate_dispatches == 0


def kernel_coverage(
    dispatches: Sequence,
    candidates: Iterable[str],
) -> Optional[CoverageReport]:
    """Share of profiled GPU kernel time attributable to the candidate.

    ``dispatches`` is a sequence of objects with ``kernel_name`` and
    ``duration_ns`` (:class:`kore.verifier.parsers.rocprofv3.KernelDispatch`), or
    of ``(name, duration_ns)`` pairs.

    Returns ``None`` -- not 0.0 -- when coverage cannot be established: no
    dispatches, zero total GPU time, or no candidate kernel names to match
    against. Returns a report with ``coverage == 0.0`` when the trace is good and
    the candidate's kernels genuinely never ran, because that is a finding.
    """
    names = [c for c in candidates if c]
    if not names:
        return None

    total = 0
    mine = 0
    n_total = 0
    n_mine = 0
    for dispatch in dispatches:
        if isinstance(dispatch, (tuple, list)):
            if len(dispatch) != 2:
                continue
            name, duration = dispatch[0], _finite(dispatch[1])
        else:
            name = getattr(dispatch, "kernel_name", None)
            duration = _finite(getattr(dispatch, "duration_ns", None))
        if not name or duration is None or duration < 0:
            continue
        total += int(duration)
        n_total += 1
        if dispatch_matches(name, names):
            mine += int(duration)
            n_mine += 1

    if n_total == 0 or total <= 0:
        return None            # nothing measured; not "zero coverage"
    return CoverageReport(
        coverage=mine / total,
        candidate_ns=mine,
        total_ns=total,
        n_candidate_dispatches=n_mine,
        n_dispatches=n_total,
    )


# --------------------------------------------------------------------------- #
# combining speedup and coverage
# --------------------------------------------------------------------------- #
def amdahl_end_to_end_speedup(local_speedup, coverage) -> Optional[float]:
    """``1 / ((1 - C) + C/S)`` -- the end-to-end speedup a local win buys.

    Returns ``None`` for a non-finite or non-positive speedup, or a coverage
    outside [0, 1]: those are broken measurements, and an "end-to-end speedup"
    computed from one would be fiction.

    ``coverage == 1.0`` with an unbounded speedup would divide by zero, so the
    formula is evaluated as written and only the finite result is returned; an
    infinite local speedup at full coverage has no end-to-end ceiling to report.
    """
    speedup = _finite(local_speedup)
    covered = _finite(coverage)
    if speedup is None or covered is None:
        return None
    if speedup <= 0.0 or not (0.0 <= covered <= 1.0):
        return None
    denominator = (1.0 - covered) + covered / speedup
    if denominator <= 0.0:
        return None
    return 1.0 / denominator


def coverage_ceiling(coverage) -> Optional[float]:
    """The best end-to-end speedup this coverage could EVER buy (S -> inf).

    ``1 / (1 - C)``. At the paper's 0.014% that is 1.00014x, so no amount of
    cleverness on that slice matters; at 86.15% it is 7.22x. Useful as turn
    feedback: it tells the model whether to keep tuning what it has or to go fuse
    more of the graph.
    """
    covered = _finite(coverage)
    if covered is None or not (0.0 <= covered < 1.0):
        return None
    return 1.0 / (1.0 - covered)


def implausible_speedup(speedup, cap: float = MAX_PLAUSIBLE_SPEEDUP) -> bool:
    """Is a claimed speedup outside what this hardware can actually deliver?

    An unmeasurable speedup (``None``) is NOT implausible -- there is no claim to
    reject -- which keeps this from becoming a way to launder missing data into a
    rejection. Matches
    :func:`kore.data.agentic_filter.classify`'s ``implausible_speedup`` rule so the
    RL-time guard and the data-time guard agree on the same number.
    """
    value = _finite(speedup)
    if value is None:
        return False
    return value > cap


@dataclass(frozen=True)
class ProfilingReward:
    """A profiling-based reward, with the terms that produced it."""

    reward: float                     # bounded [0, 1] shaping term
    end_to_end_speedup: float
    local_speedup: float
    coverage: float
    ceiling: Optional[float]

    def as_dict(self) -> dict:
        return {
            "profiling_reward": round(self.reward, 6),
            "end_to_end_speedup": round(self.end_to_end_speedup, 6),
            "local_speedup": round(self.local_speedup, 6),
            "coverage": round(self.coverage, 6),
            "coverage_ceiling": (round(self.ceiling, 6)
                                 if self.ceiling is not None else None),
        }


def profiling_reward(
    local_speedup,
    coverage,
    *,
    correct: bool,
    reward_cap: float = 4.0,
    max_plausible_speedup: float = MAX_PLAUSIBLE_SPEEDUP,
) -> Optional[ProfilingReward]:
    """Bounded profiling reward from a MEASURED speedup and a MEASURED coverage.

    The reward is ``log(S_end_to_end) / log(reward_cap)`` clamped to [0, 1]: log
    scale because the difference between 1x and 2x is worth far more than between
    5x and 6x, and clamped so one lucky measurement cannot dominate a group's
    advantage. ``reward_cap`` is the end-to-end speedup that earns full marks.

    Returns ``None`` -- no reward at all -- when:

      * the kernel is not correct. Speed is meaningless without correctness, and
        this module must never be a way to earn credit for a wrong answer;
      * either measurement is missing or non-finite. The alternative, a 0.0
        reward, is a claim that the kernel was measured and found worthless, and
        it teaches the policy that a profiler failure is indistinguishable from
        a bad kernel;
      * the claimed speedup exceeds ``max_plausible_speedup``. On these tasks the
        baseline is a production vendor kernel, so a 1541x claim is a measurement
        exploit; paying it would train exactly the exploit.

    A verified-correct kernel whose own dispatches never appear in the trace
    (coverage 0.0) yields reward 0.0, not ``None``: that is measured, and it is
    the decoy-kernel hack.
    """
    if not correct:
        return None
    speedup = _finite(local_speedup)
    covered = _finite(coverage)
    if speedup is None or covered is None:
        return None
    if speedup <= 0.0 or not (0.0 <= covered <= 1.0):
        return None
    if implausible_speedup(speedup, max_plausible_speedup):
        return None
    end_to_end = amdahl_end_to_end_speedup(speedup, covered)
    if end_to_end is None:
        return None
    cap = _finite(reward_cap)
    if cap is None or cap <= 1.0:
        return None
    scaled = math.log(max(end_to_end, 1.0)) / math.log(cap)
    return ProfilingReward(
        reward=max(0.0, min(1.0, scaled)),
        end_to_end_speedup=end_to_end,
        local_speedup=speedup,
        coverage=covered,
        ceiling=coverage_ceiling(covered),
    )


def coverage_feedback(report: CoverageReport, local_speedup=None) -> str:
    """Turn feedback that names the ceiling, so the model can act on coverage.

    Telling a model "your kernel is 8x faster" while it covers 3% of the runtime
    is the feedback that produces lazy optimisation. Telling it the end-to-end
    ceiling is 1.03x points at fusion instead.
    """
    if report.never_ran:
        return ("PROFILE: none of your kernels appear in the trace - the "
                f"{report.n_dispatches} dispatches that ran were all somebody "
                "else's. A kernel that is never called cannot be faster; check "
                "that the entry point actually dispatches it.")
    ceiling = coverage_ceiling(report.coverage)
    parts = [
        f"PROFILE: your kernels are {report.coverage * 100:.2f}% of GPU time "
        f"({report.n_candidate_dispatches}/{report.n_dispatches} dispatches)"
    ]
    if ceiling is not None:
        parts.append(
            f"so the best end-to-end speedup this can ever reach is "
            f"{ceiling:.2f}x, no matter how fast the kernel itself gets")
    end_to_end = amdahl_end_to_end_speedup(local_speedup, report.coverage)
    if end_to_end is not None:
        parts.append(f"at the measured {float(local_speedup):.2f}x local speedup "
                     f"that is {end_to_end:.2f}x end to end")
    if report.coverage < LAZY_COVERAGE_THRESHOLD:
        parts.append("fuse more of the computation into your kernel rather than "
                     "tuning this slice further")
    return ". ".join(parts) + "."


__all__ = [
    "CoverageReport",
    "FUSED_COVERAGE",
    "LAZY_COVERAGE",
    "LAZY_COVERAGE_THRESHOLD",
    "MAX_PLAUSIBLE_SPEEDUP",
    "ProfilingReward",
    "amdahl_end_to_end_speedup",
    "candidate_kernel_names",
    "coverage_ceiling",
    "coverage_feedback",
    "dispatch_matches",
    "implausible_speedup",
    "kernel_coverage",
    "profiling_reward",
]
