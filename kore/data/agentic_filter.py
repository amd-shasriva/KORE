"""Keep the agentic trajectories that teach improvement, drop the rest.

Every trajectory that ran is not training data. Kernel-Smith's framing is that the
model is being trained as *a strong local improver inside an optimization loop*,
not as a one-shot generator, so the supervision that matters is a revision which
kept the kernel correct and made it materially faster. A trajectory that reached
a correct kernel and then spent seven turns moving the speedup from 0.41x to
0.42x demonstrates the behaviour we are trying to train out.

Three things this measures, in order:

``gain``
    best correct speedup divided by the FIRST correct speedup. This is
    self-improvement inside the episode, and it is the primary signal: the
    baselines here are production AITER/hipBLASLt kernels, so demanding every
    kept trajectory beat the vendor would throw away almost everything while
    teaching nothing extra. An episode that climbs 0.35x -> 0.80x has shown the
    model exactly the skill being trained.

``vendor parity``
    a trajectory that actually reached >= 1.0x against the production baseline is
    kept regardless of gain, because arriving correct-and-competitive on the first
    try is also worth imitating.

``high-gain revisions``
    the individual turns where a correct candidate pushed the measured frontier.
    Counted per trajectory and reported, because it is the unit Kernel-Smith
    trains on and a trajectory with none of them is a trajectory with nothing to
    learn from even if its endpoint looks good.

Reward hacking is filtered separately and deliberately conservatively. KORE's
oracle is a much harder bar than a torch-eager reference - four correctness prongs
with an SNR gate mean a decoy kernel that is never called fails outright - so the
residual exploit is a *timing* exploit rather than a correctness one. Two gates
catch it: an absolute speedup cap whose value must be justified from the observed
distribution rather than picked, and a requirement that the winning measurement
was vendor-grade, since the executor only sets ``ok`` on a bench whose timing
protocol was admissible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# Measured on this repo's mixtures; used only for budgeting and length gates.
CHARS_PER_TOKEN = 3.6


@dataclass(frozen=True)
class FilterPolicy:
    """Thresholds for retention. Every default is argued in the module docstring."""

    min_gain: float = 1.15
    vendor_parity: float = 1.0
    max_speedup: float = 12.0
    min_high_gain_revisions: int = 1
    high_gain_revision_ratio: float = 1.05
    max_seq_tokens: int = 17408
    require_vendor_grade: bool = True
    keep_categories: frozenset = frozenset({"success", "repair"})


@dataclass
class TrajectoryStats:
    task_id: str = ""
    category: str = "unknown"
    success: bool = False
    turns: int = 0
    first_correct_speedup: Optional[float] = None
    best_speedup: Optional[float] = None
    gain: Optional[float] = None
    n_benches: int = 0
    n_correct_benches: int = 0
    high_gain_revisions: list = field(default_factory=list)
    best_bench_vendor_grade: bool = False
    est_tokens: float = 0.0

    @property
    def n_high_gain_revisions(self) -> int:
        return len(self.high_gain_revisions)


def _messages_text(record: dict) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    return "".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict)
    )


def revision_gains(record: dict, ratio: float = 1.05) -> list[dict]:
    """Turns where a correct candidate pushed the measured-speedup frontier.

    Walks the bench calls in order and compares each correct measurement against
    the best correct measurement before it. The executor already reports
    ``improved_frontier`` and ``delta_vs_best``, but those are recomputed here from
    the raw speedups so a record written by an older executor - or one whose
    frontier bookkeeping was reset by a lineage reseed - is still measured
    consistently.
    """
    gains: list[dict] = []
    running_best: Optional[float] = None
    for call in record.get("tool_trace") or []:
        if not isinstance(call, dict) or call.get("name") != "bench":
            continue
        result = call.get("result")
        if not isinstance(result, dict) or not result.get("correct"):
            continue
        speedup = result.get("speedup")
        if not isinstance(speedup, (int, float)) or speedup <= 0:
            continue
        if running_best is not None and speedup >= running_best * ratio:
            gains.append({
                "turn": call.get("turn"),
                "from": round(running_best, 4),
                "to": round(float(speedup), 4),
                "ratio": round(float(speedup) / running_best, 4),
            })
        if running_best is None or speedup > running_best:
            running_best = float(speedup)
    return gains


def analyze(record: dict, policy: FilterPolicy = FilterPolicy()) -> TrajectoryStats:
    """Summarize one trajectory into the quantities retention depends on."""
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    stats = TrajectoryStats(
        task_id=str(record.get("task_id") or ""),
        category=str(provenance.get("category") or "unknown"),
        success=bool(record.get("success")),
        turns=int(provenance.get("turns_used") or 0),
        est_tokens=len(_messages_text(record)) / CHARS_PER_TOKEN,
    )

    best_speedup: Optional[float] = None
    best_vendor_grade = False
    for call in record.get("tool_trace") or []:
        if not isinstance(call, dict) or call.get("name") != "bench":
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        stats.n_benches += 1
        if not result.get("correct"):
            continue
        speedup = result.get("speedup")
        if not isinstance(speedup, (int, float)) or speedup <= 0:
            continue
        stats.n_correct_benches += 1
        if stats.first_correct_speedup is None:
            stats.first_correct_speedup = float(speedup)
        if best_speedup is None or speedup > best_speedup:
            best_speedup = float(speedup)
            # ``ok`` is the executor's vendor-grade verdict: correct, no infra
            # error, and a timing protocol admissible for a published speedup.
            best_vendor_grade = bool(result.get("ok"))

    stats.best_speedup = best_speedup
    stats.best_bench_vendor_grade = best_vendor_grade
    if best_speedup is not None and stats.first_correct_speedup:
        stats.gain = best_speedup / stats.first_correct_speedup
    stats.high_gain_revisions = revision_gains(
        record, ratio=policy.high_gain_revision_ratio)
    return stats


def classify(record: dict, policy: FilterPolicy = FilterPolicy()) -> tuple[Optional[str], TrajectoryStats]:
    """Return ``(drop_reason, stats)``; ``drop_reason`` is None when kept.

    Ordered so the cheapest and most decisive rejections come first, and so the
    reported reason is the most informative one rather than whichever gate
    happened to run.
    """
    stats = analyze(record, policy)

    if stats.category not in policy.keep_categories:
        return "not_useful_category", stats
    if not stats.success:
        return "never_correct", stats
    if stats.best_speedup is None:
        # Correct but never benched: no measured evidence that anything improved.
        return "no_measured_speedup", stats
    if stats.best_speedup > policy.max_speedup:
        return "implausible_speedup", stats
    if policy.require_vendor_grade and not stats.best_bench_vendor_grade:
        return "timing_not_vendor_grade", stats
    if stats.est_tokens > policy.max_seq_tokens:
        # Truncating mid-refinement teaches the model to start an optimization
        # and never finish it.
        return "too_long", stats

    reached_parity = stats.best_speedup >= policy.vendor_parity
    improved = stats.gain is not None and stats.gain >= policy.min_gain
    enough_revisions = stats.n_high_gain_revisions >= policy.min_high_gain_revisions
    if not (reached_parity or (improved and enough_revisions)):
        return "low_gain", stats
    return None, stats


def speedup_distribution(values: Iterable[float]) -> dict:
    """Percentiles used to justify the reward-hack cap from observed data."""
    ordered = sorted(float(v) for v in values if isinstance(v, (int, float)) and v > 0)
    if not ordered:
        return {"n": 0}

    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]

    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "p50": round(at(0.50), 4),
        "p90": round(at(0.90), 4),
        "p99": round(at(0.99), 4),
        "p999": round(at(0.999), 4),
        "max": round(ordered[-1], 4),
    }


def suggest_max_speedup(values: Iterable[float], *, floor: float = 5.0) -> dict:
    """Propose a reward-hack cap from where the observed distribution thins out.

    The cap is a judgement call, so this reports the evidence behind it rather
    than only the number: the p99.9 of measured speedups and the multiple of the
    p99 it represents. A distribution with a long thin tail above the bulk is the
    signature of measurement exploits rather than of real fusion wins, which on
    this hardware land in the low single digits against a vendor kernel.
    """
    distribution = speedup_distribution(values)
    if not distribution.get("n"):
        return {"cap": floor, "basis": "no observations", **distribution}
    p99 = distribution["p99"]
    # Twice the p99, floored: wide enough that no genuine win in the observed
    # bulk is lost, tight enough that an order-of-magnitude outlier cannot pass.
    cap = max(floor, round(2.0 * p99, 2))
    return {
        "cap": cap,
        "basis": "2x observed p99, floored",
        "tail_ratio_max_over_p99": (
            round(distribution["max"] / p99, 2) if p99 else None),
        **distribution,
    }


__all__ = [
    "CHARS_PER_TOKEN",
    "FilterPolicy",
    "TrajectoryStats",
    "analyze",
    "classify",
    "revision_gains",
    "speedup_distribution",
    "suggest_max_speedup",
]
