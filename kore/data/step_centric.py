"""Decompose agentic trajectories into step-centric supervision.

Kernel-Smith's central finding is that a model trained on whole optimization
trajectories learns to imitate a *search*, while what you actually want is a
strong LOCAL IMPROVER inside an evolutionary loop -- something that, shown a
kernel and its execution feedback, reliably makes it faster without breaking it.
Their recipe converts long-horizon trajectories into step-centric supervision by
"retaining correctness-preserving, high-gain revisions", and that single change
is what separates their 235B (3.70 average speedup on KernelBench) from
Claude-4.6-opus (3.33).

Training on the whole trajectory teaches the wrong lesson in two specific ways:

  * a 5-turn trajectory whose first four turns are wrong and whose fifth is
    right trains the model on four failures for every success, and the failures
    are indistinguishable from the success at the token level;
  * it rewards *reaching* a good kernel eventually, which a model can satisfy by
    flailing, rather than *improving* on each step, which is the behaviour the
    RL stage will actually sample.

So a trajectory becomes up to N-1 independent examples, each ending at one
revision, and only the revisions worth imitating are kept. A step qualifies when
it preserves correctness -- correct before and after, or the step that FIXES a
broken kernel -- and gains meaningfully on measured speedup. Regressions,
no-op edits and lucky-but-broken revisions are dropped.

Everything here is pure data transformation on records already on disk, so it is
unit-testable on CPU and can be re-run with different thresholds without
regenerating a single episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from kore.obs import get_logger

log = get_logger("data.step_centric")

# A revision has to beat its parent by more than measurement noise. Cold-cache
# kernel timing on a shared GPU moves a couple of percent run to run, so 5% is
# the floor at which "faster" is a claim about the kernel rather than the clock.
MIN_GAIN = 0.05


class StepCentricError(RuntimeError):
    """A trajectory cannot be decomposed into steps."""


@dataclass
class Step:
    """One revision, with the evidence that it was worth keeping."""

    task_id: str
    turn: int                      # 1-based index of the revision
    messages: list                 # conversation ending at this revision
    speedup_before: Optional[float]
    speedup_after: Optional[float]
    correct_before: bool
    correct_after: bool
    gain: float
    kind: str                      # "fix" | "speedup"

    def to_row(self) -> dict:
        return {
            "messages": self.messages,
            "_source": "kernel_step_centric",
            "_task_id": self.task_id,
            "_turn": self.turn,
            "_kind": self.kind,
            "_gain": round(self.gain, 4),
            "_speedup_before": self.speedup_before,
            "_speedup_after": self.speedup_after,
        }


def _assistant_turn_indices(messages: list) -> list[int]:
    return [i for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "assistant"]


def _f(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    # A non-finite or non-positive speedup is a failed measurement, not a slow
    # kernel; treating it as 0.0 would make the next turn look like a huge gain.
    if out != out or out <= 0.0:
        return None
    return out


def extract_steps(
    record: dict,
    min_gain: float = MIN_GAIN,
    max_speedup: float = 50.0,
) -> list[Step]:
    """Return the correctness-preserving, high-gain revisions in one trajectory.

    ``max_speedup`` rejects the reward-hacking signature: a Triton kernel three
    orders of magnitude faster than its Torch reference is a decoy that never
    runs or computation that was skipped, and a step that "achieves" it teaches
    the model to cheat on the exact metric RL will later optimise.
    """
    prov = record.get("provenance") or {}
    correct = list(prov.get("turn_correct") or [])
    speedups = [_f(s) for s in (prov.get("turn_speedups") or [])]
    # turn_rewards is the fallback gain signal for trajectories written before
    # turn_speedups was persisted. It blends terms besides runtime, so it is a
    # weaker proxy -- used only when speedups are absent, never preferred.
    rewards = [_f(r) for r in (prov.get("turn_rewards") or [])]
    messages = list(record.get("messages") or [])
    task_id = str(record.get("task_id") or "")

    turns = _assistant_turn_indices(messages)
    if len(turns) < 2 or not correct:
        return []

    series = speedups if any(s is not None for s in speedups) else rewards
    n = min(len(turns), len(correct), len(series))
    steps: list[Step] = []
    for i in range(1, n):
        before, after = series[i - 1], series[i]
        c_before = bool(correct[i - 1])
        c_after = bool(correct[i])
        if not c_after:
            continue                      # never imitate a revision that broke it
        if after is not None and after > max_speedup:
            continue                      # reward-hacking signature
        if not c_before:
            kind, gain = "fix", 1.0       # the revision that made it correct
        else:
            if before is None or after is None:
                continue
            gain = (after - before) / before
            if gain < min_gain:
                continue                  # noise, or a regression
            kind = "speedup"
        steps.append(Step(
            task_id=task_id,
            turn=i + 1,
            messages=messages[: turns[i] + 1],
            speedup_before=speedups[i - 1] if i - 1 < len(speedups) else None,
            speedup_after=speedups[i] if i < len(speedups) else None,
            correct_before=c_before,
            correct_after=c_after,
            gain=gain,
            kind=kind,
        ))
    return steps


def decompose(
    records: Iterable[dict],
    min_gain: float = MIN_GAIN,
    max_speedup: float = 50.0,
) -> tuple[list[dict], dict]:
    """Decompose many trajectories, returning rows and a summary."""
    rows: list[dict] = []
    stats = {
        "trajectories": 0, "with_steps": 0, "steps": 0,
        "fix_steps": 0, "speedup_steps": 0,
    }
    for rec in records:
        stats["trajectories"] += 1
        steps = extract_steps(rec, min_gain=min_gain, max_speedup=max_speedup)
        if steps:
            stats["with_steps"] += 1
        for s in steps:
            stats["steps"] += 1
            stats["fix_steps" if s.kind == "fix" else "speedup_steps"] += 1
            rows.append(s.to_row())
    log.metric("step_centric_decompose", **stats)
    return rows, stats
