"""Decompose agentic trajectories into step-centric supervision.

Step-centric rows are not the only thing worth keeping, and this module also owns
the other half -- see :func:`extract_full_trajectory`. Measured on the overnight
campaign: 3,475 trajectories reached a correct kernel and 1,942 of them (55.9%)
produced no step row at all, because 1,576 were correct on their FIRST turn and a
step needs a "before" to improve on. Those are not weak trajectories -- their
median measured speedup is 1.58x and 507 of them cleared 2x -- they are simply
invisible to a representation built around revisions.


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


@dataclass
class FullTrajectory:
    """One whole successful episode, ending on the turn that won it."""

    task_id: str
    messages: list
    turns: int                     # assistant turns kept
    first_correct_turn: int        # 1-based
    best_turn: int                 # 1-based; the turn this row ends on
    best_speedup: Optional[float]

    def to_row(self) -> dict:
        return {
            "messages": self.messages,
            "_source": "kernel_full_trajectory",
            "_task_id": self.task_id,
            "_turns": self.turns,
            "_first_correct_turn": self.first_correct_turn,
            "_best_turn": self.best_turn,
            "_best_speedup": self.best_speedup,
        }


def extract_full_trajectory(
    record: dict,
    max_speedup: float = 50.0,
) -> Optional[FullTrajectory]:
    """The whole episode as one row, truncated at the turn that won it.

    Dr. Kernel builds its cold start from FULL multi-turn trajectories with
    execution feedback, where Kernel-Smith uses step-centric revisions. Doing only
    the latter throws away every trajectory whose win was not a *revision*: a
    first-turn success has no parent to improve on, so :func:`extract_steps`
    cannot represent it however good the kernel was.

    Two rules make this safe to train on:

    * **Never emit a trajectory that did not reach correctness.** Eight turns of
      failure teaches failure, and the campaign has 8,189 of them.
    * **End on the best correct turn.** Everything after the win is by
      construction a non-improvement or a regression, and training through it
      teaches the model to keep editing a kernel that was already right.

    The reward-hacking guard from :func:`extract_steps` applies unchanged: a
    trajectory whose best "speedup" is three orders of magnitude is a decoy, and
    ending a row on it would teach exactly the metric-cheating that RL will later
    be free to exploit.
    """
    prov = record.get("provenance") or {}
    correct = [bool(c) for c in (prov.get("turn_correct") or [])]
    speedups = [_f(s) for s in (prov.get("turn_speedups") or [])]
    messages = list(record.get("messages") or [])
    task_id = str(record.get("task_id") or "")

    turns = _assistant_turn_indices(messages)
    if not turns or not any(correct):
        return None

    first_correct = correct.index(True)
    # The winning turn is the fastest CORRECT one; with no timing anywhere, the
    # first correct turn is the only defensible end point -- it is the earliest
    # place the episode is known to be right, and later turns carry no evidence
    # that they are still right about anything.
    best_turn = first_correct
    best_speedup = None
    for index, is_correct in enumerate(correct):
        if not is_correct or index >= len(speedups):
            continue
        candidate = speedups[index]
        if candidate is None:
            continue
        if best_speedup is None or candidate > best_speedup:
            best_speedup, best_turn = candidate, index

    if best_speedup is not None and best_speedup > max_speedup:
        return None
    if best_turn >= len(turns):
        return None

    return FullTrajectory(
        task_id=task_id,
        messages=messages[: turns[best_turn] + 1],
        turns=best_turn + 1,
        first_correct_turn=first_correct + 1,
        best_turn=best_turn + 1,
        best_speedup=best_speedup,
    )


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


def decompose_with_trajectories(
    records: Iterable[dict],
    min_gain: float = MIN_GAIN,
    max_speedup: float = 50.0,
    only_residual: bool = True,
) -> tuple[list[dict], dict]:
    """Step-centric rows, plus a full-trajectory row for the successes they miss.

    ``only_residual`` (the default) emits a full trajectory ONLY when the same
    episode produced no step row. A step row's messages are a PREFIX of the whole
    trajectory, so emitting both for one episode puts the same tokens in the
    corpus twice, and content-hash dedup does not catch a prefix. Residual mode
    fills the hole -- the 55.9% of successful trajectories that currently
    contribute nothing -- without reweighting the trajectories that already do.

    Set ``only_residual=False`` to reproduce Dr. Kernel's setup, where every
    successful trajectory is a cold-start example in its own right.
    """
    rows: list[dict] = []
    stats = {
        "trajectories": 0, "with_steps": 0, "steps": 0,
        "fix_steps": 0, "speedup_steps": 0,
        "reached_correct": 0, "full_trajectories": 0,
        "full_rejected_hack": 0, "full_skipped_has_steps": 0,
        "never_correct_dropped": 0,
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

        prov = rec.get("provenance") or {}
        if not any(bool(c) for c in (prov.get("turn_correct") or [])):
            stats["never_correct_dropped"] += 1
            continue
        stats["reached_correct"] += 1
        if steps and only_residual:
            stats["full_skipped_has_steps"] += 1
            continue
        full = extract_full_trajectory(rec, max_speedup=max_speedup)
        if full is None:
            stats["full_rejected_hack"] += 1
            continue
        stats["full_trajectories"] += 1
        rows.append(full.to_row())

    log.metric("step_centric_decompose_with_trajectories", **stats)
    return rows, stats
