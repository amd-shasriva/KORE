"""Open-ended co-evolution loop: the task frontier and the policy improve together.

This is the capstone of the "verifiably-grounded open-ended kernel discovery"
paradigm. Each generation:

  1. PROPOSE a batch of tasks at the policy's competence frontier
     (proposer scores candidate TaskDescriptors by learnability p(1-p) + headroom
     regret + novelty vs the archive).
  2. ATTEMPT each task with the current policy (k tries), verify + measure each
     candidate on real hardware (correctness oracle + speedup vs the production
     baseline) -> per-descriptor outcome stats (solve rate, headroom regret).
  3. UPDATE the MAP-Elites task archive + the outcome history.
  4. DISTILL the winning trajectories (correct + >1x) back into training data
     (RFT/expert-iteration) for the next policy update.

The loop is PURE orchestration: the policy, the measurement (env), and the
distillation sink are INJECTED, so it is fully unit-testable on CPU with mocks and
GPU-agnostic. It is deliberately decoupled from run_campaign/grpo - a driver script
wires a real served policy + KoreEnv + the GRPO update into these hooks.

Why kernels make open-endedness actually work (the paradigm claim): the task space
is infinitely + cheaply generatable, every task is ground-truth VERIFIABLE, and
carries a CONTINUOUS performance-headroom regret signal - which simultaneously
solves UED's regret-estimation problem and the unverifiable-reward problem that
block open-ended RL in other domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from kore.openended.archive import TaskArchive
from kore.openended.proposer import DescriptorStats, propose
from kore.openended.task_space import TaskDescriptor

# A policy attempt: (descriptor, try_index) -> a candidate kernel source string.
PolicyFn = Callable[[TaskDescriptor, int], str]
# Measurement: (descriptor, kernel_src) -> outcome dict with at least
# {"correct": bool, "speedup": float|None, "verified": bool}.
MeasureFn = Callable[[TaskDescriptor, str], dict]
# Distillation sink: consume winning (descriptor, kernel_src, outcome) records.
DistillFn = Callable[[list[dict]], None]


@dataclass
class Outcome:
    descriptor: TaskDescriptor
    correct: bool
    verified: bool
    speedup: Optional[float]
    reward: float
    kernel_src: str


@dataclass
class GenerationReport:
    generation: int
    n_proposed: int
    n_attempts: int
    n_correct: int
    n_verified: int
    n_wins: int                 # correct AND speedup > win_tau
    mean_solve_rate: float
    archive_coverage: int
    frontier_solve_rates: list[float] = field(default_factory=list)


def _headroom_regret(speedup: Optional[float]) -> float:
    """Regret = how far below the 'genuinely beats baseline' bar (1x) the best
    attempt is, in [0,1]. Correct-but-slow kernels carry the most learnable regret;
    already-fast (>=1x) kernels carry ~0 (little left to learn)."""
    if speedup is None:
        return 1.0
    if speedup >= 1.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - speedup))


def run_generation(
    archive: TaskArchive,
    history: dict[TaskDescriptor, DescriptorStats],
    policy_fn: PolicyFn,
    measure_fn: MeasureFn,
    *,
    generation: int,
    n_tasks: int = 16,
    k_attempts: int = 4,
    win_tau: float = 1.0,
    seed: int = 0,
    distill_fn: Optional[DistillFn] = None,
) -> GenerationReport:
    """Run ONE open-ended generation: propose -> attempt+verify -> update -> distill.

    Mutates ``archive`` and ``history`` in place; returns a report. Pure except for
    the injected policy/measure/distill side effects.
    """
    proposed = propose(archive, history, n_tasks, seed=seed)
    wins: list[dict] = []
    n_attempts = n_correct = n_verified = 0
    solve_rates: list[float] = []

    for desc in proposed:
        n_solved = 0
        best_speedup: Optional[float] = None
        best_rec: Optional[dict] = None
        verified_any = False
        for i in range(k_attempts):
            src = policy_fn(desc, i)
            out = measure_fn(desc, src)
            n_attempts += 1
            correct = bool(out.get("correct"))
            verified = bool(out.get("verified"))
            su = out.get("speedup")
            reward = float(out.get("reward", 0.0))
            if correct:
                n_solved += 1
                n_correct += 1
            if verified:
                verified_any = True
                n_verified += 1
            if correct and su is not None and (best_speedup is None or su > best_speedup):
                best_speedup = su
                best_rec = {"descriptor": desc, "kernel_src": src, "correct": correct,
                            "verified": verified, "speedup": su, "reward": reward}
            if correct and su is not None and su > win_tau and verified:
                wins.append({"descriptor": desc, "kernel_src": src, "speedup": su,
                             "reward": reward, "verified": verified})

        p = n_solved / max(1, k_attempts)
        solve_rates.append(p)
        stats = DescriptorStats(solve_rate=p, headroom_regret=_headroom_regret(best_speedup),
                                attempts=k_attempts)
        history[desc] = stats
        archive.add(desc, stats, outcome=best_rec)

    if distill_fn is not None and wins:
        distill_fn(wins)

    n_wins = len(wins)
    return GenerationReport(
        generation=generation, n_proposed=len(proposed), n_attempts=n_attempts,
        n_correct=n_correct, n_verified=n_verified, n_wins=n_wins,
        mean_solve_rate=(sum(solve_rates) / len(solve_rates)) if solve_rates else 0.0,
        archive_coverage=archive.coverage(),
        frontier_solve_rates=solve_rates,
    )


def run_coevolution(
    policy_fn: PolicyFn,
    measure_fn: MeasureFn,
    *,
    generations: int = 10,
    n_tasks: int = 16,
    k_attempts: int = 4,
    win_tau: float = 1.0,
    seed: int = 0,
    distill_fn: Optional[DistillFn] = None,
    archive: Optional[TaskArchive] = None,
) -> list[GenerationReport]:
    """Run the full open-ended loop for ``generations`` generations. Returns the
    per-generation reports (the unbounded-improvement curve to plot)."""
    archive = archive if archive is not None else TaskArchive(seed=seed)
    history: dict[TaskDescriptor, DescriptorStats] = {}
    reports: list[GenerationReport] = []
    for g in range(generations):
        rep = run_generation(archive, history, policy_fn, measure_fn,
                             generation=g, n_tasks=n_tasks, k_attempts=k_attempts,
                             win_tau=win_tau, seed=seed + g, distill_fn=distill_fn)
        reports.append(rep)
    return reports
