"""Co-evolution curriculum controller: the open-ended loop, wired to real GRPO.

The GRPO trainer (:func:`kore.policy.grpo._train_grpo_fallback`) normally walks its
task list ROUND-ROBIN. This controller replaces that with the open-ended
task-frontier policy: each task-group is selected at the policy's *competence
frontier* (proposer: learnability ``4p(1-p)`` + performance-headroom regret +
archive novelty), and every group's measured outcome (solve-rate, best speedup)
is fed back into the MAP-Elites task archive so the curriculum co-evolves with the
policy - the "verifiably-grounded open-ended kernel discovery" paradigm made real
inside the training loop.

Grounding: the controller only ever returns task_ids that are ACTUALLY registered
(the intersection of the parametric task space with the trainer's allowed task
list), and selection does not mutate op/dtype off the registry, so every proposed
task is guaranteed runnable on hardware. Frontier selection ranges over the full
registered (op × dtype) menu; shape-regime is handled inside each task's own shape
sweep. (Dynamic minting of brand-new generated ops mid-run is a future extension;
this integration co-evolves the curriculum over the registered space.)

Pure control logic (no torch): CPU-unit-testable. The trainer calls
:meth:`next_task_id` per group and :meth:`record` with the group outcome.
"""

from __future__ import annotations

from typing import Optional

from kore.openended.archive import TaskArchive
from kore.openended.coevolve import _headroom_regret
from kore.openended.proposer import DEFAULT_WEIGHTS, DescriptorStats, ScoreWeights, propose
from kore.openended.task_space import TaskDescriptor, enumerate_descriptors


class CoevolutionController:
    """Frontier-targeted task selector + outcome sink for the GRPO loop.

    Parameters
    ----------
    task_ids:
        The trainer's allowed task list (registered task_ids). The controller
        operates over the subset of the parametric space whose ``task_id`` is in
        this list; any non-mappable ids (e.g. hand-authored tasks) remain
        reachable via the round-robin fallback so nothing is starved.
    seed:
        Base seed for deterministic proposal.
    batch:
        How many frontier tasks to propose per refill (defaults to ``len(menu)``
        capped, so a refill sweeps the frontier once).
    k_attempts:
        Trajectories per group (only used to annotate ``DescriptorStats.attempts``
        so the proposer's collapse guardrail engages after real evidence).
    include_vendor:
        Include vendor-baselined ops in the space.
    weights:
        Proposer scoring weights.
    """

    def __init__(self, task_ids, *, seed: int = 0, batch: Optional[int] = None,
                 k_attempts: int = 1, include_vendor: bool = True,
                 weights: ScoreWeights = DEFAULT_WEIGHTS):
        self.allowed = list(dict.fromkeys(task_ids))
        self.allowed_set = set(self.allowed)
        # Menu: registered descriptors, one representative per task_id (prefer the
        # 'primary' shape regime as the canonical niche representative).
        menu: dict[str, TaskDescriptor] = {}
        for d in enumerate_descriptors(include_vendor):
            if d.task_id not in self.allowed_set:
                continue
            cur = menu.get(d.task_id)
            if cur is None or (d.shape_regime == "primary" and cur.shape_regime != "primary"):
                menu[d.task_id] = d
        self.by_task = menu                       # task_id -> representative descriptor
        self.menu = sorted(menu.values(), key=lambda d: d.task_id)
        self.archive = TaskArchive(seed=seed)
        self.history: dict[TaskDescriptor, DescriptorStats] = {}
        self.seed = seed
        self.k_attempts = max(1, int(k_attempts))
        self.weights = weights
        self.batch = batch if batch is not None else max(1, min(64, len(self.menu)))
        self._queue: list[str] = []               # proposed task_ids to serve
        self._refills = 0
        self._rr_cursor = 0                        # fallback round-robin cursor
        self._served = 0
        self._recorded = 0

    # ------------------------------------------------------------------ #
    # selection
    # ------------------------------------------------------------------ #
    def _round_robin(self) -> str:
        tid = self.allowed[self._rr_cursor % len(self.allowed)]
        self._rr_cursor += 1
        return tid

    def _refill(self) -> None:
        """Propose a fresh frontier batch (grounded to the registered menu)."""
        proposed = propose(self.archive, self.history, self.batch,
                           seed=self.seed + self._refills, weights=self.weights,
                           mutate=False, candidate_pool=self.menu)
        self._refills += 1
        # map to registered task_ids, dedupe preserving order, keep only runnable
        seen: set[str] = set()
        q: list[str] = []
        for d in proposed:
            tid = d.task_id
            if tid in self.allowed_set and tid not in seen:
                seen.add(tid)
                q.append(tid)
        self._queue = q

    def next_task_id(self, step: int = 0, attempt: int = 0) -> str:
        """Return the next task_id to roll out (frontier-selected, or round-robin
        fallback when no registered task maps into the parametric space)."""
        self._served += 1
        if not self.menu:
            return self._round_robin()
        if not self._queue:
            self._refill()
        if not self._queue:                       # proposer returned nothing runnable
            return self._round_robin()
        return self._queue.pop(0)

    # ------------------------------------------------------------------ #
    # feedback
    # ------------------------------------------------------------------ #
    def record(self, task_id: str, solve_rate: float,
               best_speedup: Optional[float]) -> bool:
        """Feed a group's measured outcome back into the archive + history.

        Returns True if the outcome (re)claimed its archive niche's elite.
        No-op (returns False) for task_ids outside the parametric menu."""
        desc = self.by_task.get(task_id)
        if desc is None:
            return False
        stats = DescriptorStats(
            solve_rate=max(0.0, min(1.0, float(solve_rate))),
            headroom_regret=_headroom_regret(best_speedup),
            attempts=self.k_attempts,
        )
        self.history[desc] = stats
        self._recorded += 1
        return self.archive.add(desc, stats, outcome={
            "task_id": task_id, "solve_rate": stats.solve_rate,
            "best_speedup": best_speedup})

    # ------------------------------------------------------------------ #
    # observability
    # ------------------------------------------------------------------ #
    def report(self) -> dict:
        """Compact snapshot for logging the co-evolution curriculum state."""
        rates = [s.solve_rate for s in self.history.values()]
        regrets = [s.headroom_regret for s in self.history.values()]
        return {
            "menu_size": len(self.menu),
            "allowed": len(self.allowed),
            "archive_coverage": self.archive.coverage(),
            "measured_tasks": len(self.history),
            "served": self._served,
            "recorded": self._recorded,
            "refills": self._refills,
            "mean_solve_rate": (sum(rates) / len(rates)) if rates else 0.0,
            "mean_regret": (sum(regrets) / len(regrets)) if regrets else 0.0,
            "queue_remaining": len(self._queue),
        }
