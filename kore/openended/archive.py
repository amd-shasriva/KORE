"""A MAP-Elites TASK archive for open-ended co-evolution.

Mirrors the ``kore.data.evolve.MapElitesArchive`` pattern, but the elites here
are *tasks* (descriptors), not kernels: each behavior niche
(``task_space.descriptor_key``) keeps the single most **informative** task seen
for that niche, plus its outcome history. "Informative" = high learnability
(``4*p*(1-p)``) plus remaining performance-headroom regret — i.e. the task the
policy could still learn the most from — so the archive tracks the *competence
frontier* per region of behavior space rather than collapsing onto one family.

APIs: :meth:`add`/:meth:`update` (niching + fitness-gated replacement, history
always appended), :meth:`sample` (frontier-weighted sampling), :meth:`coverage`,
:meth:`best`/:meth:`frontier`, :meth:`occupied_keys` (the interface the proposer
uses for novelty).

Pure and deterministic (seeded). No torch at import; feature extraction lazily
touches torch only for the fusion/gemm families (see ``task_space``)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from kore.openended import task_space as ts
from kore.openended.proposer import DescriptorStats, clamp, learnability


def informativeness(stats: DescriptorStats) -> float:
    """How much the policy could still learn from a task: learnability + regret.

    Archive-internal fitness (novelty is archive-relative and handled by the
    proposer, so it is deliberately excluded here)."""
    return learnability(stats.solve_rate) + 0.5 * clamp(stats.headroom_regret)


@dataclass
class TaskCell:
    """The elite task for one behavior niche + its accumulated outcome history."""

    descriptor: ts.TaskDescriptor
    stats: DescriptorStats
    key: tuple
    history: list = field(default_factory=list)

    @property
    def fitness(self) -> float:
        return informativeness(self.stats)


class TaskArchive:
    """MAP-Elites grid over ``descriptor_key`` niches, keeping the most
    informative task per niche.

    ``add``/``update`` insert a ``(descriptor, stats)`` observation: the cell is
    (re)claimed when empty or when the new observation is strictly more
    informative than the incumbent, and the raw ``outcome`` (if given) is always
    appended to that niche's history."""

    def __init__(self, seed: int = 0):
        self.cells: dict[tuple, TaskCell] = {}
        self.rng = random.Random(seed)

    # -- niching / insertion ------------------------------------------------ #
    def key(self, descriptor: ts.TaskDescriptor) -> tuple:
        return ts.descriptor_key(descriptor)

    def add(self, descriptor: ts.TaskDescriptor, stats: DescriptorStats,
            outcome=None) -> bool:
        """Insert an observation; return True if it (re)claimed the niche's elite."""
        key = ts.descriptor_key(descriptor)
        cur = self.cells.get(key)
        if cur is None:
            cell = TaskCell(descriptor=descriptor, stats=stats, key=key)
            if outcome is not None:
                cell.history.append(outcome)
            self.cells[key] = cell
            return True
        if outcome is not None:
            cur.history.append(outcome)
        if informativeness(stats) > cur.fitness:
            cur.descriptor = descriptor
            cur.stats = stats
            return True
        return False

    # alias: same semantics as add (observation-driven update).
    update = add

    # -- queries ------------------------------------------------------------ #
    def occupied_keys(self) -> set:
        """The set of occupied niche keys (used by the proposer for novelty)."""
        return set(self.cells)

    def coverage(self) -> int:
        return len(self.cells)

    def __len__(self) -> int:
        return len(self.cells)

    def __contains__(self, descriptor: ts.TaskDescriptor) -> bool:
        return ts.descriptor_key(descriptor) in self.cells

    def cell(self, descriptor: ts.TaskDescriptor):
        return self.cells.get(ts.descriptor_key(descriptor))

    def cells_list(self) -> list:
        return list(self.cells.values())

    def best(self, n: int = 1) -> list:
        """The ``n`` most-informative elite cells (frontier)."""
        return sorted(self.cells.values(),
                      key=lambda c: (c.fitness, ts._sort_key(c.descriptor)),
                      reverse=True)[:n]

    def frontier(self, n: int = 1) -> list:
        """The ``n`` most-informative elite *descriptors*."""
        return [c.descriptor for c in self.best(n)]

    # -- sampling ----------------------------------------------------------- #
    def sample(self, n: int = 1, seed=None) -> list:
        """Sample ``n`` descriptors, frontier-weighted by informativeness.

        Cells with higher fitness (learnability + regret) are proportionally more
        likely, so sampling favours the competence frontier. A small floor keeps
        zero-fitness niches reachable. Deterministic when ``seed`` is given."""
        if not self.cells or n <= 0:
            return []
        rng = random.Random(seed) if seed is not None else self.rng
        cells = sorted(self.cells.values(), key=lambda c: ts._sort_key(c.descriptor))
        weights = [c.fitness + 1e-6 for c in cells]
        return [c.descriptor for c in rng.choices(cells, weights=weights, k=n)]

    # -- reporting ---------------------------------------------------------- #
    def coverage_by_field(self, field_name: str) -> dict:
        """Count of occupied niches per value of one niche field (diagnostics)."""
        idx = ts.NICHE_FIELDS.index(field_name)
        out: dict = {}
        for key in self.cells:
            out[key[idx]] = out.get(key[idx], 0) + 1
        return out

    def summary(self) -> dict:
        best = self.best(1)
        return {
            "coverage": self.coverage(),
            "families": self.coverage_by_field("family"),
            "top_fitness": best[0].fitness if best else 0.0,
            "top_task": best[0].descriptor.task_id if best else None,
        }
