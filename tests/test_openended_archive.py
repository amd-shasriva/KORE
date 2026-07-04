"""CPU-only tests for the MAP-Elites task archive."""

from __future__ import annotations

from collections import Counter

from kore.openended import task_space as ts
from kore.openended.archive import TaskArchive, informativeness
from kore.openended.proposer import DescriptorStats


def _stats(p, regret=0.0, attempts=10):
    return DescriptorStats(solve_rate=p, headroom_regret=regret, attempts=attempts)


def test_informativeness_tracks_learnability_and_regret():
    assert informativeness(_stats(0.5)) > informativeness(_stats(0.1))
    assert informativeness(_stats(0.5, regret=1.0)) > informativeness(_stats(0.5, regret=0.0))
    assert informativeness(_stats(1.0)) == 0.0


def test_add_and_niching():
    arch = TaskArchive(seed=0)
    d = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "primary")
    assert arch.add(d, _stats(0.5))
    assert len(arch) == 1
    assert d in arch
    # a descriptor mapping to the SAME niche does not grow coverage
    same_niche = ts.TaskDescriptor("genops", "unary", "abs", "bf16", "primary")
    assert ts.descriptor_key(same_niche) == ts.descriptor_key(d)
    arch.add(same_niche, _stats(0.1))  # less informative -> no takeover
    assert len(arch) == 1
    assert arch.cell(d).descriptor == d


def test_more_informative_takes_over_cell():
    arch = TaskArchive(seed=0)
    weak = ts.TaskDescriptor("genops", "unary", "abs", "bf16", "primary")
    strong = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "primary")
    arch.add(weak, _stats(0.05))          # low learnability
    claimed = arch.add(strong, _stats(0.5))  # high learnability, same niche
    assert claimed
    assert arch.cell(strong).descriptor == strong


def test_coverage_grows_with_distinct_niches():
    arch = TaskArchive(seed=0)
    seen = set()
    n_added = 0
    for d in ts.sample_descriptors(60, seed=1):
        before = len(arch)
        arch.add(d, _stats(0.5))
        key = ts.descriptor_key(d)
        if key not in seen:
            seen.add(key)
            assert len(arch) == before + 1
            n_added += 1
    assert arch.coverage() == len(seen)
    assert n_added > 1


def test_history_always_appended():
    arch = TaskArchive(seed=0)
    d = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "primary")
    arch.add(d, _stats(0.5), outcome={"gen": 0})
    arch.add(d, _stats(0.4), outcome={"gen": 1})  # worse, but history still grows
    cell = arch.cell(d)
    assert len(cell.history) == 2


def test_occupied_keys_matches_cells():
    arch = TaskArchive(seed=0)
    for d in ts.sample_descriptors(10, seed=2):
        arch.add(d, _stats(0.5))
    assert arch.occupied_keys() == set(arch.cells)


def test_best_and_frontier_ordered():
    arch = TaskArchive(seed=0)
    lo = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "minimal")
    hi = ts.TaskDescriptor("genops", "gemm_fusion", "gemm_bias", "bf16", "primary")
    arch.add(lo, _stats(0.2))
    arch.add(hi, _stats(0.5, regret=1.0))
    assert arch.best(1)[0].descriptor == hi
    assert arch.frontier(2)[0] == hi


def test_sample_favors_frontier():
    arch = TaskArchive(seed=0)
    frontier = ts.TaskDescriptor("genops", "gemm_fusion", "gemm_bias", "bf16", "primary")
    dull = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "minimal")
    arch.add(frontier, _stats(0.5, regret=1.0))  # high informativeness
    arch.add(dull, _stats(0.02))                 # ~0 informativeness
    picks = Counter(d.task_id for d in arch.sample(400, seed=0))
    assert picks[frontier.task_id] > picks[dull.task_id]


def test_sample_deterministic_and_empty():
    arch = TaskArchive(seed=0)
    assert arch.sample(5, seed=0) == []
    for d in ts.sample_descriptors(8, seed=3):
        arch.add(d, _stats(0.5))
    assert arch.sample(5, seed=1) == arch.sample(5, seed=1)


def test_coverage_by_field_and_summary():
    arch = TaskArchive(seed=0)
    for d in ts.sample_descriptors(30, seed=4):
        arch.add(d, _stats(0.5))
    by_fam = arch.coverage_by_field("family")
    assert sum(by_fam.values()) == arch.coverage()
    summ = arch.summary()
    assert summ["coverage"] == arch.coverage()
    assert "families" in summ and "top_task" in summ
