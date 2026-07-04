"""CPU tests for the open-ended co-evolution loop (mocked policy + measurement)."""

from __future__ import annotations

import pytest

from kore.openended.archive import TaskArchive
from kore.openended.coevolve import (
    _headroom_regret,
    run_coevolution,
    run_generation,
)
from kore.openended.proposer import DescriptorStats


def _mock_measure_factory(competence: float, rng_seed: int = 0):
    """A mock env whose pass-rate = ``competence``; correct kernels get a modest
    speedup. Deterministic per (descriptor, try) so tests are reproducible."""
    def measure(desc, src):
        h = hash((desc.task_id, src)) & 0xFFFF
        r = (h / 0xFFFF)
        correct = r < competence
        speedup = (0.8 + 0.6 * r) if correct else None
        return {"correct": correct, "verified": correct,
                "speedup": speedup, "reward": (speedup or 0.0)}
    return measure


def _mock_policy(desc, i):
    return f"# kernel for {desc.task_id} attempt {i}"


def test_headroom_regret_bounds():
    assert _headroom_regret(None) == 1.0
    assert _headroom_regret(1.5) == 0.0
    assert _headroom_regret(1.0) == 0.0
    assert _headroom_regret(0.4) == pytest.approx(0.6)


def test_run_generation_updates_archive_and_history():
    archive = TaskArchive(seed=0)
    history: dict = {}
    rep = run_generation(archive, history, _mock_policy,
                         _mock_measure_factory(0.5), generation=0,
                         n_tasks=8, k_attempts=4, seed=1)
    assert rep.n_proposed > 0
    assert rep.n_attempts == rep.n_proposed * 4
    assert rep.n_correct >= 0
    assert len(history) == rep.n_proposed
    assert archive.coverage() >= 1
    # every proposed descriptor got a stats entry
    for d, s in history.items():
        assert isinstance(s, DescriptorStats)
        assert 0.0 <= s.solve_rate <= 1.0


def test_wins_require_verified_and_speedup():
    """A distill sink should only receive verified, >1x records."""
    collected = []
    archive = TaskArchive(seed=0)
    history: dict = {}
    run_generation(archive, history, _mock_policy,
                   _mock_measure_factory(0.9), generation=0,
                   n_tasks=16, k_attempts=6, win_tau=1.0, seed=2,
                   distill_fn=lambda ws: collected.extend(ws))
    for w in collected:
        assert w["verified"] is True
        assert w["speedup"] > 1.0


def test_full_loop_runs_and_reports_curve():
    reports = run_coevolution(_mock_policy, _mock_measure_factory(0.5),
                              generations=5, n_tasks=8, k_attempts=4, seed=0)
    assert len(reports) == 5
    for g, rep in enumerate(reports):
        assert rep.generation == g
    # coverage is monotic non-decreasing across generations (shared archive)
    covs = [r.archive_coverage for r in reports]
    assert covs == sorted(covs)


def test_deterministic():
    r1 = run_coevolution(_mock_policy, _mock_measure_factory(0.5),
                         generations=3, n_tasks=8, k_attempts=3, seed=7)
    r2 = run_coevolution(_mock_policy, _mock_measure_factory(0.5),
                         generations=3, n_tasks=8, k_attempts=3, seed=7)
    assert [x.n_attempts for x in r1] == [x.n_attempts for x in r2]
    assert [x.n_correct for x in r1] == [x.n_correct for x in r2]
