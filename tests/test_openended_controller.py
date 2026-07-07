"""CPU tests for the co-evolution curriculum controller (GRPO integration logic)."""

from __future__ import annotations

from kore.openended.controller import CoevolutionController
from kore.openended.task_space import enumerate_descriptors


def _some_registered_task_ids(n=12):
    """A handful of real, registered generated task_ids (gen_<op>_<dtype>)."""
    ids = []
    seen = set()
    for d in enumerate_descriptors(include_vendor=True):
        if d.task_id not in seen:
            seen.add(d.task_id)
            ids.append(d.task_id)
        if len(ids) >= n:
            break
    return ids


def test_menu_restricts_to_registered_tasks():
    tids = _some_registered_task_ids(8)
    ctrl = CoevolutionController(tids + ["totally_unregistered_task"], seed=0)
    # every menu descriptor maps to an allowed id; the bogus id is never in by_task
    assert all(t in ctrl.allowed_set for t in ctrl.by_task)
    assert "totally_unregistered_task" not in ctrl.by_task
    assert set(ctrl.by_task).issubset(set(tids + ["totally_unregistered_task"]))


def test_next_task_id_always_runnable():
    tids = _some_registered_task_ids(10)
    ctrl = CoevolutionController(tids, seed=1)
    for step in range(3):
        for attempt in range(6):
            tid = ctrl.next_task_id(step, attempt)
            assert tid in ctrl.allowed_set


def test_record_updates_archive_and_history():
    tids = _some_registered_task_ids(10)
    ctrl = CoevolutionController(tids, seed=2)
    tid = ctrl.next_task_id(0, 0)
    claimed = ctrl.record(tid, solve_rate=0.5, best_speedup=0.8)
    assert claimed is True
    assert ctrl.archive.coverage() >= 1
    assert ctrl.report()["measured_tasks"] == 1
    # regret from a 0.8x kernel is 0.2
    desc = ctrl.by_task[tid]
    assert abs(ctrl.history[desc].headroom_regret - 0.2) < 1e-9


def test_record_ignores_unregistered():
    ctrl = CoevolutionController(_some_registered_task_ids(6), seed=3)
    assert ctrl.record("not_a_task", 0.5, 1.2) is False


def test_frontier_selection_prefers_learnable_tasks():
    """After recording a mix of trivial/unsolvable/frontier outcomes, the proposer
    should favor the mid-solve-rate (learnable) task over the collapsed ones."""
    tids = _some_registered_task_ids(12)
    ctrl = CoevolutionController(tids, seed=4, batch=6)
    # warm the archive: make tids[0] trivial, tids[1] unsolvable, tids[2] frontier
    for _ in range(2):
        ctrl.record(tids[0], solve_rate=1.0, best_speedup=2.0)   # trivial + fast
        ctrl.record(tids[1], solve_rate=0.0, best_speedup=None)  # unsolvable
        ctrl.record(tids[2], solve_rate=0.5, best_speedup=0.7)   # learnable + regret
    # force a refill and inspect the proposed queue
    ctrl._queue = []
    picks = [ctrl.next_task_id(9, a) for a in range(6)]
    # the learnable task should be selected; the trivial one should not dominate
    assert tids[2] in picks


def test_determinism():
    tids = _some_registered_task_ids(10)
    a = CoevolutionController(tids, seed=7)
    b = CoevolutionController(tids, seed=7)
    for step in range(2):
        for attempt in range(5):
            assert a.next_task_id(step, attempt) == b.next_task_id(step, attempt)


def test_round_robin_fallback_when_no_menu():
    """If none of the allowed ids map into the parametric space, fall back to a
    round-robin over the raw allowed list (nothing starves)."""
    ctrl = CoevolutionController(["hand_authored_a", "hand_authored_b"], seed=0)
    assert ctrl.menu == []
    picks = [ctrl.next_task_id(0, a) for a in range(4)]
    assert picks == ["hand_authored_a", "hand_authored_b",
                     "hand_authored_a", "hand_authored_b"]
