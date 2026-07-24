"""CPU concurrency contract for the per-task K-concurrency deepen refactor.

Validates the NEW machinery (collision-safe persistence + trajectory coordinator)
WITHOUT any GPU/teacher: real spawned processes hammer one shard, and the
coordinator's attempt/target bounds are checked directly. The trajectory body
(generate_wins/KoreEnv) is unchanged by the refactor and is covered elsewhere +
the GPU smoke.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import random

import scripts.deepen_wins as dw


def _persist_worker(data_root: str, tid: str, target: int, sources: list, seed: int):
    """Top-level (spawn-picklable): try to persist a shuffled mix of sources."""
    rng = random.Random(seed)
    order = list(sources) + list(sources)  # offer duplicates too
    rng.shuffle(order)
    path = dw._wins_shard(Path(data_root), tid)
    lock = dw._wins_lock(Path(data_root), tid)
    for s in order:
        dw._persist_win(path, lock, {"task_id": tid, "type": "win", "final_source": s}, target)


def _run_procs(data_root, tid, target, sources, n):
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_persist_worker, args=(str(data_root), tid, target, sources, i))
             for i in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0, f"worker exited rc={p.exitcode}"


def test_persist_win_is_collision_safe_and_target_bounded(tmp_path):
    tid = "genb_probe"
    sources = [f"kernel_source_{i}" for i in range(12)]
    target = 5
    _run_procs(tmp_path, tid, target, sources, n=8)

    path = dw._wins_shard(tmp_path, tid)
    existing, seen = dw._load_existing(path)
    # EXACTLY target distinct wins - no lost updates, no overshoot past target.
    assert len(existing) == target, f"expected {target}, got {len(existing)}"
    srcs = [dw._final_source(r) for r in existing]
    assert len(set(srcs)) == target, "duplicate final_source persisted"
    assert set(srcs) <= set(sources), "unknown source persisted"
    # Raw shard is valid JSONL end-to-end (no torn/partial lines from a race).
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == target
    for ln in lines:
        json.loads(ln)


def test_persist_win_preserves_existing_and_resumes(tmp_path):
    tid = "genb_resume"
    path = dw._wins_shard(tmp_path, tid)
    path.parent.mkdir(parents=True, exist_ok=True)
    dw._atomic_write_jsonl(path, [
        {"task_id": tid, "type": "win", "final_source": "orig_a"},
        {"task_id": tid, "type": "win", "final_source": "orig_b"},
    ])
    target = 5
    sources = [f"new_{i}" for i in range(10)]
    _run_procs(tmp_path, tid, target, sources, n=6)

    existing, _ = dw._load_existing(path)
    assert len(existing) == target
    srcs = {dw._final_source(r) for r in existing}
    assert {"orig_a", "orig_b"} <= srcs, "existing wins were lost"


def test_persist_win_already_full_is_zero_cost(tmp_path):
    tid = "genb_full"
    path = dw._wins_shard(tmp_path, tid)
    path.parent.mkdir(parents=True, exist_ok=True)
    dw._atomic_write_jsonl(path, [{"task_id": tid, "type": "win", "final_source": f"s{i}"} for i in range(3)])
    st, have = dw._persist_win(path, dw._wins_lock(tmp_path, tid),
                               {"task_id": tid, "final_source": "extra"}, target=3)
    assert st == "full" and have == 3


def test_coordinator_bounds_attempts_and_stops(tmp_path):
    mgr = mp.Manager()
    tasks_have = {"a": 0, "b": 1, "c": 3}  # c already at target
    target, oversample = 3, 3
    coord = dw.TaskCoordinator(mgr, tasks_have, target, oversample)
    budget = dict(coord.max_attempts)
    rng = random.Random(0)
    guard = 0
    while True:
        tid, status = coord.claim()
        if status in ("DONE",):
            break
        if status == "WAIT":
            break  # single-threaded sim records immediately, so WAIT => truly stuck
        guard += 1
        assert guard < 100000
        coord.record(tid, rng.random() < 0.5)
    # Already-complete task is NEVER scheduled (zero teacher calls).
    assert coord.attempts["c"] == 0
    # Per-task attempts never exceed the oversample budget (no runaway spend).
    for t in ("a", "b"):
        assert coord.attempts[t] <= budget[t], t
        assert coord.inflight[t] == 0
        # Each task either reached target or exhausted its bounded budget.
        assert coord.have[t] >= target or coord.attempts[t] >= budget[t], t


def test_coordinator_allows_k_concurrency_on_one_task(tmp_path):
    """A SINGLE remaining task must hand out multiple concurrent trajectory
    tickets (old model was 1 worker/task). Proves the ceiling is removed."""
    mgr = mp.Manager()
    coord = dw.TaskCoordinator(mgr, {"solo": 0}, target=3, oversample=3)
    gos = 0
    for _ in range(20):
        tid, status = coord.claim()  # never record -> tickets stay in flight
        if status == "GO":
            gos += 1
        else:
            break
    assert gos >= 3, f"expected multi-worker concurrency on one task, got {gos}"
