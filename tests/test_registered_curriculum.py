"""Determinism, fairness, held-out, consensus, and resume tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kore.policy.curriculum import (
    CurriculumError,
    CurriculumStateV1,
    RegisteredStratifiedScheduler,
)


TASKS = {
    "gemm_a": SimpleNamespace(
        task_id="gemm_a", operation="gemm", family="gemm", dtype="bf16", heldout=False
    ),
    "gemm_b": SimpleNamespace(
        task_id="gemm_b", operation="gemm", family="gemm", dtype="bf16", heldout=False
    ),
    "gemm_fp8": SimpleNamespace(
        task_id="gemm_fp8", operation="gemm", family="gemm", dtype="fp8", heldout=False
    ),
    "norm_a": SimpleNamespace(
        task_id="norm_a", operation="norm", family="norm", dtype="bf16", heldout=False
    ),
    "norm_b": SimpleNamespace(
        task_id="norm_b", operation="norm", family="norm", dtype="bf16", heldout=False
    ),
    "held": SimpleNamespace(
        task_id="held", operation="mla", family="mla", dtype="bf16", heldout=True
    ),
}
TRAIN_IDS = ["gemm_a", "gemm_b", "gemm_fp8", "norm_a", "norm_b"]


def _loader(task_id):
    return TASKS[task_id]


def _scheduler(ids=TRAIN_IDS, **kwargs):
    return RegisteredStratifiedScheduler(
        ids,
        task_loader=_loader,
        is_heldout_fn=lambda task: task.heldout,
        operator_family_fn=lambda task: task.family,
        **kwargs,
    )


def test_stratum_fairness_and_per_stratum_task_fairness():
    scheduler = _scheduler(seed=17)
    draws = [scheduler.next_task_id() for _ in range(99)]
    strata = [
        (TASKS[task_id].family, TASKS[task_id].dtype) for task_id in draws
    ]
    counts = {stratum: strata.count(stratum) for stratum in set(strata)}
    assert max(counts.values()) - min(counts.values()) <= 1

    # Every full local epoch visits each task in a stratum exactly once.
    gemm_bf16 = [
        task_id
        for task_id in draws
        if (TASKS[task_id].family, TASKS[task_id].dtype) == ("gemm", "bf16")
    ]
    for start in range(0, len(gemm_bf16) - 1, 2):
        assert set(gemm_bf16[start : start + 2]) == {"gemm_a", "gemm_b"}


def test_sha_counter_sequence_is_deterministic_and_input_order_independent():
    a = _scheduler(TRAIN_IDS, seed=123)
    b = _scheduler(list(reversed(TRAIN_IDS)), seed=123)
    assert a.task_set_digest == b.task_set_digest
    assert [a.next_task_id() for _ in range(50)] == [
        b.next_task_id() for _ in range(50)
    ]
    c = _scheduler(TRAIN_IDS, seed=124)
    assert [a.next_task_id() for _ in range(30)] != [
        c.next_task_id() for _ in range(30)
    ]


def test_empty_duplicate_unknown_and_heldout_rejected():
    with pytest.raises(CurriculumError, match="non-empty"):
        _scheduler([])
    with pytest.raises(CurriculumError, match="unique"):
        _scheduler(["gemm_a", "gemm_a"])
    with pytest.raises(CurriculumError, match="not registered"):
        _scheduler(["missing"])
    with pytest.raises(CurriculumError, match="held-out"):
        _scheduler(["gemm_a", "held"])
    with pytest.raises(CurriculumError, match="overlaps held-out"):
        _scheduler(["gemm_a"], explicit_heldout_ids=["gemm_a"])


def test_exact_resume_suffix_and_state_file(tmp_path):
    uninterrupted = _scheduler(seed=9)
    prefix = [uninterrupted.next_task_id() for _ in range(17)]
    assert len(prefix) == 17
    path = uninterrupted.save_json(tmp_path / "curriculum.json")
    expected_suffix = [uninterrupted.next_task_id() for _ in range(60)]

    resumed = RegisteredStratifiedScheduler.from_json(
        TRAIN_IDS,
        path,
        task_loader=_loader,
        is_heldout_fn=lambda task: task.heldout,
        operator_family_fn=lambda task: task.family,
    )
    assert resumed.draw_index == 17
    assert [resumed.next_task_id() for _ in range(60)] == expected_suffix


def test_resume_rejects_task_digest_seed_and_counter_tampering():
    scheduler = _scheduler(seed=5)
    for _ in range(8):
        scheduler.next_task_id()
    state = scheduler.state_dict()

    with pytest.raises(CurriculumError, match="task-set digest"):
        _scheduler(["gemm_a", "gemm_b"], seed=5, state=state)
    with pytest.raises(CurriculumError, match="seed mismatch"):
        _scheduler(seed=6, state=state)

    bad = dict(state)
    bad["stratum_draw_counts"] = [dict(item) for item in state["stratum_draw_counts"]]
    bad["stratum_draw_counts"][0]["draws"] += 1
    with pytest.raises(CurriculumError, match="counters"):
        _scheduler(seed=5, state=CurriculumStateV1.from_dict(bad))


def test_rank_zero_broadcast_consensus_without_follower_selection():
    schedulers = [_scheduler(seed=77) for _ in range(4)]
    bus = {}

    def root_broadcast(payload, src):
        assert src == 0 and payload is not None
        bus["payload"] = payload
        return payload

    def follower_broadcast(payload, src):
        assert src == 0 and payload is None
        return bus["payload"]

    # If a follower tries to select/reconstruct locally, this sentinel explodes.
    schedulers[1]._task_for_draw = lambda *_: (_ for _ in ()).throw(
        AssertionError("follower made an independent task decision")
    )

    for _ in range(25):
        chosen = [
            schedulers[0].next_for_rank(
                rank=0, world_size=4, broadcast=root_broadcast
            )
        ]
        chosen.extend(
            scheduler.next_for_rank(
                rank=rank, world_size=4, broadcast=follower_broadcast
            )
            for rank, scheduler in enumerate(schedulers[1:], start=1)
        )
        assert len(set(chosen)) == 1
        assert len({scheduler.state().to_dict().__repr__() for scheduler in schedulers}) == 1


def test_rank_local_state_divergence_fails_before_accepting_broadcast():
    root = _scheduler(seed=3)
    follower = _scheduler(seed=3)
    root.next_task_id()  # force divergent pre-draw state
    payload = {}

    def root_broadcast(value, _src):
        payload["value"] = value
        return value

    root.next_for_rank(rank=0, world_size=2, broadcast=root_broadcast)
    with pytest.raises(CurriculumError, match="diverged"):
        follower.next_for_rank(
            rank=1,
            world_size=2,
            broadcast=lambda _value, _src: payload["value"],
        )
