"""Exact accounting and resume tests for BudgetLedgerV1."""

from __future__ import annotations

import json

import pytest

from kore.policy.budget import (
    BudgetError,
    BudgetExceededError,
    BudgetLedgerV1,
    BudgetLimitsV1,
)


def test_every_budget_dimension_is_separate():
    ledger = BudgetLedgerV1()
    ledger.record_generated(101)
    ledger.record_optimizer(73)
    ledger.record_evaluation(
        correctness_calls=7,
        fresh_timed_calls=3,
        replay_hits=11,
        verifier_gpu_seconds=2.5,
        profiler_gpu_seconds=0.75,
    )
    ledger.record_groups(attempted=5, kept=3)
    ledger.record_feature("starpo_s", 2)
    state = ledger.to_dict()
    assert state["generated_tokens"] == 101
    assert state["optimizer_tokens"] == 73
    assert state["correctness_calls"] == 7
    assert state["fresh_timed_calls"] == 3
    assert state["replay_hits"] == 11
    assert state["verifier_gpu_seconds"] == 2.5
    assert state["profiler_gpu_seconds"] == 0.75
    assert state["groups_attempted"] == 5
    assert state["groups_kept"] == 3
    assert state["feature_invocations"] == {"starpo_s": 2}


def test_replay_hit_does_not_imply_a_physical_call():
    ledger = BudgetLedgerV1()
    ledger.record_evaluation(replay_hits=4)
    assert ledger.replay_hits == 4
    assert ledger.correctness_calls == 0
    assert ledger.fresh_timed_calls == 0
    assert ledger.verifier_gpu_seconds == 0.0


def test_timed_and_correctness_calls_must_be_reported_explicitly():
    ledger = BudgetLedgerV1()
    ledger.record_evaluation(fresh_timed_calls=1)
    assert ledger.fresh_timed_calls == 1
    assert ledger.correctness_calls == 0
    ledger.record_evaluation(correctness_calls=1)
    assert ledger.fresh_timed_calls == 1
    assert ledger.correctness_calls == 1


@pytest.mark.parametrize(
    "limits",
    [
        {"generated_tokens": -1},
        {"correctness_calls": 1.5},
        {"verifier_gpu_seconds": float("inf")},
        {"profiler_gpu_seconds": float("nan")},
        {"unknown": 1},
    ],
)
def test_invalid_limits_rejected(limits):
    with pytest.raises(BudgetError):
        BudgetLimitsV1.from_mapping(limits)


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("record_generated", (-1,)),
        ("record_optimizer", (1.2,)),
        ("record_groups", ()),
        ("record_feature", ("", 1)),
        ("record_feature", ("starpo_s", -1)),
    ],
)
def test_invalid_counter_updates_rejected(method, args):
    ledger = BudgetLedgerV1()
    if method == "record_groups":
        with pytest.raises(BudgetError):
            ledger.record_groups(kept=1)
    else:
        with pytest.raises(BudgetError):
            getattr(ledger, method)(*args)


def test_limit_exceeded_is_atomic():
    ledger = BudgetLedgerV1(limits={"generated_tokens": 10})
    ledger.record_generated(8)
    with pytest.raises(BudgetExceededError, match="generated_tokens"):
        ledger.record_generated(3)
    assert ledger.generated_tokens == 8


def test_exact_state_roundtrip_digest_and_atomic_file(tmp_path):
    ledger = BudgetLedgerV1(
        limits={"generated_tokens": 1000, "verifier_gpu_seconds": 10.0}
    )
    ledger.record_generated(99)
    ledger.record_optimizer(50)
    ledger.record_evaluation(correctness_calls=4, replay_hits=2)
    ledger.record_groups(attempted=3, kept=2)
    ledger.record_feature("avspo")
    before = ledger.digest()
    restored = BudgetLedgerV1.from_dict(ledger.to_dict())
    assert restored.to_dict() == ledger.to_dict()
    assert restored.digest() == before

    path = ledger.write_json(tmp_path / "budget.json")
    assert BudgetLedgerV1.from_dict(json.loads(path.read_text())).digest() == before
    ledger.write_json(path)
    assert BudgetLedgerV1.from_dict(json.loads(path.read_text())).digest() == before


def test_merge_sums_physical_rank_counters_and_feature_counts():
    rank0 = BudgetLedgerV1()
    rank0.record_generated(10)
    rank0.record_optimizer(4)
    rank0.record_groups(attempted=2, kept=1)
    rank0.record_feature("dynamic_sampling")
    rank1 = BudgetLedgerV1()
    rank1.record_generated(12)
    rank1.record_optimizer(5)
    rank1.record_feature("dynamic_sampling", 2)
    merged = BudgetLedgerV1.merge([rank0, rank1])
    assert merged.generated_tokens == 22
    assert merged.optimizer_tokens == 9
    assert merged.groups_attempted == 2
    assert merged.groups_kept == 1
    assert merged.feature_count("dynamic_sampling") == 3


def test_merge_rejects_incompatible_limits():
    with pytest.raises(BudgetError, match="different hard limits"):
        BudgetLedgerV1.merge(
            [
                BudgetLedgerV1(limits={"generated_tokens": 10}),
                BudgetLedgerV1(limits={"generated_tokens": 20}),
            ]
        )


def test_malformed_resume_state_rejected():
    state = BudgetLedgerV1().to_dict()
    state["schema_version"] = "BudgetLedgerV0"
    with pytest.raises(BudgetError, match="unsupported"):
        BudgetLedgerV1.from_dict(state)
    state = BudgetLedgerV1().to_dict()
    state["extra"] = 1
    with pytest.raises(BudgetError, match="unknown"):
        BudgetLedgerV1.from_dict(state)
