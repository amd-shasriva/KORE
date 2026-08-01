"""Exact accounting and resume tests for BudgetLedgerV1."""

from __future__ import annotations

import json

import pytest

from kore.policy.budget import (
    BudgetError,
    BudgetExceededError,
    BudgetLedgerV1,
    BudgetLimitsV1,
    EvaluationWork,
    charge_evaluation_work,
    check_evaluation_budget,
)
from kore.reward.reward import Observation

_EVALUATION_COUNTERS = (
    "correctness_calls",
    "fresh_timed_calls",
    "replay_hits",
    "verifier_gpu_seconds",
    "profiler_gpu_seconds",
)


def _timed_observation() -> Observation:
    """A correct candidate that was verified and then benchmarked."""
    return Observation(
        compiled=True,
        dtype="bf16",
        validation_passed=True,
        timing_requested=True,
        snr_by_shape={"primary": 61.0},
        wall_by_shape={"primary": 1.5},
        baseline_by_shape={"primary": 2.5},
    )


def _correctness_only_observation() -> Observation:
    """A correct candidate evaluated with ``do_bench=False``."""
    return Observation(
        compiled=True,
        dtype="bf16",
        validation_passed=True,
        snr_by_shape={"primary": 61.0},
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


def test_the_evaluation_path_produces_every_evaluation_counter():
    ledger = BudgetLedgerV1()
    # A verified-and-benchmarked evaluation.
    charge_evaluation_work(ledger, EvaluationWork.from_observation(
        _timed_observation(), verifier_seconds=31.5, profiler_seconds=4.25))
    # A correctness-only evaluation on the same ledger.
    charge_evaluation_work(ledger, EvaluationWork.from_observation(
        _correctness_only_observation(), verifier_seconds=8.0))
    # A cache-served evaluation.
    charge_evaluation_work(ledger, EvaluationWork.from_observation(
        _timed_observation(), verifier_seconds=0.001, replayed=True))

    assert ledger.correctness_calls == 2
    assert ledger.fresh_timed_calls == 1
    assert ledger.replay_hits == 1
    assert ledger.verifier_gpu_seconds == pytest.approx(39.5)
    assert ledger.profiler_gpu_seconds == pytest.approx(4.25)
    # None of the evaluation counters bleeds into the token/group dimensions.
    assert ledger.generated_tokens == 0
    assert ledger.optimizer_tokens == 0
    assert ledger.groups_attempted == 0


def test_observed_work_never_derives_one_counter_from_another():
    # Correctness work does not imply timing work...
    correctness = EvaluationWork.from_observation(
        _correctness_only_observation(), verifier_seconds=3.0)
    assert correctness.correctness_calls == 1
    assert correctness.fresh_timed_calls == 0
    assert correctness.profiler_gpu_seconds == 0.0

    # ...a driver refused as performance-ineligible never reached the benchmark...
    ineligible = Observation(compiled=True, validation_passed=True,
                             timing_requested=True, timing_grade="ineligible")
    assert EvaluationWork.from_observation(
        ineligible, verifier_seconds=3.0).fresh_timed_calls == 0

    # ...but a benchmark whose measurements were rejected still spent the GPU.
    rejected = Observation(compiled=True, validation_passed=True,
                           timing_requested=True, infra_error=True,
                           timing_grade="rejected")
    assert EvaluationWork.from_observation(
        rejected, verifier_seconds=3.0).fresh_timed_calls == 1

    # A replay hit performs none of it, whatever the observation carries.
    replayed = EvaluationWork.from_observation(
        _timed_observation(), verifier_seconds=12.0, profiler_seconds=1.0,
        replayed=True)
    assert replayed.to_dict() == {
        "correctness_calls": 0, "fresh_timed_calls": 0, "replay_hits": 1,
        "verifier_gpu_seconds": 0.0, "profiler_gpu_seconds": 0.0}

    # An evaluation rejected before any subprocess launched charges nothing.
    assert EvaluationWork.from_observation(
        _timed_observation(), verifier_seconds=0.0, executed=False).is_empty


@pytest.mark.parametrize(
    "work",
    [
        {"replay_hits": 1, "correctness_calls": 1},
        {"replay_hits": 1, "verifier_gpu_seconds": 1.0},
        {"verifier_gpu_seconds": 2.0},
        {"profiler_gpu_seconds": 2.0},
        {"correctness_calls": -1},
        {"fresh_timed_calls": 1.5},
        {"correctness_calls": True},
        {"verifier_gpu_seconds": float("inf")},
    ],
)
def test_dishonest_evaluation_records_rejected(work):
    with pytest.raises(BudgetError):
        EvaluationWork(**work)


@pytest.mark.parametrize("counter", _EVALUATION_COUNTERS)
def test_a_zero_limit_on_any_evaluation_counter_binds(counter):
    work = (
        EvaluationWork(replay_hits=1) if counter == "replay_hits"
        else EvaluationWork.from_observation(
            _timed_observation(), verifier_seconds=2.0, profiler_seconds=1.0)
    )
    ledger = BudgetLedgerV1(limits={counter: 0})
    # Pre-flight refuses the work before any compute is spent...
    with pytest.raises(BudgetExceededError, match=counter):
        check_evaluation_budget(ledger, work)
    # ...and the charge itself is atomic, so nothing lands.
    with pytest.raises(BudgetExceededError, match=counter):
        charge_evaluation_work(ledger, work)
    assert all(getattr(ledger, name) == 0 for name in _EVALUATION_COUNTERS)


def test_evaluation_limits_admit_work_up_to_the_cap():
    ledger = BudgetLedgerV1(
        limits={"correctness_calls": 2, "verifier_gpu_seconds": 10.0})
    work = EvaluationWork.from_observation(
        _correctness_only_observation(), verifier_seconds=5.0)
    check_evaluation_budget(ledger, work)
    charge_evaluation_work(ledger, work)
    charge_evaluation_work(ledger, work)
    assert ledger.correctness_calls == 2
    assert ledger.verifier_gpu_seconds == pytest.approx(10.0)
    with pytest.raises(BudgetExceededError, match="correctness_calls"):
        check_evaluation_budget(ledger, work)


def test_an_unbudgeted_evaluation_path_is_a_no_op():
    work = EvaluationWork.from_observation(
        _timed_observation(), verifier_seconds=1.0)
    assert check_evaluation_budget(None, work) is work
    assert charge_evaluation_work(None, work) is work


def test_evaluation_work_must_be_a_typed_record():
    ledger = BudgetLedgerV1()
    for bad in ({"correctness_calls": 1}, 1, None):
        with pytest.raises(BudgetError, match="EvaluationWork"):
            ledger.record_evaluation_work(bad)
        with pytest.raises(BudgetError, match="EvaluationWork"):
            ledger.check_evaluation(bad)


def test_evaluation_counters_merge_across_ranks_and_resume_exactly(tmp_path):
    limits = {"correctness_calls": 10}
    ranks = []
    for seconds in (4.0, 6.5):
        rank = BudgetLedgerV1(limits=limits)
        charge_evaluation_work(rank, EvaluationWork.from_observation(
            _timed_observation(), verifier_seconds=seconds,
            profiler_seconds=0.5))
        charge_evaluation_work(rank, EvaluationWork(replay_hits=3))
        ranks.append(rank)
    merged = BudgetLedgerV1.merge(ranks)
    assert merged.correctness_calls == 2
    assert merged.fresh_timed_calls == 2
    assert merged.replay_hits == 6
    assert merged.verifier_gpu_seconds == pytest.approx(10.5)
    assert merged.profiler_gpu_seconds == pytest.approx(1.0)

    path = merged.write_json(tmp_path / "budget_ledger.json")
    assert BudgetLedgerV1.from_dict(
        json.loads(path.read_text())).digest() == merged.digest()

    # Cross-rank merges still require identical hard limits.
    ranks[1].limits = BudgetLimitsV1(correctness_calls=11)
    with pytest.raises(BudgetError, match="different hard limits"):
        BudgetLedgerV1.merge(ranks)


def test_malformed_resume_state_rejected():
    state = BudgetLedgerV1().to_dict()
    state["schema_version"] = "BudgetLedgerV0"
    with pytest.raises(BudgetError, match="unsupported"):
        BudgetLedgerV1.from_dict(state)
    state = BudgetLedgerV1().to_dict()
    state["extra"] = 1
    with pytest.raises(BudgetError, match="unknown"):
        BudgetLedgerV1.from_dict(state)
