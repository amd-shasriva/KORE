"""Retention contract for agentic trajectories.

A filter bug here is silent in both directions. Too loose and the mixture teaches
the model to spend eight turns not improving, or to reproduce a measurement
exploit. Too tight and a campaign that cost node-days yields nothing. Neither
shows up as an error, so the thresholds and the reasons they fire are pinned.
"""

from __future__ import annotations

import pytest

from kore.data.agentic_filter import (
    FilterPolicy,
    analyze,
    classify,
    revision_gains,
    speedup_distribution,
    suggest_max_speedup,
)


def _bench(turn, speedup, *, correct=True, ok=True):
    return {
        "turn": turn,
        "name": "bench",
        "result": {"tool": "bench", "correct": correct, "ok": ok, "speedup": speedup},
    }


def _record(trace, *, category="success", success=True, turns=8, text="x" * 100):
    return {
        "task_id": "gemm_bf16",
        "success": success,
        "messages": [{"role": "assistant", "content": text}],
        "tool_trace": trace,
        "provenance": {"category": category, "turns_used": turns},
    }


# --------------------------------------------------------------------------- #
# Revision extraction
# --------------------------------------------------------------------------- #
def test_revision_gains_counts_only_frontier_pushes():
    record = _record([
        _bench(0, 0.40),
        _bench(1, 0.38),   # regression, not a revision worth learning
        _bench(2, 0.80),   # 2x push
        _bench(3, 0.81),   # +1.2%, below the ratio
    ])
    gains = revision_gains(record, ratio=1.05)
    assert [g["turn"] for g in gains] == [2]
    assert gains[0]["from"] == 0.40
    assert gains[0]["to"] == 0.80
    assert gains[0]["ratio"] == 2.0


def test_revision_gains_ignores_incorrect_benches():
    # A faster wrong kernel is not an improvement; counting it would train the
    # model to trade correctness for speed, which the oracle exists to prevent.
    record = _record([_bench(0, 0.5), _bench(1, 9.0, correct=False)])
    assert revision_gains(record) == []


def test_revision_gains_survive_a_lineage_reseed():
    # After a reseed the executor's own frontier bookkeeping restarts, so the
    # gains are recomputed here from raw speedups rather than trusted from the
    # record's improved_frontier flags.
    record = _record([_bench(0, 0.4), _bench(1, 1.2), _bench(2, 0.5), _bench(3, 2.0)])
    assert [g["ratio"] for g in revision_gains(record)] == [3.0, pytest.approx(1.6667, abs=1e-3)]


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def test_gain_is_measured_inside_the_episode():
    stats = analyze(_record([_bench(0, 0.35), _bench(1, 0.80)]))
    assert stats.first_correct_speedup == 0.35
    assert stats.best_speedup == 0.80
    assert stats.gain == pytest.approx(0.80 / 0.35)
    assert stats.n_correct_benches == 2


def test_vendor_grade_is_read_from_the_best_bench_not_the_last():
    stats = analyze(_record([_bench(0, 3.0, ok=True), _bench(1, 1.1, ok=False)]))
    assert stats.best_speedup == 3.0
    assert stats.best_bench_vendor_grade is True


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
def test_a_real_improvement_is_kept():
    reason, stats = classify(_record([_bench(0, 0.35), _bench(1, 0.80)]))
    assert reason is None
    assert stats.n_high_gain_revisions == 1


def test_a_trajectory_that_never_improved_is_dropped():
    # Correct throughout, ran the full eight turns, and moved nothing. This is
    # the behaviour being trained out, so it must not be in the mixture.
    reason, _ = classify(_record([_bench(0, 0.41), _bench(1, 0.42)]))
    assert reason == "low_gain"


def test_vendor_parity_is_kept_even_without_in_episode_gain():
    # Arriving correct and competitive on the first try is also worth imitating.
    reason, _ = classify(_record([_bench(0, 1.4)]))
    assert reason is None


def test_attempts_that_never_reached_correctness_are_dropped():
    reason, _ = classify(_record([], category="attempt", success=False))
    assert reason == "not_useful_category"


def test_repair_trajectories_are_retained():
    reason, _ = classify(
        _record([_bench(0, 0.3), _bench(1, 0.9)], category="repair"))
    assert reason is None


def test_correct_but_never_benched_is_dropped():
    reason, _ = classify(_record([{"turn": 0, "name": "test",
                                   "result": {"correct": True, "ok": True}}]))
    assert reason == "no_measured_speedup"


# --------------------------------------------------------------------------- #
# Reward hacking
# --------------------------------------------------------------------------- #
def test_implausible_speedup_is_rejected():
    # The 1541x seen in NVIDIA-sourced data is a measurement being gamed, not a
    # fused kernel. Against a production vendor baseline it is not even arguable.
    reason, _ = classify(_record([_bench(0, 1.0), _bench(1, 1541.94)]))
    assert reason == "implausible_speedup"


def test_untrusted_timing_is_rejected():
    # The executor clears ok when the timing protocol was not admissible for a
    # published speedup; a number measured that way cannot support retention.
    reason, _ = classify(_record([_bench(0, 0.5), _bench(1, 3.0, ok=False)]))
    assert reason == "timing_not_vendor_grade"


def test_untrusted_timing_can_be_admitted_explicitly():
    policy = FilterPolicy(require_vendor_grade=False)
    reason, _ = classify(_record([_bench(0, 0.5), _bench(1, 3.0, ok=False)]), policy)
    assert reason is None


def test_overlong_trajectories_are_dropped_rather_than_truncated():
    reason, _ = classify(_record([_bench(0, 0.5), _bench(1, 2.0)], text="x" * 200_000))
    assert reason == "too_long"


# --------------------------------------------------------------------------- #
# Cap derivation
# --------------------------------------------------------------------------- #
def test_cap_is_derived_from_the_observed_distribution():
    # A realistic bulk against a vendor baseline plus one absurd outlier: the cap
    # has to sit above the bulk and below the outlier, without being told which
    # is which.
    bulk = [0.3 + 0.01 * i for i in range(400)] + [1.0 + 0.02 * i for i in range(400)]
    evidence = suggest_max_speedup(bulk + [1541.94])
    assert evidence["cap"] >= max(bulk)
    assert evidence["cap"] < 1541.94
    assert evidence["tail_ratio_max_over_p99"] > 100


def test_cap_has_a_floor_so_a_narrow_pilot_cannot_over_tighten_it():
    # Twenty samples that all cluster near 1x must not produce a cap that would
    # reject a genuine 4x win in production.
    evidence = suggest_max_speedup([1.0] * 20)
    assert evidence["cap"] >= 5.0


def test_distribution_of_nothing_is_reported_not_invented():
    assert speedup_distribution([]) == {"n": 0}
    assert suggest_max_speedup([])["basis"] == "no observations"
