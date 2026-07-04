"""CPU-only tests for stratified RFT (>1x rejection sampling) selection."""

from __future__ import annotations

from kore.data.rejection import (
    passes_win_filter,
    stratified_rft_select,
    task_entropy,
)
from kore.data.schemas import RepairRecord, WinRecord


def _win(task, speedup, snr=40.0, src=None, tid_src=None):
    return WinRecord(
        task_id=task, trajectory=[{"role": "assistant", "content": "k"}],
        initial_wall_us=100.0, final_wall_us=50.0, speedup=speedup,
        final_source=src or f"def k_{task}_{speedup}_{tid_src or ''}(): pass",
        snr_db=snr,
    )


def test_filter_keeps_only_fast_correct_wins():
    assert passes_win_filter(_win("t", 1.2)) is True
    assert passes_win_filter(_win("t", 0.9)) is False        # slower than baseline
    assert passes_win_filter(_win("t", 1.2, snr=10.0), min_snr=25.0) is False  # incorrect
    assert passes_win_filter(RepairRecord("t", "snr_fail", "h", "e", [])) is False


def test_filter_custom_tau():
    r = _win("t", 1.1)
    assert passes_win_filter(r, tau=1.0) is True
    assert passes_win_filter(r, tau=1.2) is False


def test_dedup_keeps_fastest_identical_source():
    recs = [_win("t", 1.1, src="def k(): pass  # v1"),
            _win("t", 1.9, src="def k(): pass  # v2"),  # same code up to comments
            _win("t", 1.5, src="def k(): pass")]
    kept, report = stratified_rft_select(recs, tau=1.0)
    assert report.n_pass_filter == 3
    assert report.n_after_dedup == 1           # all normalize to the same source
    assert kept[0].speedup == 1.9              # fastest instance retained


def test_stratification_prevents_single_task_domination():
    # One easy task floods 20 wins; 6 other tasks have plenty too. With adequate
    # diversity the per-task cap must hold so 'easy' can't dominate the kept set.
    recs = [_win("easy", 1.0 + i * 0.01, src=f"e{i}") for i in range(20)]
    for t in range(6):
        recs += [_win(f"task{t}", 1.5, src=f"t{t}_{j}") for j in range(5)]
    kept, report = stratified_rft_select(recs, tau=1.0, max_total=21,
                                         per_task_frac_cap=0.20)
    assert report.n_kept == 21
    cap = int(0.20 * 21)  # = 4
    assert report.per_task["easy"] <= cap        # easy task is capped
    assert len(report.per_task) == 7             # every task represented
    assert report.task_entropy > 0.9             # near-uniform => high diversity


def test_entropy_bounds():
    assert task_entropy({}) == 0.0
    assert task_entropy({"a": 10}) == 1.0
    assert task_entropy({"a": 5, "b": 5}) == 1.0
    assert 0.0 < task_entropy({"a": 9, "b": 1}) < 1.0


def test_relaxes_cap_when_few_tasks_to_fill_budget():
    # only one task available: cap would starve, but budget should still fill
    recs = [_win("solo", 1.0 + i * 0.05, src=f"s{i}") for i in range(10)]
    kept, report = stratified_rft_select(recs, tau=1.0, max_total=8,
                                         per_task_frac_cap=0.34)
    assert report.n_kept == 8  # cap relaxed since no other task to balance with
