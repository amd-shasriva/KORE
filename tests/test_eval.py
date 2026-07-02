"""CPU-only tests for the KORE eval module (no GPU required)."""

from __future__ import annotations

import math

from kore.reward.reward import Observation
from kore.eval.fastp import fastp, fast_p_curve, geometric_mean_speedup
from kore.eval import bakeoff
from kore.eval import report


# --- fastp.py ---------------------------------------------------------------

def test_fastp_all_correct_all_2x():
    # times: baseline = 2 * actual  => speedup = 2.0 everywhere
    n = 4
    is_correct = [True] * n
    actual = [1.0] * n
    baseline = [2.0] * n
    # 2.0 > 1.0 and 2.0 > 1.5 -> all count; 2.0 is NOT > 2.0 -> none count
    assert fastp(is_correct, baseline, actual, n, 1.0) == 1.0
    assert fastp(is_correct, baseline, actual, n, 1.5) == 1.0
    assert fastp(is_correct, baseline, actual, n, 2.0) == 0.0


def test_fastp_mixed_correctness_reduces():
    n = 4
    is_correct = [True, True, False, False]
    actual = [1.0, 1.0, 1.0, 1.0]
    baseline = [2.0, 2.0, 2.0, 2.0]
    # only the 2 correct ones count at p=1.0 -> 2/4
    assert fastp(is_correct, baseline, actual, n, 1.0) == 0.5


def test_fastp_n_is_uncorrected_denominator():
    # 2 attempted (both correct+fast), but the split has n=5 => 2/5
    is_correct = [True, True]
    actual = [1.0, 1.0]
    baseline = [2.0, 2.0]
    assert fastp(is_correct, baseline, actual, 5, 1.0) == 2.0 / 5.0


def test_geometric_mean_speedup_two_2x():
    is_correct = [True, True]
    actual = [1.0, 1.0]
    baseline = [2.0, 2.0]  # both 2x
    assert abs(geometric_mean_speedup(is_correct, baseline, actual) - 2.0) < 1e-9


def test_geometric_mean_speedup_correct_only():
    is_correct = [True, False]
    actual = [1.0, 0.001]          # the wrong one would be huge if counted
    baseline = [2.0, 2.0]
    assert abs(geometric_mean_speedup(is_correct, baseline, actual) - 2.0) < 1e-9


def test_fast_p_curve_monotonic_non_increasing():
    n = 5
    is_correct = [True, True, True, False, True]
    actual = [1.0, 0.5, 2.0, 1.0, 0.9]
    baseline = [2.0, 2.0, 2.0, 2.0, 2.0]
    curve = fast_p_curve(is_correct, baseline, actual, n)
    vals = [v for _, v in curve]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-12


# --- bakeoff.py (dry_run, fabricated Observations) --------------------------

def _obs_2x() -> Observation:
    return Observation(
        compiled=True, snr_db=90.0, wall_ms=0.5, baseline_ms=1.0,
        wall_by_shape={"s": 0.5}, baseline_by_shape={"s": 1.0},
        snr_by_shape={"s": 90.0}, validation_passed=True,
    )


def _obs_incorrect() -> Observation:
    return Observation(
        compiled=True, snr_db=5.0, wall_ms=0.5, baseline_ms=1.0,
        wall_by_shape={"s": 0.5}, baseline_by_shape={"s": 1.0},
        snr_by_shape={"s": 5.0}, validation_passed=True,
    )


def _benign_policy(task, feedback):
    # No anti-hacking triggers (no torch.nn / aiter / try: / pass-inheritance).
    return "def matmul(a, b):\n    return real_kernel(a, b)\n"


def test_bakeoff_dry_run_per_policy_fastp():
    tasks = ["t1", "t2"]
    good = {"t1": [_obs_2x()], "t2": [_obs_2x()]}
    mixed = {"t1": [_obs_2x()], "t2": [_obs_incorrect()]}

    results = bakeoff.matched_budget_bakeoff(
        {"good": _benign_policy, "mixed": _benign_policy},
        tasks, budget=1, dry_run=good, mode="serial",
    )
    # NOTE: matched_budget_bakeoff uses one dry_run source; use evaluate_policy
    # directly to give each policy its own fabricated observations.
    good_res = bakeoff.evaluate_policy(_benign_policy, tasks, budget=1, dry_run=good)
    mixed_res = bakeoff.evaluate_policy(_benign_policy, tasks, budget=1, dry_run=mixed)

    assert good_res["fast_p"][1.0] == 1.0        # both correct + 2x
    assert good_res["fast_p"][2.0] == 0.0        # 2.0 not > 2.0
    assert mixed_res["fast_p"][1.0] == 0.5       # one correct of two
    assert good_res["num_correct"] == 2
    assert "good" in results["policies"] and "mixed" in results["policies"]


def test_bakeoff_serial_vs_parallel_two_numbers():
    obs_seq = [_obs_incorrect(), _obs_2x(), _obs_2x()]
    dry = {"t1": obs_seq}
    out = bakeoff.serial_vs_parallel(_benign_policy, "t1", total_budget=3, dry_run=dry)
    assert out["serial_best_speedup"] is not None
    assert out["parallel_best_speedup"] is not None
    assert abs(out["serial_best_speedup"] - 2.0) < 1e-6
    assert abs(out["parallel_best_speedup"] - 2.0) < 1e-6


def test_benches_to_best():
    # value model ranks candidate 2 highest; it is also the true best.
    value_scores = [0.1, 0.2, 0.9, 0.3]
    true_speedups = [1.1, 1.2, 3.0, 1.5]
    out = bakeoff.benches_to_best(value_scores, true_speedups)
    assert out["best_idx"] == 2
    assert out["benches_to_best"] == 1
    assert out["random_expected"] == 2.5


# --- report.py --------------------------------------------------------------

def test_report_formatting_non_empty():
    tasks = ["t1", "t2"]
    good = {"t1": [_obs_2x()], "t2": [_obs_2x()]}
    res = bakeoff.evaluate_policy(_benign_policy, tasks, budget=1, dry_run=good)
    md = report.format_fastp_report(res)
    assert isinstance(md, str) and len(md) > 0
    assert "fast_p" in md

    bake = bakeoff.matched_budget_bakeoff(
        {"good": _benign_policy}, tasks, budget=1, dry_run=good,
    )
    table = report.format_bakeoff_table(bake)
    assert isinstance(table, str) and len(table) > 0
    assert "policy" in table


def test_save_report_writes_json_and_md(tmp_path):
    tasks = ["t1"]
    good = {"t1": [_obs_2x()]}
    res = bakeoff.evaluate_policy(_benign_policy, tasks, budget=1, dry_run=good)
    paths = report.save_report(res, tmp_path / "run")
    import os
    assert os.path.exists(paths["json"]) and os.path.exists(paths["md"])


def test_e2e_module_imports():
    from kore.eval import e2e_sglang_vllm as e2e
    assert hasattr(e2e, "e2e_throughput")
    assert hasattr(e2e, "e2e_accuracy")
    assert hasattr(e2e, "Workload")
