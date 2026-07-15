"""CPU-only tests for the KORE eval module (no GPU required)."""

from __future__ import annotations

import math

from kore.reward.reward import Observation
from kore.eval.fastp import (
    fastp, fast_p_curve, geometric_mean_speedup, pass_at_k, fast_p_at_k, mean_ci,
)
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


def _obs_high_variance() -> Observation:
    # correct + would-be 10x, but the timing is NOISY (cv far above threshold): the
    # integrity gate damps it to <=1x so it can't farm fast_p.
    return Observation(
        compiled=True, snr_db=90.0, wall_ms=0.1, baseline_ms=1.0,
        wall_by_shape={"s": 0.1}, baseline_by_shape={"s": 1.0},
        snr_by_shape={"s": 90.0}, validation_passed=True, cv_pct=99.0,
    )


def _obs_excessive() -> Observation:
    # correct but an implausible ~1000x ratio (measurement artifact) -> capped.
    return Observation(
        compiled=True, snr_db=90.0, wall_ms=0.001, baseline_ms=1.0,
        wall_by_shape={"s": 0.001}, baseline_by_shape={"s": 1.0},
        snr_by_shape={"s": 90.0}, validation_passed=True,
    )


def _benign_policy(task, feedback):
    # No anti-hacking triggers (no torch.nn / aiter / try: / pass-inheritance).
    return "def matmul(a, b):\n    return real_kernel(a, b)\n"


def test_bakeoff_fastp_gates_noisy_and_excessive_speedups():
    """audit R2 soup-eval C2: fast_p must run on the timing-INTEGRITY-gated speedup,
    not the raw ratio -- a noisy (high-cv) bench is damped to <=1x and an excessive
    measurement artifact is capped, so neither can farm the headline metric."""
    from kore.config import CONFIG
    cap = float(CONFIG.excessive_speedup_flag)

    hv = bakeoff.evaluate_policy(_benign_policy, ["t1"], budget=1,
                                 dry_run={"t1": [_obs_high_variance()]})
    assert hv["num_correct"] == 1          # still counts as CORRECT
    assert hv["fast_p"][1.0] == 0.0        # but damped -> cannot farm fast_p with noise

    ex = bakeoff.evaluate_policy(_benign_policy, ["t1"], budget=1,
                                 dry_run={"t1": [_obs_excessive()]})
    assert ex["per_task"][0]["best_speedup"] == cap   # capped, NOT the raw ~1000x
    assert ex["fast_p"][1.0] == 1.0                    # a capped 10x still beats 1x honestly


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


# --- pass@k / fast_p@k (unbiased best-of-N estimators) ----------------------

def test_pass_at_k_matches_closed_form():
    # n=4, c=2, k=1 -> 0.5 ; k=2 -> 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert abs(pass_at_k(4, 2, 1) - 0.5) < 1e-12
    assert abs(pass_at_k(4, 2, 2) - (1.0 - 1.0 / 6.0)) < 1e-12
    # boundary conditions
    assert pass_at_k(4, 0, 2) == 0.0          # no successes
    assert pass_at_k(4, 4, 1) == 1.0          # all succeed
    assert pass_at_k(4, 3, 2) == 1.0          # n-c < k -> every draw hits
    assert pass_at_k(0, 0, 1) == 0.0          # degenerate
    assert pass_at_k(5, 2, 10) == pass_at_k(5, 2, 5)  # k clamped to n


def test_pass_at_k_monotonic_in_k():
    vals = [pass_at_k(8, 3, k) for k in range(1, 9)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-12


def test_fast_p_at_k_requires_correct_and_fast():
    # 4 samples: 2 correct+2x, 1 correct+slow, 1 wrong. At p=1.0 only the 2x count.
    is_correct = [True, True, True, False]
    actual = [1.0, 1.0, 4.0, 1.0]     # third is slower than baseline
    baseline = [2.0, 2.0, 2.0, 2.0]
    # c(success at p=1) = 2 -> fast_1@1 = pass_at_k(4, 2, 1) = 0.5
    assert abs(fast_p_at_k(is_correct, baseline, actual, 1, 1.0) - 0.5) < 1e-12
    # at p=1.0 with k=2 -> 1 - C(2,2)/C(4,2)
    assert abs(fast_p_at_k(is_correct, baseline, actual, 2, 1.0) - (1 - 1 / 6)) < 1e-12


def test_mean_ci_basic():
    mc = mean_ci([0.5, 0.5, 0.5])
    assert abs(mc["mean"] - 0.5) < 1e-12 and mc["ci95"] == 0.0 and mc["n"] == 3
    mc2 = mean_ci([0.0, 1.0])
    assert mc2["n"] == 2 and mc2["ci95"] > 0 and mc2["lo"] < mc2["mean"] < mc2["hi"]
    assert mean_ci([])["n"] == 0


# --- torch-eager second fast_p curve ----------------------------------------

def _obs_2x_with_torch(torch_ms: float = 3.0) -> Observation:
    o = _obs_2x()                       # 2x vs production baseline (baseline_ms=1.0)
    o.torch_baseline_ms = torch_ms      # torch-eager is torch_ms x the production time
    return o


def test_torch_eager_second_curve():
    tasks = ["t1", "t2"]
    dry = {"t1": [_obs_2x_with_torch(3.0)], "t2": [_obs_2x_with_torch(3.0)]}
    res = bakeoff.evaluate_policy(_benign_policy, tasks, budget=1, dry_run=dry)
    # speedup vs torch = (torch/prod) * prod_speedup = 3 * 2 = 6
    assert abs(res["geometric_mean_speedup_vs_torch"] - 6.0) < 1e-6
    assert res["fast_p_vs_torch"][2.0] == 1.0     # 6 > 2 (unlike production, 2 not > 2)
    assert res["fast_p"][2.0] == 0.0
    md = report.format_fastp_report(res)
    assert "vs torch-eager" in md


def test_torch_curve_absent_without_torch_times():
    tasks = ["t1"]
    dry = {"t1": [_obs_2x()]}            # no torch_baseline_ms attribute
    res = bakeoff.evaluate_policy(_benign_policy, tasks, budget=1, dry_run=dry)
    assert "fast_p_vs_torch" not in res


# --- multi-seed fast_p with CI ----------------------------------------------

def test_multiseed_fastp_mean_ci():
    tasks = ["t1", "t2"]

    def seed_dry(sd):
        if sd == 0:
            return {"t1": [_obs_2x()], "t2": [_obs_2x()]}
        return {"t1": [_obs_2x()], "t2": [_obs_incorrect_slow()]}

    agg = bakeoff.evaluate_policy_multiseed(
        _benign_policy, tasks, seeds=[0, 1, 2], budget=1, seed_dry_run=seed_dry,
    )
    assert agg["num_seeds"] == 3
    mc = agg["fast_p_mean_ci"][1.0]
    # seed0 -> 1.0, seeds 1&2 -> 0.5 => mean = (1.0+0.5+0.5)/3
    assert abs(mc["mean"] - (2.0 / 3.0)) < 1e-9
    assert mc["ci95"] > 0
    md = report.render_markdown(agg)
    assert "multi-seed" in md and "CI95" in md


def _obs_incorrect_slow() -> Observation:
    # correct=False path: SNR below the bf16 gate -> not counted correct
    return Observation(
        compiled=True, snr_db=5.0, wall_ms=0.5, baseline_ms=1.0,
        wall_by_shape={"s": 0.5}, baseline_by_shape={"s": 1.0},
        snr_by_shape={"s": 5.0}, validation_passed=True, dtype="bf16",
    )


# --- best-of-N pass@k over a parallel eval ----------------------------------

def test_best_of_n_pass_at_k_report():
    seq = [_obs_2x(), _obs_2x(), _obs_2x()]
    res = bakeoff.evaluate_policy(_benign_policy, ["t1"], budget=3,
                                  mode="parallel", dry_run={"t1": seq})
    pk = bakeoff.best_of_n_pass_at_k(res, ks=[1, 2], ps=[1.0, 2.0])
    assert pk["pass_at_k"][1] == 1.0            # all 3 correct
    assert pk["fast_p_at_k"]["k=1,p=1"] == 1.0  # all 3 correct + 2x > 1
    assert pk["fast_p_at_k"]["k=1,p=2"] == 0.0  # none > 2x
    md = report.render_markdown(pk)
    assert "pass@k" in md


# --- registry train/held-out split ------------------------------------------

def test_registry_heldout_split_is_partition():
    from kore.tasks import registry as reg
    train = {t.task_id for t in reg.train_tasks()}
    held = {t.task_id for t in reg.heldout_tasks()}
    allids = set(reg.task_ids())
    assert train.isdisjoint(held)
    assert train | held == allids
    assert len(held) >= 1
    # >=1 WHOLE operator family reserved (the attention family here)
    assert "attention" in reg.heldout_families()


def test_registry_split_tasks_deterministic_and_stable_heldout():
    from kore.tasks import registry as reg
    s0a = reg.split_tasks(0)
    s0b = reg.split_tasks(0)
    s7 = reg.split_tasks(7)
    order0 = [t.task_id for t in s0a["train"]]
    assert order0 == [t.task_id for t in s0b["train"]]      # deterministic per seed
    # held-out membership is FIXED regardless of seed (training never sees it)
    held0 = {t.task_id for t in s0a["heldout"]}
    held7 = {t.task_id for t in s7["heldout"]}
    assert held0 == held7 == {t.task_id for t in reg.heldout_tasks()}


def test_registry_operator_family_known_ops():
    from kore.tasks import registry as reg
    fam = {t.task_id: reg.operator_family(t) for t in reg.all_tasks()}
    assert fam.get("rmsnorm_aiter") == "rmsnorm"
    assert fam.get("gemm_bf16") == "gemm"
    assert fam.get("flash_attn_decode_bf16") == "attention"
