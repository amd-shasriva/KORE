"""CPU-only tests for the KORE-Bench report assembler (dry-run)."""

from __future__ import annotations

from kore.eval.korebench import run_korebench, data_scale_summary, format_report
from kore.reward.reward import Observation


def _obs(su):
    return Observation(compiled=True, validation_passed=True, snr_by_shape={"s": 99.0},
                       wall_by_shape={"s": 1.0 / su}, baseline_by_shape={"s": 1.0})


def test_korebench_dry_run_report():
    tasks = ["gemm_bf16", "gen_silu_bf16", "gen_relu_fp16"]
    # gemm fast (1.5x worst), silu just-beats (1.1x), relu slower (0.8x)
    dry = {"gemm_bf16": [_obs(1.5)], "gen_silu_bf16": [_obs(1.1)], "gen_relu_fp16": [_obs(0.8)]}
    rep = run_korebench(lambda task, fb: "k", tasks, dry_run=dry, budget=1)
    assert rep["n_tasks"] == 3
    assert rep["correct_rate"] == 1.0
    # 2 of 3 beat the baseline on the worst shape (1.5x, 1.1x); 0.8x does not
    assert abs(rep["worst_shape_win_rate_vs_baseline"] - 2 / 3) < 1e-9
    assert rep["timing_integrity_complete"] is True
    assert "gemm" in rep["per_family"] or "activation" in rep["per_family"]
    # formatting works
    assert "WORST-SHAPE win-rate" in format_report(rep)


def test_data_scale_summary_wide():
    d = data_scale_summary()
    assert d["operators"] >= 100
    assert "attention" in d["heldout_families"]
