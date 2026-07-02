"""CPU-only tests for the reconstructed reward + bench-stats modules."""

from __future__ import annotations

import math

from kore.config import CONFIG
from kore.reward.reward import Observation, RewardResult, compute_reward, scan_for_hacks
from kore.reward import stats


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def test_median_and_cv():
    assert stats.median([3.0, 1.0, 2.0]) == 2.0
    assert stats.median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert stats.cv_pct([10.0, 10.0, 10.0]) == 0.0
    assert stats.cv_pct([10.0, 12.0]) > 0.0


def test_significance_needs_gap_above_noise():
    # 50% faster, tiny noise -> significant.
    assert stats.speedup_is_significant(1.0, 0.5, 0.001, 0.001) is True
    # 1% faster, within noise floor -> not significant.
    assert stats.speedup_is_significant(1.0, 0.99, 0.02, 0.02) is False


# --------------------------------------------------------------------------- #
# anti-hack scanner
# --------------------------------------------------------------------------- #
def test_scan_flags_aiter_and_torch_delegation():
    assert scan_for_hacks("import aiter\nout = aiter.rms_norm(x)") is not None
    assert scan_for_hacks("y = torch.matmul(a, b)") is not None
    assert scan_for_hacks("try:\n    k()\nexcept Exception:\n    return ref") is not None


def test_scan_ignores_comments_and_docstrings():
    clean = (
        '"""This kernel matches aiter.rms_norm layout for MI300."""\n'
        "import triton\nimport triton.language as tl\n"
        "@triton.jit\ndef k():  # not calling torch.matmul here\n    pass\n"
    )
    assert scan_for_hacks(clean) is None


# --------------------------------------------------------------------------- #
# lexicographic reward tiers
# --------------------------------------------------------------------------- #
def test_reward_compile_fail():
    rr = compute_reward(Observation(compiled=False, error_text="boom"), "x=1")
    assert isinstance(rr, RewardResult)
    assert rr.correct is False and rr.reward == CONFIG.reward_compile_fail
    assert "compile_fail" in rr.flags


def test_reward_incorrect_when_snr_below_threshold():
    obs = Observation(compiled=True, validation_passed=True,
                      snr_by_shape={"s": 10.0})  # below bf16 threshold (25)
    rr = compute_reward(obs, "x=1", dtype="bf16")
    assert rr.correct is False and rr.tier == "incorrect"


def test_reward_correct_timed_worst_shape_speedup():
    obs = Observation(
        compiled=True, validation_passed=True,
        snr_by_shape={"a": 90.0, "b": 90.0},
        wall_by_shape={"a": 0.5, "b": 2.0},
        baseline_by_shape={"a": 1.0, "b": 3.0},  # speedups 2x and 1.5x -> worst 1.5x
    )
    rr = compute_reward(obs, "x=1", dtype="bf16")
    assert rr.correct is True and rr.tier == "correct_timed"
    assert abs(rr.speedup - 1.5) < 1e-9
    assert abs(rr.reward - (CONFIG.correctness_weight + math.log(1.5))) < 1e-9


def test_reward_hack_beats_nothing():
    obs = Observation(compiled=True, validation_passed=True, snr_by_shape={"s": 99.0},
                      wall_by_shape={"s": 0.1}, baseline_by_shape={"s": 1.0})
    rr = compute_reward(obs, "import aiter\nout = aiter.rms_norm(x)", dtype="bf16")
    assert rr.correct is False and "hack" in rr.flags
    assert rr.reward == CONFIG.reward_compile_fail


def test_excessive_speedup_flagged_and_capped():
    obs = Observation(compiled=True, validation_passed=True, snr_by_shape={"s": 99.0},
                      wall_by_shape={"s": 0.01}, baseline_by_shape={"s": 1.0})  # 100x
    rr = compute_reward(obs, "x=1", dtype="bf16")
    assert "excessive_speedup" in rr.flags
    assert abs(rr.reward - (CONFIG.correctness_weight + math.log(CONFIG.excessive_speedup_flag))) < 1e-9
