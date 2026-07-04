"""CPU-only tests for the reconstructed reward + bench-stats modules."""

from __future__ import annotations

import math

from kore.config import CONFIG
from kore.reward.reward import Observation, RewardResult, compute_reward, scan_for_hacks
from kore.reward import stats


def _expected_speed_term(su_scored: float, su_raw: float,
                         significant: bool = True, excessive: bool = False) -> float:
    """Independent re-derivation of the P4 speed reward for assertions."""
    w = CONFIG.speedup_weight
    if CONFIG.speedup_log and su_scored > 1.0:
        term = w * (1.0 + math.log(su_scored))
    else:
        term = w * max(su_scored, 0.0)
    margin = 1.0 + CONFIG.noise_floor_pct / 100.0
    if CONFIG.fast_p_bonus and ((not CONFIG.fast_p_significant_only)
                                or (significant and not excessive)):
        for thr, bonus in CONFIG.fast_p_bonus:
            if su_raw >= thr * margin:
                term += bonus
    return term


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


def test_scan_blocks_post_verdict_and_aliased_delegation():
    # post-verdict forgery channels (beat last-match parsing)
    assert scan_for_hacks("import atexit\natexit.register(lambda: print('x'))") is not None
    assert scan_for_hacks("import signal\nsignal.signal(2, lambda *a: None)") is not None
    assert scan_for_hacks("class K:\n def __del__(self): pass") is not None
    # delegation bypasses
    assert scan_for_hacks("import torch as t\ndef f(x): return t.softmax(x,-1)") is not None
    assert scan_for_hacks("import torch.nn.functional as fn\ndef f(x): return fn.gelu(x)") is not None
    assert scan_for_hacks("def f(x): return x.matmul(x)") is not None
    assert scan_for_hacks("import torch\ndef f(x): return getattr(torch,'mat'+'mul')(x,x)") is not None
    assert scan_for_hacks("import torch\nm=torch.matmul\ndef f(x): return m(x,x)") is not None
    # legitimate Triton must NOT be flagged
    assert scan_for_hacks("import triton.language as tl\n@triton.jit\ndef k():\n    y = tl.sigmoid(x)") is None


def test_scan_blocks_mode_sniffing_and_timing_manipulation():
    """A kernel that detects the bench-vs-correctness split (or manipulates the
    timed region) can fake a speedup while staying 'correct' — must be rejected."""
    # mode-sniffing via the driver CLI
    assert scan_for_hacks("import sys\nif '--bench-mode' in sys.argv:\n    fast_path()") is not None
    assert scan_for_hacks("import argparse\np=argparse.ArgumentParser()") is not None
    assert scan_for_hacks("mode = '--impl'\ndef f(): pass") is not None
    # timing manipulation
    assert scan_for_hacks("import time\ndef f(x):\n    time.sleep(0.0)\n    return x") is not None
    assert scan_for_hacks("torch.cuda.set_sync_debug_mode(0)") is not None
    # forged bench timing line
    assert scan_for_hacks("print('wall_ms: 0.001')") is not None
    # legitimate Triton autotune config must NOT be flagged
    assert scan_for_hacks(
        "import triton\n@triton.autotune(configs=[], key=['M'])\n@triton.jit\ndef k():\n    pass") is None


def test_scan_blocks_operator_and_module_and_fs_bypasses():
    """Bypasses found in audit: @ matmul operator, sys.modules delegation,
    pathlib/chmod filesystem escape, and os.environ mode-sniffing."""
    # @ matmul operator = pure vendor-GEMM delegation
    assert scan_for_hacks("def f(a, b):\n    return a @ b") is not None
    assert scan_for_hacks("def f(a, b):\n    return a@b") is not None
    # module-table access to reach torch/oracle
    assert scan_for_hacks("import sys\nreturn sys.modules['torch'].matmul(a, b)") is not None
    assert scan_for_hacks("import sys\nsys.modules.get('torch').mm(a, b)") is not None
    # filesystem escape past the open('w') pattern
    assert scan_for_hacks("from pathlib import Path\nPath('reference.py').write_text('x')") is not None
    assert scan_for_hacks("import os\nos.chmod('reference.py', 0o644)") is not None
    # env read (mode-sniff / escape channel)
    assert scan_for_hacks("import os\nif os.environ.get('BENCH'): fast()") is not None
    # a decorator STACK must NOT trip the @ operator rule (regression)
    assert scan_for_hacks(
        "import triton\nimport triton.language as tl\n"
        "@triton.autotune(configs=[], key=['M'])\n@triton.jit\n"
        "def k():\n    acc = tl.dot(a, b)\n    return acc") is None


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
    # P4 reward: correctness_weight + log-shaped speedup + significance-gated
    # fast_p threshold bonuses (worst-shape 1.5x meets 1.0/1.2/1.5 thresholds).
    # "x=1" does not parse to a FULL_KERNEL, so the format term is 0.
    expected = CONFIG.correctness_weight + _expected_speed_term(1.5, 1.5)
    assert abs(rr.reward - expected) < 1e-9
    # 1.5x clears the 1.0x and 1.2x thresholds (with the noise-floor margin) but not
    # the 1.5x threshold (needs 1.5*1.02=1.53x), so only those two bonuses fire.
    assert "fast_p>=1.0" in rr.flags and "fast_p>=1.2" in rr.flags
    assert "fast_p>=1.5" not in rr.flags


def test_correct_but_slow_still_beats_incorrect():
    """P0 regression: a correct kernel slower than baseline must NEVER score
    below an incorrect kernel, or GRPO advantages invert."""
    slow = Observation(compiled=True, validation_passed=True, snr_by_shape={"s": 99.0},
                       wall_by_shape={"s": 4.0}, baseline_by_shape={"s": 1.0})  # 0.25x
    incorrect = Observation(compiled=True, validation_passed=True, snr_by_shape={"s": 5.0})
    r_slow = compute_reward(slow, "x=1", dtype="bf16")
    r_bad = compute_reward(incorrect, "x=1", dtype="bf16")
    assert r_slow.correct is True and r_slow.reward > r_bad.reward
    # CHANGED (P1 sub-threshold shaping): the incorrect kernel now earns a small
    # continuous credit (eps_shape * clamp(snr/threshold)) instead of a flat 0,
    # so ``r_bad.reward`` is a small POSITIVE value rather than exactly 0.0. The
    # invariant that matters is unchanged: it stays strictly BELOW the correct
    # tier (correctness_weight), so a correct-but-slow kernel still dominates.
    assert 0.0 < r_bad.reward < CONFIG.correctness_weight
    assert r_slow.reward >= CONFIG.correctness_weight


def test_reward_hack_beats_nothing():
    obs = Observation(compiled=True, validation_passed=True, snr_by_shape={"s": 99.0},
                      wall_by_shape={"s": 0.1}, baseline_by_shape={"s": 1.0})
    rr = compute_reward(obs, "import aiter\nout = aiter.rms_norm(x)", dtype="bf16")
    assert rr.correct is False and "hack" in rr.flags
    # CHANGED: a flagged hack is now punished STRICTLY harder than a compile
    # failure (reward_hack < reward_compile_fail) — cheating is the unique floor.
    assert rr.reward == CONFIG.reward_hack
    assert CONFIG.reward_hack < CONFIG.reward_compile_fail
    # A hack that "passes" every gate still gets NO shaping/format credit.
    assert "shaped" not in rr.flags


def test_excessive_speedup_flagged_and_capped():
    obs = Observation(compiled=True, validation_passed=True, snr_by_shape={"s": 99.0},
                      wall_by_shape={"s": 0.01}, baseline_by_shape={"s": 1.0})  # 100x
    rr = compute_reward(obs, "x=1", dtype="bf16")
    assert "excessive_speedup" in rr.flags
    # Excessive speedup: capped to excessive_speedup_flag for the (log-shaped)
    # continuous term, and fast_p bonuses are WITHHELD (measurement-error outlier),
    # so the policy cannot farm the bonus with an implausible timing.
    expected = CONFIG.correctness_weight + _expected_speed_term(
        CONFIG.excessive_speedup_flag, 100.0, excessive=True)
    assert abs(rr.reward - expected) < 1e-9
    assert not any(f.startswith("fast_p") for f in rr.flags)


# =========================================================================== #
# Reward-shaping upgrades (P1 sub-threshold shaping / P2 format / P3 curriculum)
#
# All of the tests below keep the lexicographic anti-hack guarantee intact:
#   hack < compile_fail < incorrect(shaped) < correct-slow < correct-fast.
# =========================================================================== #
import dataclasses  # noqa: E402


_DT = "bf16"  # snr threshold 25.0


def _obs_incorrect(snr: float, dtype: str = _DT) -> Observation:
    """Compiled + validated but SNR below the gate -> incorrect tier."""
    return Observation(compiled=True, validation_passed=True,
                       snr_by_shape={"s": snr}, dtype=dtype)


def _obs_correct(speedup: float | None) -> Observation:
    """Correct kernel with a given worst-shape speedup (None = no timing)."""
    if speedup is None:
        return Observation(compiled=True, validation_passed=True, snr_by_shape={"s": 99.0})
    return Observation(compiled=True, validation_passed=True, snr_by_shape={"s": 99.0},
                       wall_by_shape={"s": 1.0 / speedup}, baseline_by_shape={"s": 1.0})


_VALID_CONTRACT = "FULL_KERNEL:\n```python\nimport triton\n@triton.jit\ndef k():\n    pass\n```"
_MALFORMED = "ANALYSIS: here is my prose answer with no kernel block at all."


# --------------------------------------------------------------------------- #
# config invariants: the numeric bounds that PROVE ordering can never break
# --------------------------------------------------------------------------- #
def test_shaping_config_bounds_guarantee_ordering():
    c = CONFIG
    # hack is the unique floor, strictly below an honest compile failure.
    assert c.reward_hack < c.reward_compile_fail < c.reward_incorrect
    # Even the MAX shaped-incorrect (full sub-threshold credit + a format bonus)
    # stays strictly below the MIN correct reward (base minus a format penalty).
    max_incorrect = c.reward_incorrect + c.eps_shape + c.format_weight
    min_correct = c.correctness_weight - c.format_weight
    assert max_incorrect < min_correct
    # The worst malformed compile/incorrect output still sits above the hack floor.
    assert c.reward_compile_fail < c.reward_incorrect - c.format_weight
    assert c.reward_hack < c.reward_compile_fail - c.format_weight


# --------------------------------------------------------------------------- #
# P1: bounded continuous sub-threshold shaping
# --------------------------------------------------------------------------- #
def test_subthreshold_shaping_is_continuous_and_monotonic():
    """Higher SNR (closer to the gate) => strictly more shaped credit, but the
    reward stays flat-incorrect-tier (< eps_shape, < correct)."""
    r_lo = compute_reward(_obs_incorrect(2.0), "x=1", dtype=_DT)
    r_mid = compute_reward(_obs_incorrect(12.5), "x=1", dtype=_DT)  # halfway to 25
    r_hi = compute_reward(_obs_incorrect(24.9), "x=1", dtype=_DT)   # just below gate
    for r in (r_lo, r_mid, r_hi):
        assert r.correct is False and r.tier == "incorrect"
    assert r_lo.reward < r_mid.reward < r_hi.reward          # monotone in progress
    # halfway SNR (12.5/25 = 0.5) -> exactly half of eps_shape of credit.
    assert abs(r_mid.reward - (CONFIG.reward_incorrect + 0.5 * CONFIG.eps_shape)) < 1e-9
    # every shaped value is strictly bounded below eps_shape and the correct tier.
    assert r_hi.reward < CONFIG.eps_shape < CONFIG.correctness_weight


def test_subthreshold_shaping_can_be_disabled():
    """With shaping off, a compiled-but-incorrect kernel is flat reward_incorrect
    (the legacy sparse behavior) — proves the knob works."""
    cfg = dataclasses.replace(CONFIG, subthreshold_shaping=False)
    rr = compute_reward(_obs_incorrect(20.0), "x=1", dtype=_DT, cfg=cfg)
    assert rr.tier == "incorrect" and rr.reward == cfg.reward_incorrect
    assert "shaped" not in rr.flags


def test_shaped_incorrect_strictly_below_every_correct():
    """The whole point of the P1 bound: no shaped-incorrect kernel, at ANY SNR,
    can reach even the slowest / no-timing correct kernel."""
    correct_min = min(
        compute_reward(_obs_correct(0.01), "x=1", dtype=_DT).reward,   # extremely slow
        compute_reward(_obs_correct(None), "x=1", dtype=_DT).reward,   # correct, no bench
    )
    for snr in (0.0, 5.0, 12.5, 20.0, 24.9):
        r = compute_reward(_obs_incorrect(snr), "x=1", dtype=_DT)
        assert r.correct is False
        assert r.reward < correct_min


# --------------------------------------------------------------------------- #
# hack / compile-fail are NEVER shaped
# --------------------------------------------------------------------------- #
def test_hack_and_compile_fail_never_shaped():
    # A hack that would otherwise "pass" correctness gets the pure hack floor.
    hack = compute_reward(_obs_correct(5.0), "import aiter\nout = aiter.rms_norm(x)", dtype=_DT)
    assert hack.tier == "hack" and hack.reward == CONFIG.reward_hack
    assert "shaped" not in hack.flags
    # A compile failure gets the pure compile-fail floor, no shaping/format even
    # if we hand it a perfectly-formatted response.
    cf = compute_reward(Observation(compiled=False, dtype=_DT), "x=1", dtype=_DT,
                        response=_VALID_CONTRACT)
    assert cf.tier == "compile_fail" and cf.reward == CONFIG.reward_compile_fail
    assert "shaped" not in cf.flags


# --------------------------------------------------------------------------- #
# P2: format-compliance term, bounded so it can never flip ordering
# --------------------------------------------------------------------------- #
def test_format_bonus_and_penalty_are_bounded():
    base_incorrect = compute_reward(_obs_incorrect(10.0), "x=1", dtype=_DT).reward
    good = compute_reward(_obs_incorrect(10.0), "x=1", dtype=_DT, response=_VALID_CONTRACT)
    bad = compute_reward(_obs_incorrect(10.0), "x=1", dtype=_DT, response=_MALFORMED)
    # valid contract adds +format_weight, malformed subtracts it, symmetric & tiny.
    assert abs(good.reward - (base_incorrect + CONFIG.format_weight)) < 1e-9
    assert abs(bad.reward - (base_incorrect - CONFIG.format_weight)) < 1e-9
    # bounded so it can NEVER flip a tier: a malformed-but-correct kernel still
    # outranks a valid-contract shaped-incorrect kernel.
    correct_bad_fmt = compute_reward(_obs_correct(0.01), "x=1", dtype=_DT, response=_MALFORMED)
    incorrect_good_fmt = compute_reward(_obs_incorrect(24.9), "x=1", dtype=_DT,
                                        response=_VALID_CONTRACT)
    assert correct_bad_fmt.correct is True
    assert correct_bad_fmt.reward > incorrect_good_fmt.reward
    # no response supplied -> no format term at all (legacy behavior preserved).
    assert abs(compute_reward(_obs_incorrect(10.0), "x=1", dtype=_DT).reward
               - base_incorrect) < 1e-9


# --------------------------------------------------------------------------- #
# P3: correctness -> latency curriculum
# --------------------------------------------------------------------------- #
def test_correctness_phase_zeroes_speed_term():
    fast = _obs_correct(4.0)
    slow = _obs_correct(0.25)
    r_fast = compute_reward(fast, "x=1", dtype=_DT, phase="correctness")
    r_slow = compute_reward(slow, "x=1", dtype=_DT, phase="correctness")
    # In the correctness phase speed is irrelevant: both correct kernels score
    # exactly correctness_weight regardless of their (very different) speedups.
    assert r_fast.correct is r_slow.correct is True
    assert abs(r_fast.reward - CONFIG.correctness_weight) < 1e-9
    assert abs(r_slow.reward - CONFIG.correctness_weight) < 1e-9
    assert "phase:correctness" in r_fast.flags


def test_latency_and_full_phases_use_speed():
    fast = _obs_correct(4.0)
    r_full = compute_reward(fast, "x=1", dtype=_DT, phase="full")
    r_latency = compute_reward(fast, "x=1", dtype=_DT, phase="latency")
    r_default = compute_reward(fast, "x=1", dtype=_DT)  # cfg.reward_phase = "full"
    expected = CONFIG.correctness_weight + _expected_speed_term(4.0, 4.0)
    for r in (r_full, r_latency, r_default):
        assert abs(r.reward - expected) < 1e-9


def test_phase_defaults_to_config():
    cfg = dataclasses.replace(CONFIG, reward_phase="correctness")
    rr = compute_reward(_obs_correct(4.0), "x=1", dtype=_DT, cfg=cfg)
    assert abs(rr.reward - cfg.correctness_weight) < 1e-9  # config drives the phase


# --------------------------------------------------------------------------- #
# P4: speedup reshape — breaks the "correct-but-slow" plateau at the 1x crossover
# --------------------------------------------------------------------------- #
def test_fast_p_bonus_creates_reward_jump_at_baseline_crossover():
    """The central plateau fix: crossing 1.0x must be a DISTINCT high-value event,
    not a marginal linear increment. A 1.01x kernel should out-reward a 0.99x one
    by ~the first fast_p bonus, giving GRPO strong group-relative advantage."""
    # below vs clearly-above the noise-floor-margined 1.0x crossover (need >=1.02x)
    just_below = compute_reward(_obs_correct(0.99), "x=1", dtype=_DT)
    just_above = compute_reward(_obs_correct(1.05), "x=1", dtype=_DT)
    assert just_below.correct and just_above.correct
    jump = just_above.reward - just_below.reward
    # dominated by the 1.0x threshold bonus (0.30), far above the ~0.05 linear step
    assert jump >= CONFIG.fast_p_bonus[0][1] * 0.9
    assert "fast_p>=1.0" in just_above.flags
    assert not any(f.startswith("fast_p") for f in just_below.flags)


def test_speedup_reshape_preserves_lexicographic_dominance():
    """No matter how the speed term is shaped, every correct kernel (even 0.01x)
    must strictly beat the best-possible incorrect kernel."""
    worst_correct = compute_reward(_obs_correct(0.01), "x=1", dtype=_DT)
    best_incorrect = compute_reward(_obs_incorrect(24.9), "x=1",
                                    dtype=_DT, response=_VALID_CONTRACT)
    assert worst_correct.correct and not best_incorrect.correct
    assert worst_correct.reward >= CONFIG.correctness_weight
    assert worst_correct.reward > best_incorrect.reward


def test_fast_p_bonus_withheld_when_timing_untrustworthy():
    """A fast speedup with high timing variance (cv > threshold) must NOT earn the
    fast_p bonus — otherwise the policy farms bonuses via noisy/lucky timings."""
    noisy = Observation(compiled=True, validation_passed=True, snr_by_shape={"s": 99.0},
                        wall_by_shape={"s": 0.5}, baseline_by_shape={"s": 1.0},  # 2x
                        cv_pct=CONFIG.cv_threshold_pct + 5.0)  # noisy
    rr = compute_reward(noisy, "x=1", dtype=_DT)
    assert rr.correct and "high_variance" in rr.flags
    assert not any(f.startswith("fast_p") for f in rr.flags)
    # speed term damped to <=1.0 (linear), so reward ~ correctness_weight + <=1.0
    assert rr.reward <= CONFIG.correctness_weight + CONFIG.speedup_weight + 1e-9


def test_log_shape_monotonic_and_continuous_at_one():
    """The continuous speed term stays monotonic and is continuous at su=1."""
    rewards = [compute_reward(_obs_correct(su), "x=1", dtype=_DT).reward
               for su in (0.25, 0.5, 0.9, 1.0, 1.1, 2.0, 4.0)]
    assert rewards == sorted(rewards)  # strictly non-decreasing in speedup


# --------------------------------------------------------------------------- #
# EXHAUSTIVE lexicographic ordering with shaping ON (the anti-hack guarantee)
# --------------------------------------------------------------------------- #
def test_full_lexicographic_ordering_with_shaping_on():
    """hack < compile_fail < incorrect(shaped, every SNR) < correct-slow
    < correct-fast — swept across SNRs, speeds, phases, and format status."""
    hack = compute_reward(_obs_correct(5.0), "y = torch.matmul(a, b)", dtype=_DT)
    compile_fail = compute_reward(Observation(compiled=False, dtype=_DT), "x=1", dtype=_DT)
    assert hack.tier == "hack" and compile_fail.tier == "compile_fail"
    assert hack.reward < compile_fail.reward

    # every shaped-incorrect outcome (any SNR, any format) beats compile_fail and
    # loses to every correct outcome.
    correct_slow = compute_reward(_obs_correct(0.1), "x=1", dtype=_DT)     # 0.1x, slow
    correct_fast = compute_reward(_obs_correct(8.0), "x=1", dtype=_DT)     # 8x, fast
    assert correct_slow.correct and correct_fast.correct
    assert correct_slow.reward < correct_fast.reward  # speed strictly orders correct

    incorrect_rewards = []
    for snr in (0.0, 3.0, 10.0, 24.9):
        for resp in (None, _VALID_CONTRACT, _MALFORMED):
            r = compute_reward(_obs_incorrect(snr), "x=1", dtype=_DT, response=resp)
            assert r.tier == "incorrect"
            incorrect_rewards.append(r.reward)

    assert max(incorrect_rewards) < correct_slow.reward     # never reaches correct
    assert min(incorrect_rewards) > compile_fail.reward     # always above compile-fail
    # full strict chain on representative points:
    assert (hack.reward < compile_fail.reward < min(incorrect_rewards)
            <= max(incorrect_rewards) < correct_slow.reward < correct_fast.reward)


def test_all_task_seeds_stay_clean_and_rewardable():
    """The 15 shipped task seeds must never be flagged as hacks (anti-hack must
    not over-fire) and, when correct, must land in the correct tier."""
    from kore.tasks.registry import all_tasks

    tasks = all_tasks()
    assert len(tasks) == 15
    for t in tasks:
        assert scan_for_hacks(t.seed_source) is None, f"{t.task_id}: seed wrongly flagged"
        rr = compute_reward(_obs_correct(2.0), t.seed_source, dtype=t.dtype)
        assert rr.correct is True and rr.tier == "correct_timed"
