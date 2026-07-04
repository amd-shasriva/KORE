"""CPU-only tests for the hardware-counter dense reward (flagship novelty)."""

from __future__ import annotations

import dataclasses

from kore.config import CONFIG
from kore.reward import profile_reward as pr
from kore.reward.reward import Observation, compute_reward


# Synthetic counter dicts (rocprofv3-style names).
def _c(valu, salu, vmem, mfma, wait):
    return {
        "SQ_INSTS_VALU": valu, "SQ_INSTS_SALU": salu, "SQ_INSTS_VMEM": vmem,
        "SQ_INSTS_VALU_MFMA_BF16": mfma, "SQ_WAIT_INST_ANY": wait,
    }


def test_issue_efficiency_and_stall_fraction():
    c = _c(valu=800, salu=100, vmem=100, mfma=0, wait=1000)  # issued=1000, wait=1000
    assert abs(pr.stall_fraction(c) - 0.5) < 1e-9
    assert abs(pr.issue_efficiency(c) - 0.5) < 1e-9


def test_score_is_one_when_candidate_matches_baseline():
    ref = _c(800, 100, 100, 0, 500)
    assert abs(pr.profile_efficiency_score(dict(ref), dict(ref)) - 1.0) < 1e-9


def test_score_rewards_fewer_stalls_and_less_traffic():
    ref = _c(valu=800, salu=100, vmem=200, mfma=0, wait=1000)
    # candidate: fewer stalls (better scheduling) AND less memory traffic
    better = _c(valu=800, salu=100, vmem=100, mfma=0, wait=200)
    worse = _c(valu=800, salu=100, vmem=400, mfma=0, wait=3000)
    s_better = pr.profile_efficiency_score(better, ref)
    s_worse = pr.profile_efficiency_score(worse, ref)
    assert s_better > s_worse
    assert 0.0 <= s_worse < s_better <= 1.0


def test_score_bounded_and_capped_at_one():
    ref = _c(800, 100, 100, 0, 1000)
    # candidate hugely better than baseline -> clamped to 1.0, never > 1
    great = _c(800, 100, 10, 0, 1)
    assert pr.profile_efficiency_score(great, ref) <= 1.0


def test_score_none_when_no_usable_counters():
    assert pr.profile_efficiency_score({}, {}) is None


# --- reward integration ---------------------------------------------------- #
def _obs(su, prof=None):
    return Observation(compiled=True, validation_passed=True, snr_by_shape={"s": 99.0},
                       wall_by_shape={"s": 1.0 / su}, baseline_by_shape={"s": 1.0},
                       profile_efficiency=prof)


def test_profile_reward_inert_by_default():
    """Default weight 0 => profile_efficiency never changes the reward."""
    with_prof = compute_reward(_obs(0.8, prof=1.0), "x=1", dtype="bf16")
    without = compute_reward(_obs(0.8, prof=None), "x=1", dtype="bf16")
    assert abs(with_prof.reward - without.reward) < 1e-9
    assert not any(f.startswith("profile") for f in with_prof.flags)


def test_profile_reward_adds_bounded_bonus_when_enabled():
    cfg = dataclasses.replace(CONFIG, profile_reward_weight=0.15)
    hi = compute_reward(_obs(0.8, prof=1.0), "x=1", dtype="bf16", cfg=cfg)
    lo = compute_reward(_obs(0.8, prof=0.0), "x=1", dtype="bf16", cfg=cfg)
    # dense signal in the correct-but-slow band: better counters -> higher reward
    assert abs((hi.reward - lo.reward) - 0.15) < 1e-9
    assert any(f.startswith("profile+") for f in hi.flags)


def test_profile_reward_never_dominates_fast_p():
    """A counter-efficient SLOW kernel must not out-reward a kernel that actually
    beats the baseline — the profiler shapes, it never leads."""
    cfg = dataclasses.replace(CONFIG, profile_reward_weight=0.15)
    slow_but_efficient = compute_reward(_obs(0.9, prof=1.0), "x=1", dtype="bf16", cfg=cfg)
    genuinely_fast = compute_reward(_obs(1.3, prof=0.0), "x=1", dtype="bf16", cfg=cfg)
    assert genuinely_fast.reward > slow_but_efficient.reward
