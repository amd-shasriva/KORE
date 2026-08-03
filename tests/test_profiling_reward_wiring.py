"""The coverage reward and MRS reach the rollout, or are provably inert.

Two separate integrity questions:

* the coverage reward must contribute NOTHING unless it is armed AND there is
  evidence the trace collector works on this hardware, because the collector fails
  safe and an armed-but-broken collector would state a reward that silently pays
  0.0 every turn;
* MRS must actually remove a rejected trajectory from the samples the estimator
  sees. A filter that logs a rejection and then trains on the trajectory anyway is
  worse than no filter, because its statistics say the data was cleaned.
"""

from __future__ import annotations

import types

import pytest

from kore.policy import grpo
from kore.policy.configs import GRPOConfig
from kore.verifier.parsers.rocprofv3 import KernelDispatch


class _Obs:
    def __init__(self, speedup=2.0, passed=True):
        self.validation_passed = passed
        self.speedup = speedup


class _TracingEnv:
    """An env that returns a fixed dispatch trace."""

    def __init__(self, dispatches):
        self._dispatches = dispatches
        self.calls = 0

    def collect_kernel_trace(self, code, shape=None):
        self.calls += 1
        return self._dispatches


CANDIDATE = """
import triton
import triton.language as tl

@triton.jit
def fused_kernel(x_ptr, y_ptr):
    pass
"""

GOOD_TRACE = [KernelDispatch("fused_kernel_0d1d", 9_000),
              KernelDispatch("void at::native::elementwise", 1_000)]


def _armed(**kw) -> GRPOConfig:
    """A config whose coverage reward is armed with a collector receipt."""
    base = dict(profiling_reward_weight=0.15,
                profiling_reward_evidence_path="data/ktrace_receipt.json",
                correctness_weight=0.3, reward_phase="all")
    base.update(kw)
    return GRPOConfig(**base)


# --------------------------------------------------------------------------- #
# 1. the arming gate
# --------------------------------------------------------------------------- #
def test_the_weight_is_zero_without_a_collector_receipt():
    """The whole gate: no evidence, no reward, however large the weight."""
    assert grpo._profiling_reward_weight(
        GRPOConfig(profiling_reward_weight=0.15)) == 0.0
    assert grpo._profiling_reward_weight(_armed()) == pytest.approx(0.15)


def test_a_weight_that_could_outrank_correctness_is_refused():
    """Shaping above ``correctness_weight`` would invert the reward's order."""
    assert grpo._profiling_reward_weight(
        _armed(profiling_reward_weight=0.3)) == 0.0
    assert grpo._profiling_reward_weight(
        _armed(profiling_reward_weight=0.9)) == 0.0
    assert grpo._profiling_reward_weight(
        _armed(profiling_reward_weight=0.29)) == pytest.approx(0.29)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf"), None])
def test_an_unusable_weight_disables_the_path(weight):
    assert grpo._profiling_reward_weight(_armed(profiling_reward_weight=weight)) == 0.0


def test_the_correctness_curriculum_phase_collects_nothing():
    """Phase 1 trains correctness only, so no speed term may be added."""
    env = _TracingEnv(GOOD_TRACE)
    term, _ = grpo._coverage_bonus(
        env, None, CANDIDATE, _Obs(), _armed(reward_phase="correctness"))
    assert term == 0.0
    assert env.calls == 0, "the profiler must not even be invoked in phase 1"


def test_an_unarmed_config_never_invokes_the_profiler():
    """The default path must be a byte-for-byte no-op, not a wasted trace."""
    env = _TracingEnv(GOOD_TRACE)
    term, feedback = grpo._coverage_bonus(
        env, None, CANDIDATE, _Obs(), GRPOConfig())
    assert (term, feedback) == (0.0, "")
    assert env.calls == 0


# --------------------------------------------------------------------------- #
# 2. the armed path
# --------------------------------------------------------------------------- #
def test_an_armed_config_rewards_coverage_and_explains_it():
    env = _TracingEnv(GOOD_TRACE)
    term, feedback = grpo._coverage_bonus(
        env, None, CANDIDATE, _Obs(speedup=2.0), _armed())
    assert env.calls == 1
    assert 0.0 < term <= 0.15
    assert "90.00%" in feedback           # 9000 / 10000 ns
    assert "PROFILE" in feedback


def test_a_kernel_that_never_ran_earns_nothing_but_is_told_why():
    """Coverage 0.0 is a measurement, and the feedback is the actionable part."""
    env = _TracingEnv([KernelDispatch("someone_elses_gemm", 5_000)])
    term, feedback = grpo._coverage_bonus(
        env, None, CANDIDATE, _Obs(speedup=6.0), _armed())
    assert term == 0.0
    assert "never" in feedback


def test_an_implausible_speedup_pays_nothing_yet_still_reports_coverage():
    env = _TracingEnv(GOOD_TRACE)
    term, feedback = grpo._coverage_bonus(
        env, None, CANDIDATE, _Obs(speedup=1541.94), _armed())
    assert term == 0.0
    assert "90.00%" in feedback


def test_the_bonus_is_never_negative():
    """Punishment belongs to the correctness tiers, not to a shaping term."""
    env = _TracingEnv(GOOD_TRACE)
    term, _ = grpo._coverage_bonus(
        env, None, CANDIDATE, _Obs(speedup=0.2), _armed())
    assert term == 0.0


@pytest.mark.parametrize("env", [
    types.SimpleNamespace(),                      # no collector at all
    _TracingEnv(None),                            # profiler unavailable
    _TracingEnv([]),                              # empty trace
    _TracingEnv([KernelDispatch("k", 0)]),        # zero total GPU time
])
def test_an_unmeasurable_trace_yields_no_reward_rather_than_zero_coverage(env):
    term, feedback = grpo._coverage_bonus(
        env, None, CANDIDATE, _Obs(), _armed())
    assert (term, feedback) == (0.0, "")


def test_a_candidate_with_no_triton_kernels_is_not_credited_with_everything():
    """With no names to match, coverage is unknowable -- not 100% and not 0%."""
    env = _TracingEnv(GOOD_TRACE)
    term, feedback = grpo._coverage_bonus(
        env, None, "def f(x): return x", _Obs(), _armed())
    assert (term, feedback) == (0.0, "")


def test_an_incorrect_candidate_is_never_profiled():
    env = _TracingEnv(GOOD_TRACE)
    term, _ = grpo._coverage_bonus(
        env, None, CANDIDATE, _Obs(passed=False), _armed())
    assert term == 0.0 and env.calls == 0


def test_a_raising_collector_cannot_break_a_rollout():
    class _Broken:
        def collect_kernel_trace(self, code, shape=None):
            raise RuntimeError("rocprofv3 exploded")

    assert grpo._coverage_bonus(
        _Broken(), None, CANDIDATE, _Obs(), _armed()) == (0.0, "")


# --------------------------------------------------------------------------- #
# 3. MRS actually removes the trajectory from the samples
# --------------------------------------------------------------------------- #
def _group():
    """Four trajectories: one gamed, one never-correct, two healthy."""
    rewards = [[0.9, 0.95], [0.2, 0.30], [0.0, 0.0], [0.5, 0.55]]
    correct = [[True, True], [True, True], [False, False], [True, True]]
    speedups = [[1.2, 1.4], [1.05, 1.1], [None, None], [1.1, 1.15]]
    infra = [[False, False]] * 4
    return rewards, correct, speedups, infra


def test_mrs_is_a_no_op_when_rejection_sampling_is_off():
    rewards, correct, speedups, infra = _group()
    out, report = grpo._apply_mrs(GRPOConfig(rejection_sampling=False),
                                 rewards, correct, speedups, infra)
    assert out is infra and report is None


def test_a_rejected_trajectory_contributes_no_samples():
    """The property that matters: rejected turns never reach the estimator."""
    rewards, correct, speedups, infra = _group()
    config = GRPOConfig(rejection_sampling=True, gamma=0.4)
    out, report = grpo._apply_mrs(config, rewards, correct, speedups, infra)
    assert report is not None and report.applied
    assert report.keep_indices == [0, 1, 3]

    _, index = grpo.build_kevin_samples(rewards, correct, 0.4, traj_infra=out)
    trajectories = {ti for ti, _ in index}
    assert trajectories == {0, 1, 3}
    assert 2 not in trajectories, "the never-correct trajectory still trained"


def test_a_gamed_trajectory_contributes_no_samples():
    rewards = [[0.9, 0.95], [0.4, 0.5], [0.9, 0.99]]
    correct = [[True, True]] * 3
    speedups = [[1.2, 1.4], [1.05, 1.1], [1.1, 1541.94]]
    infra = [[False, False]] * 3
    out, report = grpo._apply_mrs(GRPOConfig(rejection_sampling=True),
                                 rewards, correct, speedups, infra)
    assert report.rejected.get("implausible_speedup") == 1
    _, index = grpo.build_kevin_samples(rewards, correct, 0.4, traj_infra=out)
    assert 2 not in {ti for ti, _ in index}


def test_a_declining_filter_leaves_every_trajectory_trainable():
    """If filtering would collapse the group's contrast, nothing is dropped."""
    rewards = [[0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.0, 0.0]]
    correct = [[True, True], [True, True], [True, True], [False, False]]
    infra = [[False, False]] * 4
    out, report = grpo._apply_mrs(GRPOConfig(rejection_sampling=True),
                                 rewards, correct, None, infra)
    assert not report.applied
    _, index = grpo.build_kevin_samples(rewards, correct, 0.4, traj_infra=out)
    assert {ti for ti, _ in index} == {0, 1, 2, 3}


def test_mrs_preserves_a_genuine_infra_flag():
    """An infrastructure failure on a KEPT trajectory must stay dropped."""
    rewards = [[0.9, 0.95], [0.3, 0.4]]
    correct = [[True, True], [True, True]]
    infra = [[False, True], [False, False]]
    out, report = grpo._apply_mrs(GRPOConfig(rejection_sampling=True),
                                 rewards, correct, None, infra)
    assert report.keep_indices == [0, 1]
    assert out[0] == [False, True]


# --------------------------------------------------------------------------- #
# 4. the env collector fails safe
# --------------------------------------------------------------------------- #
def test_collect_kernel_trace_returns_none_without_a_profiler(monkeypatch):
    """No rocprofv3 on this box, so the honest answer is None.

    Exercised through the real method rather than a double, so the fail-safe
    contract is checked on the code a rollout would actually call.
    """
    from kore.env.kore_env import KoreEnv

    env = KoreEnv.__new__(KoreEnv)          # no __init__: no task, no GPU
    assert env.collect_kernel_trace("print('x')") is None
