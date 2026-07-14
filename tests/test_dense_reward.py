"""CPU-only tests for the dense hardware-counter reward wired into the GRPO rollout.

Two layers are covered, no GPU required:

  * the PURE roofline-aware dense-score math
    (:func:`kore.reward.profile_reward.roofline_dense_score`): a memory-bound kernel
    far from the roofline scores LOW, a near-roofline kernel scores HIGH, it uses
    :func:`profile_efficiency_score` when the vendor-reference counters are supplied,
    and it returns ``None`` when nothing is computable; and

  * the GRPO rollout hook (:func:`kore.policy.grpo._dense_profile_bonus` /
    ``_dense_profile_weight`` / ``_make_rollout_env``) with rocprofv3 counters
    INJECTED via a stub env: the shaped bonus + counter feedback are produced when
    the weight > 0, it is a byte-for-byte NO-OP when the weight == 0 (the stub's
    ``collect_counters`` is never even called), it is fail-safe on a profiler crash,
    and it is skipped in the correctness curriculum phase and for incorrect kernels.
"""

from __future__ import annotations

import types

from kore.analysis import roofline as _R
from kore.reward import profile_reward as pr
from kore.reward.reward import Observation


# --------------------------------------------------------------------------- #
# rocprofv3-style synthetic counter dicts (no GPU).
# --------------------------------------------------------------------------- #
def _counters(valu, salu, vmem, mfma, wait):
    return {
        "SQ_INSTS_VALU": valu, "SQ_INSTS_SALU": salu, "SQ_INSTS_VMEM": vmem,
        "SQ_INSTS_VALU_MFMA_BF16": mfma, "SQ_WAIT_INST_ANY": wait,
    }


# A modelable op (bf16, arithmetic intensity 0.25 => memory-bound). The bandwidth
# roofline lower bound is t_min = bytes / HBM_BW on the ACTIVE arch (gfx950/MI350X:
# 4e6 / 8e12 = 5.0e-4 ms; gfx942/MI300X was 7.5e-4). Derive the near/far runtimes
# from t_min so the test is correct on whichever arch is active.
_FLOPS = 1.0e6
_BYTES = 4.0e6
_TMIN_MS = _R.roofline(_FLOPS, _BYTES, "bf16")["t_min_ms"]
_NEAR_MS = _TMIN_MS * 1.05    # ~5% off the roofline -> near (high score)
_FAR_MS = _TMIN_MS * 10.0     # 10x t_min -> far from the roofline (low score)

# Stalling kernel (spends most cycles waiting) vs a busy one (ALUs kept fed).
_STALLING = _counters(valu=800, salu=100, vmem=500, mfma=0, wait=4000)   # low issue eff
_BUSY = _counters(valu=800, salu=100, vmem=100, mfma=2000, wait=200)     # high issue eff


# --------------------------------------------------------------------------- #
# PURE roofline_dense_score math
# --------------------------------------------------------------------------- #
def test_memory_bound_far_from_roofline_scores_low():
    far = pr.roofline_dense_score(_STALLING, flops=_FLOPS, bytes=_BYTES,
                                  measured_ms=_FAR_MS, dtype="bf16")
    assert far is not None
    assert 0.0 <= far < 0.5


def test_near_roofline_scores_high_and_beats_far():
    near = pr.roofline_dense_score(_BUSY, flops=_FLOPS, bytes=_BYTES,
                                   measured_ms=_NEAR_MS, dtype="bf16")
    far = pr.roofline_dense_score(_STALLING, flops=_FLOPS, bytes=_BYTES,
                                  measured_ms=_FAR_MS, dtype="bf16")
    assert near is not None and far is not None
    assert near > far            # dense gradient in the flat correct-but-slow band
    assert 0.8 < near <= 1.0     # near the roofline -> high


def test_uses_profile_efficiency_score_when_ref_given():
    # Same op; only the counters differ vs the tuned vendor baseline. With no
    # roofline inputs the score is the baseline-relative profile_efficiency_score
    # (blended with the candidate's own issue efficiency).
    ref = _counters(valu=800, salu=100, vmem=200, mfma=0, wait=1000)
    better = _counters(valu=800, salu=100, vmem=100, mfma=0, wait=200)   # fewer stalls, less traffic
    worse = _counters(valu=800, salu=100, vmem=400, mfma=0, wait=3000)   # more stalls, more traffic
    s_better = pr.roofline_dense_score(better, ref=ref)
    s_worse = pr.roofline_dense_score(worse, ref=ref)
    assert s_better is not None and s_worse is not None
    assert 0.0 <= s_worse < s_better <= 1.0


def test_degrades_to_issue_efficiency_from_counters_only():
    # No roofline inputs, no ref -> the score is just the candidate's issue
    # efficiency (1 - stall_fraction) from its own counters.
    busy = pr.roofline_dense_score(_BUSY)
    stalling = pr.roofline_dense_score(_STALLING)
    assert busy is not None and stalling is not None
    assert busy > stalling
    assert busy > 0.8


def test_none_when_nothing_computable():
    assert pr.roofline_dense_score({}, ref=None) is None
    assert pr.roofline_dense_score({}, ref={}) is None
    # counters unusable and no roofline inputs -> None (caller no-ops).
    assert pr.roofline_dense_score({"UNRELATED": 5}) is None


def test_score_is_bounded_even_for_super_fast_kernel():
    # attained_fraction can exceed 100% (cache reuse); the score stays clamped.
    s = pr.roofline_dense_score(_BUSY, flops=_FLOPS, bytes=_BYTES,
                                measured_ms=_NEAR_MS / 100.0, dtype="bf16")
    assert s is not None and 0.0 <= s <= 1.0


# --------------------------------------------------------------------------- #
# GRPO rollout hook (counters injected via a stub env; no GPU)
# --------------------------------------------------------------------------- #
from kore.policy import grpo  # noqa: E402 - imported after the pure helpers above


class _StubEnv:
    """Stand-in for KoreEnv exposing only the public ``collect_counters``."""

    def __init__(self, counters, raise_on_collect=False):
        self._counters = counters
        self.raise_on_collect = raise_on_collect
        self.calls = []                      # records every collect_counters(source)

    def collect_counters(self, source, shape=None):
        self.calls.append(source)
        if self.raise_on_collect:
            raise RuntimeError("rocprofv3 unavailable")
        return dict(self._counters) if self._counters is not None else None


def _stub_task(op="gemm", dtype="bf16", dims=None):
    dims = dims or {"M": 512, "N": 512, "K": 512}
    sh = types.SimpleNamespace(name="primary", dims=dims)
    return types.SimpleNamespace(
        task_id="stub", operation=op, dtype=dtype,
        shape=lambda name: sh if name == "primary" else None, shapes=[sh])


def _obs(measured_ms, correct=True):
    return Observation(compiled=True, validation_passed=correct,
                       snr_by_shape={"primary": 99.0}, wall_ms=measured_ms, dtype="bf16")


def _cfg(weight, phase="latency", agentic=False):
    return types.SimpleNamespace(profile_reward_weight=weight, reward_phase=phase,
                                 agentic=agentic)


def test_grpo_dense_bonus_active_when_weight_positive():
    env = _StubEnv(_BUSY)
    dense, fb = grpo._dense_profile_bonus(env, _stub_task(), "def k(): pass",
                                          _obs(3.0e-4), _cfg(0.15))
    assert dense > 0.0
    assert dense <= 0.15                     # bounded by the weight (shapes, never leads)
    assert env.calls == ["def k(): pass"]    # counters collected exactly once
    assert "HARDWARE COUNTERS (rocprofv3)" in fb
    assert "bottleneck=" in fb
    assert "roofline attainment" in fb


def test_grpo_dense_bonus_near_roofline_beats_far():
    task = _stub_task()
    near, _ = grpo._dense_profile_bonus(_StubEnv(_BUSY), task, "s", _obs(3.0e-4), _cfg(0.15))
    far, _ = grpo._dense_profile_bonus(_StubEnv(_STALLING), task, "s", _obs(3.0e-3), _cfg(0.15))
    assert near > far >= 0.0


def test_grpo_dense_bonus_is_byte_for_byte_noop_at_weight_zero(monkeypatch):
    # Neutralize the config/env fallbacks so weight is unambiguously 0.
    import kore.config as kcfg
    monkeypatch.setattr(kcfg.CONFIG, "profile_reward_weight", 0.0, raising=False)
    monkeypatch.delenv("KORE_PROFILE_REWARD_WEIGHT", raising=False)
    env = _StubEnv(_BUSY)
    dense, fb = grpo._dense_profile_bonus(env, _stub_task(), "s", _obs(3.0e-4), _cfg(0.0))
    assert dense == 0.0
    assert fb == ""
    assert env.calls == []                   # true no-op: no counters collected, no GPU work


def test_grpo_dense_bonus_failsafe_on_profiler_crash():
    env = _StubEnv(None, raise_on_collect=True)
    dense, fb = grpo._dense_profile_bonus(env, _stub_task(), "s", _obs(3.0e-4), _cfg(0.15))
    assert dense == 0.0 and fb == ""         # profiler failure -> dense term 0.0, no crash


def test_grpo_dense_bonus_survives_missing_counters_via_roofline():
    # rocprofv3 unavailable (collect_counters -> None, no crash): the dense reward
    # degrades to the analytical roofline attainment computed from the verified
    # bench wall-time, which needs no profiler.
    env = _StubEnv(None)
    dense, fb = grpo._dense_profile_bonus(env, _stub_task(), "s", _obs(3.0e-4), _cfg(0.15))
    assert dense > 0.0
    assert "roofline attainment" in fb
    assert env.calls == ["s"]


def test_grpo_dense_bonus_skipped_in_correctness_phase():
    env = _StubEnv(_BUSY)
    dense, fb = grpo._dense_profile_bonus(env, _stub_task(), "s", _obs(3.0e-4),
                                          _cfg(0.15, phase="correctness"))
    assert dense == 0.0 and fb == ""
    assert env.calls == []                   # phase gate short-circuits before any collection


def test_grpo_dense_bonus_noop_for_incorrect_kernel():
    env = _StubEnv(_BUSY)
    dense, fb = grpo._dense_profile_bonus(env, _stub_task(), "s",
                                          _obs(3.0e-4, correct=False), _cfg(0.15))
    assert dense == 0.0 and fb == ""
    assert env.calls == []


def test_grpo_dense_bonus_unmodelable_op_uses_counters_only():
    # op_flop_bytes returns None (no dims) -> no roofline term, but the candidate
    # counters still yield an issue-efficiency dense signal.
    task = types.SimpleNamespace(task_id="stub", operation="mystery", dtype="bf16",
                                 shape=lambda name: None, shapes=[])
    dense, fb = grpo._dense_profile_bonus(_StubEnv(_BUSY), task, "s", _obs(1.0), _cfg(0.15))
    assert dense > 0.0
    assert "bottleneck=" in fb


# --------------------------------------------------------------------------- #
# gating helper + env-profiling-disable (double-count prevention)
# --------------------------------------------------------------------------- #
def test_dense_profile_weight_prefers_config():
    assert grpo._dense_profile_weight(_cfg(0.2)) == 0.2


def test_dense_profile_weight_env_var_fallback(monkeypatch):
    import kore.config as kcfg
    monkeypatch.setattr(kcfg.CONFIG, "profile_reward_weight", 0.0, raising=False)
    monkeypatch.setenv("KORE_PROFILE_REWARD_WEIGHT", "0.07")
    cfg = types.SimpleNamespace(profile_reward_weight=0.0)   # config off -> fall back to env
    assert abs(grpo._dense_profile_weight(cfg) - 0.07) < 1e-9


def test_dense_profile_weight_zero_by_default(monkeypatch):
    import kore.config as kcfg
    monkeypatch.setattr(kcfg.CONFIG, "profile_reward_weight", 0.0, raising=False)
    monkeypatch.delenv("KORE_PROFILE_REWARD_WEIGHT", raising=False)
    assert grpo._dense_profile_weight(types.SimpleNamespace()) == 0.0


def test_make_rollout_env_passthrough_at_weight_zero(monkeypatch):
    """Weight 0 -> KoreEnv built exactly as before (no ``config`` kwarg), so a test
    double that accepts only ``task`` keeps working (byte-for-byte prior behavior)."""
    import kore.env.kore_env as ke
    import kore.config as kcfg
    monkeypatch.setattr(kcfg.CONFIG, "profile_reward_weight", 0.0, raising=False)
    monkeypatch.delenv("KORE_PROFILE_REWARD_WEIGHT", raising=False)

    class FakeEnv:                            # NB: no ``config`` parameter
        def __init__(self, task, gpu=None):
            self.task, self.gpu = task, gpu

    monkeypatch.setattr(ke, "KoreEnv", FakeEnv)
    env = grpo._make_rollout_env("T", _cfg(0.0), serial=True)
    assert isinstance(env, FakeEnv) and env.task == "T"


def test_make_rollout_env_disables_internal_profiling_when_active(monkeypatch):
    """Weight > 0 on the serial path -> KoreEnv gets a config with
    profile_reward_weight=0 so it does NOT double-count the dense bonus."""
    import kore.env.kore_env as ke
    seen = {}

    class FakeEnv:
        def __init__(self, task, config=None, gpu=None):
            seen["config"] = config

    monkeypatch.setattr(ke, "KoreEnv", FakeEnv)
    env = grpo._make_rollout_env("T", _cfg(0.15), serial=True)
    assert isinstance(env, FakeEnv)
    assert seen["config"] is not None
    assert getattr(seen["config"], "profile_reward_weight", None) == 0.0


def test_make_rollout_env_agentic_keeps_internal_profiling(monkeypatch):
    """serial=False (agentic path) never disables KoreEnv internal profiling, even
    when the weight > 0, preserving that path's exact prior behavior."""
    import kore.env.kore_env as ke
    seen = {}

    class FakeEnv:                            # no ``config`` param -> would TypeError if passed
        def __init__(self, task, gpu=None):
            seen["built"] = True

    monkeypatch.setattr(ke, "KoreEnv", FakeEnv)
    env = grpo._make_rollout_env("T", _cfg(0.15), serial=False)
    assert isinstance(env, FakeEnv) and seen.get("built")
