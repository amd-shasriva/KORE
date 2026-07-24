"""Integrity/shaping separation and reward-tier tests."""

from __future__ import annotations

import dataclasses
import copy
import math

import pytest

from kore.analysis.roofline import make_physical_model
from kore.config import CONFIG
from kore.reward.physics import (
    PhysicsSignal,
    compute_kernel_reward,
    compute_residual_reward,
    model_from_config,
    named_residual_ms,
    physics_from_measure,
    physics_signal_from_obs,
    residual_descent_frac,
    roofline_ceiling_violation_from_obs,
)
from kore.reward.reward import (
    Observation,
    compute_reward,
    roofline_ceiling_violation,
    validate_reward_config,
)
from kore.reward.shaping import FamilyShapingEvidence


MODEL = make_physical_model("mi350x")


class Shape:
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims


class Task:
    task_id = "rmsnorm_x"
    operation = "rmsnorm"
    dtype = "bf16"

    def __init__(self):
        self._shape = Shape("primary", {"M": 4096, "N": 4096})
        self.shapes = [self._shape]

    def shape(self, name):
        return self._shape if name == "primary" else None


def _evidence(family="norm", fingerprint=MODEL.fingerprint):
    return FamilyShapingEvidence(
        family=family,
        report_fingerprint="sha256:report",
        model_fingerprint=fingerprint,
        n_points=100,
        n_task_clusters=8,
        normalized_cv_r2=0.8,
        baseline_cv_r2=0.1,
        ci95=(0.5, 0.9),
        adjusted_p=0.01,
        coefficients=(0.5, 0.25, 0.05),
    )


def _correct_obs(wall=1.0, baseline=2.0):
    return Observation(
        compiled=True,
        snr_db=40.0,
        validation_passed=True,
        wall_ms=wall,
        baseline_ms=baseline,
        wall_by_shape={"primary": wall},
        baseline_by_shape={"primary": baseline},
        dtype="bf16",
    )


def _signal(**kwargs):
    values = dict(
        t_min_ms=0.5,
        measured_ms=1.0,
        model_fingerprint=MODEL.fingerprint,
        family="norm",
        stall_frac=0.2,
        occupancy=0.7,
    )
    values.update(kwargs)
    return PhysicsSignal(**values)


def test_physics_signal_rejects_nonfinite_and_out_of_range_values():
    with pytest.raises(ValueError):
        _signal(t_min_ms=math.nan)
    with pytest.raises(ValueError):
        _signal(stall_frac=1.1)
    with pytest.raises(ValueError):
        _signal(measured_ms=0.0)


def test_diagnostic_eta_is_bounded_without_evidence():
    eta, used = residual_descent_frac(_signal())
    assert eta == pytest.approx(0.5)
    assert used is False
    super_sol, used = residual_descent_frac(_signal(measured_ms=0.1))
    assert super_sol == 1.0 and used is False


def test_named_residual_unavailable_without_matching_evidence():
    signal = _signal()
    assert named_residual_ms(1.0, signal) is None
    assert named_residual_ms(
        1.0, signal, _evidence(family="gemm")) is None
    assert named_residual_ms(
        1.0, signal, _evidence(fingerprint="sha256:other")) is None


def test_passing_family_evidence_enables_bounded_residual_prediction():
    signal = _signal(stall_frac=0.2, occupancy=0.7)
    evidence = _evidence()
    # gap = .5*.2 + .25*.3 + .05 = .225; credit = .775
    credit, used = residual_descent_frac(signal, evidence=evidence)
    assert used is True
    assert credit == pytest.approx(0.775)
    assert named_residual_ms(1.0, signal, evidence) == pytest.approx(0.225)


def test_residual_reward_falls_back_to_speedup_without_evidence():
    obs = _correct_obs()
    base = compute_reward(obs, source="x=1", dtype="bf16")
    result = compute_residual_reward(
        obs, _signal(), source="x=1", dtype="bf16")
    assert result.reward == base.reward
    assert result.tier == base.tier
    assert "physics_shaping_disabled" in result.flags


def test_residual_reward_uses_evidence_only_on_correct_tier():
    obs = _correct_obs()
    result = compute_residual_reward(
        obs, _signal(), source="x=1", dtype="bf16", evidence=_evidence())
    assert result.tier == "correct_residual"
    assert result.reward == pytest.approx(CONFIG.correctness_weight + 0.775)

    bad = Observation(
        compiled=True, snr_db=1.0, validation_passed=False, wall_ms=1.0)
    assert compute_residual_reward(
        bad, _signal(), source="x=1", dtype="bf16", evidence=_evidence()
    ).tier == "incorrect"


def test_hack_and_compile_tiers_are_delegated_before_physics():
    hack = compute_residual_reward(
        _correct_obs(),
        _signal(),
        source="import aiter\nout=aiter.rms_norm(x,w)",
        dtype="bf16",
        evidence=_evidence(),
    )
    assert hack.tier == "hack" and hack.reward == CONFIG.reward_hack
    compile_fail = compute_residual_reward(
        Observation(compiled=False), _signal(), source="x=1", evidence=_evidence())
    assert compile_fail.tier == "compile_fail"


def test_reward_tier_validation_uses_runtime_errors_not_asserts():
    with pytest.raises(ValueError, match="incorrect"):
        dataclasses.replace(
            CONFIG, correctness_weight=0.05, eps_shape=0.05, format_weight=0.02)
    invalid = copy.copy(CONFIG)
    invalid.correctness_weight = 0.05
    with pytest.raises(ValueError):
        validate_reward_config(invalid)


def test_model_from_config_matches_pinned_default():
    selected = model_from_config(CONFIG)
    assert selected.fingerprint == MODEL.fingerprint


def test_physics_signal_from_obs_uses_explicit_model():
    signal = physics_signal_from_obs(Task(), _correct_obs(), MODEL)
    assert signal is not None
    assert signal.model_fingerprint == MODEL.fingerprint
    assert signal.family == "norm"
    assert 0.0 < signal.t_min_ms < signal.measured_ms


def test_physics_from_measure_carries_fingerprint():
    class Measure:
        task_id = "rmsnorm_x"
        t_min_ms = 0.5
        cand_ms = 1.0
        stall_frac = 0.2
        occupancy = 0.7

    signal = physics_from_measure(Measure(), MODEL)
    assert signal.model_fingerprint == MODEL.fingerprint
    assert signal.family == "norm"


def test_integrity_uses_compute_floor_without_cold_cache():
    task = Task()
    diagnostic = physics_signal_from_obs(task, _correct_obs(), MODEL)
    # A memory-bound T_min is not a sound warm-cache floor.
    warm = Observation(
        compiled=True,
        validation_passed=True,
        wall_by_shape={"primary": diagnostic.t_min_ms * 0.5},
        cold_cache_verified=False,
    )
    violated, _ = roofline_ceiling_violation_from_obs(
        task, warm, model=MODEL)
    assert violated is False


def test_integrity_uses_hbm_floor_with_explicit_cold_cache():
    task = Task()
    diagnostic = physics_signal_from_obs(task, _correct_obs(), MODEL)
    cold = Observation(
        compiled=True,
        validation_passed=True,
        wall_by_shape={"primary": diagnostic.t_min_ms * 0.1},
        cold_cache_verified=True,
    )
    violated, reason = roofline_ceiling_violation_from_obs(
        task, cold, model=MODEL)
    assert violated is True
    assert "cold-cache" in reason and MODEL.for_integrity().fingerprint in reason


def test_integrity_tolerance_is_validated():
    with pytest.raises(ValueError):
        roofline_ceiling_violation(0.1, 1.0, tol=-0.1)
    with pytest.raises(ValueError):
        roofline_ceiling_violation(0.1, 1.0, tol=1.0)


def test_compute_kernel_residual_mode_stays_disabled_without_artifact():
    result = compute_kernel_reward(
        _correct_obs(), "x=1", Task(), mode="residual", dtype="bf16",
        model=MODEL, physics_config=CONFIG)
    assert result.correct
    assert result.tier == "correct_timed"
    assert "physics_shaping_disabled" in result.flags


def test_compute_kernel_integrity_gate_rejects_cold_super_sol():
    task = Task()
    diagnostic = physics_signal_from_obs(task, _correct_obs(), MODEL)
    obs = _correct_obs(wall=diagnostic.t_min_ms * 0.1)
    obs.cold_cache_verified = True
    result = compute_kernel_reward(
        obs, "x=1", task, model=MODEL, roofline_gate=True, dtype="bf16")
    assert result.tier == "hack"
    assert "roofline_ceiling" in result.flags
