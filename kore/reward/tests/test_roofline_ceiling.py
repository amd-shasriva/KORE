"""Physical-impossibility integrity checks."""

from __future__ import annotations

import math

import pytest

from kore.analysis.roofline import (
    estimate_work,
    evaluate_roofline,
    make_physical_model,
)
from kore.config import CONFIG
from kore.reward.physics import (
    compute_kernel_reward,
    roofline_ceiling_violation_from_obs,
)
from kore.reward.reward import (
    Observation,
    compute_reward,
    roofline_ceiling_violation,
)


MODEL = make_physical_model("mi350x")


class Shape:
    name = "primary"
    dims = {"M": 4096, "N": 4096}


class Task:
    task_id = "rmsnorm_x"
    operation = "rmsnorm"
    dtype = "bf16"
    shapes = [Shape()]

    def shape(self, name):
        return Shape() if name == "primary" else None


def _roofline():
    return evaluate_roofline(
        estimate_work(Task.operation, Shape.dims, Task.dtype), MODEL.for_integrity())


def _correct(wall, *, cold=False):
    return Observation(
        compiled=True,
        validation_passed=True,
        snr_db=40.0,
        wall_ms=wall,
        baseline_ms=1.0,
        wall_by_shape={"primary": wall},
        baseline_by_shape={"primary": 1.0},
        cold_cache_verified=cold,
        dtype="bf16",
    )


def test_scalar_predicate_is_strict_and_validated():
    assert roofline_ceiling_violation(0.70, 1.0, tol=0.25)
    assert not roofline_ceiling_violation(0.75, 1.0, tol=0.25)
    assert not roofline_ceiling_violation(1.0, 1.0, tol=0.25)
    for bad in (-0.1, 1.0, math.nan):
        with pytest.raises(ValueError):
            roofline_ceiling_violation(0.1, 1.0, tol=bad)


def test_scalar_predicate_fails_open_on_unavailable_measurement():
    for measured, floor in (
        (None, 1.0),
        (1.0, None),
        (math.nan, 1.0),
        (1.0, math.nan),
        (0.0, 1.0),
    ):
        assert not roofline_ceiling_violation(measured, floor)


def test_warm_cache_uses_compute_floor_only():
    result = _roofline()
    assert result.t_memory_ms > result.t_compute_ms
    wall = result.t_memory_ms * 0.1
    assert wall > result.t_compute_ms
    violated, reason = roofline_ceiling_violation_from_obs(
        Task(), _correct(wall, cold=False), model=MODEL)
    assert not violated and reason == ""


def test_cold_cache_uses_full_compute_hbm_floor():
    result = _roofline()
    wall = result.t_min_ms * 0.1
    violated, reason = roofline_ceiling_violation_from_obs(
        Task(), _correct(wall, cold=True), model=MODEL)
    assert violated
    assert "compute+HBM cold-cache" in reason
    assert MODEL.for_integrity().fingerprint in reason


def test_compute_floor_rejects_even_without_cold_cache():
    result = _roofline()
    wall = result.t_compute_ms * 0.1
    violated, reason = roofline_ceiling_violation_from_obs(
        Task(), _correct(wall, cold=False), model=MODEL)
    assert violated and "mandatory compute" in reason


def test_unmodeled_operation_fails_open():
    class Unknown(Task):
        operation = "flash_attn_decode"

    violated, reason = roofline_ceiling_violation_from_obs(
        Unknown(), _correct(1e-12, cold=True), model=MODEL)
    assert not violated and reason == ""


def test_compute_reward_scalar_gate_preserves_hack_floor():
    obs = _correct(0.01)
    result = compute_reward(
        obs,
        source="x=1",
        dtype="bf16",
        roofline_gate=True,
        t_min_ms=1.0,
    )
    assert result.tier == "hack"
    assert result.reward == CONFIG.reward_hack


def test_kernel_gate_is_transparent_when_off():
    result = _roofline()
    obs = _correct(result.t_min_ms * 0.1, cold=True)
    off = compute_kernel_reward(
        obs, "x=1", Task(), model=MODEL, roofline_gate=False, dtype="bf16")
    assert off.correct and off.tier != "hack"


def test_kernel_gate_rejects_in_both_reward_modes():
    result = _roofline()
    obs = _correct(result.t_min_ms * 0.1, cold=True)
    for mode in ("speedup", "residual"):
        reward = compute_kernel_reward(
            obs,
            "x=1",
            Task(),
            mode=mode,
            model=MODEL,
            roofline_gate=True,
            dtype="bf16",
        )
        assert reward.tier == "hack"
        assert reward.reward == CONFIG.reward_hack
        assert "roofline_ceiling" in reward.flags


def test_integrity_gate_keeps_lexicographic_order():
    result = _roofline()
    hack = compute_kernel_reward(
        _correct(result.t_min_ms * 0.1, cold=True),
        "x=1",
        Task(),
        model=MODEL,
        roofline_gate=True,
        dtype="bf16",
    )
    compile_fail = compute_reward(Observation(compiled=False), source="x=1")
    incorrect = compute_reward(
        Observation(compiled=True, snr_db=1.0, validation_passed=False),
        source="x=1",
    )
    correct = compute_kernel_reward(
        _correct(result.t_min_ms * 2.0, cold=True),
        "x=1",
        Task(),
        model=MODEL,
        roofline_gate=True,
        dtype="bf16",
    )
    assert hack.reward < compile_fail.reward < incorrect.reward < correct.reward
