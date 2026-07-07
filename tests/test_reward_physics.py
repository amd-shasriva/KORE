"""CPU-only tests for the physics residual-descent reward (Phase 5).

Verifies the reward (1) delegates every anti-hack / compile / correctness gate
verbatim to the base reward, (2) is monotonic in the amount of NAMED residual
removed, (3) degrades gracefully (flagged) when PMC is unavailable, and (4)
never lets residual credit cross a tier boundary (lexicographic dominance).
"""

from __future__ import annotations

from kore.config import CONFIG
from kore.reward.physics import (
    PhysicsSignal,
    compute_residual_reward,
    named_residual_ms,
    physics_from_measure,
    residual_descent_frac,
)
from kore.reward.reward import Observation, compute_reward


def _correct_obs(wall_ms=1.0):
    return Observation(compiled=True, snr_db=40.0, wall_ms=wall_ms,
                       validation_passed=True, dtype="bf16")


# ------------- gating is delegated verbatim (anti-hack ordering) ------------- #
def test_hack_source_is_floor_and_not_correct():
    obs = _correct_obs()
    sig = PhysicsSignal(t_min_ms=0.5, measured_ms=1.0, stall_frac=0.1, occupancy=0.9)
    rr = compute_residual_reward(obs, sig, source="import aiter\nout = aiter.rms_norm(x,w)",
                                 dtype="bf16")
    assert rr.tier == "hack"
    assert rr.reward == CONFIG.reward_hack
    assert rr.correct is False


def test_compile_fail_delegated():
    obs = Observation(compiled=False, dtype="bf16")
    sig = PhysicsSignal(t_min_ms=0.5, measured_ms=1.0, stall_frac=0.1, occupancy=0.9)
    rr = compute_residual_reward(obs, sig, source="x = 1", dtype="bf16")
    assert rr.tier == "compile_fail"
    assert rr.reward == CONFIG.reward_compile_fail


def test_incorrect_delegated_matches_base():
    obs = Observation(compiled=True, snr_db=5.0, wall_ms=1.0,
                      validation_passed=False, dtype="bf16")
    sig = PhysicsSignal(t_min_ms=0.5, measured_ms=1.0, stall_frac=0.1, occupancy=0.9)
    base = compute_reward(obs, source="x = 1", dtype="bf16")
    rr = compute_residual_reward(obs, sig, source="x = 1", dtype="bf16")
    assert rr.tier == base.tier == "incorrect"
    assert abs(rr.reward - base.reward) < 1e-12


# --------------------- physics credit on the correct tier ------------------- #
def test_monotonic_in_named_residual_removed():
    obs = _correct_obs(wall_ms=1.0)
    big = PhysicsSignal(t_min_ms=0.5, measured_ms=1.0, stall_frac=0.4, occupancy=0.5)
    small = PhysicsSignal(t_min_ms=0.5, measured_ms=1.0, stall_frac=0.05, occupancy=0.9)
    r_big = compute_residual_reward(obs, big, source="", dtype="bf16")
    r_small = compute_residual_reward(obs, small, source="", dtype="bf16")
    assert r_big.tier == r_small.tier == "correct_residual"
    # less named residual -> strictly larger reward
    assert r_small.reward > r_big.reward


def test_rho_in_unit_interval_and_named_le_full():
    sig = PhysicsSignal(t_min_ms=0.5, measured_ms=1.0, stall_frac=0.3, occupancy=0.6)
    rho, pmc = residual_descent_frac(sig, 1.0)
    assert pmc is True and 0.0 < rho <= 1.0
    # eta (full residual) <= rho_named because named residual N <= full residual R
    eta = 0.5 / 1.0
    assert rho >= eta - 1e-9


def test_pmc_unavailable_falls_back_to_eta_flagged():
    obs = _correct_obs(wall_ms=1.0)
    sig = PhysicsSignal(t_min_ms=0.5, measured_ms=1.0, stall_frac=None, occupancy=None)
    rr = compute_residual_reward(obs, sig, source="", dtype="bf16")
    assert rr.tier == "correct_residual"
    assert "no_pmc" in rr.flags
    # eta fallback: credit == physics_weight * (t_min/t_meas) = 1.0 * 0.5
    assert abs(rr.reward - (CONFIG.correctness_weight + 0.5)) < 1e-9


def test_named_residual_clamped():
    # stall+occ_deficit > 1 must clamp to at most the measured wall time
    sig = PhysicsSignal(t_min_ms=0.5, measured_ms=1.0, stall_frac=0.9, occupancy=0.2)
    n = named_residual_ms(1.0, sig)
    assert n is not None and 0.0 <= n <= 1.0


def test_correct_dominates_incorrect_ceiling():
    # worst correct-residual (rho -> 0) still beats the best shaped-incorrect kernel
    obs = _correct_obs(wall_ms=1e9)  # huge wall -> rho ~ 0
    sig = PhysicsSignal(t_min_ms=1e-6, measured_ms=1e9, stall_frac=0.99, occupancy=0.01)
    rr = compute_residual_reward(obs, sig, source="", dtype="bf16")
    incorrect_ceiling = CONFIG.reward_incorrect + CONFIG.eps_shape + CONFIG.format_weight
    assert rr.reward >= CONFIG.correctness_weight - 1e-9 > incorrect_ceiling


def test_no_physics_when_no_roofline():
    obs = _correct_obs(wall_ms=1.0)
    sig = PhysicsSignal(t_min_ms=float("nan"), measured_ms=1.0, stall_frac=0.1, occupancy=0.9)
    rr = compute_residual_reward(obs, sig, source="", dtype="bf16")
    assert rr.tier == "correct_no_physics"
    assert rr.correct is True


def test_physics_from_measure_reads_attrs():
    class M:
        t_min_ms = 0.5
        cand_ms = 1.0
        stall_frac = 0.2
        occupancy = 0.7
    sig = physics_from_measure(M())
    assert sig.t_min_ms == 0.5 and sig.measured_ms == 1.0
    assert sig.stall_frac == 0.2 and sig.occupancy == 0.7


# ---- live-training reward dispatch (compute_kernel_reward) ---- #
from kore.reward.physics import compute_kernel_reward, physics_signal_from_obs  # noqa: E402


class _FakeShape:
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims


class _FakeTask:
    def __init__(self, task_id, operation, dtype, shapes):
        self.task_id = task_id
        self.operation = operation
        self.dtype = dtype
        self._shapes = {s.name: s for s in shapes}

    def shape(self, name):
        return self._shapes.get(name)


def _rms_task():
    return _FakeTask("rmsnorm_x", "rmsnorm", "bf16", [_FakeShape("primary", {"M": 4096, "N": 4096})])


def test_physics_signal_from_obs_builds_from_roofline():
    task = _FakeTask("rmsnorm_x", "rmsnorm", "bf16",
                     [_FakeShape("primary", {"M": 4096, "N": 4096}),
                      _FakeShape("big", {"M": 8192, "N": 8192})])
    obs = Observation(compiled=True, validation_passed=True,
                      wall_by_shape={"primary": 1.0, "big": 1.0}, dtype="bf16")
    sig = physics_signal_from_obs(task, obs, arch="gfx950")
    assert sig is not None and sig.t_min_ms > 0 and sig.measured_ms == 1.0


def test_dispatch_residual_uses_physics_for_modeled_op():
    obs = Observation(compiled=True, snr_db=40.0, validation_passed=True,
                      wall_by_shape={"primary": 1.0}, wall_ms=1.0, dtype="bf16")
    rr = compute_kernel_reward(obs, "kernel src", _rms_task(), mode="residual", dtype="bf16")
    assert rr.correct
    assert rr.tier in ("correct_residual", "correct_no_physics")
    assert rr.reward >= CONFIG.correctness_weight - 1e-9


def test_dispatch_speedup_default_uses_vendor_reward():
    obs = Observation(compiled=True, snr_db=40.0, validation_passed=True,
                      wall_by_shape={"primary": 1.0}, baseline_by_shape={"primary": 2.0},
                      wall_ms=1.0, baseline_ms=2.0, dtype="bf16")
    rr = compute_kernel_reward(obs, "kernel src", _rms_task(), mode="speedup", dtype="bf16")
    assert rr.correct  # 2x speedup, correct-tier


def test_dispatch_residual_falls_back_when_unmodelable():
    # op with no roofline model -> residual mode transparently uses the speedup reward
    task = _FakeTask("weird", "no_such_op", "bf16", [_FakeShape("primary", {})])
    obs = Observation(compiled=True, snr_db=40.0, validation_passed=True,
                      wall_by_shape={"primary": 1.0}, baseline_by_shape={"primary": 2.0},
                      wall_ms=1.0, baseline_ms=2.0, dtype="bf16")
    rr = compute_kernel_reward(obs, "src", task, mode="residual", dtype="bf16")
    assert rr.correct  # fell back to speedup path, still correct-tier


def test_dispatch_preserves_hack_gate_in_both_modes():
    obs = Observation(compiled=True, snr_db=40.0, validation_passed=True,
                      wall_by_shape={"primary": 1.0}, wall_ms=1.0, dtype="bf16")
    hack_src = "import aiter\nout = aiter.rms_norm(x, w)"
    for mode in ("speedup", "residual"):
        rr = compute_kernel_reward(obs, hack_src, _rms_task(), mode=mode, dtype="bf16")
        assert rr.tier == "hack" and rr.reward == CONFIG.reward_hack
