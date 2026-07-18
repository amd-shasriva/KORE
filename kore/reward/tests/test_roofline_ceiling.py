"""CPU-only tests for the roofline SPEED-OF-LIGHT hack-ceiling gate (anti-reward-hack).

The gate rejects any candidate whose MEASURED time is physically impossible below the
roofline lower bound ``T_min`` (throughput above the speed of light), which can only
come from a measurement exploit (warm cache / do-less / forged timer -- the class of
hack that inflated Sakana's CUDA agent and CUDA-L1). These tests verify:

  * the pure predicate flags super-SOL times, passes physical ones, and fail-opens on
    junk input / ``tol >= 1``;
  * the ``compute_reward`` and ``compute_kernel_reward`` gates are OFF by default and
    BYTE-IDENTICAL when off (no behavior change for the live run);
  * when ON, a super-SOL time is dropped to the hack floor in BOTH reward modes, while
    a physical time is untouched, and the lexicographic ordering
    (hack < compile < incorrect < correct) is preserved.

No GPU / no network: the roofline ``T_min`` is computed analytically and ``arch`` is
pinned to ``gfx950`` so ``detect_arch`` never shells out.
"""

from __future__ import annotations

import math

from kore.config import CONFIG
from kore.reward.physics import (
    compute_kernel_reward,
    roofline_ceiling_violation_from_obs,
)
from kore.reward.reward import (
    DEFAULT_ROOFLINE_TOL,
    Observation,
    compute_reward,
    roofline_ceiling_violation,
)

ARCH = "gfx950"


# --------------------------------------------------------------------------- #
# Test doubles (mirror the live KoreEnv task/shape duck-typing)
# --------------------------------------------------------------------------- #
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


def _rms_task(shapes=None):
    shapes = shapes or [_FakeShape("primary", {"M": 4096, "N": 4096})]
    return _FakeTask("rmsnorm_x", "rmsnorm", "bf16", shapes)


def _t_min_ms(task, name="primary"):
    """Analytic roofline T_min for a shape (no GPU)."""
    from kore.analysis.rooflines import resolve_peaks, roofline, shape_to_str

    sh = task.shape(name)
    peaks = resolve_peaks(ARCH)
    rf = roofline(task.task_id, task.operation, task.dtype,
                  shape_to_str(sh.dims), sh.dims, peaks, ARCH)
    assert rf is not None and rf.t_min_ms > 0
    return rf.t_min_ms


def _correct_obs(wall_ms, baseline_ms=None):
    return Observation(compiled=True, snr_db=40.0, wall_ms=wall_ms,
                       baseline_ms=baseline_ms, validation_passed=True, dtype="bf16")


# --------------------------------------------------------------------------- #
# 1. Pure predicate: roofline_ceiling_violation(measured_ms, t_min_ms, tol)
# --------------------------------------------------------------------------- #
def test_super_sol_time_is_flagged():
    # 10x faster than the speed of light -> impossible -> violation
    assert roofline_ceiling_violation(0.1, 1.0) is True


def test_physical_time_is_not_flagged():
    assert roofline_ceiling_violation(1.0, 1.0) is False       # exactly on the roofline
    assert roofline_ceiling_violation(2.0, 1.0) is False       # slower than the floor
    assert roofline_ceiling_violation(1e9, 1e-6) is False      # far below the roofline


def test_tolerance_band():
    # tol=0.25 -> reject only below 0.75*T_min
    t_min = 1.0
    assert roofline_ceiling_violation(0.80, t_min, tol=0.25) is False   # within tol
    assert roofline_ceiling_violation(0.70, t_min, tol=0.25) is True    # beyond tol
    # exact boundary m == t*(1-tol) is NOT a violation (strict <)
    assert roofline_ceiling_violation(0.75, t_min, tol=0.25) is False


def test_fail_open_on_bad_input():
    # missing / non-positive / NaN inputs can never be adjudicated -> never a violation
    for m, t in [(None, 1.0), (1.0, None), (0.0, 1.0), (-1.0, 1.0),
                 (1.0, 0.0), (1.0, -1.0), (math.nan, 1.0), (1.0, math.nan)]:
        assert roofline_ceiling_violation(m, t) is False


def test_tol_ge_one_never_fires():
    # tol>=1 collapses the threshold to <=0 -> fail-open (never rejects), even for ~0 time
    assert roofline_ceiling_violation(1e-12, 1.0, tol=1.0) is False
    assert roofline_ceiling_violation(1e-12, 1.0, tol=1.5) is False


def test_negative_tol_clamped_to_zero():
    # a negative tol is clamped to 0 -> any time strictly below T_min is a violation
    assert roofline_ceiling_violation(0.999, 1.0, tol=-0.5) is True
    assert roofline_ceiling_violation(1.0, 1.0, tol=-0.5) is False


def test_default_tol_constant():
    assert 0.0 < DEFAULT_ROOFLINE_TOL < 1.0


# --------------------------------------------------------------------------- #
# 2. compute_reward gate: OFF by default + byte-identical when off
# --------------------------------------------------------------------------- #
def test_compute_reward_gate_off_is_byte_identical():
    # An impossibly-fast correct kernel: with the gate OFF (default) the result MUST be
    # unchanged, even when a t_min_ms is supplied.
    obs = _correct_obs(wall_ms=0.001, baseline_ms=1.0)
    base = compute_reward(obs, source="", dtype="bf16")
    same = compute_reward(obs, source="", dtype="bf16",
                          roofline_gate=False, t_min_ms=0.5, roofline_tol=0.25)
    assert same == base                       # dataclass equality over all fields
    assert same.tier == "correct_timed" and same.correct is True
    assert "roofline_ceiling" not in same.flags


def test_compute_reward_gate_on_rejects_super_sol():
    obs = _correct_obs(wall_ms=0.001, baseline_ms=1.0)   # 0.001 ms << T_min
    rr = compute_reward(obs, source="", dtype="bf16",
                        roofline_gate=True, t_min_ms=0.5, roofline_tol=0.25)
    assert rr.tier == "hack"
    assert rr.reward == CONFIG.reward_hack
    assert rr.correct is False
    assert "hack" in rr.flags and "roofline_ceiling" in rr.flags


def test_compute_reward_gate_on_passes_physical():
    # measured AT/above the roofline is physical -> untouched (identical to gate-off)
    obs = _correct_obs(wall_ms=0.6, baseline_ms=1.0)
    off = compute_reward(obs, source="", dtype="bf16")
    on = compute_reward(obs, source="", dtype="bf16",
                        roofline_gate=True, t_min_ms=0.5, roofline_tol=0.25)
    assert on == off and on.tier == "correct_timed" and on.correct is True


def test_compute_reward_gate_noop_without_tmin():
    # gate on but no roofline supplied -> fail-open (cannot adjudicate)
    obs = _correct_obs(wall_ms=0.001, baseline_ms=1.0)
    rr = compute_reward(obs, source="", dtype="bf16", roofline_gate=True, t_min_ms=None)
    assert rr.tier == "correct_timed" and rr.correct is True


def test_compute_reward_source_hack_takes_precedence_over_ceiling():
    # a candidate that BOTH delegates and clocks a super-SOL time keeps the specific
    # source-hack reason (the scan runs first); still the hack floor either way.
    obs = _correct_obs(wall_ms=0.001, baseline_ms=1.0)
    rr = compute_reward(obs, source="import aiter\nout = aiter.rms_norm(x, w)",
                        dtype="bf16", roofline_gate=True, t_min_ms=0.5)
    assert rr.tier == "hack" and rr.reward == CONFIG.reward_hack
    assert "roofline_ceiling" not in rr.flags       # source reason won, not the ceiling
    assert "aiter" in rr.detail


def test_compute_reward_uses_min_shape_for_ceiling():
    # multi-shape: the gate must look at the FASTEST (min) shape, where an exploit shows
    obs = Observation(compiled=True, snr_db=40.0, validation_passed=True, dtype="bf16",
                      wall_by_shape={"slow": 2.0, "fast": 0.001},
                      baseline_by_shape={"slow": 4.0, "fast": 4.0})
    rr = compute_reward(obs, source="", dtype="bf16",
                        roofline_gate=True, t_min_ms=0.5, roofline_tol=0.25)
    assert rr.tier == "hack" and "roofline_ceiling" in rr.flags


def test_ceiling_hack_preserves_lexicographic_order():
    # hack (ceiling) < compile_fail < incorrect < correct, by construction of the ladder
    ceiling = compute_reward(_correct_obs(0.001, 1.0), "", dtype="bf16",
                             roofline_gate=True, t_min_ms=0.5)
    compile_fail = compute_reward(Observation(compiled=False, dtype="bf16"), "", dtype="bf16")
    incorrect = compute_reward(Observation(compiled=True, snr_db=1.0, wall_ms=1.0,
                                           validation_passed=False, dtype="bf16"),
                               "", dtype="bf16")
    correct = compute_reward(_correct_obs(0.6, 1.0), "", dtype="bf16",
                             roofline_gate=True, t_min_ms=0.5)
    assert ceiling.reward < compile_fail.reward < incorrect.reward < correct.reward
    assert ceiling.reward == CONFIG.reward_hack


# --------------------------------------------------------------------------- #
# 3. roofline_ceiling_violation_from_obs: per-shape detection (analytic T_min)
# --------------------------------------------------------------------------- #
def test_from_obs_flags_super_sol_shape():
    task = _rms_task()
    t_min = _t_min_ms(task)
    obs = Observation(compiled=True, validation_passed=True, dtype="bf16",
                      wall_by_shape={"primary": t_min * 0.1})     # 10x over SOL
    violated, why = roofline_ceiling_violation_from_obs(task, obs, tol=0.25, arch=ARCH)
    assert violated is True and "primary" in why and "T_min" in why


def test_from_obs_passes_physical_shape():
    task = _rms_task()
    t_min = _t_min_ms(task)
    obs = Observation(compiled=True, validation_passed=True, dtype="bf16",
                      wall_by_shape={"primary": t_min * 2.0})     # physical
    violated, why = roofline_ceiling_violation_from_obs(task, obs, tol=0.25, arch=ARCH)
    assert violated is False and why == ""


def test_from_obs_flags_if_any_shape_violates():
    task = _rms_task([_FakeShape("a", {"M": 4096, "N": 4096}),
                      _FakeShape("b", {"M": 4096, "N": 4096})])
    t_min = _t_min_ms(task, "a")
    obs = Observation(compiled=True, validation_passed=True, dtype="bf16",
                      wall_by_shape={"a": t_min * 3.0, "b": t_min * 0.05})  # b impossible
    violated, why = roofline_ceiling_violation_from_obs(task, obs, tol=0.25, arch=ARCH)
    assert violated is True and "b" in why


def test_from_obs_fail_open_when_unmodelable():
    task = _FakeTask("weird", "no_such_op", "bf16", [_FakeShape("primary", {})])
    obs = Observation(compiled=True, validation_passed=True, dtype="bf16",
                      wall_by_shape={"primary": 1e-9})
    violated, why = roofline_ceiling_violation_from_obs(task, obs, tol=0.25, arch=ARCH)
    assert violated is False and why == ""


# --------------------------------------------------------------------------- #
# 4. compute_kernel_reward gate: OFF by default + rejects in BOTH modes when ON
# --------------------------------------------------------------------------- #
def _impossible_obs(task):
    t_min = _t_min_ms(task)
    return Observation(compiled=True, snr_db=40.0, validation_passed=True, dtype="bf16",
                       wall_by_shape={"primary": t_min * 0.05}, wall_ms=t_min * 0.05,
                       baseline_by_shape={"primary": 1.0}, baseline_ms=1.0)


def _physical_obs(task):
    t_min = _t_min_ms(task)
    return Observation(compiled=True, snr_db=40.0, validation_passed=True, dtype="bf16",
                       wall_by_shape={"primary": t_min * 3.0}, wall_ms=t_min * 3.0,
                       baseline_by_shape={"primary": 1.0}, baseline_ms=1.0)


def test_kernel_gate_off_default_rewards_super_sol_in_both_modes():
    # WITHOUT the gate the exploit is REWARDED (this is the vulnerability the gate
    # closes): speedup mode -> correct+fast; residual mode -> rho capped at 1.0.
    task = _rms_task()
    obs = _impossible_obs(task)
    for mode in ("speedup", "residual"):
        rr = compute_kernel_reward(obs, "kernel src", task, mode=mode, dtype="bf16", arch=ARCH)
        assert rr.correct is True and rr.tier != "hack"
        assert rr.reward >= CONFIG.correctness_weight - 1e-9


def test_kernel_gate_on_rejects_super_sol_in_both_modes():
    task = _rms_task()
    obs = _impossible_obs(task)
    for mode in ("speedup", "residual"):
        rr = compute_kernel_reward(obs, "kernel src", task, mode=mode, dtype="bf16",
                                   arch=ARCH, roofline_gate=True)
        assert rr.tier == "hack", mode
        assert rr.reward == CONFIG.reward_hack
        assert "roofline_ceiling" in rr.flags


def test_kernel_gate_on_passes_physical_in_both_modes():
    task = _rms_task()
    obs = _physical_obs(task)
    for mode in ("speedup", "residual"):
        off = compute_kernel_reward(obs, "kernel src", task, mode=mode, dtype="bf16", arch=ARCH)
        on = compute_kernel_reward(obs, "kernel src", task, mode=mode, dtype="bf16",
                                   arch=ARCH, roofline_gate=True)
        assert on.correct is True and on.tier != "hack"
        assert on == off                        # gate transparent for physical times


def test_kernel_gate_off_is_byte_identical():
    task = _rms_task()
    obs = _impossible_obs(task)
    for mode in ("speedup", "residual"):
        base = compute_kernel_reward(obs, "kernel src", task, mode=mode, dtype="bf16", arch=ARCH)
        same = compute_kernel_reward(obs, "kernel src", task, mode=mode, dtype="bf16",
                                     arch=ARCH, roofline_gate=False)
        assert same == base


def test_kernel_gate_fail_open_when_unmodelable():
    # unmodelable op -> gate cannot adjudicate -> falls through to the normal reward
    task = _FakeTask("weird", "no_such_op", "bf16", [_FakeShape("primary", {})])
    obs = Observation(compiled=True, snr_db=40.0, validation_passed=True, dtype="bf16",
                      wall_by_shape={"primary": 1e-9}, wall_ms=1e-9,
                      baseline_by_shape={"primary": 1.0}, baseline_ms=1.0)
    rr = compute_kernel_reward(obs, "src", task, mode="speedup", dtype="bf16",
                               arch=ARCH, roofline_gate=True)
    assert rr.tier != "hack" and rr.correct is True
