"""CPU-only tests for the counter-grounded LIVE ``rho`` path in ``kore.reward.whitebox``.

Today the rollout calls ``phi_potential(task, obs)`` WITHOUT counters, so the shaping
potential degrades to the PMC-free ``eta = T_min/T_meas``. Once the orchestrator threads
a per-turn rocprofv3 counter dict, ``phi_potential(task, obs, counters=...)`` must return
the NAMED-RESIDUAL ``rho = T_min/(T_min+N)`` instead (the R^2~0.98 signal). These tests
lock in that math and API on CPU with synthetic counters (no GPU / no rocprofv3):

  * no counters  -> eta (flat SOL attainment, ``pmc_used=False``);
  * with counters -> rho (named residual, ``pmc_used=True``) and ``eta <= rho <= 1``;
  * the DEFAULT (no-counter) behavior is unchanged (regression guard);
  * ``physics_signal_from_counters`` attaches stall/occupancy from both the derived
    (``MemUnitStalled``/``OccupancyPercent``) and the raw resource-field vocabularies,
    and degrades to the eta base when counters are absent/unusable / the op is unmodeled.
"""

from __future__ import annotations

from kore.reward import whitebox as wb
from kore.reward.physics import PhysicsSignal, residual_descent_frac
from kore.reward.reward import Observation

ARCH = "gfx950"


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
    return _FakeTask("rmsnorm_x", "rmsnorm", "bf16",
                     [_FakeShape("primary", {"M": 4096, "N": 4096})])


def _t_min_ms(task, name="primary"):
    from kore.analysis.rooflines import resolve_peaks, roofline, shape_to_str

    sh = task.shape(name)
    peaks = resolve_peaks(ARCH)
    rf = roofline(task.task_id, task.operation, task.dtype,
                  shape_to_str(sh.dims), sh.dims, peaks, ARCH)
    assert rf is not None and rf.t_min_ms > 0
    return rf.t_min_ms


def _obs_at_eta(task, eta):
    """Correct+timed observation whose worst-shape eta == ``eta`` (wall = T_min/eta)."""
    wall = _t_min_ms(task) / eta
    return Observation(compiled=True, snr_db=40.0, validation_passed=True, dtype="bf16",
                       wall_by_shape={"primary": wall}, wall_ms=wall)


# --------------------------------------------------------------------------- #
# 1. phi_potential: no counters -> eta ; with counters -> named-residual rho
# --------------------------------------------------------------------------- #
def test_phi_no_counters_is_eta():
    task = _rms_task()
    obs = _obs_at_eta(task, eta=0.25)                      # wall = 4 * T_min
    phi = wb.phi_potential(task, obs, arch=ARCH)           # counters default None
    assert phi is not None
    assert abs(phi - 0.25) < 1e-9                          # exactly eta = T_min/T_meas
    # and it is flagged as the PMC-free path
    rho, pmc_used = wb.whitebox_attainment(task, obs, None, ARCH)
    assert pmc_used is False and abs(rho - 0.25) < 1e-9


def test_phi_with_counters_is_named_residual_rho():
    task = _rms_task()
    obs = _obs_at_eta(task, eta=0.25)                      # wall = 4 * T_min, eta = 0.25
    # MemUnitStalled=20% -> stall=0.2 ; OccupancyPercent=70% -> occ_deficit=0.3
    # N = (0.2 + 0.3) * T_meas = 0.5 * (4*T_min) = 2*T_min  (<= residual R = 3*T_min)
    # rho = T_min / (T_min + 2*T_min) = 1/3  (independent of the absolute T_min)
    counters = {"MemUnitStalled": 20.0, "OccupancyPercent": 70.0}
    phi = wb.phi_potential(task, obs, counters=counters, arch=ARCH)
    assert phi is not None
    assert abs(phi - (1.0 / 3.0)) < 1e-9                   # rho, NOT eta(0.25)
    assert phi > 0.25                                      # strictly denser than eta
    rho, pmc_used = wb.whitebox_attainment(task, obs, counters, ARCH)
    assert pmc_used is True and abs(rho - (1.0 / 3.0)) < 1e-9


def test_rho_ge_eta_invariant():
    task = _rms_task()
    obs = _obs_at_eta(task, eta=0.5)
    eta = wb.phi_potential(task, obs, arch=ARCH)
    rho = wb.phi_potential(task, obs, counters={"MemUnitStalled": 30.0,
                                                "OccupancyPercent": 60.0}, arch=ARCH)
    assert eta is not None and rho is not None
    assert rho >= eta - 1e-12                              # named residual N <= full R


def test_named_residual_clamp_collapses_rho_to_eta():
    # when the named residual would exceed the full residual it clamps, so rho == eta
    task = _rms_task()
    obs = _obs_at_eta(task, eta=0.8)                       # small residual R = 0.25*T_min
    eta = wb.phi_potential(task, obs, arch=ARCH)
    rho = wb.phi_potential(task, obs, counters={"MemUnitStalled": 90.0,
                                                "OccupancyPercent": 10.0}, arch=ARCH)
    assert abs(rho - eta) < 1e-9                           # clamped -> degrades to eta


# --------------------------------------------------------------------------- #
# 2. DEFAULT (no-counter) behavior is UNCHANGED (regression guard)
# --------------------------------------------------------------------------- #
def test_default_no_counter_behavior_unchanged():
    task = _rms_task()
    obs = _obs_at_eta(task, eta=0.4)
    base = wb.phi_potential(task, obs, arch=ARCH)
    assert base is not None
    # None and {} must both behave exactly like "no counters" (still eta)
    assert wb.phi_potential(task, obs, counters=None, arch=ARCH) == base
    assert wb.phi_potential(task, obs, counters={}, arch=ARCH) == base
    # counters present but UNUSABLE (no stall/occ vocabulary) also stays on eta
    assert wb.phi_potential(task, obs, counters={"SOME_UNKNOWN": 1}, arch=ARCH) == base


# --------------------------------------------------------------------------- #
# 3. physics_signal_from_counters: attaches terms / degrades gracefully
# --------------------------------------------------------------------------- #
def test_signal_from_counters_attaches_derived_terms():
    task = _rms_task()
    obs = _obs_at_eta(task, eta=0.3)
    sig = wb.physics_signal_from_counters(
        task, obs, {"MemUnitStalled": 25.0, "OccupancyPercent": 80.0}, ARCH)
    assert isinstance(sig, PhysicsSignal)
    assert abs(sig.stall_frac - 0.25) < 1e-9
    assert abs(sig.occupancy - 0.80) < 1e-9
    assert sig.t_min_ms > 0 and sig.measured_ms > 0
    # this signal drives the rho (pmc) path
    rho, pmc = residual_descent_frac(sig)
    assert pmc is True and 0.0 < rho <= 1.0


def test_signal_from_counters_empty_is_eta_base():
    from kore.reward.physics import physics_signal_from_obs

    task = _rms_task()
    obs = _obs_at_eta(task, eta=0.3)
    base = physics_signal_from_obs(task, obs, ARCH)
    got = wb.physics_signal_from_counters(task, obs, None, ARCH)
    assert got is not None and got.stall_frac is None and got.occupancy is None
    assert abs(got.t_min_ms - base.t_min_ms) < 1e-15
    assert abs(got.measured_ms - base.measured_ms) < 1e-15


def test_signal_from_counters_unmodelable_is_none():
    task = _FakeTask("weird", "no_such_op", "bf16", [_FakeShape("primary", {})])
    obs = Observation(compiled=True, validation_passed=True, dtype="bf16",
                      wall_by_shape={"primary": 1.0})
    assert wb.physics_signal_from_counters(
        task, obs, {"MemUnitStalled": 10.0}, ARCH) is None


# --------------------------------------------------------------------------- #
# 4. Counter -> stall / occupancy extraction (both vocabularies)
# --------------------------------------------------------------------------- #
def test_stall_frac_derived_and_raw():
    assert abs(wb.stall_frac_from_counters({"MemUnitStalled": 25.0}) - 0.25) < 1e-9
    raw = {"SQ_WAIT_INST_ANY": 10, "SQ_INSTS_VALU": 90}
    assert abs(wb.stall_frac_from_counters(raw) - 0.10) < 1e-9
    assert wb.stall_frac_from_counters({}) is None


def test_occupancy_derived_and_resource_field():
    # derived percentage
    assert abs(wb.occupancy_from_counters({"OccupancyPercent": 60.0}) - 0.6) < 1e-9
    # RAW resource fields -> est_occupancy waves/SIMD model (this path was previously a
    # silent no-op: float(Occupancy) threw and was swallowed; it must now return a frac)
    occ = wb.occupancy_from_counters({"vgpr_count": 100, "lds_bytes": 0, "num_warps": 4})
    assert occ is not None and 0.0 < occ <= 1.0
    # no derived + no resource fields -> None (never fabricated)
    assert wb.occupancy_from_counters({"SQ_INSTS_VALU": 5}) is None


def test_clamp_out_of_range_counter_values():
    # a malformed >100% derived metric clamps into [0,1] rather than exploding rho
    assert wb.stall_frac_from_counters({"MemUnitStalled": 250.0}) == 1.0
    assert wb.occupancy_from_counters({"OccupancyPercent": -5.0}) == 0.0
