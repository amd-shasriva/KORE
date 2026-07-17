"""CPU tests for the P0 white-box reward + potential-based shaping modules.

No GPU / no counters collection: the counter dicts are synthetic, and the physics
signals are constructed directly, so these validate the pure logic that turns
rocprofv3 counters into the NAMED-residual credit and the policy-invariant shaping.
"""

from __future__ import annotations

import math

from kore.reward.physics import PhysicsSignal, residual_descent_frac
from kore.reward import whitebox as wb
from kore.reward import shaping as sh


# --------------------------------------------------------------------------- #
# 1. Named-residual comes online and is never harsher than the eta fallback
# --------------------------------------------------------------------------- #
def test_named_residual_beats_flat_eta_contrast():
    # eta path (no counters): rho = T_min/T_meas = 0.5
    eta_sig = PhysicsSignal(t_min_ms=1.0, measured_ms=2.0)
    eta, pmc = residual_descent_frac(eta_sig)
    assert pmc is False and abs(eta - 0.5) < 1e-9

    # named-residual path: N = (stall + occ_deficit)*T_meas = (0.1+0.1)*2 = 0.4
    # rho = T_min/(T_min+N) = 1/1.4 ~= 0.714 -> strictly higher contrast than eta,
    # and the theorem eta <= rho <= 1 holds.
    named = PhysicsSignal(t_min_ms=1.0, measured_ms=2.0, stall_frac=0.1, occupancy=0.9)
    rho, pmc2 = residual_descent_frac(named)
    assert pmc2 is True
    assert eta <= rho <= 1.0
    assert abs(rho - (1.0 / 1.4)) < 1e-9


# --------------------------------------------------------------------------- #
# 2. Counter -> normalized stall / occupancy extraction (derived preferred)
# --------------------------------------------------------------------------- #
def test_stall_frac_prefers_derived_then_raw():
    # derived MemUnitStalled (percentage) wins
    assert abs(wb.stall_frac_from_counters({"MemUnitStalled": 25.0}) - 0.25) < 1e-9
    # raw fallback: SQ_WAIT_INST_ANY / (issued + wait)
    raw = {"SQ_WAIT_INST_ANY": 10, "SQ_INSTS_VALU": 90}
    assert abs(wb.stall_frac_from_counters(raw) - (10.0 / 100.0)) < 1e-9
    assert wb.stall_frac_from_counters({}) is None


def test_occupancy_prefers_derived():
    assert abs(wb.occupancy_from_counters({"OccupancyPercent": 60.0}) - 0.6) < 1e-9
    # no derived + no resource fields -> None (never fabricated)
    assert wb.occupancy_from_counters({"SQ_INSTS_VALU": 5}) is None


# --------------------------------------------------------------------------- #
# 3. Hack-resistant structural score: degenerate kernel ~0, efficient kernel ~1
# --------------------------------------------------------------------------- #
def test_structural_score_is_hack_resistant():
    # memset/do-less kernel: almost all cycles WAITING, ~no issued work -> ~0
    degenerate = {"SQ_WAIT_INST_ANY": 100000, "SQ_INSTS_VALU": 1, "SQ_INSTS_VMEM": 1}
    s_bad = wb.whitebox_structural_score(degenerate)
    assert s_bad is not None and s_bad < 0.05

    # well-scheduled kernel: mostly issuing real work -> high
    efficient = {"SQ_WAIT_INST_ANY": 10, "SQ_INSTS_VALU": 5000,
                 "SQ_INSTS_VMEM": 500, "MFMA_MOPS": 2000}
    s_good = wb.whitebox_structural_score(efficient)
    assert s_good is not None and s_good > 0.9
    assert s_good > s_bad

    assert wb.whitebox_structural_score({}) is None  # no usable counters -> no-op


# --------------------------------------------------------------------------- #
# 4. Potential-based shaping: telescoping invariance (Ng et al.)
# --------------------------------------------------------------------------- #
def test_pbs_discounted_sum_telescopes_to_neg_phi0():
    phis = [0.2, 0.5, 0.8]
    gamma = 0.4
    # sum_t gamma^t F_t == -Phi(s_0) == -0.2 (terminal Phi = 0)
    assert abs(sh.discounted_shaping_sum(phis, gamma) - (-0.2)) < 1e-9


def test_pbs_trajectory_return_shifts_by_constant_only():
    # Two trajectories from the SAME start potential differ in shaped discounted
    # return from their original return by EXACTLY -Phi(s_0) -> group-relative
    # advantage (r - mean) is unchanged, so PBS cannot change the optimal policy.
    gamma = 0.5
    phi0 = 0.3
    for turn_rewards, phis in [
        ([0.0, 1.0], [phi0, 0.9]),
        ([0.1, 0.2, 1.0], [phi0, 0.5, 0.95]),
    ]:
        shaped = sh.shaped_turn_rewards(turn_rewards, phis, gamma)
        orig_ret = sum((gamma ** t) * r for t, r in enumerate(turn_rewards))
        shaped_ret = sum((gamma ** t) * r for t, r in enumerate(shaped))
        assert abs((shaped_ret - orig_ret) - (-phi0)) < 1e-9


def test_pbs_none_potential_is_a_zero_boundary():
    # a turn whose kernel is incorrect/untimed (Phi=None) contributes no shaping
    phis = [None, 0.5, None]
    terms = sh.shaping_terms(phis, gamma=0.9)
    assert terms[0] == 0.0  # cur None
    assert terms[1] == 0.0  # next None
    assert terms[2] == 0.0  # cur None
    shaped = sh.shaped_turn_rewards([1.0, 2.0, 3.0], phis, gamma=0.9)
    assert shaped == [1.0, 2.0, 3.0]  # unchanged where undefined


def test_pbs_densifies_intermediate_credit():
    # monotone progress toward the roofline yields positive intermediate shaping
    # on the improving turns (denser gradient in the flat valley).
    phis = [0.2, 0.4, 0.6]
    terms = sh.shaping_terms(phis, gamma=1.0)  # gamma=1: F_t = Phi_{t+1}-Phi_t
    assert terms[0] > 0 and terms[1] > 0  # improving -> positive local credit
    assert math.isclose(terms[0], 0.2) and math.isclose(terms[1], 0.2)
