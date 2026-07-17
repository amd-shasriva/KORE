"""CPU tests for the P0d/P0b credit-assignment upgrades in grpo.py.

Validates (pure, no torch/GPU):
  * backward compatibility (legacy hard-zero gating unchanged by default);
  * P0d densification: an incorrect turn's shaped progress reward now flows into
    the gradient instead of being hard-zeroed, while correctness stays dominant;
  * P0b potential-based shaping: the trajectory return shifts by exactly the
    start-state constant -Phi(s_0) (Ng et al. policy invariance), so per-turn credit
    is densified without changing which trajectory is best;
  * build_kevin_samples threads the new per-turn potentials + credit flag.
"""

from __future__ import annotations

from kore.policy.grpo import build_kevin_samples, kevin_turn_returns


def test_legacy_gating_is_unchanged_by_default():
    tr = [0.05, 0.3]
    tc = [False, True]
    rets = kevin_turn_returns(tr, tc, gamma=0.4)
    # base=[0,0.3] -> R=[0.12, 0.3] (incorrect turn hard-zeroed, look-ahead only)
    assert abs(rets[0] - 0.12) < 1e-9 and abs(rets[1] - 0.3) < 1e-9


def test_p0d_credit_incorrect_densifies_progress():
    tr = [0.05, 0.3]
    tc = [False, True]
    legacy = kevin_turn_returns(tr, tc, gamma=0.4)
    dense = kevin_turn_returns(tr, tc, gamma=0.4, credit_incorrect=True)
    # the incorrect turn's shaped reward (0.05) now contributes -> higher return
    assert dense[0] > legacy[0]
    assert abs(dense[0] - 0.17) < 1e-9  # 0.05 + 0.4*0.3
    # correctness still dominates: the correct turn scores well above any incorrect one
    all_wrong = kevin_turn_returns([0.07, 0.05], [False, False], gamma=0.4,
                                   credit_incorrect=True)
    assert max(dense) > max(all_wrong)


def test_p0b_pbs_shifts_trajectory_return_by_start_constant():
    tr = [0.0, 1.0]
    tc = [True, True]
    gamma = 0.5
    phis = [0.3, 0.7]
    no_pbs = kevin_turn_returns(tr, tc, gamma)
    pbs = kevin_turn_returns(tr, tc, gamma, phis=phis, phi_weight=1.0)
    # trajectory return = R_0 (full discounted look-ahead). PBS shifts it by -Phi(s_0).
    assert abs((pbs[0] - no_pbs[0]) - (-phis[0])) < 1e-9
    # weight scales the (invariant) shift linearly
    pbs_half = kevin_turn_returns(tr, tc, gamma, phis=phis, phi_weight=0.5)
    assert abs((pbs_half[0] - no_pbs[0]) - (-0.5 * phis[0])) < 1e-9


def test_pbs_none_potentials_are_zero_boundaries():
    tr = [0.0, 1.0]
    tc = [False, True]
    # phis=[None, 0.7]: the t0 shaping TERM is zeroed (cur=None), but the t1 term
    # F_1 = gamma*Phi_T(=0) - 0.7 = -0.7 still propagates back to R_0 via the
    # discount -- that is correct, not a phantom credit. base=[0,1]+shaping[0,-0.7]
    # -> [0,0.3] -> R=[0.15, 0.3].
    rets = kevin_turn_returns(tr, tc, gamma=0.5, phis=[None, 0.7], phi_weight=1.0)
    assert abs(rets[0] - 0.15) < 1e-9 and abs(rets[1] - 0.3) < 1e-9
    # a FULLY-None potential list contributes zero shaping anywhere (pure boundary)
    none_rets = kevin_turn_returns(tr, tc, gamma=0.5, phis=[None, None], phi_weight=1.0)
    base = kevin_turn_returns(tr, tc, gamma=0.5)
    assert none_rets == base


def test_build_kevin_samples_threads_phis_and_credit_flag():
    traj_rewards = [[0.05, 0.4], [0.03, 0.0]]
    traj_correct = [[False, True], [False, False]]
    traj_phis = [[0.2, 0.6], [None, None]]
    returns, index = build_kevin_samples(
        traj_rewards, traj_correct, gamma=0.4,
        credit_incorrect=True, traj_phis=traj_phis, phi_weight=1.0)
    assert len(returns) == len(index) == 4
    assert index == [(0, 0), (0, 1), (1, 0), (1, 1)]
    # default path (no phis / no credit) still matches the legacy per-turn returns
    legacy, _ = build_kevin_samples(traj_rewards, traj_correct, gamma=0.4)
    assert abs(legacy[0] - 0.16) < 1e-9  # 0 + 0.4*0.4 (incorrect hard-zeroed)
