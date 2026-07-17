"""CPU tests for the P0d/P0b credit-assignment upgrades in grpo.py.

Validates (pure, no torch/GPU):
  * backward compatibility (legacy hard-zero gating unchanged by default);
  * P0d densification: an incorrect turn's shaped progress reward now flows into
    the gradient instead of being hard-zeroed, while correctness stays dominant;
  * P0b potential-based shaping: the potential is taken at the ENTERING state
    Phi(s_t) (prev turn's exit; seed = 0.0 constant), so the per-turn sample return
    R_t is action-INDEPENDENT (Ng et al. policy invariance) and the trajectory
    return R_0 shifts only by the start-state constant -Phi(s_0)=0 -- densifying
    per-turn credit without changing which trajectory is best;
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


def test_p0b_pbs_leaves_trajectory_return_invariant():
    # PBS shifts the TRAJECTORY return R_0 by -w*Phi(s_0). s_0 is the ENTERING
    # (seed) state -- a fixed per-group constant (0.0) -- so R_0 is UNCHANGED by PBS
    # for ANY interior/exit potentials and ANY weight. This is the Ng et al.
    # guarantee that shaping cannot alter the trajectory-level optimum (and Phi(s_0)
    # cancels in the GRPO group baseline). The OLD exit-state convention wrongly made
    # this shift depend on the interior potential -> not invariant; this locks it.
    tr = [0.0, 1.0]
    tc = [True, True]
    gamma = 0.5
    no_pbs = kevin_turn_returns(tr, tc, gamma)
    for phis in ([0.3, 0.7], [0.9, 0.1], [0.0, 1.0]):
        for w in (1.0, 0.5):
            pbs = kevin_turn_returns(tr, tc, gamma, phis=phis, phi_weight=w)
            assert abs(pbs[0] - no_pbs[0]) < 1e-9, (phis, w)


def test_pbs_is_policy_invariant_to_own_turn_action():
    # THE policy-invariance property (the semantic the arithmetic-only tests missed):
    # sample (i,t)'s return R_t is the advantage that multiplies turn t's OWN
    # generation, so PBS must make R_t INDEPENDENT of turn t's produced kernel
    # (its exit potential phi[t]). We vary ONLY phi[1] and require R[1] unchanged.
    tr = [0.3, 0.3, 0.3]
    tc = [True, True, True]
    gamma = 0.5
    a = kevin_turn_returns(tr, tc, gamma, phis=[0.2, 0.5, 0.8], phi_weight=1.0)
    b = kevin_turn_returns(tr, tc, gamma, phis=[0.2, 0.9, 0.8], phi_weight=1.0)  # differ only at t=1
    assert abs(a[1] - b[1]) < 1e-9, "R_1 must be invariant to turn 1's own action (phi[1])"
    assert abs(a[0] - b[0]) < 1e-9, "earlier turns' returns also cancel the later action"
    # turn 2's return legitimately depends on phi[1] (its ENTERING-state baseline,
    # which is action-independent for turn 2) -- so it is allowed to differ.


def test_pbs_none_potentials_are_zero_boundaries():
    tr = [0.0, 1.0]
    tc = [False, True]
    # phis=[None, 0.7] are EXIT potentials. Turn 0 is incorrect (exit None), so
    # turn 1's ENTERING state (= turn 0's exit) is None -> its PBS term is a boundary
    # (0); turn 0's own term is a boundary too. No shaping is fabricated, so the
    # returns collapse to the pure correctness-gated base: base=[0,1] -> R=[0.5, 1.0].
    rets = kevin_turn_returns(tr, tc, gamma=0.5, phis=[None, 0.7], phi_weight=1.0)
    assert abs(rets[0] - 0.5) < 1e-9 and abs(rets[1] - 1.0) < 1e-9
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
