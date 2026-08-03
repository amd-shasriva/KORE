"""Does GRPO's advantage estimate actually bias the multi-turn gradient, and
does TRLOO actually fix it?

"GRPO is biased in multi-turn" is easy to assert and hard to see. These tests
SHOW it, on a two-turn MDP small enough to enumerate every possible rollout group
exactly -- so every number below is an expectation computed in closed form, with
no sampling noise and no tolerance hiding a real discrepancy.

The reference gradient is not taken on faith either: the first test checks the
analytic ``grad J`` against a central finite difference of ``J``, and checks that
the per-turn REINFORCE target equals it. Only then is it used to judge estimators.

The MDP (``_SetupPayoff``): two turns, one shared policy parameter,
``pi(a=1) = sigmoid(theta) = p``. Reward 1.0 iff BOTH turns pick action 1, so
``J = p^2``. Turn 0's action strongly determines turn 1's payoff, which is exactly
the cross-turn dependence that multi-turn credit assignment lives on and that a
self-inclusive baseline contaminates.
"""

from __future__ import annotations

import itertools
import math

import pytest

from kore.policy.grpo import group_advantages
from kore.policy.trloo import (
    turn_loo_advantages,
    turn_loo_advantages_ragged,
    sample_turn_keys,
)

TRAJ = list(itertools.product([0, 1], repeat=2))   # (a0, a1)


# --------------------------------------------------------------------------- #
# exact-enumeration scaffolding
# --------------------------------------------------------------------------- #
def _sigmoid(t: float) -> float:
    return 1.0 / (1.0 + math.exp(-t))


def _grad_logpi(a: int, p: float) -> float:
    """d/dtheta log pi(a) for pi(1) = sigmoid(theta)."""
    return (1.0 - p) if a == 1 else -p


def _traj_prob(traj, p: float) -> float:
    out = 1.0
    for a in traj:
        out *= p if a == 1 else (1.0 - p)
    return out


def _returns_setup_payoff(traj, gamma: float) -> list[float]:
    """Per-turn discounted returns for the setup/payoff MDP."""
    a0, a1 = traj
    rewards = [0.0, 1.0 if (a0 == 1 and a1 == 1) else 0.0]
    r1 = rewards[1]
    return [rewards[0] + gamma * r1, r1]


def _J(p: float) -> float:
    return p * p


def _analytic_grad_J(p: float) -> float:
    """dJ/dtheta = dJ/dp * dp/dtheta = 2p * p(1-p)."""
    return 2.0 * p * p * (1.0 - p)


def _pooled_mean_advantages(returns):
    """GRPO's baseline with the std normalisation removed.

    Isolates the SELF-INCLUSION term from the std term so the two can be
    attributed separately. ``group_advantages`` is the shipped estimator and
    always divides by the pooled std; this is the same centring without it.
    """
    n = len(returns)
    mean = sum(returns) / n
    return [r - mean for r in returns]


def _expected_gradient(estimator, m_traj: int, p: float, gamma: float = 1.0,
                       returns_fn=_returns_setup_payoff, traj_space=None):
    """E[ghat] over EVERY possible group of ``m_traj`` trajectories, exactly.

    ``estimator(returns, index) -> advantages`` receives the group's flat per-turn
    returns in the same ``(returns, index)`` form ``build_kevin_samples`` emits.
    Normalised per trajectory so the target is the single-trajectory gradient.
    """
    space = traj_space if traj_space is not None else TRAJ
    total = 0.0
    for group in itertools.product(space, repeat=m_traj):
        prob = 1.0
        for traj in group:
            prob *= _traj_prob(traj, p)
        returns: list[float] = []
        index: list[tuple[int, int]] = []
        actions: list[int] = []
        for ti, traj in enumerate(group):
            for tu, r in enumerate(returns_fn(traj, gamma)):
                returns.append(r)
                index.append((ti, tu))
                actions.append(traj[tu])
        advantages = estimator(returns, index)
        total += prob * sum(
            a * _grad_logpi(act, p)
            for a, act in zip(advantages, actions)) / m_traj
    return total


def _trloo(returns, index):
    return turn_loo_advantages(returns, index)


def _grpo_pooled_std(returns, index):
    return group_advantages(list(returns))


def _grpo_pooled_mean(returns, index):
    return _pooled_mean_advantages(list(returns))


def _true_return_weighting(returns, index):
    """No baseline at all: the plain REINFORCE target."""
    return list(returns)


# --------------------------------------------------------------------------- #
# 0. the yardstick has to be verified before it is used to judge anything
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
def test_the_reference_gradient_is_itself_verified(p):
    """Analytic grad J == central finite difference, and == the REINFORCE target.

    Every later test measures an estimator against ``_analytic_grad_J``. If that
    were wrong, every conclusion drawn from it would be too, so it is checked
    against a numerical derivative of ``J`` and against the exactly-enumerated
    per-turn REINFORCE expectation.
    """
    theta = math.log(p / (1.0 - p))
    h = 1e-5
    finite_difference = (_J(_sigmoid(theta + h)) - _J(_sigmoid(theta - h))) / (2 * h)
    assert finite_difference == pytest.approx(_analytic_grad_J(p), abs=1e-7)

    # The estimator under study credits each turn with its own discounted return;
    # with gamma=1 that sum is exactly grad J by the policy gradient theorem.
    reinforce = _expected_gradient(_true_return_weighting, 3, p, gamma=1.0)
    assert reinforce == pytest.approx(_analytic_grad_J(p), abs=1e-12)


# --------------------------------------------------------------------------- #
# 1. THE BIAS: GRPO shrinks the multi-turn gradient, TRLOO does not
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m_traj,expected_fraction",
                         [(2, 1 / 2), (3, 2 / 3), (4, 3 / 4), (5, 4 / 5)])
@pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
def test_grpo_self_inclusion_shrinks_the_gradient_by_one_over_group_size(
    m_traj, expected_fraction, p
):
    """Pooled self-inclusive centring returns exactly ``(M-1)/M`` of grad J.

    This is the bias in its most benign form. The group mean contains the
    sample's own return AND the other turns of its own trajectory, and those
    other turns are downstream of the action being credited, so the baseline
    cancels part of the real long-horizon credit. On this symmetric MDP the
    leakage happens to be proportional, so it shows up as a clean shrink:
    half the gradient thrown away at M=2, a fifth at M=5.
    """
    truth = _analytic_grad_J(p)
    measured = _expected_gradient(_grpo_pooled_mean, m_traj, p)
    assert measured == pytest.approx(expected_fraction * truth, rel=1e-9)
    # Stated as the thing a reader cares about: it is NOT grad J.
    assert measured < truth


@pytest.mark.parametrize("m_traj", [2, 3, 4, 5])
@pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
def test_trloo_recovers_the_true_gradient_exactly(m_traj, p):
    """The same MDP, the same groups, an unbiased baseline: no shrink at all."""
    truth = _analytic_grad_J(p)
    measured = _expected_gradient(_trloo, m_traj, p)
    assert measured == pytest.approx(truth, rel=1e-12)


@pytest.mark.parametrize("m_traj", [3, 4, 5])
def test_grpo_std_normalisation_error_grows_with_the_group(m_traj):
    """The std term is the part that does NOT wash out as the group grows.

    Self-inclusion in the mean is an O(1/M) shrink, so it decays. Dividing by the
    pooled std -- which also contains the sample -- goes the other way: the error
    increases with M, and it depends on the policy, so no group size makes it
    disappear. This is why TRLOO is deliberately unnormalised.
    """
    p = 0.3
    truth = _analytic_grad_J(p)
    ratio = _expected_gradient(_grpo_pooled_std, m_traj, p) / truth
    # Measured: 1.41 at M=3, 1.69 at M=4, 1.91 at M=5.
    assert ratio > 1.35
    previous = _expected_gradient(_grpo_pooled_std, m_traj - 1, p) / truth
    assert ratio > previous, "std-normalised error should grow with group size"


def test_std_normalisation_error_is_policy_dependent_so_it_never_cancels():
    """A shrink by a constant is a learning-rate change; this is not constant."""
    truth_lo, truth_hi = _analytic_grad_J(0.3), _analytic_grad_J(0.7)
    ratio_lo = _expected_gradient(_grpo_pooled_std, 5, 0.3) / truth_lo
    ratio_hi = _expected_gradient(_grpo_pooled_std, 5, 0.7) / truth_hi
    assert abs(ratio_lo - ratio_hi) > 0.15, (ratio_lo, ratio_hi)


# --------------------------------------------------------------------------- #
# 2. the strongest form: GRPO can point the gradient the WRONG WAY
# --------------------------------------------------------------------------- #
#: A two-turn MDP found by exhaustive search over random reward tables where
#: pooled GRPO's EXPECTED gradient has the opposite sign to the true one. Kept as
#: literal data so the demonstration is reproducible and hand-checkable.
#: rewards[(a0, a1)] = (r0, r1)
WRONG_WAY_REWARDS = {
    (0, 0): (-0.483, -0.686),
    (0, 1): (0.014, -0.519),
    (1, 0): (-0.711, 0.753),
    (1, 1): (0.409, 0.719),
}
WRONG_WAY_P = 0.5
WRONG_WAY_M = 2
WRONG_WAY_GAMMA = 0.4


def _wrong_way_returns(traj, gamma):
    r0, r1 = WRONG_WAY_REWARDS[traj]
    return [r0 + gamma * r1, r1]


def _wrong_way_expectation(estimator):
    return _expected_gradient(
        estimator, WRONG_WAY_M, WRONG_WAY_P, gamma=WRONG_WAY_GAMMA,
        returns_fn=_wrong_way_returns)


def test_grpo_can_send_the_gradient_the_wrong_way_and_trloo_cannot():
    """The bias is directional, not merely a scale factor.

    On the symmetric MDP above the contamination happened to be proportional to
    the signal, which looks harmless. In general it is not: the cross-turn terms
    the self-inclusive baseline subtracts are unrelated in magnitude to the term
    they contaminate, so they can exceed it and reverse the sign.

    Here the true gradient is +0.171 -- a large, unambiguous "increase p" -- and
    pooled GRPO answers -0.007, i.e. decrease it. With std normalisation it
    answers -0.098, over half the true magnitude in the wrong direction. A
    trainer following either would move the policy AWAY from the optimum on this
    task while its loss curve looked perfectly healthy.
    """
    truth = _expected_gradient(
        _true_return_weighting, WRONG_WAY_M, WRONG_WAY_P,
        gamma=WRONG_WAY_GAMMA, returns_fn=_wrong_way_returns)
    assert truth == pytest.approx(+0.171350, abs=1e-6)

    grpo_mean = _wrong_way_expectation(_grpo_pooled_mean)
    grpo_std = _wrong_way_expectation(_grpo_pooled_std)
    trloo = _wrong_way_expectation(_trloo)

    assert grpo_mean == pytest.approx(-0.007337, abs=1e-6)
    assert grpo_std == pytest.approx(-0.098342, abs=1e-6)
    assert grpo_mean * truth < 0, "expected GRPO to invert the gradient here"
    assert grpo_std * truth < 0, "expected GRPO+std to invert the gradient here"

    # TRLOO gets it exactly right, sign and magnitude.
    assert trloo == pytest.approx(truth, rel=1e-12)
    assert trloo * truth > 0


def test_trloo_never_inverts_the_gradient_across_a_sweep_of_mdps():
    """A sweep rather than one lucky example: no wrong-direction case for TRLOO.

    Deterministic sweep over reward tables built from a fixed generator, so this
    is a regression test on the estimator rather than a randomised search.
    """
    grpo_inversions = 0
    checked = 0
    for seed in range(120):
        # Deterministic pseudo-random reward table in [-1, 1].
        table = {}
        h = seed * 2654435761 % (2 ** 32)
        for traj in TRAJ:
            vals = []
            for _ in range(2):
                h = (h * 1103515245 + 12345) % (2 ** 31)
                vals.append((h % 2001) / 1000.0 - 1.0)
            table[traj] = tuple(vals)

        def returns_fn(traj, gamma, _table=table):
            r0, r1 = _table[traj]
            return [r0 + gamma * r1, r1]

        for m_traj in (2, 3):
            truth = _expected_gradient(_true_return_weighting, m_traj, 0.5,
                                       gamma=0.4, returns_fn=returns_fn)
            if abs(truth) < 1e-2:
                continue          # no meaningful direction to preserve
            checked += 1
            trloo = _expected_gradient(_trloo, m_traj, 0.5, gamma=0.4,
                                       returns_fn=returns_fn)
            grpo = _expected_gradient(_grpo_pooled_mean, m_traj, 0.5, gamma=0.4,
                                      returns_fn=returns_fn)
            assert trloo == pytest.approx(truth, rel=1e-9), (seed, m_traj)
            if grpo * truth < 0:
                grpo_inversions += 1

    assert checked > 100, f"sweep degenerated to {checked} cases"
    # The point of the sweep: GRPO inverts on some of these, TRLOO on none.
    assert grpo_inversions > 0, (
        "the sweep found no GRPO inversion; it no longer demonstrates anything")


# --------------------------------------------------------------------------- #
# 3. ragged, action-dependent episode lengths (what KORE actually produces)
# --------------------------------------------------------------------------- #
#: A ragged MDP: action 1 at turn 0 COMMITS and ends the episode after one turn;
#: action 0 continues to a second turn. Episode length is therefore
#: action-dependent, which is exactly the harness's behaviour -- it breaks as
#: soon as the model stops calling tools.
_RAGGED = {
    "commit": 0.6,      # 1-turn episode, reward for committing early
    "continue": 0.1,    # turn-0 reward when continuing
    "then_0": -0.4,     # turn-1 reward after a1=0
    "then_1": 0.9,      # turn-1 reward after a1=1
}
RAGGED_SPACE = [(1,), (0, 0), (0, 1)]


def _ragged_returns(traj, gamma):
    if traj == (1,):
        rewards = [_RAGGED["commit"]]
    else:
        rewards = [_RAGGED["continue"],
                   _RAGGED["then_1"] if traj[1] == 1 else _RAGGED["then_0"]]
    out = [0.0] * len(rewards)
    running = 0.0
    for k in range(len(rewards) - 1, -1, -1):
        running = rewards[k] + gamma * running
        out[k] = running
    return out


def _trloo_dropping_baseline_less_samples(returns, index):
    """The tempting alternative: zero out any sample with no LOO baseline.

    Included so the design decision in ``kore/policy/trloo.py`` is demonstrated
    rather than asserted. Zeroing the advantage is the same thing as dropping the
    sample from the update.
    """
    counts: dict[int, int] = {}
    for _, turn in index:
        counts[turn] = counts.get(turn, 0) + 1
    advantages = turn_loo_advantages(returns, index)
    return [0.0 if counts[turn] < 2 else a
            for a, (_, turn) in zip(advantages, index)]


@pytest.mark.parametrize("m_traj", [2, 3])
@pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
def test_trloo_stays_unbiased_when_episode_length_depends_on_the_action(m_traj, p):
    """Ragged episodes do not break TRLOO, because the baseline excludes self.

    Worth checking explicitly: whether turn ``t`` exists at all depends on the
    policy's earlier actions, which is a selection effect no turn-level estimator
    creates but any of them could be damaged by. It is harmless here because the
    baseline is built from OTHER trajectories only, and
    ``E[grad log pi(a_t) | history] == 0`` holds whether or not the turn exists.
    """
    truth = _expected_gradient(
        _true_return_weighting, m_traj, p, gamma=1.0,
        returns_fn=_ragged_returns, traj_space=RAGGED_SPACE)
    measured = _expected_gradient(
        _trloo, m_traj, p, gamma=1.0,
        returns_fn=_ragged_returns, traj_space=RAGGED_SPACE)
    assert measured == pytest.approx(truth, rel=1e-9)


def test_dropping_a_baseline_less_sample_would_reintroduce_the_bias():
    """Why baseline-less samples are KEPT with a constant baseline of 0.0.

    Dropping them looks like the conservative choice -- no baseline, no claim --
    but the decision to drop is correlated with the policy's own actions, so it
    is a selection bias. Keeping the sample with a constant baseline is unbiased,
    because a constant factorises out of ``E[b grad log pi]``.
    """
    p = 0.7
    truth = _expected_gradient(
        _true_return_weighting, 2, p, gamma=1.0,
        returns_fn=_ragged_returns, traj_space=RAGGED_SPACE)
    kept = _expected_gradient(
        _trloo, 2, p, gamma=1.0,
        returns_fn=_ragged_returns, traj_space=RAGGED_SPACE)
    dropped = _expected_gradient(
        _trloo_dropping_baseline_less_samples, 2, p, gamma=1.0,
        returns_fn=_ragged_returns, traj_space=RAGGED_SPACE)

    assert kept == pytest.approx(truth, rel=1e-9)
    assert abs(dropped - truth) > 0.05 * abs(truth), (
        "dropping baseline-less samples no longer shows a bias; if the harness "
        "changed so every turn always has a peer, revisit the docstring claim")


# --------------------------------------------------------------------------- #
# 4. arithmetic contract + failure modes
# --------------------------------------------------------------------------- #
def test_baseline_uses_other_trajectories_at_the_same_turn():
    """Hand-checkable: three trajectories, two turns."""
    returns = [1.0, 4.0,     # trajectory 0: turn 0, turn 1
               2.0, 6.0,     # trajectory 1
               3.0, 8.0]     # trajectory 2
    index = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    advantages = turn_loo_advantages(returns, index)
    # turn 0 peers of trajectory 0 are {2.0, 3.0} -> baseline 2.5 -> 1.0-2.5
    assert advantages[0] == pytest.approx(1.0 - 2.5)
    # turn 1 peers of trajectory 0 are {6.0, 8.0} -> baseline 7.0 -> 4.0-7.0
    assert advantages[1] == pytest.approx(4.0 - 7.0)
    assert advantages[2] == pytest.approx(2.0 - 2.0)     # peers {1,3} -> 2.0
    assert advantages[5] == pytest.approx(8.0 - 5.0)     # peers {4,6} -> 5.0


def test_turns_are_never_pooled_across_turn_indices():
    """A turn-1 sample must not be centred against turn-0 returns.

    Later turns hold better kernels, so pooling across turns would give every
    late turn a positive advantage and every early turn a negative one no matter
    what the model did. Here turn 1 is uniformly better than turn 0 yet the
    within-turn contrast is what survives.
    """
    returns = [0.0, 10.0, 1.0, 11.0]
    index = [(0, 0), (0, 1), (1, 0), (1, 1)]
    advantages = turn_loo_advantages(returns, index)
    assert advantages == pytest.approx([-1.0, -1.0, 1.0, 1.0])
    # A pooled estimator would instead rank both turn-1 samples far above both
    # turn-0 samples, which is a statement about the turn index, not the action.
    pooled = group_advantages(returns)
    assert pooled[1] > 0 and pooled[0] < 0 and pooled[3] > 0 and pooled[2] < 0


def test_a_degenerate_group_yields_no_signal_rather_than_noise():
    """All trajectories identical at a turn -> zero advantage, not a divide blowup.

    ``group_advantages`` divides by ``std + 1e-6``, so an all-equal group there
    produces ~0 through a near-singular division. TRLOO subtracts a mean equal to
    the value and gets exact zeros, with no epsilon involved.
    """
    returns = [0.5] * 6
    index = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert turn_loo_advantages(returns, index) == [0.0] * 6


def test_a_lone_trajectory_keeps_its_return_as_the_advantage():
    """One trajectory: no peers at any turn, so the baseline is the constant 0.0.

    The alternative -- emitting 0.0 advantages -- would be a silent selection
    bias (see ``test_dropping_a_baseline_less_sample_would_reintroduce_the_bias``).
    """
    returns = [0.25, 0.75]
    index = [(0, 0), (0, 1)]
    assert turn_loo_advantages(returns, index) == pytest.approx([0.25, 0.75])


def test_a_turn_only_one_trajectory_reached_keeps_its_return():
    """Trajectory 0 ran 3 turns, trajectory 1 stopped at 2: turn 2 has no peer."""
    returns = [1.0, 2.0, 9.0, 3.0, 4.0]
    index = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    advantages = turn_loo_advantages(returns, index)
    assert advantages[0] == pytest.approx(1.0 - 3.0)
    assert advantages[1] == pytest.approx(2.0 - 4.0)
    assert advantages[2] == pytest.approx(9.0)      # constant baseline 0.0
    assert advantages[3] == pytest.approx(3.0 - 1.0)
    assert advantages[4] == pytest.approx(4.0 - 2.0)


def test_mismatched_returns_and_index_is_refused():
    """Silently zipping to the shorter list would drop real samples."""
    with pytest.raises(ValueError, match="length mismatch"):
        turn_loo_advantages([1.0, 2.0, 3.0], [(0, 0), (0, 1)])


def test_empty_group_is_empty_not_an_error():
    assert turn_loo_advantages([], []) == []


def test_ragged_helper_preserves_shape():
    advantages = turn_loo_advantages_ragged([[1.0, 2.0], [3.0], [5.0, 6.0, 7.0]])
    assert [len(row) for row in advantages] == [2, 1, 3]
    # turn 0 peers of trajectory 0: {3.0, 5.0} -> 4.0
    assert advantages[0][0] == pytest.approx(1.0 - 4.0)
    # turn 1 exists for trajectories 0 and 2 only: peer of 2.0 is 6.0
    assert advantages[0][1] == pytest.approx(2.0 - 6.0)
    # turn 2 belongs to trajectory 2 alone -> constant baseline
    assert advantages[2][2] == pytest.approx(7.0)


# --------------------------------------------------------------------------- #
# 5. the sample-key contract: refuse to guess trajectory ids
# --------------------------------------------------------------------------- #
def test_turn_keys_are_read_from_the_sample_tuple():
    samples = [[0.5, "gen", None, None, 4, None, (0, 0)],
               [0.7, "gen", None, None, 4, None, (1, 0)]]
    assert sample_turn_keys(samples) == [(0, 0), (1, 0)]


@pytest.mark.parametrize("samples", [
    [[0.5, "gen", None, None, 4, None]],                    # legacy 6-tuple
    [[0.5, "gen", None, None, 4, None, None]],              # key absent
    [[0.5, "gen", None, None, 4, None, (0,)]],              # wrong arity
    [[0.5, "gen", None, None, 4, None, ("a", 0)]],          # wrong type
    [[0.5, "gen", None, None, 4, None, (True, 0)]],         # bool is not an id
    [[0.5, "gen", None, None, 4, None, (0, 0)],
     [0.7, "gen", None, None, 4, None]],                    # one sample missing
])
def test_unusable_turn_keys_return_none_instead_of_a_guess(samples):
    """Without real trajectory ids TRLOO cannot be computed.

    Inventing ids (say, one trajectory per sample) would silently compute a
    different estimator than the config asked for, which is the failure mode this
    whole module exists to remove. ``None`` tells the caller to fall back
    explicitly and log it.
    """
    assert sample_turn_keys(samples) is None
