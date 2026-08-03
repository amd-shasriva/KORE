"""TRLOO: turn-level REINFORCE leave-one-out advantages for multi-turn RL.

Dr. Kernel (arXiv 2602.05885) names the defect this module fixes: plain GRPO has
a BIASED policy gradient in the multi-turn setting, caused by self-inclusion in
the advantage estimate. This is the algorithmic core of their result, so it is
worth being precise about what the bias is, because "GRPO is biased" is often
repeated without a statement of how.

WHY SELF-INCLUSION BIASES THE GRADIENT
--------------------------------------
REINFORCE with a baseline estimates ``grad J = E[R(a) grad log pi(a)]`` as
``(R_i - b) grad log pi(a_i)``. That is unbiased if and only if
``E[b grad log pi(a_i)] == 0``, which holds when ``b`` is INDEPENDENT of the
action ``a_i`` being credited -- then it factorises and
``E[grad log pi(a_i)] == 0`` kills it.

GRPO's baseline is the group mean, and the group mean CONTAINS sample ``i``. So
``b`` is correlated with ``a_i`` and the term does not vanish. In KORE this is
worse than the single-turn case, because :func:`~kore.policy.grpo.build_kevin_samples`
flattens ``m`` trajectories x ``n`` turns into ONE pool and
:func:`~kore.policy.grpo.group_advantages` centres on the mean of that pool.
The baseline for turn ``t`` of trajectory ``i`` therefore contains

  * ``R_{i,t}`` itself, and
  * every OTHER turn of the SAME trajectory ``i``.

The second is the damaging one: later turns of trajectory ``i`` are causally
DOWNSTREAM of the action taken at turn ``t``, so ``E[R_{i,t'} grad log pi(a_{i,t})]``
is not zero. It is the downstream credit -- precisely the long-horizon signal
multi-turn RL exists to capture -- and the self-inclusive baseline subtracts a
fraction of it.

WHAT THAT COSTS, MEASURED
-------------------------
Measured by EXACT ENUMERATION over every possible rollout group of a small
two-turn MDP (no sampling noise), with the true gradient independently checked
against a central finite difference of ``J``. See
``tests/test_trloo_advantages.py``, which reproduces all of it.

* On a symmetric "setup then payoff" MDP (reward only if both turns act well),
  pooled GRPO returns exactly ``(M-1)/M`` of the true gradient: 0.500 at M=2,
  0.750 at M=4, 0.800 at M=5. A uniform shrink, so it merely rescales the
  learning rate by the group size -- the benign case.
* Adding GRPO's std normalisation makes the error GROW with the group instead of
  shrinking: +41% at M=3, +69% at M=4, +91% at M=5, and it varies with the
  policy, so it does not vanish at any group size.
* On general (asymmetric) two-turn MDPs the bias is NOT a uniform shrink, because
  the cross-turn terms are not proportional to the term they contaminate. Over
  ~1,930 random MDPs where the true gradient is at least 0.1 in magnitude,
  pooled GRPO's EXPECTED gradient points the WRONG WAY in 7-10 of them, and
  55-58 with std normalisation. The worst case found:
  ``grad J = +0.171``, GRPO ``= -0.007``, GRPO+std ``= -0.098``, TRLOO ``= +0.171``.
* TRLOO produced ZERO wrong-direction cases in every configuration tried, with a
  worst relative error of 1.1e-13 -- floating-point roundoff.

THE FIX
-------
Leave the sample's own trajectory out of its baseline, and match turns:

    A_{i,t} = R_{i,t} - mean over j != i of R_{j,t}

The baseline now depends only on OTHER trajectories, which are independent
rollouts, so it is independent of ``a_{i,t}`` and the estimator is unbiased.
Matching by TURN INDEX rather than pooling across turns matters for a second
reason: turn-``t`` return distributions are not stationary in ``t`` (later turns
hold better kernels), so a pooled baseline is systematically mis-centred per
turn.

TWO DESIGN DECISIONS SETTLED BY MEASUREMENT, NOT TASTE
-----------------------------------------------------
1. **No std normalisation, of any kind.** Dividing by the pooled std reintroduces
   self-inclusion. Dividing by a LEAVE-ONE-OUT std looks safe -- it excludes
   sample ``i`` -- and per turn it is: it scales that turn's contribution by a
   positive constant. But the scale ``E[1/s_t]`` DIFFERS BY TURN, so it reweights
   turns against each other, and on near-degenerate groups ``1/s`` explodes. The
   measured ratio to the true gradient ranged from -15,107 to +503,846 over 1,443
   cases, including a sign flip. So TRLOO is deliberately unnormalised; scale is
   handled by ``max_grad_norm``, where it belongs.

2. **A sample with no leave-one-out baseline is KEPT with a baseline of 0.0, not
   dropped.** KORE's episodes are ragged (the harness breaks when the model stops
   calling tools) and the length is action-dependent, so turn ``t`` may exist for
   only one trajectory in the group. Dropping those samples FEELS conservative
   and is in fact a selection bias, because whether the sample exists depends on
   the policy's own earlier actions: measured worst error 2.1-4.6x with 12-15 sign
   flips over ~1,440 ragged cases. Keeping the sample with a CONSTANT baseline of
   0.0 is exactly unbiased (worst error 5e-15, zero sign flips) because a constant
   baseline factorises out. Higher variance for that one sample, zero bias.

Everything here is pure arithmetic: no torch, no GPU, no config. It is the same
``(returns, index)`` contract :func:`~kore.policy.grpo.build_kevin_samples`
already emits, so it drops in beside ``group_advantages`` without restructuring
the trainer.
"""

from __future__ import annotations

from typing import Optional, Sequence

#: Name of this estimator in ``GRPOConfig.advantage_estimator``.
TRLOO = "trloo"
#: The incumbent: pooled, self-inclusive, std-normalised group advantages.
GRPO = "grpo"

ESTIMATORS = (GRPO, TRLOO)


def turn_loo_advantages(
    returns: Sequence[float],
    index: Sequence[tuple[int, int]],
) -> list[float]:
    """Turn-level leave-one-out advantages for one rollout group.

    ``returns[k]`` is the per-turn return of the sample whose
    ``index[k] == (trajectory_id, turn_id)``. Both come straight from
    :func:`~kore.policy.grpo.build_kevin_samples`, so infra-dropped turns and
    ragged trajectories are already handled upstream.

    For each sample the baseline is the mean return of the OTHER trajectories at
    the SAME turn id. When no other trajectory reached that turn the baseline is
    0.0 and the sample is kept (see the module docstring: dropping it is a
    measurable selection bias, worth up to 4.6x and able to flip the gradient's
    sign).

    ``trajectory_id`` must be GLOBAL across the whole group. On the sharded path
    each rank owns a slice of the trajectories, and the baseline has to be taken
    over the full group, so the caller maps its local index through
    ``_rank_slice`` before building the key. Two different trajectories sharing
    an id would leak one into the other's baseline.
    """
    if len(returns) != len(index):
        raise ValueError(
            f"returns/index length mismatch: {len(returns)} vs {len(index)}")
    # Per turn id: the sum and count of every trajectory's return at that turn.
    # A leave-one-out mean is then (total - mine) / (count - 1), which is O(n)
    # overall instead of O(n^2) and, more importantly, is exact rather than an
    # incrementally-accumulated float.
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for r, (_, turn) in zip(returns, index):
        totals[turn] = totals.get(turn, 0.0) + float(r)
        counts[turn] = counts.get(turn, 0) + 1

    out: list[float] = []
    for r, (_, turn) in zip(returns, index):
        r = float(r)
        n = counts[turn]
        if n < 2:
            # No other trajectory reached this turn. A constant baseline keeps
            # the estimator unbiased; dropping the sample would not.
            out.append(r)
            continue
        baseline = (totals[turn] - r) / (n - 1)
        out.append(r - baseline)
    return out


def turn_loo_advantages_ragged(
    traj_returns: Sequence[Sequence[float]],
) -> list[list[float]]:
    """:func:`turn_loo_advantages` on a ragged per-trajectory nesting.

    Convenience for callers that hold ``[[R_00, R_01, ...], [R_10, ...], ...]``
    rather than the flat ``(returns, index)`` pair. Shape is preserved.
    """
    returns: list[float] = []
    index: list[tuple[int, int]] = []
    for ti, row in enumerate(traj_returns):
        for tu, r in enumerate(row):
            returns.append(float(r))
            index.append((ti, tu))
    flat = turn_loo_advantages(returns, index)
    out: list[list[float]] = []
    cursor = 0
    for row in traj_returns:
        out.append(flat[cursor:cursor + len(row)])
        cursor += len(row)
    return out


def sample_turn_keys(
    samples: Sequence, key_index: int = 6,
) -> Optional[list[tuple[int, int]]]:
    """Recover ``(trajectory_id, turn_id)`` from GRPO sample tuples.

    A GRPO sample is ``[ret, gen_inputs, ref_logp, old_logp, n_tokens, sc_w,
    turn_key]``; ``turn_key`` was appended for TRLOO, so older/shorter tuples
    (and the hand-built ones in the test suite) simply do not have it. Returns
    ``None`` when ANY sample lacks a usable key, which is the caller's signal to
    fall back to the pooled estimator rather than to invent trajectory ids --
    guessing them would silently compute a DIFFERENT estimator than the one the
    config asked for.
    """
    keys: list[tuple[int, int]] = []
    for s in samples:
        try:
            key = s[key_index]
        except (IndexError, KeyError, TypeError):
            return None
        if (not isinstance(key, (tuple, list)) or len(key) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool)
                           for x in key)):
            return None
        keys.append((int(key[0]), int(key[1])))
    return keys


__all__ = [
    "ESTIMATORS",
    "GRPO",
    "TRLOO",
    "sample_turn_keys",
    "turn_loo_advantages",
    "turn_loo_advantages_ragged",
]
