"""Potential-based reward shaping (PBS) for the multi-turn GRPO credit path.

The verified terminal signal (correctness x speedup) is high-contrast but SPARSE:
across the "correct-but-slow" valley it is nearly flat, so the per-turn advantage
estimator gets little gradient. We densify it with a *potential* ``Phi(s)`` (the
roofline attainment ``rho`` from :func:`kore.reward.whitebox.phi_potential`) added
as Ng-Harada-Russell potential-based shaping:

    F(s, s') = gamma * Phi(s') - Phi(s)

The Ng et al. (1999) theorem: PBS leaves the optimal policy invariant for ANY
potential and ANY weight. Concretely, the discounted sum of the shaping telescopes
to ``-Phi(s_0)`` (a constant of the START state), so:

  * the trajectory-level optimal policy is provably unchanged (no reward-hacking
    incentive can be introduced by the dense term -- a formal anti-gaming result),
  * yet the PER-TURN returns are re-distributed to credit local progress toward the
    roofline, which is exactly the dense signal the flat valley needs.

This module is PURE / CPU-only. It operates on the per-turn arrays the GRPO loop
already builds (``turn_rewards`` and a parallel list of per-turn potentials), and
returns shaped per-turn rewards with the invariance property preserved (verified by
tests). ``None`` potentials (turns where the kernel is not correct-and-timed, so
``rho`` is undefined) are treated as shaping boundaries that contribute zero -- a
conservative choice that never invents gradient where there is no measurement.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def shaping_terms(phis: Sequence[Optional[float]], gamma: float,
                  terminal_phi: float = 0.0) -> List[float]:
    """Per-turn PBS terms ``F_t = gamma*Phi(s_{t+1}) - Phi(s_t)``.

    ``phis[t]`` is the potential of the state AT turn ``t`` (i.e. after turn ``t``'s
    kernel is evaluated). The transition potential uses ``phis[t+1]`` as the "next"
    state, and ``terminal_phi`` (default 0) as ``Phi`` past the last turn -- so the
    discounted sum telescopes to ``-Phi(s_0)``.

    A ``None`` potential on either side of a transition makes that transition's term
    ``0.0`` (a shaping boundary): we never fabricate progress across a turn whose
    roofline attainment is undefined (incorrect / untimed kernel).
    """
    n = len(phis)
    out: List[float] = []
    for t in range(n):
        cur = phis[t]
        nxt = phis[t + 1] if t + 1 < n else terminal_phi
        if cur is None or nxt is None:
            out.append(0.0)
        else:
            out.append(gamma * float(nxt) - float(cur))
    return out


def shaped_turn_rewards(turn_rewards: Sequence[float], phis: Sequence[Optional[float]],
                        gamma: float, weight: float = 1.0,
                        terminal_phi: float = 0.0) -> List[float]:
    """Add ``weight * F_t`` to each per-turn reward (denser, policy-invariant credit).

    ``weight`` scales the shaping potential (any value is policy-invariant by the
    theorem; it only trades off how much of the gradient is densified). The returned
    list has the same length as ``turn_rewards``.
    """
    if len(turn_rewards) != len(phis):
        raise ValueError(f"turn_rewards ({len(turn_rewards)}) and phis ({len(phis)}) "
                         "must be the same length")
    terms = shaping_terms(phis, gamma, terminal_phi)
    return [float(r) + weight * f for r, f in zip(turn_rewards, terms)]


def discounted_shaping_sum(phis: Sequence[Optional[float]], gamma: float,
                           terminal_phi: float = 0.0) -> float:
    """The discounted sum ``sum_t gamma^t F_t`` of the shaping terms.

    By the telescoping identity this equals ``-Phi(s_0)`` (when no ``None`` boundary
    interrupts the chain and ``terminal_phi=0``). Exposed so the invariance property
    is directly checkable (and unit-tested).
    """
    terms = shaping_terms(phis, gamma, terminal_phi)
    return float(sum((gamma ** t) * f for t, f in enumerate(terms)))
