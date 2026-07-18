"""Potential-based reward shaping (PBS) for the multi-turn GRPO credit path.

The verified terminal signal (correctness x speedup) is high-contrast but SPARSE:
across the "correct-but-slow" valley it is nearly flat, so the per-turn advantage
estimator gets little gradient. We densify it with a *potential* ``Phi(s)`` -- the
roofline attainment from :func:`kore.reward.whitebox.phi_potential` (the named
residual ``rho`` when PMC counters are present, else the timing-based ``eta =
T_min/T_meas``) -- added as Ng-Harada-Russell potential-based shaping:

    F(s, s') = gamma * Phi(s') - Phi(s)

The Ng et al. (1999) theorem: for the VANILLA expected-gradient estimator, PBS
leaves the optimal policy invariant for ANY potential and ANY weight -- the
discounted shaping telescopes to ``-Phi(s_0)`` (a constant of the START state). In
that idealized setting:

  * the trajectory-level optimal policy is unchanged (the dense term adds no
    reward-hacking incentive), and
  * PBS is expected-gradient-NEUTRAL: it does not ADD directional gradient toward
    the roofline, it RE-DISTRIBUTES the existing terminal credit across turns
    (variance reduction / denser intermediate signal), which is what the flat valley
    needs. This requires the potentials passed here to be the ENTERING-state
    ``Phi(s_t)`` (see the caller ``kevin_turn_returns``, which reconstructs them
    from the per-turn EXIT potentials); feeding exit potentials would make the
    per-turn subtraction action-dependent.

HONEST CAVEAT (the KORE application, not the theorem itself): the exact invariance
is a property of the vanilla estimator. KORE feeds the ``-w*Phi(s_t)`` offset into
GRPO's std-normalized, GROUP-RELATIVE, per-turn-as-sample advantage (dividing by a
sigma that itself depends on the shifted returns), and at a correct->incorrect
boundary ``Phi(s')=None`` zeroes ``F`` and breaks the telescoping. A small BOUNDED
action-dependent leak (<= gamma*w*Phi ~ 0.06 at w=0.15) therefore survives. Treat
PBS here as an APPROXIMATE, expected-gradient-neutral STATE-DEPENDENT BASELINE that
reshapes credit -- not an exact at-any-weight guarantee. The real anti-hack spine
is the lexicographic correctness gate + bounded action space, not this term.

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

    ``phis[t]`` MUST be the potential of the state ENTERING turn ``t`` (``Phi(s_t)``,
    the kernel the agent starts turn ``t`` with) -- NOT the exit state it produces.
    The transition uses ``phis[t+1]`` as the "next" state and ``terminal_phi``
    (default 0) as ``Phi`` past the last turn, so the discounted sum telescopes to
    ``-Phi(s_0)``. Passing exit potentials instead would shift the whole chain by one
    and make the per-turn subtraction action-dependent (breaking policy invariance);
    the GRPO caller therefore reconstructs the entering-state sequence before calling.

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
