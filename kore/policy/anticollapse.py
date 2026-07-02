"""Anti-collapse ladder for multi-turn GRPO — PURE, unit-testable.

GRPO's group-relative advantage degenerates when a whole group shares the same
reward: the std -> 0 and every advantage collapses to ~0, so there is no
learning signal. The KORE plan's anti-collapse ladder addresses this in rungs:

  1. RC-GRPO (reward-conditioned rollouts): prepend a ``<|high_reward|>`` /
     ``<|low_reward|>`` control token to a fraction ``p`` of rollouts. Because
     the two conditioned modes have different mean rewards, the group variance
     has a guaranteed *floor* and can never fully collapse.
  2. GTPO (turn-level credit): assign per-turn credit (discounted future
     return, mean-centered across turns) instead of a single trajectory reward.
  3. SC-GRPO (self-consistency / all-fail): when every rollout in a group
     fails, inject a zero-mean diversity spread so exploration is still rewarded.

All functions are numpy-free pure Python so they import and test on CPU.
"""

from __future__ import annotations

import random

HIGH_REWARD_TOKEN = "<|high_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"
_VALID_TOKENS = frozenset({HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN})


def prepend_reward_token(prompt: str, token: str) -> str:
    """Prepend an RC-GRPO reward-control token to a prompt.

    ``token`` must be one of ``<|high_reward|>`` / ``<|low_reward|>``.
    """
    if token not in _VALID_TOKENS:
        raise ValueError(f"token must be one of {sorted(_VALID_TOKENS)}, got {token!r}")
    return f"{token}\n{prompt}"


def sample_reward_tokens(G: int, p_high: float, seed: int | None = None) -> list[str]:
    """Sample ``G`` reward-control tokens with ~``p_high`` fraction high-reward.

    Uses an exact count (``round(G * p_high)`` high tokens) so the realized
    fraction is as close to ``p_high`` as the group size permits, then shuffles.
    """
    if G < 0:
        raise ValueError("G must be non-negative")
    p_high = min(1.0, max(0.0, p_high))
    n_high = int(round(G * p_high))
    n_high = min(G, max(0, n_high))
    tokens = [HIGH_REWARD_TOKEN] * n_high + [LOW_REWARD_TOKEN] * (G - n_high)
    rng = random.Random(seed)
    rng.shuffle(tokens)
    return tokens


def variance_floor(
    rewards: list[float],
    reward_tokens: list[str],
    means: dict[str, float],
) -> bool:
    """RC-GRPO variance-floor check.

    With ``G`` rollouts, a fraction ``p`` conditioned on the high-reward mode
    and the two modes separated by a mean gap ``eps = |mu_high - mu_low|``, the
    expected group variance is bounded below by

        E[sigma^2] >= (G-1)/G * p*(1-p) * eps^2.

    Returns True when the *observed* group variance meets this floor, i.e. the
    reward-conditioning is doing its job and the group has not collapsed.

    Args:
        rewards: realized reward per rollout in the group.
        reward_tokens: the control token used for each rollout (parallel list).
        means: conditional mean reward per token, e.g.
            ``{"<|high_reward|>": mu_high, "<|low_reward|>": mu_low}``.
    """
    G = len(rewards)
    if G == 0:
        return False

    n_high = sum(1 for t in reward_tokens if t == HIGH_REWARD_TOKEN)
    p = n_high / G

    mu_vals = [v for v in means.values() if v is not None]
    eps_gap = (max(mu_vals) - min(mu_vals)) if len(mu_vals) >= 2 else 0.0

    floor = (G - 1) / G * p * (1.0 - p) * (eps_gap ** 2)

    mean_r = sum(rewards) / G
    var = sum((r - mean_r) ** 2 for r in rewards) / G

    return var >= floor


def sc_grpo_allfail_bonus(rewards: list[float], alpha: float) -> list[float]:
    """SC-GRPO all-fail diversity bonus.

    When every rollout in the group fails (all rewards non-positive and
    identical, so standard GRPO advantages are all ~0), return a zero-mean
    diversity spread of magnitude ``alpha`` so that distinct exploratory
    rollouts receive a non-degenerate signal. Callers should order ``rewards``
    by novelty (most novel last) for the spread to reward diversity. In any
    non-collapsed case, returns all zeros (no-op).
    """
    G = len(rewards)
    if G == 0:
        return []
    all_fail = all(r <= 0.0 for r in rewards) and (max(rewards) - min(rewards) < 1e-12)
    if not all_fail or G == 1:
        return [0.0] * G
    center = (G - 1) / 2.0
    # Zero-mean spread scaled so the extremes are +/- alpha.
    return [alpha * (i - center) / center for i in range(G)]


def gtpo_turn_credit(turn_rewards: list[float], gamma: float) -> list[float]:
    """GTPO turn-level credit assignment.

    Each turn gets its discounted future return (Kevin sum form
    ``R_t = sum_{i>=t} gamma^(i-t) s_i``) mean-centered across the turns, so
    credit is distributed per-turn rather than collapsed to a single trajectory
    reward. Returns one credit value per turn.
    """
    n = len(turn_rewards)
    if n == 0:
        return []
    returns: list[float] = [0.0] * n
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = turn_rewards[t] + gamma * running
        returns[t] = running
    mean_r = sum(returns) / n
    return [r - mean_r for r in returns]
