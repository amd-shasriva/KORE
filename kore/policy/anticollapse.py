"""Anti-collapse ladder for multi-turn GRPO - PURE, unit-testable.

GRPO's group-relative advantage degenerates when a whole group shares the same
reward: the std -> 0 and every advantage collapses to ~0, so there is no
learning signal. The KORE plan's anti-collapse ladder addresses this in rungs:

  1. RC-GRPO (reward-conditioned rollouts): prepend a ``<|high_reward|>`` /
     ``<|low_reward|>`` control token to a fraction ``p`` of rollouts. Because
     the two conditioned modes have different mean rewards, the group variance
     has a guaranteed *floor* and can never fully collapse (:func:`variance_floor`
     is the diagnostic that this floor is being met).
  2. AVSPO (virtual-sample injection): when a group's realized reward std is
     below ``tau``, inject ``k`` *virtual* reward samples into the NORMALIZATION
     statistics only (no policy-gradient term). This guarantees a variance floor
     for the group-relative advantage without adding any spurious gradient
     sample (:func:`avspo_advantages`).
  3. SC-GRPO (self-consistency): for partial-solve groups, re-score other turns
     against a correct kernel used as an in-context demo (the teacher) and weight
     each token's PG term by KL(teacher||student). :func:`scgrpo_weight_from_kl`
     is the pure aggregator that turns the per-token KL trace into a bounded
     multiplicative weight (the extra teacher forward lives in ``grpo.py``).
  4. GTPO (all-fail code-similarity shaping): when every rollout in a group
     fails, assign a graded partial reward = normalized code shingle-cosine
     similarity to the nearest correct kernel (or a reference) so an all-fail
     group still carries a non-degenerate signal (:func:`gtpo_codesim_shaping`).

All functions are numpy-free pure Python so they import and test on CPU.
"""

from __future__ import annotations

import math
import random
import re

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


def avspo_advantages(returns: list[float], tau: float, k: int = 2,
                     eps: float = 1e-6) -> list[float]:
    """AVSPO variance-floor advantages (virtual-sample injection).

    Standard GRPO group-relative advantages ``(r - mean) / (std + eps)``, except
    that when the group's realized reward std is *below* ``tau`` we inject ``k``
    virtual reward samples at ``mean +/- tau`` into the NORMALIZATION statistics
    only. The injection is mean-preserving (balanced ``+/-``) so the centering is
    unchanged, but it raises the denominator to

        std_aug = sqrt((n*var + k*tau^2) / (n + k))  >=  sqrt(k*tau^2/(n+k)) > 0,

    guaranteeing a variance FLOOR: a near-degenerate group still produces a
    finite, usable learning signal instead of collapsing to ~0. The ``k`` virtual
    samples exist ONLY in the normalization stats - they get NO policy-gradient
    term (the returned list has exactly ``len(returns)`` entries, one per real
    rollout). With ``tau <= 0`` (disabled) or a group whose std already meets the
    floor, this is exactly :func:`group_reward_std`-normalized GRPO.
    """
    n = len(returns)
    if n == 0:
        return []
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    std = math.sqrt(var)
    if tau <= 0.0 or k <= 0 or std >= tau:
        return [(r - mean) / (std + eps) for r in returns]
    aug_std = math.sqrt((n * var + k * (tau ** 2)) / (n + k))
    return [(r - mean) / (aug_std + eps) for r in returns]


def scgrpo_weight_from_kl(token_kls: list[float], scale: float = 1.0,
                          w_min: float = 0.5, w_max: float = 2.0) -> float:
    """SC-GRPO per-sample multiplicative PG weight from a per-token KL trace.

    ``token_kls[t] = KL(teacher_t || student_t)`` where the teacher is the policy
    conditioned on a correct kernel used as an in-context demo. Tokens where the
    demo-conditioned teacher disagrees most with the student carry the most
    information, so the sample's PG term is up-weighted by the mean per-token KL:

        w = clip(1 + scale * mean(token_kls), w_min, w_max).

    An empty trace (no re-scored tokens) returns the neutral weight ``1.0``. The
    weight is bounded so a single high-KL token cannot blow up the gradient.
    """
    if not token_kls:
        return 1.0
    mean_kl = sum(token_kls) / len(token_kls)
    return max(w_min, min(w_max, 1.0 + scale * mean_kl))


def _code_shingles(code: str, n: int = 3) -> dict[tuple, int]:
    """Multiset (as a count dict) of length-``n`` token shingles of ``code``.

    Tokenization is whitespace + identifier/operator split so structurally
    similar kernels (same ops, tile sizes, loop structure) share shingles even
    when identifiers differ slightly. Falls back to unigrams for short snippets.
    """
    toks = re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", code or "")
    if not toks:
        return {}
    n = max(1, min(n, len(toks)))
    counts: dict[tuple, int] = {}
    for i in range(len(toks) - n + 1):
        sh = tuple(toks[i:i + n])
        counts[sh] = counts.get(sh, 0) + 1
    return counts


def code_shingle_cosine(a: str, b: str, n: int = 3) -> float:
    """Cosine similarity in [0, 1] of two code snippets over token-shingle counts."""
    ca, cb = _code_shingles(a, n), _code_shingles(b, n)
    if not ca or not cb:
        return 0.0
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def gtpo_codesim_shaping(codes: list[str], references: list[str],
                         scale: float = 0.3, n: int = 3) -> list[float]:
    """GTPO all-fail code-similarity shaping (replaces the old positional spread).

    For an ALL-FAIL group (no correct kernel, so Kevin returns are all 0 and the
    group-relative advantage collapses), give each candidate a graded partial
    reward proportional to its maximum code shingle-cosine similarity to any
    ``references`` kernel (the nearest correct kernel seen this step, or a seed
    reference)::

        partial_i = scale * max_j cosine(codes_i, references_j)   in [0, scale].

    This rewards candidates that are structurally close to a known-good kernel
    even though none passed, restoring a non-degenerate signal. With no
    references, returns all zeros (no-op - the group stays collapsed and is
    dropped by StarPO-S).
    """
    if not references:
        return [0.0] * len(codes)
    out: list[float] = []
    for c in codes:
        best = max((code_shingle_cosine(c, r, n) for r in references), default=0.0)
        out.append(scale * best)
    return out
