"""Learnability/regret-targeted TASK PROPOSER (UED / PLR) for KORE.

Mints tasks at the *policy's competence frontier* rather than uniformly, so the
task distribution co-evolves with the policy. Each candidate descriptor is scored
by three signals:

  * **learnability** ``4*p*(1-p)`` where ``p`` is the measured solve-rate - the
    classic UED/PLR "not too easy, not too hard" curve that peaks at ``p=0.5``
    (the ``4*`` normalizes the peak to ``1.0``). This is the primary signal.

  * **headroom regret** - how much verified performance is *still on the table*
    for this task (normalized ``[0,1]``): a proxy for how much the policy could
    still learn to speed it up. High regret => worth revisiting.

  * **novelty** - distance from the current task archive's occupied niches, so
    the proposer expands into unexplored regions of behavior space.

Guardrails against collapse: descriptors with strong evidence of being
*unsolvable* (``p ~ 0``) or *trivial* (``p ~ 1``) are hard-filtered (score 0);
and :func:`propose` enforces per-niche diversity so the frontier can't collapse
onto one task family.

Pure and deterministic (all randomness is seeded). ``propose`` takes an archive
object by duck-typing (only ``.occupied_keys()`` is used), so this module does
not import :mod:`kore.openended.archive` (keeps the dependency graph acyclic).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from kore.openended import task_space as ts

# --------------------------------------------------------------------------- #
# Outcome statistics + scoring
# --------------------------------------------------------------------------- #
# collapse guardrail band: outside (P_UNSOLVABLE, P_TRIVIAL) a task with
# evidence (attempts > 0) is rejected.
P_UNSOLVABLE = 0.05
P_TRIVIAL = 0.95


@dataclass(frozen=True)
class DescriptorStats:
    """Per-descriptor outcome statistics fed to the proposer.

    ``solve_rate`` (``p``) and ``headroom_regret`` are measured; ``novelty`` is
    filled in relative to an archive at scoring time (an explicit value here acts
    as an override for testing). ``attempts`` gates the collapse guardrail - a
    descriptor with no attempts yet is never rejected as trivial/unsolvable.
    """

    solve_rate: float = 0.0
    headroom_regret: float = 0.0
    attempts: int = 0
    novelty: float = 0.0


@dataclass(frozen=True)
class ScoreWeights:
    learnability: float = 1.0
    regret: float = 0.5
    novelty: float = 0.5


DEFAULT_WEIGHTS = ScoreWeights()


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def learnability(p: float) -> float:
    """UED/PLR learnability ``4*p*(1-p)`` - 0 at p in {0,1}, peaks at 1.0 at p=0.5."""
    p = clamp(p)
    return 4.0 * p * (1.0 - p)


def is_viable(stats: DescriptorStats) -> bool:
    """False iff there is *evidence* the task is unsolvable (p~0) or trivial (p~1).

    Descriptors with no attempts yet are always viable (unknown competence)."""
    if stats.attempts <= 0:
        return True
    return P_UNSOLVABLE < stats.solve_rate < P_TRIVIAL


def score_descriptor(stats: DescriptorStats,
                     weights: ScoreWeights = DEFAULT_WEIGHTS) -> float:
    """Frontier score = learnability (+ regret + novelty), 0 if guardrail-filtered.

    With ``regret == novelty == 0`` the score reduces to learnability and thus
    peaks at ``p = 0.5``. Trivial / unsolvable descriptors (with evidence) score
    exactly ``0.0`` regardless of novelty/regret so they are never proposed."""
    if not is_viable(stats):
        return 0.0
    score = weights.learnability * learnability(stats.solve_rate)
    score += weights.regret * clamp(stats.headroom_regret)
    score += weights.novelty * clamp(stats.novelty)
    return score


# --------------------------------------------------------------------------- #
# Novelty vs the task archive
# --------------------------------------------------------------------------- #
def _key_distance(a: tuple, b: tuple) -> float:
    """Normalized Hamming distance between two niche keys (in ``[0, 1]``)."""
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)


def descriptor_novelty(desc: ts.TaskDescriptor, archive) -> float:
    """Novelty of ``desc`` vs an archive's occupied niches (``[0, 1]``).

    ``1.0`` if the archive is empty; ``0.0`` if the descriptor's own niche is
    already occupied; otherwise the min normalized Hamming distance to the
    nearest occupied niche (crowded neighbourhoods => lower novelty)."""
    if archive is None:
        return 1.0
    occupied = archive.occupied_keys()
    if not occupied:
        return 1.0
    key = ts.descriptor_key(desc)
    if key in occupied:
        return 0.0
    return min(_key_distance(key, k) for k in occupied)


# --------------------------------------------------------------------------- #
# Ranking + proposal
# --------------------------------------------------------------------------- #
def _resolve_stats(desc: ts.TaskDescriptor, history: dict, archive) -> DescriptorStats:
    """Merge measured history for ``desc`` with archive-relative novelty."""
    base = history.get(desc)
    if base is None:
        base = DescriptorStats()
    return DescriptorStats(
        solve_rate=base.solve_rate,
        headroom_regret=base.headroom_regret,
        attempts=base.attempts,
        novelty=descriptor_novelty(desc, archive),
    )


def rank_descriptors(pool, history=None, archive=None,
                     weights: ScoreWeights = DEFAULT_WEIGHTS) -> list:
    """Return ``[(score, descriptor), ...]`` sorted by score desc (deterministic).

    Ties are broken by the descriptor's total order so the ranking is stable."""
    history = history or {}
    scored = []
    for desc in pool:
        stats = _resolve_stats(desc, history, archive)
        scored.append((score_descriptor(stats, weights), desc))
    scored.sort(key=lambda item: (item[0], ts._sort_key(item[1])), reverse=True)
    return scored


def propose(archive, history, n, seed: int = 0, *,
            include_vendor: bool = True,
            weights: ScoreWeights = DEFAULT_WEIGHTS,
            mutate: bool = True,
            max_per_niche: int = 2,
            candidate_pool=None) -> list:
    """Propose ``n`` frontier tasks by selecting + mutating high-learnability ones.

    Steps: (1) build a candidate pool (measured history ∪ the full parametric
    space, unless ``candidate_pool`` is given); (2) rank by
    :func:`score_descriptor` (learnability + regret + archive-novelty); (3) walk
    the ranking, mutating each high-scoring parent (``mutate = perturb
    shape/dtype/fusion-depth``) to mint a *new* frontier task; (4) enforce
    diversity via ``max_per_niche`` so the frontier can't collapse.

    Deterministic given ``seed``. Guardrail: guardrail-filtered (trivial /
    unsolvable) descriptors have score ``0`` and are skipped in the primary pass;
    a novelty-driven fallback guarantees ``n`` tasks are still returned even when
    every measured task has collapsed to trivial/unsolvable."""
    if n <= 0:
        return []
    rng = random.Random(seed)
    history = dict(history or {})
    if candidate_pool is not None:
        pool = list(candidate_pool)
    else:
        pool = list(dict.fromkeys(
            list(history.keys()) + ts.enumerate_descriptors(include_vendor)))
    ranked = rank_descriptors(pool, history, archive, weights)

    out: list = []
    niche_count: dict = {}

    def _try_add(desc) -> bool:
        cand = ts.mutate(desc, rng) if mutate else desc
        key = ts.descriptor_key(cand)
        if niche_count.get(key, 0) >= max_per_niche or cand in out:
            return False
        out.append(cand)
        niche_count[key] = niche_count.get(key, 0) + 1
        return True

    # primary pass: only genuinely-scoring (viable) descriptors.
    for score, desc in ranked:
        if len(out) >= n:
            break
        if score <= 0.0:
            continue
        _try_add(desc)

    # fallback: keep filling toward n (used only in degenerate collapse cases).
    if len(out) < n:
        for score, desc in ranked:
            if len(out) >= n:
                break
            _try_add(desc)

    return out[:n]
