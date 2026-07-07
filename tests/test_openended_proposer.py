"""CPU-only tests for the learnability/regret-targeted task proposer."""

from __future__ import annotations

from kore.openended import task_space as ts
from kore.openended.proposer import (
    DEFAULT_WEIGHTS,
    DescriptorStats,
    ScoreWeights,
    descriptor_novelty,
    is_viable,
    learnability,
    propose,
    rank_descriptors,
    score_descriptor,
)


# --- a minimal fake archive (duck-typed: only occupied_keys is used) -------- #
class _FakeArchive:
    def __init__(self, keys):
        self._keys = set(keys)

    def occupied_keys(self):
        return set(self._keys)


# --------------------------------------------------------------------------- #
# learnability
# --------------------------------------------------------------------------- #
def test_learnability_peaks_at_half():
    assert learnability(0.5) == 1.0
    assert learnability(0.0) == 0.0
    assert learnability(1.0) == 0.0
    # strictly increasing up to 0.5, symmetric
    assert learnability(0.5) > learnability(0.3) > learnability(0.1)
    assert abs(learnability(0.2) - learnability(0.8)) < 1e-12


def test_learnability_clamps_out_of_range():
    assert learnability(-1.0) == 0.0
    assert learnability(2.0) == 0.0


# --------------------------------------------------------------------------- #
# guardrails + scoring
# --------------------------------------------------------------------------- #
def test_is_viable_bands():
    assert is_viable(DescriptorStats(solve_rate=0.5, attempts=10))
    assert not is_viable(DescriptorStats(solve_rate=0.0, attempts=10))   # unsolvable
    assert not is_viable(DescriptorStats(solve_rate=1.0, attempts=10))   # trivial
    # no evidence yet -> always viable
    assert is_viable(DescriptorStats(solve_rate=0.0, attempts=0))


def test_score_filters_trivial_and_unsolvable():
    trivial = DescriptorStats(solve_rate=0.99, attempts=20, novelty=1.0, headroom_regret=1.0)
    unsolv = DescriptorStats(solve_rate=0.01, attempts=20, novelty=1.0, headroom_regret=1.0)
    # even with max novelty + regret, guardrail forces score to 0.
    assert score_descriptor(trivial) == 0.0
    assert score_descriptor(unsolv) == 0.0


def test_score_peaks_at_half_when_only_learnability():
    w = ScoreWeights(learnability=1.0, regret=0.0, novelty=0.0)
    mid = score_descriptor(DescriptorStats(solve_rate=0.5, attempts=10), w)
    off = score_descriptor(DescriptorStats(solve_rate=0.2, attempts=10), w)
    assert mid > off
    assert mid == 1.0


def test_novelty_is_rewarded():
    base = DescriptorStats(solve_rate=0.5, attempts=10, novelty=0.0)
    novel = DescriptorStats(solve_rate=0.5, attempts=10, novelty=1.0)
    assert score_descriptor(novel) > score_descriptor(base)


def test_regret_is_rewarded():
    lo = DescriptorStats(solve_rate=0.5, attempts=10, headroom_regret=0.0)
    hi = DescriptorStats(solve_rate=0.5, attempts=10, headroom_regret=1.0)
    assert score_descriptor(hi) > score_descriptor(lo)


# --------------------------------------------------------------------------- #
# novelty vs archive
# --------------------------------------------------------------------------- #
def test_descriptor_novelty_empty_and_occupied():
    d = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "primary")
    assert descriptor_novelty(d, None) == 1.0
    assert descriptor_novelty(d, _FakeArchive([])) == 1.0
    occ = _FakeArchive([ts.descriptor_key(d)])
    assert descriptor_novelty(d, occ) == 0.0


def test_descriptor_novelty_distance():
    d = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "primary")
    other = ts.TaskDescriptor("genops", "gemm_fusion", "gemm_bias", "bf16", "primary")
    arch = _FakeArchive([ts.descriptor_key(other)])
    nov = descriptor_novelty(d, arch)
    assert 0.0 < nov <= 1.0


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #
def test_rank_prefers_high_learnability_over_trivial():
    learnable = ts.TaskDescriptor("genops", "unary", "relu", "bf16", "primary")
    trivial = ts.TaskDescriptor("genops", "unary", "abs", "bf16", "primary")
    unsolv = ts.TaskDescriptor("genops", "unary", "exp", "bf16", "primary")
    history = {
        learnable: DescriptorStats(solve_rate=0.5, attempts=20),
        trivial: DescriptorStats(solve_rate=0.99, attempts=20),
        unsolv: DescriptorStats(solve_rate=0.01, attempts=20),
    }
    ranked = rank_descriptors([learnable, trivial, unsolv], history, archive=None,
                              weights=ScoreWeights(1.0, 0.0, 0.0))
    order = [d for _, d in ranked]
    assert order[0] == learnable
    # trivial + unsolvable both scored 0
    assert ranked[-1][0] == 0.0


def test_rank_is_deterministic():
    pool = ts.sample_descriptors(20, seed=2)
    r1 = rank_descriptors(pool)
    r2 = rank_descriptors(pool)
    assert r1 == r2


# --------------------------------------------------------------------------- #
# propose
# --------------------------------------------------------------------------- #
def test_propose_returns_n_and_is_deterministic():
    out1 = propose(archive=None, history={}, n=6, seed=0)
    out2 = propose(archive=None, history={}, n=6, seed=0)
    assert out1 == out2
    assert len(out1) == 6
    assert all(isinstance(d, ts.TaskDescriptor) for d in out1)


def test_propose_enforces_niche_diversity():
    out = propose(archive=None, history={}, n=12, seed=1, max_per_niche=1)
    keys = [ts.descriptor_key(d) for d in out]
    assert len(set(keys)) == len(keys)  # every proposed task in a distinct niche


def test_propose_avoids_collapse_on_all_trivial_history():
    # every measured task is trivial/unsolvable -> proposer must still explore.
    pool = ts.sample_descriptors(40, seed=5)
    history = {}
    for i, d in enumerate(pool):
        p = 0.99 if i % 2 == 0 else 0.01
        history[d] = DescriptorStats(solve_rate=p, attempts=30)
    out = propose(archive=None, history=history, n=8, seed=3)
    assert len(out) == 8
    # proposed tasks should not simply echo the trivial/unsolvable measured ones
    trivial_ids = {d.task_id for d in pool}
    assert any(d.task_id not in trivial_ids for d in out)


def test_propose_seeks_novel_niches_given_archive():
    # archive occupies only the unary/memory-bound region; proposer should be
    # willing to leave it (novelty-driven) rather than pile back in.
    occupied = {
        ts.descriptor_key(d)
        for d in ts.enumerate_descriptors() if d.family == "unary"
    }
    arch = _FakeArchive(occupied)
    out = propose(archive=arch, history={}, n=10, seed=4, max_per_niche=1)
    fams = {d.family for d in out}
    assert fams != {"unary"}  # explored beyond the crowded region


def test_propose_zero_n():
    assert propose(archive=None, history={}, n=0, seed=0) == []
