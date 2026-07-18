"""CPU tests for the task PROPOSER, focused on the OPTIONAL competitor-anchored
``regret_vs_opus`` term (blend tasks by ``regret_vs_opus x learnability`` so the
curriculum concentrates where KORE is closest to overtaking Opus 4.8).

All pure/CPU (no torch, no GPU): the behavioral tests operate on
:func:`score_descriptor` / :func:`rank_descriptors` with ``archive=None`` (so
novelty is a constant ``1.0`` and no shape/registry lookup is needed), plus one
end-to-end :func:`propose` test over a tiny hand-built candidate pool of
``unary``/``binary``/``reduce`` descriptors (whose niche key is torch-free).

Coverage:
  * OFF-by-default regression guard: ``opus_scores=None`` (and ``{}``) is
    BYTE-IDENTICAL to the learnability+regret+novelty score/ranking/proposal.
  * blending: a high ``regret_vs_opus`` + mid learnability task ranks ABOVE a
    low-regret higher-learnability task (and the ranking flips vs OFF).
  * fail-safe: clamping of out-of-range values, NaN/inf ignored, missing ids and
    malformed maps fall back to the plain score.
  * the ``opus_regret`` weight knob monotonically shifts score + ranking, and
    ``opus_regret=0`` fully disables the term.
"""

from __future__ import annotations

import math

import pytest

from kore.openended import proposer as pr
from kore.openended import task_space as ts
from kore.openended.proposer import (DEFAULT_WEIGHTS, DescriptorStats,
                                     ScoreWeights, clamp, learnability,
                                     rank_descriptors, score_descriptor)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _desc(op: str, *, family: str = "unary", source: str = "genops",
          dtype: str = "bf16", regime: str = "primary") -> ts.TaskDescriptor:
    """A real (pure-dataclass) descriptor; no torch needed to build one."""
    return ts.TaskDescriptor(source, family, op, dtype, regime)


def _stats(p: float, *, hr: float = 0.0, attempts: int = 0,
           nov: float = 0.0) -> DescriptorStats:
    return DescriptorStats(solve_rate=p, headroom_regret=hr, attempts=attempts,
                           novelty=nov)


def _base_score(stats: DescriptorStats, w: ScoreWeights = DEFAULT_WEIGHTS) -> float:
    """The pre-opus scoring formula, recomputed from the module's own helpers.

    This is the reference the ``opus_scores=None`` path must reproduce exactly."""
    if not pr.is_viable(stats):
        return 0.0
    return (w.learnability * learnability(stats.solve_rate)
            + w.regret * clamp(stats.headroom_regret)
            + w.novelty * clamp(stats.novelty))


def _rank_ids(ranked) -> list[str]:
    return [d.task_id for _s, d in ranked]


# --------------------------------------------------------------------------- #
# 0. weights surface: new knob added, defaults + backward-compat preserved
# --------------------------------------------------------------------------- #
def test_proposer_scoreweights_backcompat_and_default_knob():
    # default knob present and documented default value
    assert DEFAULT_WEIGHTS.opus_regret == 1.0
    assert ScoreWeights().opus_regret == 1.0
    # legacy 3-field construction still works and leaves the other fields intact
    w = ScoreWeights(learnability=1.0, regret=0.5, novelty=0.5)
    assert (w.learnability, w.regret, w.novelty, w.opus_regret) == (1.0, 0.5, 0.5, 1.0)
    # and it is still a frozen, value-equal dataclass
    assert ScoreWeights() == ScoreWeights()


# --------------------------------------------------------------------------- #
# 1. REGRESSION GUARD: opus off == byte-identical to the plain score
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,hr,nov,att", [
    (0.5, 0.0, 0.0, 0),
    (0.5, 1.0, 1.0, 4),
    (0.25, 0.4, 0.7, 4),
    (0.9, 0.2, 0.3, 4),
    (0.0, 1.0, 1.0, 0),      # p=0 but no attempts -> viable, learnability 0
    (0.02, 0.9, 0.9, 4),     # unsolvable w/ evidence -> guardrail 0
    (0.99, 0.9, 0.9, 4),     # trivial w/ evidence -> guardrail 0
])
def test_proposer_regret_none_is_byte_identical_score(p, hr, nov, att):
    stats = _stats(p, hr=hr, nov=nov, attempts=att)
    expected = _base_score(stats)
    # not passing the kwarg, passing None, and passing NaN/inf must all match exactly
    assert score_descriptor(stats) == expected
    assert score_descriptor(stats, regret_vs_opus=None) == expected
    assert score_descriptor(stats, DEFAULT_WEIGHTS, regret_vs_opus=None) == expected


def test_proposer_regret_none_pins_known_values():
    # pin the exact formula so an accidental change to the base path is caught
    assert score_descriptor(_stats(0.5)) == 1.0                     # 1*4*.5*.5
    assert score_descriptor(_stats(0.5, hr=1.0, nov=1.0)) == 2.0    # +.5 +.5
    assert score_descriptor(_stats(0.25)) == pytest.approx(0.75)    # 4*.25*.75
    assert score_descriptor(_stats(0.5, attempts=4)) == 1.0         # viable band
    assert score_descriptor(_stats(0.99, attempts=4)) == 0.0        # trivial guardrail


def test_proposer_rank_none_and_empty_map_identical_to_plain():
    pool = [_desc("relu"), _desc("add", family="binary"),
            _desc("sum", family="reduce"), _desc("gelu", dtype="fp16")]
    history = {pool[0]: _stats(0.5, attempts=4), pool[1]: _stats(0.3, attempts=4),
               pool[2]: _stats(0.7, attempts=4)}
    plain = rank_descriptors(pool, history, archive=None)
    assert plain == rank_descriptors(pool, history, archive=None, opus_scores=None)
    assert plain == rank_descriptors(pool, history, archive=None, opus_scores={})


# --------------------------------------------------------------------------- #
# 2. BLENDING: high regret_vs_opus x learnability lifts a task above a
#    higher-learnability but low-regret task (and flips the ranking vs OFF)
# --------------------------------------------------------------------------- #
def test_regret_vs_opus_lifts_midlearnable_task_above_lowregret():
    a = _desc("add", family="binary")     # mid learnability, HIGH regret_vs_opus
    b = _desc("relu")                     # peak learnability, low/absent regret
    history = {a: _stats(0.2, attempts=4),  # learnability 4*.2*.8 = 0.64
               b: _stats(0.5, attempts=4)}  # learnability 1.0
    opus = {a.task_id: 0.9}               # b intentionally absent (missing-id fallback)

    off = rank_descriptors([a, b], history, archive=None)
    on = rank_descriptors([a, b], history, archive=None, opus_scores=opus)

    # OFF: higher raw learnability (b) wins; ON: the competitor term flips it to a
    assert off[0][1] is b and off[1][1] is a
    assert on[0][1] is a and on[1][1] is b
    # b's score is unchanged (it has no opus entry -> plain score)
    b_off = dict((d, s) for s, d in off)[b]
    b_on = dict((d, s) for s, d in on)[b]
    assert b_off == b_on == _base_score(_stats(0.5, nov=1.0))


def test_regret_vs_opus_additive_term_matches_formula():
    d = _desc("relu")
    stats = _stats(0.5, nov=0.3)   # learnability 1.0
    base = score_descriptor(stats)
    boosted = score_descriptor(stats, regret_vs_opus=0.7)
    # additive term = opus_regret(=1.0) * regret_vs_opus * learnability
    assert boosted == pytest.approx(base + 1.0 * 0.7 * learnability(0.5))
    assert boosted > base
    # the boost scales with learnability: a p far from 0.5 gets a smaller boost
    low_learn = _stats(0.1, nov=0.3)        # learnability 4*.1*.9 = 0.36
    boost_hi = score_descriptor(stats, regret_vs_opus=0.7) - base
    boost_lo = (score_descriptor(low_learn, regret_vs_opus=0.7)
                - score_descriptor(low_learn))
    assert boost_lo < boost_hi
    _ = d


def test_regret_vs_opus_never_revives_guardrail_filtered_task():
    # a trivial/unsolvable task (with evidence) stays exactly 0 even with max regret
    trivial = _stats(0.99, hr=1.0, nov=1.0, attempts=4)
    unsolvable = _stats(0.01, hr=1.0, nov=1.0, attempts=4)
    for s in (trivial, unsolvable):
        assert score_descriptor(s, regret_vs_opus=1.0) == 0.0


# --------------------------------------------------------------------------- #
# 3. FAIL-SAFE: clamp out-of-range, ignore NaN/inf, missing-id + malformed maps
# --------------------------------------------------------------------------- #
def test_sanitize_regret_vs_opus_helper():
    s = pr._sanitize_regret_vs_opus
    assert s(None) is None
    assert s(float("nan")) is None
    assert s(float("inf")) is None
    assert s(float("-inf")) is None
    assert s("not-a-number") is None
    assert s(0.3) == 0.3
    assert s(1.5) == 1.0          # clamp high
    assert s(-2.0) == 0.0         # clamp low
    assert s(0.0) == 0.0 and s(1.0) == 1.0


def test_regret_vs_opus_clamps_out_of_range():
    stats = _stats(0.5, nov=0.2)     # learnability 1.0
    base = score_descriptor(stats)
    # >1 clamps to 1.0
    assert score_descriptor(stats, regret_vs_opus=5.0) == pytest.approx(base + 1.0)
    # <0 clamps to 0.0 -> adds nothing
    assert score_descriptor(stats, regret_vs_opus=-3.0) == base
    # exactly-zero regret adds nothing
    assert score_descriptor(stats, regret_vs_opus=0.0) == base


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "x", None])
def test_regret_vs_opus_bad_values_ignored(bad):
    stats = _stats(0.5, nov=0.2)
    assert score_descriptor(stats, regret_vs_opus=bad) == score_descriptor(stats)


def test_regret_vs_opus_missing_ids_fall_back():
    a, b = _desc("add", family="binary"), _desc("relu")
    history = {a: _stats(0.5, attempts=4), b: _stats(0.5, attempts=4)}
    opus = {a.task_id: 0.8}          # b missing
    ranked = dict((d, s) for s, d in
                  rank_descriptors([a, b], history, archive=None, opus_scores=opus))
    assert ranked[a] == pytest.approx(_base_score(_stats(0.5, nov=1.0)) + 0.8)
    assert ranked[b] == _base_score(_stats(0.5, nov=1.0))   # unchanged


def test_lookup_regret_vs_opus_helper_and_key_forms():
    look = pr._lookup_regret_vs_opus
    d = _desc("relu")
    assert look(None, d) is None
    assert look({}, d) is None
    assert look({d.task_id: 0.4}, d) == 0.4          # keyed by task_id (documented)
    assert look({d: 0.6}, d) == 0.6                  # keyed by descriptor (convenience)
    assert look({"other_id": 0.9}, d) is None        # missing id
    assert look(["garbage"], d) is None              # malformed map -> fail-safe None


def test_rank_with_nan_map_entry_is_identical_to_plain():
    a, b = _desc("add", family="binary"), _desc("relu")
    history = {a: _stats(0.5, attempts=4), b: _stats(0.5, attempts=4)}
    plain = rank_descriptors([a, b], history, archive=None)
    noisy = rank_descriptors([a, b], history, archive=None,
                             opus_scores={a.task_id: float("nan"),
                                          b.task_id: float("inf")})
    assert plain == noisy      # NaN/inf entries are ignored -> byte-identical


# --------------------------------------------------------------------------- #
# 4. WEIGHT KNOB: monotone in score AND ranking; opus_regret=0 disables
# --------------------------------------------------------------------------- #
def test_opus_regret_weight_monotone_in_score():
    stats = _stats(0.5, nov=0.0)       # learnability 1.0
    base = score_descriptor(stats, ScoreWeights())         # opus off
    scores = [score_descriptor(stats, ScoreWeights(opus_regret=w), regret_vs_opus=0.8)
              for w in (0.0, 0.5, 1.0, 2.0)]
    assert scores[0] == base                                # weight 0 disables term
    assert scores == sorted(scores)                         # non-decreasing
    assert len(set(scores)) == 4                            # strictly increasing


def test_opus_regret_weight_monotonically_shifts_ranking():
    a = _desc("add", family="binary")     # lower base learnability, high regret
    b = _desc("relu")                     # higher base learnability, no regret
    history = {a: _stats(0.2, attempts=4), b: _stats(0.5, attempts=4)}
    opus = {a.task_id: 1.0}

    lo = rank_descriptors([a, b], history, archive=None,
                          weights=ScoreWeights(opus_regret=0.1), opus_scores=opus)
    hi = rank_descriptors([a, b], history, archive=None,
                          weights=ScoreWeights(opus_regret=2.0), opus_scores=opus)
    # small knob: raw-learnability task b still leads; large knob: a overtakes
    assert lo[0][1] is b
    assert hi[0][1] is a
    # a's score rises monotonically with the knob; b's is invariant to it
    a_scores = [dict((d, s) for s, d in
                     rank_descriptors([a, b], history, archive=None,
                                      weights=ScoreWeights(opus_regret=w),
                                      opus_scores=opus))[a]
                for w in (0.0, 0.5, 1.0, 2.0)]
    assert a_scores == sorted(a_scores) and len(set(a_scores)) == 4


# --------------------------------------------------------------------------- #
# 5. INTEGRATION: opus_scores threads through propose() and shifts the batch
# --------------------------------------------------------------------------- #
def _propose_pool():
    # four distinct niches (distinct family or dtype-precision), all torch-free keys
    d_relu16 = _desc("relu", family="unary", dtype="bf16")
    d_relu32 = _desc("relu", family="unary", dtype="fp32")
    d_add = _desc("add", family="binary", dtype="bf16")
    d_sum = _desc("sum", family="reduce", dtype="bf16")
    return d_relu16, d_relu32, d_add, d_sum


def test_propose_opus_none_is_byte_identical():
    d_relu16, d_relu32, d_add, d_sum = _propose_pool()
    pool = [d_relu16, d_relu32, d_add, d_sum]
    history = {d: _stats(0.5, attempts=4) for d in pool}
    kw = dict(candidate_pool=pool, mutate=False, seed=0)
    out_default = pr.propose(None, history, 3, **kw)
    out_none = pr.propose(None, history, 3, opus_scores=None, **kw)
    out_empty = pr.propose(None, history, 3, opus_scores={}, **kw)
    assert out_default == out_none == out_empty


def test_propose_opus_scores_concentrates_batch_on_high_regret_tasks():
    d_relu16, d_relu32, d_add, d_sum = _propose_pool()
    pool = [d_relu16, d_relu32, d_add, d_sum]
    history = {d: _stats(0.5, attempts=4) for d in pool}   # equal base scores
    boosted = {d_add.task_id, d_sum.task_id}
    opus = {d_add.task_id: 0.9, d_sum.task_id: 0.9}
    kw = dict(candidate_pool=pool, mutate=False, seed=0)

    off = {d.task_id for d in pr.propose(None, history, 2, **kw)}
    on = {d.task_id for d in pr.propose(None, history, 2, opus_scores=opus, **kw)}

    # ON: the batch is exactly the two high-regret tasks; OFF: it excludes them
    assert boosted <= on
    assert not (boosted & off)
    assert on != off


def test_propose_is_deterministic_with_opus_scores():
    d_relu16, d_relu32, d_add, d_sum = _propose_pool()
    pool = [d_relu16, d_relu32, d_add, d_sum]
    history = {d: _stats(0.5, attempts=4) for d in pool}
    opus = {d_add.task_id: 0.9, d_sum.task_id: 0.7}
    kw = dict(candidate_pool=pool, mutate=False, seed=3, opus_scores=opus)
    assert pr.propose(None, history, 3, **kw) == pr.propose(None, history, 3, **kw)


def test_module_still_imports_cleanly():
    # cheap guard mirroring the deliverable's final import check
    assert math.isfinite(learnability(0.5))
    assert hasattr(pr, "score_descriptor") and hasattr(pr, "propose")
