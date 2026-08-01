"""CPU-only tests for the KORE value / selection model.

Covers featurization shape, the 3-head ValueModel (both the sklearn backend and
the pure-numpy fallback), candidate reranking, and the replay-validation metrics
(benches-to-best, top-k recall, Spearman rank correlation).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kore.value import features as F
from kore.value.features import (
    FEATURE_NAMES,
    SCHEDULE_FEATURE_NAMES,
    extract_schedule_features,
    featurize,
    featurize_many,
)
from kore.value.model import PairwiseRanker, ValueModel
from kore.value.rerank import (
    benches_to_best,
    get_default_model,
    rank_candidates,
    rank_correlation,
    score_candidates,
    set_default_model,
    top_k_recall,
)
from kore.value.train_value import (
    groupwise_rank_corr,
    groupwise_top_k_recall,
    synthesize_groups,
    synthesize_table,
    train_ranking,
    _split_row,
    _realized_utility,
)


def _sample_meta():
    return {
        "operation": "gemm",
        "M": 2048,
        "N": 2048,
        "K": 2048,
        "dtype": "bf16",
        "diff_size": 42,
        "parent_snr": 45.0,
        "parent_wall_ms": 1.3,
        "parent_vgpr": 128,
        "pmc_bottleneck": "memory",
    }


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_featurize_fixed_length():
    v = featurize(_sample_meta())
    assert v.shape == (len(FEATURE_NAMES),)
    assert v.dtype == np.float32
    assert np.all(np.isfinite(v))


def test_featurize_handles_missing_and_oov():
    # empty / unknown values must not change vector length
    v = featurize({})
    assert v.shape == (len(FEATURE_NAMES),)
    v2 = featurize({"operation": "some_unknown_op", "dtype": "weirdtype", "dims": [10, 20]})
    assert v2.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(v2))


def test_featurize_many_shape():
    metas = [_sample_meta(), {}, {"operation": "conv", "dims": [1, 2, 3, 4, 5]}]
    X = featurize_many(metas)
    assert X.shape == (3, len(FEATURE_NAMES))
    assert featurize_many([]).shape == (0, len(FEATURE_NAMES))


# --------------------------------------------------------------------------- #
# model (both backends)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("use_sklearn", [True, False])
def test_value_model_fit_predict(use_sklearn):
    rows = synthesize_table(300, seed=3)
    metas, outs = zip(*[_split_row(r) for r in rows])
    X = featurize_many(list(metas))
    y_compile = np.array([1 if o["compiled"] else 0 for o in outs])
    y_snr = np.array([1 if o["snr_pass"] else 0 for o in outs])
    y_ls = np.array([o["log_speedup"] for o in outs])
    sw = np.array([max(o["speedup"], 0.1) for o in outs])

    model = ValueModel(use_sklearn=use_sklearn)
    model.fit(X, y_compile, y_snr, y_ls, sample_weight=sw)
    pred = model.predict(X)

    assert pred["p_compile"].shape == (len(rows),)
    assert np.all(pred["p_compile"] >= 0.0) and np.all(pred["p_compile"] <= 1.0)
    assert np.all(pred["p_snr_pass"] >= 0.0) and np.all(pred["p_snr_pass"] <= 1.0)
    assert np.all(np.isfinite(pred["e_log_speedup"]))


def test_numpy_fallback_forced():
    # Guarantee the sklearn-optional path works even when sklearn is installed.
    model = ValueModel(use_sklearn=False)
    assert model.backend == "numpy"
    rows = synthesize_table(120, seed=5)
    metas, outs = zip(*[_split_row(r) for r in rows])
    X = featurize_many(list(metas))
    model.fit(
        X,
        [o["compiled"] for o in outs],
        [o["snr_pass"] for o in outs],
        [o["log_speedup"] for o in outs],
        sample_weight=[max(o["speedup"], 0.1) for o in outs],
    )
    pred = model.predict(X)
    assert np.all(np.isfinite(pred["e_log_speedup"]))


def test_model_save_load(tmp_path):
    rows = synthesize_table(80, seed=7)
    metas, outs = zip(*[_split_row(r) for r in rows])
    X = featurize_many(list(metas))
    model = ValueModel(use_sklearn=False)
    model.fit(X, [o["compiled"] for o in outs], [o["snr_pass"] for o in outs],
              [o["log_speedup"] for o in outs])
    p = tmp_path / "m.pkl"
    model.save(str(p))
    loaded = ValueModel.load(str(p))
    a = model.predict(X)
    b = loaded.predict(X)
    assert np.allclose(a["e_log_speedup"], b["e_log_speedup"])


def test_single_class_target_is_handled():
    # all-compiled target -> classifier must not crash, returns valid probs
    X = featurize_many([_sample_meta() for _ in range(10)])
    model = ValueModel(use_sklearn=False)
    model.fit(X, np.ones(10), np.zeros(10), np.zeros(10))
    pred = model.predict(X)
    assert np.all((pred["p_compile"] >= 0) & (pred["p_compile"] <= 1))
    assert np.all((pred["p_snr_pass"] >= 0) & (pred["p_snr_pass"] <= 1))


# --------------------------------------------------------------------------- #
# rerank: scoring + ranking
# --------------------------------------------------------------------------- #
class _StubModel:
    """Deterministic model: utility increases with row index (last is best)."""

    def predict(self, X):
        n = X.shape[0]
        return {
            "p_compile": np.ones(n),
            "p_snr_pass": np.ones(n),
            "e_log_speedup": np.arange(n, dtype=float),
        }


def test_rank_candidates_best_first():
    metas = [_sample_meta() for _ in range(5)]
    order = rank_candidates(metas, model=_StubModel())
    assert order[0] == 4  # highest e_log_speedup is the last meta
    assert order == [4, 3, 2, 1, 0]
    scores = score_candidates(metas, model=_StubModel())
    assert isinstance(scores, list) and len(scores) == 5
    assert int(np.argmax(scores)) == 4


def test_score_is_validity_gated():
    # p_compile=0 must zero out the utility regardless of speedup
    class _Z:
        def predict(self, X):
            n = X.shape[0]
            return {
                "p_compile": np.array([0.0] + [1.0] * (n - 1)),
                "p_snr_pass": np.ones(n),
                "e_log_speedup": np.full(n, 2.0),
            }

    metas = [_sample_meta() for _ in range(3)]
    scores = score_candidates(metas, model=_Z())
    assert scores[0] == 0.0
    assert all(s > 0.0 for s in scores[1:])


# --------------------------------------------------------------------------- #
# rerank: P0 CONTRACT - rank_candidates(items, task=None) + untrained fallback
# --------------------------------------------------------------------------- #
_GOOD_KERNEL = """
import triton
import triton.language as tl

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        x = tl.load(a_ptr + offs, mask=offs < K, other=0.0)
        acc += tl.dot(x, x)
    tl.store(c_ptr + offs, acc, mask=offs < N)

def entry(a, b):
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
    return _mm[(1,)](a, b, num_warps=8, num_stages=2)
"""

_BAD_KERNEL = """
def entry(a, b):
    BLOCK_M, BLOCK_N, BLOCK_K = 96, 96, 48
    acc = bf16_accumulate(a, b)
    return acc
"""


class _FakeTask:
    task_id = "gemm_bf16"
    operation = "gemm"
    dtype = "bf16"
    gpu_target = "gfx942"

    class _S:
        name = "s"
        dims = {"M": 2048, "N": 2048, "K": 2048}

    shapes = [_S()]


def test_rank_candidates_accepts_sources_and_task():
    # items may be raw kernel-source strings; task supplies problem context.
    items = [_BAD_KERNEL, _GOOD_KERNEL]
    order = rank_candidates(items, task=_FakeTask())
    assert set(order) == {0, 1}
    # untrained -> heuristic fallback should still prefer the well-formed kernel
    assert order[0] == 1


def test_rank_candidates_untrained_fallback_no_model():
    # No default model installed: must not crash and must return a permutation.
    assert get_default_model() is None
    items = [{"source": _GOOD_KERNEL}, {"source": _BAD_KERNEL}, {"source": ""}]
    scores = score_candidates(items)
    assert isinstance(scores, list) and len(scores) == 3
    order = rank_candidates(items)
    assert sorted(order) == [0, 1, 2]
    assert order[0] == 0  # good kernel ranks first under the heuristic
    # empty inputs are handled
    assert rank_candidates([]) == []
    assert score_candidates([]) == []


def test_set_default_model_is_used(tmp_path):
    # installing a fitted default model makes rank_candidates use it (not heuristic)
    rows = synthesize_table(200, seed=4)
    metas, outs = zip(*[_split_row(r) for r in rows])
    X = featurize_many(list(metas))
    model = ValueModel(use_sklearn=False)
    model.fit(X, [o["compiled"] for o in outs], [o["snr_pass"] for o in outs],
              [o["log_speedup"] for o in outs])
    try:
        set_default_model(model)
        assert get_default_model() is model
        order = rank_candidates([_GOOD_KERNEL, _BAD_KERNEL], task=_FakeTask())
        assert sorted(order) == [0, 1]
    finally:
        set_default_model(None)


# --------------------------------------------------------------------------- #
# replay-validation metrics
# --------------------------------------------------------------------------- #
def test_benches_to_best_perfect_and_worst():
    true = np.array([0.1, 0.5, 0.9, 0.3, 0.2])
    n = len(true)
    # perfect predictor: pred == true -> best ranked first
    assert benches_to_best(true, true) == 1
    # worst predictor: pred anti-correlated -> best ranked last
    assert benches_to_best(-true, true) == n


def test_top_k_recall():
    true = np.array([10.0, 9.0, 8.0, 1.0, 2.0])
    # perfect agreement -> full recall
    assert top_k_recall(true, true, 3) == 1.0
    # pred that gets exactly 2 of the true top-3 right
    pred = np.array([10.0, 9.0, 0.0, 0.5, 100.0])  # top-3 pred = {4,0,1}
    r = top_k_recall(pred, true, 3)
    assert abs(r - (2.0 / 3.0)) < 1e-9


def test_rank_correlation_sane():
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(rank_correlation(true, true) - 1.0) < 1e-9
    assert abs(rank_correlation(-true, true) + 1.0) < 1e-9
    # partial agreement -> strictly between
    pred = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    rc = rank_correlation(pred, true)
    assert 0.0 < rc < 1.0


def test_rank_correlation_degenerate():
    # constant predictions -> undefined -> 0.0, no crash
    assert rank_correlation(np.ones(5), np.arange(5)) == 0.0
    assert rank_correlation([1.0], [1.0]) == 0.0


# --------------------------------------------------------------------------- #
# end-to-end: learned ranker beats random on synthetic data
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("use_sklearn", [True, False])
def test_learned_ranker_helps(use_sklearn):
    rows = synthesize_table(500, seed=11)
    metas, outs = zip(*[_split_row(r) for r in rows])
    metas = list(metas)
    split = int(len(rows) * 0.75)
    tr_m, te_m = metas[:split], metas[split:]
    tr_o, te_o = outs[:split], outs[split:]

    X = featurize_many(tr_m)
    model = ValueModel(use_sklearn=use_sklearn)
    model.fit(
        X,
        [o["compiled"] for o in tr_o],
        [o["snr_pass"] for o in tr_o],
        [o["log_speedup"] for o in tr_o],
        sample_weight=[max(o["speedup"], 0.1) for o in tr_o],
    )
    pred_scores = score_candidates(te_m, model=model)
    true_util = np.array([_realized_utility(o) for o in te_o])
    rc = rank_correlation(pred_scores, true_util)
    # a useful surrogate should be positively correlated with realized utility
    assert rc > 0.1


# --------------------------------------------------------------------------- #
# candidate-schedule features (Ansor/NLTSP: featurize the SCHEDULE)
# --------------------------------------------------------------------------- #
def test_schedule_feature_extraction_from_source():
    s = extract_schedule_features(_GOOD_KERNEL)
    assert s["has_source"] is True
    assert s["block_m"] == 128 and s["block_n"] == 128 and s["block_k"] == 64
    assert s["group_m"] == 8
    assert s["num_warps"] == 8 and s["num_stages"] == 2
    assert s["has_tl_dot"] is True
    assert s["has_fp32_acc"] is True
    assert s["has_mask"] is True
    assert s["has_reduction_loop"] is True
    assert s["blocks_mult64"] == 1.0
    assert s["tile_area"] == 128 * 128


def test_schedule_features_detects_bad_schedule():
    s = extract_schedule_features(_BAD_KERNEL)
    # 96 / 48 are not 64-multiples; no tl.dot; no fp32 accumulate
    assert s["blocks_mult64"] == 0.0
    assert s["has_tl_dot"] is False
    assert s["has_fp32_acc"] is False


def test_schedule_features_backward_compatible_layout():
    # a move without source keeps the vector length and zeros the schedule block
    v_no_src = featurize({"operation": "gemm", "M": 2048, "N": 2048, "K": 2048})
    v_src = featurize({"operation": "gemm", "M": 2048, "N": 2048, "K": 2048,
                       "source": _GOOD_KERNEL})
    assert v_no_src.shape == v_src.shape == (len(FEATURE_NAMES),)
    assert set(SCHEDULE_FEATURE_NAMES) <= set(FEATURE_NAMES)
    # the schedule block differs once a source is present
    idx = [FEATURE_NAMES.index(n) for n in SCHEDULE_FEATURE_NAMES]
    assert not np.allclose(v_no_src[idx], v_src[idx])
    # the problem block (everything before the schedule block) is identical
    first_sched = min(idx)
    assert np.allclose(v_no_src[:first_sched], v_src[:first_sched])


# --------------------------------------------------------------------------- #
# ranking objective: within-group order + throughput-weighted pairwise ranker
# --------------------------------------------------------------------------- #
def test_pairwise_ranker_learns_within_group_order():
    groups = synthesize_groups(120, group_size=6, seed=1)
    tr, te = groups[:90], groups[90:]
    model = train_ranking(tr, use_sklearn=False)

    def ranker_fn(metas):
        return model.rank_scores(featurize_many(metas))

    def untrained_fn(metas):
        # a zero-weight ranker -> identity/degenerate ordering (baseline ~0 corr)
        return PairwiseRanker().predict(featurize_many(metas))

    trained = groupwise_rank_corr(ranker_fn, te)
    baseline = groupwise_rank_corr(untrained_fn, te)
    # the ranking objective must improve within-group rank correlation
    assert trained > 0.2
    assert trained > baseline + 0.1


def test_train_ranking_attaches_ranker():
    groups = synthesize_groups(40, group_size=5, seed=2)
    model = train_ranking(groups, use_sklearn=False)
    assert model.ranker is not None and model.ranker.fitted
    X = featurize_many([_split_row(groups[0][0])[0]])
    assert np.all(np.isfinite(model.rank_scores(X)))


# --------------------------------------------------------------------------- #
# the trained ranking head is CONSULTED by the reranker (not just fitted)
# --------------------------------------------------------------------------- #
def test_score_candidates_routes_through_the_fitted_ranking_head():
    """``train_ranking`` always fits a pairwise head; ``score_candidates`` must read
    it. Regression guard: the head used to be trained on every run and ignored."""
    from kore.value import rerank as rr

    groups = synthesize_groups(120, group_size=6, seed=5)
    model = train_ranking(groups, use_sklearn=False)
    assert model.ranker is not None and model.ranker.fitted

    metas = [{"operation": "gemm", "M": 4096, "N": 4096, "K": 4096, "dtype": "bf16",
              "source": src} for src in (_GOOD_KERNEL, _BAD_KERNEL)]
    X = featurize_many(metas)

    # the reranker's utility IS the model's rank_scores when a ranker is fitted...
    assert np.allclose(rr._model_utility(model, X), model.rank_scores(X))
    # ...and that is genuinely different from the pointwise-only utility
    assert not np.allclose(rr._model_utility(model, X),
                           rr._pointwise_utility(model, X))


def test_score_candidates_falls_back_to_pointwise_without_a_ranker():
    """A model with no ranking head keeps the historical pointwise utility exactly."""
    from kore.value import rerank as rr

    rows = synthesize_table(300, seed=6)
    metas, outs = zip(*[_split_row(r) for r in rows])
    model = ValueModel(use_sklearn=False)
    model.fit(featurize_many(list(metas)),
              [o["compiled"] for o in outs],
              [o["snr_pass"] for o in outs],
              [o["log_speedup"] for o in outs])
    assert model.ranker is None

    X = featurize_many([_sample_meta()])
    assert np.allclose(rr._model_utility(model, X), rr._pointwise_utility(model, X))


def test_model_utility_tolerates_a_model_pickled_before_the_ranker_existed():
    """``rank_scores`` touches ``self.ranker``, which a pre-ranker pickle lacks. The
    reranker must probe with getattr and never raise AttributeError."""
    from kore.value import rerank as rr

    rows = synthesize_table(200, seed=7)
    metas, outs = zip(*[_split_row(r) for r in rows])
    model = ValueModel(use_sklearn=False)
    model.fit(featurize_many(list(metas)),
              [o["compiled"] for o in outs],
              [o["snr_pass"] for o in outs],
              [o["log_speedup"] for o in outs])
    del model.ranker  # exactly what an old artifact unpickles as
    assert not hasattr(model, "ranker")

    X = featurize_many([_sample_meta(), _sample_meta()])
    assert np.all(np.isfinite(rr._model_utility(model, X)))
    assert len(score_candidates([_GOOD_KERNEL, _BAD_KERNEL], model=model)) == 2


def test_groupwise_top_k_recall_rewards_getting_the_benched_candidates_right():
    """top-k recall is the metric matching how the model is USED (bench only top-k)."""
    groups = synthesize_groups(120, group_size=6, seed=8)
    tr, te = groups[:90], groups[90:]
    model = train_ranking(tr, use_sklearn=False)

    def trained_fn(metas):
        return model.rank_scores(featurize_many(metas))

    def flat_fn(metas):
        return np.zeros(len(metas))

    r_trained = groupwise_top_k_recall(trained_fn, te, 2)
    r_flat = groupwise_top_k_recall(flat_fn, te, 2)
    assert 0.0 <= r_flat <= 1.0
    assert 0.0 <= r_trained <= 1.0
    assert r_trained > r_flat
    # a perfect scorer recovers the whole true top-k
    def oracle_fn(metas):
        return [1.0] * len(metas)
    assert groupwise_top_k_recall(oracle_fn, [], 2) == 0.0


# --------------------------------------------------------------------------- #
# a STALE artifact must degrade to the heuristic, never raise into a rollout
# --------------------------------------------------------------------------- #
def test_score_candidates_survives_a_stale_feature_layout():
    """A model fit under a narrower feature layout raises on predict. Ranking is
    never worth failing a rollout over, so it must fall back to the heuristic."""
    class _StaleModel:
        fitted = True

        def predict(self, X):
            raise ValueError(f"X has {np.asarray(X).shape[1]} features, "
                             f"but this estimator is expecting 28")

    scores = score_candidates([_GOOD_KERNEL, _BAD_KERNEL], model=_StaleModel())
    assert len(scores) == 2
    assert scores[0] > scores[1]  # the heuristic still gives a varied, ordered prior
    assert rank_candidates([_GOOD_KERNEL, _BAD_KERNEL], model=_StaleModel()) == [0, 1]


def _fit_stale_model(tmp_path) -> str:
    """Save a REAL artifact fit under a NARROWER feature layout.

    This is the exact shape of the audited ``runs/value/value_model.pkl``: it was
    trained before the 19 candidate-schedule features existed, so it expects 28
    columns and raises on today's 47-column matrix.
    """
    n_narrow = len(FEATURE_NAMES) - len(SCHEDULE_FEATURE_NAMES)
    rows = synthesize_table(120, seed=21)
    metas, outs = zip(*[_split_row(r) for r in rows])
    X_narrow = featurize_many(list(metas))[:, :n_narrow]
    model = ValueModel(use_sklearn=False)
    model.fit(X_narrow,
              [o["compiled"] for o in outs],
              [o["snr_pass"] for o in outs],
              [o["log_speedup"] for o in outs])
    p = tmp_path / "stale.pkl"
    model.save(str(p))
    return str(p)


def test_load_default_model_rejects_a_stale_artifact(tmp_path):
    """``load_default_model`` must not install an artifact that cannot answer under
    the current featurizer, or the first real prediction crashes the rollout."""
    from kore.value.rerank import get_default_model, load_default_model, set_default_model

    path = _fit_stale_model(tmp_path)
    # the artifact really is unusable under the current layout
    stale = ValueModel.load(path)
    with pytest.raises(Exception):
        stale.predict(featurize_many([_sample_meta(), _sample_meta()]))

    try:
        set_default_model(None)
        assert load_default_model(path) is None
        assert get_default_model() is None  # never installed
    finally:
        set_default_model(None)


def test_load_default_model_installs_a_serviceable_artifact(tmp_path):
    from kore.value.rerank import get_default_model, load_default_model, set_default_model

    groups = synthesize_groups(40, group_size=5, seed=9)
    model = train_ranking(groups, use_sklearn=False)
    p = tmp_path / "good.pkl"
    model.save(str(p))
    try:
        set_default_model(None)
        loaded = load_default_model(str(p))
        assert loaded is not None
        assert get_default_model() is loaded
    finally:
        set_default_model(None)
