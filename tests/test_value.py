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
from kore.value.features import FEATURE_NAMES, featurize, featurize_many
from kore.value.model import ValueModel
from kore.value.rerank import (
    benches_to_best,
    rank_candidates,
    rank_correlation,
    score_candidates,
    top_k_recall,
)
from kore.value.train_value import synthesize_table, _split_row, _realized_utility


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
    order = rank_candidates(_StubModel(), metas)
    assert order[0] == 4  # highest e_log_speedup is the last meta
    assert order == [4, 3, 2, 1, 0]
    scores = score_candidates(_StubModel(), metas)
    assert scores.shape == (5,)
    assert np.argmax(scores) == 4


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
    scores = score_candidates(_Z(), metas)
    assert scores[0] == 0.0
    assert np.all(scores[1:] > 0.0)


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
    pred_scores = score_candidates(model, te_m)
    true_util = np.array([_realized_utility(o) for o in te_o])
    rc = rank_correlation(pred_scores, true_util)
    # a useful surrogate should be positively correlated with realized utility
    assert rc > 0.1
