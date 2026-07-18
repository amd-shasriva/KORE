"""CPU tests: ``score_candidates`` / ``rank_candidates`` as a REAL AlphaKernel PUCT prior.

The AlphaKernel search drives its PUCT priors from
``value_fn(sources, task) -> list[float]`` where ``sources`` are raw Triton kernel
SOURCE STRINGS (see ``kore.search.alphakernel._value_scores`` and the grpo wiring
``lambda sources, task: score_candidates(list(sources), task=task, model=vm)``).

These tests pin the contract that makes that prior actually guide search:

  * raw SOURCE STRINGS are accepted and their schedule features are read (they are
    NOT silently dropped) -- distinct sources -> distinct, finite, sensibly-ordered
    scores, with better-formed / faster-looking kernels scoring higher;
  * a missing model falls back to the varied source heuristic (never uniform);
  * a *usable but schedule-blind* model (fit without source features, so it ignores
    the kernel source and returns a constant utility) is detected and the always-
    varied heuristic is substituted -- so the prior is never a no-op;
  * a model that MEANINGFULLY discriminates the candidates is used as-is (the
    heuristic never overrides it);
  * the object / dict input paths are unchanged (backward compatible);
  * the exact grpo ``value_fn`` lambda returns a usable, VARIED ``list[float]``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kore.value.features import featurize, featurize_many, N_FEATURES
from kore.value.model import ValueModel
from kore.value.rerank import (
    _has_distinct_feature_rows,
    _heuristic_scores,
    _is_degenerate,
    _item_meta,
    _model_utility,
    _task_meta,
    get_default_model,
    rank_candidates,
    score_candidates,
    set_default_model,
)
from kore.value.train_value import (
    _split_row,
    synthesize_groups,
    synthesize_table,
    train_ranking,
)


# --------------------------------------------------------------------------- #
# Kernel-source fixtures with clearly different gfx942 "goodness"
# --------------------------------------------------------------------------- #
GOOD = """
import triton
import triton.language as tl

@triton.jit
def _gemm(a_ptr, b_ptr, c_ptr, M, N, K,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + kk, mask=kk < K, other=0.0)
        b = tl.load(b_ptr + kk, mask=kk < K, other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr, acc, mask=acc >= 0)

def entry(a, b):
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8
    return _gemm[(1,)](a, b, num_warps=8, num_stages=2)
"""

MED = """
import triton
import triton.language as tl

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    acc += tl.dot(a, b)
    tl.store(c_ptr, acc)

def entry(a, b):
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    return _mm[(1,)](a, b, num_warps=4)
"""

BAD = """
def entry(a, b):
    BLOCK_M, BLOCK_N, BLOCK_K = 96, 96, 48
    return reference_matmul(a, b)
"""

EMPTY = ""

# schedule-IDENTICAL siblings that differ only in a comment (same discipline flags,
# same tiles/warps/stages) -- the tie-breaker must still separate them.
SIB_A = GOOD
SIB_B = GOOD.replace("acc += tl.dot(a, b)", "acc += tl.dot(a, b)  # fused MAC")

# a BAD kernel padded to be very LONG: length must never outweigh the discipline
# signal (a well-formed short kernel still beats a long ill-formed one).
BAD_LONG = BAD + "\n" + ("# padding padding padding padding\n" * 400)

DISTINCT = [BAD, MED, GOOD, EMPTY]


class FakeTask:
    """Minimal Task-like object supplying problem context (as grpo would)."""

    task_id = "gemm_bf16"
    operation = "gemm"
    dtype = "bf16"

    class _S:
        name = "s"
        dims = {"M": 4096, "N": 4096, "K": 4096}

    shapes = [_S()]


# --------------------------------------------------------------------------- #
# Stub models (deterministic; do not need training)
# --------------------------------------------------------------------------- #
class ConstModel:
    """Usable + fitted, but returns a CONSTANT utility for every candidate.

    This is the schedule-blind / no-op-prior pathology: the model ignores whatever
    makes the candidates different."""

    fitted = True

    def predict(self, X):
        n = X.shape[0]
        return {
            "p_compile": np.full(n, 0.9),
            "p_snr_pass": np.full(n, 0.8),
            "e_log_speedup": np.full(n, 0.5),
        }


class RampModel:
    """Usable + fitted, discriminates candidates by row index (last is best)."""

    fitted = True

    def predict(self, X):
        n = X.shape[0]
        return {
            "p_compile": np.ones(n),
            "p_snr_pass": np.ones(n),
            "e_log_speedup": np.arange(n, dtype=float),
        }


# --------------------------------------------------------------------------- #
# 1. raw source strings -> distinct, finite, sensibly-ordered (heuristic path)
# --------------------------------------------------------------------------- #
def test_distinct_sources_give_distinct_finite_ordered_scores():
    scores = score_candidates(DISTINCT, task=None, model=None)
    assert isinstance(scores, list) and len(scores) == len(DISTINCT)
    assert all(isinstance(s, float) for s in scores)
    assert all(math.isfinite(s) for s in scores)
    # all DISTINCT (no collapse to a uniform prior)
    assert len(set(scores)) == len(scores)
    b, m, g, e = scores
    # better-formed / faster-looking kernels score strictly higher
    assert g > m > b > e == 0.0


def test_rank_candidates_best_first_from_sources():
    order = rank_candidates(DISTINCT, task=None, model=None)
    assert sorted(order) == [0, 1, 2, 3]
    assert order[0] == 2  # GOOD ranks first
    assert order[-1] == 3  # EMPTY (no source) ranks last


def test_source_is_read_not_dropped():
    # A non-empty source produces a non-zero heuristic; an empty one scores 0.
    assert score_candidates([GOOD], model=None)[0] > 0.0
    assert score_candidates([EMPTY], model=None)[0] == 0.0
    # and the source materially changes the model feature vector (it is read, not
    # silently dropped) -- the whole reason a source-string prior can be non-uniform.
    assert not np.allclose(featurize({"source": GOOD}), featurize({}))


def test_task_context_merged_without_breaking_sources():
    scores = score_candidates(DISTINCT, task=FakeTask(), model=None)
    assert len(scores) == 4 and all(math.isfinite(s) for s in scores)
    # heuristic is source-driven, so ordering holds with a task attached too
    assert scores[2] == max(scores)


# --------------------------------------------------------------------------- #
# 2. None / unusable model -> heuristic fallback (never uniform)
# --------------------------------------------------------------------------- #
def test_none_model_falls_back_to_heuristic_not_uniform():
    assert get_default_model() is None
    scores = score_candidates(DISTINCT, model=None)
    assert len(set(scores)) > 1  # NOT uniform
    assert np.allclose(scores, _heuristic_scores([_item_meta(s, {}) for s in DISTINCT]))


def test_unfitted_model_is_not_used():
    class _Unfitted:
        fitted = False

        def predict(self, X):  # pragma: no cover - must never be called
            raise AssertionError("unfitted model must not be consulted")

    scores = score_candidates(DISTINCT, model=_Unfitted())
    assert len(set(scores)) > 1  # heuristic, varied


def test_empty_input():
    assert score_candidates([]) == []
    assert rank_candidates([]) == []


# --------------------------------------------------------------------------- #
# 3. no-op-prior GUARD: schedule-blind model over DISTINCT sources -> heuristic
# --------------------------------------------------------------------------- #
def test_constant_model_over_distinct_sources_defers_to_heuristic():
    X = featurize_many([_item_meta(s, {}) for s in DISTINCT])
    raw = _model_utility(ConstModel(), X)
    # the model itself is a no-op prior over these genuinely-distinct candidates
    assert _is_degenerate(raw)
    assert _has_distinct_feature_rows(X)
    # ...but score_candidates recovers a VARIED, ordered prior via the heuristic
    scores = score_candidates(DISTINCT, model=ConstModel())
    assert len(set(scores)) > 1
    assert scores[2] == max(scores)  # GOOD still first
    assert np.allclose(scores, _heuristic_scores([_item_meta(s, {}) for s in DISTINCT]))


def test_constant_model_over_identical_sources_stays_constant():
    # identical candidates -> a constant score is CORRECT; the heuristic must not
    # manufacture spurious variation.
    items = [GOOD, GOOD, GOOD]
    X = featurize_many([_item_meta(s, {}) for s in items])
    assert not _has_distinct_feature_rows(X)
    scores = score_candidates(items, model=ConstModel())
    assert len(set(scores)) == 1
    assert scores[0] == pytest.approx(0.9 * 0.8 * math.exp(0.5))


def test_discriminating_model_is_used_as_is_not_overridden():
    # RampModel says the LAST candidate (BAD) is best; the heuristic prefers GOOD.
    # A discriminating model must win (never be overridden by the heuristic).
    items = [GOOD, BAD]
    scores = score_candidates(items, model=RampModel())
    assert scores == [pytest.approx(math.exp(0.0)), pytest.approx(math.exp(1.0))]
    assert int(np.argmax(scores)) == 1  # model's opinion, not the heuristic's


# --------------------------------------------------------------------------- #
# 4. exact grpo value_fn lambda: schedule-blind trained model -> VARIED list[float]
# --------------------------------------------------------------------------- #
def _train_blind_model(seed: int = 3, n: int = 400) -> ValueModel:
    """A ValueModel trained on synthesize_table rows, which carry NO `source` -- so
    it learns ~zero weight on the schedule features and ignores the kernel source
    (exactly the production `train_value.py __main__` path)."""
    rows = synthesize_table(n, seed=seed)
    metas, outs = zip(*[_split_row(r) for r in rows])
    X = featurize_many(list(metas))
    m = ValueModel(use_sklearn=False)
    m.fit(
        X,
        [o["compiled"] for o in outs],
        [o["snr_pass"] for o in outs],
        [o["log_speedup"] for o in outs],
        sample_weight=[max(o["speedup"], 0.1) for o in outs],
    )
    return m


@pytest.mark.parametrize("task", [None, FakeTask()])
def test_exact_grpo_value_fn_lambda_returns_varied_usable_scores(task):
    vm = _train_blind_model()
    # EXACT lambda the orchestrator builds in kore/policy/grpo.py:
    value_fn = lambda sources, task: score_candidates(list(sources), task=task, model=vm)  # noqa: E731

    # the trained model is degenerate (a no-op prior) over these distinct sources,
    # built exactly as score_candidates does (task context merged into each item)
    X = featurize_many([_item_meta(s, _task_meta(task)) for s in DISTINCT])
    assert _is_degenerate(_model_utility(vm, X))
    assert _has_distinct_feature_rows(X)

    out = value_fn(DISTINCT, task)
    assert isinstance(out, list) and len(out) == len(DISTINCT)
    assert all(isinstance(x, float) and math.isfinite(x) for x in out)
    assert len(set(out)) > 1                       # VARIED -> a real PUCT prior
    assert out[2] == max(out)                      # GOOD scores highest
    # softmax priors are non-uniform (what PUCT actually consumes)
    p = np.exp(out - np.max(out)); p = p / p.sum()
    assert float(p.max() - p.min()) > 1e-3


def test_grpo_value_fn_with_schedule_aware_model_uses_model_signal():
    # A model trained on groups that DO carry `source` learns the schedule and
    # discriminates candidates on its own -> used as-is (not the heuristic).
    groups = synthesize_groups(160, group_size=6, seed=7)
    vm = train_ranking(groups, use_sklearn=False)
    value_fn = lambda sources, task: score_candidates(list(sources), task=task, model=vm)  # noqa: E731
    # this model actually discriminates the candidates (not a no-op), so the guard
    # does NOT fire and the model's own utility is used.
    X = featurize_many([_item_meta(s, {}) for s in DISTINCT])
    assert not _is_degenerate(_model_utility(vm, X))
    out = value_fn(DISTINCT, None)
    assert len(out) == 4 and all(math.isfinite(x) for x in out)
    assert len(set(out)) > 1
    # the learned model ranks the well-formed kernel above the invalid one
    assert out[2] > out[0]
    # and it is genuinely the MODEL's signal, not the heuristic's numbers
    heur = _heuristic_scores([_item_meta(s, {}) for s in DISTINCT])
    assert not np.allclose(out, heur)


# --------------------------------------------------------------------------- #
# 5. backward compatibility: dict / object / bytes input paths
# --------------------------------------------------------------------------- #
class _Cand:
    def __init__(self, source):
        self.source = source


def test_object_and_dict_paths_match_string_path():
    strv = score_candidates([BAD, GOOD], model=None)
    objv = score_candidates([_Cand(BAD), _Cand(GOOD)], model=None)
    dictv = score_candidates([{"source": BAD}, {"source": GOOD}], model=None)
    assert np.allclose(objv, strv)
    assert np.allclose(dictv, strv)


def test_bytes_source_is_decoded_not_dropped():
    strv = score_candidates([BAD, GOOD], model=None)
    bytesv = score_candidates([BAD.encode("utf-8"), GOOD.encode("utf-8")], model=None)
    assert np.allclose(bytesv, strv)


def test_object_without_source_attr_scores_zero():
    class _NoSource:
        pass

    assert score_candidates([_NoSource()], model=None) == [0.0]


def test_existing_object_stub_from_test_value_still_ranks_by_model():
    # mirrors tests/test_value.py::test_rank_candidates_best_first (dict metas +
    # a model keyed on row index): behavior must be unchanged.
    metas = [{"operation": "gemm", "M": 2048, "N": 2048, "K": 2048} for _ in range(5)]
    order = rank_candidates(metas, model=RampModel())
    assert order == [4, 3, 2, 1, 0]


# --------------------------------------------------------------------------- #
# 6. tie-breaker properties
# --------------------------------------------------------------------------- #
def test_schedule_identical_siblings_get_distinct_scores():
    a, b = score_candidates([SIB_A, SIB_B], model=None)
    assert a != b                    # never a flat prior among siblings
    assert abs(a - b) < 0.05         # ...but the difference is a tiny tie-breaker


def test_length_tiebreaker_never_flips_discipline():
    # a well-formed SHORT kernel must beat a very LONG ill-formed one
    g, bl = score_candidates([GOOD, BAD_LONG], model=None)
    assert g > bl


def test_default_model_install_still_varied_when_blind():
    # grpo's _activate_value_ranker installs the model as default; a blind default
    # must still yield a varied prior through the guard.
    vm = _train_blind_model()
    try:
        set_default_model(vm)
        scores = score_candidates(DISTINCT)  # model=None -> uses default
        assert len(set(scores)) > 1
        assert scores[2] == max(scores)
    finally:
        set_default_model(None)


# --------------------------------------------------------------------------- #
# 7. layout invariant: featurization width is unchanged (trained-model compat)
# --------------------------------------------------------------------------- #
def test_feature_layout_width_unchanged():
    # the trained value_model.pkl was fit at this width; changing it would break
    # ValueModel.predict on load. The heuristic tie-breaker / new dict keys must
    # NOT alter the feature vector length.
    assert N_FEATURES == featurize({"source": GOOD}).shape[0]
    assert featurize({}).shape[0] == N_FEATURES
