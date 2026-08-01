"""KORE candidate reranking + replay-validation metrics.

The value model turns each candidate move into three predictions; this module
collapses them into a single scalar *utility* and ranks candidates so the
scheduler benches only the most promising ones first.

    utility = p_compile * p_snr_pass * exp(e_log_speedup)
            = E[speedup] gated by the probability the candidate is even valid.

When the model also carries a fitted pairwise RANKING head (every model trained by
``train_value.train_ranking`` does, which is what the campaign trains via
``replay_train.train_value_from_groups``), that head supplies the ordering instead:
it is fit on within-group candidate order, which is precisely what a top-k bench
selector consumes. See :func:`_model_utility`.

Analogue (KORE.pdf Sec 4.5):
  - Ansor ranks schedules by predicted throughput and measures only the top-k;
    the acquisition here is the same, with an explicit validity gate (a kernel
    that will not compile or will fail SNR has zero realizable speedup).
  - The replay-validation metrics (`benches_to_best`, `top_k_recall`,
    `rank_correlation`) quantify *measurement efficiency*: on a logged table of
    (move -> real outcome), how many fewer benches does the ranker need to reach
    the best kernel vs. random order? This is exactly how Ansor / Compiler-World-
    Models report cost-model quality. All three metrics are PURE (numpy only,
    no scipy).

P0 CONTRACT (imported by the GRPO rollout to bench only the top-k):

    rank_candidates(items, task=None) -> list[int]      # indices, best-first
    score_candidates(items, task=None) -> list[float]   # predicted utility

``items`` is a list of candidates, each either a kernel-source ``str`` or a dict
that may carry a ``source`` plus problem metadata; ``task`` (a :class:`Task` or a
plain dict, optional) supplies the shared problem context (operation, dtype,
shape) merged into every item. A candidate's schedule features are read from its
source, so the ranking is action/schedule-conditioned.

Robustness: when no fitted model is available (cold start / untrained) the
functions fall back to a cheap, deterministic source-heuristic ranking, so the
rollout can always call ``rank_candidates`` and get a sane best-first order. The
same fallback also catches a subtler failure -- a *usable* model that returns a
(near-)constant score for genuinely distinct candidates (e.g. one fit without
schedule features, so it ignores the kernel source): that is a no-op PUCT prior,
so we substitute the always-varied heuristic (see :func:`score_candidates`).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Sequence

import numpy as np

from kore.value.features import extract_schedule_features, featurize_many

_LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Default model wiring (so the rollout can call rank_candidates(items, task)
# with no model argument). Kept in-memory and explicit for test determinism.
# --------------------------------------------------------------------------- #
_DEFAULT_MODEL: Any = None


def set_default_model(model: Any) -> None:
    """Install the value model used by ``rank_candidates`` when none is passed."""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = model


def get_default_model() -> Any:
    """Return the currently installed default model (or None)."""
    return _DEFAULT_MODEL


def _model_is_serviceable(model: Any) -> bool:
    """Smoke-test a freshly-loaded model against the CURRENT featurizer.

    A pickle is only as good as the ``features`` layout it was fit under. An
    artifact trained before a feature block was added expects a narrower ``X`` and
    raises (e.g. sklearn's "X has 47 features, but ... is expecting 28") on the
    first real call - which would surface as a rollout crash, not a fallback. So we
    score a two-row probe here and refuse anything that cannot answer.
    """
    try:
        from kore.value.features import N_FEATURES

        probe = np.zeros((2, N_FEATURES), dtype=np.float64)
        out = np.asarray(_model_utility(model, probe), dtype=np.float64)
        return out.shape == (2,) and bool(np.all(np.isfinite(out)))
    except Exception:  # noqa: BLE001 - any failure means "not serviceable"
        return False


def load_default_model(path: Optional[str] = None) -> Any:
    """Load a saved :class:`ValueModel` from disk and install it as default.

    Returns the model, or None if the file is missing / unreadable / STALE (leaving
    the heuristic fallback in place). Safe to call at rollout start.

    A model is only installed once it answers a probe under the current feature
    layout (:func:`_model_is_serviceable`), so a stale artifact degrades to the
    heuristic instead of raising into the rollout on its first prediction.
    """
    from kore.value.model import ValueModel

    if path is None:
        try:
            from kore.config import CONFIG

            path = str(CONFIG.runs_dir / "value" / "value_model.pkl")
        except Exception:
            return None
    try:
        model = ValueModel.load(path)
    except Exception:
        return None
    if not _model_is_serviceable(model):
        return None
    set_default_model(model)
    return model


# --------------------------------------------------------------------------- #
# item / task -> meta plumbing
# --------------------------------------------------------------------------- #
def _task_meta(task: Any) -> dict:
    """Extract shared problem context from a Task (or dict) into a meta dict."""
    if task is None:
        return {}
    if isinstance(task, dict):
        return dict(task)
    meta: dict = {}
    for attr in ("operation", "dtype", "pmc_bottleneck"):
        v = getattr(task, attr, None)
        if v is not None:
            meta[attr] = v
    shapes = getattr(task, "shapes", None)
    if shapes:
        first = shapes[0]
        dims = getattr(first, "dims", None)
        if isinstance(dims, dict):
            meta.update(dims)
    return meta


def _item_meta(item: Any, task_meta: dict) -> dict:
    """Normalize one candidate + shared task context into a featurizer meta."""
    if isinstance(item, str):
        meta = {"source": item}
    elif isinstance(item, bytes):  # defensive: never silently drop a bytes source
        meta = {"source": item.decode("utf-8", "replace")}
    elif isinstance(item, dict):
        meta = dict(item)
    else:  # object exposing .source (e.g. a candidate record)
        meta = {"source": getattr(item, "source", "") or ""}
    for k, v in task_meta.items():
        meta.setdefault(k, v)
    return meta


def _is_usable_model(model: Any) -> bool:
    if model is None or not hasattr(model, "predict"):
        return False
    # ValueModel exposes `.fitted`; a hand-rolled stub without it is assumed ready.
    return bool(getattr(model, "fitted", True))


def _has_fitted_ranker(model: Any) -> bool:
    """True when ``model`` carries a FITTED pairwise ranking head.

    ``getattr`` (not attribute access) on purpose: a model pickled before the
    ranking head existed has no ``.ranker`` at all, and ``ValueModel.rank_scores``
    would raise ``AttributeError`` on it.
    """
    ranker = getattr(model, "ranker", None)
    return ranker is not None and bool(getattr(ranker, "fitted", False))


def _pointwise_utility(model: Any, X: np.ndarray) -> np.ndarray:
    """utility = p_compile * p_snr_pass * exp(e_log_speedup) from the 3 pointwise
    heads: E[speedup] gated by the probability the candidate is even valid."""
    pred = model.predict(X)
    p_c = np.asarray(pred["p_compile"], dtype=np.float64)
    p_s = np.asarray(pred["p_snr_pass"], dtype=np.float64)
    e_ls = np.asarray(pred["e_log_speedup"], dtype=np.float64)
    # clip exponent so a wild regressor output cannot produce inf utility
    return p_c * p_s * np.exp(np.clip(e_ls, -50.0, 50.0))


def _model_utility(model: Any, X: np.ndarray) -> np.ndarray:
    """Ordering score for the candidates in ``X`` (higher is better).

    Prefers the model's WITHIN-GROUP pairwise ranking head when one is fitted, and
    otherwise the pointwise ``p_compile * p_snr_pass * exp(e_log_speedup)`` utility.
    That is exactly the contract of :meth:`kore.value.model.ValueModel.rank_scores`,
    which we route through when it is available.

    Why prefer the ranker: ``train_value.train_ranking`` (the trainer the campaign
    runs via :func:`kore.value.replay_train.train_value_from_groups`) ALWAYS fits
    the pairwise head on within-group order - which is the signal the top-k bench
    selector and the PUCT prior actually consume - whereas the pointwise regressor
    is fit to an absolute E[log speedup] target. Reading only the pointwise heads
    trained the ranker on every run and then threw it away.

    ``X`` is the ``featurize_many`` matrix for the candidates (passed in so the
    caller can reuse it for the degeneracy/distinctness check without re-featurizing).
    """
    if _has_fitted_ranker(model):
        rank_scores = getattr(model, "rank_scores", None)
        if callable(rank_scores):
            return np.asarray(rank_scores(X), dtype=np.float64)
    return _pointwise_utility(model, X)


def _is_degenerate(arr: np.ndarray) -> bool:
    """True when scores are (near-)constant -- i.e. a uniform, non-guiding prior.

    Uses a scale-relative spread so it fires on both exact ties (spread == 0) and
    numerically-negligible spreads (which softmax would render uniform anyway),
    without ever flagging a model that meaningfully discriminates candidates.
    """
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size < 2:
        return False
    spread = float(np.nanmax(a) - np.nanmin(a))
    scale = float(np.nanmax(np.abs(a)))
    return spread <= 1e-9 * (scale + 1.0)


def _has_distinct_feature_rows(X: np.ndarray) -> bool:
    """True when >=2 candidates differ in their MODEL features (genuinely distinct).

    A degenerate model score over distinct feature rows means the model is ignoring
    what makes the candidates different (the no-op prior); over identical rows a
    constant score is correct, so we must NOT override it with the heuristic."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 2:
        return False
    return np.unique(X, axis=0).shape[0] > 1


def _heuristic_scores(metas: Sequence[dict]) -> np.ndarray:
    """Cheap source-only goodness score for the untrained cold-start path.

    Rewards the gfx942 discipline the cost model would eventually learn: MFMA via
    tl.dot, an fp32 accumulator, 64-multiple (and power-of-2) tiles, num_warps in
    {4, 8}, software pipelining, bounds masking, and a K reduction loop. A move
    with no source scores 0 (nothing to prefer).

    A tiny, bounded structural tie-breaker (source length + load/store/loop counts,
    with a small atomic penalty) is added on top so that genuinely DISTINCT sources
    that happen to share the same discipline flags still get DISTINCT scores -- the
    PUCT prior must never collapse to uniform among schedule-equivalent siblings.
    The tie-breaker is capped well below the smallest discipline increment (0.25),
    so it only ever separates otherwise-equal candidates; it never reorders tiers
    (faster / better-formed kernels still score strictly higher)."""
    out: list[float] = []
    for m in metas:
        s = extract_schedule_features(m.get("source", "") or "")
        if not s["has_source"]:
            out.append(0.0)  # truly uninformative -> 0 (no source to prefer)
            continue
        v = 0.0
        v += 1.0 if s["has_tl_dot"] else 0.0
        v += 1.0 if s["has_fp32_acc"] else 0.0
        v += 0.5 if s["has_mask"] else 0.0
        v += 0.5 if s["has_reduction_loop"] else 0.0
        v += 1.0 * float(s["blocks_mult64"])
        v += 0.5 * float(s["blocks_pow2"])
        if s["num_warps"] in (4, 8):
            v += 0.5
        if s["num_stages"] and s["num_stages"] >= 2:
            v += 0.25
        if s["tile_area"]:
            v += 0.1 * math.log1p(s["tile_area"])
        # --- bounded structural tie-breaker (< ~0.05 total; << 0.25) ---
        v += 1e-3 * math.log1p(s.get("source_len", 0))
        v += 2e-3 * math.log1p(s["n_loads"])
        v += 2e-3 * math.log1p(s["n_stores"])
        v += 2e-3 * math.log1p(s["n_loops"])
        if s.get("has_atomic"):
            v -= 8e-3  # atomics usually serialize -> mildly deprioritized
        out.append(v)
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------- #
# P0 contract: score / rank
# --------------------------------------------------------------------------- #
def score_candidates(items: Sequence[Any], task: Any = None, model: Any = None) -> list[float]:
    """Predicted utility per candidate (higher is better).

    ``items`` may be raw kernel-source ``str`` (the AlphaKernel PUCT-prior call
    site passes candidate sources), ``bytes``, a ``dict`` carrying a ``source`` +
    problem metadata, or any object exposing a ``.source`` attribute; ``task``
    supplies the shared problem context merged into every item.

    utility = p_compile * p_snr_pass * exp(e_log_speedup) under a fitted value
    model, else a source-heuristic score (see :func:`_heuristic_scores`).

    NO-OP-PRIOR GUARD: a *usable* model that returns a (near-)constant utility for
    a set of genuinely DISTINCT candidates is uninformative for PUCT -- e.g. a model
    fit without schedule/source features learns ~zero weight on them and therefore
    IGNORES the kernel source, collapsing to a uniform prior. In that case we defer
    to the always-varied source heuristic so the prior still guides the search. A
    model that meaningfully discriminates the candidates is used as-is (unchanged
    behavior); the heuristic never overrides it.

    STALE-MODEL GUARD: a model whose feature layout no longer matches the current
    featurizer RAISES on predict rather than returning something wrong. Ranking a
    candidate set is never worth failing a rollout over, so a scoring failure also
    falls back to the heuristic (loudly, via ``log``) instead of propagating."""
    task_meta = _task_meta(task)
    metas = [_item_meta(it, task_meta) for it in items]
    if not metas:
        return []
    if model is None:
        model = get_default_model()
    arr: Any = None
    if _is_usable_model(model):
        X = featurize_many(metas)
        try:
            arr = np.nan_to_num(np.asarray(_model_utility(model, X), dtype=np.float64),
                                nan=0.0, posinf=1e30, neginf=0.0)
        except Exception as e:  # noqa: BLE001 - a stale/broken model must not fail the rollout
            _LOG.warning("value model failed to score %d candidates (%s: %s); "
                         "falling back to the source heuristic",
                         len(metas), type(e).__name__, e)
            arr = None
        if arr is not None and _is_degenerate(arr) and _has_distinct_feature_rows(X):
            arr = _heuristic_scores(metas)
    if arr is None:
        arr = _heuristic_scores(metas)
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0,
                        posinf=1e30, neginf=0.0)
    return [float(x) for x in arr]


def rank_candidates(items: Sequence[Any], task: Any = None, model: Any = None) -> list[int]:
    """Return candidate indices ordered best-first by predicted utility.

    Robust to a missing / untrained model (heuristic fallback). Stable sort keeps
    input order on ties, so a degenerate all-equal score yields identity order."""
    scores = np.asarray(score_candidates(items, task=task, model=model), dtype=np.float64)
    if scores.size == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order]


# --------------------------------------------------------------------------- #
# Replay-validation metrics (PURE)
# --------------------------------------------------------------------------- #
def benches_to_best(pred_scores: Sequence[float], true_outcomes: Sequence[float]) -> int:
    """#benches (candidates ranked by pred, high-first) needed to reach true best.

    A perfect ranker returns 1 (the true best is ranked first); the worst
    possible ranker returns N (the true best is ranked last).
    """
    pred = np.asarray(pred_scores, dtype=np.float64).ravel()
    true = np.asarray(true_outcomes, dtype=np.float64).ravel()
    n = pred.shape[0]
    if n == 0:
        return 0
    best_idx = int(np.argmax(true))
    order = np.argsort(-pred, kind="stable")
    # position (1-indexed) of the true-best candidate in the predicted order
    pos = int(np.where(order == best_idx)[0][0]) + 1
    return pos


def top_k_recall(pred_scores: Sequence[float], true_outcomes: Sequence[float], k: int) -> float:
    """Fraction of the true top-k candidates recovered by the predicted top-k."""
    pred = np.asarray(pred_scores, dtype=np.float64).ravel()
    true = np.asarray(true_outcomes, dtype=np.float64).ravel()
    n = pred.shape[0]
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)
    pred_top = set(int(i) for i in np.argsort(-pred, kind="stable")[:k])
    true_top = set(int(i) for i in np.argsort(-true, kind="stable")[:k])
    return len(pred_top & true_top) / float(k)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-indexed), handling ties -- like scipy.stats.rankdata."""
    a = np.asarray(a, dtype=np.float64).ravel()
    n = a.shape[0]
    order = np.argsort(a, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # resolve ties by averaging ranks within each group of equal values
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0  # 1-indexed positions i..j
            for idx in order[i : j + 1]:
                ranks[idx] = avg
        i = j + 1
    return ranks


def rank_correlation(pred_scores: Sequence[float], true_outcomes: Sequence[float]) -> float:
    """Spearman rank correlation (no scipy). Returns 0.0 when undefined."""
    pred = np.asarray(pred_scores, dtype=np.float64).ravel()
    true = np.asarray(true_outcomes, dtype=np.float64).ravel()
    n = pred.shape[0]
    if n < 2:
        return 0.0
    rp = _rankdata(pred)
    rt = _rankdata(true)
    rp -= rp.mean()
    rt -= rt.mean()
    denom = np.sqrt((rp * rp).sum() * (rt * rt).sum())
    if denom < 1e-12:
        return 0.0
    return float((rp * rt).sum() / denom)
