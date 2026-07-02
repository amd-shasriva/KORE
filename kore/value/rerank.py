"""KORE candidate reranking + replay-validation metrics.

The value model turns each candidate move into three predictions; this module
collapses them into a single scalar *utility* and ranks candidates so the
scheduler benches only the most promising ones first.

    utility = p_compile * p_snr_pass * exp(e_log_speedup)
            = E[speedup] gated by the probability the candidate is even valid.

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
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from kore.value.features import featurize_many


def score_candidates(model, metas: Sequence[dict]) -> np.ndarray:
    """Scalar utility per candidate: p_compile * p_snr_pass * exp(e_log_speedup)."""
    X = featurize_many(metas)
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    pred = model.predict(X)
    p_c = np.asarray(pred["p_compile"], dtype=np.float64)
    p_s = np.asarray(pred["p_snr_pass"], dtype=np.float64)
    e_ls = np.asarray(pred["e_log_speedup"], dtype=np.float64)
    # clip exponent so a wild regressor output cannot produce inf utility
    return p_c * p_s * np.exp(np.clip(e_ls, -50.0, 50.0))


def rank_candidates(model, metas: Sequence[dict]) -> list[int]:
    """Return candidate indices ordered best-first by utility."""
    scores = score_candidates(model, metas)
    # negate for descending; stable sort keeps input order on ties
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
