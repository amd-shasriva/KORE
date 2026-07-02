"""KORE value model: a cheap 3-head surrogate of an expensive GPU benchmark.

Given the static features of a candidate kernel move (see `features.py`), predict
BEFORE benching:
  - p_compile   : P(the edit compiles),
  - p_snr_pass  : P(it passes the SNR correctness gate | it compiles-ish),
  - e_log_speedup : E[log speedup] (throughput-weighted regression).

Analogue (KORE.pdf Sec 4.5):
  - This is the Ansor cost-model role: a learned predictor of a measurement,
    trained on past (move -> outcome) pairs, used to rank candidates so only the
    top-k are actually measured. Ansor reports several-fold fewer real
    measurements at equal quality; Compiler-World-Models make the same point for
    compiler autotuning.
  - Throughput weighting (sample_weight ~ realized speedup) focuses the regressor
    where it matters: being accurate about the *fast* kernels at the top of the
    ranking, not the long tail of duds.

sklearn (HistGradientBoosting) is used when importable; otherwise a pure-numpy
logistic / ridge fallback keeps everything working with no extra dependency.
"""

from __future__ import annotations

import pickle
from typing import Optional

import numpy as np

try:  # sklearn is optional; import-guarded.
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - exercised only when sklearn absent
    _HAS_SKLEARN = False


# --------------------------------------------------------------------------- #
# Pure-numpy fallback estimators (no sklearn required)
# --------------------------------------------------------------------------- #
def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


class _NumpyLogistic:
    """L2-regularized logistic regression via gradient descent (class-1 prob)."""

    def __init__(self, lr: float = 0.5, n_iter: int = 400, l2: float = 1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.constant: Optional[float] = None  # set when only one class present

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        classes = np.unique(y)
        if classes.size < 2:
            # Degenerate: constant probability (smoothed away from 0/1).
            p = float(classes[0]) if classes.size == 1 else 0.5
            self.constant = min(max(p, 1e-3), 1 - 1e-3)
            return self
        self.mu, self.sd = _standardize_fit(X)
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        if sample_weight is None:
            w_s = np.ones(n)
        else:
            w_s = np.asarray(sample_weight, dtype=np.float64).ravel()
        w_s = w_s / (w_s.mean() + 1e-12)
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            z = Xs @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            err = (p - y) * w_s
            grad_w = Xs.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba1(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.constant is not None:
            return np.full(X.shape[0], self.constant)
        Xs = (X - self.mu) / self.sd
        z = Xs @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class _NumpyRidge:
    """Weighted ridge regression (closed form) on standardized features."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.mu, self.sd = _standardize_fit(X)
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        if sample_weight is None:
            w_s = np.ones(n)
        else:
            w_s = np.asarray(sample_weight, dtype=np.float64).ravel()
        w_s = np.clip(w_s, 1e-8, None)
        # augment with intercept column
        A = np.hstack([Xs, np.ones((n, 1))])
        W = A * w_s[:, None]
        reg = self.l2 * np.eye(d + 1)
        reg[-1, -1] = 0.0  # do not regularize intercept
        coef = np.linalg.solve(A.T @ W + reg, A.T @ (w_s * y))
        self.w = coef[:-1]
        self.b = float(coef[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self.mu) / self.sd
        return Xs @ self.w + self.b


# --------------------------------------------------------------------------- #
# sklearn wrappers that tolerate single-class targets
# --------------------------------------------------------------------------- #
class _SklearnClassifier:
    def __init__(self):
        self.est = None
        self.constant: Optional[float] = None

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y).ravel()
        classes = np.unique(y)
        if classes.size < 2:
            p = float(classes[0]) if classes.size == 1 else 0.5
            self.constant = min(max(p, 1e-3), 1 - 1e-3)
            return self
        self.est = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_depth=3, l2_regularization=1.0
        )
        self.est.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba1(self, X):
        X = np.asarray(X)
        if self.constant is not None:
            return np.full(X.shape[0], self.constant)
        proba = self.est.predict_proba(X)
        # index of the positive (1) class
        pos = list(self.est.classes_).index(1) if 1 in self.est.classes_ else proba.shape[1] - 1
        return proba[:, pos]


class _SklearnRegressor:
    def __init__(self):
        self.est = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.08, max_depth=3, l2_regularization=1.0
        )

    def fit(self, X, y, sample_weight=None):
        self.est.fit(X, np.asarray(y).ravel(), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(X)


# --------------------------------------------------------------------------- #
# The three-head value model
# --------------------------------------------------------------------------- #
class ValueModel:
    """Three-head surrogate: p_compile, p_snr_pass, e_log_speedup.

    Set `use_sklearn=False` to force the pure-numpy fallback (used in tests to
    guarantee the sklearn-optional path works).
    """

    def __init__(self, use_sklearn: Optional[bool] = None):
        if use_sklearn is None:
            use_sklearn = _HAS_SKLEARN
        self.use_sklearn = bool(use_sklearn and _HAS_SKLEARN)
        self.backend = "sklearn" if self.use_sklearn else "numpy"
        if self.use_sklearn:
            self.clf_compile = _SklearnClassifier()
            self.clf_snr = _SklearnClassifier()
            self.reg_speedup = _SklearnRegressor()
        else:
            self.clf_compile = _NumpyLogistic()
            self.clf_snr = _NumpyLogistic()
            self.reg_speedup = _NumpyRidge()
        self.fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_compile,
        y_snr,
        y_logspeedup,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ValueModel":
        X = np.asarray(X, dtype=np.float64)
        y_compile = np.asarray(y_compile).ravel().astype(int)
        y_snr = np.asarray(y_snr).ravel().astype(int)
        y_logspeedup = np.asarray(y_logspeedup, dtype=np.float64).ravel()

        # Classifiers may use sample_weight but do not require it.
        self.clf_compile.fit(X, y_compile, sample_weight=sample_weight)
        self.clf_snr.fit(X, y_snr, sample_weight=sample_weight)
        # Regressor is throughput-weighted (accuracy where speedup is large).
        self.reg_speedup.fit(X, y_logspeedup, sample_weight=sample_weight)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> dict:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        p_compile = np.clip(self.clf_compile.predict_proba1(X), 0.0, 1.0)
        p_snr = np.clip(self.clf_snr.predict_proba1(X), 0.0, 1.0)
        e_log = np.asarray(self.reg_speedup.predict(X), dtype=np.float64)
        e_log = np.nan_to_num(e_log, nan=0.0, posinf=50.0, neginf=-50.0)
        return {
            "p_compile": p_compile,
            "p_snr_pass": p_snr,
            "e_log_speedup": e_log,
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ValueModel":
        with open(path, "rb") as f:
            return pickle.load(f)
