"""Paired statistics for "KORE vs baseline/Opus" kernel comparisons.

A publishable claim that KORE beats a baseline (the seed policy) or the Opus
teacher must be STATISTICALLY sound, not a single-number anecdote. Because both
sides are scored on the SAME held-out tasks under a matched budget (see
:mod:`kore.eval.bakeoff` / :mod:`kore.eval.vs_opus`), the comparison is PAIRED:
per task we have KORE's speedup and the other side's speedup, and we care about the
per-task DELTA. Paired tests are far more powerful than unpaired ones here because
they cancel the huge task-to-task difficulty variance.

This module reports, for a set of paired per-task deltas (or two paired speedup
arrays), the three things a reviewer wants:

  * EFFECT SIZE   - the mean per-task delta, or (for speedups) the geometric-mean
    speedup RATIO ``exp(mean(log(kore/base)))``;
  * 95% CI        - a paired BOOTSTRAP confidence interval on that effect size
    (:func:`paired_bootstrap`), plus a bootstrap two-sided p-value;
  * P-VALUE       - a non-parametric paired significance test: the exact two-sided
    SIGN test (:func:`sign_test`) and the WILCOXON signed-rank test
    (:func:`wilcoxon_signed_rank`, normal approx with tie + continuity correction).

Everything is PURE numpy/python (no scipy, no torch) so it is trivially unit-tested
and has no heavy import cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Small pure helpers (normal CDF, average ranks, exact binomial tail).
# --------------------------------------------------------------------------- #
def _normal_cdf(z: float) -> float:
    """Standard-normal CDF via the error function (no scipy)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average (fractional) ranks, 1-based, ties share their mean rank."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _binom_sf_le(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k) for X ~ Binomial(n, p), exact (Python big-int combinatorics)."""
    if n <= 0:
        return 1.0
    k = max(0, min(k, n))
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return total


# --------------------------------------------------------------------------- #
# Paired bootstrap confidence interval + bootstrap p-value.
# --------------------------------------------------------------------------- #
@dataclass
class BootstrapResult:
    effect_size: float
    ci_lo: float
    ci_hi: float
    ci_level: float
    se: float
    p_value: float
    n: int
    n_boot: int
    statistic: str

    def to_dict(self) -> dict:
        return {"effect_size": self.effect_size, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
                "ci_level": self.ci_level, "se": self.se, "p_value": self.p_value,
                "n": self.n, "n_boot": self.n_boot, "statistic": self.statistic}

    @property
    def excludes_zero(self) -> bool:
        return self.ci_lo > 0.0 or self.ci_hi < 0.0


def paired_bootstrap(deltas: Sequence[float], *, n_boot: int = 10000, ci_level: float = 0.95,
                     seed: int = 0, statistic: str = "mean") -> BootstrapResult:
    """Bootstrap CI + two-sided p-value for the paired-delta effect size.

    Resamples the per-task deltas WITH REPLACEMENT ``n_boot`` times, recomputing the
    ``statistic`` ("mean" or "median") each time. The 95% CI is the percentile
    interval of that bootstrap distribution; the two-sided p-value is a bootstrap
    hypothesis test against ``H0: statistic == 0`` (the null distribution is the
    bootstrap distribution recentred to zero, and p is the fraction at least as
    extreme as the observed effect, with add-one smoothing).
    """
    d = np.asarray(deltas, dtype=float)
    n = d.size
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, ci_level, 0.0, 1.0, 0, n_boot, statistic)

    stat_fn = np.mean if statistic == "mean" else np.median
    obs = float(stat_fn(d))
    if n == 1:
        return BootstrapResult(obs, obs, obs, ci_level, 0.0, 1.0, 1, n_boot, statistic)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = stat_fn(d[idx], axis=1)

    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    se = float(np.std(boot, ddof=1))

    # Two-sided bootstrap p-value: recenter the bootstrap distribution to the null
    # (statistic == 0) and count draws at least as far from 0 as the observed effect.
    null = boot - obs
    p = (np.sum(np.abs(null) >= abs(obs)) + 1.0) / (n_boot + 1.0)
    return BootstrapResult(obs, lo, hi, ci_level, se, float(p), n, n_boot, statistic)


# --------------------------------------------------------------------------- #
# Non-parametric paired significance tests.
# --------------------------------------------------------------------------- #
@dataclass
class SignTestResult:
    n_pos: int
    n_neg: int
    n_effective: int
    p_value: float
    prop_positive: float

    def to_dict(self) -> dict:
        return {"n_pos": self.n_pos, "n_neg": self.n_neg, "n_effective": self.n_effective,
                "p_value": self.p_value, "prop_positive": self.prop_positive}


def sign_test(deltas: Sequence[float], *, zero_tol: float = 0.0) -> SignTestResult:
    """Exact two-sided SIGN test on paired deltas (binomial, p=0.5).

    Counts positive vs negative deltas (ties within ``zero_tol`` are dropped, the
    standard treatment), and returns the exact two-sided p-value
    ``min(1, 2 * P(X <= min(n_pos, n_neg)))`` under ``X ~ Binomial(n_eff, 0.5)``.
    Distribution-free: it only uses the SIGN of each per-task delta, so a couple of
    huge outliers cannot manufacture significance.
    """
    d = np.asarray(deltas, dtype=float)
    n_pos = int(np.sum(d > zero_tol))
    n_neg = int(np.sum(d < -zero_tol))
    n_eff = n_pos + n_neg
    if n_eff == 0:
        return SignTestResult(n_pos, n_neg, 0, 1.0, 0.0)
    k = min(n_pos, n_neg)
    p = min(1.0, 2.0 * _binom_sf_le(k, n_eff, 0.5))
    return SignTestResult(n_pos, n_neg, n_eff, float(p), n_pos / n_eff)


@dataclass
class WilcoxonResult:
    statistic: float          # W+ (sum of ranks of positive deltas)
    z: float
    p_value: float
    n_effective: int

    def to_dict(self) -> dict:
        return {"statistic": self.statistic, "z": self.z,
                "p_value": self.p_value, "n_effective": self.n_effective}


def wilcoxon_signed_rank(deltas: Sequence[float], *, correction: bool = True) -> WilcoxonResult:
    """Wilcoxon signed-rank test (normal approximation, tie + continuity corrected).

    Ranks the absolute deltas (zeros dropped, ``wilcox`` method; ties get average
    ranks), forms ``W+`` = sum of ranks with a positive delta, and standardizes with
    the tie-corrected variance. Uses W+ (not min(W+, W-)) so the resulting z keeps a
    SIGN, and reports a two-sided p-value. Pure numpy; matches
    ``scipy.stats.wilcoxon(..., mode='approx')`` closely for n >= ~10.
    """
    d = np.asarray(deltas, dtype=float)
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return WilcoxonResult(0.0, 0.0, 1.0, 0)

    ranks = _rankdata(np.abs(d))
    w_plus = float(np.sum(ranks[d > 0.0]))
    mean_w = n * (n + 1) / 4.0

    # Tie correction to the variance: subtract sum(t^3 - t)/48 over tie groups.
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts)) / 48.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term
    if var_w <= 0.0:
        return WilcoxonResult(w_plus, 0.0, 1.0, n)

    diff = w_plus - mean_w
    cc = 0.5 if correction else 0.0
    # Continuity correction shrinks |diff| toward 0 by 0.5.
    z = (diff - math.copysign(cc, diff)) / math.sqrt(var_w) if diff != 0 else 0.0
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return WilcoxonResult(w_plus, float(z), float(min(1.0, p)), n)


# --------------------------------------------------------------------------- #
# Consolidated paired comparison (effect size + 95% CI + p-value).
# --------------------------------------------------------------------------- #
@dataclass
class PairedComparison:
    n: int
    effect_size: float
    effect_kind: str
    ci: tuple
    ci_level: float
    se: float
    p_bootstrap: float
    p_sign: float
    p_wilcoxon: float
    p_value: float           # the headline p-value (Wilcoxon by default)
    significant: bool
    direction: str
    bootstrap: dict = field(default_factory=dict)
    sign: dict = field(default_factory=dict)
    wilcoxon: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n": self.n, "effect_size": self.effect_size, "effect_kind": self.effect_kind,
            "ci": list(self.ci), "ci_level": self.ci_level, "se": self.se,
            "p_bootstrap": self.p_bootstrap, "p_sign": self.p_sign,
            "p_wilcoxon": self.p_wilcoxon, "p_value": self.p_value,
            "significant": self.significant, "direction": self.direction,
            "bootstrap": self.bootstrap, "sign": self.sign, "wilcoxon": self.wilcoxon,
        }


def _direction(effect: float, tol: float = 0.0) -> str:
    if effect > tol:
        return "kore_better"
    if effect < -tol:
        return "baseline_better"
    return "tie"


def paired_comparison(kore: Optional[Sequence[float]] = None,
                      baseline: Optional[Sequence[float]] = None, *,
                      deltas: Optional[Sequence[float]] = None,
                      n_boot: int = 10000, ci_level: float = 0.95, seed: int = 0,
                      alpha: float = 0.05, headline: str = "wilcoxon") -> PairedComparison:
    """Full paired comparison: effect size + 95% CI + p-value(s).

    Provide either the per-task ``deltas`` directly, or paired ``kore`` / ``baseline``
    arrays (``deltas = kore - baseline``). Returns the mean-delta effect size with a
    bootstrap 95% CI and three p-values (bootstrap / sign / Wilcoxon); ``significant``
    is decided by the ``headline`` test (default Wilcoxon) at level ``alpha``. Use
    :func:`paired_speedup_comparison` when the paired quantities are SPEEDUPS and the
    natural effect size is a geometric-mean ratio.
    """
    if deltas is None:
        if kore is None or baseline is None:
            raise ValueError("provide deltas, or both kore and baseline arrays")
        k = np.asarray(kore, dtype=float)
        b = np.asarray(baseline, dtype=float)
        if k.shape != b.shape:
            raise ValueError(f"paired arrays must match: {k.shape} vs {b.shape}")
        d = k - b
    else:
        d = np.asarray(deltas, dtype=float)

    boot = paired_bootstrap(d, n_boot=n_boot, ci_level=ci_level, seed=seed)
    sgn = sign_test(d)
    wil = wilcoxon_signed_rank(d)

    p_headline = {"wilcoxon": wil.p_value, "sign": sgn.p_value,
                  "bootstrap": boot.p_value}.get(headline, wil.p_value)
    return PairedComparison(
        n=int(d.size),
        effect_size=boot.effect_size,
        effect_kind="mean_delta",
        ci=(boot.ci_lo, boot.ci_hi),
        ci_level=ci_level,
        se=boot.se,
        p_bootstrap=boot.p_value,
        p_sign=sgn.p_value,
        p_wilcoxon=wil.p_value,
        p_value=p_headline,
        significant=(p_headline < alpha),
        direction=_direction(boot.effect_size),
        bootstrap=boot.to_dict(), sign=sgn.to_dict(), wilcoxon=wil.to_dict(),
    )


def paired_speedup_comparison(kore_speedups: Sequence[float], baseline_speedups: Sequence[float],
                              *, n_boot: int = 10000, ci_level: float = 0.95, seed: int = 0,
                              alpha: float = 0.05, headline: str = "wilcoxon") -> PairedComparison:
    """Paired comparison of SPEEDUPS with a geometric-mean-RATIO effect size.

    Works on the log domain (``delta = log(kore) - log(baseline)``), so the effect
    size is the geometric-mean speedup ratio ``exp(mean(delta))`` and the CI is the
    exponentiated bootstrap interval - the multiplicative "KORE is X times faster
    than the baseline" statement, with a 95% CI and a Wilcoxon/sign p-value on the
    per-task log-ratios. Both inputs must be strictly positive (a side only competes
    with a correct, timed kernel; drop non-competing tasks before calling).
    """
    k = np.asarray(kore_speedups, dtype=float)
    b = np.asarray(baseline_speedups, dtype=float)
    if k.shape != b.shape:
        raise ValueError(f"paired arrays must match: {k.shape} vs {b.shape}")
    if np.any(k <= 0) or np.any(b <= 0):
        raise ValueError("speedups must be strictly positive for the log-ratio effect size")

    log_delta = np.log(k) - np.log(b)
    boot = paired_bootstrap(log_delta, n_boot=n_boot, ci_level=ci_level, seed=seed)
    sgn = sign_test(log_delta)
    wil = wilcoxon_signed_rank(log_delta)

    ratio_effect = math.exp(boot.effect_size)
    ratio_ci = (math.exp(boot.ci_lo), math.exp(boot.ci_hi))
    p_headline = {"wilcoxon": wil.p_value, "sign": sgn.p_value,
                  "bootstrap": boot.p_value}.get(headline, wil.p_value)
    return PairedComparison(
        n=int(k.size),
        effect_size=ratio_effect,
        effect_kind="geomean_speedup_ratio",
        ci=ratio_ci,
        ci_level=ci_level,
        se=boot.se,
        p_bootstrap=boot.p_value,
        p_sign=sgn.p_value,
        p_wilcoxon=wil.p_value,
        p_value=p_headline,
        # For a ratio, "no effect" is 1.0, so significance is the ratio-CI excluding 1.
        significant=(ratio_ci[0] > 1.0 or ratio_ci[1] < 1.0),
        direction=_direction(boot.effect_size),
        bootstrap=boot.to_dict(), sign=sgn.to_dict(), wilcoxon=wil.to_dict(),
    )


def format_paired_report(cmp: PairedComparison, *, name_a: str = "KORE",
                         name_b: str = "baseline") -> str:
    """Compact ASCII markdown for a :class:`PairedComparison`."""
    ratio = cmp.effect_kind == "geomean_speedup_ratio"
    eff = f"{cmp.effect_size:.4f}" + ("x" if ratio else "")
    null = "1.0" if ratio else "0.0"
    lines = [
        f"# paired comparison: {name_a} vs {name_b}",
        "",
        f"- n (paired tasks): {cmp.n}",
        f"- effect size ({cmp.effect_kind}): {eff}",
        f"- {int(cmp.ci_level * 100)}% CI: [{cmp.ci[0]:.4f}, {cmp.ci[1]:.4f}] (null={null})",
        f"- p-value (headline): {cmp.p_value:.4g}",
        f"  - bootstrap: {cmp.p_bootstrap:.4g}",
        f"  - sign test: {cmp.p_sign:.4g}",
        f"  - wilcoxon:  {cmp.p_wilcoxon:.4g}",
        f"- direction: {cmp.direction}",
        f"- significant (alpha=0.05): {cmp.significant}",
    ]
    return "\n".join(lines)


__all__ = [
    "BootstrapResult",
    "SignTestResult",
    "WilcoxonResult",
    "PairedComparison",
    "paired_bootstrap",
    "sign_test",
    "wilcoxon_signed_rank",
    "paired_comparison",
    "paired_speedup_comparison",
    "format_paired_report",
]
