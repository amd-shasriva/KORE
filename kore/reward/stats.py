"""Benchmark statistics: median, mean, std, coefficient of variation."""

from __future__ import annotations

import math


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty list")
    mid = n // 2
    return s[mid] if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def cv_pct(xs: list[float]) -> float:
    mu = mean(xs)
    if mu == 0 or math.isnan(mu):
        return float("inf")
    return 100.0 * std(xs) / abs(mu)


def paired_timing_stats(candidate_ms: list[float], baseline_ms: list[float],
                        noise_floor_pct: float = 2.0,
                        z: float = 1.96) -> dict:
    """Statistics for equal-length paired candidate/reference measurements.

    Each input element is one repeat-level median measured back-to-back on fresh,
    value-identical storage.  The paired effect is represented in log space:
    ``log(reference/candidate)``.  Its exponential is the geometric-mean
    speedup, and the normal-approximation CI is used only as a conservative
    admission/classification gate (the raw samples remain in ``Observation``).
    """
    cand = [float(x) for x in candidate_ms]
    base = [float(x) for x in baseline_ms]
    if len(cand) != len(base):
        raise ValueError(
            f"paired timing count mismatch: candidate={len(cand)} baseline={len(base)}")
    if not cand:
        raise ValueError("paired timing requires at least one sample")
    if not all(math.isfinite(x) and x > 0.0 for x in cand + base):
        raise ValueError("paired timing samples must be finite and positive")

    ratios = [b / c for c, b in zip(cand, base)]
    logs = [math.log(r) for r in ratios]
    mu = mean(logs)
    log_std = std(logs)
    half = z * log_std / math.sqrt(len(logs)) if len(logs) >= 2 else float("inf")
    ci_lo, ci_hi = mu - half, mu + half
    # Multiplicative uncertainty around the geometric-mean ratio.
    ci_half_width_pct = (100.0 * math.expm1(half)
                         if math.isfinite(half) else float("inf"))
    noise_log = math.log1p(max(0.0, float(noise_floor_pct)) / 100.0)
    if ci_lo > noise_log:
        classification = "faster"
    elif ci_hi < -noise_log:
        classification = "slower"
    else:
        classification = "tie"
    return {
        "candidate_cv_pct": cv_pct(cand),
        "baseline_cv_pct": cv_pct(base),
        "paired_ratio_cv_pct": cv_pct(ratios),
        "paired_ratios": ratios,
        "paired_log_speedups": logs,
        "log_mean": mu,
        "log_ci_lo": ci_lo,
        "log_ci_hi": ci_hi,
        "ci_half_width_pct": ci_half_width_pct,
        "geomean_speedup": math.exp(mu),
        "classification": classification,
        "n": len(cand),
    }


def publication_admission_error(stats: dict, *, min_pairs: int,
                                candidate_cv_threshold_pct: float,
                                baseline_cv_threshold_pct: float,
                                paired_ratio_cv_threshold_pct: float,
                                paired_ci_threshold_pct: float) -> str | None:
    """Return the first vendor-grade timing admission failure, else ``None``."""
    if int(stats.get("n", 0)) < int(min_pairs):
        return f"paired sample count {stats.get('n', 0)} < required {min_pairs}"
    gates = (
        ("candidate CV", stats.get("candidate_cv_pct"), candidate_cv_threshold_pct),
        ("baseline CV", stats.get("baseline_cv_pct"), baseline_cv_threshold_pct),
        ("paired ratio CV", stats.get("paired_ratio_cv_pct"),
         paired_ratio_cv_threshold_pct),
        ("paired CI half-width", stats.get("ci_half_width_pct"),
         paired_ci_threshold_pct),
    )
    for label, value, limit in gates:
        try:
            v, lim = float(value), float(limit)
        except (TypeError, ValueError):
            return f"{label} is missing or invalid"
        if not math.isfinite(v):
            return f"{label} is not finite"
        if v > lim:
            return f"{label} {v:.3f}% exceeds {lim:.3f}%"
    return None
