"""fast_p metric (KernelBench), reproduced exactly.

KORE.pdf Sec 4.7 primary metric. ``fast_p`` is the fraction of the WHOLE split
for which a candidate is both correct AND faster than the baseline by more than
a factor ``p``:

    fast_p = (1/n) * count_i( correct_i AND baseline_i/actual_i > p )

The "speed" arrays are measured TIMES (lower is better), so the per-sample
speedup is ``baseline_i / actual_i``. ``n`` is the size of the split and is the
denominator: it is an *uncorrected* count, so tasks that error out / are not
attempted still count against the score (they simply contribute 0).

This module is PURE (no GPU, no I/O) and directly testable.
"""

from __future__ import annotations

import math
from typing import Sequence

# The p-grid KORE reports on. p in {1, 1.5} are the headline numbers; the rest
# trace the fast_p curve.
DEFAULT_PS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)


def _speedup(baseline_time: float, actual_time: float) -> float:
    """Speedup = baseline/actual for TIMES (lower time is better).

    Returns 0.0 for a non-positive / non-finite candidate time so that a
    degenerate measurement never counts as a win.
    """
    if actual_time is None or baseline_time is None:
        return 0.0
    if actual_time <= 0 or not math.isfinite(actual_time):
        return 0.0
    if baseline_time <= 0 or not math.isfinite(baseline_time):
        return 0.0
    return baseline_time / actual_time


def fastp(
    is_correct: Sequence[bool],
    baseline_speed: Sequence[float],
    actual_speed: Sequence[float],
    n: int,
    p: float,
) -> float:
    """fast_p = (1/n) * count(correct_i AND baseline_i/actual_i > p).

    ``baseline_speed`` and ``actual_speed`` are TIMES (lower is better); the
    per-sample speedup is ``baseline_i / actual_i``. ``n`` is the denominator
    (the full split size, uncorrected).
    """
    if n <= 0:
        return 0.0
    count = 0
    m = min(len(is_correct), len(baseline_speed), len(actual_speed))
    for i in range(m):
        if not is_correct[i]:
            continue
        if _speedup(baseline_speed[i], actual_speed[i]) > p:
            count += 1
    return count / n


def fast_p_curve(
    is_correct: Sequence[bool],
    baseline_speed: Sequence[float],
    actual_speed: Sequence[float],
    n: int,
    ps: Sequence[float] = DEFAULT_PS,
) -> list[tuple[float, float]]:
    """Return ``[(p, fast_p)]`` over the p-grid.

    The curve is monotonically non-increasing in ``p`` (a larger threshold can
    only exclude samples), which the tests assert.
    """
    return [(float(p), fastp(is_correct, baseline_speed, actual_speed, n, p)) for p in ps]


def geometric_mean_speedup(
    is_correct: Sequence[bool],
    baseline_speed: Sequence[float],
    actual_speed: Sequence[float],
) -> float:
    """Geometric mean of the speedup over the CORRECT-only subset.

    Geometric (not arithmetic) so equal multiplicative gains weigh equally,
    matching the log-speedup reward. Returns 0.0 if nothing is correct.
    """
    logs: list[float] = []
    m = min(len(is_correct), len(baseline_speed), len(actual_speed))
    for i in range(m):
        if not is_correct[i]:
            continue
        s = _speedup(baseline_speed[i], actual_speed[i])
        if s > 0:
            logs.append(math.log(s))
    if not logs:
        return 0.0
    return math.exp(sum(logs) / len(logs))


# --------------------------------------------------------------------------- #
# Unbiased best-of-N estimators (pass@k / fast_p@k) + multi-seed CI
# --------------------------------------------------------------------------- #
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. / HumanEval).

    Given ``n`` independent samples of which ``c`` are "successes", the expected
    fraction of size-``k`` draws (without replacement) that contain at least one
    success is ``1 - C(n-c, k) / C(n, k)``. This is the standard low-variance
    estimator used for parallel best-of-N; it avoids the high variance of the
    naive ``1 - (1-c/n)**k`` plug-in.
    """
    n, c, k = int(n), int(c), int(k)
    if n <= 0 or k <= 0:
        return 0.0
    k = min(k, n)
    c = max(0, min(c, n))
    if c <= 0:
        return 0.0
    if n - c < k:  # every size-k draw must contain a success
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def fast_p_at_k(
    is_correct: Sequence[bool],
    baseline_speed: Sequence[float],
    actual_speed: Sequence[float],
    k: int,
    p: float,
    n: int | None = None,
) -> float:
    """Unbiased fast_p@k: pass@k where a "success" is a sample that is BOTH
    correct AND faster than the baseline by more than a factor ``p``.

    ``n`` overrides the sample count (defaults to the number of samples given);
    ``c`` is the number of samples that are correct AND ``baseline/actual > p``.
    """
    m = min(len(is_correct), len(baseline_speed), len(actual_speed))
    total = m if n is None else int(n)
    c = 0
    for i in range(m):
        if is_correct[i] and _speedup(baseline_speed[i], actual_speed[i]) > p:
            c += 1
    return pass_at_k(total, c, k)


def mean_ci(values: Sequence[float], z: float = 1.96) -> dict:
    """Mean and (normal-approx) confidence half-width of ``values``.

    Returns ``{mean, std, sem, ci95, lo, hi, n}``. With <2 samples the CI is 0
    (a single seed carries no variance estimate). ``z=1.96`` -> 95% CI.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0, "ci95": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    mu = sum(vals) / n
    if n < 2:
        return {"mean": mu, "std": 0.0, "sem": 0.0, "ci95": 0.0, "lo": mu, "hi": mu, "n": n}
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    ci = z * sem
    return {"mean": mu, "std": std, "sem": sem, "ci95": ci, "lo": mu - ci, "hi": mu + ci, "n": n}
