"""Benchmark statistics: median, coefficient of variation, significance."""

from __future__ import annotations

import math
from dataclasses import dataclass


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


@dataclass
class BenchStat:
    median_ms: float
    mean_ms: float
    std_ms: float
    cv_pct: float
    n: int
    stable: bool

    @classmethod
    def from_samples(cls, samples: list[float], cv_threshold_pct: float = 3.0) -> "BenchStat":
        if not samples:
            raise ValueError("no bench samples")
        return cls(median(samples), mean(samples), std(samples), cv_pct(samples),
                   len(samples), cv_pct(samples) <= cv_threshold_pct)


def speedup_is_significant(baseline_ms: float, candidate_ms: float,
                           baseline_std_ms: float, candidate_std_ms: float,
                           noise_floor_pct: float = 2.0) -> bool:
    if candidate_ms <= 0 or baseline_ms <= 0:
        return False
    gain_pct = 100.0 * (baseline_ms - candidate_ms) / baseline_ms
    combined_sigma_pct = 100.0 * math.sqrt(baseline_std_ms**2 + candidate_std_ms**2) / baseline_ms if baseline_ms > 0 else 0.0
    return gain_pct > max(2.0 * combined_sigma_pct, noise_floor_pct)
