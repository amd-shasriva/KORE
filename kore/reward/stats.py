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
