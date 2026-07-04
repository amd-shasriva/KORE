"""CPU-only tests for the performance-reward-hacking taxonomy + coverage."""

from __future__ import annotations

from kore.reward import timing_integrity as ti
from kore.reward.reward import scan_for_hacks


def test_every_perf_hack_class_has_a_defense():
    assert ti.uncovered() == []
    cov = ti.coverage()
    assert set(cov) == set(ti.PerfHackClass)


def test_scanner_actually_closes_the_static_hack_classes():
    """The classes attributed to scan_for_hacks are really caught by it."""
    assert scan_for_hacks("print('median_ms: 0.001')") is not None      # verdict forgery
    assert scan_for_hacks("import sys\nif '--bench-mode' in sys.argv: fast()") is not None  # mode sniff
    assert scan_for_hacks("import time\ntime.sleep(0.0)") is not None    # clock manipulation
    assert scan_for_hacks("def f(a, b):\n    return a @ b") is not None  # delegation


def test_report_lists_all_classes():
    r = ti.report()
    for h in ti.PerfHackClass:
        assert h.value in r


def test_coverage_is_one_to_one():
    cov = ti.coverage()
    # exactly one primary defense per class
    assert len(cov) == len(ti.PerfHackClass)
    assert len(ti.DEFENSES) == len(ti.PerfHackClass)
