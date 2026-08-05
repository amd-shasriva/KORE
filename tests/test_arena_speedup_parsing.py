"""Speedup parsing across the arena's several harness dialects.

Speedup is the number the whole benchmark turns on -- score is
``120 + speedup*100``, so a measurement that is taken and then dropped costs more
points than a kernel that was merely slow. The first full sweep lost 134 of 142
correct kernels this way: they ran, passed, reported their latency, and scored as
though they had no speedup at all, because the parser knew one dialect and the
suites use three.

Returning None is always allowed and never wrong here. Inventing a number is,
since it feeds the score directly.
"""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kore.eval.agent_kernel_arena import _parse_speedup  # noqa: E402


GEAK = ("Running benchmark on 4 configs ...\n"
        "  [0] M=64,N=64  0.1234ms\n"
        "GEAK_SHAPES_USED=[0, 1]\n"
        "GEAK_RESULT_LATENCY_MS=0.4321\n")

HIPBENCH = ("[INFO] HIP kernel foo processed 3 test cases.\n"
            "[INFO] Average: ref=1.00000ms, opt=0.50000ms, speedup=2.00x\n")

JSONBLOB = '{"speedup": 3.5, "ori_time": 2.0, "opt_time": 0.5714}\n'


def test_geak_latency_is_kept_as_a_time_not_discarded():
    """GEAK reports an absolute geomean and no ratio -- one run has nothing to
    compare against. The latency is still a real measurement, so it comes back as
    the optimized time for the caller to divide by a baseline."""
    base, opt, sp = _parse_speedup(GEAK)
    assert (base, opt, sp) == (None, 0.4321, None)


def test_ratio_dialects_are_read_directly():
    assert _parse_speedup(HIPBENCH)[2] == 2.0
    assert _parse_speedup(JSONBLOB)[2] == 3.5


def test_a_ratio_wins_over_a_latency_when_both_appear():
    """A harness that reports both is telling us the ratio it intends; deriving
    our own from its latency could disagree with it."""
    assert _parse_speedup(HIPBENCH + GEAK)[2] == 2.0


@pytest.mark.parametrize("text", [
    "", "Running benchmark on 4 configs ...\ndone\n",
    "GEAK_RESULT_LATENCY_MS=notanumber\n",
    "speedup=\n", '{"speedup": null}\n',
    "GEAK_RESULT_LATENCY_MS=0.0\n",     # a zero latency would divide to infinity
    "speedup=-1.5\n",                    # negative is not a measurement
])
def test_unusable_output_yields_no_number(text):
    assert _parse_speedup(text)[2] is None


def test_zero_latency_is_not_offered_as_a_divisor():
    """0ms is a broken measurement, and using it as a denominator would produce an
    infinite speedup and an unbounded score."""
    assert _parse_speedup("GEAK_RESULT_LATENCY_MS=0.0\n")[1] is None


def _refs(tmp_path, rows):
    (tmp_path / "baseline.shard0of2.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    from run_agent_kernel_arena import _reference_latencies
    return _reference_latencies(tmp_path)


def test_only_a_correct_baseline_is_a_valid_denominator(tmp_path):
    """Timing a reference that failed its own correctness check would inflate
    every speedup measured against it."""
    refs = _refs(tmp_path, [
        {"task_id": "good", "correct": True, "optimized_seconds": 2.0},
        {"task_id": "wrong", "correct": False, "optimized_seconds": 9.0},
        {"task_id": "untimed", "correct": True, "optimized_seconds": None},
    ])
    assert refs == {"good": 2.0}


def test_partial_baseline_is_usable_before_it_finishes(tmp_path):
    """Shard ledgers are read directly, so a baseline still in flight already
    scores the tasks it has timed. Waiting for all 402 would serialize two runs
    that do not need to be."""
    assert _refs(tmp_path, [{"task_id": "t", "correct": True,
                             "optimized_seconds": 1.5}]) == {"t": 1.5}
