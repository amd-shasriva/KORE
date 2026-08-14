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


# ------------------------------------------- the harness's own geomean ---
#: A real GEAK harness transcript. The per-shape lines come FIRST and the
#: geometric mean over all of them comes LAST, which is the trap: a generic
#: search for "speedup=" returns shape 0. Here shape 0 is deliberately the
#: least favourable, as it was in the transcript this was reproduced from.
GEAK_GEOMEAN = (
    "Benchmarking 4 shapes\n"
    "  shape [128, 128]   ref=1.2000ms  opt=1.0000ms  speedup=1.200\n"
    "  shape [512, 512]   ref=8.0000ms  opt=1.6000ms  speedup=5.000\n"
    "  shape [1024, 1024] ref=32.000ms  opt=4.0000ms  speedup=8.000\n"
    "  shape [2048, 2048] ref=64.000ms  opt=12.800ms  speedup=5.000\n"
    "Geometric mean speedup: 4.579x\n"
    "Median kernel latency: 4.0000 ms\n"
    "GEAK_RESULT_LATENCY_MS=4.000000\n"
    "GEAK_RESULT_GEOMEAN_SPEEDUP=4.5789\n"
)


def test_the_harnesss_own_geomean_wins_over_the_first_shape():
    """32 harnesses compute a geomean across every shape they timed and print it.

    Without reading it, the generic ``speedup=`` search matches the FIRST
    per-shape line. Measured against this transcript that returned 1.2x where the
    harness's own answer was 4.5789x -- a 3.8x understatement fed straight into
    ``120 + speedup*100``, and in an arbitrary direction, since it depends only on
    which shape the harness happened to print first.
    """
    base, opt, sp = _parse_speedup(GEAK_GEOMEAN)
    assert sp == pytest.approx(4.5789)
    assert sp != pytest.approx(1.2)


def test_a_json_ratio_still_wins_over_the_geomean():
    """Ordering: an explicit machine-readable ratio is the harness's most
    deliberate statement, so it outranks the printed geomean."""
    assert _parse_speedup(JSONBLOB + GEAK_GEOMEAN)[2] == 3.5


def test_a_negative_geomean_is_a_failure_sentinel_not_a_measurement():
    """flydsl2flydsl/pa_decode_swa_kernel prints GEAK_RESULT_GEOMEAN_SPEEDUP=-1
    when it could not benchmark. That is not a speedup, and it must not become
    one; falling through to the latency leaves a usable optimized time instead."""
    text = ("GEAK_RESULT_LATENCY_MS=0.5000\n"
            "GEAK_RESULT_GEOMEAN_SPEEDUP=-1\n")
    base, opt, sp = _parse_speedup(text)
    assert sp is None
    assert opt == pytest.approx(0.5)


# ------------------------------------------ per-case matched ratios ---
def test_speedup_is_the_mean_of_ratios_not_the_ratio_of_means():
    """AKA averages per-case ratios; a ratio of aggregates is dominated by the
    largest shape and is not the quantity the published bars were computed with.

    Here the two disagree by 5.5x versus 1.005x on the same measurements.
    """
    from kore.eval.agent_kernel_arena import speedup_from_cases

    base = [{"shape": [8], "execution_time_ms": 1.0},
            {"shape": [4096], "execution_time_ms": 100.0}]
    opt = [{"shape": [8], "execution_time_ms": 0.1},
           {"shape": [4096], "execution_time_ms": 100.0}]
    sp, note = speedup_from_cases(base, opt)
    assert sp == pytest.approx(5.5)       # mean of (10.0, 1.0)
    assert sp != pytest.approx(1.005)     # ratio of means, the old behaviour
    assert note is None


def test_cases_are_paired_by_identity_not_position():
    """A harness that skips a failing shape shifts every later index, which would
    divide one shape's time by another's."""
    from kore.eval.agent_kernel_arena import speedup_from_cases

    base = [{"params": {"m": 1}, "execution_time_ms": 2.0},
            {"params": {"m": 2}, "execution_time_ms": 10.0}]
    opt = [{"params": {"m": 2}, "execution_time_ms": 5.0},
           {"params": {"m": 1}, "execution_time_ms": 1.0}]
    sp, note = speedup_from_cases(base, opt)
    assert sp == pytest.approx(2.0)       # both shapes are exactly 2x
    assert note is None


def test_an_incomplete_case_match_refuses_rather_than_comparing_shapes():
    from kore.eval.agent_kernel_arena import speedup_from_cases

    base = [{"shape": [8], "execution_time_ms": 1.0},
            {"shape": [16], "execution_time_ms": 2.0}]
    opt = [{"shape": [8], "execution_time_ms": 0.5}]
    sp, note = speedup_from_cases(base, opt)
    assert sp is None
    assert "incomplete case match" in note


def test_a_nonpositive_case_time_refuses():
    """vLLM runners emit execution_time_ms: -1.0 for a case that failed. Averaging
    only the cases that worked would silently score a partial benchmark."""
    from kore.eval.agent_kernel_arena import speedup_from_cases

    base = [{"shape": [8], "execution_time_ms": 1.0}]
    opt = [{"shape": [8], "execution_time_ms": -1.0}]
    sp, note = speedup_from_cases(base, opt)
    assert sp is None
    assert "non-positive" in note


def test_a_benchmark_method_mismatch_is_flagged_not_hidden():
    """Measured on gfx950, the same torch.relu reads 0.00514ms under CUDA-graph
    capture and 0.01232ms with per-launch events -- 2.4x from the method alone.
    Dividing one by the other manufactures a speedup out of nothing."""
    from kore.eval.agent_kernel_arena import speedup_from_cases

    base = [{"shape": [8], "execution_time_ms": 1.0,
             "benchmark_method": "cuda_graph"}]
    opt = [{"shape": [8], "execution_time_ms": 0.5,
            "benchmark_method": "cuda_event_fallback"}]
    sp, note = speedup_from_cases(base, opt)
    assert sp == pytest.approx(2.0)   # the number is real...
    assert "benchmark method mismatch" in note   # ...but not about the kernel
