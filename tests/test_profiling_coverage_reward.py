"""Coverage-aware profiling rewards, and the failure modes they must refuse.

The reason coverage exists is that speedup alone cannot separate real
optimisation from lazy optimisation. Dr. Kernel's example -- a generated kernel
accounting for 0.014% of GPU time versus 86.15% for the same task fused properly
-- is asserted here against the published numbers, because it is the entire
justification for the metric.

The bulk of these tests are about what the reward REFUSES to produce. A profiling
reward that quietly returns 0.0 when the profiler failed teaches the policy that
a profiler failure and a worthless kernel are the same event, and a reward that
pays out on an implausible speedup trains the measurement exploit directly.
"""

from __future__ import annotations

import math

import pytest

from kore.reward.coverage import (
    FUSED_COVERAGE,
    LAZY_COVERAGE,
    MAX_PLAUSIBLE_SPEEDUP,
    amdahl_end_to_end_speedup,
    candidate_kernel_names,
    coverage_ceiling,
    coverage_feedback,
    dispatch_matches,
    implausible_speedup,
    kernel_coverage,
    profiling_reward,
)
from kore.verifier.parsers.rocprofv3 import (
    KernelDispatch,
    parse_kernel_dispatches,
)


def _d(name, ns):
    return KernelDispatch(kernel_name=name, duration_ns=ns)


# --------------------------------------------------------------------------- #
# 1. the paper's motivating example, in numbers
# --------------------------------------------------------------------------- #
def test_the_published_lazy_versus_fused_example_reproduces():
    """0.014% coverage caps the end-to-end win at 1.00014x; 86.15% caps it at 7.22x.

    This is why coverage is combined by Amdahl's law rather than added to
    speedup: at 0.014% no local speedup can matter, and any reward scheme that
    pays for one is paying for the wrong behaviour.
    """
    assert coverage_ceiling(LAZY_COVERAGE) == pytest.approx(1.00014, abs=1e-5)
    assert coverage_ceiling(FUSED_COVERAGE) == pytest.approx(7.2202, abs=1e-3)

    # A big local speedup on the irrelevant slice buys essentially nothing...
    lazy = amdahl_end_to_end_speedup(10.0, LAZY_COVERAGE)
    assert lazy == pytest.approx(1.000126, abs=1e-5)
    # ... while the same 10x on the fused kernel is a real 4.45x.
    # 1 / ((1 - 0.8615) + 0.8615/10) = 1 / 0.22465 = 4.4514
    fused = amdahl_end_to_end_speedup(10.0, FUSED_COVERAGE)
    assert fused == pytest.approx(4.4514, abs=1e-3)

    lazy_reward = profiling_reward(10.0, LAZY_COVERAGE, correct=True)
    fused_reward = profiling_reward(10.0, FUSED_COVERAGE, correct=True)
    assert lazy_reward is not None and fused_reward is not None
    assert lazy_reward.reward < 0.001
    assert fused_reward.reward > 0.9
    # The whole point, stated as an inequality on the reward itself.
    assert fused_reward.reward > 100 * lazy_reward.reward


def test_identical_local_speedups_are_not_identically_rewarded():
    """Two kernels, same measured speedup, different share of the work."""
    sliver = profiling_reward(8.0, 0.02, correct=True)
    most = profiling_reward(8.0, 0.90, correct=True)
    assert sliver.local_speedup == most.local_speedup == 8.0
    assert most.reward > 10 * sliver.reward


# --------------------------------------------------------------------------- #
# 2. coverage from a trace
# --------------------------------------------------------------------------- #
def test_coverage_is_the_share_of_gpu_time_the_candidate_accounts_for():
    dispatches = [_d("fused_attn_kernel_0d1d", 8_000),
                  _d("void at::native::elementwise", 1_000),
                  _d("hipblaslt_gemm", 1_000)]
    report = kernel_coverage(dispatches, {"fused_attn_kernel"})
    assert report is not None
    assert report.coverage == pytest.approx(0.8)
    assert report.candidate_ns == 8_000 and report.total_ns == 10_000
    assert report.n_candidate_dispatches == 1 and report.n_dispatches == 3
    assert not report.never_ran


def test_a_kernel_that_never_ran_is_measured_as_zero_not_unknown():
    """The decoy-kernel hack: ship something fast that nothing calls.

    Coverage 0.0 has to be a finding, distinct from "no profile", or the hack is
    indistinguishable from a profiler failure.
    """
    dispatches = [_d("void at::native::gemm", 5_000), _d("aiter_rmsnorm", 5_000)]
    report = kernel_coverage(dispatches, {"my_fast_kernel"})
    assert report is not None
    assert report.coverage == 0.0
    assert report.never_ran
    # It still earns a reward of exactly 0.0 -- measured, not withheld.
    reward = profiling_reward(8.0, report.coverage, correct=True)
    assert reward is not None and reward.reward == 0.0
    assert reward.end_to_end_speedup == pytest.approx(1.0)


def test_an_implausible_claim_is_refused_before_coverage_is_considered():
    """Order matters, so pin it: implausibility wins over a coverage of 0.0.

    A decoy kernel claiming 50x on 0% coverage trips both guards. Refusing on the
    implausible timing is the more informative verdict -- it says the MEASUREMENT
    is not admissible, rather than reporting a well-measured zero -- and it keeps
    the RL-time cap agreeing with the data-time one.
    """
    assert profiling_reward(50.0, 0.0, correct=True) is None
    assert profiling_reward(8.0, 0.0, correct=True) is not None


def test_specialised_triton_suffixes_still_count_as_the_candidate():
    """Requiring a right-hand word boundary would call this kernel "never ran"."""
    for traced in ("add_kernel", "add_kernel_0d1d2d3de", "add_kernel_warps4"):
        assert dispatch_matches(traced, {"add_kernel"}), traced
    # A different identifier that merely embeds the name is not a match.
    assert not dispatch_matches("my_add_kernel", {"add_kernel"})


def test_candidate_kernel_names_are_read_from_the_triton_decorators():
    source = """
import triton
import triton.language as tl

@triton.jit
def fused_rmsnorm_kernel(x_ptr, y_ptr, N: tl.constexpr):
    pass

@triton.jit()
def helper_kernel(a, b):
    pass

def not_a_kernel(x):
    pass
"""
    assert candidate_kernel_names(source) == {"fused_rmsnorm_kernel",
                                             "helper_kernel"}
    assert candidate_kernel_names("") == set()


def test_coverage_is_bounded_even_when_dispatches_overlap():
    """Concurrent streams make durations sum past wall time; a share cannot exceed 1."""
    dispatches = [_d("k_kernel", 10_000), _d("k_kernel", 10_000),
                  _d("other", 1)]
    report = kernel_coverage(dispatches, {"k_kernel"})
    assert 0.0 <= report.coverage <= 1.0
    assert report.n_candidate_dispatches == 2


# --------------------------------------------------------------------------- #
# 3. coverage refuses to invent a number
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dispatches,candidates", [
    ([], {"k"}),                                  # nothing profiled
    ([_d("k", 0), _d("other", 0)], {"k"}),        # zero total GPU time
    ([_d("k", 100)], set()),                      # no candidate names to match
    ([_d("k", 100)], {""}),                       # blank name is not a name
])
def test_unknowable_coverage_returns_none(dispatches, candidates):
    """None means "cannot say", 0.0 means "measured and it was zero"."""
    assert kernel_coverage(dispatches, candidates) is None


def test_corrupt_dispatch_rows_are_skipped_not_absorbed():
    dispatches = [_d("k_kernel", 100), _d("k_kernel", -5), _d("", 900),
                  ("other_kernel", 100), ("bad",), _d("x", float("nan"))]
    report = kernel_coverage(dispatches, {"k_kernel"})
    # 100 candidate ns out of 200 usable ns; the negative, nameless, malformed
    # and non-finite rows contribute to neither side.
    assert report.coverage == pytest.approx(0.5)
    assert report.total_ns == 200


# --------------------------------------------------------------------------- #
# 4. the reward refuses rather than guessing
# --------------------------------------------------------------------------- #
def test_an_incorrect_kernel_earns_no_profiling_reward_at_all():
    """Speed is meaningless without correctness, and this must never be a door."""
    assert profiling_reward(5.0, 0.9, correct=False) is None


@pytest.mark.parametrize("speedup,coverage", [
    (None, 0.5),                 # no timing
    (5.0, None),                 # no profile
    (float("nan"), 0.5),
    (float("inf"), 0.5),
    (5.0, float("nan")),
    (0.0, 0.5),                  # a zero speedup is a failed measurement
    (-2.0, 0.5),
    (5.0, -0.1),                 # coverage outside [0, 1] is a broken trace
    (5.0, 1.5),
    (True, 0.5),                 # a bool is not a measurement
])
def test_a_reward_that_cannot_be_grounded_is_withheld_not_zeroed(speedup, coverage):
    """0.0 would assert "measured, and worthless"; None says "not measured".

    Collapsing the two teaches the policy that a profiler failure costs the same
    as a bad kernel, which makes breaking the profiler a viable strategy.
    """
    assert profiling_reward(speedup, coverage, correct=True) is None


def test_an_implausible_speedup_is_refused_rather_than_paid():
    """1541.94x against a production vendor kernel is a gamed measurement.

    That exact value reached an early SFT mixture from third-party data, which is
    why the cap is enforced at reward time and not only at data-selection time.
    """
    assert implausible_speedup(1541.94)
    assert profiling_reward(1541.94, 0.9, correct=True) is None
    # Just under the cap still pays, so the guard is a cliff at a known place.
    assert implausible_speedup(MAX_PLAUSIBLE_SPEEDUP - 0.01) is False
    assert profiling_reward(MAX_PLAUSIBLE_SPEEDUP - 0.01, 0.9,
                            correct=True) is not None


def test_an_unmeasurable_speedup_is_not_called_implausible():
    """No claim, nothing to reject -- otherwise missing data becomes a rejection."""
    assert implausible_speedup(None) is False
    assert implausible_speedup(float("nan")) is False


def test_the_reward_is_bounded_so_one_measurement_cannot_own_the_group():
    """GRPO/TRLOO advantages are group-relative, so an unbounded term dominates."""
    for speedup in (1.0, 2.0, 5.0, MAX_PLAUSIBLE_SPEEDUP - 0.01):
        for coverage in (0.0, 0.3, 0.999, 1.0):
            reward = profiling_reward(speedup, coverage, correct=True)
            assert reward is not None, (speedup, coverage)
            assert 0.0 <= reward.reward <= 1.0, (speedup, coverage, reward)


def test_a_slowdown_earns_zero_rather_than_a_negative_shaping_term():
    """The correctness tiers own punishment; this term only ever adds."""
    reward = profiling_reward(0.25, 0.9, correct=True)
    assert reward is not None
    assert reward.end_to_end_speedup < 1.0
    assert reward.reward == 0.0


def test_a_degenerate_reward_cap_is_refused():
    assert profiling_reward(2.0, 0.5, correct=True, reward_cap=1.0) is None
    assert profiling_reward(2.0, 0.5, correct=True, reward_cap=0.0) is None
    assert profiling_reward(2.0, 0.5, correct=True,
                            reward_cap=float("nan")) is None


def test_full_coverage_reduces_to_the_local_speedup():
    """Amdahl with C=1 is the identity, which is the sanity check on the formula."""
    assert amdahl_end_to_end_speedup(6.0, 1.0) == pytest.approx(6.0)
    assert coverage_ceiling(1.0) is None       # no finite ceiling to report


def test_zero_coverage_means_no_end_to_end_change():
    assert amdahl_end_to_end_speedup(100.0, 0.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 5. feedback the model can act on
# --------------------------------------------------------------------------- #
def test_feedback_names_the_ceiling_so_the_model_fuses_instead_of_tuning():
    report = kernel_coverage([_d("k_kernel", 300), _d("other", 9_700)],
                             {"k_kernel"})
    text = coverage_feedback(report, local_speedup=8.0)
    assert "3.00%" in text
    assert "1.03x" in text            # the ceiling at 3% coverage
    assert "fuse more" in text


def test_feedback_for_a_kernel_that_never_ran_says_so_plainly():
    report = kernel_coverage([_d("someone_else", 1_000)], {"mine_kernel"})
    text = coverage_feedback(report)
    assert "never called" in text or "never" in text
    assert "1 dispatches" in text


# --------------------------------------------------------------------------- #
# 6. the trace reader
# --------------------------------------------------------------------------- #
def test_kernel_dispatches_parse_from_timestamps(tmp_path):
    csv_path = tmp_path / "trace.csv"
    csv_path.write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "add_kernel,1000,3000\n"
        "void at::native::x,3000,3500\n"
    )
    dispatches = parse_kernel_dispatches(csv_path)
    assert [(d.kernel_name, d.duration_ns) for d in dispatches] == [
        ("add_kernel", 2000), ("void at::native::x", 500)]


def test_an_explicit_duration_column_is_preferred(tmp_path):
    csv_path = tmp_path / "trace.csv"
    csv_path.write_text("Kernel_Name,Duration\nk,777\n")
    assert parse_kernel_dispatches(csv_path)[0].duration_ns == 777


def test_a_negative_interval_is_dropped_rather_than_absolute_valued(tmp_path):
    """An inverted timestamp pair means the export is wrong, not that time flowed backwards."""
    csv_path = tmp_path / "trace.csv"
    csv_path.write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "good,100,200\n"
        "inverted,900,100\n"
        "nameless_row_has_no_kernel,,\n"
    )
    dispatches = parse_kernel_dispatches(csv_path)
    assert [d.kernel_name for d in dispatches] == ["good"]


def test_an_empty_trace_is_empty_and_a_missing_one_raises(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("Kernel_Name,Duration\n")
    assert parse_kernel_dispatches(empty) == []
    with pytest.raises(FileNotFoundError):
        parse_kernel_dispatches(tmp_path / "nope.csv")


def test_the_counter_parser_still_ignores_timestamps(tmp_path):
    """The counter path deliberately does not source latency from the profiler.

    Adding a timestamp reader must not have changed that, so this pins it: a
    timestamp column must not appear as a counter.
    """
    from kore.verifier.parsers.rocprofv3 import parse_rocprofv3_csv

    csv_path = tmp_path / "counters.csv"
    csv_path.write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp,SQ_INSTS_VMEM\n"
        "add_kernel,1000,3000,42\n"
    )
    pmc = parse_rocprofv3_csv(csv_path)[0]
    assert pmc.counters == {"SQ_INSTS_VMEM": 42}
    assert "Start_Timestamp" not in pmc.counters
