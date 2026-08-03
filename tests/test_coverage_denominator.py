"""Regression tests for what coverage measures, and what it must not be used for.

These encode facts measured on gfx950 (docs/evidence/coverage_denominator.md).
They are deliberately hardware-free: the point is not to re-measure the GPU on
every CI run, it is to stop the conclusions from being quietly undone by a
config edit or a refactor of the driver.

The finding they defend: coverage is computed on the CANDIDATE over a
denominator containing fixed harness work, so it rises as the kernel gets
slower. That makes it a working decoy detector and an unusable Amdahl ``p``.
"""
from __future__ import annotations

import inspect
import os

from kore.policy.configs import GRPOConfig
from kore.policy.rejection import profiling_rejection_sample
from kore.reward.coverage import (
    amdahl_end_to_end_speedup, kernel_coverage, profiling_reward)


# --------------------------------------------------------------------------- #
# the inversion itself
# --------------------------------------------------------------------------- #
def test_fixed_harness_makes_coverage_rise_as_the_kernel_speeds_up():
    """A faster kernel takes a smaller share of a denominator it does not shrink.

    This is the mechanism behind the measured 0.057 (correct seed) vs 0.739 (46x
    slower copy). Reproduced here in arithmetic so the property is pinned even
    though CI has no GPU: hold the harness at the measured ~344k ns of bench
    setup and vary only the kernel.
    """
    harness_ns = 344_000          # measured, constant across all three variants
    kernel_fast_ns = 283_519      # measured: gen_add_bf16 seed
    kernel_slow_ns = 10_997_126   # measured: same kernel, 46x slowed

    fast = kernel_coverage([("_add_kernel", kernel_fast_ns),
                            ("harness", harness_ns)], ["_add_kernel"])
    slow = kernel_coverage([("_add_kernel", kernel_slow_ns),
                            ("harness", harness_ns)], ["_add_kernel"])

    assert fast is not None and slow is not None
    assert slow.coverage > fast.coverage, (
        "coverage must be understood as increasing with kernel cost; if this "
        "ever fails the denominator changed and the evidence doc is stale")

    # And the consequence that makes it unusable as a reward: the SLOWER kernel
    # earns the larger shaping term at equal measured speedup.
    r_fast = profiling_reward(2.0, fast.coverage, correct=True)
    r_slow = profiling_reward(2.0, slow.coverage, correct=True)
    assert r_fast is not None and r_slow is not None
    assert r_slow.reward > r_fast.reward


def test_amdahl_p_must_not_come_from_the_candidate():
    """Amdahl is monotonic in speedup only when ``p`` is held fixed.

    With ``p`` fixed (a property of the baseline, as Amdahl defines it) a faster
    kernel is worth more. Letting ``p`` fall as the kernel improves -- which is
    what measuring coverage on the candidate does -- can reverse that.
    """
    p = 0.45
    assert (amdahl_end_to_end_speedup(4.0, p)
            > amdahl_end_to_end_speedup(2.0, p))

    # Same 2x -> 4x improvement, but with p dragged down the way a shrinking
    # share of a fixed harness drags it: the "improvement" now scores worse.
    assert (amdahl_end_to_end_speedup(4.0, 0.15)
            < amdahl_end_to_end_speedup(2.0, 0.45))


# --------------------------------------------------------------------------- #
# the part that IS sound
# --------------------------------------------------------------------------- #
def test_decoy_reads_as_zero_and_is_rejected():
    """coverage 0.0 on a healthy trace is the validated signal: keep rejecting it."""
    dispatches = [("at::native::vectorized_elementwise_kernel", 500_000),
                  ("__amd_rocclr_copyBuffer", 120_000)]
    report = kernel_coverage(dispatches, ["_add_kernel"])
    assert report is not None
    assert report.coverage == 0.0
    assert report.never_ran

    verdict = profiling_rejection_sample(
        correct=True, speedup=9.0, coverage=report.coverage,
        min_coverage=GRPOConfig.prs_min_coverage)
    assert not verdict.keep
    assert verdict.reason == "kernel_never_ran"


def test_unmeasurable_coverage_is_not_zero_coverage():
    """No trace must not be laundered into "the kernel never ran"."""
    assert kernel_coverage([], ["_add_kernel"]) is None
    assert kernel_coverage([("_add_kernel", 1000)], []) is None


# --------------------------------------------------------------------------- #
# configuration must stay safe against the measured distribution
# --------------------------------------------------------------------------- #
def test_correct_seed_kernels_survive_the_default_prs_threshold():
    """Every coverage measured from a CORRECT seed must be accepted by default.

    These are the ten values observed on gfx950. The spread is harness overhead,
    not kernel quality, so any threshold that rejects one of them is rejecting
    correct work. gen_relu_fp32 (0.095) and softmax_bf16 (0.036) are the two the
    old 0.1 default would have thrown away.
    """
    measured_correct = [0.5866, 0.5494, 0.4618, 0.4520, 0.4487,
                        0.2995, 0.2908, 0.1614, 0.0952, 0.0363]
    for coverage in measured_correct:
        verdict = profiling_rejection_sample(
            correct=True, speedup=1.5, coverage=coverage,
            min_coverage=GRPOConfig.prs_min_coverage)
        assert verdict.keep, (
            f"coverage {coverage} came from a correct reference kernel; "
            f"prs_min_coverage={GRPOConfig.prs_min_coverage} rejects it")


def test_profiling_reward_stays_disarmed_by_default():
    """A receipt proves the trace path works, not that the magnitude means p."""
    assert GRPOConfig.profiling_reward_weight == 0.0


# --------------------------------------------------------------------------- #
# the partial fix stays wired
# --------------------------------------------------------------------------- #
def test_bench_only_env_var_suppresses_post_timing_correctness():
    """driver_main must honour KORE_TRACE_BENCH_ONLY.

    Without it the profiled process re-runs correctness -- random inputs, the
    ATen reference, an allclose reduction -- and on gen_add_bf16 that was 279 of
    289 dispatches.
    """
    from kore.tasks import _genops
    src = inspect.getsource(_genops.driver_main)
    assert "KORE_TRACE_BENCH_ONLY" in src
    # Suppression must apply to the candidate branch, where correctness reruns.
    assert "_run_correctness" in src


def test_collect_kernel_trace_sets_bench_only():
    """The env var is useless unless the collector actually sets it."""
    from kore.env import kore_env
    src = inspect.getsource(kore_env.KoreEnv.collect_kernel_trace)
    assert 'KORE_TRACE_BENCH_ONLY' in src
    assert '"1"' in src


def test_bench_only_is_an_env_var_not_a_required_flag():
    """Drivers predating the fix must ignore it, not fail argparse.

    A CLI flag would make collect_kernel_trace return None for every
    hand-written driver, silently removing them from coverage entirely.
    """
    from kore.tasks import _genops
    src = inspect.getsource(_genops.driver_main)
    assert "--trace-bench-only" not in src
    assert os.environ.get("KORE_TRACE_BENCH_ONLY") in (None, "", "0", "1",
                                                       "true", "True")
