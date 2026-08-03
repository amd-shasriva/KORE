"""MRS and PRS: what they reject, and what they refuse to reject.

The interesting tests here are the second kind. A rejection filter in front of a
GROUP-RELATIVE estimator can destroy the contrast the estimator learns from, so
the filter has to be able to decline. And a filter that rejects everything it
cannot measure silently narrows training to the measurable subset while looking
like a quality improvement.
"""

from __future__ import annotations

import pytest

from kore.policy.rejection import (
    ARITHMETIC,
    GEOMETRIC,
    MINIMUM,
    aggregate_quality,
    multi_turn_rejection_sample,
    profiling_rejection_sample,
    trajectory_verdict,
    turn_quality,
)


# --------------------------------------------------------------------------- #
# 1. per-turn quality and its aggregation
# --------------------------------------------------------------------------- #
def test_an_incorrect_turn_has_zero_quality():
    assert turn_quality(0.9, correct=False) == 0.0
    assert turn_quality(0.9, correct=True) == pytest.approx(0.9)


def test_quality_is_clamped_and_rejects_unusable_rewards():
    assert turn_quality(5.0, correct=True) == 1.0            # clamped
    assert turn_quality(-1.0, correct=True) == 0.0
    assert turn_quality(None, correct=True) == 0.0
    assert turn_quality(float("nan"), correct=True) == 0.0
    assert turn_quality(True, correct=True) == 0.0           # bool is not a reward


def test_geometric_aggregation_is_dominated_by_the_weakest_turn():
    """One broken turn must not be averaged away by three good ones.

    This is why geometric is the default: the arithmetic mean of the same trace
    still looks like a decent trajectory.
    """
    trace = [0.9, 0.9, 0.9, 0.0]
    assert aggregate_quality(trace, GEOMETRIC) == 0.0
    assert aggregate_quality(trace, ARITHMETIC) == pytest.approx(0.675)
    assert aggregate_quality(trace, MINIMUM) == 0.0

    uneven = [0.9, 0.1]
    assert aggregate_quality(uneven, GEOMETRIC) == pytest.approx(0.3, abs=1e-9)
    assert aggregate_quality(uneven, ARITHMETIC) == pytest.approx(0.5)


def test_an_empty_trace_has_no_quality_rather_than_zero_quality():
    assert aggregate_quality([], GEOMETRIC) is None


def test_an_unknown_aggregate_is_refused():
    with pytest.raises(ValueError, match="aggregate must be one of"):
        aggregate_quality([0.5], "harmonic")


# --------------------------------------------------------------------------- #
# 2. per-trajectory verdicts
# --------------------------------------------------------------------------- #
def test_a_trajectory_that_never_became_correct_is_rejected():
    v = trajectory_verdict(0, [0.1, 0.1], [False, False])
    assert not v.keep and v.reason == "never_correct"


def test_a_gamed_measurement_is_rejected_before_anything_else():
    """A trajectory must not pass on its good turns while one turn is a hack."""
    v = trajectory_verdict(0, [0.9, 0.95], [True, True],
                           turn_speedups=[1.4, 1541.94])
    assert not v.keep and v.reason == "implausible_speedup"
    assert v.best_speedup == pytest.approx(1541.94)


def test_a_healthy_improving_trajectory_is_kept():
    v = trajectory_verdict(0, [0.5, 0.8], [True, True], turn_speedups=[1.1, 1.9])
    assert v.keep and v.reason == ""
    assert v.improved and v.best_speedup == pytest.approx(1.9)


def test_lazy_optimisation_is_only_rejected_when_improvement_is_required():
    """A flat trajectory is the lazy one, but arriving good on turn 1 is not lazy."""
    flat = dict(turn_rewards=[0.8, 0.8], turn_correct=[True, True],
                turn_speedups=[1.5, 1.5])
    assert trajectory_verdict(0, flat["turn_rewards"], flat["turn_correct"],
                              turn_speedups=flat["turn_speedups"]).keep
    strict = trajectory_verdict(0, flat["turn_rewards"], flat["turn_correct"],
                                turn_speedups=flat["turn_speedups"],
                                require_improvement=True)
    assert not strict.keep and strict.reason == "no_improvement"

    # A single measured turn cannot be judged on improvement -- there is no
    # earlier measurement to improve on -- so it is not rejected as lazy.
    single = trajectory_verdict(0, [0.8], [True], turn_speedups=[1.5],
                                require_improvement=True)
    assert single.keep


def test_an_empty_trajectory_is_rejected_as_having_no_turns():
    v = trajectory_verdict(0, [], [])
    assert not v.keep and v.reason == "no_turns" and v.quality is None


# --------------------------------------------------------------------------- #
# 3. MRS declines rather than destroying the group's contrast
# --------------------------------------------------------------------------- #
def test_mrs_drops_the_failures_when_contrast_survives():
    rewards = [[0.9, 0.95], [0.2, 0.3], [0.0, 0.0], [0.5, 0.55]]
    correct = [[True, True], [True, True], [False, False], [True, True]]
    report = multi_turn_rejection_sample(rewards, correct)
    assert report.applied
    assert report.keep_indices == [0, 1, 3]
    assert report.rejected == {"never_correct": 1}


def test_mrs_declines_when_only_one_trajectory_would_survive():
    """A single survivor has no peer, so TRLOO could not form a baseline at all.

    Keeping the whole group is strictly better than handing the estimator a group
    it cannot compute an advantage for.
    """
    rewards = [[0.9, 0.95], [0.0], [0.0], [0.0]]
    correct = [[True, True], [False], [False], [False]]
    report = multi_turn_rejection_sample(rewards, correct)
    assert not report.applied
    assert report.skipped_reason == "too_few_survivors"
    assert report.keep_indices == [0, 1, 2, 3]


def test_mrs_declines_when_the_survivors_would_all_look_identical():
    """Filtering to a zero-variance group turns a training step into a no-op.

    The mean reward of the survivors looks excellent, which is exactly why this
    has to be checked rather than trusted.
    """
    rewards = [[0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.0, 0.0]]
    correct = [[True, True], [True, True], [True, True], [False, False]]
    report = multi_turn_rejection_sample(rewards, correct)
    assert not report.applied
    assert report.skipped_reason == "would_collapse_variance"
    assert len(report.keep_indices) == 4


def test_a_gamed_trajectory_stays_rejected_even_when_the_filter_declines():
    """The one rejection that is never traded off against group variance."""
    rewards = [[0.9, 0.9], [0.9, 0.9], [0.9, 0.9]]
    correct = [[True, True], [True, True], [True, True]]
    speedups = [[1.2, 1.3], [1.2, 1.3], [1.1, 1541.94]]
    report = multi_turn_rejection_sample(rewards, correct,
                                        traj_speedups=speedups)
    # The survivors are identical, so the filter declines...
    assert not report.applied
    assert report.skipped_reason == "would_collapse_variance"
    # ... but the gamed trajectory does not come back.
    assert report.keep_indices == [0, 1]
    assert report.rejected == {"implausible_speedup": 1}


def test_mrs_report_is_serialisable_for_the_step_log():
    rewards = [[0.9], [0.4], [0.0]]
    correct = [[True], [True], [False]]
    summary = multi_turn_rejection_sample(rewards, correct).as_dict()
    assert summary["total"] == 3
    assert summary["kept"] == 2
    assert summary["applied"] is True
    assert summary["rejected"] == {"never_correct": 1}


def test_mrs_on_an_empty_group_declines_rather_than_raising():
    report = multi_turn_rejection_sample([], [])
    assert not report.applied and report.keep_indices == []


def test_ragged_turn_counts_are_handled():
    rewards = [[0.9, 0.95, 0.99], [0.3], [0.0, 0.0]]
    correct = [[True, True, True], [True], [False, False]]
    report = multi_turn_rejection_sample(rewards, correct)
    assert report.keep_indices == [0, 1]


# --------------------------------------------------------------------------- #
# 4. PRS on profiling signals
# --------------------------------------------------------------------------- #
def test_prs_rejects_a_kernel_that_never_dispatched():
    """The decoy hack: correctness satisfied by the reference, kernel unused."""
    v = profiling_rejection_sample(correct=True, speedup=6.0, coverage=0.0)
    assert not v.keep and v.reason == "kernel_never_ran"


def test_prs_rejects_a_speedup_on_a_sliver_of_the_runtime():
    v = profiling_rejection_sample(correct=True, speedup=9.0, coverage=0.02)
    assert not v.keep and v.reason == "lazy_optimisation"
    # The same speedup over most of the runtime is a real win.
    assert profiling_rejection_sample(correct=True, speedup=9.0,
                                      coverage=0.86).keep


def test_prs_rejects_an_implausible_speedup_before_looking_at_coverage():
    v = profiling_rejection_sample(correct=True, speedup=1541.94, coverage=0.9)
    assert not v.keep and v.reason == "implausible_speedup"


def test_prs_tests_correctness_first():
    v = profiling_rejection_sample(correct=False, speedup=2.0, coverage=0.9)
    assert not v.keep and v.reason == "incorrect"


def test_prs_has_no_opinion_without_a_profile_by_default():
    """Rejecting every unprofiled candidate would narrow training silently.

    The profiler is not available for every task on every node, so a missing
    profile must mean "no opinion", not "bad kernel" -- otherwise the training
    distribution quietly becomes the profilable subset while the filter's
    statistics look like a quality improvement.
    """
    assert profiling_rejection_sample(correct=True, speedup=2.0,
                                      coverage=None).keep
    strict = profiling_rejection_sample(correct=True, speedup=2.0,
                                        coverage=None, require_profile=True)
    assert not strict.keep and strict.reason == "no_profile"


@pytest.mark.parametrize("coverage", [-0.1, 1.4, float("nan"), float("inf")])
def test_prs_treats_a_broken_trace_as_no_profile_not_as_a_bad_kernel(coverage):
    assert profiling_rejection_sample(correct=True, speedup=2.0,
                                      coverage=coverage).keep
    strict = profiling_rejection_sample(correct=True, speedup=2.0,
                                        coverage=coverage, require_profile=True)
    assert not strict.keep and strict.reason == "no_profile"


def test_prs_keeps_a_candidate_at_exactly_the_coverage_threshold():
    """The threshold is a floor, not a gap: 0.10 coverage passes at min 0.10."""
    assert profiling_rejection_sample(correct=True, speedup=2.0, coverage=0.10,
                                      min_coverage=0.10).keep
    assert not profiling_rejection_sample(correct=True, speedup=2.0,
                                          coverage=0.0999,
                                          min_coverage=0.10).keep
