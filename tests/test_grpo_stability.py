"""Dual-clip PPO and rollout/training mismatch correction.

Two failure modes with the same shape: one sample acquiring an unbounded share of
the gradient. Dual-clip bounds it on the negative-advantage side of the PPO
surrogate; truncated importance sampling bounds it when the policy that generated
a sample is not quite the policy being trained.

The most important test in this file is the one showing that applying the
dual-clip floor UNCONDITIONALLY -- rather than only where the advantage is
negative -- silently replaces the objective for positive advantages. That is an
easy mistake to make, it produces no error, and it removes the clipping entirely.
"""

from __future__ import annotations

import math

import pytest

from kore.policy.configs import GRPOConfig
from kore.policy.grpo import (
    clip_higher_ratio,
    dual_clip_ratio,
    mismatch_weight,
)


# --------------------------------------------------------------------------- #
# 1. dual-clip: the unbounded side of the surrogate
# --------------------------------------------------------------------------- #
def test_the_plain_surrogate_is_unbounded_for_a_negative_advantage():
    """The defect dual-clip fixes, shown on the shipped function.

    With A < 0 the ``min`` selects ``r*A``, so the surrogate falls without limit
    as the ratio grows and the loss ``-surrogate`` grows without limit with it.
    """
    advantage = -1.0
    values = [clip_higher_ratio(r, advantage) for r in (2.0, 10.0, 100.0, 1e6)]
    assert values == pytest.approx([-2.0, -10.0, -100.0, -1e6])
    assert values[-1] < values[0]          # strictly worse, without bound


def test_dual_clip_floors_the_negative_advantage_surrogate():
    """With A < 0 the surrogate is ``r*A``, so the floor binds once ``r > c``."""
    advantage = -1.0
    for ratio in (10.0, 100.0, 1e6):
        floored = dual_clip_ratio(ratio, advantage, c=3.0)
        assert floored == pytest.approx(-3.0), ratio
    # At or below c the ordinary surrogate is already above the floor.
    assert dual_clip_ratio(1.5, advantage, c=3.0) == pytest.approx(-1.5)
    assert dual_clip_ratio(2.0, advantage, c=3.0) == pytest.approx(-2.0)
    assert dual_clip_ratio(3.0, advantage, c=3.0) == pytest.approx(-3.0)


def test_dual_clip_leaves_a_positive_advantage_alone():
    """The clip-higher upper bound must survive, not be replaced by c*A."""
    for ratio in (0.5, 1.0, 1.5, 10.0):
        assert dual_clip_ratio(ratio, 1.0, c=3.0) == pytest.approx(
            clip_higher_ratio(ratio, 1.0)), ratio


def test_applying_the_floor_unconditionally_would_destroy_the_clip():
    """Why ``dual_clip_ratio`` tests the advantage's sign.

    With A > 0 and c = 3, ``c*A`` exceeds the clip-higher ceiling ``(1+hi)*A``,
    so an unconditional ``max(surrogate, c*A)`` returns ``c*A`` for EVERY ratio.
    The objective would become a constant multiple of the advantage with the
    importance ratio removed entirely -- no error, no clipping, no PPO.
    """
    advantage, c = 1.0, 3.0
    correct = dual_clip_ratio(2.0, advantage, c=c)
    naive = max(clip_higher_ratio(2.0, advantage), c * advantage)
    assert naive == pytest.approx(3.0)
    assert correct == pytest.approx(1.28)      # (1 + hi) * A, hi defaults to 0.28
    assert naive != pytest.approx(correct)


@pytest.mark.parametrize("c", [0.0, None, 1.0, -1.0])
def test_a_disabled_floor_reproduces_the_previous_surrogate_exactly(c):
    """A strict extension: the incumbent objective must be recoverable."""
    for advantage in (-2.0, -0.5, 0.0, 0.5, 2.0):
        for ratio in (0.1, 0.9, 1.0, 1.1, 5.0):
            assert dual_clip_ratio(ratio, advantage, c=c) == clip_higher_ratio(
                ratio, advantage), (c, advantage, ratio)


def test_a_zero_advantage_is_unaffected_by_the_floor():
    assert dual_clip_ratio(100.0, 0.0, c=3.0) == pytest.approx(0.0)


def test_dual_clip_stays_differentiable_on_a_torch_ratio():
    """The clamped region must yield a defined zero gradient, not a dropped graph."""
    torch = pytest.importorskip("torch")

    ratio = torch.tensor(50.0, requires_grad=True)
    loss = -dual_clip_ratio(ratio, -1.0, c=3.0)
    assert torch.is_tensor(loss)
    loss.backward()
    assert ratio.grad is not None
    assert float(ratio.grad) == pytest.approx(0.0)   # deep inside the floor

    # Inside the unclipped band the gradient flows normally.
    live = torch.tensor(1.0, requires_grad=True)
    (-dual_clip_ratio(live, -1.0, c=3.0)).backward()
    assert float(live.grad) == pytest.approx(1.0)


def test_dual_clip_bounds_the_loss_a_single_sample_can_contribute():
    """The property that matters at the batch level, stated directly."""
    worst_plain = max(-clip_higher_ratio(r, -1.0) for r in (10.0, 1e3, 1e6))
    worst_floored = max(-dual_clip_ratio(r, -1.0, c=3.0)
                        for r in (10.0, 1e3, 1e6))
    assert worst_plain >= 1e6
    assert worst_floored == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# 2. mismatch correction
# --------------------------------------------------------------------------- #
def test_an_identical_rollout_and_training_logprob_needs_no_correction():
    assert mismatch_weight(-1.5, -1.5) == pytest.approx(1.0)


def test_the_weight_is_the_ratio_of_the_two_policies():
    assert mismatch_weight(math.log(0.5), math.log(0.25)) == pytest.approx(2.0)
    assert mismatch_weight(math.log(0.25), math.log(0.5)) == pytest.approx(0.5)


def test_the_upper_tail_is_truncated_and_the_lower_tail_is_not():
    """Variance comes from large weights; a small one merely discounts a sample."""
    assert mismatch_weight(0.0, -10.0, cap=2.0) == pytest.approx(2.0)
    assert mismatch_weight(-10.0, 0.0, cap=2.0) == pytest.approx(math.exp(-10.0))


@pytest.mark.parametrize("train,rollout", [
    (None, -1.0),
    (-1.0, None),
    (None, None),
    (float("nan"), -1.0),
    (-1.0, float("inf")),
    (True, -1.0),
])
def test_a_missing_logprob_means_no_correction_not_a_guessed_one(train, rollout):
    """Exactly 1.0: the sample is used as-is rather than reweighted by a guess."""
    assert mismatch_weight(train, rollout) == 1.0


def test_an_overflowing_ratio_is_capped_rather_than_becoming_inf():
    """exp(1000) overflows; an inf weight would poison the whole step."""
    weight = mismatch_weight(1000.0, -1000.0, cap=2.0)
    assert math.isfinite(weight)
    assert weight == pytest.approx(2.0)


def test_an_uncapped_weight_is_returned_untruncated():
    assert mismatch_weight(math.log(8.0), 0.0, cap=0.0) == pytest.approx(8.0)


# --------------------------------------------------------------------------- #
# 3. config contracts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("c", [0.5, 1.0])
def test_a_dual_clip_c_that_reads_enabled_but_is_inert_is_refused(c):
    """(0, 1] would state a bound the update does not apply."""
    with pytest.raises(ValueError, match="dual_clip_c"):
        GRPOConfig(dual_clip_c=c).validate()


@pytest.mark.parametrize("c", [0.0, -1.0, 1.01, 3.0, 10.0])
def test_a_disabled_or_real_dual_clip_c_is_accepted(c):
    assert GRPOConfig(dual_clip_c=c).validate() is None


def test_a_non_finite_dual_clip_c_is_refused():
    with pytest.raises(ValueError, match="dual_clip_c"):
        GRPOConfig(dual_clip_c=float("nan")).validate()


def test_mismatch_correction_requires_a_cap_above_one():
    """A cap at or below 1.0 clamps every weight into a uniform LR change."""
    with pytest.raises(ValueError, match="mismatch_weight_cap"):
        GRPOConfig(mismatch_correction=True, mismatch_weight_cap=1.0).validate()
    with pytest.raises(ValueError, match="mismatch_weight_cap"):
        GRPOConfig(mismatch_correction=True, mismatch_weight_cap=0.0).validate()
    assert GRPOConfig(mismatch_correction=True,
                      mismatch_weight_cap=2.0).validate() is None
    # Disabled, so the cap is not load-bearing and is not policed.
    assert GRPOConfig(mismatch_correction=False,
                      mismatch_weight_cap=0.0).validate() is None


def test_the_default_recipe_enables_the_dual_clip_floor():
    """It costs nothing when nothing is off-policy, and bounds the tail when it is."""
    assert GRPOConfig().dual_clip_c > 1.0


# --------------------------------------------------------------------------- #
# 4. the diagnostic
# --------------------------------------------------------------------------- #
def test_mismatch_stat_is_null_when_no_sample_carries_both_logprobs():
    """A null is honest; a 1.0 would assert the policies were measured to agree."""
    from kore.policy.grpo import _grpo_step_mismatch_stat

    samples = [[0.5, ("p", "g"), None, -1.0, 4, None, (0, 0)]]
    assert _grpo_step_mismatch_stat([samples]) == (None, None)


def test_mismatch_stat_reports_the_measured_divergence():
    from kore.policy.grpo import _grpo_step_mismatch_stat

    # Field 7 carries the training-side log-prob; field 3 the rollout-side one.
    samples = [
        [0.5, ("p", "g"), None, math.log(0.25), 4, None, (0, 0), math.log(0.5)],
        [0.5, ("p", "g"), None, math.log(0.5), 4, None, (1, 0), math.log(0.5)],
    ]
    mean, peak = _grpo_step_mismatch_stat([samples], cap=4.0)
    assert peak == pytest.approx(2.0)
    assert mean == pytest.approx(1.5)
