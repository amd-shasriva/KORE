"""CPU-only tests for the adaptive GRPO step controller + overlong masking."""

from __future__ import annotations

from kore.policy.dynamic import DynamicStepController
from kore.policy.grpo import is_overlong


# --- overlong (DAPO) filtering -------------------------------------------- #
def test_is_overlong_flags_truncated_responses():
    assert is_overlong(16384, 16384, 512) is True           # at the cap
    assert is_overlong(16000, 16384, 512) is True           # within buffer
    assert is_overlong(15871, 16384, 512) is False          # just outside buffer
    assert is_overlong(100, 16384, 512) is False            # short response


def test_is_overlong_disabled_when_cap_unset():
    assert is_overlong(99999, 0, 512) is False
    assert is_overlong(99999, -1, 512) is False


# --- adaptive step controller --------------------------------------------- #
def test_never_stops_before_min_steps():
    c = DynamicStepController(min_steps=10, max_steps=1000, patience=3)
    for step in range(9):
        assert c.update(step, metric=0.0) is False          # flat, but < min_steps


def test_stops_on_plateau_after_min_steps():
    c = DynamicStepController(min_steps=5, max_steps=1000, patience=3, min_delta=1e-3)
    # climb for a while (each step improves)
    stop = False
    for step in range(10):
        stop = c.update(step, metric=float(step))
        assert stop is False
    # now plateau: no improvement -> stop after `patience` steps past the best
    results = [c.update(step, metric=9.0) for step in range(10, 20)]
    assert any(results)
    assert "plateau" in c.stopped_reason


def test_hard_cap_at_max_steps():
    c = DynamicStepController(min_steps=1, max_steps=5, patience=100)
    stops = [c.update(step, metric=float(step)) for step in range(5)]  # always improving
    assert stops[-1] is True and "max_steps" in c.stopped_reason
    assert stops[:-1] == [False, False, False, False]


def test_improvement_resets_patience():
    c = DynamicStepController(min_steps=1, max_steps=1000, patience=3, min_delta=1e-3)
    # improve, stall 2, improve again, stall 2 -> should NOT stop (patience reset)
    seq = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    stops = [c.update(step, m) for step, m in enumerate(seq)]
    assert not any(stops)
