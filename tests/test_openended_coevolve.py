"""CPU tests for the headroom-regret signal the co-evolution curriculum ranks on.

The standalone propose/attempt/update/distill loop that used to live in
``kore.openended.coevolve`` was removed as unreachable duplication: the loop that
actually runs is :class:`kore.openended.controller.CoevolutionController`, which
multi-turn GRPO drives. ``_headroom_regret`` is the piece the live controller
imports (``controller.py`` -> ``DescriptorStats(headroom_regret=...)``), so it is
what this module pins.
"""

from __future__ import annotations

import pytest

from kore.openended.coevolve import _headroom_regret


def test_headroom_regret_bounds():
    assert _headroom_regret(None) == 1.0
    assert _headroom_regret(1.5) == 0.0
    assert _headroom_regret(1.0) == 0.0
    assert _headroom_regret(0.4) == pytest.approx(0.6)


@pytest.mark.parametrize("speedup", [-5.0, 0.0, 0.25, 0.5, 0.999, 1.0, 2.0, 1e9])
def test_headroom_regret_is_always_a_bounded_probability(speedup):
    """The proposer multiplies this into a learnability score, so it must stay in
    [0, 1] for every speedup the perf oracle can hand back (incl. nonsense)."""
    r = _headroom_regret(speedup)
    assert 0.0 <= r <= 1.0


def test_headroom_regret_is_monotone_non_increasing_in_speedup():
    """More speedup => strictly no more regret: the ordering the curriculum needs."""
    grid = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 3.0]
    regrets = [_headroom_regret(s) for s in grid]
    assert regrets == sorted(regrets, reverse=True)
    # an unmeasured / failed attempt carries the MAXIMUM regret (nothing learned yet)
    assert _headroom_regret(None) >= max(regrets)


def test_controller_consumes_headroom_regret():
    """Guard the live wiring: the in-training controller is the real caller."""
    import inspect

    from kore.openended import controller

    assert controller._headroom_regret is _headroom_regret
    assert "_headroom_regret(" in inspect.getsource(controller)
