"""Headroom-regret signal for the open-ended co-evolution curriculum.

The co-evolution loop itself lives in :class:`kore.openended.controller.
CoevolutionController`, which multi-turn GRPO drives directly when ``coevolve`` is
enabled. This module holds the one piece that loop is built on: the CONTINUOUS
performance-headroom regret derived from a measured speedup.

Why kernels make open-endedness actually work (the paradigm claim): the task space
is infinitely + cheaply generatable, every task is ground-truth VERIFIABLE, and
carries a CONTINUOUS performance-headroom regret signal - which simultaneously
solves UED's regret-estimation problem and the unverifiable-reward problem that
block open-ended RL in other domains. ``_headroom_regret`` is that signal: it
converts a verified speedup measurement into the learnability score the proposer
(:mod:`kore.openended.proposer`) and the MAP-Elites archive
(:mod:`kore.openended.archive`) rank candidate tasks by.
"""

from __future__ import annotations

from typing import Optional


def _headroom_regret(speedup: Optional[float]) -> float:
    """Regret = how far below the 'genuinely beats baseline' bar (1x) the best
    attempt is, in [0,1]. Correct-but-slow kernels carry the most learnable regret;
    already-fast (>=1x) kernels carry ~0 (little left to learn)."""
    if speedup is None:
        return 1.0
    if speedup >= 1.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - speedup))
