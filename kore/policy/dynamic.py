"""Adaptive training-horizon controller for GRPO.

A fixed step count is a guess: too few and the policy never moves off the
correct-but-slow plateau; too many and compute is burned after convergence. The
``DynamicStepController`` keeps training WHILE the monitored signal (rollout
reward mean, or held-out fast_p) is still improving and stops once it plateaus,
bounded by ``[min_steps, max_steps]``.

Pure and deterministic. In distributed training the monitored metric MUST be a
cross-rank-identical value (e.g. the gathered group reward mean) so every rank
takes the same stop decision in lockstep — otherwise ranks desynchronize.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DynamicStepController:
    min_steps: int
    max_steps: int
    patience: int
    min_delta: float = 1e-3
    best: float = -math.inf
    best_step: int = -1
    stopped_reason: str = ""

    def update(self, step: int, metric: float) -> bool:
        """Record ``metric`` at ``step`` (0-indexed) and return whether to STOP.

        Stop when: the hard cap ``max_steps`` is reached, OR (past ``min_steps``)
        the metric hasn't improved by > ``min_delta`` for ``patience`` steps.
        """
        if metric > self.best + self.min_delta:
            self.best = metric
            self.best_step = step

        done = step + 1  # steps completed
        if done >= self.max_steps:
            self.stopped_reason = f"reached max_steps={self.max_steps}"
            return True
        if done < self.min_steps:
            return False
        if self.best_step >= 0 and (step - self.best_step) >= self.patience:
            self.stopped_reason = (
                f"plateau: no >{self.min_delta} gain for {self.patience} steps "
                f"(best={self.best:.4f} @ step {self.best_step})")
            return True
        return False
