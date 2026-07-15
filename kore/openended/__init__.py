"""Open-ended co-evolution machinery for KORE (the "task frontier" half).

A verifiably-grounded open-ended kernel-discovery paradigm where the *task
distribution* co-evolves with the policy:

  * :mod:`kore.openended.task_space` - a PARAMETRIC, verifiable task descriptor +
    space built directly on the existing KORE parametric op registries
    (``kore.tasks._genops`` + ``kore.tasks.vendor_ops``). Each descriptor is a
    concrete, gradable task; ``descriptor_features`` gives MAP-Elites behavior
    dimensions and ``descriptor_key`` the archive niche.

  * :mod:`kore.openended.proposer` - a learnability/regret-targeted task PROPOSER
    (UED/PLR): score candidate tasks by learnability ``p*(1-p)`` combined with
    performance-headroom regret and novelty vs the archive, with guardrails
    against collapse (no unsolvable / trivial tasks; enforced diversity).

  * :mod:`kore.openended.archive` - a MAP-Elites TASK archive keeping the most
    *informative* task per behavior niche + its outcome history.

Everything here is pure and CPU-only (torch is only imported lazily, inside
functions, when the underlying op registries need it), so the whole task
frontier is unit-testable without a GPU. The GRPO-facing adapter
(:class:`kore.openended.controller.CoevolutionController`) wires this frontier
into multi-turn GRPO when ``coevolve`` is enabled (see
``configs/grpo_14b_full.json``); the standalone :mod:`kore.openended.coevolve`
loop stays runner-agnostic for offline study.
"""

from __future__ import annotations
