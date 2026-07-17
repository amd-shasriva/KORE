"""KORE verified, ε-typed transformation calculus (the P2 paradigm component).

A small, PURE (CPU-only, stdlib) package that turns Triton-kernel optimization
into a *verified action space*: a library of source->source rewrites, each typed
by its relation to the kernel it edits -

    exact  (≡)   bit-preserving  (scheduling / layout / occupancy / masks / fp32 acc)
    approx (≈_ε) numeric contract within tolerance ε  (re-tiling / split-K /
                 downcast / reassociated reductions / fast reciprocal)

- and an :class:`ErrorBudget` that meters how much numerical drift a rewrite
trajectory may introduce. The calculus composes relations (``exact ⊔ exact =
exact``; any approx ⇒ approx) and carries the WEAKEST (max-ε) contract, while the
budget blocks approx moves once its tolerance is exhausted.

Public API
  - :class:`Transformation`  - one typed rewrite (``apply`` / ``side_conditions`` /
    ``epsilon`` / metadata).
  - :data:`LIBRARY` (+ :data:`EXACT`, :data:`APPROX`, :func:`get`) - the >=12
    transforms.
  - :class:`ErrorBudget` (+ :func:`default_budget`, :data:`DEFAULT_BUDGET_TABLE`) -
    per-(op, dtype) ε accounting.
  - :func:`apply_sequence` - run a rewrite trajectory with verification + budget.
  - :func:`admissible_actions` - the currently-legal move set (the RL action space).
  - :func:`action_menu` - the static transformation menu.

Typical use as an RL action space::

    from kore.transform import ErrorBudget, admissible_actions, apply_sequence
    budget = ErrorBudget.for_op(task.operation, task.dtype)
    actions = admissible_actions(kernel_src, budget)          # legal moves now
    a = actions[i]                                            # policy picks one
    new_src, applied, rejected, state = apply_sequence(
        kernel_src, [a.as_step()], budget)                   # apply + account
"""

from __future__ import annotations

from kore.transform.budget import (
    DEFAULT_BUDGET_TABLE,
    RELATION_APPROX,
    RELATION_EXACT,
    RELATIONS,
    BudgetStep,
    ErrorBudget,
    compose_eps,
    compose_relation,
    default_budget,
)
from kore.transform.calculus import (
    Action,
    Step,
    Transformation,
    action_menu,
    admissible_actions,
    apply_sequence,
)
from kore.transform.library import (
    APPROX,
    BY_NAME,
    EXACT,
    LIBRARY,
    default_library,
    get,
)

__all__ = [
    # relations / budget
    "RELATION_EXACT",
    "RELATION_APPROX",
    "RELATIONS",
    "compose_relation",
    "compose_eps",
    "ErrorBudget",
    "BudgetStep",
    "default_budget",
    "DEFAULT_BUDGET_TABLE",
    # calculus
    "Transformation",
    "Action",
    "Step",
    "apply_sequence",
    "admissible_actions",
    "action_menu",
    # library
    "LIBRARY",
    "EXACT",
    "APPROX",
    "BY_NAME",
    "get",
    "default_library",
]
