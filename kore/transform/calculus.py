"""The verified, ε-typed transformation calculus (KORE P2 paradigm).

This is the core of the transform package: a :class:`Transformation` abstraction
and the two calculus operations the RL loop consumes:

  * :func:`apply_sequence` - apply a trajectory of ``(transform, params)`` steps
    to a Triton kernel source, enforcing (a) each transform's *side conditions*
    (admissibility of its params), (b) the ε-*budget* for approximate moves, and
    (c) actual applicability (a transform returns ``None`` when its pattern is
    absent). It returns the rewritten source plus a full audit trail.
  * :func:`admissible_actions` - the currently-legal set of ``(transform, params)``
    moves for a given source and budget. This is the RL ACTION SPACE: it shrinks
    as the ε-budget is spent (approx moves drop out once they no longer fit),
    exactly the "verified action space" the P2 paradigm calls for.

A :class:`Transformation` is a pure, deterministic source->source rewrite tagged
with its relation (``exact`` ≡ or ``approx`` ≈_ε), the numeric knob / token it
edits (metadata mirroring ``kore.policy.format`` / ``kore.value.features``), a
side-condition predicate, an ε-cost model, and a small candidate-parameter grid
used to enumerate the action space.

Everything here is PURE (stdlib only). The concrete library of transforms lives
in ``kore.transform.library`` and is imported lazily inside the two engine
functions so the dependency graph stays acyclic (library -> calculus -> budget).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from kore.transform.budget import (
    RELATION_APPROX,
    RELATION_EXACT,
    ErrorBudget,
)

# A step is (transform-or-name, params). ``params`` may be omitted -> {}.
Step = tuple[Any, dict]


@dataclass(frozen=True)
class Transformation:
    """A single verified source->source rewrite of a Triton kernel.

    Fields
      name       stable identifier (the action name in the RL action space).
      relation   ``exact`` (≡, bit-preserving) or ``approx`` (≈_ε, numeric contract).
      knob       the numeric knob / token this edits, in the vocabulary of
                 ``kore.policy.format`` and ``kore.value.features`` (e.g.
                 ``num_warps``, ``block``, ``group_m``, ``vectorize``,
                 ``accumulator``, ``dtype``, ``reduction``, ``split_k``).
      summary    one-line human description.
      apply_fn   ``(src, **params) -> str | None`` - the rewrite, or ``None`` when
                 it is structurally inapplicable to ``src``.
      cond_fn    ``(src, **params) -> list[str]`` - side-condition violations;
                 empty list == admissible params. ``None`` means no conditions.
      eps_fn     ``(**params) -> float`` - ε cost of an approx move (param
                 dependent). Ignored for exact transforms (their ε is 0).
      grid_fn    ``(src) -> list[dict]`` - a small candidate-parameter grid used
                 to enumerate the action space. ``None`` -> a single ``{}`` action.
      default_eps  ε for an approx transform with no ``eps_fn``.
    """

    name: str
    relation: str
    knob: str
    summary: str
    apply_fn: Callable[..., Optional[str]]
    cond_fn: Optional[Callable[..., list[str]]] = None
    eps_fn: Optional[Callable[..., float]] = None
    grid_fn: Optional[Callable[[str], list[dict]]] = None
    default_eps: float = 0.0

    # -- relation helpers ------------------------------------------------- #
    def is_exact(self) -> bool:
        return self.relation == RELATION_EXACT

    def is_approx(self) -> bool:
        return self.relation == RELATION_APPROX

    # -- core behavior ---------------------------------------------------- #
    def apply(self, src: str, **params) -> Optional[str]:
        """Rewrite ``src``; ``None`` when structurally inapplicable. Never raises."""
        try:
            return self.apply_fn(src, **params)
        except Exception:  # a rewrite must never crash the calculus / RL loop
            return None

    def side_conditions(self, src: str, **params) -> list[str]:
        """Param/precondition violations; empty == admissible. Never raises."""
        if self.cond_fn is None:
            return []
        try:
            return list(self.cond_fn(src, **params))
        except Exception as e:  # a broken predicate rejects rather than crashes
            return [f"side-condition error: {e}"]

    def epsilon(self, **params) -> float:
        """ε this move would spend (0 for exact transforms)."""
        if self.is_exact():
            return 0.0
        if self.eps_fn is not None:
            try:
                return max(0.0, float(self.eps_fn(**params)))
            except Exception:
                return max(0.0, float(self.default_eps))
        return max(0.0, float(self.default_eps))

    def candidate_params(self, src: str) -> list[dict]:
        """Small grid of candidate params for action-space enumeration."""
        if self.grid_fn is None:
            return [{}]
        try:
            grid = list(self.grid_fn(src))
        except Exception:
            return [{}]
        return grid or [{}]

    def admissible(self, src: str, budget: Optional[ErrorBudget] = None, **params) -> bool:
        """True iff params pass side-conditions, the move actually changes ``src``,
        and (for approx) the ε-budget can afford it."""
        if self.side_conditions(src, **params):
            return False
        if budget is not None and self.is_approx() and not budget.admissible(self, **params):
            return False
        new = self.apply(src, **params)
        return new is not None and new != src

    def as_metadata(self) -> dict:
        """Static descriptor (menu row) - no source required."""
        return {
            "name": self.name,
            "relation": self.relation,
            "knob": self.knob,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class Action:
    """One legal move in the RL action space (a transform + concrete params)."""

    name: str
    params: dict
    relation: str
    eps: float
    knob: str

    def as_step(self) -> Step:
        return (self.name, dict(self.params))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "params": dict(self.params),
            "relation": self.relation,
            "eps": round(self.eps, 6),
            "knob": self.knob,
        }


# --------------------------------------------------------------------------- #
# Library resolution (lazy, to keep library -> calculus -> budget acyclic)
# --------------------------------------------------------------------------- #
def _resolve_library(library: Optional[Sequence[Transformation]]) -> list[Transformation]:
    if library is not None:
        return list(library)
    from kore.transform.library import LIBRARY  # lazy import
    return list(LIBRARY)


def _index(library: Sequence[Transformation]) -> dict[str, Transformation]:
    return {t.name: t for t in library}


def _resolve_transform(
    ref: Any, index: dict[str, Transformation]
) -> Optional[Transformation]:
    if isinstance(ref, Transformation):
        return ref
    if isinstance(ref, str):
        return index.get(ref)
    return None


def _split_step(entry: Any) -> Step:
    """Normalize a trajectory entry into ``(transform_ref, params_dict)``.

    Accepts ``(ref, params)``, a bare ``ref``, or ``{"name"/"transform", "params"}``.
    """
    if isinstance(entry, dict):
        ref = entry.get("transform", entry.get("name"))
        params = entry.get("params", {}) or {}
        return ref, dict(params)
    if isinstance(entry, (tuple, list)):
        if len(entry) == 2:
            ref, params = entry
            return ref, dict(params or {})
        if len(entry) == 1:
            return entry[0], {}
    return entry, {}


# --------------------------------------------------------------------------- #
# apply_sequence - execute a rewrite trajectory with verification + budget
# --------------------------------------------------------------------------- #
def apply_sequence(
    src: str,
    steps: Iterable[Step],
    budget: ErrorBudget,
    library: Optional[Sequence[Transformation]] = None,
) -> tuple[str, list[dict], list[dict], dict]:
    """Apply a trajectory of transforms to ``src`` under an ε-budget.

    Each step is gated, in order, by:
      1. **side conditions** - illegal params (e.g. a non-64-multiple BLOCK) are
         rejected without touching the source or budget;
      2. **budget** - an approx move whose ε no longer fits ``budget.remaining()``
         is rejected (the ``≈_ε`` contract would overrun the tolerance);
      3. **applicability** - a transform that returns ``None`` (its pattern is
         absent) or that leaves the source unchanged is rejected.
    A step that passes all three is committed: the source is rewritten and, for an
    approx move, its ε is spent (exact moves are recorded but cost nothing).

    Returns ``(new_src, applied, rejected, budget_state)`` where ``applied`` /
    ``rejected`` are audit records and ``budget_state`` is ``budget.state()`` -
    which carries the composed relation and the weakest (max-ε) trajectory
    contract. Pure apart from mutating the passed-in ``budget``.
    """
    lib = _resolve_library(library)
    index = _index(lib)

    cur = src
    applied: list[dict] = []
    rejected: list[dict] = []

    for entry in steps:
        ref, params = _split_step(entry)
        tf = _resolve_transform(ref, index)
        name = tf.name if tf is not None else (ref if isinstance(ref, str) else str(ref))

        if tf is None:
            rejected.append({"name": name, "params": params,
                             "reason": "unknown_transform"})
            continue

        violations = tf.side_conditions(cur, **params)
        if violations:
            rejected.append({"name": name, "params": params,
                             "relation": tf.relation, "reason": "side_condition",
                             "violations": violations})
            continue

        eps = tf.epsilon(**params)
        if tf.is_approx() and not budget.admissible(tf, **params):
            rejected.append({"name": name, "params": params,
                             "relation": tf.relation, "reason": "budget_exhausted",
                             "eps": round(eps, 6),
                             "remaining": round(budget.remaining(), 6)})
            continue

        new = tf.apply(cur, **params)
        if new is None or new == cur:
            rejected.append({"name": name, "params": params,
                             "relation": tf.relation, "reason": "inapplicable"})
            continue

        # Commit: rewrite + account. Approx spends ε; exact is recorded free.
        if tf.is_approx():
            budget.spend(eps, name=tf.name, relation=RELATION_APPROX)
        else:
            budget.record_exact(tf.name)
        cur = new
        applied.append({"name": name, "params": params, "relation": tf.relation,
                        "knob": tf.knob, "eps": round(eps, 6),
                        "remaining": round(budget.remaining(), 6)})

    return cur, applied, rejected, budget.state()


# --------------------------------------------------------------------------- #
# admissible_actions - the (budget-constrained) RL action space
# --------------------------------------------------------------------------- #
def admissible_actions(
    src: str,
    budget: ErrorBudget,
    library: Optional[Sequence[Transformation]] = None,
) -> list[Action]:
    """The currently-legal transformation set for ``src`` under ``budget``.

    An action ``(transform, params)`` is legal iff its params pass the
    transform's side-conditions, the transform actually changes ``src`` (its
    pattern is present and the params are not a no-op), and - for approx moves -
    its ε still fits the remaining budget. Exact moves are always budget-legal.

    This is the RL action space and it is *monotone in the budget*: spending ε can
    only remove approx actions, never add any, so the action set shrinks as the
    trajectory consumes its numerical tolerance.
    """
    lib = _resolve_library(library)
    actions: list[Action] = []
    for tf in lib:
        for params in tf.candidate_params(src):
            if tf.side_conditions(src, **params):
                continue
            if tf.is_approx() and not budget.admissible(tf, **params):
                continue
            new = tf.apply(src, **params)
            if new is None or new == src:
                continue
            actions.append(Action(
                name=tf.name, params=dict(params), relation=tf.relation,
                eps=tf.epsilon(**params), knob=tf.knob,
            ))
    return actions


def action_menu(library: Optional[Sequence[Transformation]] = None) -> list[dict]:
    """The static transformation menu (name, relation, knob, summary) - no source."""
    return [t.as_metadata() for t in _resolve_library(library)]
