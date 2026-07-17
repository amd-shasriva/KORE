"""ε-budget accounting for the KORE transformation calculus.

A rewrite trajectory is a sequence of source-to-source transforms, each tagged
with a *relation* to the kernel it rewrites:

  * ``exact``  (``≡``)   - bit-preserving. The rewritten kernel is provably the
    same numeric function (scheduling / layout / occupancy knobs, added boundary
    masks, ensured-fp32 accumulation, ...). Composing exact with exact stays
    exact and never touches the budget.
  * ``approx`` (``≈_ε``) - a numerical *contract* rather than an identity: the
    output is guaranteed within a tolerance ``ε`` (re-tiling / K-split /
    reassociated reductions / downcast IO / fast reciprocal). Approx moves SPEND
    from a finite budget.

Two orthogonal quantities are tracked, and BOTH are exposed, because they answer
different questions:

  1. **Cumulative spend** (``spent`` / :meth:`ErrorBudget.remaining`): a
     conservative additive meter of how much numerical drift the trajectory has
     been *allowed* to introduce. This is what gates admissibility - an approx
     move whose ``ε`` no longer fits the remaining budget is inadmissible.
  2. **Composed relation** (:meth:`ErrorBudget.composed_relation` /
     :meth:`ErrorBudget.weakest_eps`): the *type the result carries*. Relation
     composition is a lattice join - ``exact ⊔ exact = exact``, anything with an
     approx step is ``approx`` - and the composed contract carries the WEAKEST
     (largest-``ε``, i.e. ``max``) guarantee of any step, since a chain of
     numeric contracts is only as strong as its loosest link.

This module is PURE (stdlib only) and imports nothing from KORE, so it is safe
to import anywhere (mirrors the dependency-free discipline of ``kore.policy.format``).
``ErrorBudget.admissible`` is duck-typed over the ``Transformation`` protocol
(it reads ``.relation`` and ``.epsilon(**params)``) so ``budget`` never has to
import ``calculus`` / ``library`` - keeping the dependency graph acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Relation algebra
# --------------------------------------------------------------------------- #
RELATION_EXACT = "exact"      # ≡  bit-preserving identity
RELATION_APPROX = "approx"    # ≈_ε numerical contract within tolerance ε
RELATIONS: tuple[str, ...] = (RELATION_EXACT, RELATION_APPROX)

# Numerical floor: ε deltas below this are treated as zero so float round-off in
# the additive meter can never spuriously reject an otherwise-affordable move.
_EPS_TOL = 1e-9


def compose_relation(r1: str, r2: str) -> str:
    """Join two relations: exact iff BOTH are exact, else approx.

    This is the lattice join with ``exact`` as bottom (strongest) and ``approx``
    as top (weakest): once a trajectory goes approximate it can never recover a
    bit-exact guarantee.
    """
    return RELATION_EXACT if (r1 == RELATION_EXACT and r2 == RELATION_EXACT) else RELATION_APPROX


def compose_eps(e1: float, e2: float) -> float:
    """Compose two step tolerances into the trajectory's carried ``ε``.

    A chain of ``≈_ε`` contracts is only as strong as its loosest link, so the
    result carries the WEAKEST (``max``) tolerance - not the sum. (The additive
    sum is metered separately as the spend; see the module docstring.)
    """
    return max(float(e1 or 0.0), float(e2 or 0.0))


# --------------------------------------------------------------------------- #
# Per-(operation, dtype) default budget table
# --------------------------------------------------------------------------- #
# Low-precision dtypes already tolerate more numerical error (their SNR gate is
# relaxed - see ``KoreConfig.snr_threshold_for``), so they get a LARGER ε budget:
# an approx rewrite that is invisible under bf16/fp8 rounding would be a real
# regression in fp32. The per-op scale reflects reduction depth / error
# sensitivity: a deep GEMM/attention reduction accumulates rounding, an
# elementwise op barely does (so it can spend more freely).
_DTYPE_TOLERANCE: dict[str, float] = {
    "fp32": 0.02,
    "tf32": 0.04,
    "fp16": 0.06,
    "bf16": 0.10,
    "fp8": 0.25,
    "int8": 0.25,
    "mxfp6": 0.30,
    "fp6": 0.30,
    "mxfp4": 0.35,
    "fp4": 0.35,
}
_OP_SCALE: dict[str, float] = {
    "gemm": 1.0,
    "matmul": 1.0,
    "conv": 1.0,
    "attention": 0.8,
    "reduction": 0.7,
    "norm": 0.9,
    "softmax": 0.8,
    "elementwise": 1.5,
}
# Fallbacks for out-of-vocabulary op / dtype.
_DEFAULT_DTYPE_TOL = 0.05
_DEFAULT_OP_SCALE = 1.0


def _norm_dtype(dtype: str) -> str:
    """Collapse a dtype string to a budget-table key (mirrors kore.value.features)."""
    d = (dtype or "").lower()
    if "bf16" in d or "bfloat16" in d:
        return "bf16"
    if "fp16" in d or "float16" in d or "half" in d:
        return "fp16"
    if "fp8" in d or "float8" in d:
        return "fp8"
    if "int8" in d or "i8" in d:
        return "int8"
    if "mxfp4" in d or "fp4" in d:
        return "mxfp4"
    if "mxfp6" in d or "fp6" in d:
        return "mxfp6"
    if "tf32" in d:
        return "tf32"
    if "fp32" in d or "float32" in d or "float" in d or d == "":
        return "fp32"
    return "fp32"


def _norm_op(operation: str) -> str:
    """Collapse an operation string to a budget-table key."""
    o = (operation or "").lower()
    for known in _OP_SCALE:
        if known in o:
            return known
    return ""


def default_budget(operation: str, dtype: str) -> float:
    """Default total ε budget for an ``(operation, dtype)`` task.

    ``budget = dtype_tolerance * op_scale`` - larger for low-precision dtypes and
    shallow reductions, tighter for fp32 and deep reductions. Returns a positive
    float; unknown op/dtype fall back to conservative defaults.
    """
    dt = _DTYPE_TOLERANCE.get(_norm_dtype(dtype), _DEFAULT_DTYPE_TOL)
    scale = _OP_SCALE.get(_norm_op(operation), _DEFAULT_OP_SCALE)
    return round(dt * scale, 4)


# Precomputed cross product, exposed as the documented default table. Built from
# the same formula as :func:`default_budget` so the two can never drift.
DEFAULT_BUDGET_TABLE: dict[tuple[str, str], float] = {
    (op, dt): round(tol * scale, 4)
    for op, scale in _OP_SCALE.items()
    for dt, tol in _DTYPE_TOLERANCE.items()
}


def _transform_eps(transform, **params) -> float:
    """ε a transform would spend, duck-typed over the Transformation protocol."""
    fn = getattr(transform, "epsilon", None)
    if callable(fn):
        return max(0.0, float(fn(**params)))
    return max(0.0, float(getattr(transform, "default_eps", 0.0)))


def _transform_relation(transform) -> str:
    return getattr(transform, "relation", RELATION_EXACT)


# --------------------------------------------------------------------------- #
# Budget tracker
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BudgetStep:
    """One recorded step of a trajectory's ε-accounting."""

    name: str
    relation: str
    eps: float


@dataclass
class ErrorBudget:
    """A finite ε budget consumed by approximate rewrites along a trajectory.

    Construct directly with a ``total`` or via :meth:`for_op` to pull the
    per-(op, dtype) default. Exact steps are recorded (for provenance) but never
    reduce the budget; approx steps :meth:`spend` from it and are :meth:`admissible`
    only while their ``ε`` still fits ``remaining()``.
    """

    total: float
    spent: float = 0.0
    steps: list[BudgetStep] = field(default_factory=list)

    # -- construction ----------------------------------------------------- #
    @classmethod
    def for_op(cls, operation: str, dtype: str, total: float | None = None) -> "ErrorBudget":
        """Budget seeded from the per-(op, dtype) default (override with ``total``)."""
        t = float(total) if total is not None else default_budget(operation, dtype)
        return cls(total=t)

    # -- queries ---------------------------------------------------------- #
    def remaining(self) -> float:
        """Unspent budget (never negative)."""
        return max(0.0, self.total - self.spent)

    def exhausted(self) -> bool:
        return self.remaining() <= _EPS_TOL

    def would_exceed(self, eps: float) -> bool:
        """True iff spending ``eps`` would overrun the remaining budget."""
        return float(eps) > self.remaining() + _EPS_TOL

    def admissible(self, transform, **params) -> bool:
        """Is ``transform`` legal to apply under the current budget?

        Exact transforms are ALWAYS admissible (they never touch the budget). An
        approx transform is admissible only while the budget is live and its
        ``ε`` still fits what remains - so approx moves are automatically blocked
        once the budget is exhausted.
        """
        if _transform_relation(transform) == RELATION_EXACT:
            return True
        if self.exhausted():
            return False
        eps = _transform_eps(transform, **params)
        # A zero-ε approx move is still gated on a live budget (it is a numeric
        # contract, not an identity), but costs nothing once admitted.
        if eps <= _EPS_TOL:
            return True
        return not self.would_exceed(eps)

    # -- mutation --------------------------------------------------------- #
    def spend(self, eps: float, name: str = "", relation: str = RELATION_APPROX) -> float:
        """Charge ``eps`` to the budget and record the step. Returns ``remaining()``.

        ``eps`` is clamped to be non-negative. Callers should gate on
        :meth:`admissible` first; ``spend`` itself does not refuse an overrun (it
        is the low-level meter), it just records it.
        """
        eps = max(0.0, float(eps))
        self.spent += eps
        self.steps.append(BudgetStep(name or "?", relation, eps))
        return self.remaining()

    def record_exact(self, name: str = "") -> None:
        """Record an exact step for provenance without touching the budget."""
        self.steps.append(BudgetStep(name or "?", RELATION_EXACT, 0.0))

    # -- composed contract ------------------------------------------------ #
    def composed_relation(self) -> str:
        """The relation the whole trajectory carries (approx iff any step was)."""
        rel = RELATION_EXACT
        for s in self.steps:
            rel = compose_relation(rel, s.relation)
        return rel

    def weakest_eps(self) -> float:
        """The trajectory's carried ``ε`` - the WEAKEST (max) step tolerance."""
        eps = 0.0
        for s in self.steps:
            if s.relation == RELATION_APPROX:
                eps = compose_eps(eps, s.eps)
        return eps

    def cumulative_eps(self) -> float:
        """Total ε spent (the additive meter, distinct from the composed max)."""
        return self.spent

    def state(self) -> dict:
        """A JSON-serializable snapshot for logging / the calculus return value."""
        return {
            "total": round(self.total, 6),
            "spent": round(self.spent, 6),
            "remaining": round(self.remaining(), 6),
            "relation": self.composed_relation(),
            "weakest_eps": round(self.weakest_eps(), 6),
            "cumulative_eps": round(self.cumulative_eps(), 6),
            "exhausted": self.exhausted(),
            "n_steps": len(self.steps),
            "n_approx": sum(1 for s in self.steps if s.relation == RELATION_APPROX),
            "steps": [
                {"name": s.name, "relation": s.relation, "eps": round(s.eps, 6)}
                for s in self.steps
            ],
        }
