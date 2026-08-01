"""Production wiring for the multi-pronged correctness oracle.

:mod:`kore.verify.equivalence` describes a four-prong oracle; the *shipped*
correctness path (the task driver in ``kore.tasks._genops`` plus
:class:`kore.env.kore_env.KoreEnv`) historically ran only three of them - many
reseeded random trials, an enumerated adversarial battery, and a determinism
re-check. This module is the bridge that puts the **metamorphic** prong on the
production path and makes the resulting claim auditable:

* :func:`metamorphic_plan_for_task` decides, from the *versioned task taxonomy*
  (``kore.tasks.taxonomy`` is the family authority and is only read here),
  whether a task's operator family has a metamorphic identity that is
  **provably** true of the true operator. Anything not proven is
  ``applicable=False`` - a relation that merely *probably* holds would
  false-reject honest kernels, which is far worse than not running it.
* :data:`RUNNER_SHIM_SOURCE` + :mod:`kore.verify.runner` execute the relations
  against the candidate on the GPU, in the environment's own staged workdir.
* :func:`parse_metamorphic_report` / :class:`OracleReport` carry the evidence
  back, including the honest ``(1-p)**m`` false-accept bound.

Everything here is pure CPU/stdlib (no torch, no numpy at import time), so the
mapping and reporting logic is unit-testable without a GPU.

Why the metamorphic prong is cheap
----------------------------------
Every relation is a *candidate-only self-consistency* check: ``f(P.x)`` versus
``P.f(x)`` needs no second oracle evaluation, so the prong costs candidate
launches on a small shape rather than reference work. It nonetheless catches a
class the random prong structurally cannot: a "pointwise" kernel that secretly
mixes elements across a row agrees with the oracle on every sampled *value* only
if it is also right about *locality*, and the random prong never varies locality.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from kore.verify.equivalence import false_accept_probability
from kore.verify.metamorphic import metamorphic_relations

__all__ = [
    "METAMORPHIC_PROTOCOL",
    "METAMORPHIC_MARKER",
    "GENOPS_METAMORPHIC_OP_CLASS",
    "METAMORPHIC_DTYPES",
    "DEFAULT_MAX_ELEMENTS",
    "MetamorphicPlan",
    "ProngStatus",
    "OracleReport",
    "PRONG_STATES",
    "metamorphic_plan_for_task",
    "select_metamorphic_shape",
    "expected_output_elements",
    "op_class_for_reference",
    "task_output_op_class",
    "build_oracle_report",
    "generic_adversarial_families",
    "driver_random_trials",
    "parse_metamorphic_report",
    "format_metamorphic_report",
    "sanitize_detail",
    "shape_spec",
    "parse_shape_spec",
    "RUNNER_SHIM_NAME",
    "RUNNER_SHIM_SOURCE",
]


# --------------------------------------------------------------------------- #
# Wire protocol between the environment and the GPU-side runner
# --------------------------------------------------------------------------- #
METAMORPHIC_PROTOCOL = "kore-metamorphic-v1"
METAMORPHIC_MARKER = "KORE_METAMORPHIC:"
_METAMORPHIC_LINE = re.compile(r"^KORE_METAMORPHIC:\s*(\{[^\n]+\})\s*$", re.MULTILINE)

# The runner's *decision* is carried by the same ``SNR:`` / ``allclose:``
# literals the driver uses, deliberately: ``kore.reward.scan_for_hacks`` already
# rejects any candidate source that prints those literals, and the environment
# already takes the LAST match (which is the runner's, printed after the
# candidate has finished running). The JSON line below is DIAGNOSTIC ONLY - it
# is not forgery-protected, so it may enrich a report but must never be the
# thing that turns a rejection into an acceptance.
DEFAULT_MAX_ELEMENTS = 1 << 22  # 4Mi elements: bounds the CPU-side f64 compare


# --------------------------------------------------------------------------- #
# Which operator families have a PROVEN metamorphic identity
# --------------------------------------------------------------------------- #
# ``kore.tasks.taxonomy`` owns the family names; this table owns the (much
# stronger) claim that a family's *contract* implies a specific algebraic
# identity. It is deliberately restricted to the ``kore.tasks._genops``
# generator source families, because there the operator semantics are fixed by
# the generator's own spec rather than inferred from a name:
#
#   unary   f(x)        [M,N] -> [M,N], applied independently per element
#   binary  f(x, y)     [M,N] x [M,N] -> [M,N], per element
#   fusion  f(a, b[,c]) all [M,N] -> [M,N], a pointwise chain per element
#     => permutation equivariance (rows and columns), block locality, and
#        reshape invariance all hold exactly for the true operator.
#
#   reduce  g(x)        [M,N] -> [M], an order-invariant per-row reduce
#                       (sum/mean/max/min/l1/l2/rms/max_abs)
#     => column-permutation invariance, row-permutation equivariance, and row
#        locality all hold exactly for the true operator.
#
# ``gemm_fusion`` is EXCLUDED on purpose: a K-contraction is neither elementwise
# nor an order-invariant row reduction, and the generic relations permute every
# operand along the same axis, which for ``A[M,K] @ B[K,N]`` permutes A's rows
# together with B's rows and is simply not an identity of matmul.
GENOPS_METAMORPHIC_OP_CLASS: Mapping[str, str] = MappingProxyType({
    "unary": "elementwise",
    "binary": "elementwise",
    "fusion": "elementwise",
    "reduce": "reduction",
})

# Why an otherwise-known genops family is still skipped (reported verbatim).
GENOPS_METAMORPHIC_EXCLUSIONS: Mapping[str, str] = MappingProxyType({
    "gemm_fusion": (
        "a K-contraction is neither elementwise nor an order-invariant row "
        "reduction; the generic relations are not identities of matmul"
    ),
})

# Storage dtypes with a calibrated :class:`kore.verify.equivalence.Tolerance`.
# int8/fp8 generated tasks carry quantization structure (scales, packing) that
# the plain float relations would break, so they are never planned.
METAMORPHIC_DTYPES: frozenset[str] = frozenset({"fp32", "fp16", "bf16"})

# Relations need at least two rows to split/permute, and two columns to permute.
_MIN_ROWS = 2
_MIN_COLS = 2

# Mirrors ``kore.tasks._genops._GENERIC_ADV_FAMILIES`` for REPORTING only (never
# for a verdict), so a driver-side rename can at worst make a report vaguer.
_GENERIC_ADV_FAMILIES_FALLBACK = ("unary", "binary", "reduce", "fusion", "gemm_fusion")


def generic_adversarial_families() -> tuple[str, ...]:
    """Families for which the driver builds its generic adversarial fills."""
    try:  # read-only: the driver owns this list
        from kore.tasks._genops import _GENERIC_ADV_FAMILIES  # noqa: PLC0415

        return tuple(str(f) for f in _GENERIC_ADV_FAMILIES)
    except Exception:  # noqa: BLE001 - reporting must never break an evaluation
        return _GENERIC_ADV_FAMILIES_FALLBACK


def driver_random_trials() -> int:
    """Reseeded random trials the driver runs per shape (mirrors the driver)."""
    try:  # read-only: keeps the reported bound honest if the driver's floor moves
        from kore.tasks._genops import _num_correct_trials  # noqa: PLC0415

        return max(1, int(_num_correct_trials()))
    except Exception:  # noqa: BLE001
        try:
            return max(5, int(os.environ.get("KORE_CORRECTNESS_TRIALS", "5")))
        except ValueError:
            return 5


@dataclass(frozen=True)
class MetamorphicPlan:
    """Whether (and how) the metamorphic prong may run for one task."""

    applicable: bool
    reason: str
    op_class: str = "generic"
    source_family: str = ""
    product_family: str = ""
    dtype: str = ""
    relations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "reason": self.reason,
            "op_class": self.op_class,
            "source_family": self.source_family,
            "product_family": self.product_family,
            "dtype": self.dtype,
            "relations": list(self.relations),
        }


def _not_applicable(reason: str, **kw: Any) -> MetamorphicPlan:
    return MetamorphicPlan(applicable=False, reason=reason, **kw)


def metamorphic_plan_for_task(task: Any) -> MetamorphicPlan:
    """Decide the metamorphic op class for ``task`` (fail-closed).

    Returns ``applicable=False`` with an explicit ``reason`` whenever the task's
    operator contract does not *prove* the relation set - an unproven relation
    would false-reject honest kernels, so "unknown" must mean "do not run",
    never "assume elementwise". The task's family is taken from the versioned
    taxonomy authority (:mod:`kore.tasks.taxonomy`, read-only): the generator
    source family in ``task.yaml`` must classify cleanly *and* agree with the
    canonical product leaf the taxonomy assigns it, so a taxonomy change
    silently disables the prong rather than silently changing its meaning.
    """
    task_id = str(getattr(task, "task_id", "") or "")
    raw = getattr(task, "raw", None)
    if not isinstance(raw, Mapping):
        return _not_applicable("task has no task.yaml metadata mapping")
    if not raw.get("generated"):
        return _not_applicable(
            "hand-authored task: operator semantics are not fixed by a generator "
            "spec, so no structural identity is proven")
    if not task_id.startswith("gen_"):
        return _not_applicable(
            f"task {task_id!r} is not a kore.tasks._genops (gen_*) task; only the "
            "genops elementwise/reduction families have proven relations")

    source_family = str(raw.get("op_family") or "").strip().lower()
    if not source_family:
        return _not_applicable("task.yaml declares no op_family")
    if source_family in GENOPS_METAMORPHIC_EXCLUSIONS:
        return _not_applicable(
            f"family {source_family!r} excluded: "
            f"{GENOPS_METAMORPHIC_EXCLUSIONS[source_family]}",
            source_family=source_family)
    op_class = GENOPS_METAMORPHIC_OP_CLASS.get(source_family)
    if op_class is None:
        return _not_applicable(
            f"family {source_family!r} has no proven metamorphic identity",
            source_family=source_family)

    # The taxonomy is the family authority: classify through it, and require the
    # canonical leaf it assigns to match what this family is supposed to be.
    try:
        from kore.tasks.taxonomy import (  # noqa: PLC0415 - read-only authority
            GENOPS_OPERATION_OVERRIDES,
            GENOPS_SOURCE_FAMILIES,
            product_family_for_task,
        )

        product_family = product_family_for_task(task, strict=True)
        operation = str(getattr(task, "operation", "") or "").strip().lower()
        expected = GENOPS_OPERATION_OVERRIDES.get(
            operation, GENOPS_SOURCE_FAMILIES.get(source_family))
    except Exception as exc:  # noqa: BLE001 - unclassifiable => do not run
        return _not_applicable(
            f"taxonomy could not classify the task ({type(exc).__name__}): {exc}",
            source_family=source_family)
    if product_family is None or product_family != expected:
        return _not_applicable(
            f"taxonomy leaf {product_family!r} does not match the {source_family!r} "
            f"generator contract (expected {expected!r})",
            source_family=source_family)

    dtype = str(getattr(task, "dtype", "") or "").strip().lower()
    if dtype not in METAMORPHIC_DTYPES:
        return _not_applicable(
            f"dtype {dtype!r} is outside the calibrated float tolerances "
            f"{sorted(METAMORPHIC_DTYPES)}",
            source_family=source_family, product_family=str(product_family))

    relations = tuple(rel.name for rel in metamorphic_relations(op_class))
    if not relations:
        return _not_applicable(
            f"op class {op_class!r} declares no relations",
            source_family=source_family, product_family=str(product_family),
            dtype=dtype)
    return MetamorphicPlan(
        applicable=True,
        reason=(f"genops {source_family!r} -> op_class {op_class!r} "
                f"({len(relations)} relations)"),
        op_class=op_class,
        source_family=source_family,
        product_family=str(product_family),
        dtype=dtype,
        relations=relations,
    )


# --------------------------------------------------------------------------- #
# Shape selection (cheap, bounded, explicit)
# --------------------------------------------------------------------------- #
def shape_spec(dims: Mapping[str, int]) -> str:
    """Render dims the way ``Shape.as_args`` / the driver's parser expect."""
    return ",".join(f"{k}={int(v)}" for k, v in dims.items()) if dims else "default"


def parse_shape_spec(spec: str) -> dict[str, int]:
    """Inverse of :func:`shape_spec` (mirrors ``_genops._parse_shape``)."""
    if not spec or spec == "default":
        return {"M": 4096, "N": 8192}
    out: dict[str, int] = {}
    for kv in spec.split(","):
        key, _, value = kv.partition("=")
        out[key.strip()] = int(value)
    return out


def expected_output_elements(op_class: str, dims: Mapping[str, int]) -> Optional[int]:
    """Elements in one reference output for ``op_class`` at ``dims``.

    Known exactly for the planned families (elementwise preserves ``[M, N]``; a
    row reduction yields ``[M]``) and ``None`` otherwise, so a bound is reported
    only where the comparison count is a fact rather than a guess.
    """
    try:
        m = int(dims["M"])
        n = int(dims["N"])
    except (KeyError, TypeError, ValueError):
        return None
    if m <= 0 or n <= 0:
        return None
    if op_class == "elementwise":
        return m * n
    if op_class == "reduction":
        return m
    return None


# Output extent per genops source family. This is wider than
# :data:`GENOPS_METAMORPHIC_OP_CLASS` on purpose: a GEMM epilogue has no usable
# metamorphic identity but its output extent is still exactly ``[M, N]``, so the
# false-accept bound can be reported honestly for it too.
_GENOPS_OUTPUT_OP_CLASS: Mapping[str, str] = MappingProxyType({
    "unary": "elementwise",
    "binary": "elementwise",
    "fusion": "elementwise",
    "gemm_fusion": "elementwise",
    "reduce": "reduction",
})


def op_class_for_reference(ref: Any) -> str:
    """Metamorphic op class implied by a loaded ``reference.py``, else generic.

    The environment plans from ``task.yaml``; this is the same decision taken
    from the reference module instead, for a caller that already has it loaded
    (the task driver). It is deliberately the SAME table, so the two agree by
    construction, and it is fail-closed for anything it does not recognize.
    """
    family = str(getattr(ref, "family", "") or "").strip().lower()
    dtype = str(getattr(ref, "dtype_name", "") or "").strip().lower()
    if dtype and dtype not in METAMORPHIC_DTYPES:
        return "generic"
    return GENOPS_METAMORPHIC_OP_CLASS.get(family, "generic")


def task_output_op_class(task: Any) -> str:
    """Op class describing a task's OUTPUT EXTENT (for the bound), else generic."""
    raw = getattr(task, "raw", None)
    if not isinstance(raw, Mapping) or not raw.get("generated"):
        return "generic"
    if not str(getattr(task, "task_id", "") or "").startswith("gen_"):
        return "generic"
    family = str(raw.get("op_family") or "").strip().lower()
    return _GENOPS_OUTPUT_OP_CLASS.get(family, "generic")


def select_metamorphic_shape(
    shapes: Sequence[Any],
    *,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> tuple[Optional[dict[str, int]], Optional[dict[str, int]], str]:
    """Pick the cheapest usable ``[M, N]`` shape for the metamorphic run.

    Returns ``(dims_used, dims_declared, note)``. The relations are structural,
    so the *smallest* declared shape is as much evidence as the largest while
    costing far less; when even that exceeds ``max_elements`` the row count is
    reduced (columns are preserved so a multi-block reduction loop still runs)
    and the note records the substitution. ``(None, None, reason)`` when no
    declared shape is usable - the caller must then treat the prong as unable to
    run rather than as passing.
    """
    usable: list[dict[str, int]] = []
    for shape in shapes or ():
        dims = dict(getattr(shape, "dims", {}) or {})
        if set(dims) != {"M", "N"}:
            continue
        try:
            m, n = int(dims["M"]), int(dims["N"])
        except (TypeError, ValueError):
            continue
        if m < _MIN_ROWS or n < _MIN_COLS:
            continue
        usable.append({"M": m, "N": n})
    if not usable:
        return None, None, (
            f"no requested shape is a 2-D [M, N] with M>={_MIN_ROWS}, N>={_MIN_COLS}")

    declared = min(usable, key=lambda d: (d["M"] * d["N"], d["M"], d["N"]))
    cap = max(int(max_elements), _MIN_ROWS * _MIN_COLS)
    if declared["M"] * declared["N"] <= cap:
        return dict(declared), dict(declared), f"declared shape {shape_spec(declared)}"
    rows = max(_MIN_ROWS, cap // declared["N"])
    used = {"M": int(rows), "N": declared["N"]}
    return used, dict(declared), (
        f"row-capped {shape_spec(declared)} -> {shape_spec(used)} "
        f"({cap} element cap)")


# --------------------------------------------------------------------------- #
# Runner report: encode / decode / sanitize
# --------------------------------------------------------------------------- #
_VERDICT_LITERALS = re.compile(
    r"(SNR|allclose|max_diff|median_ms|wall_ms|SHAPE_BEGIN|KORE_TIMING_PAIR|"
    r"KORE_DRIVER_CAPABILITIES|KORE_METAMORPHIC)",
    re.IGNORECASE,
)


def sanitize_detail(text: Any, limit: int = 320) -> str:
    """Neutralize verdict literals inside free text before it is printed.

    Candidate exception messages end up in diagnostics. They are printed before
    the runner's own verdict (so a last-match parse already wins), but stripping
    the protocol literals removes the channel entirely rather than relying on
    ordering.
    """
    s = " ".join(str(text).split())
    s = _VERDICT_LITERALS.sub("<redacted>", s)
    return s[:limit]


def _finite(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def format_metamorphic_report(payload: Mapping[str, Any]) -> str:
    """Render the diagnostic ``KORE_METAMORPHIC:`` line (single line, JSON)."""
    body = dict(payload)
    body["protocol"] = METAMORPHIC_PROTOCOL
    return METAMORPHIC_MARKER + " " + json.dumps(
        body, sort_keys=True, separators=(",", ":"), default=str)


def parse_metamorphic_report(text: str) -> Optional[dict[str, Any]]:
    """Parse the LAST diagnostic report line, or ``None``.

    DIAGNOSTIC ONLY. The caller must take its accept/reject decision from the
    forgery-scanned ``allclose:`` / ``SNR:`` literals and use this payload only
    to enrich the report (and only after cross-checking it against them).
    """
    matches = list(_METAMORPHIC_LINE.finditer(text or ""))
    if not matches:
        return None
    try:
        payload = json.loads(matches[-1].group(1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("protocol") != METAMORPHIC_PROTOCOL:
        return None
    return payload


# --------------------------------------------------------------------------- #
# The consumer-visible oracle report
# --------------------------------------------------------------------------- #
#: A prong is exactly one of these. Only ``pass``/``fail`` are verdicts:
#: ``off`` (gated out), ``not-applicable`` (no proven relation for this task),
#: ``inconclusive`` (meant to run, produced nothing) and ``unknown`` (this
#: process cannot observe whether it ran) all mean the prong contributed NO
#: evidence, so a consumer that wants four-prong coverage must check for them.
PRONG_STATES = ("pass", "fail", "off", "not-applicable", "inconclusive", "unknown")


@dataclass(frozen=True)
class ProngStatus:
    """One prong's state plus where its evidence physically came from."""

    name: str
    kind: str
    state: str
    evidence: str
    detail: str = ""

    @property
    def contributed(self) -> bool:
        """True only when the prong actually produced a verdict."""
        return self.state in ("pass", "fail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "state": self.state,
            "evidence": self.evidence, "detail": self.detail,
        }


@dataclass(frozen=True)
class OracleReport:
    """What the production oracle actually checked for one candidate.

    ``false_accept_bound`` is the honest ``(1-p)**m`` statistical bound from
    :func:`kore.verify.equivalence.false_accept_probability`, evaluated at
    ``reference_defect_fraction`` over the ``m`` element comparisons the random
    prong performed. It bounds *lucky random misses only*; the deterministic
    prongs contribute certainty on the regimes they enumerate, not probability.
    It is ``None`` (with ``bound_basis`` saying why) whenever ``m`` is not known
    exactly, so a number is never invented.
    """

    task_id: str
    verified: bool
    prongs: tuple[ProngStatus, ...] = ()
    random_trials_per_shape: int = 0
    random_shapes: int = 0
    random_elements: Optional[int] = None
    reference_defect_fraction: float = 1e-4
    false_accept_bound: Optional[float] = None
    #: ``log10`` of the same bound. With ``m`` in the hundreds of millions the
    #: bound underflows a float64 to exactly 0.0, which reads as "no risk stated"
    #: rather than "astronomically small"; the exponent stays informative.
    false_accept_bound_log10: Optional[float] = None
    bound_basis: str = ""
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def prong(self, name: str) -> Optional[ProngStatus]:
        for p in self.prongs:
            if p.name == name:
                return p
        return None

    def live_prongs(self) -> tuple[str, ...]:
        """Prongs that produced a verdict for this candidate."""
        return tuple(p.name for p in self.prongs if p.contributed)

    def failed_prongs(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.prongs if p.state == "fail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "verified": self.verified,
            "prongs": [p.to_dict() for p in self.prongs],
            "live_prongs": list(self.live_prongs()),
            "random_trials_per_shape": self.random_trials_per_shape,
            "random_shapes": self.random_shapes,
            "random_elements": self.random_elements,
            "reference_defect_fraction": self.reference_defect_fraction,
            "false_accept_bound": self.false_accept_bound,
            "false_accept_bound_log10": self.false_accept_bound_log10,
            "bound_basis": self.bound_basis,
            "detail": self.detail,
            **({"extra": self.extra} if self.extra else {}),
        }

    def summary(self) -> str:
        tag = "VERIFIED" if self.verified else "REJECTED"
        if self.false_accept_bound is None:
            bound = "n/a"
        elif self.false_accept_bound > 0.0:
            bound = f"{self.false_accept_bound:.3e}"
        else:
            bound = f"10^{self.false_accept_bound_log10:.1f}"
        lines = [
            f"[{tag}] {self.task_id}: {len(self.live_prongs())}/4 prongs live; "
            f"false_accept<={bound} at p={self.reference_defect_fraction:g}"
        ]
        for p in self.prongs:
            lines.append(
                f"  - {p.name:<12} [{p.state}] {p.evidence}"
                + (f" ({p.detail})" if p.detail else ""))
        if self.bound_basis:
            lines.append(f"  bound: {self.bound_basis}")
        if self.detail:
            lines.append(f"  {self.detail}")
        return "\n".join(lines)


def build_oracle_report(
    *,
    task_id: str,
    verified: bool,
    prongs: Sequence[ProngStatus],
    op_class: str,
    shape_dims: Sequence[Mapping[str, int]],
    trials_per_shape: Optional[int] = None,
    defect_fraction: float = 1e-4,
    detail: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> OracleReport:
    """Assemble an :class:`OracleReport`, computing the bound only when exact.

    ``m`` is ``trials_per_shape`` times the summed reference-output element
    count over every shape the random prong covered. For a row reduction that
    count is the number of output rows, which *understates* the input domain the
    trials actually sampled - i.e. the resulting bound is conservative (an upper
    bound), which is the safe direction.
    """
    trials = int(driver_random_trials() if trials_per_shape is None else trials_per_shape)
    per_shape = [expected_output_elements(op_class, dims) for dims in shape_dims]
    if shape_dims and all(count is not None for count in per_shape):
        elements = trials * sum(int(c) for c in per_shape)  # type: ignore[arg-type]
        bound = false_accept_probability(defect_fraction, elements)
        p = min(max(float(defect_fraction), 0.0), 1.0)
        bound_log10 = (elements * math.log1p(-p) / math.log(10.0)
                       if 0.0 < p < 1.0 else None)
        basis = (
            f"(1-p)^m with p={defect_fraction:g} and m={elements} element "
            f"comparisons ({trials} reseeded trials x {len(per_shape)} shape(s); "
            f"op_class={op_class!r}); bounds LUCKY RANDOM MISSES only - the "
            "deterministic prongs contribute certainty on enumerated regimes"
        )
    else:
        elements = None
        bound = None
        bound_log10 = None
        basis = (
            f"not computed: the per-shape comparison count is unknown for "
            f"op_class={op_class!r} (the driver does not report it), so no "
            "number is invented"
        )
    return OracleReport(
        task_id=str(task_id),
        verified=bool(verified),
        prongs=tuple(prongs),
        random_trials_per_shape=trials,
        random_shapes=len(list(shape_dims)),
        random_elements=elements,
        reference_defect_fraction=float(defect_fraction),
        false_accept_bound=bound,
        false_accept_bound_log10=bound_log10,
        bound_basis=basis,
        detail=detail,
        extra=dict(extra or {}),
    )


# --------------------------------------------------------------------------- #
# The workdir shim (mirrors the generated task driver.py exactly)
# --------------------------------------------------------------------------- #
RUNNER_SHIM_NAME = "kore_metamorphic.py"

#: Staged into the evaluation workdir next to ``driver.py`` and invoked with the
#: identical ``[python, <script>, --shape ...]`` argv shape, so the metamorphic
#: run goes through exactly the same execution boundary as a correctness run.
RUNNER_SHIM_SOURCE = '''"""KORE metamorphic prong entry (staged by kore.env.kore_env).

Mirrors the generated task driver shim: put the staged task directory first on
sys.path, import the task's reference module, and hand it to the runner.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import reference as ref  # noqa: E402
from kore.verify.runner import runner_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(runner_main(ref, _here))
'''
