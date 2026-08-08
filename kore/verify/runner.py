"""GPU-side runner for the production metamorphic prong.

Executed as a subprocess from :class:`kore.env.kore_env.KoreEnv`, inside the
same throwaway workdir the correctness driver runs in, through the same shim
shape (see :data:`kore.verify.production.RUNNER_SHIM_SOURCE`). It imports the
staged task ``reference.py`` (for the task's own input generator and entry name)
and the candidate ``kernel.py``, applies
:func:`kore.verify.metamorphic.metamorphic_relations` for the op class the
environment planned, and reduces the result with the SAME pure decision logic
the CPU unit tests drive (:func:`kore.verify.equivalence.equivalence_verdict`).

No reference evaluation happens here at all: every relation compares the
candidate against *itself* under a semantics-preserving input transform, which
is what makes this prong cheap and what lets it catch structural defects (a
"pointwise" kernel that secretly mixes elements across a row) that agree with
the oracle on every value the random prong happens to sample.

Verdict channel
---------------
The decision is published as ``SNR:`` / ``allclose:`` - the same literals the
task driver uses - because ``kore.reward.scan_for_hacks`` already rejects any
candidate source containing them, and the environment already takes the LAST
match, which is printed here after the candidate has finished executing. The
``KORE_METAMORPHIC:`` JSON line is diagnostic only.

Exit status: ``0`` when a verdict (pass or fail) was produced, ``3`` when the
prong could not run. ``3`` must never be read as a pass.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from kore.verify.equivalence import (
    ProngSamples,
    Tolerance,
    equivalence_verdict,
    tolerance_for,
)
from kore.verify.metamorphic import metamorphic_relations
from kore.verify.production import (
    METAMORPHIC_DTYPES,
    format_metamorphic_report,
    generic_adversarial_families,
    parse_shape_spec,
    sanitize_detail,
    shape_spec,
)

__all__ = ["runner_main", "metamorphic_check", "MetamorphicOutcome", "build_parser"]

_INCONCLUSIVE = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KORE metamorphic prong runner")
    p.add_argument("--shape", default="default",
                   help='shape spec, e.g. "M=64,N=512"')
    p.add_argument("--op-class", default="generic",
                   choices=["elementwise", "reduction", "generic"])
    p.add_argument("--source-family", default="",
                   help="genops source family the environment planned for")
    p.add_argument("--dtype", default="fp32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shape-declared", default="",
                   help="the declared shape this run stands in for (reporting)")
    return p


def _emit(payload: dict[str, Any]) -> None:
    print(format_metamorphic_report(payload))


def _emit_inconclusive(reason: str, payload: dict[str, Any]) -> int:
    """Report that no verdict could be produced. Deliberately prints NO
    ``allclose:`` line, so a caller that keys off the protected literal cannot
    mistake an unrunnable prong for a passing one."""
    body = {k: v for k, v in payload.items() if v is not None}
    body.update({"state": "inconclusive", "verified": False,
                 "reason": sanitize_detail(reason)})
    _emit(body)
    print(f"KORE_METAMORPHIC_INCONCLUSIVE: {sanitize_detail(reason)}")
    return _INCONCLUSIVE


def _load_candidate(task_dir: str, entry: str):
    # Dispatches on what the task staged: a HIP candidate is kernel.hip and
    # there is no kernel.py beside it, so loading Python unconditionally failed
    # in the loader rather than in the compile it was meant to exercise.
    from kore.env.hip_toolchain import load_candidate_module

    return getattr(load_candidate_module(task_dir), entry)


def _clone(value):
    import torch

    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    if isinstance(value, list):
        return [_clone(v) for v in value]
    return value


def _snr_text(value: Optional[float]) -> str:
    """Format an SNR the way the driver does (finite sentinels for +/-inf)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-999.00"
    if value == math.inf:
        return "999.00"
    if value == -math.inf:
        return "-999.00"
    return f"{float(value):.2f}"


def _adversarial_battery(ref: Any) -> str:
    """Which adversarial battery the DRIVER would use for this reference.

    Reporting only - the adversarial prong's verdict comes from the driver.
    """
    if hasattr(ref, "adversarial_inputs"):
        return "authored"
    if str(getattr(ref, "family", "") or "") in generic_adversarial_families():
        return "generic"
    return "none"


@dataclass
class MetamorphicOutcome:
    """Result of one metamorphic prong run (no I/O, no process assumptions).

    ``state`` is ``"verdict"`` when ``verified`` means something and
    ``"inconclusive"`` when the prong could not run at all. Callers must treat
    the latter as "no evidence", never as a pass.
    """

    state: str
    verified: bool = False
    reason: str = ""
    failures: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    worst_rel_err: float = math.inf
    worst_snr_db: float = -math.inf
    candidate_calls: int = 0
    output_elements: Optional[int] = None
    operands: Optional[int] = None
    input_elements: Optional[int] = None
    rtol: Optional[float] = None
    snr_db_min: Optional[float] = None

    @property
    def conclusive(self) -> bool:
        return self.state == "verdict"

    def detail(self) -> str:
        return "; ".join(self.failures) if self.failures else self.reason


def _inconclusive(reason: str) -> MetamorphicOutcome:
    return MetamorphicOutcome(state="inconclusive", reason=sanitize_detail(reason))


def metamorphic_check(ref: Any, candidate: Any, *, shape_spec: str,
                      op_class: str, dtype: str, source_family: str = "",
                      seed: int = 0) -> MetamorphicOutcome:
    """Apply the op class's identities to ``candidate``; pure of any I/O.

    ``ref`` supplies only the task's shape parser and input generator - the
    oracle is never evaluated, which is what makes this prong cheap. This is the
    single implementation of the prong: :func:`runner_main` wraps it for the
    subprocess the environment spawns, and the task driver can call it directly
    (see the driver patch in ``kore/verify/README.md``) to run it in-process.
    """
    if op_class not in ("elementwise", "reduction"):
        return _inconclusive(f"op_class {op_class!r} has no metamorphic relations")
    if dtype not in METAMORPHIC_DTYPES:
        return _inconclusive(f"dtype {dtype!r} is outside the calibrated tolerances")
    # The caller planned from task.yaml; the reference module is the second,
    # independent witness. A disagreement means the metadata and the generated
    # operator have drifted, and running an unproven relation set on a drifted
    # operator is exactly how an honest kernel gets false-rejected.
    ref_family = str(getattr(ref, "family", "") or "")
    if source_family and ref_family != source_family:
        return _inconclusive(
            f"reference.family {ref_family!r} != planned {source_family!r}")

    relations = metamorphic_relations(op_class)
    if not relations:
        return _inconclusive(f"no relations for op_class {op_class!r}")
    names = [r.name for r in relations]

    try:
        import torch  # noqa: PLC0415 - GPU-only path
    except Exception as exc:  # noqa: BLE001
        return _inconclusive(f"torch unavailable: {type(exc).__name__}")

    try:
        shape = ref.parse_shape(shape_spec)
    except Exception as exc:  # noqa: BLE001
        return _inconclusive(
            f"reference could not parse shape {shape_spec!r}: {type(exc).__name__}")
    try:
        inputs = ref.get_inputs(shape, device="cuda", seed=int(seed))
    except Exception as exc:  # noqa: BLE001 - a broken generator is not a verdict
        return _inconclusive(
            f"reference.get_inputs failed: {type(exc).__name__}: {exc}")
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    inputs = tuple(inputs)

    declared_arity = getattr(ref, "arity", None)
    if declared_arity is not None and int(declared_arity) != len(inputs):
        return _inconclusive(
            f"reference arity {declared_arity} != {len(inputs)} generated inputs")
    if not all(torch.is_tensor(t) and t.is_floating_point() for t in inputs):
        return _inconclusive(
            "the relations are defined for plain float operands only")
    if not all(tuple(t.shape) == tuple(inputs[0].shape) for t in inputs):
        return _inconclusive(
            "operands do not share one shape, so a shared permutation is not "
            "an identity of the operator")

    tol: Tolerance = tolerance_for(dtype)
    base = dict(relations=names, operands=len(inputs),
                input_elements=int(inputs[0].numel()),
                rtol=tol.metamorphic_rtol, snr_db_min=tol.metamorphic_snr_db_min)

    # Every relation composes two candidate evaluations, so each call gets its
    # own storage: an in-place kernel must not be able to corrupt the operand
    # the other half of the identity is computed from.
    calls = {"n": 0}
    # Synchronize after every call so an async launch failure surfaces at a
    # defined point rather than inside an unrelated relation. Skipped when the
    # operands never left the host, which keeps this same code path drivable by
    # a CPU-only harness (and by the unit tests).
    synchronize = (any(bool(getattr(t, "is_cuda", False)) for t in inputs)
                   and torch.cuda.is_available())

    def candidate_fn(*xs):
        calls["n"] += 1
        out = candidate(*_clone(tuple(xs)))
        if synchronize:
            torch.cuda.synchronize()
        return out

    # One untransformed call first: it pins down the output element count the
    # caller cross-checks its false-accept bound against, and it separates "this
    # candidate cannot run at all" from "this candidate violates an identity".
    try:
        probe = candidate_fn(*inputs)
    except Exception as exc:  # noqa: BLE001
        return MetamorphicOutcome(
            state="verdict", verified=False, candidate_calls=calls["n"],
            failures=[sanitize_detail(
                f"untransformed call raised {type(exc).__name__}: {exc}")],
            **base)
    output_elements = int(probe.numel()) if torch.is_tensor(probe) else None
    del probe

    pairs: list[tuple] = []
    labels: list[str] = []
    errors: list[str] = []
    for rel in relations:
        try:
            lhs, rhs = rel.apply(candidate_fn, inputs)
        except Exception as exc:  # noqa: BLE001 - a crash under a relation is a FAIL
            errors.append(sanitize_detail(
                f"{rel.name}: {type(exc).__name__}: {exc}"))
            # A guaranteed-failing pair keeps the crash inside the same verdict
            # path as a numerical violation (never a silent skip).
            pairs.append((np.array([math.nan]), np.array([0.0])))
            labels.append(f"{rel.name}:{type(exc).__name__}")
        else:
            pairs.append((lhs, rhs))
            labels.append(rel.name)

    verdict = equivalence_verdict(
        [ProngSamples("metamorphic", "metamorphic", pairs, labels=labels)], tol)
    prong = verdict.prong("metamorphic")
    passed = bool(prong is not None and prong.passed)
    failures = []
    if not passed and prong is not None and prong.detail:
        failures.append(sanitize_detail(prong.detail))
    failures.extend(errors)
    return MetamorphicOutcome(
        state="verdict", verified=passed, failures=failures,
        worst_rel_err=float(prong.worst_rel_err) if prong else math.inf,
        worst_snr_db=float(prong.worst_snr_db) if prong else -math.inf,
        candidate_calls=calls["n"], output_elements=output_elements, **base)


def runner_main(ref: Any, task_dir: str, argv=None) -> int:
    """Run the metamorphic prong for the staged ``ref`` / ``kernel.py`` pair."""
    args = build_parser().parse_args(argv)
    op_class = str(args.op_class)
    dtype = str(args.dtype).lower()
    base: dict[str, Any] = {
        "op_class": op_class,
        "source_family": args.source_family,
        "dtype": dtype,
        "shape": args.shape,
        "shape_declared": args.shape_declared or args.shape,
        "adversarial_battery": _adversarial_battery(ref),
    }

    entry = getattr(ref, "entry_name", None)
    if not entry:
        return _emit_inconclusive("reference declares no entry_name", base)
    base["entry"] = str(entry)
    try:
        candidate = _load_candidate(task_dir, str(entry))
    except Exception as exc:  # noqa: BLE001 - the driver already judged compilation
        return _emit_inconclusive(
            f"candidate import failed: {type(exc).__name__}: "
            f"{sanitize_detail(exc)}", base)

    outcome = metamorphic_check(
        ref, candidate, shape_spec=args.shape, op_class=op_class, dtype=dtype,
        source_family=args.source_family, seed=int(args.seed))
    base.update({
        "relations": outcome.relations,
        "operands": outcome.operands,
        "input_elements": outcome.input_elements,
    })
    if not outcome.conclusive:
        return _emit_inconclusive(outcome.reason, base)
    return _verdict(base, passed=outcome.verified,
                    worst_rel=outcome.worst_rel_err, worst_snr=outcome.worst_snr_db,
                    tol=tolerance_for(dtype), calls=outcome.candidate_calls,
                    output_elements=outcome.output_elements,
                    failures=list(outcome.failures), shape=args.shape)


def _verdict(base: dict[str, Any], *, passed: bool, worst_rel: float,
             worst_snr: float, tol: Tolerance, calls: int,
             output_elements: Optional[int], failures: list[str],
             shape: str) -> int:
    """Publish the prong verdict on the forgery-scanned literal channel."""
    _emit({
        **base,
        "state": "verdict",
        "verified": bool(passed),
        "candidate_calls": int(calls),
        "output_elements": output_elements,
        "worst_rel_err": (worst_rel if math.isfinite(worst_rel) else None),
        "worst_snr_db": (worst_snr if math.isfinite(worst_snr) else None),
        "metamorphic_rtol": tol.metamorphic_rtol,
        "metamorphic_snr_db_min": tol.metamorphic_snr_db_min,
        "failures": list(failures),
        "shape_used": shape_spec(parse_shape_spec(shape)),
    })
    if failures:
        # Printed BEFORE the verdict literals so a last-match parse still lands
        # on the runner's own verdict.
        print("METAMORPHIC_FAIL: " + "; ".join(failures)[:600])
    print(f"SNR: {_snr_text(worst_snr)} dB")
    print(f"allclose: {bool(passed)}")
    return 0
