"""KORE stage gates - enforce "specialize without regressing" (KORE.pdf Sec 5).

The KORE definition of best-in-world is *conjunctive*: top kernel numbers **AND**
matching-or-beating the base model on **every** general benchmark. A candidate
that raises fast_p by copying the reference or by trading away general chat/code/
reasoning is a failure, not a win. These gates make that contract executable so a
training campaign can hard-stop (or auto-reject a checkpoint) the moment
specialization silently regresses a general capability.

Two gates:
  - :class:`StageGate` - the full promotion gate. PASS iff (a) the targeted
    *kernel* metric(s) strictly improve AND (b) NO *general* metric drops by more
    than ``epsilon``. Used to promote a checkpoint to the next stage / crown a
    new best.
  - :func:`retention_gate` - the general-only guardrail (no kernel-improvement
    requirement): PASS iff no general metric regresses beyond ``epsilon`` vs base.
    Cheap to run every step as an early tripwire.

All metrics are treated as **higher-is-better** in ``[0, ...]`` (accuracy, pass@1,
fast_p, geomean speedup). If you gate on a lower-is-better metric (e.g. latency),
pass its negation or a reciprocal. This module is PURE (no I/O, no heavy imports)
and directly testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

# Default tolerance: a general metric may dip by at most this much (absolute)
# before it counts as a regression. 0.005 = half a point of accuracy, i.e. noise.
DEFAULT_EPSILON = 0.005


@dataclass
class MetricDelta:
    """Per-key before/after delta with a regression/improvement verdict."""

    key: str
    before: Optional[float]
    after: Optional[float]
    delta: Optional[float]
    kind: str  # "kernel" | "general"
    regressed: bool = False
    improved: bool = False


@dataclass
class GateResult:
    """Outcome of a gate evaluation.

    ``passed``       : did the gate pass?
    ``regressions``  : keys that regressed (general dropped > epsilon, or a
                       required kernel key failed to strictly improve).
    ``improvements`` : keys that strictly improved.
    ``detail``       : structured per-key deltas + reasons for the verdict.
    """

    passed: bool
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


def _get(metrics: Mapping[str, float], key: str) -> tuple[Optional[float], Optional[str]]:
    """Return one finite metric, distinguishing missing from malformed values."""
    try:
        if key not in metrics or metrics.get(key) is None:
            return None, "missing"
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "not numeric"
    if not math.isfinite(value):
        return None, "not finite"
    return value, None


def _epsilon_error(epsilon: float) -> Optional[str]:
    try:
        eps = float(epsilon)
    except (TypeError, ValueError, OverflowError):
        return "epsilon is not numeric"
    if not math.isfinite(eps):
        return "epsilon is not finite"
    if eps < 0.0:
        return "epsilon must be non-negative"
    return None


class GateError(AssertionError):
    """Raised by :func:`assert_gate_or_raise` when a gate fails."""

    def __init__(self, result: GateResult, message: str):
        self.result = result
        super().__init__(message)


class StageGate:
    """Promotion gate combining a kernel-improvement test and a no-regression test.

    Contract (``evaluate``):
      PASS iff
        (a) **every** targeted kernel metric in ``kernel_keys`` strictly improves
            (``after > before``), AND
        (b) **no** general metric in ``general_keys`` drops by more than
            ``epsilon`` (``after >= before - epsilon``).

    Rationale: (a) is the specialization objective (we only crown a checkpoint
    that actually pushed the kernel frontier); (b) is the retention guarantee that
    prevents silent regression of general chat/code/reasoning. Missing keys are
    reported as failures (a metric we promised to track but did not measure is
    treated conservatively as a failure, not a silent pass).
    """

    def __init__(self, epsilon: float = DEFAULT_EPSILON, *, require_all_kernel: bool = True):
        self.epsilon = float(epsilon)
        # If False, a single strictly-improving kernel key suffices (with none of
        # the others regressing). Default True = every kernel key must improve.
        self.require_all_kernel = require_all_kernel

    def evaluate(
        self,
        before: Mapping[str, float],
        after: Mapping[str, float],
        *,
        kernel_keys: Iterable[str],
        general_keys: Iterable[str],
        epsilon: Optional[float] = None,
    ) -> GateResult:
        raw_epsilon = self.epsilon if epsilon is None else epsilon
        try:
            eps = float(raw_epsilon)
        except (TypeError, ValueError, OverflowError):
            eps = float("nan")
        kernel_keys = list(kernel_keys)
        general_keys = list(general_keys)

        deltas: dict[str, MetricDelta] = {}
        regressions: list[str] = []
        improvements: list[str] = []
        reasons: list[str] = []

        epsilon_error = _epsilon_error(raw_epsilon)
        if epsilon_error:
            reasons.append(epsilon_error)
        if not general_keys:
            reasons.append("no general_keys provided; stage gate requires full-source retention metrics")
        overlap = sorted(set(kernel_keys) & set(general_keys))
        if overlap:
            reasons.append(f"metrics cannot be both kernel and general: {overlap}")

        # --- (a) kernel: must strictly improve ---
        kernel_improved_flags: list[bool] = []
        for k in kernel_keys:
            b, b_error = _get(before, k)
            a, a_error = _get(after, k)
            if b_error or a_error:
                reasons.append(
                    f"kernel key '{k}' invalid "
                    f"(before={b_error or b}, after={a_error or a})"
                )
                regressions.append(k)
                kernel_improved_flags.append(False)
                deltas[k] = MetricDelta(k, b, a, None, "kernel", regressed=True)
                continue
            assert b is not None and a is not None
            d = a - b
            improved = d > 0.0
            md = MetricDelta(k, b, a, d, "kernel", improved=improved, regressed=not improved)
            deltas[k] = md
            kernel_improved_flags.append(improved)
            if improved:
                improvements.append(k)
            else:
                regressions.append(k)
                reasons.append(f"kernel key '{k}' did not strictly improve (Δ={d:+.4f})")

        if kernel_keys:
            kernel_ok = all(kernel_improved_flags) if self.require_all_kernel else any(kernel_improved_flags)
        else:
            # No kernel target specified => (a) is vacuously unmet: a stage gate
            # must have a kernel objective to promote on.
            kernel_ok = False
            reasons.append("no kernel_keys provided; stage gate requires a kernel objective")

        # --- (b) general: none may drop by more than epsilon ---
        general_ok = bool(general_keys) and epsilon_error is None and not overlap
        for k in general_keys:
            b, b_error = _get(before, k)
            a, a_error = _get(after, k)
            if b_error or a_error:
                reasons.append(
                    f"general key '{k}' invalid "
                    f"(before={b_error or b}, after={a_error or a})"
                )
                regressions.append(k)
                general_ok = False
                deltas[k] = MetricDelta(k, b, a, None, "general", regressed=True)
                continue
            assert b is not None and a is not None
            d = a - b
            regressed = d < -eps
            md = MetricDelta(k, b, a, d, "general", improved=d > 0.0, regressed=regressed)
            deltas[k] = md
            if d > 0.0:
                improvements.append(k)
            if regressed:
                general_ok = False
                regressions.append(k)
                reasons.append(f"general key '{k}' regressed beyond epsilon (Δ={d:+.4f}, eps={eps})")

        passed = bool(kernel_ok and general_ok)
        detail = {
            "epsilon": eps if math.isfinite(eps) else None,
            "require_all_kernel": self.require_all_kernel,
            "kernel_keys": kernel_keys,
            "general_keys": general_keys,
            "kernel_ok": kernel_ok,
            "general_ok": general_ok,
            "deltas": {k: vars(v) for k, v in deltas.items()},
            "reasons": reasons,
        }
        # De-duplicate while preserving order.
        regressions = list(dict.fromkeys(regressions))
        improvements = list(dict.fromkeys(improvements))
        return GateResult(passed=passed, regressions=regressions, improvements=improvements, detail=detail)


def retention_gate(
    base_scores: Mapping[str, float],
    candidate_scores: Mapping[str, float],
    epsilon: float = DEFAULT_EPSILON,
) -> GateResult:
    """General-only guardrail: PASS iff no shared general metric regresses > epsilon.

    Unlike :class:`StageGate` this imposes NO kernel-improvement requirement - it
    is the cheap early tripwire you run every training step against the base
    model's general scores (e.g. the output of
    :func:`kore.eval.retention.run_retention_suite`'s ``scores``). Only keys
    present in ``base_scores`` are checked; extra candidate keys are ignored.
    """
    try:
        eps = float(epsilon)
    except (TypeError, ValueError, OverflowError):
        eps = float("nan")
    keys = list(base_scores.keys())
    regressions: list[str] = []
    improvements: list[str] = []
    reasons: list[str] = []
    deltas: dict[str, dict] = {}
    epsilon_error = _epsilon_error(epsilon)
    passed = bool(keys) and epsilon_error is None
    if not keys:
        reasons.append("base_scores is empty; retention gate cannot pass vacuously")
    if epsilon_error:
        reasons.append(epsilon_error)
    for k in keys:
        b, b_error = _get(base_scores, k)
        a, a_error = _get(candidate_scores, k)
        if b_error or a_error:
            reasons.append(
                f"general key '{k}' invalid "
                f"(base={b_error or b}, candidate={a_error or a})"
            )
            regressions.append(k)
            passed = False
            deltas[k] = {"key": k, "before": b, "after": a, "delta": None, "regressed": True}
            continue
        assert b is not None and a is not None
        d = a - b
        regressed = False if epsilon_error else d < -eps
        if d > 0.0:
            improvements.append(k)
        if regressed:
            passed = False
            regressions.append(k)
            reasons.append(f"general key '{k}' regressed beyond epsilon (Δ={d:+.4f}, eps={eps})")
        deltas[k] = {"key": k, "before": b, "after": a, "delta": d, "regressed": regressed}
    detail = {
        "epsilon": eps if math.isfinite(eps) else None,
        "kind": "retention_only",
        "checked_keys": keys,
        "deltas": deltas,
        "reasons": reasons,
    }
    return GateResult(passed=passed, regressions=regressions, improvements=improvements, detail=detail)


def format_gate_report(result: GateResult, *, title: str = "KORE stage gate") -> str:
    """Human-readable markdown report for a :class:`GateResult`."""
    lines: list[str] = []
    status = "PASS ✅" if result.passed else "FAIL ❌"
    lines.append(f"# {title}: {status}")
    lines.append("")
    eps = result.detail.get("epsilon")
    if eps is not None:
        lines.append(f"- epsilon (max allowed general drop): {eps}")
    if "kernel_keys" in result.detail:
        lines.append(f"- kernel objective improved: {result.detail.get('kernel_ok')}")
        lines.append(f"- general retention held: {result.detail.get('general_ok')}")
    lines.append("")

    deltas = result.detail.get("deltas", {})
    if deltas:
        lines.append("| metric | kind | before | after | Δ | verdict |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for k, d in deltas.items():
            kind = d.get("kind", "-")
            before = d.get("before")
            after = d.get("after")
            delta = d.get("delta")

            def _f(x):
                try:
                    return "-" if x is None or not math.isfinite(float(x)) else f"{float(x):.4f}"
                except (TypeError, ValueError, OverflowError):
                    return "-"

            if d.get("regressed"):
                verdict = "regressed"
            elif d.get("improved"):
                verdict = "improved"
            else:
                verdict = "flat"
            lines.append(f"| {k} | {kind} | {_f(before)} | {_f(after)} | {_f(delta)} | {verdict} |")
        lines.append("")

    if result.improvements:
        lines.append(f"**Improvements**: {', '.join(result.improvements)}")
    if result.regressions:
        lines.append(f"**Regressions**: {', '.join(result.regressions)}")
    reasons = result.detail.get("reasons") or []
    if reasons:
        lines.append("")
        lines.append("**Reasons:**")
        for r in reasons:
            lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)


def assert_gate_or_raise(
    before: Mapping[str, float],
    after: Mapping[str, float],
    *,
    kernel_keys: Iterable[str],
    general_keys: Iterable[str],
    epsilon: float = DEFAULT_EPSILON,
    require_all_kernel: bool = True,
    title: str = "KORE stage gate",
) -> GateResult:
    """Evaluate the :class:`StageGate` and raise :class:`GateError` if it fails.

    Intended for campaign use: a checkpoint that fails is not promoted, and the
    raised error carries the human-readable report so the failure is legible in
    logs / CI. Returns the :class:`GateResult` on PASS.
    """
    gate = StageGate(epsilon=epsilon, require_all_kernel=require_all_kernel)
    result = gate.evaluate(before, after, kernel_keys=kernel_keys, general_keys=general_keys)
    if not result.passed:
        raise GateError(result, format_gate_report(result, title=title))
    return result


__all__ = [
    "DEFAULT_EPSILON",
    "MetricDelta",
    "GateResult",
    "GateError",
    "StageGate",
    "retention_gate",
    "assert_gate_or_raise",
    "format_gate_report",
]
