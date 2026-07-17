"""Robust-kbench-style hardening of the EVAL correctness verdict.

A published kernel result is only as trustworthy as its correctness gate. Naive
``allclose`` on a handful of random inputs is easily *passed* by kernels that are
not actually correct - a constant/memset kernel that happens to match on the easy
elements, or a precision-DOWNGRADED kernel whose error hides under a loose
tolerance. Robust-KernelBench hardened its verdict against exactly these; this
module brings the same rigor to KORE's EVAL-time correctness decision.

Every check here is a PURE function of a candidate-output CALLABLE and a torch
REFERENCE callable (plus a deterministic input factory), so it is fully CPU/torch
testable with fake candidate functions and never needs a GPU or the KORE driver.
The battery:

  * :func:`check_random_inits`      - many reseeded random initializations.
  * :func:`check_adversarial_regimes` - the enumerated hard fills (zeros / ones /
    neg-ones / large / neg-large / small / sign-alternating / nan-inf structure),
    with non-finite-STRUCTURE-aware comparison (a correct kernel reproduces the
    reference's NaN/Inf positions and inf signs exactly).
  * :func:`check_noncontiguous`     - NON-contiguous (transposed / strided) inputs
    that break kernels which silently assume a contiguous layout.
  * :func:`check_differential_oracle` - a DIFFERENTIAL oracle: recompute the
    reference in fp64 and require the candidate to be no less accurate (relative to
    the fp64 truth) than the reference's own dtype rounding, by more than a small
    factor. This is what catches a precision downgrade that ``allclose`` waves through.
  * METAMORPHIC relations the true function must obey and a correct kernel must
    therefore preserve: :func:`metamorphic_permutation_invariance` (reductions),
    :func:`metamorphic_homogeneity` (``f(ax)=a f(x)`` for linear ops), and
    :func:`metamorphic_additive_response` (additivity for fusions).

:func:`robust_correctness` runs the applicable battery and HALTS ON THE FIRST
MISMATCH, returning the failing check and its metrics - the maximum-scrutiny verdict
a publishable claim needs.

Import-safe / offline: torch is imported lazily inside every function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# A candidate/reference is any callable ``(*inputs) -> Tensor | tuple[Tensor,...]``.
OutputFn = Callable[..., object]
# An input factory maps a seed -> the input tuple (deterministic per seed).
InputFactory = Callable[[int], tuple]


# --------------------------------------------------------------------------- #
# Config + result records.
# --------------------------------------------------------------------------- #
@dataclass
class RobustConfig:
    """Tolerances + trial counts for the robust correctness battery.

    ``snr_db_min`` + (``atol``, ``rtol``) define the per-comparison agreement bar
    (the same shape as KORE's own SNR-gate + allclose). The differential oracle uses
    ``differential_tol_factor`` * (reference's fp64 error) + ``differential_abs_floor``
    as the candidate's allowed fp64-relative error, so a candidate that is much less
    accurate than its own dtype warrants is rejected even when it passes ``allclose``.
    """

    n_random_trials: int = 8
    snr_db_min: float = 40.0
    atol: float = 1e-2
    rtol: float = 1e-2
    differential_tol_factor: float = 10.0
    differential_abs_floor: float = 1e-3
    metamorphic_alpha: float = 2.0      # positive scale for the homogeneity relation
    base_seed: int = 0


@dataclass
class CheckResult:
    name: str
    passed: bool
    reason: str = ""
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed,
                "reason": self.reason, "metrics": self.metrics}


@dataclass
class RobustReport:
    passed: bool
    checks: list = field(default_factory=list)
    failed_check: Optional[str] = None

    @property
    def n_checks(self) -> int:
        return len(self.checks)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "failed_check": self.failed_check,
                "n_checks": self.n_checks, "checks": [c.to_dict() for c in self.checks]}

    def summary(self) -> str:
        head = "ROBUST-CORRECT" if self.passed else f"REJECTED@{self.failed_check}"
        lines = [f"[{head}] {sum(c.passed for c in self.checks)}/{self.n_checks} checks passed"]
        for c in self.checks:
            tag = "ok" if c.passed else "FAIL"
            lines.append(f"    [{tag:4s}] {c.name}: {c.reason}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Comparison primitives (non-finite-structure aware; mirrors the KORE driver).
# --------------------------------------------------------------------------- #
def _snr_db(out, ref_out) -> float:
    o, r = out.double(), ref_out.double()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


def _as_tuple(x):
    return x if isinstance(x, (tuple, list)) else (x,)


def _clone_inputs(inputs):
    import torch
    return tuple(t.clone() if torch.is_tensor(t) else t for t in inputs)


def compare_outputs(out, ref_out, *, atol: float, rtol: float) -> tuple[float, float, bool]:
    """(worst_snr_db, max_abs_diff, allclose_ok) over single- or multi-tensor output.

    Non-finite-STRUCTURE aware: a correct kernel must reproduce the reference's
    NaN/Inf positions (and inf signs) exactly, and match closely on the finite
    subset. This is what makes the adversarial NaN/Inf regime a real check rather
    than an automatic false-reject.
    """
    import torch

    outs, refs = _as_tuple(out), _as_tuple(ref_out)
    if len(outs) != len(refs):
        return -999.0, float("inf"), False
    worst, maxd, ok = 999.0, 0.0, True
    for o, r in zip(outs, refs):
        if not torch.is_tensor(o) or not torch.is_tensor(r):
            ok = ok and (o == r)
            continue
        if tuple(o.shape) != tuple(r.shape):
            return -999.0, float("inf"), False
        of, rf = o.double(), r.double()
        rfin, ofin = torch.isfinite(rf), torch.isfinite(of)
        if not torch.equal(rfin, ofin):
            return -999.0, float("inf"), False
        rinf = torch.isinf(rf)
        if bool(rinf.any()) and not torch.equal(torch.sign(rf[rinf]), torch.sign(of[rinf])):
            return -999.0, float("inf"), False
        of_c, rf_c = (of, rf) if bool(rfin.all()) else (of[rfin], rf[rfin])
        if rf_c.numel() == 0:
            continue
        worst = min(worst, _snr_db(of_c, rf_c))
        maxd = max(maxd, (of_c - rf_c).abs().max().item())
        ok = ok and bool(torch.allclose(of_c, rf_c, atol=atol, rtol=rtol))
    return worst, maxd, ok


def _agree(out, ref_out, cfg: RobustConfig) -> tuple[bool, float, float]:
    """Return (agrees, worst_snr_db, max_diff): allclose AND SNR above the gate."""
    snr, maxd, ok = compare_outputs(out, ref_out, atol=cfg.atol, rtol=cfg.rtol)
    return (ok and snr >= cfg.snr_db_min), snr, maxd


def _rel_err(out, truth) -> float:
    """Relative L2 error ||out - truth|| / (||truth|| + eps), on the finite subset."""
    import torch
    o, t = out.double(), truth.double()
    fin = torch.isfinite(t) & torch.isfinite(o)
    if not bool(fin.all()):
        o, t = o[fin], t[fin]
    if t.numel() == 0:
        return 0.0
    denom = t.norm().item()
    num = (o - t).norm().item()
    return num / (denom + 1e-30)


# --------------------------------------------------------------------------- #
# Check 1: many random initializations.
# --------------------------------------------------------------------------- #
def check_random_inits(cand: OutputFn, ref: OutputFn, make_inputs: InputFactory,
                       *, cfg: RobustConfig = RobustConfig()) -> CheckResult:
    """Reseeded random inputs: the candidate must agree with the reference on all."""
    worst_snr = 999.0
    for s in range(cfg.n_random_trials):
        inputs = make_inputs(cfg.base_seed + s)
        r = ref(*_clone_inputs(inputs))
        try:
            o = cand(*_clone_inputs(inputs))
        except Exception as e:  # noqa: BLE001 - a candidate that throws is incorrect
            return CheckResult("random_inits", False,
                               f"candidate raised on seed {s}: {type(e).__name__}: {e}",
                               {"seed": s})
        agrees, snr, maxd = _agree(o, r, cfg)
        worst_snr = min(worst_snr, snr)
        if not agrees:
            return CheckResult("random_inits", False,
                               f"mismatch on seed {s} (SNR {snr:.1f} dB, max_diff {maxd:.3e})",
                               {"seed": s, "snr_db": snr, "max_diff": maxd})
    return CheckResult("random_inits", True, f"{cfg.n_random_trials} trials agree "
                       f"(worst SNR {worst_snr:.1f} dB)", {"worst_snr_db": worst_snr})


# --------------------------------------------------------------------------- #
# Check 2: enumerated adversarial input regimes.
# --------------------------------------------------------------------------- #
def adversarial_fills(inputs) -> list[tuple[str, tuple]]:
    """Yield (regime_name, filled_inputs) over the hard fills.

    Each fill preserves every input's shape/dtype/device but replaces its FLOAT
    values with a canonical hard regime (integer/index inputs are left intact). The
    sign-alternating parity is built in int64 (never in the tensor dtype) so an fp16
    tensor cannot overflow to Inf and poison the fill. The ``nan_inf`` regime injects
    a structured pattern of NaN / +Inf / -Inf so the comparison exercises the
    non-finite-structure check.
    """
    import torch

    def _sign_alt(t):
        parity = (torch.arange(t.numel(), device=t.device) % 2) * 2 - 1
        return parity.to(t.dtype).reshape(t.shape)

    def _nan_inf(t):
        flat = torch.zeros(t.numel(), dtype=t.dtype, device=t.device)
        if flat.numel() >= 1:
            flat[0::4] = float("nan")
            flat[1::4] = float("inf")
            flat[2::4] = float("-inf")
        return flat.reshape(t.shape)

    patterns = {
        "zeros": lambda t: torch.zeros_like(t),
        "ones": lambda t: torch.ones_like(t),
        "neg_ones": lambda t: -torch.ones_like(t),
        "large": lambda t: torch.full_like(t, 1.0e3),
        "neg_large": lambda t: torch.full_like(t, -1.0e3),
        "small": lambda t: torch.full_like(t, 1.0e-3),
        "sign_alt": _sign_alt,
        "nan_inf": _nan_inf,
    }

    def _fill(fill, t):
        return fill(t) if (torch.is_tensor(t) and torch.is_floating_point(t)) else t

    return [(name, tuple(_fill(fill, t) for t in inputs)) for name, fill in patterns.items()]


def check_adversarial_regimes(cand: OutputFn, ref: OutputFn, make_inputs: InputFactory,
                              *, cfg: RobustConfig = RobustConfig(),
                              regimes: Optional[Sequence[str]] = None) -> CheckResult:
    """The candidate must match the reference on every enumerated hard regime."""
    base = make_inputs(cfg.base_seed)
    want = set(regimes) if regimes is not None else None
    for name, adv in adversarial_fills(base):
        if want is not None and name not in want:
            continue
        r = ref(*_clone_inputs(adv))
        try:
            o = cand(*_clone_inputs(adv))
        except Exception as e:  # noqa: BLE001
            return CheckResult("adversarial_regimes", False,
                               f"candidate raised on regime '{name}': {type(e).__name__}: {e}",
                               {"regime": name})
        agrees, snr, maxd = _agree(o, r, cfg)
        if not agrees:
            return CheckResult("adversarial_regimes", False,
                               f"mismatch on regime '{name}' (SNR {snr:.1f} dB, "
                               f"max_diff {maxd:.3e})",
                               {"regime": name, "snr_db": snr, "max_diff": maxd})
    return CheckResult("adversarial_regimes", True, "all hard regimes agree")


# --------------------------------------------------------------------------- #
# Check 3: non-contiguous / strided inputs.
# --------------------------------------------------------------------------- #
def _make_noncontiguous(t):
    """A NON-contiguous tensor with the SAME shape + values as ``t``.

    For >=2D: swap the last two dims through a contiguous buffer then swap back, so
    the logical layout is identical but the strides are transposed. For 1D: embed in
    a width-2 buffer and slice a column (stride 2). Scalars / non-float pass through.
    """
    import torch
    if not torch.is_tensor(t) or not torch.is_floating_point(t) or t.dim() == 0:
        return t
    if t.dim() >= 2:
        nc = t.transpose(-1, -2).contiguous().transpose(-1, -2)
    else:
        buf = torch.empty(t.shape[0], 2, dtype=t.dtype, device=t.device)
        buf[:, 0] = t
        nc = buf[:, 0]
    return nc


def check_noncontiguous(cand: OutputFn, ref: OutputFn, make_inputs: InputFactory,
                        *, cfg: RobustConfig = RobustConfig()) -> CheckResult:
    """Feed non-contiguous inputs: a kernel assuming contiguity reads wrong memory."""
    import torch

    base = make_inputs(cfg.base_seed)
    nc = tuple(_make_noncontiguous(t) for t in base)
    if all((not torch.is_tensor(t)) or t.is_contiguous() for t in nc):
        return CheckResult("noncontiguous", True, "no non-contiguous variant applicable")
    r = ref(*_clone_inputs(nc))
    try:
        o = cand(*_clone_inputs(nc))
    except Exception as e:  # noqa: BLE001
        return CheckResult("noncontiguous", False,
                           f"candidate raised on non-contiguous inputs: {type(e).__name__}: {e}")
    agrees, snr, maxd = _agree(o, r, cfg)
    if not agrees:
        return CheckResult("noncontiguous", False,
                           f"mismatch on non-contiguous inputs (SNR {snr:.1f} dB, "
                           f"max_diff {maxd:.3e})", {"snr_db": snr, "max_diff": maxd})
    return CheckResult("noncontiguous", True, f"agrees on strided inputs (SNR {snr:.1f} dB)")


# --------------------------------------------------------------------------- #
# Check 4: differential oracle (candidate vs fp64 recompute of the reference).
# --------------------------------------------------------------------------- #
def _to_double(inputs):
    import torch
    return tuple(t.double() if (torch.is_tensor(t) and torch.is_floating_point(t)) else t
                 for t in inputs)


def check_differential_oracle(cand: OutputFn, ref: OutputFn, make_inputs: InputFactory,
                              *, cfg: RobustConfig = RobustConfig()) -> CheckResult:
    """Reject a candidate less accurate than its dtype warrants (precision downgrade).

    Recompute the reference in fp64 (the high-precision TRUTH), then compare the
    candidate's fp64-relative error to the REFERENCE's own fp64-relative error (its
    intrinsic dtype rounding). Fail when the candidate's error exceeds
    ``tol_factor * ref_err + abs_floor`` - i.e. it is materially less accurate than a
    faithful implementation would be, even if it slips past ``allclose``.
    """
    import torch

    inputs = make_inputs(cfg.base_seed)
    truth = _as_tuple(ref(*_clone_inputs(_to_double(inputs))))
    ref_native = _as_tuple(ref(*_clone_inputs(inputs)))
    try:
        cand_out = _as_tuple(cand(*_clone_inputs(inputs)))
    except Exception as e:  # noqa: BLE001
        return CheckResult("differential_oracle", False,
                           f"candidate raised: {type(e).__name__}: {e}")
    if not (len(truth) == len(ref_native) == len(cand_out)):
        return CheckResult("differential_oracle", False, "output arity mismatch vs reference")

    worst_ref, worst_cand = 0.0, 0.0
    for tr, rn, co in zip(truth, ref_native, cand_out):
        if not torch.is_tensor(tr):
            continue
        if tuple(co.shape) != tuple(tr.shape):
            return CheckResult("differential_oracle", False, "candidate output shape mismatch")
        worst_ref = max(worst_ref, _rel_err(rn, tr))
        worst_cand = max(worst_cand, _rel_err(co, tr))

    allowed = cfg.differential_tol_factor * worst_ref + cfg.differential_abs_floor
    metrics = {"cand_rel_err": worst_cand, "ref_rel_err": worst_ref, "allowed": allowed}
    if worst_cand > allowed:
        return CheckResult("differential_oracle", False,
                           f"precision downgrade: candidate fp64-rel-err {worst_cand:.3e} > "
                           f"allowed {allowed:.3e} (ref {worst_ref:.3e})", metrics)
    return CheckResult("differential_oracle", True,
                       f"fp64-accuracy OK (cand {worst_cand:.3e} <= {allowed:.3e})", metrics)


# --------------------------------------------------------------------------- #
# Metamorphic relations.
# --------------------------------------------------------------------------- #
def _index_select(t, axis: int, idx):
    return t.index_select(axis if axis >= 0 else t.dim() + axis, idx)


def metamorphic_permutation_invariance(cand: OutputFn, ref: OutputFn,
                                       make_inputs: InputFactory, *,
                                       cfg: RobustConfig = RobustConfig(),
                                       arg_index: int = 0, axis: int = -1) -> CheckResult:
    """Permutation-invariance (reductions): permuting the reduced axis must not
    change the output. Applied only when the REFERENCE is itself invariant on this
    axis; then the candidate MUST be invariant too, else it is flagged (e.g. a
    reduction that secretly depends on element order)."""
    import torch

    inputs = list(make_inputs(cfg.base_seed))
    x = inputs[arg_index]
    if not (torch.is_tensor(x) and x.dim() >= 1):
        return CheckResult("metamorphic_permutation", True, "not applicable (arg not a tensor)")
    ax = axis if axis >= 0 else x.dim() + axis
    n = x.shape[ax]
    g = torch.Generator(device=x.device).manual_seed(cfg.base_seed + 7)
    perm = torch.randperm(n, generator=g, device=x.device)

    perm_inputs = list(inputs)
    perm_inputs[arg_index] = _index_select(x, ax, perm)

    # Applicability: is the true function (reference) invariant on this axis?
    ref_base = ref(*_clone_inputs(tuple(inputs)))
    ref_perm = ref(*_clone_inputs(tuple(perm_inputs)))
    ref_invariant, _, _ = _agree(ref_perm, ref_base, cfg)
    if not ref_invariant:
        return CheckResult("metamorphic_permutation", True,
                           "not applicable (reference not permutation-invariant on this axis)")

    cand_base = cand(*_clone_inputs(tuple(inputs)))
    cand_perm = cand(*_clone_inputs(tuple(perm_inputs)))
    agrees, snr, maxd = _agree(cand_perm, cand_base, cfg)
    if not agrees:
        return CheckResult("metamorphic_permutation", False,
                           f"candidate NOT permutation-invariant (SNR {snr:.1f} dB, "
                           f"max_diff {maxd:.3e}) though the reference is",
                           {"snr_db": snr, "max_diff": maxd})
    return CheckResult("metamorphic_permutation", True, "permutation-invariance preserved")


def metamorphic_homogeneity(cand: OutputFn, ref: OutputFn, make_inputs: InputFactory,
                            *, cfg: RobustConfig = RobustConfig(),
                            arg_index: int = 0) -> CheckResult:
    """Homogeneity (linear ops): ``f(a*x) == a*f(x)``. Verified applicable on the
    reference first (with a positive scale so relu-family ops qualify); then the
    candidate must satisfy it - a candidate whose scale response is wrong is flagged."""
    import torch

    a = float(cfg.metamorphic_alpha)
    inputs = list(make_inputs(cfg.base_seed))
    x = inputs[arg_index]
    if not torch.is_tensor(x):
        return CheckResult("metamorphic_homogeneity", True, "not applicable (arg not a tensor)")

    scaled = list(inputs)
    scaled[arg_index] = (x.double() * a).to(x.dtype)

    ref_base = _as_tuple(ref(*_clone_inputs(tuple(inputs))))
    ref_scaled = _as_tuple(ref(*_clone_inputs(tuple(scaled))))
    ref_expect = tuple((rb.double() * a) for rb in ref_base)
    ref_ok, _, _ = _agree(ref_scaled, ref_expect, cfg)
    if not ref_ok:
        return CheckResult("metamorphic_homogeneity", True,
                           "not applicable (reference not homogeneous under this scale)")

    cand_base = _as_tuple(cand(*_clone_inputs(tuple(inputs))))
    cand_scaled = _as_tuple(cand(*_clone_inputs(tuple(scaled))))
    cand_expect = tuple((cb.double() * a) for cb in cand_base)
    agrees, snr, maxd = _agree(cand_scaled, cand_expect, cfg)
    if not agrees:
        return CheckResult("metamorphic_homogeneity", False,
                           f"candidate breaks f(ax)=a f(x) (SNR {snr:.1f} dB, "
                           f"max_diff {maxd:.3e})", {"snr_db": snr, "max_diff": maxd})
    return CheckResult("metamorphic_homogeneity", True, "homogeneity preserved")


def metamorphic_additive_response(cand: OutputFn, ref: OutputFn, make_inputs: InputFactory,
                                  *, cfg: RobustConfig = RobustConfig(),
                                  arg_index: int = 0) -> CheckResult:
    """Additivity (fusions): the OUTPUT response to an additive perturbation of one
    input must match the reference's. Perturb argument ``arg_index`` by a fixed delta
    and require ``cand(x+d) - cand(x) == ref(x+d) - ref(x)``. A constant/memset kernel
    (zero response) or a mis-fused chain is flagged; a faithful fusion passes."""
    import torch

    inputs = list(make_inputs(cfg.base_seed))
    x = inputs[arg_index]
    if not torch.is_tensor(x):
        return CheckResult("metamorphic_additivity", True, "not applicable (arg not a tensor)")

    g = torch.Generator(device=x.device).manual_seed(cfg.base_seed + 11)
    delta = torch.randn(x.shape, generator=g, device=x.device, dtype=torch.float32).to(x.dtype)
    pert = list(inputs)
    pert[arg_index] = (x.double() + delta.double()).to(x.dtype)

    ref_resp = _as_tuple(ref(*_clone_inputs(tuple(pert))))[0].double() - \
        _as_tuple(ref(*_clone_inputs(tuple(inputs))))[0].double()
    cand_resp = _as_tuple(cand(*_clone_inputs(tuple(pert))))[0].double() - \
        _as_tuple(cand(*_clone_inputs(tuple(inputs))))[0].double()
    agrees, snr, maxd = _agree(cand_resp, ref_resp, cfg)
    if not agrees:
        return CheckResult("metamorphic_additivity", False,
                           f"candidate additive response differs (SNR {snr:.1f} dB, "
                           f"max_diff {maxd:.3e})", {"snr_db": snr, "max_diff": maxd})
    return CheckResult("metamorphic_additivity", True, "additive response matches reference")


# family -> the metamorphic relations that apply to it.
_METAMORPHIC_BY_FAMILY: dict[str, tuple] = {
    "gemm": (metamorphic_homogeneity,),
    "gemm_fusion": (metamorphic_homogeneity,),
    "matmul": (metamorphic_homogeneity,),
    "reduce": (metamorphic_permutation_invariance,),
    "reduction": (metamorphic_permutation_invariance,),
    "softmax": (metamorphic_permutation_invariance,),
    "fusion": (metamorphic_additive_response,),
}


def metamorphic_relations_for(family: Optional[str]) -> tuple:
    """The metamorphic checkers applicable to an operator ``family`` (empty if none)."""
    return _METAMORPHIC_BY_FAMILY.get((family or "").lower(), ())


# --------------------------------------------------------------------------- #
# Orchestrator: run the battery, halt on first mismatch.
# --------------------------------------------------------------------------- #
def robust_correctness(cand: OutputFn, ref: OutputFn, make_inputs: InputFactory, *,
                       family: Optional[str] = None,
                       cfg: RobustConfig = RobustConfig(),
                       extra_metamorphic: Sequence[Callable] = (),
                       run_metamorphic: bool = True) -> RobustReport:
    """Run the maximum-scrutiny correctness battery, HALTING ON THE FIRST MISMATCH.

    Order: random inits -> adversarial regimes -> non-contiguous -> differential
    oracle -> metamorphic relations (those applicable to ``family`` plus any in
    ``extra_metamorphic``). Returns a :class:`RobustReport` whose ``passed`` is True
    only if every applicable check passed; ``failed_check`` names the first failure.
    """
    checks: list[CheckResult] = []

    def _run(fn) -> Optional[RobustReport]:
        res = fn(cand, ref, make_inputs, cfg=cfg)
        checks.append(res)
        if not res.passed:
            return RobustReport(passed=False, checks=checks, failed_check=res.name)
        return None

    for fn in (check_random_inits, check_adversarial_regimes,
               check_noncontiguous, check_differential_oracle):
        halted = _run(fn)
        if halted is not None:
            return halted

    if run_metamorphic:
        relations = tuple(metamorphic_relations_for(family)) + tuple(extra_metamorphic)
        for fn in relations:
            halted = _run(fn)
            if halted is not None:
                return halted

    return RobustReport(passed=True, checks=checks, failed_check=None)


# --------------------------------------------------------------------------- #
# Bridge: build an InputFactory from a KernelBench spec (integration helper).
# --------------------------------------------------------------------------- #
def inputs_factory_from_spec(spec, shape=None, *, device: str = "cpu") -> InputFactory:
    """Adapt a :class:`kore.eval.kernelbench_amd.KernelBenchSpec` to an InputFactory.

    Uses the spec's first shape by default. Lets the robust battery run directly over
    a KernelBench problem's reference + input generator (``robust_correctness(
    spec.reference, ..., inputs_factory_from_spec(spec), family=spec.family)``).
    """
    shp = shape if shape is not None else spec.input_shapes[0]

    def factory(seed: int) -> tuple:
        return spec.make_inputs(shp, device, seed)

    return factory


__all__ = [
    "RobustConfig",
    "CheckResult",
    "RobustReport",
    "compare_outputs",
    "adversarial_fills",
    "check_random_inits",
    "check_adversarial_regimes",
    "check_noncontiguous",
    "check_differential_oracle",
    "metamorphic_permutation_invariance",
    "metamorphic_homogeneity",
    "metamorphic_additive_response",
    "metamorphic_relations_for",
    "robust_correctness",
    "inputs_factory_from_spec",
]
