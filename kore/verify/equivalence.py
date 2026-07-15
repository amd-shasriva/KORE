"""Verification-in-the-loop correctness oracle for KORE.

This is the *verified reward* half of KORE's verifiably-grounded kernel paradigm.
The shipped correctness gate (``kore/reward/reward.py`` + ``kore/env/kore_env.py``)
accepts a candidate kernel when it clears an SNR threshold on a handful (5) of
reseeded random trials plus one determinism re-check. That gate is cheap and
effective, but it leaves a well-known reward-hacking surface open:

  * **Lucky pass** - a kernel that is wrong on a thin slice of the input domain
    (e.g. exactly ``x == 0``, denormals, near-``inf`` magnitudes, all-equal rows)
    can clear an SNR gate whose random draws essentially never land in that slice.
  * **Edge-case miss** - the naive random distribution (``randn``) under-samples the
    structured regimes where numerical kernels actually break (overflow knots,
    activation kinks at ``0/±1/±3/±6``, masked tail elements, sparse spikes).

This module replaces/augments the single-SNR verdict with a **multi-pronged
equivalence oracle** that is far stronger and much harder to hack. It is:

  * **Self-contained** - new files under ``kore/verify/`` only; nothing here is wired
    into the live reward yet.
  * **CPU-testable** - the *decision logic* (:func:`equivalence_verdict`) is a pure
    function over arrays of candidate/reference outputs, so the entire accept/reject
    behaviour is unit-testable on CPU with numpy (no GPU, no Triton). ``torch`` is
    imported lazily, only inside the GPU-facing orchestration paths.

The four prongs
---------------
1. **Random (statistical).** Many reseeded random trials (default 64, vs the gate's
   5), each checked with BOTH a *tight per-element relative-error bound* and an
   aggregate SNR floor. This drives the statistical false-accept probability down
   exponentially in the number of element comparisons.
2. **Adversarial (deterministic / exhaustive over a curated set).** A fixed battery
   of structured inputs - zeros, ones, ``±1``, all-equal, large/small magnitudes,
   denormals, ``±inf``-adjacent, sign-alternating, sparse spikes, a signed ramp, and
   activation-knot boundaries - that deterministically exercises exactly the regimes
   random sampling misses. See :func:`kore.verify.adversarial.adversarial_inputs`.
3. **Metamorphic (deterministic / structural).** Algebraic relations the *true* op
   must satisfy regardless of its point values (elementwise permutation- &
   reshape-equivariance and block/locality; order-invariance and row-independence for
   reductions). These are candidate-only self-consistency checks that catch
   structural cheats (e.g. a "pointwise" kernel that secretly reduces across
   elements) even when point values look right. See
   :func:`kore.verify.metamorphic.metamorphic_relations`.
4. **Determinism (deterministic).** Repeated runs on identical input must agree to
   within a tight tolerance, so a partly-random-output kernel that clears a gate by
   luck is rejected (mirrors ``kore_env._determinism_stable``, but per-element).

What is PROVABLE vs STATISTICAL (read :func:`false_accept_probability` too)
---------------------------------------------------------------------------
Floating-point kernels cannot be bit-exact against an fp64 oracle, so this oracle is
"sound-ish", not a formal proof of functional equality. Concretely:

  * **PROVABLE (deterministic prongs).** For the *checkable op class* - pure
    elementwise unary/binary maps and order-invariant per-row reductions - a
    candidate that is wrong on ANY point in the curated adversarial set, or that
    violates ANY metamorphic identity, or that is non-deterministic, is rejected
    **with certainty** (no luck involved): those prongs re-run the same fixed inputs
    every time. This is what kills the *lucky-pass* class: the canonical hard regimes
    (zeros / denormals / overflow knots / all-equal / sparse / boundary shapes) are
    checked exhaustively rather than sampled, so a kernel cannot "get lucky" and skip
    them. That is the supported claim: *provably no lucky-pass on the checkable op
    class via exhaustive adversarial + tight-bound multi-trial + metamorphic
    verification.*
  * **STATISTICAL (random prong).** For value-dependent defects that survive every
    deterministic prong, detection is probabilistic. If a defect manifests on a
    fraction ``p`` of the element domain, the probability it survives ``m`` independent
    in-tolerance element comparisons is at most ``(1 - p)**m`` (see
    :func:`false_accept_probability`). With ``m`` in the millions (64 trials ×
    thousands of elements) even a ``p = 1e-4`` defect is caught with overwhelming
    probability. This is a bound on *lucky* random misses, not a proof of equality.

Honest false-accept characterisation. A wrong kernel is accepted only if it
simultaneously (a) is deterministic, (b) satisfies every metamorphic identity of the
op class, (c) agrees with the oracle on every adversarial regime, AND (d) differs from
the oracle on a random-domain set of measure so small that all ``m`` comparisons stay
in tolerance - a joint event bounded by ``(1 - p)**m`` in the statistical prong and
identically zero in the deterministic prongs for the enumerated regimes. The returned
:class:`VerificationResult` reports the worst per-element relative error, the worst
SNR, which prongs passed, and the numeric false-accept bound at a reference defect
fraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "Tolerance",
    "tolerance_for",
    "PairComparison",
    "compare_pair",
    "ProngSamples",
    "ProngResult",
    "VerificationResult",
    "equivalence_verdict",
    "false_accept_probability",
    "verify_equivalence",
]


# --------------------------------------------------------------------------- #
# Tolerances (dtype-aware, mirrors kore.config.snr_threshold_for spirit)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tolerance:
    """Per-prong acceptance tolerance.

    A prong passes only when BOTH bounds hold on every (candidate, reference) pair:
      * worst per-element relative error ``|c - r| / (|r| + atol) <= rtol``  (tight,
        catches localised defects an aggregate norm would average away), and
      * aggregate ``SNR = 20*log10(||r|| / ||c - r||) >= snr_db_min``        (global,
        catches broad low-amplitude drift), plus exact agreement on non-finite
        (``nan`` / ``±inf``) positions.
    """

    rtol: float = 3e-3
    atol: float = 1e-4
    snr_db_min: float = 50.0
    # determinism is checked against the kernel's OWN first run (same inputs), so it
    # must be near-exact; a partly-random kernel blows past this immediately.
    determinism_rtol: float = 1e-5
    determinism_snr_db_min: float = 80.0
    # metamorphic identities compose two candidate evaluations, so allow a hair more
    # fp slack than a single point comparison.
    metamorphic_rtol: float = 6e-3
    metamorphic_snr_db_min: float = 46.0
    # reference defect fraction used to report the headline statistical bound.
    reference_defect_fraction: float = 1e-4


def tolerance_for(dtype: str) -> Tolerance:
    """Return a :class:`Tolerance` calibrated for ``dtype``.

    fp32 gets a tight bound; low-precision (fp16/bf16/fp8/fp4) storage rounds to a
    handful of mantissa bits, so the per-element bound and SNR floors are relaxed to
    the levels the KORE gate itself uses (``snr_threshold_lowp``-class values).
    """
    d = (dtype or "").lower()
    low = any(t in d for t in ("fp16", "bf16", "float16", "bfloat16",
                               "fp8", "fp4", "mxfp4", "mxfp8"))
    if low:
        return Tolerance(
            rtol=3e-2, atol=1e-2, snr_db_min=30.0,
            determinism_rtol=1e-3, determinism_snr_db_min=40.0,
            metamorphic_rtol=5e-2, metamorphic_snr_db_min=28.0,
        )
    return Tolerance()


# --------------------------------------------------------------------------- #
# Array helpers (numpy is the comparison substrate; torch is lazy elsewhere)
# --------------------------------------------------------------------------- #
def _to_f64(a) -> np.ndarray:
    """Coerce a numpy array or torch CPU/GPU tensor to a float64 numpy array.

    torch is imported lazily and only when the object actually looks like a tensor,
    so importing this module never requires torch (let alone a GPU). bf16/fp16 are
    upcast on the torch side because numpy has no native bf16.
    """
    if isinstance(a, np.ndarray):
        return a.astype(np.float64, copy=False)
    mod = type(a).__module__ or ""
    if mod.startswith("torch"):
        import torch  # lazy

        if isinstance(a, torch.Tensor):
            return a.detach().to(dtype=torch.float64, device="cpu").numpy()
    return np.asarray(a, dtype=np.float64)


def _snr_db(noise: float, signal: float) -> float:
    if noise == 0.0:
        return math.inf
    if signal == 0.0:
        return -math.inf
    return 20.0 * math.log10(signal / noise)


# --------------------------------------------------------------------------- #
# Pure per-pair comparison
# --------------------------------------------------------------------------- #
@dataclass
class PairComparison:
    """Result of comparing one (candidate, reference) output pair."""

    ok: bool
    worst_rel_err: float
    snr_db: float
    n_elements: int
    reason: str = ""


def compare_pair(actual, expected, tol: Tolerance,
                 rtol: Optional[float] = None,
                 snr_db_min: Optional[float] = None) -> PairComparison:
    """Compare one output pair under ``tol`` (pure; numpy/torch-cpu arrays).

    ``rtol`` / ``snr_db_min`` override the tolerance's defaults (used for the tighter
    determinism / looser metamorphic prongs). Non-finite handling is strict: a defect
    that produces a ``nan`` where the oracle is finite (or a ``+inf`` where the oracle
    is ``-inf``) is a hard failure with ``worst_rel_err = inf``.
    """
    r = tol.rtol if rtol is None else rtol
    s = tol.snr_db_min if snr_db_min is None else snr_db_min

    a = _to_f64(actual)
    e = _to_f64(expected)
    if a.shape != e.shape:
        return PairComparison(False, math.inf, -math.inf, e.size,
                              f"shape mismatch {a.shape} vs {e.shape}")

    af, ef = a.ravel(), e.ravel()
    n = ef.size
    if n == 0:
        return PairComparison(True, 0.0, math.inf, 0, "empty")

    fin = np.isfinite(af) & np.isfinite(ef)
    nonfin = ~fin
    # Non-finite positions must agree EXACTLY (inf sign, or both nan).
    nonfinite_ok = True
    if nonfin.any():
        an, en = af[nonfin], ef[nonfin]
        agree = (an == en) | (np.isnan(an) & np.isnan(en))
        nonfinite_ok = bool(agree.all())

    if fin.any():
        diff = np.abs(af[fin] - ef[fin])
        rel = diff / (np.abs(ef[fin]) + tol.atol)
        worst_rel = float(rel.max())
        noise = float(np.linalg.norm(diff))
        signal = float(np.linalg.norm(ef[fin]))
        snr = _snr_db(noise, signal)
    else:
        worst_rel = 0.0
        snr = math.inf

    if not nonfinite_ok:
        worst_rel = math.inf

    ok = nonfinite_ok and worst_rel <= r and snr >= s
    reason = ""
    if not ok:
        if not nonfinite_ok:
            reason = "non-finite (nan/inf) mismatch vs reference"
        elif worst_rel > r:
            reason = f"per-element rel-err {worst_rel:.3e} > rtol {r:.1e}"
        else:
            reason = f"SNR {snr:.1f} dB < min {s:.1f} dB"
    return PairComparison(ok, worst_rel, snr, n, reason)


# --------------------------------------------------------------------------- #
# Prongs: uniform (actual, expected) representation
# --------------------------------------------------------------------------- #
# Every prong reduces to a list of (actual, expected) array pairs that must be equal
# within tolerance, so the decision logic is one uniform code path:
#   random       : (candidate_out, reference_out)          per reseeded trial
#   adversarial  : (candidate_out, reference_out)          per structured input
#   metamorphic  : (transformed_candidate, relation_rhs)   per identity
#   determinism  : (candidate_run_i, candidate_run_0)      per repeat
_KIND_TOL = {
    "random": (None, None),
    "adversarial": (None, None),
    "metamorphic": ("metamorphic_rtol", "metamorphic_snr_db_min"),
    "determinism": ("determinism_rtol", "determinism_snr_db_min"),
}


@dataclass
class ProngSamples:
    """Input to the decision logic: one prong's (actual, expected) array pairs."""

    name: str
    kind: str  # "random" | "adversarial" | "metamorphic" | "determinism"
    pairs: Sequence[tuple]  # each: (actual_array, expected_array)
    required: bool = True
    labels: Optional[Sequence[str]] = None  # optional per-pair names (for diagnostics)


@dataclass
class ProngResult:
    """Per-prong verdict."""

    name: str
    kind: str
    passed: bool
    n_pairs: int
    n_elements: int
    worst_rel_err: float
    worst_snr_db: float
    required: bool = True
    detail: str = ""


@dataclass
class VerificationResult:
    """Structured multi-pronged equivalence verdict.

    ``verified`` is the headline accept/reject. ``confidence`` is ``1 -
    false_accept_bound`` where the bound is the statistical random-prong miss
    probability at ``Tolerance.reference_defect_fraction`` (the deterministic prongs
    contribute certainty for the enumerated regimes, not probability). ``prongs``
    records which of the four prongs passed and their worst-case numbers.
    """

    verified: bool
    confidence: float
    prongs: list[ProngResult]
    worst_rel_err: float
    worst_snr_db: float
    false_accept_bound: float
    n_random_trials: int
    n_random_elements: int
    detail: str = ""
    extra: dict = field(default_factory=dict)

    def prong(self, name: str) -> Optional[ProngResult]:
        for p in self.prongs:
            if p.name == name:
                return p
        return None

    def passed_prongs(self) -> list[str]:
        return [p.name for p in self.prongs if p.passed]

    def failed_prongs(self) -> list[str]:
        return [p.name for p in self.prongs if not p.passed]

    def false_accept_probability(self, defect_fraction: float) -> float:
        """Statistical upper bound that a defect on ``defect_fraction`` of the element
        domain survives the random prong (``(1 - p)**m``). See module docstring."""
        return false_accept_probability(defect_fraction, self.n_random_elements)

    def summary(self) -> str:
        tag = "VERIFIED" if self.verified else "REJECTED"
        parts = [
            f"[{tag}] confidence={self.confidence:.6f} "
            f"worst_rel_err={self.worst_rel_err:.3e} worst_snr={self.worst_snr_db:.1f}dB "
            f"false_accept<={self.false_accept_bound:.2e}"
        ]
        for p in self.prongs:
            mark = "pass" if p.passed else "FAIL"
            parts.append(
                f"  - {p.name:<12} [{mark}] pairs={p.n_pairs} "
                f"worst_rel_err={p.worst_rel_err:.3e} snr={p.worst_snr_db:.1f}dB"
                + (f"  ({p.detail})" if p.detail else "")
            )
        if self.detail:
            parts.append(f"  {self.detail}")
        return "\n".join(parts)


def false_accept_probability(defect_fraction: float, n_elements: int) -> float:
    """``(1 - p)**m``: the max probability a defect present on fraction ``p`` of the
    element domain survives ``m`` independent in-tolerance element comparisons.

    This bounds the *statistical* (random-prong) false-accept only. The deterministic
    prongs (adversarial / metamorphic / determinism) add certainty for the enumerated
    regimes and are NOT part of this bound.
    """
    p = min(max(float(defect_fraction), 0.0), 1.0)
    m = max(int(n_elements), 0)
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if m > 0 else 1.0
    return math.exp(m * math.log1p(-p))


# --------------------------------------------------------------------------- #
# THE DECISION LOGIC (pure; fully CPU-unit-testable)
# --------------------------------------------------------------------------- #
def equivalence_verdict(prong_results: Sequence[ProngSamples],
                        tol: Tolerance) -> VerificationResult:
    """Pure accept/reject decision given per-prong (candidate, reference) arrays.

    ``prong_results`` is a list of :class:`ProngSamples`; each carries the raw
    ``(actual, expected)`` output pairs for one prong (see the prong table above).
    No GPU / torch / kernel execution happens here - this is exactly the surface the
    unit tests drive with synthetic arrays.

    A candidate is ``verified`` iff **every required prong passes**. A prong passes iff
    every one of its pairs satisfies the tight per-element rel-err bound AND the SNR
    floor for that prong's kind (determinism tighter, metamorphic slightly looser),
    with strict non-finite agreement.
    """
    prongs: list[ProngResult] = []
    worst_rel_all = 0.0
    worst_snr_all = math.inf
    n_random_trials = 0
    n_random_elements = 0

    for ps in prong_results:
        rtol_attr, snr_attr = _KIND_TOL.get(ps.kind, (None, None))
        rtol = getattr(tol, rtol_attr) if rtol_attr else None
        snr_min = getattr(tol, snr_attr) if snr_attr else None

        prong_ok = True
        prong_worst_rel = 0.0
        prong_worst_snr = math.inf
        n_pairs = 0
        n_elems = 0
        fail_reason = ""
        for i, pair in enumerate(ps.pairs):
            actual, expected = pair
            cmp = compare_pair(actual, expected, tol, rtol=rtol, snr_db_min=snr_min)
            n_pairs += 1
            n_elems += cmp.n_elements
            prong_worst_rel = max(prong_worst_rel, cmp.worst_rel_err)
            prong_worst_snr = min(prong_worst_snr, cmp.snr_db)
            if not cmp.ok and prong_ok:
                prong_ok = False
                label = (ps.labels[i] if ps.labels and i < len(ps.labels)
                         else f"case[{i}]")
                fail_reason = f"{label}: {cmp.reason}"

        prongs.append(ProngResult(
            name=ps.name, kind=ps.kind, passed=prong_ok, n_pairs=n_pairs,
            n_elements=n_elems, worst_rel_err=prong_worst_rel,
            worst_snr_db=prong_worst_snr, required=ps.required, detail=fail_reason,
        ))
        # aggregate worst-case across all prongs (finite pairs only for snr)
        worst_rel_all = max(worst_rel_all, prong_worst_rel)
        if math.isfinite(prong_worst_snr):
            worst_snr_all = min(worst_snr_all, prong_worst_snr)
        if ps.kind == "random":
            n_random_trials += n_pairs
            n_random_elements += n_elems

    if not math.isfinite(worst_snr_all):
        worst_snr_all = math.inf

    verified = all(p.passed for p in prongs if p.required) and len(prongs) > 0
    fa_bound = false_accept_probability(tol.reference_defect_fraction, n_random_elements)
    confidence = (1.0 - fa_bound) if verified else 0.0

    failed = [p.name for p in prongs if not p.passed and p.required]
    if verified:
        detail = (f"all {len(prongs)} prongs passed; "
                  f"random={n_random_trials} trials / {n_random_elements} elems")
    elif not prongs:
        detail = "no prongs supplied"
    else:
        first = next((p for p in prongs if not p.passed and p.required), None)
        detail = "rejected by prong(s): " + ", ".join(failed)
        if first and first.detail:
            detail += f" - {first.detail}"

    return VerificationResult(
        verified=verified, confidence=confidence, prongs=prongs,
        worst_rel_err=worst_rel_all, worst_snr_db=worst_snr_all,
        false_accept_bound=fa_bound, n_random_trials=n_random_trials,
        n_random_elements=n_random_elements, detail=detail,
    )


# --------------------------------------------------------------------------- #
# Orchestrator: run the kernel, collect prong samples, then apply the verdict
# --------------------------------------------------------------------------- #
def _call(fn: Callable, inputs: tuple):
    """Call ``fn(*inputs)`` and return the output, or an Exception instance."""
    try:
        return fn(*inputs)
    except Exception as exc:  # noqa: BLE001 - a crashing candidate is a rejection
        return exc


def _err_pair(exc: Exception, ref_out):
    """Represent a candidate crash as a guaranteed-failing (nan, reference) pair."""
    e = _to_f64(ref_out)
    bad = np.full_like(e, np.nan)
    return (bad, e)


def verify_equivalence(
    candidate_fn: Callable,
    reference_fn: Callable,
    input_gen: Callable,
    dtype: str = "fp32",
    *,
    shape=None,
    op_class: str = "elementwise",
    arity: Optional[int] = None,
    n_random: int = 64,
    n_determinism: int = 3,
    device: str = "cpu",
    tol: Optional[Tolerance] = None,
    adversarial: bool = True,
    metamorphic: bool = True,
    seed0: int = 0,
) -> VerificationResult:
    """Run the full multi-pronged equivalence check on a candidate kernel.

    Parameters
    ----------
    candidate_fn, reference_fn
        ``(*inputs) -> array``. ``reference_fn`` is the fp64/fp32 oracle. Both may run
        on ``device`` (``"cpu"`` for tests, ``"cuda"`` in the live loop); outputs are
        moved to CPU/float64 for the verdict.
    input_gen
        ``(shape, dtype, seed, device) -> tuple[inputs]`` - matches the KORE task
        ``get_inputs`` convention (extended with an explicit ``device``/``dtype``).
    shape
        Passed straight to ``input_gen`` / the generators (a dict like ``{"M":..,
        "N":..}`` or a tuple). Defaults to ``(64, 128)`` when omitted.
    op_class
        ``"elementwise"`` | ``"reduction"`` | ``"generic"`` - selects the adversarial
        battery layout and the metamorphic identity set.
    arity
        Number of input operands. Inferred from ``input_gen`` output if omitted.

    Returns a :class:`VerificationResult`. ``torch`` is only touched (lazily) if the
    kernels/generators use it; the orchestration itself is torch-free.
    """
    if shape is None:
        shape = (64, 128)
    tol = tol or tolerance_for(dtype)

    # infer arity from a probe draw
    probe = input_gen(shape, dtype, seed0, device)
    if not isinstance(probe, (tuple, list)):
        probe = (probe,)
    probe = tuple(probe)
    if arity is None:
        arity = len(probe)

    prongs: list[ProngSamples] = []

    # ---- 1. random prong (statistical) ------------------------------------ #
    rnd_pairs = []
    rnd_labels = []
    for t in range(n_random):
        inputs = input_gen(shape, dtype, seed0 + 1000 + t, device)
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        inputs = tuple(inputs)
        ref_out = reference_fn(*inputs)
        cand = _call(candidate_fn, inputs)
        if isinstance(cand, Exception):
            rnd_pairs.append(_err_pair(cand, ref_out))
            rnd_labels.append(f"trial{t}:{type(cand).__name__}")
        else:
            rnd_pairs.append((cand, ref_out))
            rnd_labels.append(f"trial{t}")
    prongs.append(ProngSamples("random", "random", rnd_pairs, labels=rnd_labels))

    # ---- 2. adversarial prong (deterministic) ------------------------------ #
    if adversarial:
        from kore.verify.adversarial import adversarial_inputs

        adv_pairs = []
        adv_labels = []
        for name, inputs in adversarial_inputs(shape, dtype, arity=arity,
                                               op_class=op_class, device=device):
            ref_out = reference_fn(*inputs)
            cand = _call(candidate_fn, inputs)
            if isinstance(cand, Exception):
                adv_pairs.append(_err_pair(cand, ref_out))
                adv_labels.append(f"{name}:{type(cand).__name__}")
            else:
                adv_pairs.append((cand, ref_out))
                adv_labels.append(name)
        prongs.append(ProngSamples("adversarial", "adversarial", adv_pairs,
                                   labels=adv_labels))

    # ---- 3. metamorphic prong (deterministic / structural) ----------------- #
    if metamorphic:
        from kore.verify.metamorphic import metamorphic_relations

        meta_pairs = []
        meta_labels = []
        base_inputs = input_gen(shape, dtype, seed0 + 7, device)
        if not isinstance(base_inputs, (tuple, list)):
            base_inputs = (base_inputs,)
        base_inputs = tuple(base_inputs)
        for rel in metamorphic_relations(op_class):
            try:
                lhs, rhs = rel.apply(candidate_fn, base_inputs)
            except Exception as exc:  # noqa: BLE001
                e = _to_f64(reference_fn(*base_inputs))
                lhs, rhs = np.full_like(e, np.nan), e
                meta_labels.append(f"{rel.name}:{type(exc).__name__}")
            else:
                meta_labels.append(rel.name)
            meta_pairs.append((lhs, rhs))
        if meta_pairs:
            prongs.append(ProngSamples("metamorphic", "metamorphic", meta_pairs,
                                       labels=meta_labels))

    # ---- 4. determinism prong (deterministic) ------------------------------ #
    if n_determinism >= 2:
        det_inputs = input_gen(shape, dtype, seed0 + 13, device)
        if not isinstance(det_inputs, (tuple, list)):
            det_inputs = (det_inputs,)
        det_inputs = tuple(det_inputs)
        runs = []
        for _ in range(n_determinism):
            out = _call(candidate_fn, det_inputs)
            runs.append(out)
        det_pairs = []
        det_labels = []
        run0 = runs[0]
        if isinstance(run0, Exception):
            # already caught as a crash in the random prong; represent as a failure
            det_pairs.append((np.array([np.nan]), np.array([0.0])))
            det_labels.append(f"run0:{type(run0).__name__}")
        else:
            for i in range(1, len(runs)):
                ri = runs[i]
                if isinstance(ri, Exception):
                    e = _to_f64(run0)
                    det_pairs.append((np.full_like(e, np.nan), e))
                    det_labels.append(f"run{i}:{type(ri).__name__}")
                else:
                    det_pairs.append((ri, run0))
                    det_labels.append(f"run{i}_vs_run0")
        if det_pairs:
            prongs.append(ProngSamples("determinism", "determinism", det_pairs,
                                       labels=det_labels))

    return equivalence_verdict(prongs, tol)
