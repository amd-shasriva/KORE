"""Adversarial / structured input battery for the equivalence oracle.

Random ``randn`` sampling - what the shipped SNR gate uses - under-samples exactly
the regimes where numerical kernels break: exact zeros, sign boundaries, overflow
knots, denormals, all-equal rows, sparse spikes. A *lucky-pass* kernel is one that is
wrong only on such a thin slice and never gets caught by random draws. This module
enumerates those regimes DETERMINISTICALLY, so a kernel that is wrong on any of them
is rejected with certainty (not luck) - the provable half of the oracle for the
checkable op class.

Pure numpy generation with a lazy torch cast (so importing this module never needs a
GPU). Values are produced in float64 and cast to the task ``dtype`` on the requested
``device`` (numpy on CPU, torch on cuda / when torch tensors are requested).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Optional

import numpy as np

# The coevolution layer (appended below) reuses the oracle's pure comparison logic and
# dtype-aware tolerances. This import is safe: ``equivalence`` has no top-level
# dependency on this module (it imports ``adversarial_inputs`` lazily inside
# ``verify_equivalence``), and ``kore.verify.__init__`` always loads ``equivalence``
# before ``adversarial``, so there is no import cycle. Mirrors ``metamorphic.py``.
from kore.verify.equivalence import Tolerance, _to_f64, compare_pair, tolerance_for

__all__ = ["adversarial_patterns", "adversarial_inputs", "dtype_extremes", "dtype_max"]


def dtype_extremes(dtype: str) -> tuple[float, float, float]:
    """Return ``(big, small, tiny)`` magnitudes safe for ``dtype``.

    ``big`` is a large finite value chosen so that even an amplifying op (``x*x``, a
    128-wide row sum) stays finite in ``dtype`` - so it stresses magnitude without
    *gratuitously* overflowing benign kernels. ``small`` is a small-but-normal value
    and ``tiny`` a denormal-ish subnormal magnitude. (The dedicated ``inf_adjacent``
    regime, at the dtype's max finite value, is where overflow/saturation is probed.)
    """
    d = (dtype or "").lower()
    if "fp16" in d or "float16" in d:
        return 1.0e2, 1.0e-3, 6.0e-8      # fp16 max 65504; 1e2^2=1e4 stays finite
    if "bf16" in d or "bfloat16" in d:
        return 1.0e18, 1.0e-18, 9.0e-41   # bf16 shares fp32 exponent range
    if "fp8" in d or "fp4" in d or "mxfp4" in d or "mxfp8" in d:
        return 6.0e0, 1.0e-2, 1.0e-3      # fp8 e4m3 max ~448
    return 1.0e18, 1.0e-18, 1.0e-40       # fp32: 1e18^2=1e36 < 3.4e38


def dtype_max(dtype: str) -> float:
    """Largest finite value representable in ``dtype`` (for the inf-adjacent regime)."""
    d = (dtype or "").lower()
    if "fp16" in d or "float16" in d:
        return float(np.finfo(np.float16).max)      # 65504
    if "bf16" in d or "bfloat16" in d:
        return 3.3895313892515355e38                # bf16 max (fp32 exponent, 8-bit mant)
    if "fp8" in d or "fp4" in d or "mxfp4" in d or "mxfp8" in d:
        return 448.0                                # fp8 e4m3fn max
    return float(np.finfo(np.float32).max)          # 3.4028235e38


def adversarial_patterns(shape, dtype: str, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """Return a list of ``(name, float64 ndarray)`` structured single-operand tensors.

    The battery (each of ``shape``): zeros, ones, negative ones, a nonzero constant
    (all-equal), large ``±`` magnitudes, small magnitudes, denormal-ish, a signed ramp
    that crosses every activation knot, sign-alternating checkerboard, a sparse-spike
    tensor (mostly zero), ``±`` inf-adjacent (the dtype's max finite value), explicit
    activation-knot values (``0, ±1, ±3, ±6, ±0.5, 2.0``), and a mixed-magnitude tile.
    """
    dims = _dims(shape)
    n = int(np.prod(dims)) if dims else 1
    big, small, tiny = dtype_extremes(dtype)
    rng = np.random.default_rng(seed)

    def full(v):
        return np.full(dims, float(v), dtype=np.float64)

    pats: list[tuple[str, np.ndarray]] = [
        ("zeros", full(0.0)),
        ("ones", full(1.0)),
        ("neg_ones", full(-1.0)),
        ("all_equal_const", full(0.7)),
        ("large_pos", full(big)),
        ("large_neg", full(-big)),
        ("small_pos", full(small)),
        ("denormal", full(tiny)),
    ]

    # signed ramp across [-big_ramp, +big_ramp]: crosses 0 and all activation knots.
    big_ramp = min(big, 8.0) if ("fp16" in dtype.lower()) else 8.0
    ramp = np.linspace(-big_ramp, big_ramp, n, dtype=np.float64).reshape(dims)
    pats.append(("signed_ramp", ramp))

    # sign-alternating checkerboard of unit magnitude.
    alt = np.where((np.arange(n) % 2 == 0), 1.0, -1.0).astype(np.float64).reshape(dims)
    pats.append(("sign_alternating", alt))

    # sparse: almost all zero, a few large spikes (both signs).
    sparse = np.zeros(n, dtype=np.float64)
    if n > 0:
        idx = rng.choice(n, size=max(1, n // 64), replace=False)
        sparse[idx] = rng.choice([-1.0, 1.0], size=idx.size) * (big / 1e3 if big > 1e3 else big)
    pats.append(("sparse_spikes", sparse.reshape(dims)))

    # inf-adjacent: the dtype's max finite value (±). One more amplifying op tips to
    # inf, so this probes overflow/saturation handling; the verdict matches inf/nan
    # positions EXACTLY, so a kernel must saturate the same way the oracle does.
    mx = dtype_max(dtype)
    pats.append(("inf_adjacent_pos", full(mx)))
    pats.append(("inf_adjacent_neg", full(-mx)))

    # explicit activation knots tiled across the tensor (0, ±1, ±3, ±6, ±0.5, 2.0).
    knots = np.array([0.0, 1.0, -1.0, 3.0, -3.0, 6.0, -6.0, 0.5, -0.5, 2.0],
                     dtype=np.float64)
    tiled = np.resize(knots, n).reshape(dims)
    pats.append(("activation_knots", tiled))

    # mixed magnitudes in one tensor (small + large + normal interleaved).
    mixed = np.resize(np.array([small, big / 1e3 if big > 1e3 else big, 1.0, -1.0,
                                small, -big / 1e3 if big > 1e3 else -big], dtype=np.float64),
                      n).reshape(dims)
    pats.append(("mixed_magnitude", mixed))

    return pats


def adversarial_inputs(shape, dtype: str, arity: int = 1,
                       op_class: str = "elementwise",
                       device: str = "cpu",
                       seed: int = 0) -> Iterator[tuple[str, tuple]]:
    """Yield ``(case_name, inputs_tuple)`` adversarial cases for an ``arity``-operand op.

    For each structured pattern we emit:
      * an **all-operands** case (every operand set to the pattern), and
      * for ``arity > 1``, a **single-operand** case per operand slot (that slot gets
        the pattern; the others get a fixed benign random draw), so a defect that only
        triggers on, say, a zero *second* operand is still exercised.

    Operand-0 patterns are additionally biased positive for the positive-domain ops
    (sqrt/log/reciprocal/rsqrt) is the caller's concern; the reference oracle defines
    the truth, and non-finite outputs (e.g. ``log(0) = -inf``) are matched exactly by
    the verdict's strict non-finite handling.
    """
    pats = adversarial_patterns(shape, dtype, seed=seed)
    rng = np.random.default_rng(seed + 101)
    dims = _dims(shape)
    benign = rng.standard_normal(dims).astype(np.float64) if dims else np.array([0.3])

    for name, arr in pats:
        # all operands = pattern
        inputs = tuple(_cast(arr, dtype, device) for _ in range(arity))
        yield (f"all::{name}", inputs)
        # single-operand injection (only meaningful for arity > 1)
        if arity > 1:
            for slot in range(arity):
                ops = []
                for j in range(arity):
                    ops.append(_cast(arr if j == slot else benign, dtype, device))
                yield (f"op{slot}::{name}", tuple(ops))


# --------------------------------------------------------------------------- #
# shape / cast helpers
# --------------------------------------------------------------------------- #
def _dims(shape) -> tuple:
    """Normalise a KORE shape (dict ``{'M':..,'N':..[,'K':..]}`` or a tuple) to dims."""
    if shape is None:
        return (64, 128)
    if isinstance(shape, dict):
        if "M" in shape and "N" in shape:
            return (int(shape["M"]), int(shape["N"]))
        return tuple(int(v) for v in shape.values())
    if isinstance(shape, (tuple, list)):
        return tuple(int(v) for v in shape)
    return (int(shape),)


def _cast(arr_f64: np.ndarray, dtype: str, device: str):
    """Cast a float64 numpy array to ``dtype`` on ``device`` (numpy CPU or torch)."""
    if device == "cpu":
        np_dt = _numpy_dtype(dtype)
        if np_dt is not None:
            return arr_f64.astype(np_dt)
        # bf16 / fp8 have no numpy dtype -> fall through to torch even on cpu.
    import torch  # lazy

    tdt = _torch_dtype(dtype)
    t = torch.from_numpy(np.ascontiguousarray(arr_f64))
    return t.to(device=device, dtype=tdt)


def _numpy_dtype(dtype: str):
    d = (dtype or "").lower()
    if "fp16" in d or "float16" in d:
        return np.float16
    if "fp32" in d or "float32" in d:
        return np.float32
    if "fp64" in d or "float64" in d:
        return np.float64
    if "bf16" in d or "bfloat16" in d:
        return None  # numpy has no bf16
    return np.float32


def _torch_dtype(dtype: str):
    import torch

    d = (dtype or "").lower()
    if "fp16" in d or "float16" in d:
        return torch.float16
    if "bf16" in d or "bfloat16" in d:
        return torch.bfloat16
    if "fp32" in d or "float32" in d:
        return torch.float32
    if "fp64" in d or "float64" in d:
        return torch.float64
    return torch.float32


# =========================================================================== #
# Coevolutionary adversarial test-case generation   (ADDITIVE · OFF BY DEFAULT)
# =========================================================================== #
# The battery above (``adversarial_patterns`` / ``adversarial_inputs``) is a FIXED,
# hand-curated enumeration. Its guarantee is exactly its limitation: it can only ever
# reject a kernel that is wrong on a regime someone thought to write down. A kernel
# wrong on a regime NOT in that list - e.g. a near-tie at a reduction's argmax, an
# activation-kink neighbourhood the ramp steps over, a magnitude between the "safe big"
# and "inf-adjacent" anchors - still lucky-passes both the random gate and the fixed
# battery.
#
# This section adds a *minimal-criterion coevolution* (MCC) search that GROWS the
# adversarial set: a population of parametric test-case genomes that mutate to try to
# BREAK kernels which currently pass, so the correctness bar escalates automatically
# and numerical-tolerance / edge-case loopholes are auto-discovered (then closed by
# folding them back into the deterministic battery - the FMSP "find-then-patch" loop,
# applied to kernel correctness).
#
# Everything here is:
#   * PURE CPU DATA. A test-case is a small genome; ``TestCase.build`` materialises
#     float64 numpy arrays (cast to the task dtype on ``device``). The search never
#     touches a GPU/Triton/the env - the caller INJECTS how a candidate is run
#     (``run_candidate``), so the orchestrator can dispatch to the real env while this
#     module stays a pure, unit-testable search over arrays.
#   * ADDITIVE + OFF BY DEFAULT. Nothing above is modified; no existing path calls any
#     of this. A caller opts in explicitly (``coevolve_tests`` / ``fold_breaking_cases``
#     -> ``verify_equivalence(..., adversarial_inputs_fn=...)``).
#   * FAIL-SAFE. A candidate that raises is treated as a break; a case whose reference
#     is ill-defined fails the minimal criterion and is discarded (never crashes the
#     loop).
#
# HONEST SCOPE. This is a directed *search over a repertoire of regimes*, not a proof.
# It finds breaking inputs far more reliably than random sampling for defects that live
# on a thin slice of one of its families (kinks, denormals, extremes, near-ties, sparse
# spikes) and escalates into regions no fixed prior samples; it cannot find a defect
# whose regime is outside every family (same limitation as any hypothesis-space search)
# and it does not *prove* correctness. What IS certain is the fold step: once a breaking
# input is discovered it is added to the DETERMINISTIC battery, so from then on the
# oracle rejects that defect with certainty (no luck), exactly like the hand-curated
# regimes.

__all__ += [
    "TestCase",
    "list_families",
    "generate_cases",
    "mutate_case",
    "crossover_cases",
    "CaseOutcome",
    "RoundStats",
    "CoevolutionResult",
    "coevolve_tests",
    "RandomSearchResult",
    "random_search",
    "FoldResult",
    "fold_breaking_cases",
    "make_strengthened_inputs",
]


# Activation kinks the fixed battery already enumerates ...
_KNOTS = (0.0, 1.0, -1.0, 3.0, -3.0, 6.0, -6.0, 0.5, -0.5, 2.0)
# ... PLUS extra "interesting" locations the fixed battery does NOT enumerate (rational
# / irrational points where branchy or table-based kernels break). The generator's
# repertoire is deliberately BROADER than the fixed enumerated set.
_KINK_LOCATIONS = _KNOTS + (1.5, 2.5, 4.0, math.pi / 2.0, math.pi, math.e, 10.0, 100.0)
# Shape/dtype/layout metamorphic perturbations applied to a materialised operand. All
# are value-preserving up to layout (safe: the reference re-defines truth on the
# perturbed input), so folding them can never make the oracle reject a correct kernel.
_PERTURBATIONS = ("none", "reverse_rows", "reverse_cols", "roll_cols", "transpose")


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), lo), hi))


def _full(dims, v: float) -> np.ndarray:
    return np.full(dims, float(v), dtype=np.float64)


def _stable_seed(sig) -> int:
    """Process-stable seed from a genome signature (``hash()`` is salted; md5 is not)."""
    return int(hashlib.md5(repr(sig).encode("utf-8")).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# Regime families. Each is a pure-data recipe with a uniform signature so the search
# is generic: sample(rng, dtype)->genes, mutate(genes, rng, scale, dtype)->genes,
# build(dims, dtype, genes, rng)->float64 ndarray, difficulty(genes, dtype)->float.
# Initial ``sample`` genes are deliberately EASY so escalation is observable and so a
# same-budget random sampler drawing from the family prior misses thin defects that
# only compounding mutation reaches.
# --------------------------------------------------------------------------- #
def _constant_sample(rng, dtype):
    return {"value": float(rng.uniform(-2.0, 2.0))}


def _constant_mutate(p, rng, scale, dtype):
    return {"value": float(p["value"] + rng.normal(0.0, 0.5 * scale))}


def _constant_build(dims, dtype, p, rng):
    return _full(dims, p["value"])


def _constant_diff(p, dtype):
    return 0.0


def _kink_sample(rng, dtype):
    return {
        "loc": float(rng.choice(_KINK_LOCATIONS)),
        "side": float(rng.choice((-1.0, 1.0))),
        "log_eps": float(rng.uniform(-3.0, 0.0)),   # eps in [1e-3, 1]: easy, off-kink
    }


def _kink_mutate(p, rng, scale, dtype):
    q = dict(p)
    q["log_eps"] = _clip(p["log_eps"] + rng.normal(0.0, 0.7 * scale), -14.0, 0.0)
    if rng.random() < 0.15 * scale:
        q["loc"] = float(rng.choice(_KINK_LOCATIONS))
    if rng.random() < 0.10 * scale:
        q["side"] = -float(q["side"])
    return q


def _kink_build(dims, dtype, p, rng):
    return _full(dims, p["loc"] + p["side"] * (10.0 ** p["log_eps"]))


def _kink_diff(p, dtype):
    return _clip(-p["log_eps"], 0.0, 14.0)      # closer to the kink -> harder


def _subnormal_range(dtype):
    d = (dtype or "").lower()
    if "fp16" in d or "float16" in d:
        return (-4.5, -7.5)     # fp16 smallest normal ~6.1e-5; subnormal down to ~6e-8
    if "fp8" in d or "fp4" in d or "mxfp4" in d or "mxfp8" in d:
        return (-1.0, -3.0)
    return (-37.0, -44.0)       # fp32 / bf16 / fp64: subnormal boundary ~1.2e-38


def _denorm_sample(rng, dtype):
    near, _ = _subnormal_range(dtype)
    return {"log_mag": float(rng.uniform(near - 1.0, near + 1.0)),
            "sign": float(rng.choice((-1.0, 1.0)))}


def _denorm_mutate(p, rng, scale, dtype):
    near, deep = _subnormal_range(dtype)
    q = dict(p)
    q["log_mag"] = _clip(p["log_mag"] + rng.normal(0.0, 0.7 * scale), deep - 2.0, near + 2.0)
    if rng.random() < 0.1 * scale:
        q["sign"] = -float(q["sign"])
    return q


def _denorm_build(dims, dtype, p, rng):
    return _full(dims, p["sign"] * (10.0 ** p["log_mag"]))


def _denorm_diff(p, dtype):
    near, _ = _subnormal_range(dtype)
    return _clip(near - p["log_mag"], 0.0, 15.0)    # deeper into subnormals -> harder


def _extreme_anchor(dtype):
    return math.log10(dtype_extremes(dtype)[0]), math.log10(dtype_max(dtype))


def _extreme_sample(rng, dtype):
    lo, _ = _extreme_anchor(dtype)
    return {"log_mag": float(rng.uniform(lo - 1.0, lo + 1.0)),
            "sign": float(rng.choice((-1.0, 1.0)))}


def _extreme_mutate(p, rng, scale, dtype):
    lo, hi = _extreme_anchor(dtype)
    q = dict(p)
    q["log_mag"] = _clip(p["log_mag"] + rng.normal(0.0, 0.5 * scale), lo - 2.0, hi)
    if rng.random() < 0.1 * scale:
        q["sign"] = -float(q["sign"])
    return q


def _extreme_build(dims, dtype, p, rng):
    return _full(dims, p["sign"] * (10.0 ** p["log_mag"]))


def _extreme_diff(p, dtype):
    lo, hi = _extreme_anchor(dtype)
    return _clip((p["log_mag"] - lo) / max(hi - lo, 1e-9) * 15.0, 0.0, 15.0)


def _near_tie_sample(rng, dtype):
    return {"log_gap": float(rng.uniform(-4.0, -1.0)), "base": 1.0}  # gap 1e-4..0.1: easy


def _near_tie_mutate(p, rng, scale, dtype):
    q = dict(p)
    q["log_gap"] = _clip(p["log_gap"] + rng.normal(0.0, 0.8 * scale), -14.0, 0.0)
    return q


def _near_tie_build(dims, dtype, p, rng):
    dims = _dims(dims) if not isinstance(dims, tuple) else dims
    n_last = dims[-1] if len(dims) >= 1 else 1
    total = int(np.prod(dims)) if dims else 1
    n = max(int(n_last), 2)
    m = max(total // n, 1)
    base = float(p["base"])
    gap = 10.0 ** float(p["log_gap"])
    arr = np.full((m, n), base - 2.0, dtype=np.float64)
    arr[:, 0] = base                # the row max lives at column 0 ...
    arr[:, -1] = base - gap         # ... its near-tie partner at the far column
    return arr.reshape(dims) if int(np.prod(dims)) == m * n else arr


def _near_tie_diff(p, dtype):
    return _clip(-p["log_gap"], 0.0, 14.0)      # tighter tie -> harder


def _sparse_sample(rng, dtype):
    return {"log_density": float(rng.uniform(-1.5, -0.5)),
            "log_mag": float(rng.uniform(0.0, 2.0)),
            "sign": float(rng.choice((-1.0, 1.0)))}


def _sparse_mutate(p, rng, scale, dtype):
    hi = math.log10(max(dtype_extremes(dtype)[0], 10.0))
    q = dict(p)
    q["log_density"] = _clip(p["log_density"] + rng.normal(0.0, 0.5 * scale), -4.0, 0.0)
    q["log_mag"] = _clip(p["log_mag"] + rng.normal(0.0, 0.5 * scale), -2.0, hi)
    if rng.random() < 0.1 * scale:
        q["sign"] = -float(q["sign"])
    return q


def _sparse_build(dims, dtype, p, rng):
    n = int(np.prod(dims)) if dims else 1
    arr = np.zeros(n, dtype=np.float64)
    k = min(max(1, int(n * (10.0 ** p["log_density"]))), n)
    idx = rng.choice(n, size=k, replace=False)
    arr[idx] = p["sign"] * (10.0 ** p["log_mag"])
    return arr.reshape(dims)


def _sparse_diff(p, dtype):
    return _clip(-p["log_density"], 0.0, 4.0)   # sparser -> harder


@dataclass(frozen=True)
class _Family:
    name: str
    sample: Callable        # (rng, dtype) -> genes dict
    mutate: Callable        # (genes, rng, scale, dtype) -> genes dict
    build: Callable         # (dims, dtype, genes, rng) -> float64 ndarray
    difficulty: Callable    # (genes, dtype) -> float


_FAMILIES: dict[str, _Family] = {
    "constant": _Family("constant", _constant_sample, _constant_mutate,
                        _constant_build, _constant_diff),
    "kink_neighborhood": _Family("kink_neighborhood", _kink_sample, _kink_mutate,
                                 _kink_build, _kink_diff),
    "denormal_sweep": _Family("denormal_sweep", _denorm_sample, _denorm_mutate,
                              _denorm_build, _denorm_diff),
    "extreme_magnitude": _Family("extreme_magnitude", _extreme_sample, _extreme_mutate,
                                 _extreme_build, _extreme_diff),
    "near_tie": _Family("near_tie", _near_tie_sample, _near_tie_mutate,
                        _near_tie_build, _near_tie_diff),
    "sparse_spike": _Family("sparse_spike", _sparse_sample, _sparse_mutate,
                            _sparse_build, _sparse_diff),
}


def list_families() -> tuple:
    """Names of the regime families the generator can propose."""
    return tuple(_FAMILIES.keys())


def _resolve_families(families) -> list:
    if families is None:
        return list(_FAMILIES.keys())
    names = [families] if isinstance(families, str) else list(families)
    bad = [n for n in names if n not in _FAMILIES]
    if bad:
        raise KeyError(f"unknown families {bad}; known={list(_FAMILIES)}")
    return names


def _apply_perturbation(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none" or arr.ndim == 0:
        return arr
    if kind == "reverse_rows":
        return np.ascontiguousarray(arr[::-1])
    if kind == "reverse_cols":
        return np.ascontiguousarray(arr[..., ::-1])
    if kind == "roll_cols":
        return np.roll(arr, shift=max(1, arr.shape[-1] // 3), axis=-1)
    if kind == "transpose":
        return np.ascontiguousarray(arr.T) if arr.ndim >= 2 else arr
    return arr


# --------------------------------------------------------------------------- #
# The genome
# --------------------------------------------------------------------------- #
@dataclass
class TestCase:
    """One parametric adversarial test-case (a genome), materialisable as pure data.

    ``family`` selects a regime recipe; ``params`` are its evolvable genes; ``arity`` /
    ``op_index`` place the pattern into a multi-operand op (other slots get a fixed
    benign draw, like :func:`adversarial_inputs`); ``perturbation`` is an optional
    shape/layout metamorphic variant. :meth:`build` returns a tuple of operand arrays.
    """

    __test__ = False    # not a pytest test class (its name matches "Test*")

    family: str
    params: dict
    arity: int = 1
    op_index: int = 0
    perturbation: str = "none"
    benign_seed: int = 0x5EED

    def difficulty(self, dtype: str = "fp32") -> float:
        """Grounded, bounded hardness of this genome (higher = closer to a defect)."""
        fam = _FAMILIES.get(self.family)
        if fam is None:
            return 0.0
        try:
            return float(fam.difficulty(self.params, dtype))
        except Exception:      # noqa: BLE001 - difficulty must never crash the loop
            return 0.0

    def signature(self) -> tuple:
        """Hashable identity for dedup / archiving (genes rounded to 12 places)."""
        items = tuple(sorted((k, round(float(v), 12)) for k, v in self.params.items()))
        return (self.family, items, int(self.arity), int(self.op_index), self.perturbation)

    def _seed(self) -> int:
        return _stable_seed(self.signature()) ^ (int(self.benign_seed) & 0xFFFFFFFF)

    def build(self, shape=None, dtype: str = "fp32", device: str = "cpu") -> tuple:
        """Materialise the operand tuple (float64 cast to ``dtype`` on ``device``)."""
        fam = _FAMILIES.get(self.family)
        if fam is None:
            raise KeyError(f"unknown family {self.family!r}")
        dims = _dims(shape)
        rng = np.random.default_rng(self._seed())
        pat = np.asarray(fam.build(dims, dtype, self.params, rng), dtype=np.float64)
        pat = _apply_perturbation(pat, self.perturbation)
        if self.arity <= 1:
            arrs = [pat]
        else:
            benign = np.random.default_rng(self._seed() + 777).standard_normal(pat.shape)
            slot = min(int(self.op_index), self.arity - 1)
            arrs = [pat if j == slot else benign for j in range(self.arity)]
        return tuple(_cast(a, dtype, device) for a in arrs)

    def describe(self) -> str:
        genes = ", ".join(f"{k}={v:.6g}" for k, v in sorted(self.params.items()))
        extra = "" if self.arity == 1 else f" op{self.op_index}/{self.arity}"
        pert = "" if self.perturbation == "none" else f" [{self.perturbation}]"
        return f"{self.family}({genes}){extra}{pert}"


# --------------------------------------------------------------------------- #
# Generator + evolutionary operators (pure data)
# --------------------------------------------------------------------------- #
def generate_cases(n: int, rng, *, families=None, dtype: str = "fp32", arity: int = 1,
                   op_indices=None, perturbations=("none",)) -> list:
    """Propose ``n`` random adversarial genomes across ``families`` (pure data)."""
    names = _resolve_families(families)
    ops = list(op_indices) if op_indices is not None else list(range(max(1, int(arity))))
    perts = list(perturbations) if perturbations else ["none"]
    out = []
    for _ in range(int(n)):
        fam = names[int(rng.integers(len(names)))]
        params = _FAMILIES[fam].sample(rng, dtype)
        op = ops[int(rng.integers(len(ops)))] if arity > 1 else 0
        pert = perts[int(rng.integers(len(perts)))]
        out.append(TestCase(fam, params, arity=int(arity), op_index=int(op),
                            perturbation=str(pert), benign_seed=int(rng.integers(1, 2**31))))
    return out


def mutate_case(case: TestCase, rng, scale: float = 1.0, *, dtype: str = "fp32",
                families=None, perturbations=("none",)) -> TestCase:
    """Return a mutated copy of ``case`` (perturb genes; rarely jump family/layout)."""
    fam = _FAMILIES[case.family]
    params = fam.mutate(dict(case.params), rng, scale, dtype)
    new_family = case.family
    if families and rng.random() < 0.05 * scale:      # rare family jump keeps diversity
        names = _resolve_families(families)
        jumped = names[int(rng.integers(len(names)))]
        if jumped != case.family:
            new_family, params = jumped, _FAMILIES[jumped].sample(rng, dtype)
    pert = case.perturbation
    if perturbations and rng.random() < 0.10 * scale:
        perts = list(perturbations)
        pert = perts[int(rng.integers(len(perts)))]
    op = case.op_index
    if case.arity > 1 and rng.random() < 0.10 * scale:
        op = int(rng.integers(case.arity))
    return replace(case, family=new_family, params=params, perturbation=str(pert),
                   op_index=int(op))


def crossover_cases(a: TestCase, b: TestCase, rng, *, dtype: str = "fp32") -> TestCase:
    """Recombine two genomes (uniform gene crossover within a family)."""
    if a.family != b.family:
        return replace(a if a.difficulty(dtype) >= b.difficulty(dtype) else b)
    params = {k: (a.params if rng.random() < 0.5 else b.params).get(k, v)
              for k, v in a.params.items()}
    op = a.op_index if rng.random() < 0.5 else b.op_index
    pert = a.perturbation if rng.random() < 0.5 else b.perturbation
    return replace(a, params=params, op_index=int(op), perturbation=str(pert))


# --------------------------------------------------------------------------- #
# Evaluation (injectable, pure CPU)
# --------------------------------------------------------------------------- #
def _default_run(fn: Callable, inputs: tuple):
    """Call ``fn(*inputs)``; a raised exception is a disagreement (returned as-is)."""
    try:
        return fn(*inputs)
    except Exception as exc:      # noqa: BLE001 - a crashing candidate is a break
        return exc


def _default_minimal_criterion(ref_out, inputs) -> bool:
    """A case is admissible only if the reference defines a non-empty truth on it."""
    if isinstance(ref_out, Exception):
        return False
    try:
        return int(np.asarray(_to_f64(ref_out)).size) > 0
    except Exception:      # noqa: BLE001
        return False


@dataclass
class CaseOutcome:
    """Per-case evaluation record for one round."""

    case: TestCase
    valid: bool
    broke: bool
    n_broken: int
    severities: list        # per-candidate worst per-element rel-err (inf on crash)
    fitness: float
    difficulty: float
    detail: str = ""


def _archive_severity(oc: CaseOutcome) -> float:
    vals = [s for s in oc.severities if isinstance(s, float)]
    if any(not math.isfinite(s) for s in vals):
        return math.inf
    return max(vals) if vals else 0.0


def _evaluate_case(case, reference_fn, candidate_fns, tol, *, shape, dtype, device,
                   run_candidate, run_reference, compare, minimal_criterion,
                   broken_ids, difficulty_weight) -> CaseOutcome:
    try:
        inputs = case.build(shape, dtype, device=device)
    except Exception as exc:      # noqa: BLE001 - an unbuildable genome is inadmissible
        return CaseOutcome(case, False, False, 0, [], -math.inf, 0.0, f"build-error:{exc}")

    ref_out = run_reference(reference_fn, inputs)
    diff = case.difficulty(dtype)
    if not minimal_criterion(ref_out, inputs):
        return CaseOutcome(case, False, False, 0, [], -math.inf, diff, "minimal-criterion")

    severities: list = []
    n_broken = 0
    newly = 0
    for ci, cand in enumerate(candidate_fns):
        out = run_candidate(cand, inputs)
        if isinstance(out, Exception):
            severities.append(math.inf)
            n_broken += 1
            newly += (ci not in broken_ids)
            continue
        cmp = compare(out, ref_out, tol)
        severities.append(float(cmp.worst_rel_err))
        if not cmp.ok:
            n_broken += 1
            newly += (ci not in broken_ids)

    broke = n_broken > 0
    if broke:
        # Breaking cases dominate; among them prefer more breaks, freshly-broken (still
        # -passing) candidates, then the harder (tighter) genome.
        fitness = 1.0e6 * n_broken + 1.0e5 * newly + diff
    else:
        # No break yet: the DRIVE is difficulty (open-ended escalation into harder
        # regimes) plus a sub-unit proximity term (how close to tolerance we got), which
        # guides the search when the defect boundary is graded rather than sharp.
        rtol = tol.rtol if tol.rtol > 0 else 1e-9
        finite = [s for s in severities if math.isfinite(s)]
        prox = min(max(finite) / rtol, 0.999) if finite else 0.0
        fitness = difficulty_weight * diff + prox
    return CaseOutcome(case, True, broke, n_broken, severities, float(fitness), diff)


# --------------------------------------------------------------------------- #
# Coevolution loop
# --------------------------------------------------------------------------- #
@dataclass
class RoundStats:
    """Per-round telemetry (escalation is visible in the difficulty trend)."""

    round: int
    n_valid: int
    n_breaking: int
    best_fitness: float
    mean_elite_difficulty: float
    max_difficulty: float
    cumulative_breaks: int          # distinct breaking genomes archived so far
    n_candidates_broken: int        # distinct candidates broken so far


@dataclass
class CoevolutionResult:
    """Outcome of :func:`coevolve_tests`."""

    breaking_cases: list            # archived breaking genomes, most-severe/hardest first
    rounds: list                    # list[RoundStats]
    final_population: list          # list[TestCase]
    broke_any: bool
    n_candidates_broken: int
    dtype: str
    shape: Any
    tol: Tolerance
    n_evaluations: int

    def best_case(self) -> Optional[TestCase]:
        return self.breaking_cases[0] if self.breaking_cases else None

    def families_found(self) -> list:
        return sorted({c.family for c in self.breaking_cases})

    def difficulty_trend(self) -> list:
        return [r.mean_elite_difficulty for r in self.rounds]

    def summary(self) -> str:
        head = (f"[coevolve] evals={self.n_evaluations} broke_any={self.broke_any} "
                f"breaking={len(self.breaking_cases)} "
                f"candidates_broken={self.n_candidates_broken} "
                f"families={self.families_found()}")
        if self.rounds:
            d0 = self.rounds[0].mean_elite_difficulty
            d1 = self.rounds[-1].mean_elite_difficulty
            head += f" elite_difficulty {d0:.2f}->{d1:.2f}"
        best = self.best_case()
        if best is not None:
            head += f"\n  hardest breaking case: {best.describe()} " \
                    f"(difficulty={best.difficulty(self.dtype):.2f})"
        return head


def coevolve_tests(reference_fn: Callable, candidate_fns, *, shape=None,
                   dtype: str = "fp32", arity: int = 1, seed: int = 0, rounds: int = 25,
                   population_size: int = 48, elite_frac: float = 0.25, families=None,
                   tol: Optional[Tolerance] = None, device: str = "cpu",
                   run_candidate: Optional[Callable] = None,
                   run_reference: Optional[Callable] = None,
                   compare: Optional[Callable] = None,
                   minimal_criterion: Optional[Callable] = None,
                   difficulty_weight: float = 1.0, mutation_scale: float = 1.0,
                   op_indices=None, perturbations=("none",), archive_size: int = 64,
                   fresh_per_round: Optional[int] = None) -> CoevolutionResult:
    """Coevolve a population of adversarial test-cases to BREAK ``candidate_fns``.

    Minimal-criterion coevolution: each round every genome is materialised (pure CPU),
    the reference oracle defines truth, and each candidate is run VIA THE INJECTED
    ``run_candidate`` (default: a direct in-process call; the orchestrator supplies a
    GPU/env runner). A genome is rewarded for exposing a disagreement (candidate vs
    reference, using ``compare`` - default :func:`compare_pair` at ``tol``); breaking
    genomes dominate selection, with a bonus for breaking still-passing candidates so
    difficulty escalates across the candidate set. Absent a break, selection is driven
    by (bounded, grounded) difficulty + proximity-to-tolerance, so the population
    escalates into ever-harder regimes each round - reaching thin slices no fixed prior
    samples. Deterministic given ``seed``. Never imports torch / touches a GPU itself.

    Parameters
    ----------
    reference_fn : ``(*inputs) -> array``   the fp64/fp32 oracle (defines truth).
    candidate_fns : callable | list          kernel(s) to try to break.
    run_candidate / run_reference : ``(fn, inputs) -> out``  INJECTION POINT. Default
        calls in-process and treats a raised exception as a break. Pass a custom runner
        to dispatch to the real (GPU) env - this module then stays pure CPU.
    compare : ``(actual, expected, tol) -> PairComparison``  break predicate.
    families : which regime families to draw from (default: all of :func:`list_families`).
    difficulty_weight : strength of the open-ended escalation drive (0 disables it).

    Returns a :class:`CoevolutionResult` whose ``breaking_cases`` can be fed to
    :func:`fold_breaking_cases`.
    """
    if callable(candidate_fns):
        candidate_fns = [candidate_fns]
    candidate_fns = list(candidate_fns)
    if not candidate_fns:
        raise ValueError("coevolve_tests needs at least one candidate_fn")

    tol = tol or tolerance_for(dtype)
    run_candidate = run_candidate or _default_run
    run_reference = run_reference or _default_run
    compare = compare or compare_pair
    minimal_criterion = minimal_criterion or _default_minimal_criterion
    fams = _resolve_families(families)

    rng = np.random.default_rng(seed)
    pop = generate_cases(population_size, rng, families=fams, dtype=dtype, arity=arity,
                         op_indices=op_indices, perturbations=perturbations)
    k = max(2, int(round(population_size * elite_frac)))
    n_fresh = fresh_per_round if fresh_per_round is not None else max(1, population_size // 10)

    archive: dict = {}          # signature -> (severity, TestCase)
    broken_ids: set = set()
    stats: list = []
    n_eval = 0

    for r in range(int(rounds)):
        outcomes = []
        for case in pop:
            oc = _evaluate_case(
                case, reference_fn, candidate_fns, tol, shape=shape, dtype=dtype,
                device=device, run_candidate=run_candidate, run_reference=run_reference,
                compare=compare, minimal_criterion=minimal_criterion,
                broken_ids=broken_ids, difficulty_weight=difficulty_weight)
            outcomes.append(oc)
            n_eval += 1

        round_breaks = 0
        for oc in outcomes:
            if not oc.broke:
                continue
            round_breaks += 1
            sig = oc.case.signature()
            sev = _archive_severity(oc)
            prev = archive.get(sig)
            if prev is None or sev > prev[0]:
                archive[sig] = (sev, oc.case)
            for ci, s in enumerate(oc.severities):
                if (not math.isfinite(s)) or s > tol.rtol:
                    broken_ids.add(ci)

        valid_sorted = sorted((o for o in outcomes if o.valid),
                              key=lambda o: o.fitness, reverse=True)
        elite = valid_sorted[:k]
        elite_diff = [o.difficulty for o in elite]
        stats.append(RoundStats(
            round=r, n_valid=len(valid_sorted), n_breaking=round_breaks,
            best_fitness=(valid_sorted[0].fitness if valid_sorted else -math.inf),
            mean_elite_difficulty=(float(np.mean(elite_diff)) if elite_diff else 0.0),
            max_difficulty=max((o.difficulty for o in valid_sorted), default=0.0),
            cumulative_breaks=len(archive), n_candidates_broken=len(broken_ids)))

        if not elite:      # whole population inadmissible -> resample fresh
            pop = generate_cases(population_size, rng, families=fams, dtype=dtype,
                                 arity=arity, op_indices=op_indices,
                                 perturbations=perturbations)
            continue

        next_pop = [o.case for o in elite]     # elitism
        next_pop += generate_cases(n_fresh, rng, families=fams, dtype=dtype, arity=arity,
                                   op_indices=op_indices, perturbations=perturbations)
        while len(next_pop) < population_size:
            pa = elite[int(rng.integers(len(elite)))].case
            pb = elite[int(rng.integers(len(elite)))].case
            child = crossover_cases(pa, pb, rng, dtype=dtype)
            child = mutate_case(child, rng, scale=mutation_scale, dtype=dtype,
                                families=fams, perturbations=perturbations)
            next_pop.append(child)
        pop = next_pop[:population_size]

    ranked = sorted(archive.values(),
                    key=lambda t: (t[0], t[1].difficulty(dtype)), reverse=True)
    breaking = [c for _, c in ranked][:archive_size]
    return CoevolutionResult(
        breaking_cases=breaking, rounds=stats, final_population=pop,
        broke_any=bool(breaking), n_candidates_broken=len(broken_ids),
        dtype=dtype, shape=shape, tol=tol, n_evaluations=n_eval)


# --------------------------------------------------------------------------- #
# Random-search baseline (the honest control: what plain sampling finds)
# --------------------------------------------------------------------------- #
@dataclass
class RandomSearchResult:
    breaking_cases: list        # TestCase (family mode); empty in "natural" mode
    n_samples: int
    n_breaking: int
    broke_any: bool
    mode: str


def random_search(reference_fn: Callable, candidate_fns, *, shape=None,
                  dtype: str = "fp32", seed: int = 0, n_samples: int = 1000,
                  mode: str = "family", families=None, arity: int = 1,
                  tol: Optional[Tolerance] = None, device: str = "cpu",
                  run_candidate: Optional[Callable] = None,
                  run_reference: Optional[Callable] = None,
                  compare: Optional[Callable] = None,
                  minimal_criterion: Optional[Callable] = None,
                  perturbations=("none",), op_indices=None) -> RandomSearchResult:
    """Undirected baseline: draw ``n_samples`` cases, keep the ones that break.

    ``mode="natural"`` samples ``randn`` inputs (exactly what the shipped SNR gate
    does); ``mode="family"`` draws random genomes from the family priors WITHOUT any
    selection/mutation. Provided so callers/tests can quantify the coevolution's
    advantage honestly (same budget, no directed escalation).
    """
    if callable(candidate_fns):
        candidate_fns = [candidate_fns]
    candidate_fns = list(candidate_fns)
    tol = tol or tolerance_for(dtype)
    run_candidate = run_candidate or _default_run
    run_reference = run_reference or _default_run
    compare = compare or compare_pair
    minimal_criterion = minimal_criterion or _default_minimal_criterion
    rng = np.random.default_rng(seed)

    def _breaks(inputs) -> bool:
        ref_out = run_reference(reference_fn, inputs)
        if not minimal_criterion(ref_out, inputs):
            return False
        for cand in candidate_fns:
            out = run_candidate(cand, inputs)
            if isinstance(out, Exception) or not compare(out, ref_out, tol).ok:
                return True
        return False

    breaking: list = []
    n_breaking = 0
    if mode == "natural":
        dims = _dims(shape)
        for _ in range(int(n_samples)):
            inputs = tuple(_cast(rng.standard_normal(dims), dtype, device)
                           for _ in range(max(1, int(arity))))
            if _breaks(inputs):
                n_breaking += 1
        return RandomSearchResult([], int(n_samples), n_breaking, n_breaking > 0, mode)

    for case in generate_cases(n_samples, rng, families=families, dtype=dtype,
                               arity=arity, op_indices=op_indices,
                               perturbations=perturbations):
        try:
            inputs = case.build(shape, dtype, device=device)
        except Exception:      # noqa: BLE001
            continue
        if _breaks(inputs):
            n_breaking += 1
            breaking.append(case)
    return RandomSearchResult(breaking, int(n_samples), n_breaking, n_breaking > 0, mode)


# --------------------------------------------------------------------------- #
# Folding discovered breaks back into the DETERMINISTIC battery (the patch)
# --------------------------------------------------------------------------- #
@dataclass
class FoldResult:
    """Discovered breaks folded into a strengthened, deterministic regime set.

    ``adversarial_inputs_fn()`` returns a drop-in replacement for
    :func:`adversarial_inputs` (same signature) that yields the FIXED battery followed
    by the folded breaking cases, so
    ``verify_equivalence(..., adversarial_inputs_fn=fold.adversarial_inputs_fn())``
    now rejects the defect with certainty. ``tolerance`` is the base tolerance
    (optionally tightened - a heuristic, see :func:`fold_breaking_cases`).
    """

    cases: list
    tolerance: Tolerance
    n_folded: int
    base_included: bool = True

    def adversarial_inputs_fn(self) -> Callable:
        return make_strengthened_inputs(self.cases, include_base=self.base_included)

    def regime_names(self) -> list:
        return [f"folded::{c.family}" for c in self.cases]


def make_strengthened_inputs(folded_cases, include_base: bool = True) -> Callable:
    """Build an :func:`adversarial_inputs`-compatible generator = fixed battery (+ opt)
    then the folded breaking genomes, rebuilt at the caller's ``shape``/``dtype``."""
    folded = list(folded_cases)

    def _gen(shape, dtype: str, arity: int = 1, op_class: str = "elementwise",
             device: str = "cpu", seed: int = 0):
        if include_base:
            yield from adversarial_inputs(shape, dtype, arity=arity, op_class=op_class,
                                          device=device, seed=seed)
        for i, case in enumerate(folded):
            c = case
            if c.arity != arity or c.op_index >= max(1, arity):
                c = replace(c, arity=int(arity), op_index=min(c.op_index, max(0, arity - 1)))
            try:
                inputs = c.build(shape, dtype, device=device)
            except Exception:      # noqa: BLE001 - a folded case must never break the run
                continue
            yield (f"folded::{c.family}[{i}]", inputs)

    return _gen


def fold_breaking_cases(cases, *, base_tol: Optional[Tolerance] = None,
                        dtype: str = "fp32", tighten_tolerance: bool = False,
                        max_cases: int = 32, include_base: bool = True) -> FoldResult:
    """Fold discovered breaking genomes into a strengthened deterministic regime set.

    Dedups by genome signature, keeps the hardest ``max_cases``. The SOUND
    strengthening is regime-based: the exact breaking inputs are added to the
    deterministic battery, so the oracle then rejects that defect with certainty.
    ``tighten_tolerance`` additionally halves ``rtol`` and raises the SNR floor - this
    is a HEURISTIC (it can cause false rejects of legitimately-noisy kernels) and is
    OFF by default; prefer regime strengthening.
    """
    seen: dict = {}
    for c in cases:
        sig = c.signature()
        if sig not in seen or c.difficulty(dtype) > seen[sig].difficulty(dtype):
            seen[sig] = c
    folded = sorted(seen.values(), key=lambda c: c.difficulty(dtype),
                    reverse=True)[:max_cases]
    tol = base_tol or tolerance_for(dtype)
    if tighten_tolerance:
        tol = replace(tol, rtol=tol.rtol * 0.5, snr_db_min=tol.snr_db_min + 6.0,
                      metamorphic_rtol=tol.metamorphic_rtol * 0.5)
    return FoldResult(cases=folded, tolerance=tol, n_folded=len(folded),
                      base_included=include_base)
