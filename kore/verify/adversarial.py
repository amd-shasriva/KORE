"""Adversarial / structured input battery for the equivalence oracle.

Random ``randn`` sampling — what the shipped SNR gate uses — under-samples exactly
the regimes where numerical kernels break: exact zeros, sign boundaries, overflow
knots, denormals, all-equal rows, sparse spikes. A *lucky-pass* kernel is one that is
wrong only on such a thin slice and never gets caught by random draws. This module
enumerates those regimes DETERMINISTICALLY, so a kernel that is wrong on any of them
is rejected with certainty (not luck) — the provable half of the oracle for the
checkable op class.

Pure numpy generation with a lazy torch cast (so importing this module never needs a
GPU). Values are produced in float64 and cast to the task ``dtype`` on the requested
``device`` (numpy on CPU, torch on cuda / when torch tensors are requested).
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

__all__ = ["adversarial_patterns", "adversarial_inputs", "dtype_extremes", "dtype_max"]


def dtype_extremes(dtype: str) -> tuple[float, float, float]:
    """Return ``(big, small, tiny)`` magnitudes safe for ``dtype``.

    ``big`` is a large finite value chosen so that even an amplifying op (``x*x``, a
    128-wide row sum) stays finite in ``dtype`` — so it stresses magnitude without
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
    tensor (mostly zero), and explicit activation-knot values (``0, ±1, ±3, ±6``).
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

    # explicit activation knots tiled across the tensor (0, ±1, ±3, ±6, ±0.5).
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
