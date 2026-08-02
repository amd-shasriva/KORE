"""Operator-generation engine: scale the KORE task suite from a hand-authored core
to 200+ generated operator tasks (this engine emits the 201 ``gen_*`` tasks).

A KORE task = task.yaml + reference.py (torch oracle + inputs) + seed_triton.py
(a compiling starter kernel) + driver.py (the verifier contract). Hand-writing
hundreds of those is infeasible *and* error-prone, so this engine generates them
from a declarative op spec:

  * ``make_reference(op, family, dtype)`` -> the reference.py namespace (parse_shape,
    get_inputs, ref_fn oracle, baseline_fn production path, arity, entry_name).
  * ``seed_source(op, family, dtype)`` -> a REAL, compiling Triton seed kernel
    (the policy's starting point to optimize; the pointwise/reduce math is inlined,
    not a shim, so the policy has genuine code to edit).
  * ``driver_main(ref, task_dir)`` -> the generic KernelForge driver: multi-trial
    correctness + cold-cache timing + the POST-TIMING anti-hack re-verification
    (candidate module cached so a stateful invocation-count kernel is caught), in
    ONE place (rather than duplicating the driver into every generated task dir).

Generated ops use the torch FRAMEWORK op as the production baseline (the honest
ROCm serving path for elementwise/reduction ops, exactly like the shipped
gelu_tanh/softmax tasks). Every generated op is verifiable by construction: the
Triton seed computes the same math (fp32) as the torch oracle.

For the multi-op ``fusion`` / ``gemm_fusion`` families the honest bar is the
COMPILER-FUSED kernel, and that is the DEFAULT (see ``compile_baseline_status``):
a candidate that fuses an elementwise chain must beat ``torch.compile``, not the
unfused eager chain, or the "speedup" is just the absence of a compiler. Dropping
back to the eager bar requires an explicit falsey ``KORE_COMPILE_BASELINE`` and is
recorded (``baseline_kind``, ``baseline_compile_opt_out``, and a process warning).

All pure/CPU-importable (torch/triton imported lazily inside the GPU paths) so
registry discovery never needs a GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# dtypes
# --------------------------------------------------------------------------- #
# name -> (torch dtype attr, triton dtype literal, snr gate dB)
DTYPES: dict[str, tuple[str, str, float]] = {
    "bf16": ("bfloat16", "tl.bfloat16", 30.0),
    "fp16": ("float16", "tl.float16", 30.0),
    "fp32": ("float32", "tl.float32", 40.0),
    # fp8 e4m3 - OCP e4m3fn on gfx950/CDNA4 (MI350X/MI355X, the native CDNA4
    # format; gfx942/CDNA3 used FNUZ). Used by quantized GEMM vendor ops. The
    # oracle dequantizes the SAME fp8 operands, so the gate measures the kernel's
    # fp32 accumulation fidelity (bf16 output) - a ~25 dB bar, not the quant error.
    "fp8": ("float8_e4m3fn", "tl.float8e4nv", 25.0),
    # int8 symmetric (W8A8): int8-in / bf16-out quantized GEMM (per-row/col scales).
    "int8": ("int8", "tl.int8", 25.0),
}


def _torch_dtype(name: str):
    import torch
    return getattr(torch, DTYPES[name][0])


def tl_round_half_even(v: str) -> str:
    """Triton EXPRESSION rounding ``v`` half-to-EVEN -- exactly ``torch.round``.

    Every integer quantizer in this corpus has a ``round(x / scale)`` oracle, and
    ``torch.round`` breaks ties to even.  Rounding half AWAY FROM ZERO instead
    (``floor(v + 0.5)`` for positive ``v``) is off by one code on every exact
    tie, and ties are not a measure-zero set here: the operand is a bf16/fp16
    value divided by a scale derived from that same tensor, so ~0.3% of a
    quantized tensor lands exactly on ``.5`` and ~0.11% of codes come out
    different.  That is a systematic, reproducible disagreement with the oracle,
    not rounding noise, and no tolerance should absorb it.

    Written with ``floor`` rather than ``libdevice.rint`` so a generated seed
    needs no import beyond ``triton.language`` and does not depend on where a
    given Triton release puts its device-function shims.  Sub-expressions repeat
    but are common-subexpression-eliminated.
    """
    down = f"tl.floor({v} + 0.5)"
    return (f"tl.where({down} - ({v}) == 0.5, "
            f"2.0 * tl.floor({down} * 0.5), {down})")


# torch.compile'd baseline cache (one per fusion/gemm_fusion op+dtype).
_FUSED_BASELINE_CACHE: dict = {}

COMPILE_BASELINE_ENV = "KORE_COMPILE_BASELINE"
_TRUTHY = ("1", "true", "yes", "on")
_FALSEY = ("0", "false", "no", "off")
_OPT_OUT_ANNOUNCED: set = set()


def compile_baseline_status() -> dict:
    """Resolve the fusion/gemm_fusion baseline bar, and say WHERE it came from.

    The compiler-fused kernel is the bar a practitioner already has for free, so
    it is the DEFAULT: grading a fused-kernel candidate against unfused eager
    torch measures the absence of ``torch.compile``, not the candidate (this
    module's own docstring history called that speedup inflation).

    Opting back down to the eager bar is possible but must be DELIBERATE: only an
    explicitly falsey ``KORE_COMPILE_BASELINE`` disables it.  Unset, empty, and
    unrecognized values all fail CLOSED onto the honest bar rather than silently
    reverting to the inflated one, and the opt-out is reported in the returned
    status (which lands in the reference namespace as ``baseline_kind`` /
    ``baseline_compile_opt_out``) and warned about once per process.
    """
    raw = os.environ.get(COMPILE_BASELINE_ENV)
    text = (raw or "").strip().lower()
    if text in _FALSEY:
        return {"enabled": False, "declared": raw, "source": "env_opt_out"}
    if text in _TRUTHY:
        return {"enabled": True, "declared": raw, "source": "env"}
    if text:
        return {"enabled": True, "declared": raw, "source": "unrecognized_value"}
    return {"enabled": True, "declared": raw, "source": "default"}


def _compile_baseline_enabled() -> bool:
    """Whether fusion/gemm_fusion grade against the COMPILER-FUSED baseline.

    Default ON (see :func:`compile_baseline_status`).  A deliberate opt-out is
    announced once per process so an eager-bar run is never silent.
    """
    status = compile_baseline_status()
    if not status["enabled"] and "opt_out" not in _OPT_OUT_ANNOUNCED:
        _OPT_OUT_ANNOUNCED.add("opt_out")
        import warnings
        warnings.warn(
            f"{COMPILE_BASELINE_ENV}={status['declared']!r} disables the "
            "compiler-fused baseline: fusion / gemm_fusion tasks will be timed "
            "against UNFUSED eager torch, which inflates every measured speedup. "
            "The resolver reports 'eager', which records as "
            "baseline_kind='torch' (schemas.BASELINE_KIND_TORCH) -- 'eager' "
            "never appears in a stored record.",
            RuntimeWarning, stacklevel=3)
    return status["enabled"]


def _fused_baseline(fn, key: str):
    """Return ``torch.compile(fn)`` (fused, cached) when enabled, else ``fn``.

    Compilation is the honest multi-kernel-fusion bar (the compiler fuses the
    elementwise chain / GEMM epilogue), so the candidate must beat the FUSED kernel
    rather than unfused eager. Any compile failure degrades to eager (never fatal).

    ``fn`` must be a PURE tensor function: dynamo traces whatever it is handed, so
    wrapping a callable that itself re-reads the environment makes tracing fail.
    """
    if not _compile_baseline_enabled():
        return fn
    if key not in _FUSED_BASELINE_CACHE:
        try:
            import torch
            _FUSED_BASELINE_CACHE[key] = torch.compile(fn)
        except Exception:  # noqa: BLE001 - torch.compile unavailable/unsupported
            _FUSED_BASELINE_CACHE[key] = fn
    return _FUSED_BASELINE_CACHE[key]


# --------------------------------------------------------------------------- #
# Operator specs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UnarySpec:
    tl_expr: str                      # fp32 Triton expr in terms of `x`
    torch_fn: Callable                # torch fn (elementwise) for oracle/baseline
    domain: str = "signed"            # "signed" (randn) | "pos" (|randn|+0.1)


@dataclass(frozen=True)
class BinarySpec:
    tl_expr: str                      # fp32 Triton expr in terms of `x`, `y`
    torch_fn: Callable                # torch fn (a, b) -> tensor
    domain_b: str = "signed"          # domain of the 2nd operand


@dataclass(frozen=True)
class ReduceSpec:
    # per-row reduction [M,N] -> [M]; combine over a fp32 accumulator block.
    init: str                         # initial accumulator value (fp32 literal)
    other: str                        # masked-load fill (identity for the combine)
    combine: str                      # combine(acc_block, x_block) fp32 expr
    final: str                        # block -> scalar reduction (tl.sum/tl.max)
    post: str                         # scalar post-op in terms of `v` and `N`
    torch_fn: Callable                # torch fn (x) -> [M] oracle/baseline


@dataclass(frozen=True)
class GemmFusionSpec:
    """A GEMM with a FUSED epilogue (bias add + activation). This is the COMPUTE-
    BOUND high-value class: torch runs it as SEPARATE kernels (matmul -> hipBLASLt,
    then + bias, then activation), each an extra HBM round-trip of the [M,N] output;
    a fused Triton kernel keeps the tile in registers between matmul and epilogue.
    Baseline = the torch multi-kernel chain (matmul dispatches to the hipBLASLt
    vendor GEMM), so beating it is a genuine fusion win against a production baseline."""
    has_bias: bool
    act: str                          # "none" | "relu" | "gelu" | "silu"


@dataclass(frozen=True)
class FusionSpec:
    """A pointwise FUSION of 2-3 ops. The Triton seed computes the whole chain in
    ONE pass (one HBM round-trip); the torch baseline runs it as SEPARATE eager
    ops (multiple kernels / round-trips), so there is GENUINE speedup headroom vs
    torch-eager - unlike a single elementwise op where torch is already near
    roofline. This is the honest high-headroom operator class (KernelBench-L2 style)."""
    tl_expr: str                      # fp32 Triton expr in terms of `a`, `b`(, `c`)
    torch_fn: Callable                # torch composition (multi-kernel) baseline+oracle
    arity: int = 2                    # 2 or 3 inputs


def _lazy():
    import torch
    import torch.nn.functional as F
    return torch, F


# ---- unary elementwise (fp32 math; store in task dtype) --------------------
def _unary_specs() -> dict[str, UnarySpec]:
    import torch
    import torch.nn.functional as F
    return {
        "relu":        UnarySpec("tl.maximum(x, 0.0)", torch.relu),
        "relu6":       UnarySpec("tl.minimum(tl.maximum(x, 0.0), 6.0)", F.relu6),
        "leaky_relu":  UnarySpec("tl.where(x > 0.0, x, 0.01 * x)", lambda x: F.leaky_relu(x, 0.01)),
        "silu":        UnarySpec("x * tl.sigmoid(x)", F.silu),
        "sigmoid":     UnarySpec("tl.sigmoid(x)", torch.sigmoid),
        "hardsigmoid": UnarySpec("tl.minimum(tl.maximum(x / 6.0 + 0.5, 0.0), 1.0)", F.hardsigmoid),
        "tanh":        UnarySpec("2.0 * tl.sigmoid(2.0 * x) - 1.0", torch.tanh),
        "hardtanh":    UnarySpec("tl.minimum(tl.maximum(x, -1.0), 1.0)", F.hardtanh),
        "hardswish":   UnarySpec("x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0", F.hardswish),
        "softsign":    UnarySpec("x / (1.0 + tl.abs(x))", F.softsign),
        "elu":         UnarySpec("tl.where(x > 0.0, x, tl.exp(x) - 1.0)", F.elu),
        "softplus":    UnarySpec("tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))", F.softplus),
        "mish":        UnarySpec(
            "x * (2.0 * tl.sigmoid(2.0 * tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))) - 1.0)",
            F.mish),
        "gelu_tanh":   UnarySpec(
            "0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * (0.7978845608028654 * "
            "(x + 0.044715 * x * x * x))) - 1.0))",
            lambda x: F.gelu(x, approximate="tanh")),
        "gelu_quick":  UnarySpec("x * tl.sigmoid(1.702 * x)", lambda x: x * torch.sigmoid(1.702 * x)),
        "exp":         UnarySpec("tl.exp(x)", torch.exp),
        "abs":         UnarySpec("tl.abs(x)", torch.abs),
        "neg":         UnarySpec("-x", torch.neg),
        "square":      UnarySpec("x * x", torch.square),
        "sign":        UnarySpec("tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))", torch.sign),
        # positive-domain ops (inputs are |randn|+0.1 so they are well-defined)
        "sqrt":        UnarySpec("tl.sqrt(x)", torch.sqrt, domain="pos"),
        "rsqrt":       UnarySpec("1.0 / tl.sqrt(x)", torch.rsqrt, domain="pos"),
        "reciprocal":  UnarySpec("1.0 / x", torch.reciprocal, domain="pos"),
        "log":         UnarySpec("tl.log(x)", torch.log, domain="pos"),
    }


def _binary_specs() -> dict[str, BinarySpec]:
    import torch
    return {
        "add":      BinarySpec("x + y", torch.add),
        "mul":      BinarySpec("x * y", torch.mul),
        "sub":      BinarySpec("x - y", torch.sub),
        "maximum":  BinarySpec("tl.maximum(x, y)", torch.maximum),
        "minimum":  BinarySpec("tl.minimum(x, y)", torch.minimum),
        "add_relu": BinarySpec("tl.maximum(x + y, 0.0)", lambda a, b: torch.relu(a + b)),
        "mul_sig":  BinarySpec("x * tl.sigmoid(y)", lambda a, b: a * torch.sigmoid(b)),
        "div":      BinarySpec("x / y", torch.div, domain_b="pos"),
    }


def _reduce_specs() -> dict[str, ReduceSpec]:
    import torch
    return {
        "row_sum":  ReduceSpec("0.0", "0.0", "acc + x", "tl.sum(acc, axis=0)", "v",
                               lambda x: x.sum(-1)),
        "row_mean": ReduceSpec("0.0", "0.0", "acc + x", "tl.sum(acc, axis=0)", "v / N",
                               lambda x: x.mean(-1)),
        "row_max":  ReduceSpec("-3.0e38", "-3.0e38", "tl.maximum(acc, x)",
                               "tl.max(acc, axis=0)", "v", lambda x: x.amax(-1)),
        "row_l2":   ReduceSpec("0.0", "0.0", "acc + x * x", "tl.sum(acc, axis=0)",
                               "tl.sqrt(v)", lambda x: x.norm(p=2, dim=-1)),
        # --- v2 additive reductions (single-pass, fp32 accumulate; identity-safe
        #     masked fills so no engine change is needed) ---
        "row_rms":     ReduceSpec("0.0", "0.0", "acc + x * x", "tl.sum(acc, axis=0)",
                                  "tl.sqrt(v / N)", lambda x: x.pow(2).mean(-1).sqrt()),
        "row_l1":      ReduceSpec("0.0", "0.0", "acc + tl.abs(x)", "tl.sum(acc, axis=0)",
                                  "v", lambda x: x.abs().sum(-1)),
        "row_max_abs": ReduceSpec("0.0", "0.0", "tl.maximum(acc, tl.abs(x))",
                                  "tl.max(acc, axis=0)", "v", lambda x: x.abs().amax(-1)),
        "row_min":     ReduceSpec("3.0e38", "3.0e38", "tl.minimum(acc, x)",
                                  "tl.min(acc, axis=0)", "v", lambda x: x.amin(-1)),
    }


def _fusion_specs() -> dict[str, FusionSpec]:
    """Pointwise fusions with REAL headroom vs torch-eager multi-kernel.

    torch runs each op as a separate kernel (a+b -> kernel1, silu -> kernel2), so a
    single fused Triton kernel saves the intermediate HBM round-trips. These are the
    honest, high-headroom operator tasks (the baseline is torch-eager BY DESIGN, and
    beating it is a genuine fusion win, not a copy-loop race)."""
    import torch
    import torch.nn.functional as F

    def _silu(t): return F.silu(t)
    def _gelu(t): return F.gelu(t, approximate="tanh")

    return {
        # 2-input fusions (a, b both [M,N])
        "add_gelu":     FusionSpec(
            "0.5 * (a + b) * (1.0 + (2.0 * tl.sigmoid(2.0 * (0.7978845608028654 * "
            "((a + b) + 0.044715 * (a + b) * (a + b) * (a + b)))) - 1.0))",
            lambda a, b: _gelu(a + b), 2),
        "add_silu":     FusionSpec("(a + b) * tl.sigmoid(a + b)", lambda a, b: _silu(a + b), 2),
        "silu_mul":     FusionSpec("(a * tl.sigmoid(a)) * b", lambda a, b: _silu(a) * b, 2),
        "gelu_mul":     FusionSpec(
            "(0.5 * a * (1.0 + (2.0 * tl.sigmoid(2.0 * (0.7978845608028654 * "
            "(a + 0.044715 * a * a * a))) - 1.0))) * b",
            lambda a, b: _gelu(a) * b, 2),
        "sigmoid_mul":  FusionSpec("tl.sigmoid(a) * b", lambda a, b: torch.sigmoid(a) * b, 2),
        "mul_relu":     FusionSpec("tl.maximum(a * b, 0.0)", lambda a, b: torch.relu(a * b), 2),
        "mul_tanh":     FusionSpec("2.0 * tl.sigmoid(2.0 * (a * b)) - 1.0",
                                   lambda a, b: torch.tanh(a * b), 2),
        # 3-input fusions (a, b, c all [M,N])
        "fma":          FusionSpec("a * b + c", lambda a, b, c: a * b + c, 3),
        "fma_relu":     FusionSpec("tl.maximum(a * b + c, 0.0)",
                                   lambda a, b, c: torch.relu(a * b + c), 3),
        "fma_gelu":     FusionSpec(
            "0.5 * (a * b + c) * (1.0 + (2.0 * tl.sigmoid(2.0 * (0.7978845608028654 * "
            "((a * b + c) + 0.044715 * (a * b + c) * (a * b + c) * (a * b + c)))) - 1.0))",
            lambda a, b, c: _gelu(a * b + c), 3),
        "add_add_relu": FusionSpec("tl.maximum(a + b + c, 0.0)",
                                   lambda a, b, c: torch.relu(a + b + c), 3),
        "add_mul":      FusionSpec("(a + b) * c", lambda a, b, c: (a + b) * c, 3),
        # --- v2 additive pointwise fusions (real GLU/gated + fused-act chains that
        #     torch runs as multiple kernels -> genuine fusion headroom) ---
        "reglu":        FusionSpec("tl.maximum(a, 0.0) * b", lambda a, b: torch.relu(a) * b, 2),
        "sub_relu":     FusionSpec("tl.maximum(a - b, 0.0)", lambda a, b: torch.relu(a - b), 2),
        "add_mul_relu": FusionSpec("tl.maximum((a + b) * c, 0.0)",
                                   lambda a, b, c: torch.relu((a + b) * c), 3),
        "mul_add_tanh": FusionSpec("2.0 * tl.sigmoid(2.0 * (a * b + c)) - 1.0",
                                   lambda a, b, c: torch.tanh(a * b + c), 3),
    }


def _gemm_fusion_specs() -> dict[str, GemmFusionSpec]:
    """GEMM + fused bias/activation epilogues (compute-bound, hipBLASLt-baselined)."""
    return {
        "gemm_bias":        GemmFusionSpec(True, "none"),
        "gemm_relu":        GemmFusionSpec(False, "relu"),
        "gemm_gelu":        GemmFusionSpec(False, "gelu"),
        "gemm_silu":        GemmFusionSpec(False, "silu"),
        "gemm_bias_relu":   GemmFusionSpec(True, "relu"),
        "gemm_bias_gelu":   GemmFusionSpec(True, "gelu"),
        "gemm_bias_silu":   GemmFusionSpec(True, "silu"),
        # --- v2 additive GEMM epilogues (compute-bound, hipBLASLt-baselined;
        #     saturating acts are adversarial-safe) ---
        "gemm_tanh":        GemmFusionSpec(False, "tanh"),
        "gemm_sigmoid":     GemmFusionSpec(False, "sigmoid"),
        "gemm_bias_tanh":   GemmFusionSpec(True, "tanh"),
        "gemm_bias_sigmoid": GemmFusionSpec(True, "sigmoid"),
    }


# torch activation (fp32 oracle + native baseline) per act code.
def _torch_act(name: str):
    import torch
    import torch.nn.functional as F
    return {
        "none": lambda y: y,
        "relu": torch.relu,
        "gelu": lambda y: F.gelu(y, approximate="tanh"),
        "silu": F.silu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
    }[name]


# Triton fp32 epilogue activation on `acc` (libdevice-free), per act code.
_TL_ACT = {
    "none": "",
    "relu": "    acc = tl.maximum(acc, 0.0)\n",
    "gelu": ("    _gi = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)\n"
             "    acc = 0.5 * acc * (1.0 + (2.0 * tl.sigmoid(2.0 * _gi) - 1.0))\n"),
    "silu": "    acc = acc * tl.sigmoid(acc)\n",
    "tanh": "    acc = 2.0 * tl.sigmoid(2.0 * acc) - 1.0\n",
    "sigmoid": "    acc = tl.sigmoid(acc)\n",
}


# op registry: name -> (family, spec)
def _registry() -> dict[str, tuple[str, object]]:
    reg: dict[str, tuple[str, object]] = {}
    for n, s in _unary_specs().items():
        reg[n] = ("unary", s)
    for n, s in _binary_specs().items():
        reg[n] = ("binary", s)
    for n, s in _reduce_specs().items():
        reg[n] = ("reduce", s)
    for n, s in _fusion_specs().items():
        reg[n] = ("fusion", s)
    for n, s in _gemm_fusion_specs().items():
        reg[n] = ("gemm_fusion", s)
    return reg


def op_names() -> list[str]:
    return sorted(_registry())


# --------------------------------------------------------------------------- #
# reference.py namespace (thin shim calls this)
# --------------------------------------------------------------------------- #
def _parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 8192}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


# --------------------------------------------------------------------------- #
# Parallel-scan primitives for the sequence / state-space breadth baselines
# --------------------------------------------------------------------------- #
# The breadth sequence engines (``kore.tasks.breadth.seq`` and ``ssm_ext``) grade
# a candidate against ``baseline_fn``.  Writing that baseline as the SAME eager
# ``for t in range(L)`` recurrence the fp32 oracle uses makes the performance bar
# a Python interpreter loop: thousands of tiny kernel launches for one op, so any
# correct fused kernel "wins" by orders of magnitude.  That is measurement
# inflation, not a speedup.
#
# These helpers are the torch formulations a practitioner actually runs - the
# chunked / parallel-scan forms used by Mamba-2's ``ssd_minimal_discrete`` and by
# the chunked linear-attention kernels - so the bar is bandwidth-bound torch
# instead of launch-bound Python.  Each is mathematically the SAME recurrence
# with a different association order, and each accumulates in fp32 (the precision
# a real scan kernel keeps its state in), so the baseline stays numerically
# equivalent to the oracle: a faster baseline cannot become a wrong baseline.
#
# All torch imports are lazy (module stays CPU/GPU-free at registry-discovery
# time) and every helper works on arbitrary leading batch dimensions.

def _tril_mask(T: int, device):
    import torch
    return torch.ones((T, T), dtype=torch.bool, device=device).tril()


def _segsum_exp(log_decay):
    """``M[..., j, i] = exp(sum_{r=i+1..j} log_decay[..., r])`` for ``i <= j``, else 0.

    The causal decay matrix of a scalar-decay recurrence.  Built from a cumulative
    sum so the masked-in entries always have a NON-POSITIVE exponent (the decays
    are <= 1), which is what keeps the chunked forms overflow-free.
    """
    import torch
    T = log_decay.shape[-1]
    cs = torch.cumsum(log_decay, dim=-1)
    m = cs[..., :, None] - cs[..., None, :]
    return torch.exp(torch.where(_tril_mask(T, m.device), m,
                                 torch.full_like(m, float("-inf"))))


def _assoc_scan_lastdim(a, b):
    """Inclusive scan ``h_t = a_t * h_{t-1} + b_t`` (h_{-1}=0) over the LAST dim.

    Hillis-Steele doubling on the associative pair operator
    ``(a1, b1) . (a2, b2) = (a1 * a2, a2 * b1 + b2)`` - ceil(log2(L)) vectorized
    steps instead of L sequential ones.  Deliberately uses only the multiplies and
    adds of the recurrence itself (no log/exp, no division), so it stays exact for
    a gate of exactly 0 (a segment reset) and needs no bound on the gate
    magnitude, unlike a cumulative-product-quotient formulation.
    """
    import torch
    L = a.shape[-1]
    a = a.to(torch.float32, copy=True)
    h = b.to(torch.float32, copy=True).expand_as(a).contiguous()
    d = 1
    while d < L:
        a_prev = a[..., : L - d].clone()
        h_prev = h[..., : L - d].clone()
        h[..., d:] += a[..., d:] * h_prev
        a[..., d:] *= a_prev
        d *= 2
    return h


def _chunk_scalar_decay_scan(log_decay, K, V, Q, chunk: int = 64):
    """Chunked, loop-free form of the SCALAR-decay state-space recurrence

        S_t = exp(log_decay_t) * S_{t-1} + K_t (outer) V_t ;   y_t = Q_t^T S_t

    ``log_decay[..., L]``, ``K``/``Q`` ``[..., L, N]``, ``V[..., L, P]`` ->
    ``y[..., L, P]``.  Intra-chunk is the decay-weighted causal score matmul;
    inter-chunk is a decay-weighted sum over per-chunk states.  This is the
    Mamba-2 SSD / chunkwise-retention formulation.  EVERY exponent evaluated is
    <= 0, so nothing can overflow at any chunk size or sequence length.
    """
    import torch
    import torch.nn.functional as F

    L = log_decay.shape[-1]
    C = max(1, min(int(chunk), L))
    nc = (L + C - 1) // C
    pad = nc * C - L
    la, Kf, Vf, Qf = (t.float() for t in (log_decay, K, V, Q))
    if pad:
        # Zero-pad the TAIL only (decay 1, zero inputs): the recurrence is causal,
        # so padded steps cannot influence any of the first L outputs.
        la = F.pad(la, (0, pad))
        Kf, Vf, Qf = (F.pad(t, (0, 0, 0, pad)) for t in (Kf, Vf, Qf))
    bat = tuple(la.shape[:-1])
    N, P = Kf.shape[-1], Vf.shape[-1]
    la = la.reshape(*bat, nc, C)
    Kf = Kf.reshape(*bat, nc, C, N)
    Vf = Vf.reshape(*bat, nc, C, P)
    Qf = Qf.reshape(*bat, nc, C, N)

    cs = torch.cumsum(la, dim=-1)                             # [..., nc, C]
    y = ((Qf @ Kf.transpose(-1, -2)) * _segsum_exp(la)) @ Vf  # intra-chunk

    Kd = Kf * torch.exp(cs[..., -1:] - cs).unsqueeze(-1)      # decay to chunk end
    U = Kd.transpose(-1, -2) @ Vf                             # [..., nc, N, P]
    S_end = torch.einsum("...cd,...dnp->...cnp", _segsum_exp(cs[..., -1]), U)
    S_in = torch.cat([torch.zeros_like(S_end[..., :1, :, :]),
                      S_end[..., :-1, :, :]], dim=-3)         # state entering chunk
    y = y + (Qf * torch.exp(cs).unsqueeze(-1)) @ S_in         # inter-chunk
    return y.reshape(*bat, nc * C, P)[..., :L, :]


def _chunk_dimgated_scan(log_decay, K, V, Q, chunk: int = 32):
    """Chunked, loop-free form of the PER-KEY-DIM gated recurrence

        S_t[i, j] = exp(log_decay_t[i]) * S_{t-1}[i, j] + K_t[i] * V_t[j]
        y_t[j]    = sum_i Q_t[i] * S_t[i, j]

    ``log_decay``/``K``/``Q`` ``[..., L, N]``, ``V[..., L, P]`` -> ``y[..., L, P]``
    (Gated Linear Attention / HGRN2).  Unlike the scalar-decay case the intra-chunk
    score matmul needs the factorization ``exp(cs_j - cs_i) = exp(cs_j) exp(-cs_i)``,
    whose second factor grows with the chunk length; the default chunk of 32 keeps
    it far inside fp32 range while the (unbounded) inter-chunk term stays in the
    overflow-free ``exp(<= 0)`` form.
    """
    import torch
    import torch.nn.functional as F

    L = log_decay.shape[-2]
    C = max(1, min(int(chunk), L))
    nc = (L + C - 1) // C
    pad = nc * C - L
    la, Kf, Vf, Qf = (t.float() for t in (log_decay, K, V, Q))
    if pad:
        la, Kf, Vf, Qf = (F.pad(t, (0, 0, 0, pad)) for t in (la, Kf, Vf, Qf))
    bat = tuple(la.shape[:-2])
    N, P = Kf.shape[-1], Vf.shape[-1]
    la = la.reshape(*bat, nc, C, N)
    Kf = Kf.reshape(*bat, nc, C, N)
    Vf = Vf.reshape(*bat, nc, C, P)
    Qf = Qf.reshape(*bat, nc, C, N)

    cs = torch.cumsum(la, dim=-2)                             # [..., nc, C, N]
    Qe = Qf * torch.exp(cs)
    Ke = Kf * torch.exp(-cs)
    A = (Qe @ Ke.transpose(-1, -2)) * _tril_mask(C, cs.device)
    y = A @ Vf                                                # intra-chunk

    gl = cs[..., -1:, :]                                      # [..., nc, 1, N]
    U = (Kf * torch.exp(gl - cs)).transpose(-1, -2) @ Vf      # [..., nc, N, P]
    G = torch.cumsum(gl.squeeze(-2), dim=-2)                  # [..., nc, N]
    M = torch.exp(G[..., :, None, :] - G[..., None, :, :]) * \
        _tril_mask(nc, G.device).unsqueeze(-1)
    S_end = torch.einsum("...cdn,...dnp->...cnp", M, U)
    S_in = torch.cat([torch.zeros_like(S_end[..., :1, :, :]),
                      S_end[..., :-1, :, :]], dim=-3)
    y = y + Qe @ S_in                                         # inter-chunk
    return y.reshape(*bat, nc * C, P)[..., :L, :]


def _lti_fft_conv(kernel, u):
    """Causal LTI convolution ``y[..., t] = sum_{s<=t} kernel[..., t-s] u[..., s]``.

    The S4/S4D production path: a time-INVARIANT diagonal SSM is a long causal
    convolution, evaluated in O(L log L) with a zero-padded FFT rather than by
    stepping the recurrence.  ``kernel`` broadcasts against ``u`` over the leading
    dims; both carry time on the last axis.
    """
    import torch
    L = u.shape[-1]
    n = 1
    while n < 2 * L:
        n *= 2
    yf = torch.fft.rfft(kernel.float(), n=n) * torch.fft.rfft(u.float(), n=n)
    return torch.fft.irfft(yf, n=n)[..., :L]


def _blocked_selective_scan(dt, x, A, B_, C_):
    """Mamba-1 selective SSM core, as a BLOCKED scan.

        h_t[d, n] = exp(dt_t[d] A[d, n]) h_{t-1}[d, n] + dt_t[d] B_t[n] x_t[d]
        y_t[d]    = sum_n C_t[n] h_t[d, n]

    ``dt``/``x`` ``[B, L, D]`` (dt already softplus'd), ``A[D, N]``,
    ``B_``/``C_`` ``[B, L, N]`` -> ``y[B, L, D]``.

    The decay varies with BOTH d and n, so - unlike the scalar-decay (Mamba-2 SSD)
    case - there is no [C, C] score matrix to chunk with, and a fully materialized
    parallel scan would need multi-gigabyte [B, L, D, N] intermediates.  This is
    what torch can honestly do instead: run all ``L/C`` chunks in parallel for
    ``C`` steps, scan the ``L/C`` chunk-boundary states, then add each chunk's
    carried-in contribution - O(2C + L/C) launches instead of O(L).  It is exactly
    the recurrence, reassociated, in fp32.
    """
    import torch
    import torch.nn.functional as F

    Bs, L, D = x.shape
    N = A.shape[1]
    C = 32 if L <= 4096 else 64
    C = max(1, min(C, L))
    nc = (L + C - 1) // C
    pad = nc * C - L
    dtf, xf, Af = dt.float(), x.float(), A.float()
    Bf, Cf = B_.float(), C_.float()
    if pad:  # decay 1 (dt=0) and zero input on the causal tail
        dtf, xf = (F.pad(t, (0, 0, 0, pad)) for t in (dtf, xf))
        Bf, Cf = (F.pad(t, (0, 0, 0, pad)) for t in (Bf, Cf))
    dt4 = dtf.reshape(Bs, nc, C, D)
    x4 = xf.reshape(Bs, nc, C, D)
    B4 = Bf.reshape(Bs, nc, C, N)
    C4 = Cf.reshape(Bs, nc, C, N)

    dev, f32 = x.device, torch.float32
    h = torch.zeros(Bs, nc, D, N, dtype=f32, device=dev)
    dA = torch.empty_like(h)                           # reused decay scratch
    y = torch.empty(C, Bs, nc, D, dtype=f32, device=dev)   # time-major: contiguous
    for j in range(C):                                 # all chunks, in parallel
        dtj = dt4[:, :, j]
        torch.mul(dtj.unsqueeze(-1), Af, out=dA).exp_()
        h.mul_(dA).addcmul_((dtj * x4[:, :, j]).unsqueeze(-1),
                            B4[:, :, j].unsqueeze(-2))
        y[j] = torch.matmul(h, C4[:, :, j].unsqueeze(-1)).squeeze(-1)

    if nc > 1:                                         # otherwise nothing is carried
        a_end = torch.exp(dt4.sum(dim=2).unsqueeze(-1) * Af)   # per-chunk decay
        h_in = torch.zeros_like(h)
        carry = torch.zeros(Bs, D, N, dtype=f32, device=dev)
        for c in range(nc):                            # scan the chunk boundaries
            h_in[:, c] = carry
            carry = a_end[:, c] * carry + h[:, c]

        # carried-in contribution: exp(P_j * A) * h_in accumulates the SAME
        # per-step decay, so it rides the recurrence instead of re-exponentiating
        # a prefix (which would need an unmaterializable [B, L, D, N] tensor).
        for j in range(C):
            torch.mul(dt4[:, :, j].unsqueeze(-1), Af, out=dA).exp_()
            h_in.mul_(dA)
            y[j] += torch.matmul(h_in, C4[:, :, j].unsqueeze(-1)).squeeze(-1)
    return y.permute(1, 2, 0, 3).reshape(Bs, nc * C, D)[:, :L]


# --------------------------------------------------------------------------- #
# Vendor-baseline resolution (Wave1 aiter_ref wrappers)
# --------------------------------------------------------------------------- #
# The honest performance bar for a generated op is the kernel the production
# serving stack actually calls, when one exists for this (family, dtype).  Where
# AITER/hipBLASLt ship a real fused kernel we point ``baseline_fn`` at the thin
# wrapper in ``kore.tasks.aiter_ref``; families with NO vendor kernel (plain
# elementwise unary/binary, row reductions, non-gated fusions, w4a16/int4, and
# norm/softmax BACKWARD which is not generated here) keep the torch baseline and
# are labeled ``eager``/``framework``.  The whole vendor path is gated behind
# ``KORE_USE_VENDOR_BASELINE`` (default ON, toggleable) so a torch-only bar can be
# forced.  ALL vendor imports are lazy (inside the returned closures) so CPU
# ``kore tasks`` discovery never needs a GPU or the aiter runtime.

# Generated FUSION ops that ARE the LLM gated activations AITER ships as fused
# kernels: the generated op takes two [M,N] operands (a, b) and computes
# ``act(a) * b`` -- identical to AITER ``<act>_and_mul`` on ``cat([a, b], -1)``.
_VENDOR_GATED_FUSION = {
    "silu_mul": "aiter_silu_and_mul",
    "gelu_mul": "aiter_gelu_tanh_and_mul",
}


def _use_vendor_baseline() -> bool:
    """Whether to resolve generated baselines to vendor kernels (default ON)."""
    return os.environ.get("KORE_USE_VENDOR_BASELINE", "1").strip().lower() in (
        "1", "true", "yes", "on")


def _vendor_baseline_kind(op: str, family: str, dtype: str) -> str:
    """Static label for the baseline this (op, family, dtype) will use:
    ``vendor`` (an AITER/hipBLASLt production kernel), ``torch_compile``
    (compiler-fused torch bar for fusion/gemm_fusion when KORE_COMPILE_BASELINE
    is on), or ``eager`` (plain torch multi-kernel).  Independent of GPU: it
    reflects which code path ``_vendor_baseline`` selects."""
    if _use_vendor_baseline():
        if family == "gemm_fusion":
            return "vendor"           # hipBLASLt dense GEMM / torch._scaled_mm fp8
        if family == "fusion" and op in _VENDOR_GATED_FUSION:
            return "vendor"           # AITER fused gated activation
    # No vendor kernel for this family/op -> the torch bar.
    if family in ("fusion", "gemm_fusion"):
        return "torch_compile" if _compile_baseline_enabled() else "eager"
    return "eager"


def _vendor_baseline(op: str, family: str, dtype: str, torch_baseline,
                     eager_fn=None):
    """Return the baseline callable for a generated op.

    When ``KORE_USE_VENDOR_BASELINE`` is on AND a production kernel exists for
    this (family, dtype), return a closure that calls the vendor wrapper (imports
    lazily; any failure degrades to ``torch_baseline`` so a missing aiter runtime
    never breaks a bench).  Otherwise return ``torch_baseline`` unchanged.

    ``eager_fn`` is the PURE torch composition behind ``torch_baseline`` (which
    already resolves the compile gate itself).  The hipBLASLt epilogue path
    compiles ``eager_fn``, never ``torch_baseline`` - handing dynamo a callable
    that re-reads ``os.environ`` makes tracing fail outright, so double-wrapping
    would break every gemm_fusion bench the moment the fused bar is on."""
    if not _use_vendor_baseline():
        return torch_baseline

    # --- FUSION: AITER fused gated activations (silu_mul / gelu_mul) ----------
    if family == "fusion" and op in _VENDOR_GATED_FUSION:
        wrapper_name = _VENDOR_GATED_FUSION[op]

        def _vendor_gated(a, b):
            import torch  # noqa: F401 - keep torch import lazy/local
            from kore.tasks import aiter_ref
            wrapper = getattr(aiter_ref, wrapper_name)
            # AITER <act>_and_mul consumes one [M, 2*inter] tensor and returns
            # act(first_half) * second_half -> identical to the generated op's
            # act(a) * b on cat([a, b], -1).
            return wrapper(torch.cat([a, b], dim=-1))

        return _vendor_gated

    # --- GEMM_FUSION: hipBLASLt fused-epilogue GEMM / torch._scaled_mm (fp8) --
    if family == "gemm_fusion":
        spec: GemmFusionSpec = _registry()[op][1]
        act = _torch_act(spec.act)
        has_bias = spec.has_bias

        if dtype == "fp8":
            def _vendor_scaled_mm(*xs):
                # OCP-fp8 (e4m3fn) fused GEMM via torch._scaled_mm -> hipBLASLt.
                # Per-tensor descales recovered from the fp8 operands; bias fused
                # in the scaled_mm epilogue, activation applied on the bf16 out.
                import torch
                from kore.tasks.aiter_ref import _mark_baseline
                a, b = xs[0], xs[1]
                one = torch.tensor(1.0, dtype=torch.float32, device=a.device)
                bias = xs[2].to(torch.bfloat16) if has_bias else None
                try:
                    y = torch._scaled_mm(
                        a, b, scale_a=one, scale_b=one,
                        bias=bias, out_dtype=torch.bfloat16,
                    )
                    _mark_baseline("hipblaslt_vendor")
                except Exception:  # noqa: BLE001 - scaled_mm unsupported -> hipBLASLt bf16
                    _mark_baseline("hipblaslt_vendor")
                    y = torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16))
                    if has_bias:
                        y = y + xs[2].to(torch.bfloat16)
                return act(y)

            return _vendor_scaled_mm

        # The eager matmul+bias+act epilogue authored by the caller (identical math
        # to the torch bar) - a pure tensor function, so compile can fuse it.
        torch_baseline_gemm = eager_fn if eager_fn is not None else torch_baseline

        def _vendor_hipblaslt_epilogue(*xs):
            # bf16/fp16: torch.matmul dispatches to the hipBLASLt tuned GEMM (the
            # production dense-GEMM library); the bias+activation epilogue is the
            # fused-epilogue bar.  torch.compile fuses the epilogue INTO the GEMM
            # (default; still hipBLASLt underneath).
            from kore.tasks.aiter_ref import _mark_baseline
            _mark_baseline("hipblaslt_vendor")
            return _fused_baseline(torch_baseline_gemm, f"gemm_fusion:{op}:{dtype}")(*xs)

        return _vendor_hipblaslt_epilogue

    # No vendor kernel: keep the torch baseline (eager/torch_compile).
    return torch_baseline


def make_reference(op: str, family: str, dtype: str) -> dict:
    """Build the reference.py module namespace for a generated op."""
    import torch

    tdt = _torch_dtype(dtype)
    spec = _registry()[op][1]

    def _mk(domain: str):
        def gen(shape, device="cuda", seed=0):
            g = torch.Generator(device=device).manual_seed(seed)
            M, N = shape["M"], shape["N"]
            t = torch.randn((M, N), generator=g, device=device, dtype=torch.float32)
            if domain == "pos":
                t = t.abs() + 0.1
            return t.to(tdt)
        return gen

    if family == "unary":
        s: UnarySpec = spec
        gx = _mk(s.domain)

        def get_inputs(shape, device="cuda", seed=0):
            return (gx(shape, device, seed),)

        def ref_fn(x):
            return s.torch_fn(x.float()).to(x.dtype)

        def baseline_fn(x):
            return s.torch_fn(x)

        arity = 1
        baseline_kind = _vendor_baseline_kind(op, family, dtype)
    elif family == "binary":
        s: BinarySpec = spec
        ga = _mk("signed")
        gb = _mk(s.domain_b)

        def get_inputs(shape, device="cuda", seed=0):
            return (ga(shape, device, seed), gb(shape, device, seed + 1))

        def ref_fn(x, y):
            return s.torch_fn(x.float(), y.float()).to(x.dtype)

        def baseline_fn(x, y):
            return s.torch_fn(x, y)

        arity = 2
        baseline_kind = _vendor_baseline_kind(op, family, dtype)
    elif family == "reduce":
        s: ReduceSpec = spec
        gx = _mk("signed")

        def get_inputs(shape, device="cuda", seed=0):
            return (gx(shape, device, seed),)

        def ref_fn(x):
            return s.torch_fn(x.float()).to(x.dtype)

        def baseline_fn(x):
            return s.torch_fn(x)

        arity = 1
        baseline_kind = _vendor_baseline_kind(op, family, dtype)
    elif family == "fusion":
        s: FusionSpec = spec
        gen = _mk("signed")

        def get_inputs(shape, device="cuda", seed=0):
            return tuple(gen(shape, device, seed + i) for i in range(s.arity))

        def ref_fn(*xs):
            return s.torch_fn(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            # Honest fused bar (DEFAULT): torch.compile FUSES the elementwise chain
            # into one kernel, so the candidate must beat the COMPILER, not unfused
            # eager (which would inflate the speedup). Eager multi-kernel only on a
            # deliberate opt-out or if compile is unavailable.
            return _fused_baseline(s.torch_fn, f"fusion:{op}:{dtype}")(*xs)

        baseline_fn = _vendor_baseline(op, family, dtype, baseline_fn)
        baseline_kind = _vendor_baseline_kind(op, family, dtype)
        arity = s.arity
    elif family == "gemm_fusion":
        s: GemmFusionSpec = spec
        act = _torch_act(s.act)

        def get_inputs(shape, device="cuda", seed=0):
            g = torch.Generator(device=device).manual_seed(seed)
            M, N, K = shape["M"], shape["N"], shape["K"]
            # 1/sqrt(K) scale keeps the accumulated GEMM magnitude ~O(1) (stable bf16).
            sc = 1.0 / (K ** 0.5)
            a = (torch.randn((M, K), generator=g, device=device, dtype=torch.float32) * sc).to(tdt)
            b = (torch.randn((K, N), generator=g, device=device, dtype=torch.float32) * sc).to(tdt)
            if s.has_bias:
                bias = (torch.randn((N,), generator=g, device=device, dtype=torch.float32)).to(tdt)
                return (a, b, bias)
            return (a, b)

        def ref_fn(*xs):
            a, b = xs[0].float(), xs[1].float()
            y = a @ b
            if s.has_bias:
                y = y + xs[2].float()
            return act(y).to(xs[0].dtype)

        def _eager_gemm_epilogue(*xs):
            y = torch.matmul(xs[0], xs[1])
            if s.has_bias:
                y = y + xs[2]
            return act(y)

        def baseline_fn(*xs):
            # Honest fused bar (DEFAULT): torch.compile fuses the bias+activation
            # EPILOGUE into the hipBLASLt GEMM, so the candidate must beat the
            # compiler-fused epilogue-GEMM, not the unfused matmul+bias+act chain
            # (which would inflate the speedup). Eager only on a deliberate opt-out.
            return _fused_baseline(_eager_gemm_epilogue, f"gemm_fusion:{op}:{dtype}")(*xs)

        baseline_fn = _vendor_baseline(op, family, dtype, baseline_fn,
                                       eager_fn=_eager_gemm_epilogue)
        baseline_kind = _vendor_baseline_kind(op, family, dtype)
        arity = 3 if s.has_bias else 2
    else:
        raise ValueError(f"unknown family {family!r}")

    ns = {
        "parse_shape": _parse_shape,
        "get_inputs": get_inputs,
        "ref_fn": ref_fn,
        "baseline_fn": baseline_fn,
        "arity": arity,
        "entry_name": op,
        "dtype_name": dtype,
        "family": family,
        "baseline_kind": baseline_kind,
        # True only when an explicit falsey KORE_COMPILE_BASELINE dropped a
        # fusion/gemm_fusion task back to the unfused eager bar, so a run graded
        # against the inflated baseline carries that fact on the reference itself.
        "baseline_compile_opt_out": (
            family in ("fusion", "gemm_fusion")
            and compile_baseline_status()["source"] == "env_opt_out"),
        "mutates_input": False,
    }
    ns[f"{op}_ref"] = ref_fn   # conventional alias
    return ns


# --------------------------------------------------------------------------- #
# Triton seed source (a REAL compiling starter kernel)
# --------------------------------------------------------------------------- #
_UNARY_TMPL = '''"""GENERATED seed Triton kernel for the {op} ({dtype}) activation.

Elementwise {op}, 2D-tiled, fp32 math, {tldt} store. A correct-but-naive starting
point the KORE policy learns to optimize against the framework production baseline.
Regenerate via kore/tasks/generate_ops.py - do not hand-edit.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, y_ptr, stride_xm, stride_ym, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    y = {expr}
    tl.store(y_ptr + row * stride_ym + offs, y.to({tldt}), mask=mask)


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _{op}_kernel[grid](x, y, x.stride(0), y.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    return y
'''

_BINARY_TMPL = '''"""GENERATED seed Triton kernel for the {op} ({dtype}) binary op.

Elementwise {op}(a, b), 2D-tiled, fp32 math, {tldt} store. Regenerate via
kore/tasks/generate_ops.py - do not hand-edit.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _{op}_kernel(a_ptr, b_ptr, o_ptr, stride_am, stride_bm, stride_om, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(a_ptr + row * stride_am + offs, mask=mask, other=1.0).to(tl.float32)
    y = tl.load(b_ptr + row * stride_bm + offs, mask=mask, other=1.0).to(tl.float32)
    o = {expr}
    tl.store(o_ptr + row * stride_om + offs, o.to({tldt}), mask=mask)


def {op}(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N = a.shape
    o = torch.empty_like(a)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _{op}_kernel[grid](a, b, o, a.stride(0), b.stride(0), o.stride(0), N,
                       BLOCK_N=BLOCK_N, num_warps=4)
    return o
'''

_REDUCE_TMPL = '''"""GENERATED seed Triton kernel for the {op} ({dtype}) row reduction.

Per-row reduction [M,N]->[M], fp32 accumulate, {tldt} store. Regenerate via
kore/tasks/generate_ops.py - do not hand-edit.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, y_ptr, stride_xm, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32) + ({init})
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=({other})).to(tl.float32)
        acc = {combine}
    v = {final}
    v = {post}
    tl.store(y_ptr + row, v.to({tldt}))


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty((M,), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    _{op}_kernel[(M,)](x, y, x.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    return y
'''


_FUSION2_TMPL = '''"""GENERATED seed Triton kernel for the {op} ({dtype}) fusion.

Pointwise FUSION out = f(a, b) computed in ONE pass. torch-eager runs this as
separate kernels, so a fused kernel saves HBM round-trips -> real speedup headroom.
Regenerate via kore/tasks/generate_ops.py - do not hand-edit.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _{op}_kernel(a_ptr, b_ptr, o_ptr, sa, sb, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    a = tl.load(a_ptr + row * sa + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * sb + offs, mask=mask, other=0.0).to(tl.float32)
    o = {expr}
    tl.store(o_ptr + row * so + offs, o.to({tldt}), mask=mask)


def {op}(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N = a.shape
    o = torch.empty_like(a)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _{op}_kernel[grid](a, b, o, a.stride(0), b.stride(0), o.stride(0), N,
                       BLOCK_N=BLOCK_N, num_warps=4)
    return o
'''

_FUSION3_TMPL = '''"""GENERATED seed Triton kernel for the {op} ({dtype}) fusion.

Pointwise FUSION out = f(a, b, c) computed in ONE pass (vs torch-eager multi-kernel).
Regenerate via kore/tasks/generate_ops.py - do not hand-edit.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _{op}_kernel(a_ptr, b_ptr, c_ptr, o_ptr, sa, sb, sc, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    a = tl.load(a_ptr + row * sa + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * sb + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + row * sc + offs, mask=mask, other=0.0).to(tl.float32)
    o = {expr}
    tl.store(o_ptr + row * so + offs, o.to({tldt}), mask=mask)


def {op}(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    M, N = a.shape
    o = torch.empty_like(a)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _{op}_kernel[grid](a, b, c, o, a.stride(0), b.stride(0), c.stride(0), o.stride(0), N,
                       BLOCK_N=BLOCK_N, num_warps=4)
    return o
'''


_GEMM_TMPL = '''"""GENERATED seed Triton GEMM + fused epilogue for {op} ({dtype}).

C = act(A @ B [+ bias]) in ONE kernel (fp32 accumulate, {tldt} store). torch runs
this as matmul (-> hipBLASLt) + bias + activation = SEPARATE kernels, so fusing
saves HBM round-trips of the [M,N] output -> real headroom vs the vendor path.
Grouped tiling + K-mask (ROCm/CDNA-safe on gfx942/gfx950, libdevice-free act).
Regenerate via kore/tasks/generate_ops.py - do not hand-edit.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _{op}_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
{bias_block}{act_block}    c = acc.to({tldt})
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def {op}({args}) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype={torch_dt})
    if M <= 16:
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, nw, ns = 16, 128, 64, 1, 4, 2
    else:
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, nw, ns = 128, 128, 32, 8, 4, 2
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _{op}_kernel[grid](
        a, b, c, {bias_arg},
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=nw, num_stages=ns,
    )
    return c
'''


def seed_source(op: str, family: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]
    if family == "gemm_fusion":
        s: GemmFusionSpec = _registry()[op][1]
        torch_dt = f"torch.{DTYPES[dtype][0]}"
        if s.has_bias:
            bias_block = ("    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, "
                          "other=0.0).to(tl.float32)\n    acc += bias[None, :]\n")
            args, bias_arg = "a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor", "bias"
        else:
            bias_block = ""
            args, bias_arg = "a: torch.Tensor, b: torch.Tensor", "a"  # dummy ptr, unused
        return _GEMM_TMPL.format(op=op, dtype=dtype, tldt=tldt, torch_dt=torch_dt,
                                 bias_block=bias_block, act_block=_TL_ACT[s.act],
                                 args=args, bias_arg=bias_arg)
    if family == "unary":
        s: UnarySpec = _registry()[op][1]
        return _UNARY_TMPL.format(op=op, dtype=dtype, tldt=tldt, expr=s.tl_expr)
    if family == "binary":
        s = _registry()[op][1]
        return _BINARY_TMPL.format(op=op, dtype=dtype, tldt=tldt, expr=s.tl_expr)
    if family == "reduce":
        s = _registry()[op][1]
        return _REDUCE_TMPL.format(op=op, dtype=dtype, tldt=tldt, init=s.init,
                                   other=s.other, combine=s.combine, final=s.final,
                                   post=s.post)
    if family == "fusion":
        s = _registry()[op][1]
        tmpl = _FUSION3_TMPL if s.arity == 3 else _FUSION2_TMPL
        return tmpl.format(op=op, dtype=dtype, tldt=tldt, expr=s.tl_expr)
    raise ValueError(family)


# --------------------------------------------------------------------------- #
# Generic driver (correctness + cold-cache bench + post-timing anti-hack)
# --------------------------------------------------------------------------- #
DRIVER_CAPABILITY_PROTOCOL = 2
DRIVER_PROTOCOL_ID = "kore-paired-v2"
PUBLICATION_GUARANTEES = {
    "bench_both": True,
    "multi_shape": True,
    "paired_samples": True,
    "raw_samples": True,
    "fresh_inputs_per_pair": True,
    "balanced_ab_ba": True,
    "mutation_semantics": True,
    "postcheck_all_shapes": True,
}
_CAPABILITY_DEFAULTS = {
    "protocol": DRIVER_CAPABILITY_PROTOCOL,
    "protocol_id": DRIVER_PROTOCOL_ID,
    "performance_eligible": False,
    "bench_both": False,
    "multi_shape": False,
    "paired_samples": False,
    "raw_samples": False,
    "fresh_inputs_per_pair": False,
    "balanced_ab_ba": False,
    "mutation_semantics": False,
    "postcheck_all_shapes": False,
}


def publication_driver_capabilities() -> dict:
    return {
        "protocol": DRIVER_CAPABILITY_PROTOCOL,
        "protocol_id": DRIVER_PROTOCOL_ID,
        "performance_eligible": True,
        **PUBLICATION_GUARANTEES,
    }


def emit_driver_capabilities(overrides: Optional[dict] = None) -> None:
    """Emit the versioned verifier/driver capability handshake.

    Drivers must opt in to every batched-timing guarantee.  The environment
    accepts this exact machine-readable line only; merely mentioning a helper or
    CLI flag in driver source is never treated as evidence of support.
    """
    caps = dict(_CAPABILITY_DEFAULTS)
    caps.update(overrides or {})
    print("KORE_DRIVER_CAPABILITIES: "
          + json.dumps(caps, sort_keys=True, separators=(",", ":")))


def _snr_db(out, ref_out) -> float:
    o, r = out.float(), ref_out.float()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


# --------------------------------------------------------------------------- #
# Elementwise correctness tolerance
# --------------------------------------------------------------------------- #
# Two finite-precision evaluations of the same mathematical function do not
# produce the same bits.  What a correctness gate has to decide is whether the
# disagreement is attributable to the arithmetic of the DECLARED OUTPUT FORMAT --
# which is a property of the dtype and of the tensor's scale, never a universal
# constant.
#
# The tolerance is built from the two things that actually generate the
# difference:
#
# 1. REPRESENTATION.  Both results are stored in the output format, so two exact
#    values that straddle a representable boundary land on ADJACENT codes.
#    Agreement can only ever be asserted to within a small number of code steps.
#    ``torch.finfo(d).eps`` is that step, relative, for a float format; for an
#    integer-coded (quantized) output it is exactly 1.
#
# 2. ACCUMULATION.  The rounding error of a reduction is bounded by the magnitude
#    of the ACCUMULATION, not by the magnitude of the RESULT.  An output element
#    that is small only because its terms cancelled still carries the absolute
#    error the summation incurred, so a relative (``rtol * |r|``) tolerance is the
#    wrong shape for any reduction: it shrinks exactly where the error does not.
#    The scale must come from the tensor.  ``max|r|`` over the ORACLE's finite
#    entries is the available bound on any single element's accumulation scale.
#    (The RMS would be too tight for the heterogeneous tensors here -- an
#    attention-backward dK has peak/RMS ~= 32 and each row's error tracks its own
#    row's accumulation, so the max-norm bound has to be referenced to the peak,
#    the same convention LAPACK's scaled-residual tests use.)
#
# giving
#
#     |o - r|  <=  ULP_STEPS * step(dtype) * peak(r)
#
# The peak is taken from the reference, never the candidate, so a kernel cannot
# widen its own tolerance by emitting a large value.
#
# What the previous fixed ``atol = rtol = 1e-2`` did instead was apply an
# fp32-calibrated number to every format: it demands 1% relative agreement from
# fp8_e4m3, whose own relative resolution is 12.5%, so NO fp8 kernel could ever
# satisfy it; and 0.01 absolute agreement from an int8 code, where the
# quantizer's own rounding boundary moves a code by a full LSB.
#
# This gate is one of TWO that must both hold; the task's declared SNR gate is
# the other.  They are complementary and neither is redundant: SNR is a global
# (L2) measure that a sparse defect -- one masked tail, one boundary row -- barely
# moves, and this elementwise bound is a max-norm measure that a small
# broad-spectrum bias barely moves.

# Representable steps of the output format that two correct implementations of
# the same math may differ by.  Measured on gfx950 across this corpus, correct
# seeds sit at 0.5-0.8 steps; 2 leaves ~3x headroom without admitting a
# difference the format could not have produced.
CORRECTNESS_ULP_STEPS = 2.0
# Integer-CODED (quantizer) outputs are EXACT by default.  A pure quantizer --
# input tensor in, codes out -- is a deterministic function of bits both
# implementations already hold, so any disagreement is a real difference in the
# rounding rule or the scale expression, not noise.  Allowing even one code here
# hides exactly that: a seed that rounded ties away from zero while its oracle
# used torch.round (ties to even) disagreed on 0.11% of codes and still passed.
# Ops whose quantizer consumes a COMPUTED value (a normalized activation, an Adam
# update) declare ``code_tolerance_steps = 1`` instead -- see
# :func:`_tolerance_declarations`.
CORRECTNESS_CODE_STEPS = 0.0
# Relative slack for REASSOCIATING an fp32 reduction.  Recursive summation of K
# terms drifts by ~sqrt(K) * 2**-24; the widest reductions in this corpus are
# ~2**14 elements, i.e. 7.6e-6, and 2**-16 is the next binade up.  It sits below
# every low-precision format's own resolution, so it only ever binds for an fp32
# output -- where the storage step (2**-23) is far finer than the arithmetic.
FP32_REASSOC_SLACK = 2.0 ** -16
# Integer outputs that carry a QUANTIZER CODE, so "one step" means one LSB of the
# code.  Every other integer output in this corpus is an index, count or offset,
# where "off by one" is a different answer rather than a rounder one.
_CODED_INT_DTYPES = frozenset({"int8", "uint8"})


def _format_step(dtype, peak: float) -> tuple[float, str]:
    """One representable step of ``dtype`` at magnitude ``peak``, and its kind.

    ``kind`` is ``"code"`` for an integer-coded (quantized) output, ``"index"``
    for an integer index/count/mask, and ``"float"`` otherwise.  For a float
    format the step is absolute: the format's relative resolution scaled by
    ``peak``, the oracle tensor's largest finite magnitude -- floored at the
    SUBNORMAL spacing, because below the smallest normal a float's spacing stops
    shrinking with the value and becomes constant.  Several backward passes in
    this corpus land entirely in fp16's subnormal range (peak ~3e-07 against a
    smallest normal of 6.1e-05), where ``eps * peak`` under-states the true step
    by two orders of magnitude and would fail a kernel that is off by one code.
    """
    import torch

    if dtype == torch.bool or not dtype.is_floating_point:
        name = str(dtype).rsplit(".", 1)[-1]
        return (1.0, "code") if name in _CODED_INT_DTYPES else (0.0, "index")
    info = torch.finfo(dtype)
    resolution = max(float(info.eps), FP32_REASSOC_SLACK)
    subnormal = float(info.tiny) * float(info.eps)   # constant spacing below `tiny`
    return max(resolution * abs(float(peak)), subnormal), "float"


def correctness_tolerance(dtype, peak: float,
                          code_steps: float = CORRECTNESS_CODE_STEPS
                          ) -> tuple[float, float, str]:
    """``(absolute tolerance, one format step, kind)`` for an output of ``dtype``.

    ``peak`` must be the ORACLE's largest finite magnitude, never the
    candidate's, so a kernel cannot widen its own tolerance.  An index/count
    output gets a tolerance of exactly zero, and its step is one index position
    so that a disagreement still reports in a meaningful unit.
    """
    step, kind = _format_step(dtype, peak)
    if kind == "index":
        return 0.0, 1.0, kind
    steps = code_steps if kind == "code" else CORRECTNESS_ULP_STEPS
    return steps * step, step, kind


# --------------------------------------------------------------------------- #
# Per-op tolerance declarations
# --------------------------------------------------------------------------- #
# Some ops cannot be judged from the OUTPUT TENSOR'S DTYPE alone, so the
# reference declares the missing fact.  Every declaration defaults to off, so an
# op that says nothing is judged by the plain format-step gate above.
#
# * ``output_value_grid`` -- the output is stored in one format but its VALUES
#   live on a coarser grid, because the op ends in a quantizer.  A requantizing
#   GEMM writes bf16, but every value it can produce is an fp8 code times a
#   scale, so its true granularity is the fp8 step (16x the bf16 step) and a
#   one-code disagreement is not a defect.
#
# * ``selection_rel_tol`` / ``selection_index_tol`` -- the output is a
#   DISCONTINUOUS function of a floating-point cumulative sum (top-p nucleus,
#   typical-mass mask, inverse-CDF sampling).  The token sitting exactly at the
#   mass threshold moves across it under any last-bit change in the
#   probabilities, so two correct implementations differ by that token's own
#   mass (float outputs) or by a few positions along the monotone CDF (index
#   outputs).  Measured on gfx950, an equally-correct reassociation of the
#   oracle's own softmax flips the boundary on ~1-2% of rows.  This is a
#   property of the operator, not of the kernel, and no tolerance on the inputs
#   removes it -- so it is declared per op and bounded, never applied globally.
# * ``code_tolerance_steps`` -- the quantizer's INPUT is itself computed (a
#   normalized activation, an Adam update), so the two implementations feed the
#   rounding boundary slightly different fp32 values and one code step is
#   unavoidable.  A PURE quantizer declares nothing and must match exactly.
def _tolerance_declarations(ref) -> dict:
    grid = getattr(ref, "output_value_grid", None)
    return {
        "grid": grid if isinstance(grid, (tuple, list)) else (grid,),
        "selection_rel": float(getattr(ref, "selection_rel_tol", 0.0) or 0.0),
        "selection_index": int(getattr(ref, "selection_index_tol", 0) or 0),
        "code_steps": float(getattr(ref, "code_tolerance_steps",
                                    CORRECTNESS_CODE_STEPS)),
    }


def _num_correct_trials() -> int:
    try:
        return max(5, int(os.environ.get("KORE_CORRECTNESS_TRIALS", "5")))
    except ValueError:
        return 5


def _flush_l2() -> None:
    import torch
    if getattr(_flush_l2, "_buf", None) is None:
        _flush_l2._buf = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    _flush_l2._buf.zero_()


def _time_median(fn, warmup: int, iters: int) -> float:
    """Warmup, then time ``iters`` cold-cache iterations; return the median ms.

    Same protocol as :func:`_time_fn` (L2 flush per iter under KORE_BENCH_COLD,
    CUDA-event timing, median of sorted samples) but RETURNS the median instead of
    printing - so a single process can time several impls/runs back-to-back."""
    import torch
    cold = os.environ.get("KORE_BENCH_COLD", "1") != "0"
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if cold:
            _flush_l2()
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    return times[len(times) // 2]


def _time_fn(fn, warmup: int, iters: int) -> int:
    import torch
    cold = os.environ.get("KORE_BENCH_COLD", "1") != "0"
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if cold:
            _flush_l2()
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    for t in times:
        print(f"wall_ms: {t:.4f}")
    print(f"median_ms: {times[len(times) // 2]:.4f}")
    return 0


def _load_candidate(task_dir: str, entry: str):
    # cache the module so a stateful kernel's globals persist bench -> post-timing
    # re-verification (anti invocation-count timing hack).
    if getattr(_load_candidate, "_mod", None) is None:
        path = os.path.join(task_dir, "kernel.py")
        spec = importlib.util.spec_from_file_location("candidate_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _load_candidate._mod = mod
    return getattr(_load_candidate._mod, entry)


# Families whose inputs are plain float tensors, so the generic adversarial fills
# (fill every float input with a hard regime) are a valid, exhaustive-of-the-
# qualitative-cases verification battery. Quantized/structured-input ops (fp8/int8
# GEMM, etc.) must instead author their own ``adversarial_inputs`` on the reference.
_GENERIC_ADV_FAMILIES = ("unary", "binary", "reduce", "fusion", "gemm_fusion")


def _adversarial_fills(inputs):
    """Structured hard inputs that break lucky-pass / edge-case-missing kernels
    (verification-in-the-loop). Each fill preserves every input's shape/dtype/device
    but replaces its FLOAT values with a canonical hard regime (integer/index inputs
    are left intact). Yields ``(name, inputs_tuple)``."""
    import torch
    patterns = {
        "zeros": lambda t: torch.zeros_like(t),
        "ones": lambda t: torch.ones_like(t),
        "neg_ones": lambda t: -torch.ones_like(t),
        "large": lambda t: torch.full_like(t, 1.0e3),
        "neg_large": lambda t: torch.full_like(t, -1.0e3),
        "small": lambda t: torch.full_like(t, 1.0e-3),
        # Alternating ±1. Build the parity in int64 (torch.arange) - NOT via a
        # cumsum in the tensor's own dtype: an fp16/bf16 cumsum over a large tensor
        # overflows (fp16 caps at 65504) to inf, and inf % 2 == nan, which silently
        # poisoned the input with NaNs and false-rejected correct fp16 kernels.
        "sign_alt": lambda t: ((torch.arange(t.numel(), device=t.device) % 2) * 2 - 1)
                                .to(t.dtype).reshape(t.shape),
    }
    def _fill(fill, t):
        return fill(t) if (torch.is_tensor(t) and torch.is_floating_point(t)) else t

    for name, fill in patterns.items():
        yield name, tuple(_fill(fill, t) for t in inputs)


def _adversarial_sets(ref, shape):
    """Op-class-aware adversarial input battery (or None if not checkable).

    Priority: an op-authored ``ref.adversarial_inputs(shape, device=...)`` (used by
    vendor/quantized ops that must respect fp8/int8 quantization + scale structure);
    otherwise the generic float fills for the plain-float generated families."""
    if hasattr(ref, "adversarial_inputs"):
        return list(ref.adversarial_inputs(shape, device="cuda"))
    if getattr(ref, "family", None) in _GENERIC_ADV_FAMILIES:
        return list(_adversarial_fills(ref.get_inputs(shape, device="cuda", seed=0)))
    return None


def _as_tuple(x):
    return x if isinstance(x, (tuple, list)) else (x,)


def _output_pairs(out, ref_out, expected_dtypes=None):
    """Return contract-checked output pairs, or ``None`` on any ABI mismatch.

    Tuple/list structure and arity are exact.  Every leaf must be a tensor with
    the oracle's shape and its declared dtype (the oracle dtype by default).
    """
    import torch

    out_seq = isinstance(out, (tuple, list))
    ref_seq = isinstance(ref_out, (tuple, list))
    if out_seq != ref_seq:
        return None
    outs, refs = _as_tuple(out), _as_tuple(ref_out)
    if len(outs) != len(refs):
        return None
    dtypes = list(expected_dtypes) if expected_dtypes is not None else [
        r.dtype if torch.is_tensor(r) else None for r in refs
    ]
    if len(dtypes) != len(refs):
        return None
    pairs = []
    for o, r, dtype in zip(outs, refs, dtypes):
        if not torch.is_tensor(o) or not torch.is_tensor(r):
            return None
        if tuple(o.shape) != tuple(r.shape) or o.dtype != dtype:
            return None
        pairs.append((o, r))
    return pairs


def _clone_value(value):
    """Recursively clone tensor storage while preserving container structure."""
    import torch
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_value(v) for v in value)
    if isinstance(value, list):
        return [_clone_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    return value


def _clone_inputs(inputs):
    """Clone all nested tensor storage for isolated candidate/reference calls."""
    return tuple(_clone_value(t) for t in inputs)


def _make_paired_invokers(inputs, candidate_call, baseline_call,
                          mutates_input: bool):
    """Build value-identical, storage-disjoint callables for one timing pair."""
    candidate_inputs = _clone_inputs(inputs)
    baseline_inputs = _clone_inputs(inputs)

    def _args(seed):
        return _clone_inputs(seed) if mutates_input else seed

    return (
        lambda: candidate_call(_args(candidate_inputs)),
        lambda: baseline_call(_args(baseline_inputs)),
    )


def _compare_outputs(out, ref_out, atol=None, rtol=None, expected_dtypes=None,
                     stats=None, declared=None):
    """SNR/max_diff/agreement over single-tensor OR multi-output (tuple) results.

    Returns ``(worst_snr_db, max_abs_diff, agree_all)`` - the worst SNR and the
    logical-AND of the elementwise agreement test across every output tensor.

    The elementwise test is :func:`correctness_tolerance`: a per-output-tensor
    bound in representable steps of that tensor's own storage format, scaled by
    the oracle's peak magnitude.  ``atol``/``rtol`` are honoured only when a
    caller passes them explicitly (the hand-authored ``_attn_common`` /
    ``_moe_common`` drivers do), in which case the legacy
    ``|o - r| <= atol + rtol * |r|`` test is used unchanged.

    NON-FINITE-AWARE: a correct kernel must reproduce the reference's NaN/Inf
    STRUCTURE exactly (same non-finite positions, same inf sign), and match
    closely on the FINITE elements. This is essential for the adversarial battery:
    hard fills legitimately drive an op out of its finite range (rsqrt/log/sqrt of
    a negative -> NaN, 1/0 -> Inf) in BOTH the reference and a correct candidate.
    The old code compared with plain ``allclose`` (NaN != NaN -> False) and
    ``_snr_db`` (ref norm inf/nan -> -999), so it FALSE-REJECTED correct kernels on
    those regimes. Matching the non-finite structure is correctness, not a hack: a
    kernel that is wrong anywhere still fails on the finite elements or the
    structure check, and on the many finite adversarial regimes.

    ``stats``, when a dict is supplied, receives the worst elementwise error
    expressed in representable steps of the output format -- the dtype-normalised
    number that makes a verdict readable across bf16/fp8/int8 outputs."""
    import torch
    legacy = atol is not None or rtol is not None
    atol = 1e-2 if atol is None else atol
    rtol = 1e-2 if rtol is None else rtol
    decl = declared or {"grid": (None,), "selection_rel": 0.0,
                        "selection_index": 0, "code_steps": CORRECTNESS_CODE_STEPS}
    pairs = _output_pairs(out, ref_out, expected_dtypes=expected_dtypes)
    if pairs is None:
        return -999.0, float("inf"), False
    worst, maxd, ok = 999.0, 0.0, True
    for index, (o, r) in enumerate(pairs):
        of, rf = o.float(), r.float()
        rnan, onan = torch.isnan(rf), torch.isnan(of)
        rpos, opos = torch.isposinf(rf), torch.isposinf(of)
        rneg, oneg = torch.isneginf(rf), torch.isneginf(of)
        # NaN, +Inf, and -Inf are distinct semantic values.  Their masks must
        # match independently; a candidate cannot substitute NaN for either Inf
        # sign (or vice versa) merely because both are "non-finite".
        if not (torch.equal(rnan, onan)
                and torch.equal(rpos, opos)
                and torch.equal(rneg, oneg)):
            return -999.0, float("inf"), False
        rfin = ~(rnan | rpos | rneg)
        # Compare magnitudes on the finite subset only.
        if bool(rfin.all()):
            of_c, rf_c = of, rf
        else:
            of_c, rf_c = of[rfin], rf[rfin]
        if rf_c.numel() == 0:
            continue  # entirely non-finite and structurally matched -> agreement
        worst = min(worst, _snr_db(of_c, rf_c))
        diff = (of_c - rf_c).abs().max().item()
        maxd = max(maxd, diff)
        if legacy:
            ok = ok and bool(torch.allclose(of_c, rf_c, atol=atol, rtol=rtol))
            continue
        grids = decl["grid"]
        grid = grids[index] if index < len(grids) else None
        peak = rf_c.abs().max().item()
        tol, step, kind = correctness_tolerance(grid or r.dtype, peak,
                                                decl["code_steps"])
        if kind == "index":
            tol = float(decl["selection_index"])
        else:
            tol += decl["selection_rel"] * peak
        ok = ok and (diff <= tol)
        if stats is not None:
            # Report the MOST BINDING output: its disagreement and the bound it
            # was judged against, in the same unit (one representable step, or
            # one index position).  Taking the two maxima independently would
            # pair a number from one output with a bound from another.
            used, allowed = diff / step, tol / step
            if used - allowed >= stats.get("margin", -float("inf")):
                stats["margin"] = used - allowed
                stats["steps"], stats["limit"] = used, allowed
            stats.setdefault("kinds", set()).add(kind)
    return worst, maxd, ok


def _run_correctness(ref, task_dir, shape) -> int:
    import os
    import torch
    fn = _load_candidate(task_dir, ref.entry_name)
    worst, maxd, ok = 999.0, 0.0, True
    stats: dict = {}
    declared = _tolerance_declarations(ref)
    for s in range(_num_correct_trials()):
        inputs = ref.get_inputs(shape, device="cuda", seed=s)
        r = ref.ref_fn(*_clone_inputs(inputs))
        try:
            o = fn(*_clone_inputs(inputs))
        except Exception as e:  # noqa: BLE001
            print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
            print(f"CANDIDATE_ERROR: {type(e).__name__}: {e}")
            return 0
        torch.cuda.synchronize()
        snr, md, cok = _compare_outputs(o, r, stats=stats, declared=declared)
        worst = min(worst, snr); maxd = max(maxd, md); ok = ok and cok

    # Verification-in-the-loop: enumerated adversarial regimes. Opt-in via
    # KORE_VERIFIED_CORRECTNESS=1 so default gates are unchanged. A kernel correct
    # on random inputs but wrong at e.g. x==0 is rejected here with certainty (no
    # lucky-pass on the enumerated regimes). Covers unary/binary/reduce/fusion/
    # gemm_fusion (generic fills) + any op with an authored adversarial battery.
    if os.environ.get("KORE_VERIFIED_CORRECTNESS") == "1":
        adv_sets = _adversarial_sets(ref, shape)
        for name, adv in (adv_sets or []):
            r = ref.ref_fn(*_clone_inputs(adv))
            try:
                o = fn(*_clone_inputs(adv))
            except Exception as e:  # noqa: BLE001
                print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
                print(f"ADVERSARIAL_ERROR[{name}]: {type(e).__name__}: {e}")
                return 0
            torch.cuda.synchronize()
            snr, md, cok = _compare_outputs(o, r, stats=stats, declared=declared)
            worst = min(worst, snr); maxd = max(maxd, md)
            if not cok:
                ok = False
                print(f"ADVERSARIAL_FAIL[{name}]: SNR {snr:.2f} dB")

    print(f"SNR: {worst:.2f} dB"); print(f"allclose: {ok}"); print(f"max_diff: {maxd:.6f}")
    # The dtype-normalised form of max_diff: how many representable steps of the
    # output format the candidate and the oracle disagree by, beside the bound it
    # was judged against.  ``max_diff`` alone is unreadable across bf16/fp8/int8
    # outputs; this pair is what the gate actually tested, in one unit, including
    # any allowance the op declared (see :func:`correctness_tolerance` and
    # :func:`_tolerance_declarations`).
    print(f"format_steps: {stats.get('steps', 0.0):.4f} "
          f"limit: {stats.get('limit', CORRECTNESS_ULP_STEPS):.4f} "
          f"kinds: {','.join(sorted(stats.get('kinds', ()))) or 'none'}")
    return 0


def _build_bench_fn(ref, task_dir, shape, impl):
    """Build the no-arg callable timed for ``impl`` (candidate|reference).

    Each call to this builder draws an independent seed-0 input set.  Thus the
    candidate and reference see value-identical but storage-disjoint tensors: an
    in-place or malicious candidate cannot poison the reference's timed inputs.
    In-place contract ops (``mutates_input``) additionally get a fresh clone per
    invocation, applied identically to both implementations."""
    inputs = _clone_inputs(ref.get_inputs(shape, device="cuda", seed=0))
    base = ref.baseline_fn if impl in ("reference", "torch") else \
        _load_candidate(task_dir, ref.entry_name)
    if getattr(ref, "mutates_input", False):
        return lambda: base(*_clone_inputs(inputs))
    return lambda: base(*inputs)


def _build_bench_pair(ref, task_dir, shape, _pair_index):
    """Fresh, storage-isolated candidate/reference callables for one pair."""
    if not hasattr(ref, "mutates_input"):
        raise RuntimeError("reference must declare mutates_input for paired timing")
    inputs = ref.get_inputs(shape, device="cuda", seed=0)
    candidate = _load_candidate(task_dir, ref.entry_name)
    return _make_paired_invokers(
        inputs,
        lambda xs: candidate(*xs),
        lambda xs: ref.baseline_fn(*xs),
        bool(ref.mutates_input),
    )


def _run_bench(ref, task_dir, shape, impl, warmup, iters) -> int:
    return _time_fn(_build_bench_fn(ref, task_dir, shape, impl), warmup, iters)


def _time_pair(cand, refr, warmup, iters, candidate_first):
    """Time one candidate/reference pair in the requested order."""
    if candidate_first:
        return (_time_median(cand, warmup, iters),
                _time_median(refr, warmup, iters))
    rm = _time_median(refr, warmup, iters)
    cm = _time_median(cand, warmup, iters)
    return cm, rm


def _run_paired_samples(ref, task_dir, shape, warmup, iters, repeat,
                        build_pair) -> int:
    """Emit raw repeat-level candidate/reference pairs in balanced AB/BA order."""
    import random
    candidate_first = bool(random.getrandbits(1))
    for run in range(max(1, repeat)):
        # Build from a fresh canonical input allocation for EVERY pair.  The
        # builder clones it independently for candidate/reference and applies
        # the task-declared mutation policy to every timed invocation.
        cand, refr = build_pair(ref, task_dir, shape, run)
        w = random.randint(max(4, warmup - 3), warmup + 4)
        it = random.randint(max(8, iters - 5), iters + 6)
        first = candidate_first if run % 2 == 0 else not candidate_first
        cm, rm = _time_pair(cand, refr, w, it, first)
        if not (math.isfinite(cm) and cm > 0.0
                and math.isfinite(rm) and rm > 0.0):
            raise RuntimeError("timing pair produced a non-finite/non-positive sample")
        ratio = rm / cm
        payload = {
            "pair": run,
            "order": "AB" if first else "BA",
            "baseline_kind": getattr(ref, "baseline_kind", None),
            "candidate_ms": cm,
            "baseline_ms": rm,
            "ratio": ratio,
            "log_speedup": math.log(ratio),
        }
        print("KORE_TIMING_PAIR: "
              + json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


def _run_paired_bench_all_shapes(
        ref, task_dir, shape_specs, warmup, iters, repeat,
        build_pair=_build_bench_pair, postcheck=None) -> int:
    """Run the complete publication-grade paired protocol for every shape."""
    postcheck = postcheck or _run_correctness
    for spec in shape_specs:
        shape = ref.parse_shape(spec)
        print(f"SHAPE_BEGIN {spec}")
        _run_paired_samples(
            ref, task_dir, shape, warmup, iters, repeat, build_pair)
        # Every requested shape is re-verified on late invocations of the same
        # cached candidate module.  The environment validates each shape block,
        # so one early failing postcheck cannot be hidden by a later passing one.
        postcheck(ref, task_dir, shape)
    return 0


def _run_bench_both(ref, task_dir, shape, warmup, iters, repeat) -> int:
    return _run_paired_samples(
        ref, task_dir, shape, warmup, iters, repeat, _build_bench_pair)


def _run_bench_all_shapes(ref, task_dir, shape_specs, warmup, iters, repeat) -> int:
    return _run_paired_bench_all_shapes(
        ref, task_dir, shape_specs, warmup, iters, repeat)


def driver_main(ref, task_dir: str, argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="default")
    p.add_argument("--mode", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--bench-both", action="store_true",
                   help="time candidate+reference back-to-back, --repeat runs, in ONE process")
    p.add_argument("--shapes", default=None,
                   help="semicolon-separated shape specs: bench ALL of them in one process")
    p.add_argument("--repeat", type=int, default=1,
                   help="in-process timed runs (used with --bench-both)")
    p.add_argument("--impl", default="candidate", choices=["candidate", "reference", "torch"])
    p.add_argument("--kore-driver-capabilities", action="store_true",
                   help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.kore_driver_capabilities:
        if hasattr(ref, "mutates_input"):
            emit_driver_capabilities(publication_driver_capabilities())
        else:
            emit_driver_capabilities({
                "ineligible_reason": "reference does not declare mutates_input",
            })
        return 0
    shape = ref.parse_shape(a.shape)
    if a.bench_both:
        # fast + contention-fair timing of BOTH impls in one process, then the
        # post-timing anti-hack correctness re-verification on the (cached) candidate.
        if a.shapes:
            specs = [s for s in a.shapes.split(";") if s != ""] or [a.shape]
            rc = _run_bench_all_shapes(ref, task_dir, specs, a.warmup, a.iters, a.repeat)
        else:
            rc = _run_bench_both(ref, task_dir, shape, a.warmup, a.iters, a.repeat)
            _run_correctness(ref, task_dir, shape)
        return rc
    if a.bench_mode:
        rc = _run_bench(ref, task_dir, shape, a.impl, a.warmup, a.iters)
        # post-timing correctness re-verification (anti stateful timing hack): runs
        # on LATE invocations of the cached candidate module.
        if a.impl == "candidate":
            _run_correctness(ref, task_dir, shape)
        return rc
    return _run_correctness(ref, task_dir, shape)
