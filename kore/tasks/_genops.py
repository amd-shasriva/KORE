"""Operator-generation engine for scaling the KORE task suite (15 -> 100+).

A KORE task = task.yaml + reference.py (torch oracle + inputs) + seed_triton.py
(a compiling starter kernel) + driver.py (the verifier contract). Hand-writing
120 of those is infeasible *and* error-prone, so this engine generates them from a
declarative op spec:

  * ``make_reference(op, family, dtype)`` -> the reference.py namespace (parse_shape,
    get_inputs, ref_fn oracle, baseline_fn production path, arity, entry_name).
  * ``seed_source(op, family, dtype)`` -> a REAL, compiling Triton seed kernel
    (the policy's starting point to optimize; the pointwise/reduce math is inlined,
    not a shim, so the policy has genuine code to edit).
  * ``driver_main(ref, task_dir)`` -> the generic KernelForge driver: multi-trial
    correctness + cold-cache timing + the POST-TIMING anti-hack re-verification
    (candidate module cached so a stateful invocation-count kernel is caught), in
    ONE place (no 15x driver duplication for generated ops).

Generated ops use the torch FRAMEWORK op as the production baseline (the honest
ROCm serving path for elementwise/reduction ops, exactly like the shipped
gelu_tanh/softmax tasks). Every generated op is verifiable by construction: the
Triton seed computes the same math (fp32) as the torch oracle.

All pure/CPU-importable (torch/triton imported lazily inside the GPU paths) so
registry discovery never needs a GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
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
    # fp8 e4m3fnuz (gfx942/CDNA3): used by quantized GEMM vendor ops. The oracle
    # dequantizes the SAME fp8 operands, so the gate measures the kernel's fp32
    # accumulation fidelity (bf16 output) — a ~25 dB bar, not the quant error.
    "fp8": ("float8_e4m3fnuz", "tl.float8e4b8", 25.0),
    # int8 symmetric (W8A8): int8-in / bf16-out quantized GEMM (per-row/col scales).
    "int8": ("int8", "tl.int8", 25.0),
}


def _torch_dtype(name: str):
    import torch
    return getattr(torch, DTYPES[name][0])


# torch.compile'd baseline cache (one per fusion/gemm_fusion op+dtype).
_FUSED_BASELINE_CACHE: dict = {}


def _compile_baseline_enabled() -> bool:
    """Grade fusion/gemm_fusion against the COMPILER-FUSED baseline (honest bar)
    when KORE_COMPILE_BASELINE is truthy — closes the 'beat unfused eager' speedup
    inflation. Off by default so unit tests / CPU dry-runs stay eager + cheap."""
    return os.environ.get("KORE_COMPILE_BASELINE", "").strip().lower() in (
        "1", "true", "yes", "on")


def _fused_baseline(fn, key: str):
    """Return ``torch.compile(fn)`` (fused, cached) when enabled, else ``fn``.

    Compilation is the honest multi-kernel-fusion bar (the compiler fuses the
    elementwise chain / GEMM epilogue), so the candidate must beat the FUSED kernel
    rather than unfused eager. Any compile failure degrades to eager (never fatal)."""
    if not _compile_baseline_enabled():
        return fn
    if key not in _FUSED_BASELINE_CACHE:
        try:
            import torch
            _FUSED_BASELINE_CACHE[key] = torch.compile(fn)
        except Exception:  # noqa: BLE001 — torch.compile unavailable/unsupported
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
    torch-eager — unlike a single elementwise op where torch is already near
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
    }[name]


# Triton fp32 epilogue activation on `acc` (libdevice-free), per act code.
_TL_ACT = {
    "none": "",
    "relu": "    acc = tl.maximum(acc, 0.0)\n",
    "gelu": ("    _gi = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)\n"
             "    acc = 0.5 * acc * (1.0 + (2.0 * tl.sigmoid(2.0 * _gi) - 1.0))\n"),
    "silu": "    acc = acc * tl.sigmoid(acc)\n",
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
    elif family == "fusion":
        s: FusionSpec = spec
        gen = _mk("signed")

        def get_inputs(shape, device="cuda", seed=0):
            return tuple(gen(shape, device, seed + i) for i in range(s.arity))

        def ref_fn(*xs):
            return s.torch_fn(*[t.float() for t in xs]).to(xs[0].dtype)

        def baseline_fn(*xs):
            # Honest fused bar: torch.compile FUSES the elementwise chain into one
            # kernel (when KORE_COMPILE_BASELINE=1), so the candidate must beat the
            # COMPILER, not unfused eager (which would inflate the speedup). Falls
            # back to eager multi-kernel when compile is off/unavailable.
            return _fused_baseline(s.torch_fn, f"fusion:{op}:{dtype}")(*xs)

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
            # Honest fused bar: torch.compile fuses the bias+activation EPILOGUE into
            # the hipBLASLt GEMM (when KORE_COMPILE_BASELINE=1), so the candidate must
            # beat the compiler-fused epilogue-GEMM, not the unfused matmul+bias+act
            # chain (which would inflate the speedup). Falls back to eager otherwise.
            return _fused_baseline(_eager_gemm_epilogue, f"gemm_fusion:{op}:{dtype}")(*xs)

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
    }
    ns[f"{op}_ref"] = ref_fn   # conventional alias
    return ns


# --------------------------------------------------------------------------- #
# Triton seed source (a REAL compiling starter kernel)
# --------------------------------------------------------------------------- #
_UNARY_TMPL = '''"""GENERATED seed Triton kernel for the {op} ({dtype}) activation.

Elementwise {op}, 2D-tiled, fp32 math, {tldt} store. A correct-but-naive starting
point the KORE policy learns to optimize against the framework production baseline.
Regenerate via kore/tasks/generate_ops.py — do not hand-edit.
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
kore/tasks/generate_ops.py — do not hand-edit.
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
kore/tasks/generate_ops.py — do not hand-edit.
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
Regenerate via kore/tasks/generate_ops.py — do not hand-edit.
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
Regenerate via kore/tasks/generate_ops.py — do not hand-edit.
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
Grouped tiling + K-mask (ROCm/gfx942-safe, libdevice-free act). Regenerate via
kore/tasks/generate_ops.py — do not hand-edit.
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
def _snr_db(out, ref_out) -> float:
    o, r = out.float(), ref_out.float()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


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
        "sign_alt": lambda t: (torch.ones_like(t.reshape(-1)).cumsum(0) % 2 * 2 - 1)
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


def _clone_inputs(inputs):
    """Clone tensor inputs so an IN-PLACE candidate/oracle (e.g. fused_add_rmsnorm)
    can't corrupt the shared inputs between the reference and candidate calls."""
    import torch
    return tuple(t.clone() if torch.is_tensor(t) else t for t in inputs)


def _compare_outputs(out, ref_out):
    """SNR/max_diff/allclose over single-tensor OR multi-output (tuple) results.

    Returns ``(worst_snr_db, max_abs_diff, allclose_all)`` — the worst SNR and the
    logical-AND of allclose across every output tensor."""
    import torch
    outs, refs = _as_tuple(out), _as_tuple(ref_out)
    worst, maxd, ok = 999.0, 0.0, True
    for o, r in zip(outs, refs):
        worst = min(worst, _snr_db(o, r))
        maxd = max(maxd, (o.float() - r.float()).abs().max().item())
        ok = ok and bool(torch.allclose(o.float(), r.float(), atol=1e-2, rtol=1e-2))
    return worst, maxd, ok


def _run_correctness(ref, task_dir, shape) -> int:
    import os
    import torch
    fn = _load_candidate(task_dir, ref.entry_name)
    worst, maxd, ok = 999.0, 0.0, True
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
        snr, md, cok = _compare_outputs(o, r)
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
            snr, md, cok = _compare_outputs(o, r)
            worst = min(worst, snr); maxd = max(maxd, md)
            if not cok:
                ok = False
                print(f"ADVERSARIAL_FAIL[{name}]: SNR {snr:.2f} dB")

    print(f"SNR: {worst:.2f} dB"); print(f"allclose: {ok}"); print(f"max_diff: {maxd:.6f}")
    return 0


def _run_bench(ref, task_dir, shape, impl, warmup, iters) -> int:
    inputs = ref.get_inputs(shape, device="cuda", seed=0)
    base = ref.baseline_fn if impl in ("reference", "torch") else \
        _load_candidate(task_dir, ref.entry_name)
    if getattr(ref, "mutates_input", False):
        # In-place ops (e.g. fused_add_rmsnorm) mutate their inputs, so a repeated
        # timing loop must feed a fresh clone each call. The clone cost is applied
        # IDENTICALLY to candidate + baseline, so the speedup ratio stays fair.
        fn = lambda: base(*_clone_inputs(inputs))
    else:
        fn = lambda: base(*inputs)
    return _time_fn(fn, warmup, iters)


def driver_main(ref, task_dir: str, argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="default")
    p.add_argument("--mode", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--impl", default="candidate", choices=["candidate", "reference", "torch"])
    a = p.parse_args(argv)
    shape = ref.parse_shape(a.shape)
    if a.bench_mode:
        rc = _run_bench(ref, task_dir, shape, a.impl, a.warmup, a.iters)
        # post-timing correctness re-verification (anti stateful timing hack): runs
        # on LATE invocations of the cached candidate module.
        if a.impl == "candidate":
            _run_correctness(ref, task_dir, shape)
        return rc
    return _run_correctness(ref, task_dir, shape)
