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
}


def _torch_dtype(name: str):
    import torch
    return getattr(torch, DTYPES[name][0])


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


# op registry: name -> (family, spec)
def _registry() -> dict[str, tuple[str, object]]:
    reg: dict[str, tuple[str, object]] = {}
    for n, s in _unary_specs().items():
        reg[n] = ("unary", s)
    for n, s in _binary_specs().items():
        reg[n] = ("binary", s)
    for n, s in _reduce_specs().items():
        reg[n] = ("reduce", s)
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


def seed_source(op: str, family: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]
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


def _run_correctness(ref, task_dir, shape) -> int:
    import torch
    fn = _load_candidate(task_dir, ref.entry_name)
    worst, maxd, ok = 999.0, 0.0, True
    for s in range(_num_correct_trials()):
        inputs = ref.get_inputs(shape, device="cuda", seed=s)
        r = ref.ref_fn(*inputs)
        try:
            o = fn(*inputs)
        except Exception as e:  # noqa: BLE001
            print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
            print(f"CANDIDATE_ERROR: {type(e).__name__}: {e}")
            return 0
        torch.cuda.synchronize()
        worst = min(worst, _snr_db(o, r))
        maxd = max(maxd, (o.float() - r.float()).abs().max().item())
        ok = ok and torch.allclose(o.float(), r.float(), atol=1e-2, rtol=1e-2)
    print(f"SNR: {worst:.2f} dB"); print(f"allclose: {ok}"); print(f"max_diff: {maxd:.6f}")
    return 0


def _run_bench(ref, task_dir, shape, impl, warmup, iters) -> int:
    inputs = ref.get_inputs(shape, device="cuda", seed=0)
    if impl in ("reference", "torch"):
        fn = lambda: ref.baseline_fn(*inputs)
    else:
        cand = _load_candidate(task_dir, ref.entry_name)
        fn = lambda: cand(*inputs)
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
