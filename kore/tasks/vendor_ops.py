"""Vendor-baselined task authoring engine: hard ops graded against REAL AITER kernels.

Unlike the generated elementwise/fusion tasks (torch-eager baseline), these tasks
grade the policy against the ACTUAL production vendor kernel AMD's serving stack
calls — ``aiter.rms_norm`` / ``aiter.layer_norm`` / ``aiter.silu_and_mul`` /
``aiter.gelu_tanh_and_mul`` — the honest "beat the vendor library" bar. Each task
is authored semi-automatically from a per-op template + a model-shape/dtype sweep,
so the vendor-baselined suite scales without hundreds of hand-written files.

Contract (matches _genops so the generic driver works): make_vendor_reference()
returns the reference.py namespace (parse_shape/get_inputs/ref_fn oracle/baseline_fn
AITER/arity/entry_name); vendor_seed_source() returns a REAL Triton starter kernel;
the driver is the shared kore.tasks._genops.driver_main.

torch/aiter imported lazily (registry discovery never needs a GPU/aiter).
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# op -> family metadata; each op has a bespoke oracle/baseline/seed (below).
VENDOR_OPS: tuple[str, ...] = ("rmsnorm", "layernorm", "silu_mul", "gelu_mul")

# Real production shapes (hidden dims / gated-MLP widths) per op class, per the
# KORE-Bench blueprint (Llama-3 / Qwen3 / Mixtral / DeepSeek-V3).
_NORM_SHAPES = {  # x[M, N] ; N = hidden
    "minimal": {"M": 64, "N": 2048},
    "primary": {"M": 4096, "N": 8192},
    "validation": [{"M": 8192, "N": 4096}, {"M": 2048, "N": 7168},
                   {"M": 4096, "N": 8191}],   # DeepSeek hidden + non-pow2 tail
}
_GATE_SHAPES = {  # x[M, 2*inter] ; N = 2*inter (input width)
    "minimal": {"M": 64, "N": 1024},
    "primary": {"M": 4096, "N": 28672},        # Llama-3 8B MLP: 2*14336
    "validation": [{"M": 8192, "N": 22016}, {"M": 2048, "N": 8192},
                   {"M": 4096, "N": 28670}],   # 2*11008, small, non-pow2 tail
}

VENDOR_SHAPES = {"rmsnorm": _NORM_SHAPES, "layernorm": _NORM_SHAPES,
                 "silu_mul": _GATE_SHAPES, "gelu_mul": _GATE_SHAPES}
VENDOR_DTYPES = ("bf16", "fp16")
EPS = 1e-6


# --------------------------------------------------------------------------- #
# reference.py namespace (torch fp32 oracle + AITER production baseline)
# --------------------------------------------------------------------------- #
def make_vendor_reference(op: str, dtype: str) -> dict:
    import torch
    import torch.nn.functional as F
    from kore.tasks import aiter_ref

    tdt = getattr(torch, DTYPES[dtype][0])

    def _randn(shape, device, seed, scale=1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device, dtype=torch.float32) * scale).to(tdt)

    if op == "rmsnorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            x = _randn((M, N), device, seed)
            w = (torch.randn((N,), generator=torch.Generator(device=device).manual_seed(seed + 1),
                             device=device, dtype=torch.float32) * 0.1 + 1.0).to(tdt)
            return (x, w)

        def ref_fn(x, w):
            xf = x.float()
            y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
            return y.to(x.dtype)

        def baseline_fn(x, w):
            return aiter_ref.aiter_rms_norm(x, w, EPS)

        arity = 2

    elif op == "layernorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            x = _randn((M, N), device, seed)
            g = torch.Generator(device=device).manual_seed(seed + 1)
            w = (torch.randn((N,), generator=g, device=device, dtype=torch.float32) * 0.1 + 1.0).to(tdt)
            b = _randn((N,), device, seed + 2, scale=0.1)
            return (x, w, b)

        def ref_fn(x, w, b):
            return F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), EPS).to(x.dtype)

        def baseline_fn(x, w, b):
            return aiter_ref.aiter_layer_norm(x, w, b, EPS)

        arity = 3

    elif op in ("silu_mul", "gelu_mul"):
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]  # N = 2*inter
            return (_randn((M, N), device, seed),)

        if op == "silu_mul":
            def ref_fn(x):
                inter = x.shape[-1] // 2
                g, u = x[:, :inter].float(), x[:, inter:].float()
                return (F.silu(g) * u).to(x.dtype)

            def baseline_fn(x):
                return aiter_ref.aiter_silu_and_mul(x)
        else:
            def ref_fn(x):
                inter = x.shape[-1] // 2
                g, u = x[:, :inter].float(), x[:, inter:].float()
                return (F.gelu(g, approximate="tanh") * u).to(x.dtype)

            def baseline_fn(x):
                return aiter_ref.aiter_gelu_tanh_and_mul(x)

        arity = 1
    else:
        raise ValueError(f"unknown vendor op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op, "dtype_name": dtype}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Real Triton starter seeds (the policy optimizes these against the AITER bar)
# --------------------------------------------------------------------------- #
_RMSNORM_SEED = '''"""GENERATED vendor-baselined RMSNorm seed ({dtype}) vs aiter.rms_norm.
One program/row: fp32 mean-square, rsqrt, weight, {tldt} store. Regenerate via
kore/tasks/generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (x * rstd * w).to({tldt}), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _rmsnorm_kernel[(M,)](x, weight, y, x.stride(0), N, eps,
                          BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''

_LAYERNORM_SEED = '''"""GENERATED vendor-baselined LayerNorm seed ({dtype}) vs aiter.layer_norm.
One program/row: fp32 mean+var, affine, {tldt} store. Regenerate via
kore/tasks/generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, sm, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sm + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * sm + offs, (xc * rstd * w + b).to({tldt}), mask=mask)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _layernorm_kernel[(M,)](x, weight, bias, y, x.stride(0), N, eps,
                            BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y
'''

_GATE_SEED = '''"""GENERATED vendor-baselined {op} seed ({dtype}) vs aiter {op}.
Gated MLP activation x[M,2*inter] -> {op_desc}(gate)*up [M,inter], {tldt} store.
Regenerate via kore/tasks/generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, y_ptr, sxm, sym, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    gate = tl.load(x_ptr + row * sxm + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * sxm + N + offs, mask=mask, other=0.0).to(tl.float32)
    act = {act_expr}
    tl.store(y_ptr + row * sym + offs, (act * up).to({tldt}), mask=mask)


def {op}(x: torch.Tensor) -> torch.Tensor:
    M, two_n = x.shape
    N = two_n // 2
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _{op}_kernel[grid](x, y, x.stride(0), y.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    return y
'''


def vendor_seed_source(op: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]
    if op == "rmsnorm":
        return _RMSNORM_SEED.format(dtype=dtype, tldt=tldt)
    if op == "layernorm":
        return _LAYERNORM_SEED.format(dtype=dtype, tldt=tldt)
    if op == "silu_mul":
        return _GATE_SEED.format(op="silu_mul", op_desc="silu", dtype=dtype, tldt=tldt,
                                 act_expr="gate * tl.sigmoid(gate)")
    if op == "gelu_mul":
        gelu = ("0.5 * gate * (1.0 + (2.0 * tl.sigmoid(2.0 * (0.7978845608028654 * "
                "(gate + 0.044715 * gate * gate * gate))) - 1.0))")
        return _GATE_SEED.format(op="gelu_mul", op_desc="gelu_tanh", dtype=dtype, tldt=tldt,
                                 act_expr=gelu)
    raise ValueError(op)
