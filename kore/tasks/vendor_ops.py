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
VENDOR_OPS: tuple[str, ...] = ("rmsnorm", "layernorm", "silu_mul", "gelu_mul",
                               "softmax", "gemm_a8w8", "fused_add_rmsnorm", "rope")

# ops whose vendor BASELINE mutates its inputs in place (so the bench loop must
# feed a fresh clone each timed call — see _genops._run_bench mutates_input path).
VENDOR_MUTATES_INPUT: frozenset[str] = frozenset({"fused_add_rmsnorm"})

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

_SOFTMAX_SHAPES = {  # x[M, N] ; softmax over N (attention logits / vocab rows)
    "minimal": {"M": 64, "N": 1024},
    "primary": {"M": 8192, "N": 8192},
    "validation": [{"M": 4096, "N": 32768}, {"M": 16384, "N": 2048},
                   {"M": 8192, "N": 8191}],   # large vocab, wide batch, non-pow2 tail
}
_FP8_GEMM_SHAPES = {  # XQ[M,K] @ WQ[N,K]^T -> [M,N] bf16 (fp8 a8w8 serving GEMM)
    "minimal": {"M": 128, "N": 128, "K": 256},
    "primary": {"M": 4096, "N": 4096, "K": 4096},
    "validation": [{"M": 8192, "N": 8192, "K": 1024}, {"M": 2048, "N": 14336, "K": 8192},
                   {"M": 4096, "N": 4096, "K": 4095}],  # MLP up-proj, decode, non-pow2 K
}

_ROPE_SHAPES = {  # x[S,B,H,D] NEOX rotary embedding; freqs[S,1,1,D//2] angles
    "minimal": {"S": 128, "B": 1, "H": 8, "D": 64},
    "primary": {"S": 4096, "B": 1, "H": 32, "D": 128},   # Llama-3 8B attention
    "validation": [{"S": 2048, "B": 2, "H": 32, "D": 128}, {"S": 8192, "B": 1, "H": 40, "D": 128},
                   {"S": 4096, "B": 1, "H": 32, "D": 64}],  # batched, GQA-wide, half head-dim
}

VENDOR_SHAPES = {"rmsnorm": _NORM_SHAPES, "layernorm": _NORM_SHAPES,
                 "silu_mul": _GATE_SHAPES, "gelu_mul": _GATE_SHAPES,
                 "softmax": _SOFTMAX_SHAPES, "gemm_a8w8": _FP8_GEMM_SHAPES,
                 "fused_add_rmsnorm": _NORM_SHAPES, "rope": _ROPE_SHAPES}
VENDOR_DTYPES = ("bf16", "fp16")
ROPE_BASE = 10000.0
# Per-op dtype override (defaults to VENDOR_DTYPES). a8w8 GEMM sweeps fp8 + int8
# (both 8-bit-in / bf16-out; the seed is dtype-agnostic: 8-bit -> fp32 accumulate).
VENDOR_OP_DTYPES = {"gemm_a8w8": ("fp8", "int8")}


def vendor_op_dtypes(op: str) -> tuple[str, ...]:
    """The dtype sweep for a vendor op (per-op override or the global default)."""
    return VENDOR_OP_DTYPES.get(op, VENDOR_DTYPES)


EPS = 1e-6
INT8_MAX = 127.0


def _quant_a8w8(x, qdtype: str):
    """Per-tensor symmetric quantization to fp8 (e4m3fnuz) or int8, returning
    ``(q, scale)`` with ``x ~= q.float() * scale`` (scale is a scalar fp32 tensor)."""
    import torch

    from kore.tasks import aiter_ref

    if qdtype == "fp8":
        return aiter_ref.per_tensor_quant_fp8(x)
    if qdtype == "int8":
        amax = x.abs().max().clamp(min=1e-12)
        scale = (amax / INT8_MAX).to(torch.float32)
        q = (x.float() / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        return q, scale.reshape(())
    raise ValueError(f"unknown a8w8 quant dtype {qdtype!r}")


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

    elif op == "softmax":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            return (_randn((M, N), device, seed, scale=2.0),)  # logit-scale rows

        def ref_fn(x):
            return torch.softmax(x.float(), dim=-1).to(x.dtype)

        def baseline_fn(x):
            return aiter_ref.torch_softmax_lastdim(x)  # ROCm MIOpen fused softmax

        arity = 1

    elif op == "gemm_a8w8":
        qdtype = dtype  # "fp8" or "int8"

        def get_inputs(shape, device="cuda", seed=0):
            M, N, K = shape["M"], shape["N"], shape["K"]
            g = torch.Generator(device=device).manual_seed(seed)
            a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32)
            w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
            xq, sx = _quant_a8w8(a, qdtype)
            wq, sw = _quant_a8w8(w, qdtype)
            return (xq, wq, sx.repeat(M, 1).contiguous(), sw.repeat(1, N).contiguous())

        def ref_fn(xq, wq, x_scale, w_scale):
            a_deq = xq.float() * x_scale.float()               # [M,K]
            w_deq = wq.float() * w_scale.float().reshape(-1, 1)  # [N,K]
            return (a_deq @ w_deq.t()).to(torch.bfloat16)

        def baseline_fn(xq, wq, x_scale, w_scale):
            return aiter_ref.aiter_gemm_a8w8(xq, wq, x_scale, w_scale,
                                             out_dtype=torch.bfloat16)

        arity = 4

    elif op == "fused_add_rmsnorm":
        def get_inputs(shape, device="cuda", seed=0):
            M, N = shape["M"], shape["N"]
            x = _randn((M, N), device, seed)
            residual = _randn((M, N), device, seed + 1)
            w = (torch.randn((N,), generator=torch.Generator(device=device).manual_seed(seed + 2),
                             device=device, dtype=torch.float32) * 0.1 + 1.0).to(tdt)
            return (x, residual, w)

        def ref_fn(x, residual, w):
            added = x.float() + residual.float()
            var = added.pow(2).mean(dim=-1, keepdim=True)
            y = added * torch.rsqrt(var + EPS) * w.float()
            return y.to(x.dtype), added.to(x.dtype)   # (normed, new_residual)

        def baseline_fn(x, residual, w):
            return aiter_ref.aiter_fused_add_rms_norm(x, residual, w, EPS)

        arity = 3

    elif op == "rope":
        def get_inputs(shape, device="cuda", seed=0):
            S, B, H, D = shape["S"], shape["B"], shape["H"], shape["D"]
            g = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn((S, B, H, D), generator=g, device=device, dtype=torch.float32).to(tdt)
            inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, D, 2, device=device,
                                                          dtype=torch.float32) / D))
            t = torch.arange(S, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq).view(S, 1, 1, D // 2).contiguous()
            return (x, freqs)

        def ref_fn(x, freqs):
            xf = x.float()
            D = xf.shape[-1]
            cos = torch.cos(freqs).float()
            sin = torch.sin(freqs).float()
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
            x1, x2 = xf[..., : D // 2], xf[..., D // 2:]
            rot = torch.cat([-x2, x1], dim=-1)
            return (xf * cos + rot * sin).to(x.dtype)

        def baseline_fn(x, freqs):
            return aiter_ref.aiter_rope_neox(x, freqs)

        arity = 2
    else:
        raise ValueError(f"unknown vendor op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op, "dtype_name": dtype,
          "family": f"vendor_{op}", "mutates_input": op in VENDOR_MUTATES_INPUT}
    ns["adversarial_inputs"] = _make_adversarial_inputs(op, get_inputs, dtype)
    ns[f"{op}_ref"] = ref_fn
    return ns


def _make_adversarial_inputs(op: str, get_inputs, dtype: str = "bf16"):
    """Op-class-aware adversarial input battery (verification-in-the-loop).

    Plain-float vendor ops reuse the generic float fills (zeros/ones/large/small/
    sign-alt), which are exhaustive of the qualitative regimes. Quantized GEMM ops
    must build the battery in FLOAT then quantize, so the fp8 codes + scales stay a
    valid dequantizable pair (filling the fp8 tensors directly would be nonsense)."""
    import torch

    from kore.tasks._genops import _adversarial_fills

    def _generic(shape, device="cuda", seed=0):
        return list(_adversarial_fills(get_inputs(shape, device=device, seed=seed)))

    if op != "gemm_a8w8":
        return _generic

    def _quant_gemm(shape, device="cuda", seed=0):
        M, N, K = shape["M"], shape["N"], shape["K"]
        regimes = {
            "zeros": (torch.zeros((M, K)), torch.zeros((N, K))),
            "ones": (torch.ones((M, K)), torch.ones((N, K))),
            "large": (torch.full((M, K), 100.0), torch.full((N, K), 100.0)),
            "small": (torch.full((M, K), 1e-2), torch.full((N, K), 1e-2)),
            "mixed_sign": (torch.ones((M, K)).cumsum(1) % 2 * 2 - 1,
                           torch.ones((N, K)).cumsum(1) % 2 * 2 - 1),
        }
        out = []
        for name, (a, w) in regimes.items():
            a = a.to(device=device, dtype=torch.float32)
            w = w.to(device=device, dtype=torch.float32)
            xq, sx = _quant_a8w8(a, dtype)   # dtype: "fp8" | "int8"
            wq, sw = _quant_a8w8(w, dtype)
            out.append((name, (xq, wq, sx.repeat(M, 1).contiguous(),
                               sw.repeat(1, N).contiguous())))
        return out

    return _quant_gemm


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


_SOFTMAX_SEED = '''"""GENERATED vendor-baselined row-softmax seed ({dtype}) vs torch/MIOpen softmax.
Online (streaming) softmax: pass 1 running max+sum, pass 2 normalize+store, so any
row width N fits regardless of BLOCK_N. Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, y_ptr, sm, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk_max = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk_max)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + base + offs, (tl.exp(x - m) / s).to({tldt}), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    _softmax_kernel[(M,)](x, y, x.stride(0), N, BLOCK_N=1024, num_warps=8)
    return y
'''

_FP8_GEMM_SEED = '''"""GENERATED vendor-baselined a8w8 GEMM seed ({dtype}) vs aiter.gemm_a8w8.
Y = (XQ*x_scale) @ (WQ*w_scale)^T, bf16 out. 8-bit (fp8/int8) operands up-converted
in-register to fp32, fp32 accumulate, scales on the accumulator. Dtype-agnostic:
the load->fp32 path handles both fp8 e4m3fnuz and int8. Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _gemm_a8w8_kernel(a_ptr, b_ptr, c_ptr, xs_ptr, ws_ptr, M, N, K,
                      stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      GROUP_M: tl.constexpr):
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
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=kmask, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    xs = tl.load(xs_ptr + offs_cm, mask=offs_cm < M, other=0.0).to(tl.float32)
    ws = tl.load(ws_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    acc = acc * xs[:, None] * ws[None, :]
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def gemm_a8w8(xq: torch.Tensor, wq: torch.Tensor,
              x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    M, K = xq.shape
    N, _ = wq.shape
    c = torch.empty((M, N), device=xq.device, dtype=torch.bfloat16)
    xs = x_scale.reshape(-1).contiguous()
    ws = w_scale.reshape(-1).contiguous()
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _gemm_a8w8_kernel[grid](xq, wq, c, xs, ws, M, N, K,
                            xq.stride(0), xq.stride(1), wq.stride(0), wq.stride(1),
                            c.stride(0), c.stride(1),
                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                            GROUP_M=GROUP_M, num_warps=4, num_stages=2)
    return c
'''


_FUSED_ADD_RMSNORM_SEED = '''"""GENERATED vendor-baselined fused add-RMSNorm seed ({dtype}) vs aiter.fused_add_rms_norm_cu.
added = x + residual (the new residual); y = RMSNorm(added) * weight. One program
per row, fp32 accumulate, {tldt} store. Returns (y, added) — the candidate writes
NEW tensors (the vendor baseline is in-place). Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, y_ptr, added_ptr, sm, N, eps,
                              BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + r
    tl.store(added_ptr + base + offs, added.to({tldt}), mask=mask)
    var = tl.sum(added * added, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (added * rstd * w).to({tldt}), mask=mask)


def fused_add_rmsnorm(x, residual, weight, eps: float = 1e-6):
    M, N = x.shape
    y = torch.empty_like(x)
    added = torch.empty_like(x)
    _fused_add_rmsnorm_kernel[(M,)](x, residual, weight, y, added, x.stride(0), N, eps,
                                    BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, added
'''


_ROPE_SEED = '''"""GENERATED vendor-baselined NEOX RoPE seed ({dtype}) vs aiter.rope_fwd.
x[S,B,H,D], freqs[S,1,1,D//2] angles. One program per (s,b,h) row; half-width
rotate-NEOX identity (o1=x1*cos-x2*sin, o2=x2*cos+x1*sin), fp32 math, {tldt} store.
Regenerate via generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _rope_kernel(x_ptr, f_ptr, y_ptr, B, H, D,
                 sxs, sxb, sxh, sxd, sfs, HALF: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    tmp = pid // H
    b = tmp % B
    s = tmp // B
    base = s * sxs + b * sxb + h * sxh
    offs = tl.arange(0, HALF)
    x1 = tl.load(x_ptr + base + offs * sxd).to(tl.float32)
    x2 = tl.load(x_ptr + base + (offs + HALF) * sxd).to(tl.float32)
    theta = tl.load(f_ptr + s * sfs + offs).to(tl.float32)
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    tl.store(y_ptr + base + offs * sxd, (x1 * cos - x2 * sin).to({tldt}))
    tl.store(y_ptr + base + (offs + HALF) * sxd, (x2 * cos + x1 * sin).to({tldt}))


def rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    S, B, H, D = x.shape
    y = torch.empty_like(x)
    f = freqs.reshape(S, D // 2)
    _rope_kernel[(S * B * H,)](x, f, y, B, H, D,
                               x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                               f.stride(0), HALF=D // 2, num_warps=4)
    return y
'''


def vendor_seed_source(op: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]
    if op == "softmax":
        return _SOFTMAX_SEED.format(dtype=dtype, tldt=tldt)
    if op == "gemm_a8w8":
        return _FP8_GEMM_SEED.format(dtype=dtype)
    if op == "fused_add_rmsnorm":
        return _FUSED_ADD_RMSNORM_SEED.format(dtype=dtype, tldt=tldt)
    if op == "rope":
        return _ROPE_SEED.format(dtype=dtype, tldt=tldt)
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
