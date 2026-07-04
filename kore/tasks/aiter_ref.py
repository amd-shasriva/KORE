"""Shared AITER baseline helpers for KORE tasks.

The whole point of the AMD-correct tasks: the *performance baseline* is the
kernel the production serving stack actually calls (AITER), not unfused torch.
This module centralizes the thin AITER wrappers + the fp8 quantization helpers
so each task's driver measures the honest bar.

Import-safe: AITER (and torch) are imported lazily inside the wrappers so that
`kore tasks` / registry discovery never require a GPU or the aiter runtime.

gfx942 (MI325X) fp8 note: AMD CDNA3 uses the **FNUZ** fp8 encoding, so the
correct e4m3 dtype is ``torch.float8_e4m3fnuz`` (NOT the OCP ``e4m3fn``). Using
e4m3fn silently changes the numeric range/bias and mismatches AITER/hipBLASLt.
"""

from __future__ import annotations

import torch

# gfx942 / CDNA3 fp8 e4m3 is the FNUZ variant.
FP8_DTYPE = torch.float8_e4m3fnuz
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 240.0


# --- RMSNorm family -------------------------------------------------------
def aiter_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """AITER CK RMSNorm: ``aiter.rms_norm(input, weight, epsilon)`` -> Tensor."""
    import aiter

    return aiter.rms_norm(x, weight, eps)


def aiter_fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
):
    """AITER fused add + RMSNorm, matching in-place CU semantics.

    ``aiter.fused_add_rms_norm_cu(input, residual_in, weight, epsilon)`` mutates
    both tensors in place (returns None):
      * ``input``        <- RMSNorm(input + residual_in) * weight
      * ``residual_in``  <- input + residual_in   (the new residual)

    We operate on the passed tensors directly (caller owns cloning for a fair
    benchmark) and return ``(normed, new_residual)`` = ``(x, residual)``.
    """
    import aiter

    aiter.fused_add_rms_norm_cu(x, residual, weight, eps)
    return x, residual


# --- Gated MLP activation -------------------------------------------------
def aiter_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """AITER ``silu_and_mul(out, input)`` (in-place into out).

    Input is (M, 2*inter); returns SiLU(x[:, :inter]) * x[:, inter:] as (M, inter).
    """
    import aiter

    inter = x.shape[-1] // 2
    out = torch.empty((*x.shape[:-1], inter), dtype=x.dtype, device=x.device)
    aiter.silu_and_mul(out, x)
    return out


def aiter_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    """AITER ``gelu_tanh_and_mul(out, input)`` (in-place into out).

    Input is (M, 2*inter); returns GELU-tanh(x[:, :inter]) * x[:, inter:] as (M, inter).
    The tanh-approx GELU is the LLM-standard gated activation (GeGLU).
    """
    import aiter

    inter = x.shape[-1] // 2
    out = torch.empty((*x.shape[:-1], inter), dtype=x.dtype, device=x.device)
    aiter.gelu_tanh_and_mul(out, x)
    return out


def aiter_layer_norm_noaffine_ok(x, weight, bias, eps: float) -> torch.Tensor:
    """AITER LayerNorm wrapper (thin alias to aiter.layer_norm) for the vendor tasks."""
    import aiter

    return aiter.layer_norm(x, weight, bias, eps)


# --- fp8 GEMM -------------------------------------------------------------
def per_tensor_quant_fp8(x: torch.Tensor):
    """Per-tensor symmetric quantization to fp8 e4m3fnuz.

    Returns ``(xq, scale)`` where ``scale`` is a scalar fp32 tensor and
    ``x ≈ xq.float() * scale``.
    """
    amax = x.abs().max().clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    xq = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq, scale.reshape(())


def aiter_gemm_a8w8(
    xq: torch.Tensor,
    wq: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """AITER fp8 GEMM: ``aiter.gemm_a8w8(XQ, WQ, x_scale, w_scale, dtype=...)``.

    Layout (CK): XQ [M, K], WQ [N, K] (computes ``X @ W^T``), x_scale [M, 1],
    w_scale [1, N], both fp32. Returns [M, N] in ``out_dtype``.
    """
    import aiter

    return aiter.gemm_a8w8(xq, wq, x_scale, w_scale, dtype=out_dtype)


# --- Batched / grouped GEMM ----------------------------------------------
def aiter_batched_gemm_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """AITER batched bf16 GEMM: ``aiter.batched_gemm_bf16(A, B, out)`` (in-place).

    Layout (CK): A [B, M, K], B [B, N, K] (so it computes ``A @ B^T`` per batch),
    out [B, M, N] bf16, fp32 accumulation. This is the vendor batched GEMM the
    serving stack calls for batched attention/MoE projections.
    """
    import aiter

    B, M, _ = a.shape
    N = b.shape[1]
    out = torch.empty((B, M, N), dtype=torch.bfloat16, device=a.device)
    aiter.batched_gemm_bf16(a, b, out)
    return out


# --- Dense bf16 GEMM ------------------------------------------------------
def hipblaslt_gemm_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Production dense bf16 GEMM baseline: ``torch.matmul(A, B)``.

    On ROCm, ``torch.matmul`` for bf16 dense matmul dispatches straight to
    **hipBLASLt** (the vendor tuned GEMM library that the serving stack uses),
    so this *is* the real production bar — not an unfused torch loop. A[M,K] @
    B[K,N] -> [M,N], fp32 accumulate, bf16 output.
    """
    return torch.matmul(a, b)


# --- LayerNorm ------------------------------------------------------------
def aiter_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """AITER CK LayerNorm: ``aiter.layer_norm(input, weight, bias, epsilon)``.

    2D row LayerNorm over the last dim (mean + variance subtraction), affine
    with weight+bias. Returns a tensor of the same shape/dtype.
    """
    import aiter

    return aiter.layer_norm(x, weight, bias, eps)


# --- Softmax --------------------------------------------------------------
def torch_softmax_lastdim(x: torch.Tensor) -> torch.Tensor:
    """Production row-softmax baseline: ``torch.softmax(x, dim=-1)``.

    AITER exposes no standalone dense row-softmax (only ``topk_softmax`` for MoE
    routing), so the honest production op is the framework path: on ROCm
    ``torch.softmax`` lowers to a fused MIOpen/rocm softmax kernel. Documented as
    the framework production baseline per the KORE ABI.
    """
    return torch.softmax(x, dim=-1)


# --- GELU (tanh approximation) -------------------------------------------
def torch_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """Production tanh-approx GELU baseline: ``F.gelu(x, approximate='tanh')``.

    AITER only ships *gated* GELU (``gelu_and_mul`` / ``gelu_tanh_and_mul``), not
    a standalone elementwise GELU activation, so the honest production op is the
    framework path: ``torch.nn.functional.gelu`` lowers to a fused rocm
    elementwise kernel. Documented as the framework production baseline.
    """
    import torch.nn.functional as F

    return F.gelu(x, approximate="tanh")


# --- RoPE (rotary position embedding) ------------------------------------
def aiter_rope_neox(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """AITER RoPE: ``aiter.rope_fwd`` (NEOX-style, full head-dim rotation).

    ``x`` is (S, B, H, D); ``freqs`` is (S, 1, 1, D//2) of rotation *angles*
    (the op computes cos/sin internally). Call convention:
    ``rope_fwd(input, freqs, rotate_style=0 (NEOX), reuse_freqs_front_part=True,
    nope_first=False)`` -> rotated tensor (S, B, H, D). This is the vendor HIP
    rope kernel used by the serving stack.
    """
    import aiter

    return aiter.rope_fwd(x, freqs, 0, True, False, False)


# --- Dynamic per-token fp8 quantization ----------------------------------
def aiter_dynamic_per_token_quant(x: torch.Tensor):
    """AITER dynamic per-token (rowwise) fp8 quant to e4m3fnuz.

    ``aiter.dynamic_per_token_scaled_quant(out, input, scales)`` writes the fp8
    codes into ``out`` [M,N] and the per-row fp32 scales into ``scales`` [M,1]
    in place (returns None). ``x ≈ out.float() * scales``. This is the vendor
    quant kernel the serving stack calls for W8A8 / fp8 activation quant.
    """
    import aiter

    M, N = x.shape
    out = torch.empty((M, N), dtype=FP8_DTYPE, device=x.device)
    scales = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    aiter.dynamic_per_token_scaled_quant(out, x, scales)
    return out, scales
