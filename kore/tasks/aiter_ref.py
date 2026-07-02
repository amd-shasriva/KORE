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
