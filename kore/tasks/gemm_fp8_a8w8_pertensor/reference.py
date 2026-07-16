"""Reference + inputs for fp8 (a8w8) GEMM with STATIC PER-TENSOR scales.

Activation A[M,K] and weight W[N,K] are each quantized to fp8 with a SINGLE scalar scale
(per-tensor amax / FP8_MAX), broadcast into the CK per-row / per-col layout
(x_scale[M,1], w_scale[1,N]). Computes ``Y = (A_deq) @ (W_deq)^T`` in bf16.

This is the per-tensor complement of the per-token draft: the scale-application code path
is the same shape ([M,1] x [1,N]) but every row/col shares a value, so a kernel that
correctly folds per-tensor scales AND one that overfits to distinct-per-row scales are
distinguished across the family.

fp8 is arch-selected via the live ``kore.tasks.aiter_ref`` (OCP e4m3fn on gfx950). Oracle
= exact fp32 matmul of the dequantized operands (scale applied EXACTLY once).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._quant_common import matmul_a8w8_fp32, quant_per_tensor_fp8  # noqa: E402

ENTRY = "gemm"
ATOL = 5e-1
RTOL = 5e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096, "K": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (xq[M,K] fp8, wq[N,K] fp8, x_scale[M,1] fp32, w_scale[1,N] fp32)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    M, N, K = shape["M"], shape["N"], shape["K"]
    a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32)
    w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
    xq, sx = quant_per_tensor_fp8(a)                 # sx scalar
    wq, sw = quant_per_tensor_fp8(w)                 # sw scalar
    x_scale = sx.reshape(1, 1).repeat(M, 1).contiguous()   # [M,1] (broadcast per-tensor)
    w_scale = sw.reshape(1, 1).repeat(1, N).contiguous()   # [1,N]
    return (xq, wq, x_scale, w_scale)


def reference_output(shape, inputs):
    """Exact fp32 dequant-matmul oracle -> bf16 [M,N]."""
    xq, wq, x_scale, w_scale = inputs
    return matmul_a8w8_fp32(xq, wq, x_scale, w_scale)


def candidate_output(fn, shape, inputs):
    xq, wq, x_scale, w_scale = inputs
    return fn(xq, wq, x_scale, w_scale)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER ``gemm_a8w8`` (CK fp8 scaled GEMM)."""
    import torch

    from kore.tasks.aiter_ref import aiter_gemm_a8w8

    xq, wq, x_scale, w_scale = inputs
    return aiter_gemm_a8w8(xq, wq, x_scale, w_scale, out_dtype=torch.bfloat16)
