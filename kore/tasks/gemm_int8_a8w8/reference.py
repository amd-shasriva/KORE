"""Reference + inputs for int8 W8A8 GEMM (per-token activation, per-channel weight).

Activation A[M,K] is quantized to int8 with a per-ROW (per-token) scale ``x_scale[M,1]``
and weight W[N,K] with a per-OUTPUT-CHANNEL scale ``w_scale[1,N]`` (symmetric,
round-to-nearest, +/-127). Computes ``Y = (A_deq) @ (W_deq)^T`` in bf16.

Layout matches AITER ``gemm_a8w8`` (CK, int8 path): XQ[M,K] int8, WQ[N,K] int8,
x_scale[M,1], w_scale[1,N] fp32.

Correctness oracle: exact fp32 matmul of the dequantized int8 operands, each scale
applied EXACTLY once. int8 is near-exact, so the SNR gate mainly rejects a wrong-scale or
bf16-accumulation kernel (the classic W8A8 bugs).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._quant_common import matmul_a8w8_fp32, quant_rowwise_int8  # noqa: E402

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
    """Returns (xq[M,K] int8, wq[N,K] int8, x_scale[M,1] fp32, w_scale[1,N] fp32)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    M, N, K = shape["M"], shape["N"], shape["K"]
    a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32)
    w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
    xq, x_scale = quant_rowwise_int8(a)                  # [M,K], [M,1]
    wq, w_scale_col = quant_rowwise_int8(w)              # [N,K], [N,1]
    w_scale = w_scale_col.reshape(1, N).contiguous()     # CK [1,N]
    return (xq, wq, x_scale, w_scale)


def reference_output(shape, inputs):
    """Exact fp32 dequant-matmul oracle -> bf16 [M,N]."""
    xq, wq, x_scale, w_scale = inputs
    return matmul_a8w8_fp32(xq, wq, x_scale, w_scale)


def candidate_output(fn, shape, inputs):
    xq, wq, x_scale, w_scale = inputs
    return fn(xq, wq, x_scale, w_scale)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER ``gemm_a8w8`` (CK int8 W8A8 scaled GEMM)."""
    import torch

    from kore.tasks.aiter_ref import aiter_gemm_a8w8

    xq, wq, x_scale, w_scale = inputs
    return aiter_gemm_a8w8(xq, wq, x_scale, w_scale, out_dtype=torch.bfloat16)
