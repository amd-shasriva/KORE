"""Reference + inputs for DeepSeek-V3 BLOCK-SCALED fp8 GEMM.

The activation A[M,K] is quantized to fp8 per 1x128 group along K (a per-token-group
scale ``x_scale[M, K//128]``); the weight W[N,K] is quantized to fp8 per 128x128 block
(``w_scale[N//128, K//128]``). Computes ``Y = (A_deq) @ (W_deq)^T`` in bf16, where the
per-128-K-block dequant is applied on the fp32 accumulator. This is the DeepSeek-V3 /
SGLang serving GEMM.

fp8 is arch-selected via the live ``kore.tasks.aiter_ref`` (OCP e4m3fn on gfx950).

Correctness oracle: exact fp32 blockwise dequant-matmul, applying each block scale
EXACTLY once (the classic block-scale bug is mis-indexing the K-block or applying the
weight-block scale on the wrong axis; the oracle pins the correct
``xd[m,k]=xq*xs[m,k//128]``, ``wd[n,k]=wq*ws[n//128,k//128]``). get_inputs GUARDS the
128-alignment (K=4095 is illegal for the scale groups and raises).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._quant_common import (  # noqa: E402
    BLK,
    matmul_blockscale_fp32,
    quant_1x128_fp8,
    quant_128x128_fp8,
)

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
    """Returns (xq[M,K] fp8, wq[N,K] fp8, xs[M,K//128] fp32, ws[N//128,K//128] fp32)."""
    import torch

    M, N, K = shape["M"], shape["N"], shape["K"]
    # Block-scale alignment guard (edge S6): K and N MUST be multiples of 128.
    assert K % BLK == 0, f"block-scale needs K %% {BLK} == 0 (got K={K}); K=4095 is illegal"
    assert N % BLK == 0, f"block-scale needs N %% {BLK} == 0 (got N={N})"
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn((M, K), generator=g, device=device, dtype=torch.float32) * 0.2
    W = torch.randn((N, K), generator=g, device=device, dtype=torch.float32) * 0.2
    xq, xs = quant_1x128_fp8(X)
    wq, ws = quant_128x128_fp8(W)
    return (xq, wq, xs, ws)


def reference_output(shape, inputs):
    """Exact fp32 blockwise dequant-matmul oracle -> bf16 [M,N]."""
    xq, wq, xs, ws = inputs
    return matmul_blockscale_fp32(xq, wq, xs, ws)


def candidate_output(fn, shape, inputs):
    xq, wq, xs, ws = inputs
    return fn(xq, wq, xs, ws)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER ``gemm_a8w8_blockscale`` (CK 1x128 act / 128x128 weight)."""
    import torch

    import aiter

    from kore.tasks.aiter_ref import _mark_baseline

    xq, wq, xs, ws = inputs
    out = aiter.gemm_a8w8_blockscale(xq, wq, xs, ws, dtype=torch.bfloat16)
    _mark_baseline("aiter_vendor")
    return out
