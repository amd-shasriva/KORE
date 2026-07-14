"""Reference + inputs for the fp8 (a8w8) GEMM task.

Quantized inference GEMM: activation A and weight W are both fp8 e4m3 with
per-tensor fp32 scales, output bf16. Computes ``Y = (A_deq) @ (W_deq)^T``.

fp8 format is arch-selected via ``FP8_DTYPE``: OCP ``e4m3fn`` on gfx950/CDNA4
(MI350X/MI355X — the native format AITER/hipBLASLt use there); FNUZ
``e4m3fnuz`` on gfx942/CDNA3. The two differ in exponent bias / -0/inf encoding,
so the candidate + oracle must use the SAME (arch) dtype.

Layout (matches AITER ``gemm_a8w8`` CK): XQ [M, K], WQ [N, K] (so the op does
``X @ W^T``), x_scale [M, 1] fp32, w_scale [1, N] fp32.

Correctness oracle: exact torch-fp32 matmul of the DEQUANTIZED fp8 inputs
(so the fp8 rounding is shared by candidate + reference; the gate measures the
kernel's numerical fidelity, not the quantization itself).
"""

from __future__ import annotations

import torch

from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096, "K": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def _per_tensor_quant(x: torch.Tensor):
    amax = x.abs().max().clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    xq = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq, scale


def get_inputs(shape: dict, dtype=None, device="cuda", seed: int = 0):
    """Returns (xq, wq, x_scale, w_scale).

    XQ:[M,K] fp8, WQ:[N,K] fp8, x_scale:[M,1] fp32, w_scale:[1,N] fp32
    (per-tensor scales broadcast into the CK per-row/per-col layout).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    M, N, K = shape["M"], shape["N"], shape["K"]
    a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32)
    w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
    xq, sx = _per_tensor_quant(a)
    wq, sw = _per_tensor_quant(w)
    x_scale = sx.repeat(M, 1).contiguous()   # [M,1]
    w_scale = sw.repeat(1, N).contiguous()   # [1,N]
    return xq, wq, x_scale, w_scale


def matmul_ref(xq, wq, x_scale, w_scale) -> torch.Tensor:
    """Exact fp32 oracle on the dequantized fp8 inputs -> bf16."""
    a_deq = xq.float() * x_scale.float()              # [M,K]
    w_deq = wq.float() * w_scale.float().reshape(-1, 1)  # [N,K]
    return (a_deq @ w_deq.t()).to(torch.bfloat16)
