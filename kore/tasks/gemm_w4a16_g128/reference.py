"""Reference + inputs for group-wise ASYMMETRIC int4 weight GEMM (W4A16, AWQ/GPTQ).

Weight-only 4-bit GEMM for memory-bound decode, in the dominant production layout: the
activation A[M,K] stays bf16; the weight W[N,K] is quantized to unsigned 4-bit codes
0..15 with an ASYMMETRIC (zero-point) affine map per group of ``group=128`` consecutive
K elements:
    W_deq[n,k] = (code[n,k] - zero[n, k//group]) * scale[n, k//group]
    Y = A @ W_deq^T            (bf16 out)
Codes are packed 2 nibbles/byte along K -> w_packed[N,K//2] uint8; per-group fp32
scale[N,K//group] and uint8 zero[N,K//group].

This complements the live per-channel SYMMETRIC ``gemm_w4a16``. The zero-point is the
classic bug source (dropping it, or applying it after the scale) -- the oracle pins the
exact ``(code - zero) * scale`` order.

Correctness oracle: exact fp32 matmul of the dequantized int4 weight (zero-point + scale
applied EXACTLY once). The int4 rounding is shared by candidate + reference. Layout-
agnostic: with TA=1 the activation A is a non-contiguous [M,K] view (transpose edge L1);
the oracle uses ``a.float()`` and matmul, so it is correct for any A stride.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._quant_common import matmul_w4a16_group_fp32, quant_pack_int4_group_asym  # noqa: E402

ENTRY = "gemm"
ATOL = 5e-1
RTOL = 5e-2
GROUP = 128


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 16, "N": 4096, "K": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (a[M,K] bf16, w_packed[N,K//2] u8, scale[N,K//128] fp32, zero[N,K//128] u8)."""
    import torch

    M, N, K = shape["M"], shape["N"], shape["K"]
    assert K % GROUP == 0, f"group int4 needs K %% {GROUP} == 0 (got K={K})"
    g = torch.Generator(device=device).manual_seed(seed)
    if bool(shape.get("TA", 0)):
        # Transposed A: a is a NON-CONTIGUOUS [M,K] view of a [K,M] bf16 buffer (L1).
        a = torch.randn((K, M), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16).t()
    else:
        a = torch.randn((M, K), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    w = torch.randn((N, K), generator=g, device=device, dtype=torch.float32)
    w_packed, scale, zero = quant_pack_int4_group_asym(w, GROUP)
    return (a, w_packed, scale, zero)


def reference_output(shape, inputs):
    """Exact fp32 grouped dequant-matmul oracle -> bf16 [M,N]."""
    a, w_packed, scale, zero = inputs
    return matmul_w4a16_group_fp32(a, w_packed, scale, zero, shape["K"], GROUP)


def candidate_output(fn, shape, inputs):
    a, w_packed, scale, zero = inputs
    return fn(a, w_packed, scale, zero)


def baseline_output(shape, inputs):
    """REAL vendor bar: materialize the int4 weight to bf16 + hipBLASLt matmul."""
    import torch

    from kore.tasks._quant_common import unpack_dequant_int4_group_asym
    from kore.tasks.aiter_ref import hipblaslt_gemm_bf16

    a, w_packed, scale, zero = inputs
    w_deq = unpack_dequant_int4_group_asym(w_packed, scale, zero, shape["K"], GROUP)
    return hipblaslt_gemm_bf16(a, w_deq.to(torch.bfloat16).t().contiguous())
