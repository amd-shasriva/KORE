"""Reference + inputs for the dense bf16 GEMM task.

Dense matmul ``Y = A @ B`` with A[M,K], B[K,N], bf16 inputs, fp32 accumulation,
bf16 output — the workhorse of prefill (square, compute-bound) and decode
(tiny-M GEMV, memory-bound) serving.

Correctness oracle: exact torch-fp32 matmul of the (fp32-upcast) bf16 inputs, so
the gate measures the kernel's fp32-accumulation fidelity, not input rounding.
Perf baseline (driver --impl reference): ``torch.matmul`` which on ROCm dispatches
to hipBLASLt — the real vendor GEMM the serving stack calls.
"""

from __future__ import annotations

import torch


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"M": 4096, "N": 4096, "K": 4096}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (a, b): A[M,K] bf16, B[K,N] bf16."""
    g = torch.Generator(device=device).manual_seed(seed)
    M, N, K = shape["M"], shape["N"], shape["K"]
    # 1/sqrt(K) scaling keeps the fp32 accumulator well-conditioned so the SNR
    # gate reflects accumulation fidelity rather than bf16 output saturation.
    a = (torch.randn((M, K), generator=g, device=device, dtype=torch.float32) / (K ** 0.5)).to(dtype)
    b = torch.randn((K, N), generator=g, device=device, dtype=torch.float32).to(dtype)
    return a, b


def matmul_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact fp32 oracle: (A_f32 @ B_f32) -> bf16."""
    return (a.float() @ b.float()).to(torch.bfloat16)
