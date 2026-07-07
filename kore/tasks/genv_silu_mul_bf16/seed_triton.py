"""GENERATED vendor-baselined silu_mul seed (bf16) vs aiter silu_mul.
Gated MLP activation x[M,2*inter] -> silu(gate)*up [M,inter], tl.bfloat16 store.
Regenerate via kore/tasks/generate_vendor_ops.py."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _silu_mul_kernel(x_ptr, y_ptr, sxm, sym, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    gate = tl.load(x_ptr + row * sxm + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * sxm + N + offs, mask=mask, other=0.0).to(tl.float32)
    act = gate * tl.sigmoid(gate)
    tl.store(y_ptr + row * sym + offs, (act * up).to(tl.bfloat16), mask=mask)


def silu_mul(x: torch.Tensor) -> torch.Tensor:
    M, two_n = x.shape
    N = two_n // 2
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _silu_mul_kernel[grid](x, y, x.stride(0), y.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    return y
