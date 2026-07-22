"""GENERATED breadth smp_frequency_penalty seed (fp32). OpenAI frequency penalty (subtract coef * count). Naive but correct; the
data-dependent selection runs host-side in torch (the policy fuses it)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _elem_kernel(x_ptr, a_ptr, o_ptr, sx, sa, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + row * sa + offs, mask=mask, other=0.0).to(tl.float32)
    o = x - 0.5 * a
    tl.store(o_ptr + row * so + offs, o.to(tl.float32), mask=mask)


def smp_frequency_penalty(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _elem_kernel[grid](x, counts, o, x.stride(0), counts.stride(0), o.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    return o
