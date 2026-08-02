"""GENERATED breadth smp_gumbel_max seed (bf16). argmax(logits + supplied gumbel noise) - the Gumbel-max sampler. Naive but correct; the
data-dependent selection runs host-side in torch (the policy fuses it)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _gumbel_kernel(x_ptr, g_ptr, o_ptr, sx, sg, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * sg + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * so + offs, x + g, mask=mask)


def smp_gumbel_max(x: torch.Tensor, gumbel: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _gumbel_kernel[grid](x, gumbel, y, x.stride(0), gumbel.stride(0), y.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    return y.argmax(-1).to(torch.int64)
