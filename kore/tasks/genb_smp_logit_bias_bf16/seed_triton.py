"""GENERATED breadth smp_logit_bias seed (bf16). logits[M,V] + additive per-token bias. Naive but correct; the
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
    o = x + a
    tl.store(o_ptr + row * so + offs, o.to(tl.bfloat16), mask=mask)


def smp_logit_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _elem_kernel[grid](x, bias, o, x.stride(0), bias.stride(0), o.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    return o
