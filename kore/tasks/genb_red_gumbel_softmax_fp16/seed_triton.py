"""GENERATED breadth red_gumbel_softmax seed (fp16). logits[M,N] + gumbel[M,N]
-> softmax((logits + gumbel)/tau) over the last dim, tau=0.5. Stable streaming
max + rescaled exp-sum (fp32), then a normalized write. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_gumbel_softmax_kernel(x_ptr, g_ptr, o_ptr, sx, sg, so, N, INV_T, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * sg + offs, mask=mask, other=0.0).to(tl.float32)
        z = tl.where(mask, (x + g) * INV_T, -float("inf"))
        blk = tl.max(z, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(z - new_m), 0.0), axis=0)
        m = new_m
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * sg + offs, mask=mask, other=0.0).to(tl.float32)
        z = (x + g) * INV_T - m
        tl.store(o_ptr + row * so + offs, (tl.exp(z) / s).to(tl.float16), mask=mask)


def red_gumbel_softmax(logits: torch.Tensor, gumbel: torch.Tensor) -> torch.Tensor:
    M, N = logits.shape
    o = torch.empty_like(logits)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _red_gumbel_softmax_kernel[(M,)](logits, gumbel, o, logits.stride(0), gumbel.stride(0),
                                     o.stride(0), N, 2.0, BLOCK_N=BLOCK_N, num_warps=8)
    return o
