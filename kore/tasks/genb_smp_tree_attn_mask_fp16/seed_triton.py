"""GENERATED breadth smp_tree_attn_mask seed (fp16). tree-attention mask from a parent table (node attends to its ancestors). Naive but correct; the
data-dependent selection runs host-side in torch (the policy fuses it)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _copy_kernel(x_ptr, o_ptr, sx, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * so + offs, x.to(tl.float16), mask=mask)


def smp_tree_attn_mask(parent: torch.Tensor) -> torch.Tensor:
    M, T = parent.shape
    mask = torch.zeros((M, T, T), device=parent.device, dtype=torch.float32)
    par = parent.tolist()
    for m in range(M):
        pm = par[m]
        for it in range(T):
            mask[m, it, it] = 1.0
            a = pm[it]
            while a >= 0:
                mask[m, it, a] = 1.0
                a = pm[a]
    flat = mask.reshape(M * T, T)
    o = torch.empty((M * T, T), device=parent.device, dtype=torch.float16)
    BLOCK_N = triton.next_power_of_2(T)
    _copy_kernel[(M * T,)](flat, o, flat.stride(0), o.stride(0), T, BLOCK_N=BLOCK_N, num_warps=1)
    return o.reshape(M, T, T)
