"""GENERATED breadth smp_topk_mask_k20 seed (bf16). keep the top-20 logits per row, mask the rest to -inf. Naive but correct; the
data-dependent selection runs host-side in torch (the policy fuses it)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _mask_kernel(x_ptr, thr_ptr, o_ptr, sx, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float('inf')).to(tl.float32)
    thr = tl.load(thr_ptr + row).to(tl.float32)
    o = tl.where(x >= thr, x, -float('inf'))
    tl.store(o_ptr + row * so + offs, o.to(tl.bfloat16), mask=mask)


def smp_topk_mask_k20(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    K = 20
    thr = torch.topk(x, K, dim=-1).values[:, -1].contiguous()
    o = torch.empty_like(x)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(N, BLOCK_N))
    _mask_kernel[grid](x, thr, o, x.stride(0), o.stride(0), N, BLOCK_N=BLOCK_N, num_warps=4)
    return o
