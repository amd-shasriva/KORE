"""GENERATED breadth grad_clip_global_norm seed (bf16). grads[G,N] clipped by their
GLOBAL L2 norm IN PLACE. Kernel 1: per-row fp32 sum-of-squares -> [G]; host: total
norm + coef = min(max_norm/(total+1e-06), 1). Kernel 2: scale grads by coef.
Matches torch.nn.utils.clip_grad_norm_ over the stacked grads."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _gc_sumsq_kernel(g_ptr, part_ptr, sm, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    acc = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(g_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    tl.store(part_ptr + row, acc)


@triton.jit
def _gc_scale_kernel(g_ptr, numel, coef, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    tl.store(g_ptr + offs, (x * coef).to(tl.bfloat16), mask=mask)


def grad_clip_global_norm(grads: torch.Tensor, max_norm) -> torch.Tensor:
    G, N = grads.shape
    part = torch.empty((G,), device=grads.device, dtype=torch.float32)
    _gc_sumsq_kernel[(G,)](grads, part, grads.stride(0), N, BLOCK=1024, num_warps=8)
    total_norm = torch.sqrt(part.sum())
    coef = min(max_norm / (total_norm.item() + 1e-06), 1.0)
    numel = grads.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _gc_scale_kernel[grid](grads, numel, coef, BLOCK=BLOCK, num_warps=4)
    return grads
