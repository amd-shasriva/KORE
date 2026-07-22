"""GENERATED breadth tr_grad_accum_scale seed (fp16). Fused scaled gradient accumulation."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_grad_accum_scale_kernel(accum_ptr, grad_ptr, scale, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    accum = tl.load(accum_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    accum = accum + scale * grad
    tl.store(accum_ptr + offs, accum.to(tl.float16), mask=mask)


def tr_grad_accum_scale(accum, grad, scale):
    numel = accum.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_grad_accum_scale_kernel[grid](accum, grad, scale, numel, BLOCK=BLOCK, num_warps=4)
    return accum
