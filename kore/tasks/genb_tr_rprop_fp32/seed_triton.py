"""GENERATED breadth tr_rprop seed (fp32). Fused Rprop step (per-elem step sizes)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_rprop_kernel(param_ptr, grad_ptr, prev_ptr, step_size_ptr, etam, etap, smin, smax, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    prev = tl.load(prev_ptr + offs, mask=mask).to(tl.float32)
    step_size = tl.load(step_size_ptr + offs, mask=mask).to(tl.float32)
    sign = grad * prev
    mult = tl.where(sign > 0.0, etap, tl.where(sign < 0.0, etam, 1.0))
    step_size = tl.minimum(tl.maximum(step_size * mult, smin), smax)
    g2 = tl.where(sign < 0.0, 0.0, grad)
    gs = tl.where(g2 > 0.0, 1.0, tl.where(g2 < 0.0, -1.0, 0.0))
    param = param - gs * step_size
    prev = g2
    tl.store(param_ptr + offs, param.to(tl.float32), mask=mask)
    tl.store(prev_ptr + offs, prev.to(tl.float32), mask=mask)
    tl.store(step_size_ptr + offs, step_size.to(tl.float32), mask=mask)


def tr_rprop(param, grad, prev, step_size, etam, etap, smin, smax, step):
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_rprop_kernel[grid](param, grad, prev, step_size, etam, etap, smin, smax, numel, BLOCK=BLOCK, num_warps=4)
    return param, prev, step_size
