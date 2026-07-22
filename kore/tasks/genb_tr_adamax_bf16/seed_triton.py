"""GENERATED breadth tr_adamax seed (bf16). Fused Adamax step."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_adamax_kernel(param_ptr, grad_ptr, exp_avg_ptr, exp_inf_ptr, b1, b2, eps, wd, clr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    exp_avg = tl.load(exp_avg_ptr + offs, mask=mask).to(tl.float32)
    exp_inf = tl.load(exp_inf_ptr + offs, mask=mask).to(tl.float32)
    grad = grad + wd * param
    exp_avg = exp_avg + (1.0 - b1) * (grad - exp_avg)
    exp_inf = tl.maximum(b2 * exp_inf, tl.abs(grad) + eps)
    param = param - clr * exp_avg / exp_inf
    tl.store(param_ptr + offs, param.to(tl.bfloat16), mask=mask)
    tl.store(exp_avg_ptr + offs, exp_avg.to(tl.bfloat16), mask=mask)
    tl.store(exp_inf_ptr + offs, exp_inf.to(tl.bfloat16), mask=mask)


def tr_adamax(param, grad, exp_avg, exp_inf, lr, b1, b2, eps, wd, step):
    numel = param.numel()
    clr = lr / (1.0 - b1 ** step)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_adamax_kernel[grid](param, grad, exp_avg, exp_inf, b1, b2, eps, wd, clr, numel, BLOCK=BLOCK, num_warps=4)
    return param, exp_avg, exp_inf
