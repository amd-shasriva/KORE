"""GENERATED breadth tr_adabelief seed (bf16). Fused AdaBelief step (variance of grad-EMA)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_adabelief_kernel(param_ptr, grad_ptr, exp_avg_ptr, exp_avg_var_ptr, lr, b1, b2, eps, wd, bc1, bc2, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    exp_avg = tl.load(exp_avg_ptr + offs, mask=mask).to(tl.float32)
    exp_avg_var = tl.load(exp_avg_var_ptr + offs, mask=mask).to(tl.float32)
    param = param * (1.0 - lr * wd)
    exp_avg = b1 * exp_avg + (1.0 - b1) * grad
    diff = grad - exp_avg
    exp_avg_var = b2 * exp_avg_var + (1.0 - b2) * diff * diff + eps
    mhat = exp_avg / bc1
    shat = exp_avg_var / bc2
    param = param - lr * mhat / (tl.sqrt(shat) + eps)
    tl.store(param_ptr + offs, param.to(tl.bfloat16), mask=mask)
    tl.store(exp_avg_ptr + offs, exp_avg.to(tl.bfloat16), mask=mask)
    tl.store(exp_avg_var_ptr + offs, exp_avg_var.to(tl.bfloat16), mask=mask)


def tr_adabelief(param, grad, exp_avg, exp_avg_var, lr, b1, b2, eps, wd, step):
    numel = param.numel()
    bc1 = 1.0 - b1 ** step
    bc2 = 1.0 - b2 ** step
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_adabelief_kernel[grid](param, grad, exp_avg, exp_avg_var, lr, b1, b2, eps, wd, bc1, bc2, numel, BLOCK=BLOCK, num_warps=4)
    return param, exp_avg, exp_avg_var
