"""GENERATED breadth tr_adamw seed (fp16). Fused Adam-family step."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_adamw_kernel(param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr, lr, wd, b1, b2, eps, step_size, bc2sqrt, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    exp_avg = tl.load(exp_avg_ptr + offs, mask=mask).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offs, mask=mask).to(tl.float32)
    param = param * (1.0 - lr * wd)
    exp_avg = exp_avg + (1.0 - b1) * (grad - exp_avg)
    exp_avg_sq = b2 * exp_avg_sq + (1.0 - b2) * grad * grad
    denom = tl.sqrt(exp_avg_sq) / bc2sqrt + eps
    param = param - step_size * exp_avg / denom
    tl.store(param_ptr + offs, param.to(tl.float16), mask=mask)
    tl.store(exp_avg_ptr + offs, exp_avg.to(tl.float16), mask=mask)
    tl.store(exp_avg_sq_ptr + offs, exp_avg_sq.to(tl.float16), mask=mask)


def tr_adamw(param, grad, exp_avg, exp_avg_sq, lr, b1, b2, eps, wd, step):
    numel = param.numel()
    bc1 = 1.0 - b1 ** step
    step_size = lr / bc1
    bc2sqrt = (1.0 - b2 ** step) ** 0.5
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_adamw_kernel[grid](param, grad, exp_avg, exp_avg_sq, lr, wd, b1, b2, eps, step_size, bc2sqrt, numel, BLOCK=BLOCK, num_warps=4)
    return param, exp_avg, exp_avg_sq
