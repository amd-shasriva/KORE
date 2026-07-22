"""GENERATED breadth tr_radam seed (fp32). Fused RAdam step (variance rectification)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_radam_kernel(param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr, b1, b2, eps, step_coef, plain_coef, use_rect, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    exp_avg = tl.load(exp_avg_ptr + offs, mask=mask).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offs, mask=mask).to(tl.float32)
    exp_avg = exp_avg + (1.0 - b1) * (grad - exp_avg)
    exp_avg_sq = b2 * exp_avg_sq + (1.0 - b2) * grad * grad
    den = tl.sqrt(exp_avg_sq) + eps
    upd = tl.where(use_rect != 0, step_coef * exp_avg / den, plain_coef * exp_avg)
    param = param - upd
    tl.store(param_ptr + offs, param.to(tl.float32), mask=mask)
    tl.store(exp_avg_ptr + offs, exp_avg.to(tl.float32), mask=mask)
    tl.store(exp_avg_sq_ptr + offs, exp_avg_sq.to(tl.float32), mask=mask)


def tr_radam(param, grad, exp_avg, exp_avg_sq, lr, b1, b2, eps, wd, step):
    numel = param.numel()
    bc1 = 1.0 - b1 ** step
    bc2 = 1.0 - b2 ** step
    rho_inf = 2.0 / (1.0 - b2) - 1.0
    rho_t = rho_inf - 2.0 * step * (b2 ** step) / bc2
    if rho_t > 5.0:
        rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)) ** 0.5
        step_coef = lr * rect * (bc2 ** 0.5) / bc1
        plain_coef = 0.0
        use_rect = 1
    else:
        step_coef = 0.0
        plain_coef = lr / bc1
        use_rect = 0
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_radam_kernel[grid](param, grad, exp_avg, exp_avg_sq, b1, b2, eps, step_coef, plain_coef, use_rect, numel, BLOCK=BLOCK, num_warps=4)
    return param, exp_avg, exp_avg_sq
