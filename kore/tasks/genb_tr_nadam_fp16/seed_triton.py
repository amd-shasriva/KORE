"""GENERATED breadth tr_nadam seed (fp16). Fused NAdam step."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_nadam_kernel(param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr, b1, b2, eps, bc2, c1, c2, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    exp_avg = tl.load(exp_avg_ptr + offs, mask=mask).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offs, mask=mask).to(tl.float32)
    exp_avg = exp_avg + (1.0 - b1) * (grad - exp_avg)
    exp_avg_sq = b2 * exp_avg_sq + (1.0 - b2) * grad * grad
    den = tl.sqrt(exp_avg_sq / bc2) + eps
    param = param + grad * c1 / den + exp_avg * c2 / den
    tl.store(param_ptr + offs, param.to(tl.float16), mask=mask)
    tl.store(exp_avg_ptr + offs, exp_avg.to(tl.float16), mask=mask)
    tl.store(exp_avg_sq_ptr + offs, exp_avg_sq.to(tl.float16), mask=mask)


def tr_nadam(param, grad, exp_avg, exp_avg_sq, lr, b1, b2, eps, wd, momentum_decay, step):
    numel = param.numel()
    bc2 = 1.0 - b2 ** step
    mu_t = b1 * (1.0 - 0.5 * 0.96 ** (step * momentum_decay))
    mu_next = b1 * (1.0 - 0.5 * 0.96 ** ((step + 1) * momentum_decay))
    mu_prod = 1.0
    for _i in range(1, step + 1):
        mu_prod = mu_prod * b1 * (1.0 - 0.5 * 0.96 ** (_i * momentum_decay))
    c1 = -lr * (1.0 - mu_t) / (1.0 - mu_prod)
    c2 = -lr * mu_next / (1.0 - mu_prod * mu_next)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_nadam_kernel[grid](param, grad, exp_avg, exp_avg_sq, b1, b2, eps, bc2, c1, c2, numel, BLOCK=BLOCK, num_warps=4)
    return param, exp_avg, exp_avg_sq
