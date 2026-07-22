"""GENERATED breadth tr_adadelta seed (fp16). Fused Adadelta step."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_adadelta_kernel(param_ptr, grad_ptr, square_avg_ptr, acc_delta_ptr, lr, rho, eps, wd, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    square_avg = tl.load(square_avg_ptr + offs, mask=mask).to(tl.float32)
    acc_delta = tl.load(acc_delta_ptr + offs, mask=mask).to(tl.float32)
    grad = grad + wd * param
    square_avg = rho * square_avg + (1.0 - rho) * grad * grad
    delta = tl.sqrt(acc_delta + eps) / tl.sqrt(square_avg + eps) * grad
    acc_delta = rho * acc_delta + (1.0 - rho) * delta * delta
    param = param - lr * delta
    tl.store(param_ptr + offs, param.to(tl.float16), mask=mask)
    tl.store(square_avg_ptr + offs, square_avg.to(tl.float16), mask=mask)
    tl.store(acc_delta_ptr + offs, acc_delta.to(tl.float16), mask=mask)


def tr_adadelta(param, grad, square_avg, acc_delta, lr, rho, eps, wd):
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_adadelta_kernel[grid](param, grad, square_avg, acc_delta, lr, rho, eps, wd, numel, BLOCK=BLOCK, num_warps=4)
    return param, square_avg, acc_delta
