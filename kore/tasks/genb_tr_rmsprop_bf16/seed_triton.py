"""GENERATED breadth tr_rmsprop seed (bf16). Fused RMSprop step."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_rmsprop_kernel(param_ptr, grad_ptr, square_avg_ptr, lr, alpha, eps, wd, momentum, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    square_avg = tl.load(square_avg_ptr + offs, mask=mask).to(tl.float32)
    grad = grad + wd * param
    square_avg = alpha * square_avg + (1.0 - alpha) * grad * grad
    avg = tl.sqrt(square_avg) + eps
    param = param - lr * grad / avg
    tl.store(param_ptr + offs, param.to(tl.bfloat16), mask=mask)
    tl.store(square_avg_ptr + offs, square_avg.to(tl.bfloat16), mask=mask)


def tr_rmsprop(param, grad, square_avg, lr, alpha, eps, wd, momentum):
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_rmsprop_kernel[grid](param, grad, square_avg, lr, alpha, eps, wd, momentum, numel, BLOCK=BLOCK, num_warps=4)
    return param, square_avg
