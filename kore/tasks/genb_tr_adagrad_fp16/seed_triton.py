"""GENERATED breadth tr_adagrad seed (fp16). Fused Adagrad step."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_adagrad_kernel(param_ptr, grad_ptr, state_sum_ptr, clr, eps, wd, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    state_sum = tl.load(state_sum_ptr + offs, mask=mask).to(tl.float32)
    grad = grad + wd * param
    state_sum = state_sum + grad * grad
    param = param - clr * grad / (tl.sqrt(state_sum) + eps)
    tl.store(param_ptr + offs, param.to(tl.float16), mask=mask)
    tl.store(state_sum_ptr + offs, state_sum.to(tl.float16), mask=mask)


def tr_adagrad(param, grad, state_sum, lr, eps, wd, lr_decay, step):
    numel = param.numel()
    clr = lr / (1.0 + (step - 1) * lr_decay)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_adagrad_kernel[grid](param, grad, state_sum, clr, eps, wd, numel, BLOCK=BLOCK, num_warps=4)
    return param, state_sum
