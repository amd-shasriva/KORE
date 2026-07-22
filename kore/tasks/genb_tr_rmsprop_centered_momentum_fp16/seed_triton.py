"""GENERATED breadth tr_rmsprop_centered_momentum seed (fp16). Fused RMSprop step."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_rmsprop_centered_momentum_kernel(param_ptr, grad_ptr, square_avg_ptr, grad_avg_ptr, momentum_buffer_ptr, lr, alpha, eps, wd, momentum, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    param = tl.load(param_ptr + offs, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    square_avg = tl.load(square_avg_ptr + offs, mask=mask).to(tl.float32)
    grad_avg = tl.load(grad_avg_ptr + offs, mask=mask).to(tl.float32)
    momentum_buffer = tl.load(momentum_buffer_ptr + offs, mask=mask).to(tl.float32)
    grad = grad + wd * param
    square_avg = alpha * square_avg + (1.0 - alpha) * grad * grad
    grad_avg = grad_avg + (1.0 - alpha) * (grad - grad_avg)
    avg = tl.sqrt(square_avg - grad_avg * grad_avg) + eps
    momentum_buffer = momentum * momentum_buffer + grad / avg
    param = param - lr * momentum_buffer
    tl.store(param_ptr + offs, param.to(tl.float16), mask=mask)
    tl.store(square_avg_ptr + offs, square_avg.to(tl.float16), mask=mask)
    tl.store(grad_avg_ptr + offs, grad_avg.to(tl.float16), mask=mask)
    tl.store(momentum_buffer_ptr + offs, momentum_buffer.to(tl.float16), mask=mask)


def tr_rmsprop_centered_momentum(param, grad, square_avg, grad_avg, momentum_buffer, lr, alpha, eps, wd, momentum):
    numel = param.numel()
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _tr_rmsprop_centered_momentum_kernel[grid](param, grad, square_avg, grad_avg, momentum_buffer, lr, alpha, eps, wd, momentum, numel, BLOCK=BLOCK, num_warps=4)
    return param, square_avg, grad_avg, momentum_buffer
