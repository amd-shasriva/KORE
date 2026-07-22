"""GENERATED breadth tr_grad_zero_center seed (fp32). Gradient centralization (per-row mean-0)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_grad_zero_center_kernel(g_ptr, sm, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    acc = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(g_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    mean = acc / N
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(g_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(g_ptr + base + offs, (x - mean).to(tl.float32), mask=m)


def tr_grad_zero_center(grads):
    G, N = grads.shape
    _tr_grad_zero_center_kernel[(G,)](grads, grads.stride(0), N, BLOCK=1024, num_warps=8)
    return grads
