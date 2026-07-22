"""GENERATED breadth tr_agc seed (fp32). Adaptive gradient clipping (NFNets)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_agc_kernel(p_ptr, g_ptr, sm, N, clip, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    pn = 0.0
    gn = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        p = tl.load(p_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        pn += tl.sum(p * p, axis=0)
        gn += tl.sum(g * g, axis=0)
    pnorm = tl.maximum(tl.sqrt(pn), eps)
    gnorm = tl.sqrt(gn)
    coef = tl.minimum(clip * pnorm / (gnorm + 1e-12), 1.0)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        g = tl.load(g_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(g_ptr + base + offs, (g * coef).to(tl.float32), mask=m)


def tr_agc(params, grads, clip, eps):
    G, N = grads.shape
    _tr_agc_kernel[(G,)](params, grads, grads.stride(0), N, clip, eps, BLOCK=1024, num_warps=8)
    return grads
