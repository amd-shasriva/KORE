"""GENERATED breadth tr_grad_clip_per_layer seed (fp32). Per-layer (per-row) grad-norm clip."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tr_grad_clip_per_layer_kernel(grads_ptr, out_ptr, sm, N, arg0, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    acc = 0.0
    acc2 = 0.0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(grads_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    coef = tl.minimum(arg0 / (tl.sqrt(acc) + 1e-06), 1.0)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(grads_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(grads_ptr + base + offs, (x * coef).to(tl.float32), mask=m)


def tr_grad_clip_per_layer(grads, max_norm):
    G, N = grads.shape
    _tr_grad_clip_per_layer_kernel[(G,)](grads, grads, grads.stride(0), N, max_norm, BLOCK=1024, num_warps=8)
    return grads
