"""GENERATED breadth red_soft_cross_entropy seed (fp16). logits[M,V], q[M,V] (a
distribution) -> -sum_j q_j log_softmax(logits)_j = lse*sum_j q_j - sum_j q_j x_j.
Pass 1 the streaming lse; pass 2 the weighted sums. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_soft_cross_entropy_kernel(x_ptr, q_ptr, o_ptr, sx, sq, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    bx = row * sx
    bq = row * sq
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + bx + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    wsum = 0.0
    dot = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + bx + offs, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(q_ptr + bq + offs, mask=mask, other=0.0).to(tl.float32)
        wsum += tl.sum(tl.where(mask, q, 0.0), axis=0)
        dot += tl.sum(tl.where(mask, q * x, 0.0), axis=0)
    tl.store(o_ptr + row, (lse * wsum - dot).to(tl.float16))


def red_soft_cross_entropy(logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    o = torch.empty((M,), device=logits.device, dtype=logits.dtype)
    _red_soft_cross_entropy_kernel[(M,)](logits, q, o, logits.stride(0), q.stride(0), V,
                                         BLOCK=1024, num_warps=8)
    return o
