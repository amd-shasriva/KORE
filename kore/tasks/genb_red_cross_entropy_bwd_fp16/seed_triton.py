"""GENERATED breadth red_cross_entropy_bwd seed (fp16). logits[M,V]+targets[M] ->
dlogits = softmax(logits) - onehot(target) [M,V] (the gradient of the sum-reduced
CE). Pass 1 the streaming max-subtracted lse; pass 2 writes exp(logit-lse), minus 1
at the target column. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_cross_entropy_bwd_kernel(x_ptr, t_ptr, o_ptr, sx, so, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    bx = row * sx
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
    tgt = tl.load(t_ptr + row)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + bx + offs, mask=mask, other=-float("inf")).to(tl.float32)
        grad = tl.exp(x - lse)
        grad = grad - tl.where(offs == tgt, 1.0, 0.0)
        tl.store(o_ptr + row * so + offs, grad.to(tl.float16), mask=mask)


def red_cross_entropy_bwd(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    o = torch.empty_like(logits)
    _red_cross_entropy_bwd_kernel[(M,)](logits, targets, o, logits.stride(0), o.stride(0), V,
                                        BLOCK=1024, num_warps=8)
    return o
