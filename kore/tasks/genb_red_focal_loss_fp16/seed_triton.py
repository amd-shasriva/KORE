"""GENERATED breadth red_focal_loss seed (fp16). logits[M,V]+targets[M] -> the
multiclass focal loss -(1-pt)^2 * log(pt), pt = softmax(logits)[target] (gamma=2).
Streaming max-subtracted lse; log(pt) = logit[target] - lse (stable). tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_focal_loss_kernel(x_ptr, t_ptr, o_ptr, sx, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sx
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tgt = tl.load(t_ptr + row)
    xt = tl.load(x_ptr + base + tgt).to(tl.float32)
    logpt = xt - lse
    pt = tl.exp(logpt)
    omp = 1.0 - pt
    tl.store(o_ptr + row, (-(omp * omp) * logpt).to(tl.float16))


def red_focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    o = torch.empty((M,), device=logits.device, dtype=logits.dtype)
    _red_focal_loss_kernel[(M,)](logits, targets, o, logits.stride(0), V, BLOCK=1024, num_warps=8)
    return o
