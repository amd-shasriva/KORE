"""GENERATED breadth cross_entropy seed (bf16). logits[M,V] + targets[M] -> mean CE.
One program per row: streaming (online) fp32 logsumexp so any vocab width V fits;
loss[m] = logsumexp(logits[m]) - logits[m, target[m]]; the row losses are then
mean-reduced. Naive starting point (per-row + a torch mean); the policy fuses it."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cross_entropy_kernel(logits_ptr, tgt_ptr, loss_ptr, sm, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    m = -float("inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tgt = tl.load(tgt_ptr + row)
    xt = tl.load(logits_ptr + base + tgt).to(tl.float32)
    tl.store(loss_ptr + row, lse - xt)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    loss = torch.empty((M,), device=logits.device, dtype=torch.float32)
    _cross_entropy_kernel[(M,)](logits, targets, loss, logits.stride(0), V,
                                BLOCK=1024, num_warps=8)
    return loss.mean().to(logits.dtype)
