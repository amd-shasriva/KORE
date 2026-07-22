"""GENERATED breadth label_smoothing_ce seed (fp16). logits[M,V] + targets[M].
Per-row streaming fp32 pass tracks logsumexp AND the row sum of logits, so
loss[m] = (1-eps)*(lse - logit[target]) + eps*(lse - mean_v logit); eps=0.1.
Matches F.cross_entropy(..., label_smoothing=eps). Then mean over rows."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _label_smoothing_ce_kernel(logits_ptr, tgt_ptr, loss_ptr, sm, V, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    m = -float("inf")
    s = 0.0
    ssum = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
        m = new_m
        xs = tl.load(logits_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        ssum += tl.sum(xs, axis=0)
    lse = m + tl.log(s)
    tgt = tl.load(tgt_ptr + row)
    xt = tl.load(logits_ptr + base + tgt).to(tl.float32)
    nll = lse - xt
    smooth = lse - ssum / V
    tl.store(loss_ptr + row, (1.0 - eps) * nll + eps * smooth)


def label_smoothing_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    loss = torch.empty((M,), device=logits.device, dtype=torch.float32)
    _label_smoothing_ce_kernel[(M,)](logits, targets, loss, logits.stride(0), V,
                                     0.1, BLOCK=1024, num_warps=8)
    return loss.mean().to(logits.dtype)
