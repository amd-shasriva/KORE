"""GENERATED breadth red_label_smoothing_ce seed (bf16). logits[M,V]+targets[M].
Per-row streaming pass tracks logsumexp AND the row sum of logits, so
loss = (1-eps)*(lse - logit[target]) + eps*(lse - mean_v logit_v), eps=0.1.
Matches F.cross_entropy(..., label_smoothing=eps). tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_label_smoothing_ce_kernel(x_ptr, t_ptr, o_ptr, sx, V, EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sx
    m = -float("inf")
    s = 0.0
    ssum = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(x_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
        xs = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        ssum += tl.sum(xs, axis=0)
    lse = m + tl.log(s)
    tgt = tl.load(t_ptr + row)
    xt = tl.load(x_ptr + base + tgt).to(tl.float32)
    nll = lse - xt
    smooth = lse - ssum / V
    tl.store(o_ptr + row, ((1.0 - EPS) * nll + EPS * smooth).to(tl.bfloat16))


def red_label_smoothing_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, V = logits.shape
    o = torch.empty((M,), device=logits.device, dtype=logits.dtype)
    _red_label_smoothing_ce_kernel[(M,)](logits, targets, o, logits.stride(0), V,
                                         0.1, BLOCK=1024, num_warps=8)
    return o
