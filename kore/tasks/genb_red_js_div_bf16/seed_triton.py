"""GENERATED breadth red_js_div seed (bf16). logits_p[M,V], logits_q[M,V] ->
Jensen-Shannon divergence between softmax(p) and softmax(q) per row:
0.5*sum p*log(p/m) + 0.5*sum q*log(q/m), m=(p+q)/2. Stable max-subtracted lse for
p and q, then the combine pass. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_js_div_kernel(p_ptr, q_ptr, o_ptr, sp, sq, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    bp = row * sp
    bq = row * sq
    mp = -float("inf")
    sps = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(p_ptr + bp + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(mp, blk)
        sps = sps * tl.exp(mp - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        mp = new_m
    lse_p = mp + tl.log(sps)
    mq = -float("inf")
    sqs = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(q_ptr + bq + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(mq, blk)
        sqs = sqs * tl.exp(mq - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        mq = new_m
    lse_q = mq + tl.log(sqs)
    acc = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        xp = tl.load(p_ptr + bp + offs, mask=mask, other=0.0).to(tl.float32)
        xq = tl.load(q_ptr + bq + offs, mask=mask, other=0.0).to(tl.float32)
        lp_i = xp - lse_p
        lq_i = xq - lse_q
        p = tl.exp(lp_i)
        q = tl.exp(lq_i)
        logmm = tl.log(0.5 * (p + q))
        tp = tl.where(p > 0.0, p * (lp_i - logmm), 0.0)
        tq = tl.where(q > 0.0, q * (lq_i - logmm), 0.0)
        term = 0.5 * (tp + tq)
        acc += tl.sum(tl.where(mask, term, 0.0), axis=0)
    tl.store(o_ptr + row, acc.to(tl.bfloat16))


def red_js_div(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    M, V = logits_p.shape
    o = torch.empty((M,), device=logits_p.device, dtype=logits_p.dtype)
    _red_js_div_kernel[(M,)](logits_p, logits_q, o, logits_p.stride(0), logits_q.stride(0), V,
                             BLOCK=1024, num_warps=8)
    return o
