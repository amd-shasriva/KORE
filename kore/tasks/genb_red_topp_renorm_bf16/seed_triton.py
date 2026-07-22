"""GENERATED breadth red_topp_renorm seed (bf16). logits[M,N] -> top-p (nucleus)
renormalized probabilities, p=0.9. Partial-fusion starting point: a Triton kernel
computes the numerically-stable softmax; the data-dependent nucleus selection (sort
+ cumulative-mass threshold) and renormalization run host-side in torch. Fusing the
selection into the kernel (a streaming threshold search) is the optimization target.
tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_topp_softmax_kernel(x_ptr, o_ptr, sx, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    m = -float("inf")
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float("inf")).to(tl.float32)
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + row * so + offs, (tl.exp(x - m) / s).to(tl.bfloat16), mask=mask)


def red_topp_renorm(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    M, N = logits.shape
    probs = torch.empty_like(logits)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _red_topp_softmax_kernel[(M,)](logits, probs, logits.stride(0), probs.stride(0), N,
                                   BLOCK_N=BLOCK_N, num_warps=8)
    pf = probs.float()
    sp, si = torch.sort(pf, dim=-1, descending=True)
    excl = sp.cumsum(-1) - sp
    keep_sorted = excl <= p
    keep = torch.zeros_like(pf, dtype=torch.bool).scatter_(-1, si, keep_sorted)
    masked = torch.where(keep, pf, torch.zeros_like(pf))
    return (masked / masked.sum(-1, keepdim=True)).to(logits.dtype)
