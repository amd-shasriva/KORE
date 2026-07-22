"""GENERATED breadth smp_topp_sample seed (bf16). top-p renormalize then inverse-CDF sample via supplied u. Naive but correct; the
data-dependent selection runs host-side in torch (the policy fuses it)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _sm_kernel(x_ptr, o_ptr, sx, so, N, INV_T, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    m = -float('inf')
    s = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=-float('inf')).to(tl.float32) * INV_T
        blk = tl.max(x, axis=0)
        new_m = tl.maximum(m, blk)
        s = s * tl.exp(m - new_m) + tl.sum(tl.where(mask, tl.exp(x - new_m), 0.0), axis=0)
        m = new_m
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32) * INV_T
        tl.store(o_ptr + row * so + offs, tl.exp(x - m) / s, mask=mask)


def smp_topp_sample(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    probs = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_N = 1024 if N > 1024 else triton.next_power_of_2(N)
    _sm_kernel[(M,)](x, probs, x.stride(0), probs.stride(0), N, 1.0, BLOCK_N=BLOCK_N, num_warps=8)
    sp, si = torch.sort(probs, dim=-1, descending=True)
    excl = sp.cumsum(-1) - sp
    keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, si, excl <= 0.9)
    masked = torch.where(keep, probs, torch.zeros_like(probs))
    pp = masked / masked.sum(-1, keepdim=True)
    cdf = pp.cumsum(-1)
    idx = torch.searchsorted(cdf, u.float().view(-1, 1), right=True).squeeze(-1)
    return idx.clamp_(max=N - 1).to(torch.int64)
