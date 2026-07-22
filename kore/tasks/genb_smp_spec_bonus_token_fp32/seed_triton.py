"""GENERATED breadth smp_spec_bonus_token seed (fp32). bonus token: inverse-CDF sample from the target distribution. Naive but correct; the
data-dependent selection runs host-side in torch (the policy fuses it)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cumsum_kernel(p_ptr, o_ptr, sp, so, N):
    row = tl.program_id(0)
    acc = 0.0
    for i in range(0, N):
        v = tl.load(p_ptr + row * sp + i).to(tl.float32)
        acc += v
        tl.store(o_ptr + row * so + i, acc)


def smp_spec_bonus_token(p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    M, N = p.shape
    cdf = torch.empty((M, N), device=p.device, dtype=torch.float32)
    _cumsum_kernel[(M,)](p, cdf, p.stride(0), cdf.stride(0), N, num_warps=1)
    idx = torch.searchsorted(cdf, u.float().view(-1, 1), right=True).squeeze(-1)
    return idx.clamp_(max=N - 1).to(torch.int64)
