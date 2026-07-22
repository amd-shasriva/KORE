"""GENERATED breadth kl_div seed (fp16). log_p[M,V], q[M,V] -> KL(q || p) batchmean.
Per-row fp32 sum of q*(log q - log_p) (0*log0 -> 0), then sum over rows / M (batch).
Matches F.kl_div(log_p, q, reduction='batchmean')."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _kl_div_kernel(logp_ptr, q_ptr, out_ptr, sm, V, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * sm
    acc = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        lp = tl.load(logp_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(q_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        term = tl.where(q > 0.0, q * (tl.log(q) - lp), 0.0)
        acc += tl.sum(term, axis=0)
    tl.store(out_ptr + row, acc)


def kl_div(log_p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    M, V = log_p.shape
    rows = torch.empty((M,), device=log_p.device, dtype=torch.float32)
    _kl_div_kernel[(M,)](log_p, q, rows, log_p.stride(0), V, BLOCK=1024, num_warps=8)
    return (rows.sum() / M).to(log_p.dtype)
