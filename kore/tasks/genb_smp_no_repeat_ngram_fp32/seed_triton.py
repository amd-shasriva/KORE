"""GENERATED breadth smp_no_repeat_ngram seed (fp32). block tokens that would repeat a previously-seen n-gram. Naive but correct; the
data-dependent selection runs host-side in torch (the policy fuses it)."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _elem_kernel(x_ptr, a_ptr, o_ptr, sx, sa, so, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * sx + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + row * sa + offs, mask=mask, other=0.0).to(tl.float32)
    o = tl.where(a > 0.0, -float('inf'), x)
    tl.store(o_ptr + row * so + offs, o.to(tl.float32), mask=mask)


def smp_no_repeat_ngram(x: torch.Tensor, prev_ids: torch.Tensor) -> torch.Tensor:
    M, V = x.shape
    L = prev_ids.shape[1]
    n = 3
    ban = torch.zeros((M, V), device=x.device, dtype=torch.float32)
    if L >= n:
        rows = prev_ids.tolist()
        for i in range(M):
            row = rows[i]
            suffix = row[L - n + 1:L]
            for j in range(0, L - n + 1):
                if row[j:j + n - 1] == suffix:
                    ban[i, row[j + n - 1]] = 1.0
    o = torch.empty_like(x)
    BLOCK_N = 1024
    grid = (M, triton.cdiv(V, BLOCK_N))
    _elem_kernel[grid](x, ban, o, x.stride(0), ban.stride(0), o.stride(0), V, BLOCK_N=BLOCK_N, num_warps=4)
    return o
