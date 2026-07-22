"""GENERATED breadth red_cosine_sim seed (fp32). a[M,N], b[M,N] -> per-row cosine
similarity <a,b>/(||a|| ||b||). Single streaming fp32 pass accumulates the dot and
both squared norms; denominator floored at 1e-08. tl.float32 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_cosine_sim_kernel(a_ptr, b_ptr, o_ptr, sa, sb, N, EPS, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        a = tl.load(a_ptr + row * sa + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + row * sb + offs, mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(tl.where(mask, a * b, 0.0), axis=0)
        na += tl.sum(tl.where(mask, a * a, 0.0), axis=0)
        nb += tl.sum(tl.where(mask, b * b, 0.0), axis=0)
    denom = tl.maximum(tl.sqrt(na) * tl.sqrt(nb), EPS)
    tl.store(o_ptr + row, (dot / denom).to(tl.float32))


def red_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N = a.shape
    o = torch.empty((M,), device=a.device, dtype=a.dtype)
    BLOCK_N = 1024
    _red_cosine_sim_kernel[(M,)](a, b, o, a.stride(0), b.stride(0), N, 1e-08,
                                 BLOCK_N=BLOCK_N, num_warps=8)
    return o
