"""GENERATED breadth cumsum seed (fp16). x[..., L] -> cumulative sum over the last dim.
One program per flattened row; a sequential fp32 running-sum scan over L (naive but
correct). The policy replaces the serial loop with a parallel prefix (Blelloch) scan.
tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cumsum_kernel(x_ptr, y_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    acc = 0.0
    for i in range(0, L):
        v = tl.load(x_ptr + base + i).to(tl.float32)
        acc = acc + v
        tl.store(y_ptr + base + i, acc.to(tl.float16))


def cumsum(x: torch.Tensor) -> torch.Tensor:
    L = x.shape[-1]
    xf = x.contiguous().reshape(-1, L)
    y = torch.empty_like(xf)
    _cumsum_kernel[(xf.shape[0],)](xf, y, L, xf.stride(0), num_warps=1)
    return y.reshape(x.shape)
