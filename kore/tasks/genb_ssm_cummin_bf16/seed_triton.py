"""GENERATED breadth ssm_cummin seed (bf16). Cumulative minimum over the last
dim. One program per flattened row; a sequential running-min scan (naive but
correct). The policy replaces the serial loop with a parallel prefix (min) scan.
tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_cummin_kernel(x_ptr, y_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    acc = 1.0e30
    for i in range(0, L):
        v = tl.load(x_ptr + base + i).to(tl.float32)
        acc = tl.minimum(acc, v)
        tl.store(y_ptr + base + i, acc.to(tl.bfloat16))


def ssm_cummin(x: torch.Tensor) -> torch.Tensor:
    L = x.shape[-1]
    xf = x.contiguous().reshape(-1, L)
    y = torch.empty_like(xf)
    _ssm_cummin_kernel[(xf.shape[0],)](xf, y, L, xf.stride(0), num_warps=1)
    return y.reshape(x.shape)
