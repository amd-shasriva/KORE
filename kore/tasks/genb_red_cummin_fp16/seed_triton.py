"""GENERATED breadth red_cummin seed (fp16). x[M,N] -> cumulative min over the
last dim. One program per row; a sequential fp32 running-min scan (naive but
correct; the policy replaces the serial loop with a parallel prefix scan). tl.float16."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_cummin_kernel(x_ptr, o_ptr, sx, so, N):
    row = tl.program_id(0)
    run = float("inf")
    for i in range(0, N):
        v = tl.load(x_ptr + row * sx + i).to(tl.float32)
        run = tl.minimum(run, v)
        tl.store(o_ptr + row * so + i, run.to(tl.float16))


def red_cummin(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    _red_cummin_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, num_warps=1)
    return o
