"""GENERATED breadth red_logcumsumexp seed (fp16). x[M,N] -> cumulative
log-sum-exp over the last dim. One program per row; a sequential fp32 running
(max, rescaled-sum) scan (numerically stable, naive but correct; the policy
replaces the serial loop with a parallel prefix scan). tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_logcumsumexp_kernel(x_ptr, o_ptr, sx, so, N):
    row = tl.program_id(0)
    run_m = -float("inf")
    run_s = 0.0
    for i in range(0, N):
        v = tl.load(x_ptr + row * sx + i).to(tl.float32)
        new_m = tl.maximum(run_m, v)
        run_s = run_s * tl.exp(run_m - new_m) + tl.exp(v - new_m)
        run_m = new_m
        tl.store(o_ptr + row * so + i, (run_m + tl.log(run_s)).to(tl.float16))


def red_logcumsumexp(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    o = torch.empty_like(x)
    _red_logcumsumexp_kernel[(M,)](x, o, x.stride(0), o.stride(0), N, num_warps=1)
    return o
