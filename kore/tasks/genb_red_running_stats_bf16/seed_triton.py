"""GENERATED breadth red_running_stats seed (bf16). x[M,N] -> (mean, biased var)
over dim 0 (batch) -> [N], [N] (batchnorm-style feature statistics). One program
per column-block; TWO fp32 passes over the rows (stable centered variance). tl.bfloat16."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _red_running_stats_kernel(x_ptr, mean_ptr, var_ptr, sr, sc, M, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < N
    s = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in range(0, M):
        x = tl.load(x_ptr + r * sr + cols * sc, mask=cmask, other=0.0).to(tl.float32)
        s += x
    mean = s / M
    ss = tl.zeros([BLOCK_N], dtype=tl.float32)
    for r in range(0, M):
        x = tl.load(x_ptr + r * sr + cols * sc, mask=cmask, other=0.0).to(tl.float32)
        d = x - mean
        ss += d * d
    tl.store(mean_ptr + cols, mean.to(tl.bfloat16), mask=cmask)
    tl.store(var_ptr + cols, (ss / M).to(tl.bfloat16), mask=cmask)


def red_running_stats(x: torch.Tensor):
    M, N = x.shape
    mean = torch.empty((N,), device=x.device, dtype=x.dtype)
    var = torch.empty((N,), device=x.device, dtype=x.dtype)
    BLOCK_N = 256
    grid = (triton.cdiv(N, BLOCK_N),)
    _red_running_stats_kernel[grid](x, mean, var, x.stride(0), x.stride(1), M, N,
                                    BLOCK_N=BLOCK_N, num_warps=4)
    return mean, var
