"""GENERATED breadth ssm_logcumsumexp seed (fp16). Stable log-cumsum-exp over
the last dim. One program per flattened row; a streaming (running max m, running
sum s) fp32 scan. The policy replaces the serial loop with a parallel prefix
(log-space) scan. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_logcumsumexp_kernel(x_ptr, y_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    m = -1.0e30
    s = 0.0
    for i in range(0, L):
        v = tl.load(x_ptr + base + i).to(tl.float32)
        mn = tl.maximum(m, v)
        s = s * tl.exp(m - mn) + tl.exp(v - mn)
        m = mn
        tl.store(y_ptr + base + i, (m + tl.log(s)).to(tl.float16))


def ssm_logcumsumexp(x: torch.Tensor) -> torch.Tensor:
    L = x.shape[-1]
    xf = x.contiguous().reshape(-1, L)
    y = torch.empty_like(xf)
    _ssm_logcumsumexp_kernel[(xf.shape[0],)](xf, y, L, xf.stride(0), num_warps=1)
    return y.reshape(x.shape)
