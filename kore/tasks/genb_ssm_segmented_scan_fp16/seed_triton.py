"""GENERATED breadth ssm_segmented_scan seed (fp16). Blelloch segmented
cumulative sum: reset flag r_t==1 starts a new segment. Gated recurrence
h_t = (1 - r_t) * h_{t-1} + x_t over the last dim. One program per flattened
row; sequential fp32 scan (the associative segmented operator makes this a
parallel prefix scan the policy builds). Inputs x[...,L], reset[...,L]. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_segmented_scan_kernel(x_ptr, r_ptr, h_ptr, L, srow):
    row = tl.program_id(0)
    base = row * srow
    h = 0.0
    for i in range(0, L):
        xv = tl.load(x_ptr + base + i).to(tl.float32)
        rv = tl.load(r_ptr + base + i).to(tl.float32)
        h = (1.0 - rv) * h + xv
        tl.store(h_ptr + base + i, h.to(tl.float16))


def ssm_segmented_scan(x: torch.Tensor, reset: torch.Tensor) -> torch.Tensor:
    L = x.shape[-1]
    xf = x.contiguous().reshape(-1, L)
    rf = reset.contiguous().reshape(-1, L)
    h = torch.empty_like(xf)
    _ssm_segmented_scan_kernel[(xf.shape[0],)](xf, rf, h, L, xf.stride(0), num_warps=1)
    return h.reshape(x.shape)
