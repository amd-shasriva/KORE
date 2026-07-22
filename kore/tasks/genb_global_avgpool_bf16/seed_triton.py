"""GENERATED breadth global_avgpool seed (bf16) vs torch global mean.
Global average over spatial dims: one program per (n, c) row reduces all H*W elements
in BLOCK-wide chunks (fp32 accumulate), output [N, C], tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _global_avgpool_kernel(x_ptr, y_ptr, HW, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, HW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HW
        xv = tl.load(x_ptr + row * HW + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(xv, axis=0)
    tl.store(y_ptr + row, (acc / HW.to(tl.float32)).to(tl.bfloat16))


def global_avgpool(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    HW = H * W
    y = torch.empty((N, C), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    _global_avgpool_kernel[grid](x.contiguous(), y, HW, BLOCK=1024, num_warps=4)
    return y
