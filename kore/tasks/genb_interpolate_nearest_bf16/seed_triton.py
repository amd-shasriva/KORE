"""GENERATED breadth interpolate_nearest seed (bf16) vs torch F.interpolate.
2x nearest upsample: source index = dst // 2. One program per (n, c, oh), output width
vectorized gather, tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _interpolate_nearest_kernel(x_ptr, y_ptr, C, H, W, OH, OW,
                                sxn, sxc, sxh, sxw, syn, syc, syh, syw, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    ih = oh // 2
    iw = ow // 2
    base = n * sxn + c * sxc
    xv = tl.load(x_ptr + base + ih * sxh + iw * sxw, mask=ow_mask, other=0.0)
    y_off = n * syn + c * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, xv.to(tl.bfloat16), mask=ow_mask)


def interpolate_nearest(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    OH, OW = 2 * H, 2 * W
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * C * OH,)
    _interpolate_nearest_kernel[grid](x, y, C, H, W, OH, OW,
                                      x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                      y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                                      BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
