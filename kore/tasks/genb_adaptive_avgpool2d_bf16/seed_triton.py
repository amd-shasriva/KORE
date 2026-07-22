"""GENERATED breadth adaptive_avgpool2d seed (bf16) vs torch F.adaptive_avg_pool2d.
Output is a fixed 7x7 grid; with H,W divisible by 7 each output cell averages a
contiguous (H//7)x(W//7) window (== the exact adaptive result). One program per
(n, c, oh), output width vectorized, tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _adaptive_avgpool2d_kernel(x_ptr, y_ptr, C, H, W, OH, OW, KH, KW,
                               sxn, sxc, sxh, sxw, syn, syc, syh, syw, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for a in range(0, KH):
        ih = oh * KH + a
        for b in range(0, KW):
            iw = ow * KW + b
            m = ow_mask & (ih < H) & (iw < W)
            xv = tl.load(x_ptr + n * sxn + c * sxc + ih * sxh + iw * sxw,
                         mask=m, other=0.0).to(tl.float32)
            acc += xv
    acc = acc / (KH * KW).to(tl.float32)
    y_off = n * syn + c * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, acc.to(tl.bfloat16), mask=ow_mask)


def adaptive_avgpool2d(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    OH, OW = 7, 7
    KH = H // OH
    KW = W // OW
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * C * OH,)
    _adaptive_avgpool2d_kernel[grid](x, y, C, H, W, OH, OW, KH, KW,
                                     x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                     y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                                     BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
