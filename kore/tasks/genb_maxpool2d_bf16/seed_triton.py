"""GENERATED breadth maxpool2d seed (bf16) vs torch F.max_pool2d(2).
2x2 stride-2 max pool: one program per (n, c, oh), max over the 2x2 window across a
vectorized output width, tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _maxpool2d_kernel(x_ptr, y_ptr, C, H, W, OH, OW,
                      sxn, sxc, sxh, sxw, syn, syc, syh, syw, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32) - 1e38
    for kh in range(0, 2):
        ih = oh * 2 + kh
        for kw in range(0, 2):
            iw = ow * 2 + kw
            m = ow_mask & (ih < H) & (iw < W)
            xv = tl.load(x_ptr + n * sxn + c * sxc + ih * sxh + iw * sxw,
                         mask=m, other=-1e38).to(tl.float32)
            acc = tl.maximum(acc, xv)
    y_off = n * syn + c * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, acc.to(tl.bfloat16), mask=ow_mask)


def maxpool2d(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    OH = (H - 2) // 2 + 1
    OW = (W - 2) // 2 + 1
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * C * OH,)
    _maxpool2d_kernel[grid](x, y, C, H, W, OH, OW,
                            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                            BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
