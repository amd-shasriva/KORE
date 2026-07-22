"""GENERATED breadth depthwise_conv2d seed (bf16) vs torch F.conv2d(groups=C).
Naive depthwise conv: one program per (n, c, oh); each output channel convolves only
its own input channel over (kh, kw). fp32 accumulate, output width vectorized, tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _depthwise_conv2d_kernel(x_ptr, w_ptr, b_ptr, y_ptr, C, H, W, OH, OW,
                             sxn, sxc, sxh, sxw, swo, swh, sww, syn, syc, syh, syw,
                             STRIDE: tl.constexpr, PAD: tl.constexpr, DIL: tl.constexpr,
                             KH: tl.constexpr, KW: tl.constexpr, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for kh in range(0, KH):
        ih = oh * STRIDE - PAD + kh * DIL
        h_ok = (ih >= 0) & (ih < H)
        for kw in range(0, KW):
            iw = ow * STRIDE - PAD + kw * DIL
            m = ow_mask & h_ok & (iw >= 0) & (iw < W)
            xv = tl.load(x_ptr + n * sxn + c * sxc + ih * sxh + iw * sxw,
                         mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + c * swo + kh * swh + kw * sww).to(tl.float32)
            acc += xv * wv
    acc += tl.load(b_ptr + c).to(tl.float32)
    y_off = n * syn + c * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, acc.to(tl.bfloat16), mask=ow_mask)


def depthwise_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    KH, KW = weight.shape[2], weight.shape[3]
    STRIDE, PAD, DIL = 1, 1, 1
    OH = (H + 2 * PAD - DIL * (KH - 1) - 1) // STRIDE + 1
    OW = (W + 2 * PAD - DIL * (KW - 1) - 1) // STRIDE + 1
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * C * OH,)
    _depthwise_conv2d_kernel[grid](x, weight, bias, y, C, H, W, OH, OW,
                                   x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                   weight.stride(0), weight.stride(2), weight.stride(3),
                                   y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                                   STRIDE=STRIDE, PAD=PAD, DIL=DIL, KH=KH, KW=KW,
                                   BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
