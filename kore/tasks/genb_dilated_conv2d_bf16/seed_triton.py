"""GENERATED breadth dilated_conv2d seed (bf16) vs torch F.conv2d.
Naive direct convolution: one program per (n, cout, oh) output row; fp32 accumulate
over (cin, kh, kw) with the output width vectorized. Correct-but-slow starting point
the KORE policy optimizes against the torch baseline. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _dilated_conv2d_kernel(x_ptr, w_ptr, b_ptr, y_ptr, Cin, H, W, Cout, OH, OW,
                 sxn, sxc, sxh, sxw, swo, swc, swh, sww, syn, syc, syh, syw,
                 STRIDE: tl.constexpr, PAD: tl.constexpr, DIL: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    co = tmp % Cout
    n = tmp // Cout
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for ci in range(0, Cin):
        for kh in range(0, KH):
            ih = oh * STRIDE - PAD + kh * DIL
            h_ok = (ih >= 0) & (ih < H)
            for kw in range(0, KW):
                iw = ow * STRIDE - PAD + kw * DIL
                m = ow_mask & h_ok & (iw >= 0) & (iw < W)
                xv = tl.load(x_ptr + n * sxn + ci * sxc + ih * sxh + iw * sxw,
                             mask=m, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + co * swo + ci * swc + kh * swh + kw * sww).to(tl.float32)
                acc += xv * wv
    acc += tl.load(b_ptr + co).to(tl.float32)
    y_off = n * syn + co * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, acc.to(tl.bfloat16), mask=ow_mask)


def dilated_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape
    STRIDE, PAD, DIL = 1, 2, 2
    OH = (H + 2 * PAD - DIL * (KH - 1) - 1) // STRIDE + 1
    OW = (W + 2 * PAD - DIL * (KW - 1) - 1) // STRIDE + 1
    y = torch.empty((N, Cout, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * Cout * OH,)
    _dilated_conv2d_kernel[grid](x, weight, bias, y, Cin, H, W, Cout, OH, OW,
                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                       weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
                       y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                       STRIDE=STRIDE, PAD=PAD, DIL=DIL, KH=KH, KW=KW,
                       BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
