"""GENERATED breadth cv_conv_transpose2d_k2_s2 seed (fp16). Naive transposed conv2d / deconv (K=2, S=2) vs torch F.conv_transpose2d; gather form (scatter inverse) - one program per (n, cout, oh) with a stride-modulo source map, fp32 accumulate. Upsampling headroom. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cv_conv_transpose2d_k2_s2_kernel(x_ptr, w_ptr, b_ptr, y_ptr, Cin, H, W, Cout, OH, OW,
                 sxn, sxc, sxh, sxw, swi, swo, swh, sww, syn, syc, syh, syw,
                 STRIDE: tl.constexpr, PAD: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    co = tmp % Cout
    n = tmp // Cout
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for cin in range(0, Cin):
        for kh in range(0, KH):
            numh = oh + PAD - kh
            ih = numh // STRIDE
            h_ok = (numh >= 0) & ((numh % STRIDE) == 0) & (ih < H)
            for kw in range(0, KW):
                numw = ow + PAD - kw
                iw = numw // STRIDE
                m = ow_mask & h_ok & (numw >= 0) & ((numw % STRIDE) == 0) & (iw < W)
                xv = tl.load(x_ptr + n * sxn + cin * sxc + ih * sxh + iw * sxw,
                             mask=m, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + cin * swi + co * swo + kh * swh + kw * sww).to(tl.float32)
                acc += xv * wv
    acc += tl.load(b_ptr + co).to(tl.float32)
    tl.store(y_ptr + n * syn + co * syc + oh * syh + ow * syw, acc.to(tl.float16), mask=ow_mask)


def cv_conv_transpose2d_k2_s2(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, Cin, H, W = x.shape
    Cout = weight.shape[1]
    KH, KW = weight.shape[2], weight.shape[3]
    STRIDE, PAD = 2, 0
    OH = (H - 1) * STRIDE - 2 * PAD + (KH - 1) + 1
    OW = (W - 1) * STRIDE - 2 * PAD + (KW - 1) + 1
    y = torch.empty((N, Cout, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * Cout * OH,)
    _cv_conv_transpose2d_k2_s2_kernel[grid](x, weight, bias, y, Cin, H, W, Cout, OH, OW,
                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                       weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
                       y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                       STRIDE=STRIDE, PAD=PAD, KH=KH, KW=KW, BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
