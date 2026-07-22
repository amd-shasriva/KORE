"""GENERATED breadth cv_conv3d_3x3x3_s1 seed (bf16). Naive direct conv3d (K=3x3x3, S=1) vs torch F.conv3d; one program per (n, cout, od, oh), fp32 accumulate over (cin, kd, kh, kw), output width vectorized. Volumetric implicit-GEMM headroom. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cv_conv3d_3x3x3_s1_kernel(x_ptr, w_ptr, b_ptr, y_ptr, Cin, Din, H, W, Cout, OD, OH, OW,
                 sxn, sxc, sxd, sxh, sxw, swo, swc, swd, swh, sww,
                 syn, syc, syd, syh, syw,
                 STRIDE: tl.constexpr, PAD: tl.constexpr,
                 KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    t = pid // OH
    od = t % OD
    t2 = t // OD
    co = t2 % Cout
    n = t2 // Cout
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for ci in range(0, Cin):
        for kd in range(0, KD):
            idp = od * STRIDE - PAD + kd
            d_ok = (idp >= 0) & (idp < Din)
            for kh in range(0, KH):
                ih = oh * STRIDE - PAD + kh
                h_ok = d_ok & (ih >= 0) & (ih < H)
                for kw in range(0, KW):
                    iw = ow * STRIDE - PAD + kw
                    m = ow_mask & h_ok & (iw >= 0) & (iw < W)
                    xv = tl.load(x_ptr + n * sxn + ci * sxc + idp * sxd + ih * sxh + iw * sxw,
                                 mask=m, other=0.0).to(tl.float32)
                    wv = tl.load(w_ptr + co * swo + ci * swc + kd * swd + kh * swh + kw * sww).to(tl.float32)
                    acc += xv * wv
    acc += tl.load(b_ptr + co).to(tl.float32)
    tl.store(y_ptr + n * syn + co * syc + od * syd + oh * syh + ow * syw,
             acc.to(tl.bfloat16), mask=ow_mask)


def cv_conv3d_3x3x3_s1(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, Cin, Din, H, W = x.shape
    Cout = weight.shape[0]
    KD, KH, KW = weight.shape[2], weight.shape[3], weight.shape[4]
    STRIDE, PAD = 1, 1
    OD = (Din + 2 * PAD - (KD - 1) - 1) // STRIDE + 1
    OH = (H + 2 * PAD - (KH - 1) - 1) // STRIDE + 1
    OW = (W + 2 * PAD - (KW - 1) - 1) // STRIDE + 1
    y = torch.empty((N, Cout, OD, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * Cout * OD * OH,)
    _cv_conv3d_3x3x3_s1_kernel[grid](x, weight, bias, y, Cin, Din, H, W, Cout, OD, OH, OW,
                       x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
                       weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
                       y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
                       STRIDE=STRIDE, PAD=PAD, KD=KD, KH=KH, KW=KW, BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
