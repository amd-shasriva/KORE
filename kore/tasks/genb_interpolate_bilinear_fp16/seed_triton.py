"""GENERATED breadth interpolate_bilinear seed (fp16) vs torch F.interpolate.
2x bilinear upsample, align_corners=False: source coord = 0.5*(dst+0.5)-0.5 (clamped
>=0), 4-neighbor weighted blend. One program per (n, c, oh), output width vectorized,
tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _interpolate_bilinear_kernel(x_ptr, y_ptr, C, H, W, OH, OW,
                                 sxn, sxc, sxh, sxw, syn, syc, syh, syw, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    fh = (oh + 0.5) * 0.5 - 0.5
    fh = tl.maximum(fh, 0.0)
    h0 = fh.to(tl.int32)
    h1 = tl.minimum(h0 + 1, H - 1)
    lh = fh - h0.to(tl.float32)
    fw = (ow.to(tl.float32) + 0.5) * 0.5 - 0.5
    fw = tl.maximum(fw, 0.0)
    w0 = fw.to(tl.int32)
    w1 = tl.minimum(w0 + 1, W - 1)
    lw = fw - w0.to(tl.float32)
    base = n * sxn + c * sxc
    v00 = tl.load(x_ptr + base + h0 * sxh + w0 * sxw, mask=ow_mask, other=0.0).to(tl.float32)
    v01 = tl.load(x_ptr + base + h0 * sxh + w1 * sxw, mask=ow_mask, other=0.0).to(tl.float32)
    v10 = tl.load(x_ptr + base + h1 * sxh + w0 * sxw, mask=ow_mask, other=0.0).to(tl.float32)
    v11 = tl.load(x_ptr + base + h1 * sxh + w1 * sxw, mask=ow_mask, other=0.0).to(tl.float32)
    top = v00 * (1.0 - lw) + v01 * lw
    bot = v10 * (1.0 - lw) + v11 * lw
    out = top * (1.0 - lh) + bot * lh
    y_off = n * syn + c * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, out.to(tl.float16), mask=ow_mask)


def interpolate_bilinear(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    OH, OW = 2 * H, 2 * W
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * C * OH,)
    _interpolate_bilinear_kernel[grid](x, y, C, H, W, OH, OW,
                                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                       y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                                       BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
