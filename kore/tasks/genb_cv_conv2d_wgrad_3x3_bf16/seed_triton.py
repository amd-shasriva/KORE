"""GENERATED breadth cv_conv2d_wgrad_3x3 seed (bf16). Naive conv2d dWeight / wgrad (K=3, stride 1) vs torch.nn.grad.conv2d_weight; one program per (cout, cin) reducing grad_y * x over (n, oh, ow) per tap. The hard training backward kernel. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cv_conv2d_wgrad_3x3_kernel(x_ptr, gy_ptr, gw_ptr, N, Cin, H, W, Cout, OH, OW,
                 sxn, sxc, sxh, sxw, sgn, sgc, sgh, sgw, swo, swc, swh, sww,
                 PAD: tl.constexpr, DIL: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    cin = pid % Cin
    co = pid // Cin
    for kh in range(0, KH):
        for kw in range(0, KW):
            acc = 0.0
            for n in range(0, N):
                for oh in range(0, OH):
                    ow = tl.arange(0, BLOCK_OW)
                    ow_mask = ow < OW
                    ih = oh - PAD + kh * DIL
                    ih_ok = (ih >= 0) & (ih < H)
                    iw = ow - PAD + kw * DIL
                    m = ow_mask & ih_ok & (iw >= 0) & (iw < W)
                    gv = tl.load(gy_ptr + n * sgn + co * sgc + oh * sgh + ow * sgw,
                                 mask=ow_mask, other=0.0).to(tl.float32)
                    xv = tl.load(x_ptr + n * sxn + cin * sxc + ih * sxh + iw * sxw,
                                 mask=m, other=0.0).to(tl.float32)
                    acc += tl.sum(gv * xv, axis=0)
            tl.store(gw_ptr + co * swo + cin * swc + kh * swh + kw * sww, acc.to(tl.bfloat16))


def cv_conv2d_wgrad_3x3(x: torch.Tensor, grad_y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    N, Cin, H, W = x.shape
    Cout = weight.shape[0]
    KH, KW = weight.shape[2], weight.shape[3]
    OH, OW = grad_y.shape[2], grad_y.shape[3]
    PAD, DIL = 1, 1
    gw = torch.empty((Cout, Cin, KH, KW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (Cout * Cin,)
    _cv_conv2d_wgrad_3x3_kernel[grid](x, grad_y, gw, N, Cin, H, W, Cout, OH, OW,
                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                       grad_y.stride(0), grad_y.stride(1), grad_y.stride(2), grad_y.stride(3),
                       gw.stride(0), gw.stride(1), gw.stride(2), gw.stride(3),
                       PAD=PAD, DIL=DIL, KH=KH, KW=KW, BLOCK_OW=BLOCK_OW, num_warps=4)
    return gw
