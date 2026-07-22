"""GENERATED breadth cv_conv2d_dgrad_1x1 seed (fp16). Naive conv2d dInput / dgrad (K=1, stride 1) vs torch.nn.grad.conv2d_input; one program per (n, cin, ih) accumulating grad_y * weight over (cout, kh, kw). The hard training backward kernel. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cv_conv2d_dgrad_1x1_kernel(gy_ptr, w_ptr, gx_ptr, Cin, H, W, Cout, OH, OW,
                 sgn, sgc, sgh, sgw, swo, swc, swh, sww, sxn, sxc, sxh, sxw,
                 PAD: tl.constexpr, DIL: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_W: tl.constexpr):
    pid = tl.program_id(0)
    ih = pid % H
    tmp = pid // H
    cin = tmp % Cin
    n = tmp // Cin
    iw = tl.arange(0, BLOCK_W)
    iw_mask = iw < W
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for co in range(0, Cout):
        for kh in range(0, KH):
            oh = ih + PAD - kh * DIL
            oh_ok = (oh >= 0) & (oh < OH)
            for kw in range(0, KW):
                ow = iw + PAD - kw * DIL
                m = iw_mask & oh_ok & (ow >= 0) & (ow < OW)
                gv = tl.load(gy_ptr + n * sgn + co * sgc + oh * sgh + ow * sgw,
                             mask=m, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + co * swo + cin * swc + kh * swh + kw * sww).to(tl.float32)
                acc += gv * wv
    tl.store(gx_ptr + n * sxn + cin * sxc + ih * sxh + iw * sxw, acc.to(tl.float16), mask=iw_mask)


def cv_conv2d_dgrad_1x1(x: torch.Tensor, grad_y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    N, Cin, H, W = x.shape
    Cout = weight.shape[0]
    KH, KW = weight.shape[2], weight.shape[3]
    OH, OW = grad_y.shape[2], grad_y.shape[3]
    PAD, DIL = 0, 1
    gx = torch.empty((N, Cin, H, W), device=x.device, dtype=x.dtype)
    BLOCK_W = triton.next_power_of_2(W)
    grid = (N * Cin * H,)
    _cv_conv2d_dgrad_1x1_kernel[grid](grad_y, weight, gx, Cin, H, W, Cout, OH, OW,
                       grad_y.stride(0), grad_y.stride(1), grad_y.stride(2), grad_y.stride(3),
                       weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
                       gx.stride(0), gx.stride(1), gx.stride(2), gx.stride(3),
                       PAD=PAD, DIL=DIL, KH=KH, KW=KW, BLOCK_W=BLOCK_W, num_warps=4)
    return gx
