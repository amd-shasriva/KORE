"""GENERATED breadth cv_separable_conv2d_3x3 seed (bf16). Naive depthwise-separable conv2d = depthwise KxK (groups=C) then pointwise 1x1, as two chained kernels vs torch two-conv baseline; the policy fuses the two passes into one. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cv_separable_conv2d_3x3_dw_kernel(x_ptr, w_ptr, t_ptr, C, H, W,
                    sxn, sxc, sxh, sxw, swo, swh, sww, stn, stc, sth, stw,
                    PAD: tl.constexpr, K: tl.constexpr, BLOCK_W: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % H
    tmp = pid // H
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_W)
    ow_mask = ow < W
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for kh in range(0, K):
        ih = oh - PAD + kh
        h_ok = (ih >= 0) & (ih < H)
        for kw in range(0, K):
            iw = ow - PAD + kw
            m = ow_mask & h_ok & (iw >= 0) & (iw < W)
            xv = tl.load(x_ptr + n * sxn + c * sxc + ih * sxh + iw * sxw,
                         mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + c * swo + kh * swh + kw * sww).to(tl.float32)
            acc += xv * wv
    tl.store(t_ptr + n * stn + c * stc + oh * sth + ow * stw, acc.to(tl.bfloat16), mask=ow_mask)


@triton.jit
def _cv_separable_conv2d_3x3_pw_kernel(t_ptr, w_ptr, b_ptr, y_ptr, C, Cout, H, W,
                    stn, stc, sth, stw, swo, swc, syn, syc, syh, syw,
                    BLOCK_W: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % H
    tmp = pid // H
    co = tmp % Cout
    n = tmp // Cout
    ow = tl.arange(0, BLOCK_W)
    ow_mask = ow < W
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for c in range(0, C):
        tv = tl.load(t_ptr + n * stn + c * stc + oh * sth + ow * stw,
                     mask=ow_mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + co * swo + c * swc).to(tl.float32)
        acc += tv * wv
    acc += tl.load(b_ptr + co).to(tl.float32)
    tl.store(y_ptr + n * syn + co * syc + oh * syh + ow * syw, acc.to(tl.bfloat16), mask=ow_mask)


def cv_separable_conv2d_3x3(x: torch.Tensor, dw: torch.Tensor, pw: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    Cout = pw.shape[0]
    PAD, K = 1, 3
    t = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)
    BLOCK_W = triton.next_power_of_2(W)
    _cv_separable_conv2d_3x3_dw_kernel[(N * C * H,)](x, dw, t, C, H, W,
                                  x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                  dw.stride(0), dw.stride(2), dw.stride(3),
                                  t.stride(0), t.stride(1), t.stride(2), t.stride(3),
                                  PAD=PAD, K=K, BLOCK_W=BLOCK_W, num_warps=4)
    y = torch.empty((N, Cout, H, W), device=x.device, dtype=x.dtype)
    _cv_separable_conv2d_3x3_pw_kernel[(N * Cout * H,)](t, pw, bias, y, C, Cout, H, W,
                                     t.stride(0), t.stride(1), t.stride(2), t.stride(3),
                                     pw.stride(0), pw.stride(1),
                                     y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                                     BLOCK_W=BLOCK_W, num_warps=4)
    return y
