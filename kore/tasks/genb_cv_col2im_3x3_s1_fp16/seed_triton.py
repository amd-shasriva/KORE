"""GENERATED breadth cv_col2im_3x3_s1 seed (fp16). Naive col2im / fold (K=3, stride 1, square) vs torch F.fold; one program per (n, c, ih) overlap-adding the columns that map to it. The im2col adjoint. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl

import math


@triton.jit
def _cv_col2im_3x3_s1_kernel(c_ptr, y_ptr, C, H, W, OH, OW, ROWS,
                 scn, scr, scl, syn, syc, syh, syw,
                 PAD: tl.constexpr, DIL: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_W: tl.constexpr):
    pid = tl.program_id(0)
    ih = pid % H
    tmp = pid // H
    c = tmp % C
    n = tmp // C
    iw = tl.arange(0, BLOCK_W)
    iw_mask = iw < W
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for kh in range(0, KH):
        oh = ih + PAD - kh * DIL
        oh_ok = (oh >= 0) & (oh < OH)
        for kw in range(0, KW):
            ow = iw + PAD - kw * DIL
            m = iw_mask & oh_ok & (ow >= 0) & (ow < OW)
            row = (c * KH + kh) * KW + kw
            l = oh * OW + ow
            cv = tl.load(c_ptr + n * scn + row * scr + l * scl, mask=m, other=0.0).to(tl.float32)
            acc += cv
    tl.store(y_ptr + n * syn + c * syc + ih * syh + iw * syw, acc.to(tl.float16), mask=iw_mask)


def cv_col2im_3x3_s1(cols: torch.Tensor) -> torch.Tensor:
    N, ROWS, LTOT = cols.shape
    KH, KW = 3, 3
    PAD, DIL = 1, 1
    C = ROWS // (KH * KW)
    H = int(round(math.sqrt(LTOT)))
    W = H
    OH = (H + 2 * PAD - DIL * (KH - 1) - 1) + 1
    OW = (W + 2 * PAD - DIL * (KW - 1) - 1) + 1
    y = torch.empty((N, C, H, W), device=cols.device, dtype=cols.dtype)
    BLOCK_W = triton.next_power_of_2(W)
    grid = (N * C * H,)
    _cv_col2im_3x3_s1_kernel[grid](cols, y, C, H, W, OH, OW, ROWS,
                       cols.stride(0), cols.stride(1), cols.stride(2),
                       y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                       PAD=PAD, DIL=DIL, KH=KH, KW=KW, BLOCK_W=BLOCK_W, num_warps=4)
    return y
