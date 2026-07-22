"""GENERATED breadth cv_im2col_3x3_s1 seed (fp16). Naive im2col / unfold (K=3, S=1) vs torch F.unfold; one program per (n, cin*kh*kw row, L-block) gathering the padded patch. The implicit-GEMM lowering primitive. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cv_im2col_3x3_s1_kernel(x_ptr, y_ptr, C, H, W, OH, OW, ROWS, LTOT,
                 sxn, sxc, sxh, sxw, syn, syr, syl,
                 STRIDE: tl.constexpr, PAD: tl.constexpr, DIL: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_L: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    row = pid0 % ROWS
    n = pid0 // ROWS
    kw = row % KW
    r2 = row // KW
    kh = r2 % KH
    cin = r2 // KH
    l = pid1 * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = l < LTOT
    oh = l // OW
    ow = l % OW
    ih = oh * STRIDE - PAD + kh * DIL
    iw = ow * STRIDE - PAD + kw * DIL
    m = l_mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    xv = tl.load(x_ptr + n * sxn + cin * sxc + ih * sxh + iw * sxw, mask=m, other=0.0)
    tl.store(y_ptr + n * syn + row * syr + l * syl, xv.to(tl.float16), mask=l_mask)


def cv_im2col_3x3_s1(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    KH, KW = 3, 3
    STRIDE, PAD, DIL = 1, 1, 1
    OH = (H + 2 * PAD - DIL * (KH - 1) - 1) // STRIDE + 1
    OW = (W + 2 * PAD - DIL * (KW - 1) - 1) // STRIDE + 1
    ROWS = C * KH * KW
    LTOT = OH * OW
    y = torch.empty((N, ROWS, LTOT), device=x.device, dtype=x.dtype)
    BLOCK_L = 128
    grid = (N * ROWS, triton.cdiv(LTOT, BLOCK_L))
    _cv_im2col_3x3_s1_kernel[grid](x, y, C, H, W, OH, OW, ROWS, LTOT,
                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                       y.stride(0), y.stride(1), y.stride(2),
                       STRIDE=STRIDE, PAD=PAD, DIL=DIL, KH=KH, KW=KW,
                       BLOCK_L=BLOCK_L, num_warps=4)
    return y
