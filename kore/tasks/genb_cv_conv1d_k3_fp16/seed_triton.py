"""GENERATED breadth cv_conv1d_k3 seed (fp16). Naive same conv1d (K=3, groups-generic) vs torch F.conv1d; one program per (n, cout, L-block), fp32 accumulate over (cin, k). The audio / SSM short conv - scan/fusion headroom. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _cv_conv1d_k3_kernel(x_ptr, w_ptr, b_ptr, y_ptr, Cin, L, Cout, OL, GIN, GOUT,
                 sxn, sxc, sxl, swo, swc, swk, syn, syc, syl,
                 PADL: tl.constexpr, K: tl.constexpr, BLOCK_OL: tl.constexpr):
    row = tl.program_id(0)
    lblk = tl.program_id(1)
    co = row % Cout
    n = row // Cout
    g = co // GOUT
    ol = lblk * BLOCK_OL + tl.arange(0, BLOCK_OL)
    ol_mask = ol < OL
    acc = tl.zeros((BLOCK_OL,), dtype=tl.float32)
    for ci in range(0, GIN):
        cin = g * GIN + ci
        for k in range(0, K):
            il = ol - PADL + k
            m = ol_mask & (il >= 0) & (il < L)
            xv = tl.load(x_ptr + n * sxn + cin * sxc + il * sxl, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + co * swo + ci * swc + k * swk).to(tl.float32)
            acc += xv * wv
    acc += tl.load(b_ptr + co).to(tl.float32)
    tl.store(y_ptr + n * syn + co * syc + ol * syl, acc.to(tl.float16), mask=ol_mask)


def cv_conv1d_k3(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, Cin, L = x.shape
    Cout = weight.shape[0]
    GIN = weight.shape[1]
    GROUPS = Cin // GIN
    GOUT = Cout // GROUPS
    K = weight.shape[2]
    PADL = 1
    OL = L
    y = torch.empty((N, Cout, OL), device=x.device, dtype=x.dtype)
    BLOCK_OL = 128
    grid = (N * Cout, triton.cdiv(OL, BLOCK_OL))
    _cv_conv1d_k3_kernel[grid](x, weight, bias, y, Cin, L, Cout, OL, GIN, GOUT,
                       x.stride(0), x.stride(1), x.stride(2),
                       weight.stride(0), weight.stride(1), weight.stride(2),
                       y.stride(0), y.stride(1), y.stride(2),
                       PADL=PADL, K=K, BLOCK_OL=BLOCK_OL, num_warps=4)
    return y
