"""GENERATED breadth ssd_chunk_scan seed (bf16). Simplified Mamba-2 SSD scalar-decay scan.
x[B,L,D], a[B,L] (scalar decay), B_/C[B,L,N] -> y[B,L,D]. One program per (b, d) keeps an
fp32 state h[N] and scans over L: h = a*h + x*B_ ; y = sum_n C*h. Naive sequential scan; a
real SSD kernel processes time in chunks (matmul the intra-chunk term). tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssd_chunk_scan_kernel(x_ptr, a_ptr, B_ptr, C_ptr, y_ptr, L, D, N,
                           sx_b, sx_l, sx_d, sa_b, sa_l, sB_b, sB_l, sB_n,
                           NB: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D
    n = tl.arange(0, NB)
    nmask = n < N
    h = tl.zeros([NB], dtype=tl.float32)
    for l in range(0, L):
        a_v = tl.load(a_ptr + b * sa_b + l * sa_l).to(tl.float32)
        off_ld = b * sx_b + l * sx_l + d * sx_d
        x_v = tl.load(x_ptr + off_ld).to(tl.float32)
        off_ln = b * sB_b + l * sB_l + n * sB_n
        Bv = tl.load(B_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        Cv = tl.load(C_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        h = a_v * h + x_v * Bv
        y_v = tl.sum(tl.where(nmask, Cv * h, 0.0), axis=0)
        tl.store(y_ptr + off_ld, y_v.to(tl.bfloat16))


def ssd_chunk_scan(x, a, B_, C):
    Bsz, L, D = x.shape
    N = B_.shape[-1]
    x = x.contiguous(); a = a.contiguous(); B_ = B_.contiguous(); C = C.contiguous()
    y = torch.empty_like(x)
    NB = triton.next_power_of_2(N)
    _ssd_chunk_scan_kernel[(Bsz * D,)](
        x, a, B_, C, y, L, D, N,
        x.stride(0), x.stride(1), x.stride(2),
        a.stride(0), a.stride(1),
        B_.stride(0), B_.stride(1), B_.stride(2), NB=NB, num_warps=1)
    return y
