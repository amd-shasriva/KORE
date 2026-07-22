"""GENERATED breadth ssm_mamba2_ssd_c256_n128 seed (fp16). Mamba-2 SSD, multi-head scalar decay.
x[B,L,H,P], dt[B,L,H], A[H], B_/C[B,L,H,N] -> y[B,L,H,P]. One program per (b,h,p)
keeps an fp32 state h[N] and scans over L: dt=softplus; a=exp(dt*A); h=a*h+(dt*x)*B_;
y=sum_n C*h. Naive sequential scan; a real SSD kernel processes time in chunks
(matmul the intra-chunk term). tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_mamba2_ssd_c256_n128_kernel(x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, y_ptr, L, H, P, N,
                 sx_b, sx_l, sx_h, sx_p, sdt_b, sdt_l, sdt_h, sA_h,
                 sB_b, sB_l, sB_h, sB_n, NB: tl.constexpr):
    pid = tl.program_id(0)
    p = pid % P
    tmp = pid // P
    hh = tmp % H
    b = tmp // H
    n = tl.arange(0, NB)
    nmask = n < N
    A_h = tl.load(A_ptr + hh * sA_h).to(tl.float32)
    state = tl.zeros([NB], dtype=tl.float32)
    for l in range(0, L):
        dt = tl.load(dt_ptr + b * sdt_b + l * sdt_l + hh * sdt_h).to(tl.float32)
        dt = tl.where(dt > 20.0, dt, tl.log(1.0 + tl.exp(dt)))
        a = tl.exp(dt * A_h)
        xoff = b * sx_b + l * sx_l + hh * sx_h + p * sx_p
        xv = tl.load(x_ptr + xoff).to(tl.float32)
        boff = b * sB_b + l * sB_l + hh * sB_h + n * sB_n
        Bv = tl.load(B_ptr + boff, mask=nmask, other=0.0).to(tl.float32)
        Cv = tl.load(C_ptr + boff, mask=nmask, other=0.0).to(tl.float32)
        state = a * state + (dt * xv) * Bv
        y_v = tl.sum(tl.where(nmask, Cv * state, 0.0), axis=0)
        tl.store(y_ptr + xoff, y_v.to(tl.float16))


def ssm_mamba2_ssd_c256_n128(x, dt, A, B_, C):
    B, L, H, P = x.shape
    N = B_.shape[-1]
    x = x.contiguous(); dt = dt.contiguous(); A = A.contiguous()
    B_ = B_.contiguous(); C = C.contiguous()
    y = torch.empty_like(x)
    NB = triton.next_power_of_2(N)
    _ssm_mamba2_ssd_c256_n128_kernel[(B * H * P,)](
        x, dt, A, B_, C, y, L, H, P, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        dt.stride(0), dt.stride(1), dt.stride(2), A.stride(0),
        B_.stride(0), B_.stride(1), B_.stride(2), B_.stride(3),
        NB=NB, num_warps=1)
    return y
