"""GENERATED breadth selective_scan seed (bf16). Mamba-1 selective SSM core.
u/delta[B,L,D], A[D,N], B_/C[B,L,N], D_[D] -> y[B,L,D]. One program per (b, d) keeps
an fp32 state h[N] and scans over L: dt=softplus(delta); dA=exp(dt*A); dBu=dt*B_*u;
h=dA*h+dBu; y=sum_n C*h + D_*u. Naive sequential scan (the policy fuses/chunks it). tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _selective_scan_kernel(u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, Dskip_ptr, y_ptr,
                           L, D, N,
                           su_b, su_l, su_d, sA_d, sA_n, sB_b, sB_l, sB_n, sDs,
                           NB: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D
    n = tl.arange(0, NB)
    nmask = n < N
    Arow = tl.load(A_ptr + d * sA_d + n * sA_n, mask=nmask, other=0.0).to(tl.float32)
    Dd = tl.load(Dskip_ptr + d * sDs).to(tl.float32)
    h = tl.zeros([NB], dtype=tl.float32)
    for l in range(0, L):
        off_ld = b * su_b + l * su_l + d * su_d
        u_v = tl.load(u_ptr + off_ld).to(tl.float32)
        dt = tl.load(delta_ptr + off_ld).to(tl.float32)
        dt = tl.where(dt > 20.0, dt, tl.log(1.0 + tl.exp(dt)))   # softplus
        off_ln = b * sB_b + l * sB_l + n * sB_n
        Bv = tl.load(B_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        Cv = tl.load(C_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        dA = tl.exp(dt * Arow)
        dBu = dt * Bv * u_v
        h = dA * h + dBu
        y_v = tl.sum(tl.where(nmask, Cv * h, 0.0), axis=0) + Dd * u_v
        tl.store(y_ptr + off_ld, y_v.to(tl.bfloat16))


def selective_scan(u, delta, A, B_, C, D_):
    Bsz, L, D = u.shape
    N = A.shape[1]
    u = u.contiguous(); delta = delta.contiguous(); A = A.contiguous()
    B_ = B_.contiguous(); C = C.contiguous(); D_ = D_.contiguous()
    y = torch.empty_like(u)
    NB = triton.next_power_of_2(N)
    _selective_scan_kernel[(Bsz * D,)](
        u, delta, A, B_, C, D_, y, L, D, N,
        u.stride(0), u.stride(1), u.stride(2),
        A.stride(0), A.stride(1),
        B_.stride(0), B_.stride(1), B_.stride(2),
        D_.stride(0), NB=NB, num_warps=1)
    return y
