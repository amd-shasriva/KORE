"""GENERATED breadth ssm_s4d seed (fp16). S4D diagonal LTI SSM. u[B,D,L],
Abar/Bbar/C[D,N] -> y[B,D,L]. One program per (b, d) keeps an fp32 diagonal state
h[N] and scans over L: a=exp(Abar); h = a*h + Bbar*u; y = sum_n C*h. Naive
sequential scan; the policy exploits time-invariance (long convolution / chunked
scan). tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_s4d_kernel(u_ptr, Abar_ptr, Bbar_ptr, C_ptr, y_ptr, D, L, N,
                    su_b, su_d, su_l, sA_d, sA_n, NB: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D
    n = tl.arange(0, NB)
    nmask = n < N
    Aoff = d * sA_d + n * sA_n
    a = tl.exp(tl.load(Abar_ptr + Aoff, mask=nmask, other=0.0).to(tl.float32))
    Bb = tl.load(Bbar_ptr + Aoff, mask=nmask, other=0.0).to(tl.float32)
    Cc = tl.load(C_ptr + Aoff, mask=nmask, other=0.0).to(tl.float32)
    h = tl.zeros([NB], dtype=tl.float32)
    base = b * su_b + d * su_d
    for l in range(0, L):
        uv = tl.load(u_ptr + base + l * su_l).to(tl.float32)
        h = a * h + Bb * uv
        y_v = tl.sum(tl.where(nmask, Cc * h, 0.0), axis=0)
        tl.store(y_ptr + base + l * su_l, y_v.to(tl.float16))


def ssm_s4d(u, Abar, Bbar, C):
    Bsz, D, L = u.shape
    N = Abar.shape[1]
    u = u.contiguous(); Abar = Abar.contiguous(); Bbar = Bbar.contiguous(); C = C.contiguous()
    y = torch.empty_like(u)
    NB = triton.next_power_of_2(N)
    _ssm_s4d_kernel[(Bsz * D,)](
        u, Abar, Bbar, C, y, D, L, N,
        u.stride(0), u.stride(1), u.stride(2),
        Abar.stride(0), Abar.stride(1), NB=NB, num_warps=1)
    return y
