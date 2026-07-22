"""GENERATED breadth ssm_conv_selective seed (bf16). Mamba conv+SSM: causal depthwise conv
(+SiLU) then the Mamba-1 selective scan. u[B,L,D], conv_w[D,K], delta[B,L,D],
A[D,N], B_/C[B,L,N] -> y[B,L,D]. The conv+SiLU projection is done in torch; one
program per (b,d) then scans over L: dt=softplus(delta); h=exp(dt*A)*h+(dt*B_)*uc;
y=sum_n C*h. The policy fuses the conv into the scan + chunks it. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl
import torch.nn.functional as F


@triton.jit
def _ssm_conv_selective_kernel(uc_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, y_ptr, L, D, N,
                 su_b, su_l, su_d, sA_d, sA_n, sB_b, sB_l, sB_n, NB: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D
    n = tl.arange(0, NB)
    nmask = n < N
    Arow = tl.load(A_ptr + d * sA_d + n * sA_n, mask=nmask, other=0.0).to(tl.float32)
    h = tl.zeros([NB], dtype=tl.float32)
    for l in range(0, L):
        off_ld = b * su_b + l * su_l + d * su_d
        uv = tl.load(uc_ptr + off_ld).to(tl.float32)
        dt = tl.load(delta_ptr + off_ld).to(tl.float32)
        dt = tl.where(dt > 20.0, dt, tl.log(1.0 + tl.exp(dt)))
        off_ln = b * sB_b + l * sB_l + n * sB_n
        Bv = tl.load(B_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        Cv = tl.load(C_ptr + off_ln, mask=nmask, other=0.0).to(tl.float32)
        h = tl.exp(dt * Arow) * h + (dt * Bv) * uv
        y_v = tl.sum(tl.where(nmask, Cv * h, 0.0), axis=0)
        tl.store(y_ptr + off_ld, y_v.to(tl.bfloat16))


def ssm_conv_selective(u, conv_w, delta, A, B_, C):
    B, L, D = u.shape
    N = A.shape[1]
    K = conv_w.shape[1]
    ut = u.transpose(1, 2).contiguous()
    uc = F.conv1d(F.pad(ut, (K - 1, 0)), conv_w[:, None, :], None, groups=D)
    uc = F.silu(uc).transpose(1, 2).contiguous()
    delta = delta.contiguous(); A = A.contiguous(); B_ = B_.contiguous(); C = C.contiguous()
    y = torch.empty_like(uc)
    NB = triton.next_power_of_2(N)
    _ssm_conv_selective_kernel[(B * D,)](
        uc, delta, A, B_, C, y, L, D, N,
        uc.stride(0), uc.stride(1), uc.stride(2), A.stride(0), A.stride(1),
        B_.stride(0), B_.stride(1), B_.stride(2), NB=NB, num_warps=1)
    return y
