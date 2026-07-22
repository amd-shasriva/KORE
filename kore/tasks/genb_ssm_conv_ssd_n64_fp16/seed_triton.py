"""GENERATED breadth ssm_conv_ssd_n64 seed (fp16). Mamba conv+SSM: causal depthwise conv
(+SiLU) then a scalar-decay SSD scan. x[B,L,D], conv_w[D,K], a[B,L], B_/C[B,L,N] ->
y[B,L,D]. The conv+SiLU input projection is done in torch; one program per (b,d)
then keeps an fp32 state h[N] and scans over L: h=sigmoid(a)*h + xc*B_; y=sum_n C*h.
The policy fuses the conv into the scan kernel + chunks it. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl
import torch.nn.functional as F


@triton.jit
def _ssm_conv_ssd_n64_kernel(xc_ptr, a_ptr, B_ptr, C_ptr, y_ptr, L, D, N,
                 s_b, s_l, s_d, sa_b, sa_l, sB_b, sB_l, sB_n, NB: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // D
    d = pid % D
    n = tl.arange(0, NB)
    nmask = n < N
    h = tl.zeros([NB], dtype=tl.float32)
    for l in range(0, L):
        dec = tl.sigmoid(tl.load(a_ptr + b * sa_b + l * sa_l).to(tl.float32))
        xoff = b * s_b + l * s_l + d * s_d
        xv = tl.load(xc_ptr + xoff).to(tl.float32)
        boff = b * sB_b + l * sB_l + n * sB_n
        Bv = tl.load(B_ptr + boff, mask=nmask, other=0.0).to(tl.float32)
        Cv = tl.load(C_ptr + boff, mask=nmask, other=0.0).to(tl.float32)
        h = dec * h + xv * Bv
        y_v = tl.sum(tl.where(nmask, Cv * h, 0.0), axis=0)
        tl.store(y_ptr + xoff, y_v.to(tl.float16))


def ssm_conv_ssd_n64(x, conv_w, a, B_, C):
    B, L, D = x.shape
    N = B_.shape[-1]
    K = conv_w.shape[1]
    xt = x.transpose(1, 2).contiguous()
    xc = F.conv1d(F.pad(xt, (K - 1, 0)), conv_w[:, None, :], None, groups=D)
    xc = F.silu(xc).transpose(1, 2).contiguous()
    a = a.contiguous(); B_ = B_.contiguous(); C = C.contiguous()
    y = torch.empty_like(xc)
    NB = triton.next_power_of_2(N)
    _ssm_conv_ssd_n64_kernel[(B * D,)](
        xc, a, B_, C, y, L, D, N,
        xc.stride(0), xc.stride(1), xc.stride(2), a.stride(0), a.stride(1),
        B_.stride(0), B_.stride(1), B_.stride(2), NB=NB, num_warps=1)
    return y
