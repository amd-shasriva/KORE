"""GENERATED breadth ssm_conv_ssd_n128 seed (fp16). Mamba conv+SSM: causal depthwise conv
(+SiLU) then a scalar-decay SSD scan. x[B,L,D], conv_w[D,K], a[B,L], B_/C[B,L,N] ->
y[B,L,D]. A Triton causal-convolution kernel materializes the SiLU projection; one
scan program per (b,d) then keeps an fp32 state h[N] over L. The policy fuses the
two honest Triton stages and chunks the recurrence. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_conv_ssd_n128_kernel(xc_ptr, a_ptr, B_ptr, C_ptr, y_ptr, L, D, N,
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


def ssm_conv_ssd_n128(x, conv_w, a, B_, C):
    B, L, D = x.shape
    N = B_.shape[-1]
    xc = _causal_conv_silu(x, conv_w)
    a = a.contiguous(); B_ = B_.contiguous(); C = C.contiguous()
    y = torch.empty_like(xc)
    NB = triton.next_power_of_2(N)
    _ssm_conv_ssd_n128_kernel[(B * D,)](
        xc, a, B_, C, y, L, D, N,
        xc.stride(0), xc.stride(1), xc.stride(2), a.stride(0), a.stride(1),
        B_.stride(0), B_.stride(1), B_.stride(2), NB=NB, num_warps=1)
    return y


@triton.jit
def _causal_conv_silu_kernel(x_ptr, w_ptr, out_ptr, L, D,
                             sx_b, sx_l, sx_d, sw_d, sw_k,
                             K: tl.constexpr, BLOCK_D: tl.constexpr):
    bl = tl.program_id(0)
    b = bl // L
    pos = bl % L
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = d < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in range(0, K):
        src_pos = pos - (K - 1) + k
        valid = dmask & (src_pos >= 0)
        xv = tl.load(x_ptr + b * sx_b + src_pos * sx_l + d * sx_d,
                     mask=valid, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + d * sw_d + k * sw_k,
                     mask=dmask, other=0.0).to(tl.float32)
        acc += xv * wv
    acc = acc * tl.sigmoid(acc)
    tl.store(out_ptr + b * sx_b + pos * sx_l + d * sx_d,
             acc.to(out_ptr.dtype.element_ty), mask=dmask)


def _causal_conv_silu(x, conv_w):
    B, L, D = x.shape
    K = conv_w.shape[1]
    x = x.contiguous()
    conv_w = conv_w.contiguous()
    out = torch.empty_like(x)
    BLOCK_D = 256
    _causal_conv_silu_kernel[(B * L, triton.cdiv(D, BLOCK_D))](
        x, conv_w, out, L, D,
        x.stride(0), x.stride(1), x.stride(2),
        conv_w.stride(0), conv_w.stride(1),
        K=K, BLOCK_D=BLOCK_D)
    return out
