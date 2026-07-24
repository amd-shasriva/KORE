"""GENERATED breadth ssm_conv_selective seed (fp16). Mamba conv+SSM: causal depthwise conv
(+SiLU) then the Mamba-1 selective scan. u[B,L,D], conv_w[D,K], delta[B,L,D],
A[D,N], B_/C[B,L,N] -> y[B,L,D]. A Triton causal-convolution kernel materializes
the SiLU projection; one scan program per (b,d) then computes the selective
recurrence. The policy fuses the two honest Triton stages and chunks it. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


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
        tl.store(y_ptr + off_ld, y_v.to(tl.float16))


def ssm_conv_selective(u, conv_w, delta, A, B_, C):
    B, L, D = u.shape
    N = A.shape[1]
    uc = _causal_conv_silu(u, conv_w)
    delta = delta.contiguous(); A = A.contiguous(); B_ = B_.contiguous(); C = C.contiguous()
    y = torch.empty_like(uc)
    NB = triton.next_power_of_2(N)
    _ssm_conv_selective_kernel[(B * D,)](
        uc, delta, A, B_, C, y, L, D, N,
        uc.stride(0), uc.stride(1), uc.stride(2), A.stride(0), A.stride(1),
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
