"""GENERATED breadth ssm_rwkv5_wkv seed. RWKV wkv (num/den + bonus), data_dependent=False.
k/v[B,L,C] -> y[B,L,C]. One program per (b,c) keeps fp32 (S, Z) and scans over L:
wkv=(S+exp(u+k)v)/(Z+exp(u+k)); w=softplus(decay); S=exp(-w)S+exp(k)v;
Z=exp(-w)Z+exp(k). tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_rwkv5_wkv_kernel(k_ptr, v_ptr, w_ptr, u_ptr, y_ptr, C, L,
                 sk_b, sk_l, sk_c, sw_c, su_c, SP: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C
    uc = tl.load(u_ptr + c * su_c).to(tl.float32)
    S = 0.0
    Z = 0.0
    for l in range(0, L):
        off = b * sk_b + l * sk_l + c * sk_c
        kt = tl.load(k_ptr + off).to(tl.float32)
        vt = tl.load(v_ptr + off).to(tl.float32)
        ek = tl.exp(kt)
        eb = tl.exp(uc + kt)
        tl.store(y_ptr + off, ((S + eb * vt) / (Z + eb)).to(tl.bfloat16))
        wl = tl.load(w_ptr + c * sw_c).to(tl.float32)
        wt = tl.where(wl > 20.0, wl, tl.log(1.0 + tl.exp(wl)))
        dec = tl.exp(-wt)
        S = dec * S + ek * vt
        Z = dec * Z + ek


def ssm_rwkv5_wkv(k, v, w, u):
    B, L, C = k.shape
    k = k.contiguous(); v = v.contiguous(); w = w.contiguous(); u = u.contiguous()
    y = torch.empty_like(v)
    _ssm_rwkv5_wkv_kernel[(B * C,)](k, v, w, u, y, C, L,
                          k.stride(0), k.stride(1), k.stride(2),
                          w.stride(0), u.stride(0), SP=0, num_warps=1)
    return y
