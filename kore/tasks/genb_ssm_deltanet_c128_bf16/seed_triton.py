"""GENERATED breadth ssm_deltanet_c128 seed. (Gated) DeltaNet delta-rule linear attention.
k is L2-normalized. One program per (b,h,e) keeps the fp32 state column s[Dh] and
scans over L: [s=alpha*s;] pred=sum_d k_l[d]*s[d]; s += beta*k_l*(v_l[e]-pred);
y_l[e]=sum_d q_l[d]*s[d]. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_deltanet_c128_kernel(q_ptr, k_ptr, v_ptr, be_ptr, y_ptr, H, L, Dh, s_b, s_h, s_l, s_d,
                 sbe_b, sbe_h, sbe_l, DB: tl.constexpr):
    pid = tl.program_id(0)
    e = pid % Dh
    tmp = pid // Dh
    hh = tmp % H
    bb = tmp // H
    dd = tl.arange(0, DB)
    dmask = dd < Dh
    bh = bb * s_b + hh * s_h
    beh = bb * sbe_b + hh * sbe_h
    s = tl.zeros([DB], dtype=tl.float32)
    for l in range(0, L):
        off = bh + l * s_l
        beta = tl.sigmoid(tl.load(be_ptr + beh + l * sbe_l).to(tl.float32))
        krow = tl.load(k_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        pred = tl.sum(tl.where(dmask, krow * s, 0.0), axis=0)
        v_le = tl.load(v_ptr + off + e * s_d).to(tl.float32)
        s = s + beta * krow * (v_le - pred)
        qrow = tl.load(q_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        y_le = tl.sum(tl.where(dmask, qrow * s, 0.0), axis=0)
        tl.store(y_ptr + off + e * s_d, y_le.to(tl.bfloat16))


def ssm_deltanet_c128(q, k, v, beta):
    B, H, L, Dh = q.shape
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-06)
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    beta = beta.contiguous()
    
    y = torch.empty_like(v)
    DB = triton.next_power_of_2(Dh)
    _ssm_deltanet_c128_kernel[(B * H * Dh,)](
        q, k, v, beta, y, H, L, Dh,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        DB=DB, num_warps=1)
    return y
