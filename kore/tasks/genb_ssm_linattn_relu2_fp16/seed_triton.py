"""GENERATED breadth ssm_linattn_relu2 seed. Causal linear attention, feature map relu2
(normalize=False). q/k/v[B,H,L,Dh] -> y[B,H,L,Dh]. One program per
(b,h,e) keeps the fp32 state column s[Dh] (and normalizer z[Dh]) and scans over L:
s += phi(k_l)*v_l[e]; num = sum_d phi(q_l)[d]*s[d]. tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_linattn_relu2_kernel(q_ptr, k_ptr, v_ptr, y_ptr, H, L, Dh, s_b, s_h, s_l, s_d,
                 DB: tl.constexpr):
    pid = tl.program_id(0)
    e = pid % Dh
    tmp = pid // Dh
    hh = tmp % H
    bb = tmp // H
    dd = tl.arange(0, DB)
    dmask = dd < Dh
    bh = bb * s_b + hh * s_h
    s = tl.zeros([DB], dtype=tl.float32)
    for l in range(0, L):
        off = bh + l * s_l
        krow = tl.load(k_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        phik = tl.maximum(krow, 0.0) * tl.maximum(krow, 0.0)
        phik = tl.where(dmask, phik, 0.0)
        v_le = tl.load(v_ptr + off + e * s_d).to(tl.float32)
        s = s + phik * v_le
        qrow = tl.load(q_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        phiq = tl.maximum(qrow, 0.0) * tl.maximum(qrow, 0.0)
        num = tl.sum(tl.where(dmask, phiq * s, 0.0), axis=0)
        y_le = num
        tl.store(y_ptr + off + e * s_d, y_le.to(tl.float16))


def ssm_linattn_relu2(q, k, v):
    B, H, L, Dh = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    y = torch.empty_like(v)
    DB = triton.next_power_of_2(Dh)
    _ssm_linattn_relu2_kernel[(B * H * Dh,)](q, k, v, y, H, L, Dh,
                               q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                               DB=DB, num_warps=1)
    return y
