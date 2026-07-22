"""GENERATED breadth linear_attention seed (fp16). Causal linear attention, phi=elu+1.
q/k/v[B,H,L,Dh] -> y[B,H,L,Dh]. One program per (b, h, e) keeps an fp32 state column s[Dh]
(= S[:, e]) and scans over L: s += phi(k_l) * v_l[e]; y_l[e] = sum_d phi(q_l)[d] * s[d].
Naive sequential scan (the policy chunks/parallelizes it). tl.float16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _linear_attention_kernel(q_ptr, k_ptr, v_ptr, y_ptr, H, L, Dh,
                             s_b, s_h, s_l, s_d, DB: tl.constexpr):
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
        krow = tl.load(k_ptr + bh + l * s_l + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        phik = tl.where(krow > 0.0, krow + 1.0, tl.exp(krow))     # elu(k)+1
        v_le = tl.load(v_ptr + bh + l * s_l + e * s_d).to(tl.float32)
        s = s + phik * v_le
        qrow = tl.load(q_ptr + bh + l * s_l + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        phiq = tl.where(qrow > 0.0, qrow + 1.0, tl.exp(qrow))     # elu(q)+1
        y_le = tl.sum(tl.where(dmask, phiq * s, 0.0), axis=0)
        tl.store(y_ptr + bh + l * s_l + e * s_d, y_le.to(tl.float16))


def linear_attention(q, k, v):
    Bsz, H, L, Dh = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    y = torch.empty_like(q)
    DB = triton.next_power_of_2(Dh)
    _linear_attention_kernel[(Bsz * H * Dh,)](
        q, k, v, y, H, L, Dh,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), DB=DB, num_warps=1)
    return y
