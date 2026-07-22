"""GENERATED breadth ssm_gated_retention_c64 seed (bf16). Gated retention (data-dependent SCALAR
decay). q/k/v[B,H,L,Dh], gate logits gl[B,H,L] -> y[B,H,L,Dh]. One program per
(b,h,e) keeps the fp32 state column s[Dh] and scans over L: a=sigmoid(gl_l);
s = a*s + k_l*v_l[e]; y_l[e]=sum_d q_l[d]*s[d]. tl.bfloat16 store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _ssm_gated_retention_c64_kernel(q_ptr, k_ptr, v_ptr, g_ptr, y_ptr, H, L, Dh, s_b, s_h, s_l, s_d,
                 sg_b, sg_h, sg_l, DB: tl.constexpr):
    pid = tl.program_id(0)
    e = pid % Dh
    tmp = pid // Dh
    hh = tmp % H
    bb = tmp // H
    dd = tl.arange(0, DB)
    dmask = dd < Dh
    bh = bb * s_b + hh * s_h
    gh = bb * sg_b + hh * sg_h
    s = tl.zeros([DB], dtype=tl.float32)
    for l in range(0, L):
        off = bh + l * s_l
        a = tl.sigmoid(tl.load(g_ptr + gh + l * sg_l).to(tl.float32))
        krow = tl.load(k_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        v_le = tl.load(v_ptr + off + e * s_d).to(tl.float32)
        s = a * s + krow * v_le
        qrow = tl.load(q_ptr + off + dd * s_d, mask=dmask, other=0.0).to(tl.float32)
        y_le = tl.sum(tl.where(dmask, qrow * s, 0.0), axis=0)
        tl.store(y_ptr + off + e * s_d, y_le.to(tl.bfloat16))


def ssm_gated_retention_c64(q, k, v, gl):
    B, H, L, Dh = q.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); gl = gl.contiguous()
    y = torch.empty_like(v)
    DB = triton.next_power_of_2(Dh)
    _ssm_gated_retention_c64_kernel[(B * H * Dh,)](q, k, v, gl, y, H, L, Dh,
                               q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                               gl.stride(0), gl.stride(1), gl.stride(2),
                               DB=DB, num_warps=1)
    return y
