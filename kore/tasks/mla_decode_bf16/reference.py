"""Reference + inputs for bf16 MLA (Multi-head Latent Attention) decode.

DeepSeek-V2/V3 MLA: the KV cache is compressed to a low-rank latent ``c_kv``
[S, d_c] (d_c = kv_lora_rank << H*D) plus a decoupled-RoPE key ``k_pe`` [S, d_rope]
shared across heads. Per head h the key/value are UP-projected from the latent
    k_nope[h] = c_kv @ W_UK[h]^T          # [S, D_nope]
    v[h]      = c_kv @ W_UV[h]^T          # [S, D_v]
and the score is the concat dot ``q_nope . k_nope + q_pe . k_pe``. This is the
structurally NOVEL attention variant KORE holds out to measure true generalization
(the policy never trains on it).

Decode: Sq query tokens (typically 1) attend the whole S-token latent cache (no
causal mask). Correctness oracle: the exact fp32 MLA math above. Layout:
    q_nope [B,Sq,H,D_nope], q_pe [B,Sq,H,d_rope],
    c_kv   [B,S,d_c],       k_pe [B,S,d_rope],
    W_UK   [H,D_nope,d_c],  W_UV [H,D_v,d_c]  ->  out [B,Sq,H,D_v]
"""

from __future__ import annotations

import math

import torch


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 1, "H": 16, "S": 2048, "DC": 512, "DNOPE": 128, "DROPE": 64,
                "DV": 128, "SQ": 1}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (q_nope, q_pe, c_kv, k_pe, w_uk, w_uv)."""
    g = torch.Generator(device=device).manual_seed(seed)
    B, H, S = shape["B"], shape["H"], shape["S"]
    DC, DNOPE, DROPE, DV, SQ = (shape["DC"], shape["DNOPE"], shape["DROPE"],
                                shape["DV"], shape["SQ"])

    def rn(*sz, s=0):
        gg = torch.Generator(device=device).manual_seed(seed + s)
        return torch.randn(sz, generator=gg, device=device, dtype=torch.float32)

    q_nope = rn(B, SQ, H, DNOPE, s=1).to(dtype)
    q_pe = rn(B, SQ, H, DROPE, s=2).to(dtype)
    c_kv = rn(B, S, DC, s=3).to(dtype)
    k_pe = rn(B, S, DROPE, s=4).to(dtype)
    # up-projection weights scaled by 1/sqrt(d_c) so k_nope/v stay O(1)
    w_uk = (rn(H, DNOPE, DC, s=5) / math.sqrt(DC)).to(dtype)
    w_uv = (rn(H, DV, DC, s=6) / math.sqrt(DC)).to(dtype)
    return q_nope, q_pe, c_kv, k_pe, w_uk, w_uv


def mla_ref(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv) -> torch.Tensor:
    """Exact fp32 MLA decode oracle -> bf16, layout [B,Sq,H,D_v]."""
    B, SQ, H, DNOPE = q_nope.shape
    DROPE = q_pe.shape[3]
    scale = 1.0 / math.sqrt(DNOPE + DROPE)
    qn = q_nope.float(); qp = q_pe.float()
    ckv = c_kv.float(); kpe = k_pe.float()
    wuk = w_uk.float(); wuv = w_uv.float()
    out = torch.empty((B, SQ, H, w_uv.shape[1]), device=q_nope.device, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            k_nope = ckv[b] @ wuk[h].t()           # [S, DNOPE]
            v = ckv[b] @ wuv[h].t()                # [S, DV]
            s_nope = qn[b, :, h] @ k_nope.t()      # [SQ, S]
            s_pe = qp[b, :, h] @ kpe[b].t()        # [SQ, S]
            scores = (s_nope + s_pe) * scale       # [SQ, S]
            p = torch.softmax(scores, dim=-1)
            out[b, :, h] = p @ v                   # [SQ, DV]
    return out.to(torch.bfloat16)


def mla_batched(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv) -> torch.Tensor:
    """Vectorized torch MLA (materialize per-head K/V via einsum) -> bf16.

    The realistic non-fused serving bar (rocBLAS batched GEMMs + SDPA-style softmax);
    the fused MLA kernel beats it by never materializing per-head K_nope/V.
    """
    B, SQ, H, DNOPE = q_nope.shape
    DROPE = q_pe.shape[3]
    scale = 1.0 / math.sqrt(DNOPE + DROPE)
    qn, qp = q_nope.float(), q_pe.float()
    ckv, kpe = c_kv.float(), k_pe.float()
    wuk, wuv = w_uk.float(), w_uv.float()
    k_nope = torch.einsum("bsc,hnc->bhsn", ckv, wuk)     # [B,H,S,DNOPE]
    v = torch.einsum("bsc,hvc->bhsv", ckv, wuv)          # [B,H,S,DV]
    s_nope = torch.einsum("bqhn,bhsn->bhqs", qn, k_nope)
    s_pe = torch.einsum("bqhr,bsr->bhqs", qp, kpe)
    scores = (s_nope + s_pe) * scale                     # [B,H,SQ,S]
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqs,bhsv->bqhv", p, v)          # [B,SQ,H,DV]
    return out.to(torch.bfloat16)
