"""Reference + inputs for bf16 variable-length causal (GQA) flash attention.

Varlen (ragged-batch) prefill: sequences of DIFFERENT lengths are packed into one
tensor with ``cu_seqlens`` (cumulative offsets), so there is no padding waste — the
real serving path for mixed-length prompts (``flash_attn_varlen_func``). Each
sequence attends only within itself (causal). Correctness oracle: per-sequence
exact fp32 causal SDPA; the SNR gate measures the flash kernel's online-softmax
fidelity + correct cu_seqlens indexing.

Layout: q ``[T, H, D]``, k/v ``[T, KV, D]`` (T = sum of seqlens), cu_seqlens
``[B+1]`` int32 (shared by q and k for self-attention prefill).
"""

from __future__ import annotations

import math
from itertools import accumulate

import torch
import torch.nn.functional as F


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 4, "H": 32, "KV": 8, "D": 128, "SMAX": 2048}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (q, k, v, cu_seqlens, max_seqlen): q:[T,H,D], k/v:[T,KV,D] packed."""
    g = torch.Generator(device=device).manual_seed(seed)
    B, H, KV, D, SMAX = shape["B"], shape["H"], shape["KV"], shape["D"], shape["SMAX"]
    lo = max(1, SMAX // 2)
    lens = [int(torch.randint(lo, SMAX + 1, (1,), generator=g, device=device).item())
            for _ in range(B)]
    T = sum(lens)
    q = torch.randn((T, H, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    k = torch.randn((T, KV, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    v = torch.randn((T, KV, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    cu = torch.tensor([0] + list(accumulate(lens)), dtype=torch.int32, device=device)
    return q, k, v, cu, max(lens)


def attn_ref(q, k, v, cu_seqlens, max_seqlen, causal: bool = True) -> torch.Tensor:
    """Exact per-sequence fp32 causal GQA attention -> bf16, layout [T,H,D]."""
    T, H, D = q.shape
    KV = k.shape[1]
    scale = 1.0 / math.sqrt(D)
    rep = H // KV
    out = torch.empty_like(q)
    cu = cu_seqlens.tolist()
    for b in range(len(cu) - 1):
        s, e = cu[b], cu[b + 1]
        if e <= s:
            continue
        qf = q[s:e].float().transpose(0, 1).unsqueeze(0)   # [1,H,L,D]
        kf = k[s:e].float().transpose(0, 1).unsqueeze(0)   # [1,KV,L,D]
        vf = v[s:e].float().transpose(0, 1).unsqueeze(0)
        if rep > 1:
            kf = kf.repeat_interleave(rep, dim=1)
            vf = vf.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal, scale=scale)
        out[s:e] = o[0].transpose(0, 1).to(q.dtype)         # [L,H,D]
    return out
