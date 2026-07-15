"""Reference + inputs for the bf16 (GQA) flash-attention *decode* task.

Decode = a single query token (seq_q = 1) attending over a long KV context.
Correctness oracle: exact fp32 non-causal SDPA with grouped-query attention (the
lone query attends to the full KV window). Perf baseline (driver
``--impl reference``): AITER ``flash_attn_func`` with seq_q=1 - the real decode
attention bar.

Layout (matches AITER ``flash_attn_func``): q ``[B, 1, H, D]``, k/v
``[B, Skv, KV, D]`` (KV<=H, H % KV == 0).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 8, "H": 32, "KV": 8, "Skv": 4096, "D": 128}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (q, k, v) with q:[B,1,H,D], k/v:[B,Skv,KV,D] in ``dtype``."""
    g = torch.Generator(device=device).manual_seed(seed)
    B, H, KV, Skv, D = shape["B"], shape["H"], shape["KV"], shape["Skv"], shape["D"]
    q = torch.randn((B, 1, H, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    k = torch.randn((B, Skv, KV, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    v = torch.randn((B, Skv, KV, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    return q, k, v


def attn_ref(q, k, v) -> torch.Tensor:
    """Exact fp32 non-causal GQA decode oracle -> bf16, layout [B,1,H,D]."""
    B, Sq, H, D = q.shape
    KV = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    qf = q.float().transpose(1, 2)   # [B,H,1,D]
    kf = k.float().transpose(1, 2)   # [B,KV,Skv,D]
    vf = v.float().transpose(1, 2)
    out = F.scaled_dot_product_attention(
        qf, kf, vf, is_causal=False, scale=scale, enable_gqa=(KV != H)
    )
    return out.transpose(1, 2).to(q.dtype)   # [B,1,H,D]
