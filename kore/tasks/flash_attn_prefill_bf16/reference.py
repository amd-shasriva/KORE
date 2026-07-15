"""Reference + inputs for the bf16 causal (GQA) flash-attention *prefill* task.

Correctness oracle: exact fp32 scaled-dot-product-attention with a causal mask
and grouped-query attention (Q heads share KV heads). This is mathematically the
softmax attention the flash kernel approximates; the SNR gate measures the flash
kernel's online-softmax numerical fidelity, not a different algorithm.

Perf baseline (driver ``--impl reference``): AITER ``flash_attn_func`` (CK/ASM
FMHA) - the real prefill serving bar.

Layout (matches AITER ``flash_attn_func``): q ``[B, S, H, D]``, k/v
``[B, S, KV, D]`` (KV<=H, H % KV == 0), causal aligned to the bottom-right.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 1, "H": 32, "KV": 8, "S": 2048, "D": 128}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Returns (q, k, v) with q:[B,S,H,D], k/v:[B,S,KV,D] in ``dtype``."""
    g = torch.Generator(device=device).manual_seed(seed)
    B, H, KV, S, D = shape["B"], shape["H"], shape["KV"], shape["S"], shape["D"]
    q = torch.randn((B, S, H, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    k = torch.randn((B, S, KV, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    v = torch.randn((B, S, KV, D), generator=g, device=device, dtype=torch.float32).to(dtype)
    return q, k, v


def attn_ref(q, k, v, causal: bool = True) -> torch.Tensor:
    """Exact fp32 causal GQA attention oracle -> bf16, layout [B,S,H,D]."""
    B, S, H, D = q.shape
    KV = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    qf = q.float().transpose(1, 2)   # [B,H,S,D]
    kf = k.float().transpose(1, 2)   # [B,KV,S,D]
    vf = v.float().transpose(1, 2)
    out = F.scaled_dot_product_attention(
        qf, kf, vf, is_causal=causal, scale=scale, enable_gqa=(KV != H)
    )
    return out.transpose(1, 2).to(q.dtype)   # [B,S,H,D]
