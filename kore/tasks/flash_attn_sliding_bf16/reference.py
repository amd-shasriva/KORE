"""Reference + inputs for bf16 sliding-window causal (GQA) flash attention.

Sliding-window attention (Mistral / Gemma / gpt-oss): query i attends only to keys
in ``(i - W, i]`` (a causal band of width W), so cost is O(S*W) not O(S^2) and the
kernel can SKIP key blocks fully outside the band. Correctness oracle: exact fp32
SDPA with the sliding-window causal mask; the SNR gate measures the flash kernel's
online-softmax fidelity, not a different algorithm.

Layout matches AITER ``flash_attn_func``: q ``[B,S,H,D]``, k/v ``[B,S,KV,D]``
(KV<=H, H % KV == 0). GQA is expanded manually for the oracle (robust to the torch
version's ``enable_gqa`` support).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 1, "H": 32, "KV": 8, "S": 4096, "D": 128, "W": 1024}
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


def window_of(shape: dict) -> int:
    return int(shape.get("W", 1024))


def attn_ref(q, k, v, window: int) -> torch.Tensor:
    """Exact fp32 sliding-window causal GQA attention -> bf16, layout [B,S,H,D].

    query i attends to keys j with ``i - window < j <= i``.
    """
    B, S, H, D = q.shape
    KV = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    qf = q.float().transpose(1, 2)                       # [B,H,S,D]
    kf = k.float().transpose(1, 2)                       # [B,KV,S,D]
    vf = v.float().transpose(1, 2)
    rep = H // KV
    if rep > 1:
        kf = kf.repeat_interleave(rep, dim=1)            # [B,H,S,D]
        vf = vf.repeat_interleave(rep, dim=1)
    i = torch.arange(S, device=q.device)[:, None]
    j = torch.arange(S, device=q.device)[None, :]
    allow = (j <= i) & (j > i - window)                  # sliding-window causal band
    mask = torch.zeros((S, S), device=q.device, dtype=torch.float32)
    mask = mask.masked_fill(~allow, float("-inf"))
    out = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask, scale=scale)
    return out.transpose(1, 2).to(q.dtype)               # [B,S,H,D]
