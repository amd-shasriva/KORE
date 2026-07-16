"""Reference + inputs for bf16 MQA (multi-query, KV == 1) flash-attention *decode*.

Decode = a single query token (seq_q = 1) attending over a long KV context, with ALL
query heads sharing a SINGLE kv head (KV == 1, the multi-query extreme). Correctness
oracle: exact fp32 non-causal SDPA (the lone query sees the full KV window) with the
single kv head broadcast to every query head. Complements the existing GQA decode task
with the KV == 1 memory-bandwidth-extreme regime.

Layout (matches AITER ``flash_attn_func`` seq_q=1 decode): q ``[B,1,H,D]``, k/v
``[B,Skv,1,D]``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._attn_common import expand_kv, sdpa_fp32  # noqa: E402

ENTRY = "flash_attn_decode"
ATOL = 2e-2
RTOL = 2e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 8, "H": 32, "KV": 1, "Skv": 4096, "D": 128}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (q, k, v): q ``[B,1,H,D]`` bf16, k/v ``[B,Skv,1,D]`` bf16 (MQA)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    B, H, KV, Skv, D = shape["B"], shape["H"], shape["KV"], shape["Skv"], shape["D"]
    q = torch.randn((B, 1, H, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn((B, Skv, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn((B, Skv, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    return (q, k, v)


def reference_output(shape, inputs):
    """Exact fp32 non-causal MQA decode oracle -> bf16, layout ``[B,1,H,D]``."""
    q, k, v = inputs
    B, Sq, H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    qf = q.float().transpose(1, 2)                    # [B,H,1,D]
    kf = expand_kv(k.float().transpose(1, 2), H)      # [B,1,Skv,D] -> [B,H,Skv,D]
    vf = expand_kv(v.float().transpose(1, 2), H)
    out = sdpa_fp32(qf, kf, vf, scale, attn_mask=None)   # single query sees all KV
    return out.transpose(1, 2).to(q.dtype)            # [B,1,H,D]


def candidate_output(fn, shape, inputs):
    q, k, v = inputs
    return fn(q, k, v)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER CK/ASM FMHA, seq_q=1 decode (MQA, Hkv=1)."""
    from kore.tasks.aiter_ref_attn import aiter_flash_attn

    q, k, v = inputs
    return aiter_flash_attn(q, k, v, causal=False)
