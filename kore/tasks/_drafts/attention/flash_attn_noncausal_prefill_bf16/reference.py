"""Reference + inputs for bf16 NON-causal (bidirectional) GQA flash-attention prefill.

Full bidirectional attention (every query attends to every key, no causal mask) with
grouped-query heads (blueprint A4). Used by encoder-style / prefix / embedding passes.
Correctness oracle: exact fp32 SDPA with NO mask; the SNR gate measures the flash
kernel's online-softmax fidelity.

Layout matches AITER ``flash_attn_func``: q ``[B,S,H,D]``, k/v ``[B,S,KV,D]``
(KV <= H, H % KV == 0).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _attn_common import expand_kv, sdpa_fp32  # noqa: E402

ENTRY = "flash_attn"
ATOL = 2e-2
RTOL = 2e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 1, "H": 32, "KV": 8, "S": 2048, "D": 128}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (q, k, v): q ``[B,S,H,D]`` bf16, k/v ``[B,S,KV,D]`` bf16."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    B, H, KV, S, D = shape["B"], shape["H"], shape["KV"], shape["S"], shape["D"]
    q = torch.randn((B, S, H, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn((B, S, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn((B, S, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    return (q, k, v)


def reference_output(shape, inputs):
    """Exact fp32 NON-causal GQA attention oracle -> bf16, layout ``[B,S,H,D]``."""
    q, k, v = inputs
    B, S, H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    qf = q.float().transpose(1, 2)                    # [B,H,S,D]
    kf = expand_kv(k.float().transpose(1, 2), H)
    vf = expand_kv(v.float().transpose(1, 2), H)
    out = sdpa_fp32(qf, kf, vf, scale, attn_mask=None)   # bidirectional
    return out.transpose(1, 2).to(q.dtype)            # [B,S,H,D]


def candidate_output(fn, shape, inputs):
    q, k, v = inputs
    return fn(q, k, v, causal=False)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER CK/ASM FMHA non-causal prefill."""
    from kore.tasks.aiter_ref_attn import aiter_flash_attn

    q, k, v = inputs
    return aiter_flash_attn(q, k, v, causal=False)
