"""Reference + inputs for bf16 CHUNKED-prefill (GQA) causal flash-attention.

Chunked prefill (the vLLM / continuous-batching serving pattern): a chunk of ``Sq`` NEW
query tokens attends to a longer KV context of ``Skv`` keys (Skv >= Sq), where the new
tokens are the LAST Sq positions of the context. Causal is BOTTOM-RIGHT aligned: query i
(global position ``Skv - Sq + i``) attends to keys ``j <= Skv - Sq + i`` (so the last new
token sees the whole context, the first new token sees ``Skv - Sq + 1`` keys). This is
how a long prompt is prefilled in fixed-size chunks against the already-cached prefix
(blueprint A6). Correctness oracle: exact fp32 GQA SDPA with the bottom-right causal mask.

Layout matches AITER ``flash_attn_func`` (Sq != Skv, causal): q ``[B,Sq,H,D]``, k/v
``[B,Skv,KV,D]``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kore.tasks._attn_common import causal_mask, expand_kv, sdpa_fp32  # noqa: E402

ENTRY = "flash_attn"
ATOL = 2e-2
RTOL = 2e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 1, "H": 32, "KV": 8, "Sq": 512, "Skv": 4096, "D": 128}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (q, k, v): q ``[B,Sq,H,D]`` bf16, k/v ``[B,Skv,KV,D]`` bf16 (Sq <= Skv)."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    B, H, KV, Sq, Skv, D = shape["B"], shape["H"], shape["KV"], shape["Sq"], shape["Skv"], shape["D"]
    q = torch.randn((B, Sq, H, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn((B, Skv, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn((B, Skv, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    return (q, k, v)


def reference_output(shape, inputs):
    """Exact fp32 bottom-right-causal GQA chunked-prefill oracle -> bf16 ``[B,Sq,H,D]``."""
    q, k, v = inputs
    B, Sq, H, D = q.shape
    Skv = k.shape[1]
    scale = 1.0 / (D ** 0.5)
    qf = q.float().transpose(1, 2)                    # [B,H,Sq,D]
    kf = expand_kv(k.float().transpose(1, 2), H)      # [B,H,Skv,D]
    vf = expand_kv(v.float().transpose(1, 2), H)
    mask = causal_mask(Sq, Skv, q.device, q_offset=Skv - Sq)   # bottom-right causal
    out = sdpa_fp32(qf, kf, vf, scale, attn_mask=mask)
    return out.transpose(1, 2).to(q.dtype)            # [B,Sq,H,D]


def candidate_output(fn, shape, inputs):
    q, k, v = inputs
    return fn(q, k, v, causal=True)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER ``flash_attn_func`` causal with Sq < Skv (bottom-right).

    NOTE (verify on gfx950): confirm the installed AITER uses BOTTOM-RIGHT causal
    alignment for Sq != Skv (last query attends the full context) to match this oracle.
    See VERIFICATION_CHECKLIST.md."""
    from kore.tasks.aiter_ref_attn import aiter_flash_attn

    q, k, v = inputs
    return aiter_flash_attn(q, k, v, causal=True)
