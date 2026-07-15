"""Reference + inputs for bf16 sliding-window (GQA) flash-attention *decode*.

Sliding-window DECODE (Mistral / Gemma / gpt-oss decode step): the single new query
token (at global position Skv-1) attends only to the most recent ``W`` keys, i.e. keys
``j`` with ``Skv-1-W < j <= Skv-1``. A windowed decode kernel reads only the last W of
the Skv KV entries (bounded work + KV traffic regardless of context length), which is the
whole point of SWA at decode. Correctness oracle: exact fp32 windowed SDPA over the KV
cache; the SNR gate measures the flash kernel's online-softmax fidelity + correct window
band. Complements the existing sliding-window PREFILL task with the decode regime.

Layout (matches AITER ``flash_attn_func`` seq_q=1 decode): q ``[B,1,H,D]``, k/v
``[B,Skv,KV,D]``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _attn_common import expand_kv, sdpa_fp32, sliding_window_mask  # noqa: E402

ENTRY = "flash_attn_decode"
ATOL = 2e-2
RTOL = 2e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 8, "H": 32, "KV": 8, "Skv": 8192, "D": 128, "W": 1024}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def window_of(shape) -> int:
    return int(shape.get("W", 1024))


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (q, k, v): q ``[B,1,H,D]`` bf16, k/v ``[B,Skv,KV,D]`` bf16."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    B, H, KV, Skv, D = shape["B"], shape["H"], shape["KV"], shape["Skv"], shape["D"]
    q = torch.randn((B, 1, H, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn((B, Skv, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn((B, Skv, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    return (q, k, v)


def reference_output(shape, inputs):
    """Exact fp32 sliding-window decode oracle -> bf16, layout ``[B,1,H,D]``.

    The lone query (global position Skv-1) attends to keys j with Skv-1-W < j <= Skv-1."""
    q, k, v = inputs
    B, Sq, H, D = q.shape
    Skv = k.shape[1]
    W = window_of(shape)
    scale = 1.0 / (D ** 0.5)
    qf = q.float().transpose(1, 2)                    # [B,H,1,D]
    kf = expand_kv(k.float().transpose(1, 2), H)      # [B,H,Skv,D]
    vf = expand_kv(v.float().transpose(1, 2), H)
    mask = sliding_window_mask(1, Skv, W, q.device, q_offset=Skv - 1)   # [1,Skv]
    out = sdpa_fp32(qf, kf, vf, scale, attn_mask=mask)
    return out.transpose(1, 2).to(q.dtype)            # [B,1,H,D]


def candidate_output(fn, shape, inputs):
    q, k, v = inputs
    return fn(q, k, v, window=window_of(shape))


def baseline_output(shape, inputs):
    """REAL vendor perf bar: AITER dense (full-KV) seq_q=1 decode FMHA. The windowed
    decode kernel beats it by reading only the last W keys of the KV cache."""
    from kore.tasks.aiter_ref_attn import aiter_flash_attn

    q, k, v = inputs
    return aiter_flash_attn(q, k, v, causal=False)
