"""Reference + inputs for bf16 attention-SINK (GQA) causal flash-attention prefill.

Attention sinks (gpt-oss / StreamingLLM): each query head has a learned scalar SINK
logit that participates in the softmax DENOMINATOR but has NO value vector, i.e. it is an
always-available "no-op" attention slot that lets a head attend to "nothing" by leaking
probability mass out of the real keys. This stabilizes long-context / streaming decoding.

Exact gpt-oss formulation (this oracle): with per-head sink s_h,
    p_j = exp(logit_j - m) / (sum_k exp(logit_k - m) + exp(s_h - m)),  m = max(max_k logit_k, s_h)
    out_i = sum_j p_j v_j    (the sink contributes to the denominator only)
equivalently: append a column of value s_h (with a zero value vector) to the logits,
softmax, then drop the sink column. Correctness oracle: exact fp32 causal GQA attention
WITH the sink term; the SNR gate measures the flash kernel's online-softmax + sink
fidelity.

Layout matches AITER ``flash_attn_func``: q ``[B,S,H,D]``, k/v ``[B,S,KV,D]``,
plus per-head sink ``[H]`` fp32.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _attn_common import causal_mask, expand_kv, sdpa_fp32  # noqa: E402

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
    """Returns (q, k, v, sink): q ``[B,S,H,D]`` bf16, k/v ``[B,S,KV,D]`` bf16, sink ``[H]`` fp32."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    B, H, KV, S, D = shape["B"], shape["H"], shape["KV"], shape["S"], shape["D"]
    q = torch.randn((B, S, H, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn((B, S, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn((B, S, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    # Per-head learned sink logit, O(1) so it is a meaningful (not dominant) denominator
    # term against typical scaled q.k logits.
    sink = torch.randn((H,), generator=g, device=device, dtype=torch.float32)
    return (q, k, v, sink)


def reference_output(shape, inputs):
    """Exact fp32 causal GQA attention WITH per-head sink -> bf16, layout ``[B,S,H,D]``."""
    q, k, v, sink = inputs
    B, S, H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    qf = q.float().transpose(1, 2)                    # [B,H,S,D]
    kf = expand_kv(k.float().transpose(1, 2), H)
    vf = expand_kv(v.float().transpose(1, 2), H)
    mask = causal_mask(S, S, q.device)
    out = sdpa_fp32(qf, kf, vf, scale, attn_mask=mask, sink=sink.float())
    return out.transpose(1, 2).to(q.dtype)            # [B,S,H,D]


def candidate_output(fn, shape, inputs):
    q, k, v, sink = inputs
    return fn(q, k, v, sink, causal=True)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER ``flash_attn_func`` with the per-head ``sink_ptr``.

    NOTE (verify on gfx950): the exact sink_ptr dtype/shape/semantics expected by the
    installed AITER must be confirmed to match this oracle (per-head additive sink logit
    in the softmax denominator). See VERIFICATION_CHECKLIST.md."""
    import aiter

    q, k, v, sink = inputs
    return aiter.flash_attn_func(q, k, v, causal=True, sink_ptr=sink)
