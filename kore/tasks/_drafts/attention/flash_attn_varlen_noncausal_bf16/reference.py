"""Reference + inputs for bf16 NON-causal VARLEN / packed (GQA) flash attention.

Bidirectional variable-length (ragged) attention: sequences of DIFFERENT lengths are
packed contiguously with NO padding, delimited by ``cu_seqlens``, and each sequence
attends to ALL of its own tokens (no causal mask) -- the packed layout used by
encoder / embedding / prefix-LM batches. This is DISTINCT from the existing live
``flash_attn_varlen_bf16`` task (which is CAUSAL varlen prefill baselined against padded
dense FMHA): this draft is NON-causal and is graded against the TRUE ragged vendor kernel
``aiter.flash_attn_varlen_func`` (no padding waste at all). Correctness oracle: exact
fp32 full (non-causal) GQA SDPA computed per-sequence over its own slice.

Layout (matches AITER ``flash_attn_varlen_func``): q ``[total_tokens, H, D]``, k/v
``[total_tokens, KV, D]``, ``cu_seqlens`` int32 ``[num_seqs + 1]``. The seqlen partition
is DETERMINISTIC given the shape (seed-independent) and always includes a length-1 and a
max-length sequence (the ragged edges).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _attn_common import expand_kv, sdpa_fp32  # noqa: E402

ENTRY = "flash_attn_varlen"
ATOL = 2e-2
RTOL = 2e-2


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 4, "H": 32, "KV": 8, "S": 4096, "D": 128}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def seqlens_of(shape) -> list:
    """Deterministic ragged sequence-length partition for ``shape`` (seed-independent).

    B sequences with lengths in ``[1, S]``, ALWAYS including a length-1 and a length-S
    sequence so the ragged edges (single-token + max-length) are exercised every run."""
    import torch

    B, S = int(shape["B"]), int(shape["S"])
    g = torch.Generator().manual_seed(0x5EED + B * 100003 + S)
    ls = torch.randint(1, S + 1, (B,), generator=g).tolist()
    ls[0] = 1
    ls[-1] = S
    return ls


def _cu_seqlens(shape, device):
    import torch

    ls = seqlens_of(shape)
    cu = [0]
    for L in ls:
        cu.append(cu[-1] + L)
    return torch.tensor(cu, dtype=torch.int32, device=device)


def get_inputs(shape: dict, device="cuda", seed: int = 0):
    """Returns (q, k, v, cu_seqlens): q ``[T,H,D]`` bf16, k/v ``[T,KV,D]`` bf16, cu int32."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    H, KV, D = shape["H"], shape["KV"], shape["D"]
    cu = _cu_seqlens(shape, device)
    total = int(cu[-1].item())
    q = torch.randn((total, H, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn((total, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn((total, KV, D), generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
    return (q, k, v, cu)


def reference_output(shape, inputs):
    """Exact fp32 per-sequence NON-causal GQA attention oracle -> bf16, layout ``[T,H,D]``."""
    import torch

    q, k, v, cu = inputs
    total, H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    out = torch.empty((total, H, D), dtype=torch.bfloat16, device=q.device)
    cu_list = cu.tolist()
    for s in range(len(cu_list) - 1):
        a, b = cu_list[s], cu_list[s + 1]
        L = b - a
        if L <= 0:
            continue
        qs = q[a:b].float().transpose(0, 1).unsqueeze(0)                    # [1,H,L,D]
        ks = expand_kv(k[a:b].float().transpose(0, 1).unsqueeze(0), H)      # [1,H,L,D]
        vs = expand_kv(v[a:b].float().transpose(0, 1).unsqueeze(0), H)
        o = sdpa_fp32(qs, ks, vs, scale, attn_mask=None)                    # bidirectional
        out[a:b] = o.squeeze(0).transpose(0, 1).to(torch.bfloat16)         # [L,H,D]
    return out


def candidate_output(fn, shape, inputs):
    q, k, v, cu = inputs
    max_seqlen = int((cu[1:] - cu[:-1]).max().item())
    return fn(q, k, v, cu, max_seqlen, causal=False)


def baseline_output(shape, inputs):
    """REAL vendor bar: AITER ``flash_attn_varlen_func`` (CK/ASM ragged FMHA, non-causal)."""
    import aiter

    q, k, v, cu = inputs
    max_seqlen = int((cu[1:] - cu[:-1]).max().item())
    return aiter.flash_attn_varlen_func(q, k, v, cu, cu, max_seqlen, max_seqlen, causal=False)
