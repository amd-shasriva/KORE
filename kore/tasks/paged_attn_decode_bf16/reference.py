"""Reference + inputs for the bf16 paged-KV-cache decode attention task.

vLLM/SGLang-style paged attention: the KV cache is stored in fixed-size pages
(``block_size=16``) and gathered per sequence via a block table. Each sequence
contributes one decode query (seq_q=1) attending over its whole context.

Correctness oracle: exact fp32 attention that gathers K/V from the paged cache
(honouring the block table + per-sequence context length, including a partial
last page) and does non-causal softmax attention. Perf baseline (driver
``--impl reference``): AITER ROCm custom paged attention
(``aiter.paged_attention_rocm``) — the real paged-decode serving bar.

KV-cache layout (AITER / vLLM, x = 16 // itemsize = 8 for bf16):
    key_cache   : [num_blocks, KV, D // x, block_size, x]
    value_cache : [num_blocks, KV, D, block_size]
    query       : [num_seqs, H, D]
    block_tables: [num_seqs, max_blocks_per_seq] (int32)
    seq_lens    : [num_seqs] (int32)
"""

from __future__ import annotations

import math

import torch

BLOCK_SIZE = 16


def parse_shape(shape_str: str) -> dict:
    if not shape_str or shape_str == "default":
        return {"B": 8, "H": 32, "KV": 8, "Skv": 4096, "D": 128}
    out = {}
    for kv in shape_str.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def get_inputs(shape: dict, dtype=torch.bfloat16, device="cuda", seed: int = 0):
    """Build (query, key_cache, value_cache, block_tables, seq_lens, block_size,
    scale). All sequences share context length ``Skv`` (which may leave a partial
    last page). Each sequence is assigned its own contiguous set of pages."""
    g = torch.Generator(device=device).manual_seed(seed)
    B, H, KV, Skv, D = shape["B"], shape["H"], shape["KV"], shape["Skv"], shape["D"]
    bs = BLOCK_SIZE
    x = 16 // torch.tensor([], dtype=dtype).element_size()  # 8 for bf16
    max_blocks = (Skv + bs - 1) // bs
    num_blocks = B * max_blocks

    key_cache = torch.randn(
        (num_blocks, KV, D // x, bs, x), generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    value_cache = torch.randn(
        (num_blocks, KV, D, bs), generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    query = torch.randn((B, H, D), generator=g, device=device, dtype=torch.float32).to(dtype)

    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).view(B, max_blocks)
    seq_lens = torch.full((B,), Skv, device=device, dtype=torch.int32)
    scale = 1.0 / math.sqrt(D)
    return query, key_cache, value_cache, block_tables, seq_lens, bs, scale


def _gather_kv(key_cache, value_cache, block_table, ctx, kv_h, bs):
    """Gather K,V [ctx, D] in fp32 for one (sequence, kv-head)."""
    x = key_cache.shape[-1]
    D = value_cache.shape[2]
    ks, vs = [], []
    n_pages = (ctx + bs - 1) // bs
    for p in range(n_pages):
        page = int(block_table[p].item())
        n_tok = min(bs, ctx - p * bs)
        # key: [D//x, n_tok, x] -> [n_tok, D]
        kblk = key_cache[page, kv_h, :, :n_tok, :].permute(1, 0, 2).reshape(n_tok, D)
        # value: [D, n_tok] -> [n_tok, D]
        vblk = value_cache[page, kv_h, :, :n_tok].permute(1, 0)
        ks.append(kblk.float()); vs.append(vblk.float())
    return torch.cat(ks, 0), torch.cat(vs, 0)


def attn_ref(query, key_cache, value_cache, block_tables, seq_lens, block_size, scale):
    """Exact fp32 paged decode oracle -> bf16, output [num_seqs, H, D]."""
    B, H, D = query.shape
    KV = key_cache.shape[1]
    group = H // KV
    out = torch.empty((B, H, D), device=query.device, dtype=query.dtype)
    for s in range(B):
        ctx = int(seq_lens[s].item())
        for h in range(H):
            kv_h = h // group
            k, v = _gather_kv(key_cache, value_cache, block_tables[s], ctx, kv_h, block_size)
            q = query[s, h].float()
            logits = (k @ q) * scale               # [ctx]
            p = torch.softmax(logits, dim=0)
            out[s, h] = (p @ v).to(query.dtype)
    return out
