"""Breadth SERVING/INFERENCE ATTENTION task-authoring engine (torch-baselined).

A second, non-overlapping flash-attention pass (sibling of ``attn_ext.py``) that
widens the KORE suite with the *inference/serving frontier* attention kernels that
power vLLM / SGLang - the hard, irregular-shape attention that a naive torch
implementation runs quadratically and memory-bound, so a fused Triton tile that
streams KV blocks with a running (max, sum) online softmax has genuine headroom:

  * varlen / ragged-batch prefill  - packed (cu_seqlens) self-attention, per
    sequence, causal + non-causal (no cross-sequence attention).
  * chunked prefill                - a query chunk attends a *growing* KV context
    (SK > SQ), the new queries pinned to the last SQ absolute positions.
  * cross-attention (enc-dec)       - decoder queries attend encoder keys/values
    (full, non-causal), both prefill (SQ > 1) and single-step (SQ == 1).
  * KV-cache append + attention     - concatenate freshly produced (k, v) onto the
    cache, then attend (== full attention on the concatenated KV).
  * block-local / windowed prefill  - local attention over the last ``W`` keys
    (W in {256, 1024}), and dilated/strided sparse attention (stride ``d``).
  * GQA / MQA decode                - q_len == 1 against a long KV cache, head_dim
    {64, 128, 256}.
  * relative-position bias          - T5-style bucketed learned relative bias.
  * custom additive mask            - an arbitrary additive float bias/mask.
  * fp8 (e4m3) grouped/multi-query prefill - CDNA4/gfx950 fp8 attention, bf16 out.

These are DISJOINT from ``attn_ext`` (which owns the dense MHA/GQA/MQA grid, ALiBi,
softcap, sink, backward) and DISJOINT from the RESERVED held-out eval families
(MLA / paged / latent-cache) - every op here is TRAIN.

Contract mirrors ``kore/tasks/breadth/attn_ext.py`` so the shared ``_genops``
driver + generator consume it unchanged:

    OPS / OP_DTYPES / SHAPES              module-level task catalog
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn fp32 oracle, baseline_fn torch eager, arity,
        entry_name, dtype_name, family=f"breadth_{op}", mutates_input=False)
    seed_source(op, dtype) -> str         a naive, COMPILING, correct online-softmax
        flash-attention Triton seed (defines ``def <op>(*inputs)``).

CORRECTNESS is paramount: every ``ref_fn`` computes in fp32 (fp8 upcast) and casts
back, with the EXACT varlen / chunked / cross / cache / window / dilation / bias
semantics, and is validated on CPU against an INDEPENDENT torch path (a hand-written
O(S^2) einsum softmax, F.scaled_dot_product_attention, per-sequence loops) at tight
fp32 tol. torch / triton are imported lazily (registry discovery needs no GPU).
"""

from __future__ import annotations

from dataclasses import dataclass

from kore.tasks._genops import DTYPES, _parse_shape

# T5 relative-position-bucket hyper-parameters (shared by ref_fn and the seed).
_T5_NUM_BUCKETS = 32
_T5_MAX_DIST = 128


# --------------------------------------------------------------------------- #
# per-op specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Attn2:
    family: str             # varlen|chunked|cross|cache|window|dilated|decode|
                            # relbias|custommask|fp8
    variant: str            # "mha" | "gqa" | "mqa"
    head_dim: int           # 64 | 128 | 256
    causal: bool = False
    window: int = 0         # 0 = full; else sliding-window width W
    dilation: int = 1       # 1 = dense; else strided/dilated stride
    fp8: bool = False
    step: bool = False       # cross-attention single decode step (SQ == 1)


def _kv_heads(variant: str, H: int) -> int:
    """KV-head count for the attention variant (GQA uses a group of 4)."""
    if variant == "mha":
        return H
    if variant == "mqa":
        return 1
    return max(1, H // 4)


# --------------------------------------------------------------------------- #
# task catalog (exactly 32 ops, all prefixed ``attn2_``; NO mla/paged/latent)
# --------------------------------------------------------------------------- #
_SPECS: dict[str, _Attn2] = {}
_RESERVED = ("mla", "paged", "latent")


def _add(name: str, spec: _Attn2) -> None:
    assert name.startswith("attn2_"), name
    assert not any(bad in name for bad in _RESERVED), name
    assert name not in _SPECS, name
    _SPECS[name] = spec


# --- varlen / ragged-batch prefill (cu_seqlens): causal + non-causal (4)
_add("attn2_varlen_mha_causal", _Attn2("varlen", "mha", 128, causal=True))
_add("attn2_varlen_gqa_causal", _Attn2("varlen", "gqa", 128, causal=True))
_add("attn2_varlen_mha_noncausal", _Attn2("varlen", "mha", 128, causal=False))
_add("attn2_varlen_gqa_noncausal", _Attn2("varlen", "gqa", 128, causal=False))

# --- chunked prefill: query chunk attends a growing KV context (3)
_add("attn2_chunked_mha_causal", _Attn2("chunked", "mha", 128, causal=True))
_add("attn2_chunked_gqa_causal", _Attn2("chunked", "gqa", 128, causal=True))
_add("attn2_chunked_mqa_causal", _Attn2("chunked", "mqa", 128, causal=True))

# --- cross-attention (encoder-decoder): prefill + single step (4)
_add("attn2_cross_mha_prefill", _Attn2("cross", "mha", 128, causal=False))
_add("attn2_cross_gqa_prefill", _Attn2("cross", "gqa", 128, causal=False))
_add("attn2_cross_mha_step", _Attn2("cross", "mha", 128, causal=False, step=True))
_add("attn2_cross_gqa_step", _Attn2("cross", "gqa", 128, causal=False, step=True))

# --- KV-cache append + attention (3)
_add("attn2_cacheappend_mha", _Attn2("cache", "mha", 128, causal=True))
_add("attn2_cacheappend_gqa", _Attn2("cache", "gqa", 128, causal=True))
_add("attn2_cacheappend_mqa", _Attn2("cache", "mqa", 128, causal=True))

# --- block-local / windowed + dilated/strided sparse prefill (6)
_add("attn2_window256_mha_causal", _Attn2("window", "mha", 128, causal=True, window=256))
_add("attn2_window256_gqa_causal", _Attn2("window", "gqa", 128, causal=True, window=256))
_add("attn2_window1024_mha_causal", _Attn2("window", "mha", 128, causal=True, window=1024))
_add("attn2_window1024_gqa_causal", _Attn2("window", "gqa", 128, causal=True, window=1024))
_add("attn2_dilated_mha_causal", _Attn2("dilated", "mha", 128, causal=True, dilation=4))
_add("attn2_dilated_gqa_causal", _Attn2("dilated", "gqa", 128, causal=True, dilation=4))

# --- GQA / MQA decode (q_len == 1, long KV) at head_dim {64,128,256} (6)
_add("attn2_decode_gqa_hd64", _Attn2("decode", "gqa", 64))
_add("attn2_decode_gqa_hd128", _Attn2("decode", "gqa", 128))
_add("attn2_decode_gqa_hd256", _Attn2("decode", "gqa", 256))
_add("attn2_decode_mqa_hd64", _Attn2("decode", "mqa", 64))
_add("attn2_decode_mqa_hd128", _Attn2("decode", "mqa", 128))
_add("attn2_decode_mqa_hd256", _Attn2("decode", "mqa", 256))

# --- relative-position bias (T5 bucketed) + custom additive mask (3)
_add("attn2_relbias_t5_mha_causal", _Attn2("relbias", "mha", 128, causal=True))
_add("attn2_relbias_t5_gqa_causal", _Attn2("relbias", "gqa", 128, causal=True))
_add("attn2_custommask_mha", _Attn2("custommask", "mha", 128, causal=False))

# --- fp8 (e4m3) grouped / multi-query prefill, bf16 output (3)
_add("attn2_fp8_gqa_prefill_causal", _Attn2("fp8", "gqa", 128, causal=True, fp8=True))
_add("attn2_fp8_mqa_prefill_causal", _Attn2("fp8", "mqa", 128, causal=True, fp8=True))
_add("attn2_fp8_gqa_prefill_noncausal", _Attn2("fp8", "gqa", 128, causal=False, fp8=True))

OPS: list[str] = list(_SPECS.keys())

# fp8 ops sweep only fp8 (bf16 out); the rest sweep bf16/fp16 (the fp32 oracle
# casts back). Materialized dict so a generator can iterate OPS x dtypes directly.
DEFAULT_DTYPES: list[str] = ["bf16", "fp16"]
OP_DTYPES: dict[str, list[str]] = {
    op: (["fp8"] if _SPECS[op].fp8 else list(DEFAULT_DTYPES)) for op in OPS
}


def op_dtypes(op: str) -> list[str]:
    """The dtype sweep for an op (per-op override or the global default)."""
    return OP_DTYPES.get(op, DEFAULT_DTYPES)


# --------------------------------------------------------------------------- #
# realistic serving shapes (B in {1,2,4,8}, heads {16,32}, GQA/MQA kv-heads,
# seqlen {512,1024,2048,4096,8192}, head_dim {64,128,256}, non-power-of-2 tails).
# Layout: q[B,H,SQ,D], k/v[B,HKV,SK,D] -> out[B,H,SQ,D] for dense families; varlen
# packs [T,H,D] with cu_seqlens; cache carries a context length + new tokens.
# Minimal shapes stay CPU-cheap for the unit tests.
# --------------------------------------------------------------------------- #
def _shapes_for(spec: _Attn2) -> dict:
    D = spec.head_dim
    v = spec.variant

    def kv(H):
        return _kv_heads(v, H)

    fam = spec.family
    if fam == "varlen":
        def mk(B, H, S):
            return {"B": B, "H": H, "HKV": kv(H), "S": S, "D": D}
        return {
            "minimal": mk(2, 8, 64),
            "primary": mk(4, 32, 2048),
            "validation": [mk(8, 16, 1024), mk(2, 32, 4096), mk(3, 16, 2047)],
        }
    if fam == "cache":
        def mk(B, H, CTX, NEW):
            return {"B": B, "H": H, "HKV": kv(H), "SKctx": CTX, "SQ": NEW, "D": D}
        return {
            "minimal": mk(1, 8, 64, 16),
            "primary": mk(2, 32, 4096, 256),
            "validation": [mk(4, 16, 2048, 128), mk(1, 32, 8192, 64), mk(2, 16, 2047, 65)],
        }
    if fam == "chunked":
        def mk(B, H, SQ, SK):                       # SK = context + chunk (growing)
            return {"B": B, "H": H, "HKV": kv(H), "SQ": SQ, "SK": SK, "D": D}
        return {
            "minimal": mk(1, 8, 32, 96),
            "primary": mk(2, 32, 512, 2560),
            "validation": [mk(4, 16, 256, 4352), mk(1, 32, 128, 2176), mk(2, 16, 200, 1223)],
        }
    if fam == "cross":
        def mk(B, H, SQ, SK):
            return {"B": B, "H": H, "HKV": kv(H), "SQ": SQ, "SK": SK, "D": D}
        if spec.step:                               # single decoder step
            return {
                "minimal": mk(1, 8, 1, 48),
                "primary": mk(4, 32, 1, 2048),
                "validation": [mk(8, 16, 1, 1024), mk(2, 32, 1, 4096), mk(2, 16, 1, 1023)],
            }
        return {
            "minimal": mk(1, 8, 24, 48),
            "primary": mk(2, 32, 1024, 768),
            "validation": [mk(4, 16, 512, 333), mk(1, 32, 777, 4096), mk(2, 16, 200, 1023)],
        }
    if fam == "decode":
        def mk(B, H, SK):
            return {"B": B, "H": H, "HKV": kv(H), "SQ": 1, "SK": SK, "D": D}
        return {
            "minimal": mk(1, 8, 128),
            "primary": mk(4, 32, 4096),
            "validation": [mk(1, 32, 8192), mk(8, 16, 2048), mk(2, 32, 2047)],
        }
    # window / dilated / relbias / custommask / fp8  -> self-attention (SQ == SK)
    def mk(B, H, S):
        return {"B": B, "H": H, "HKV": kv(H), "SQ": S, "SK": S, "D": D}
    return {
        "minimal": mk(1, 8, 64),
        "primary": mk(2, 32, 4096),
        "validation": [mk(4, 16, 512), mk(1, 32, 2048), mk(2, 16, 2047)],
    }


SHAPES: dict[str, dict] = {op: _shapes_for(_SPECS[op]) for op in OPS}


# --------------------------------------------------------------------------- #
# T5 relative-position bucket (unidirectional / causal) - buckets a query/key
# distance into ``num_buckets`` bins (exact for small, log-spaced for large).
# --------------------------------------------------------------------------- #
def _t5_bucket(SQ: int, SK: int, num_buckets: int, max_distance: int, device):
    import math

    import torch

    i = torch.arange(SQ, device=device)[:, None]
    j = torch.arange(SK, device=device)[None, :]
    qpos = (SK - SQ) + i                                  # last query pinned to last key
    n = torch.clamp(qpos - j, min=0)                      # past distance (>= 0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    large = max_exact + (
        torch.log(n.float().clamp(min=1) / max_exact)
        / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).long()
    large = torch.clamp(large, max=num_buckets - 1)
    return torch.where(is_small, n, large)                # [SQ, SK] long


def _custom_mask(SQ: int, SK: int, device, seed: int):
    """A deterministic 'custom' additive float mask: a soft per-position bias plus a
    hard (-1e4) block pattern; key 0 is always kept so no query row is fully masked."""
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    soft = torch.randn((SQ, SK), generator=g, device=device, dtype=torch.float32) * 0.5
    hard = torch.randn((SQ, SK), generator=g, device=device, dtype=torch.float32)
    m = soft.clone()
    m[hard > 1.0] = -1e4
    m[:, 0] = soft[:, 0]
    return m


def _varlen_lens(B: int, S: int) -> list[int]:
    """Deterministic ragged sequence lengths in (S/2, S], last one made non-pow2."""
    lens = []
    for b in range(B):
        L = S - (b * S) // (2 * max(1, B))
        lens.append(max(8, min(S, L)))
    lens[-1] = max(8, lens[-1] - 1)
    return lens


# --------------------------------------------------------------------------- #
# fp32-capable attention core (shared by ref_fn / baseline_fn). Operates on the
# tensors AS GIVEN (ref_fn upcasts to fp32 first; baseline_fn keeps compute dtype).
# Alignment: query row i sits at absolute position (SK - SQ) + i (chunked / cache /
# self-attention), so the last SQ queries are the most recent positions.
# --------------------------------------------------------------------------- #
def _attn_core(q, k, v, *, scale, causal=False, window=0, dilation=1,
               bias=None, add_mask=None):
    import torch

    B, H, SQ, D = q.shape
    HKV = k.shape[1]
    SK = k.shape[2]
    group = H // HKV
    if group > 1:                                   # GQA / MQA broadcast of KV heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale      # [B,H,SQ,SK]
    if bias is not None:                            # relative-position bias [H,SQ,SK]
        scores = scores + bias
    if add_mask is not None:                        # custom additive mask [SQ,SK]
        scores = scores + add_mask

    if causal or window > 0 or dilation > 1:
        i = torch.arange(SQ, device=q.device)[:, None]
        j = torch.arange(SK, device=q.device)[None, :]
        qpos = (SK - SQ) + i
        allowed = torch.ones((SQ, SK), dtype=torch.bool, device=q.device)
        if causal:
            allowed = allowed & (j <= qpos)
        if window > 0:
            allowed = allowed & (qpos - j < window)
        if dilation > 1:                            # strided: keep every d-th past key
            allowed = allowed & (j <= qpos) & (((qpos - j) % dilation) == 0)
        scores = scores.masked_fill(~allowed[None, None], torch.finfo(scores.dtype).min)

    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)                        # [B,H,SQ,D]


# --------------------------------------------------------------------------- #
# reference.py namespace (torch fp32 oracle + torch eager perf baseline)
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch

    spec = _SPECS[op]
    tdt = getattr(torch, DTYPES[dtype][0])
    out_dtype = torch.bfloat16 if spec.fp8 else tdt     # fp8 attention -> bf16 out
    fam = spec.family

    def _randn(shape_, device, seed, scale=1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape_, generator=g, device=device,
                            dtype=torch.float32) * scale).to(tdt)

    def _scale(q):
        return 1.0 / (q.shape[-1] ** 0.5)

    # ---- dense self/cross attention (chunked, cross, window, dilated, decode, fp8)
    if fam in ("chunked", "cross", "window", "dilated", "decode", "fp8"):
        def get_inputs(shape, device="cuda", seed=0):
            B, H, HKV = shape["B"], shape["H"], shape["HKV"]
            SQ, SK, D = shape["SQ"], shape["SK"], shape["D"]
            q = _randn((B, H, SQ, D), device, seed)
            k = _randn((B, HKV, SK, D), device, seed + 1)
            v = _randn((B, HKV, SK, D), device, seed + 2)
            return (q, k, v)

        def ref_fn(q, k, v):
            qf, kf, vf = q.float(), k.float(), v.float()
            return _attn_core(qf, kf, vf, scale=_scale(qf), causal=spec.causal,
                              window=spec.window, dilation=spec.dilation).to(out_dtype)

        def baseline_fn(q, k, v):
            cdt = torch.bfloat16 if spec.fp8 else q.dtype     # fp8 can't matmul: bf16
            qf, kf, vf = q.to(cdt), k.to(cdt), v.to(cdt)
            return _attn_core(qf, kf, vf, scale=_scale(qf), causal=spec.causal,
                              window=spec.window, dilation=spec.dilation).to(out_dtype)

        arity = 3

    # ---- relative-position bias (T5 bucketed): inputs (q, k, v, bias_table)
    elif fam == "relbias":
        def get_inputs(shape, device="cuda", seed=0):
            B, H, HKV = shape["B"], shape["H"], shape["HKV"]
            SQ, SK, D = shape["SQ"], shape["SK"], shape["D"]
            q = _randn((B, H, SQ, D), device, seed)
            k = _randn((B, HKV, SK, D), device, seed + 1)
            v = _randn((B, HKV, SK, D), device, seed + 2)
            bt = _randn((H, _T5_NUM_BUCKETS), device, seed + 3)   # learned bias table
            return (q, k, v, bt)

        def _bias(bt, SQ, SK, device):
            buckets = _t5_bucket(SQ, SK, _T5_NUM_BUCKETS, _T5_MAX_DIST, device)
            return bt[:, buckets]                             # [H, SQ, SK]

        def ref_fn(q, k, v, bt):
            qf, kf, vf = q.float(), k.float(), v.float()
            bias = _bias(bt.float(), qf.shape[2], kf.shape[2], q.device)
            return _attn_core(qf, kf, vf, scale=_scale(qf), causal=True,
                              bias=bias).to(out_dtype)

        def baseline_fn(q, k, v, bt):
            cdt = q.dtype
            qf, kf, vf = q.to(cdt), k.to(cdt), v.to(cdt)
            bias = _bias(bt.to(cdt), qf.shape[2], kf.shape[2], q.device)
            return _attn_core(qf, kf, vf, scale=_scale(qf), causal=True,
                              bias=bias).to(out_dtype)

        arity = 4

    # ---- custom additive mask: inputs (q, k, v, mask[SQ,SK])
    elif fam == "custommask":
        def get_inputs(shape, device="cuda", seed=0):
            B, H, HKV = shape["B"], shape["H"], shape["HKV"]
            SQ, SK, D = shape["SQ"], shape["SK"], shape["D"]
            q = _randn((B, H, SQ, D), device, seed)
            k = _randn((B, HKV, SK, D), device, seed + 1)
            v = _randn((B, HKV, SK, D), device, seed + 2)
            mask = _custom_mask(SQ, SK, device, seed + 3).to(tdt)
            return (q, k, v, mask)

        def ref_fn(q, k, v, mask):
            qf, kf, vf = q.float(), k.float(), v.float()
            return _attn_core(qf, kf, vf, scale=_scale(qf),
                              add_mask=mask.float()).to(out_dtype)

        def baseline_fn(q, k, v, mask):
            cdt = q.dtype
            qf, kf, vf = q.to(cdt), k.to(cdt), v.to(cdt)
            return _attn_core(qf, kf, vf, scale=_scale(qf),
                              add_mask=mask.to(cdt)).to(out_dtype)

        arity = 4

    # ---- varlen / ragged-batch packed prefill: inputs (q, k, v, cu_seqlens)
    elif fam == "varlen":
        def get_inputs(shape, device="cuda", seed=0):
            B, H, HKV = shape["B"], shape["H"], shape["HKV"]
            S, D = shape["S"], shape["D"]
            lens = _varlen_lens(B, S)
            cu = [0]
            for L in lens:
                cu.append(cu[-1] + L)
            T = cu[-1]
            q = _randn((T, H, D), device, seed)
            k = _randn((T, HKV, D), device, seed + 1)
            v = _randn((T, HKV, D), device, seed + 2)
            cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
            return (q, k, v, cu_t)

        def _run_varlen(q, k, v, cu, compute_dtype):
            cul = cu.tolist()
            H = q.shape[1]
            out = torch.empty((q.shape[0], H, q.shape[2]), dtype=out_dtype, device=q.device)
            for b in range(len(cul) - 1):
                s, e = cul[b], cul[b + 1]
                if e <= s:
                    continue
                qs = q[s:e].transpose(0, 1).unsqueeze(0).to(compute_dtype)
                ks = k[s:e].transpose(0, 1).unsqueeze(0).to(compute_dtype)
                vs = v[s:e].transpose(0, 1).unsqueeze(0).to(compute_dtype)
                o = _attn_core(qs, ks, vs, scale=_scale(qs), causal=spec.causal)
                out[s:e] = o[0].transpose(0, 1).to(out_dtype)
            return out

        def ref_fn(q, k, v, cu):
            return _run_varlen(q, k, v, cu, torch.float32)

        def baseline_fn(q, k, v, cu):
            return _run_varlen(q, k, v, cu, q.dtype)

        arity = 4

    # ---- KV-cache append + attention: inputs (q, k_cache, v_cache, k_new, v_new)
    elif fam == "cache":
        def get_inputs(shape, device="cuda", seed=0):
            B, H, HKV = shape["B"], shape["H"], shape["HKV"]
            CTX, NEW, D = shape["SKctx"], shape["SQ"], shape["D"]
            q = _randn((B, H, NEW, D), device, seed)
            kc = _randn((B, HKV, CTX, D), device, seed + 1)
            vc = _randn((B, HKV, CTX, D), device, seed + 2)
            kn = _randn((B, HKV, NEW, D), device, seed + 3)
            vn = _randn((B, HKV, NEW, D), device, seed + 4)
            return (q, kc, vc, kn, vn)

        def ref_fn(q, kc, vc, kn, vn):
            qf = q.float()
            k = torch.cat([kc, kn], dim=2).float()
            v = torch.cat([vc, vn], dim=2).float()
            return _attn_core(qf, k, v, scale=_scale(qf), causal=True).to(out_dtype)

        def baseline_fn(q, kc, vc, kn, vn):
            cdt = q.dtype
            qf = q.to(cdt)
            k = torch.cat([kc, kn], dim=2).to(cdt)
            v = torch.cat([vc, vn], dim=2).to(cdt)
            return _attn_core(qf, k, v, scale=_scale(qf), causal=True).to(out_dtype)

        arity = 5

    else:
        raise ValueError(f"unknown family {fam!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}", "mutates_input": False}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive (correct, compiling) Triton flash-attention seeds - the policy's start.
# One forward program per (query-block, batch*head) streams KV blocks with an
# online (max, sum) softmax (fp32 math), GQA via kv_head = head // group; optional
# causal / sliding-window / dilated / additive-bias. Family wrappers preprocess in
# torch (varlen unpacks per sequence; cache concatenates; relbias builds the T5
# bias; custommask expands the mask) then call the SAME proven flash core.
# --------------------------------------------------------------------------- #
_SEED_HEADER = (
    "from __future__ import annotations\n"
    "import torch\n"
    "import triton\n"
    "import triton.language as tl\n\n\n"
)

_FWD_KERNEL_SRC = '''@triton.jit
def _attn2_fwd(Q, K, V, O, Bias, sm_scale, H, HKV, SQ, SK,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
               IS_CAUSAL: tl.constexpr, WINDOW: tl.constexpr, DILATION: tl.constexpr,
               USE_BIAS: tl.constexpr):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    group = H // HKV
    off_hkv = off_h // group

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_row = (off_b * H + off_h) * SQ + offs_m
    q_mask = offs_m[:, None] < SQ
    q = tl.load(Q + q_row[:, None] * HEAD_DIM + offs_d[None, :], mask=q_mask, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q_pos = (SK - SQ) + offs_m
    kv_base = (off_b * HKV + off_hkv) * SK

    lo = 0
    if WINDOW > 0:
        lo = start_m * BLOCK_M + (SK - SQ) - WINDOW + 1
        lo = tl.maximum((lo // BLOCK_N) * BLOCK_N, 0)
    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M + (SK - SQ), SK)
    else:
        hi = SK

    for start_n in range(lo, hi, BLOCK_N):
        n = start_n + offs_n
        n_mask = n < SK
        k = tl.load(K + (kv_base + n)[None, :] * HEAD_DIM + offs_d[:, None],
                    mask=n_mask[None, :], other=0.0).to(tl.float32)
        qk = tl.dot(q, k) * sm_scale
        if USE_BIAS:
            b = tl.load(Bias + (off_h * SQ + offs_m[:, None]) * SK + n[None, :],
                        mask=q_mask & n_mask[None, :], other=0.0).to(tl.float32)
            qk = qk + b
        keep = n_mask[None, :]
        if IS_CAUSAL:
            keep = keep & (n[None, :] <= q_pos[:, None])
        if WINDOW > 0:
            keep = keep & (q_pos[:, None] - n[None, :] < WINDOW)
        if DILATION > 1:
            keep = keep & (n[None, :] <= q_pos[:, None]) & (((q_pos[:, None] - n[None, :]) % DILATION) == 0)
        qk = tl.where(keep, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + (kv_base + n)[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(O + q_row[:, None] * HEAD_DIM + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=q_mask)


'''

_T5_HELPER_SRC = '''def _t5_rel_bias(bias_table, SQ, SK, num_buckets, max_distance):
    """T5 bucketed relative-position bias [H, SQ, SK] (unidirectional / causal)."""
    import math
    device = bias_table.device
    i = torch.arange(SQ, device=device)[:, None]
    j = torch.arange(SK, device=device)[None, :]
    qpos = (SK - SQ) + i
    n = torch.clamp(qpos - j, min=0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    large = max_exact + (torch.log(n.float().clamp(min=1) / max_exact)
                         / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
    large = torch.clamp(large, max=num_buckets - 1)
    buckets = torch.where(is_small, n, large)
    return bias_table.float()[:, buckets].contiguous()


'''


def _wrap_dense(op: str, spec: _Attn2) -> str:
    out_dt = "torch.bfloat16" if spec.fp8 else "q.dtype"
    block_m = 16 if spec.family == "decode" else 64
    return f'''def {op}(q, k, v):
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    B, H, SQ, D = q.shape
    HKV = k.shape[1]
    SK = k.shape[2]
    o = torch.empty((B, H, SQ, D), dtype={out_dt}, device=q.device)
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_M = {block_m}
    BLOCK_N = 64
    grid = (triton.cdiv(SQ, BLOCK_M), B * H)
    _attn2_fwd[grid](
        q, k, v, o, q, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL={bool(spec.causal)}, WINDOW={int(spec.window)}, DILATION={int(spec.dilation)},
        USE_BIAS=False, num_warps=4)
    return o
'''


def _wrap_relbias(op: str, spec: _Attn2) -> str:
    return f'''def {op}(q, k, v, bias_table):
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    B, H, SQ, D = q.shape
    HKV = k.shape[1]
    SK = k.shape[2]
    bias = _t5_rel_bias(bias_table, SQ, SK, {_T5_NUM_BUCKETS}, {_T5_MAX_DIST})
    o = torch.empty((B, H, SQ, D), dtype=q.dtype, device=q.device)
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(SQ, BLOCK_M), B * H)
    _attn2_fwd[grid](
        q, k, v, o, bias, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=True, WINDOW=0, DILATION=1, USE_BIAS=True, num_warps=4)
    return o
'''


def _wrap_custommask(op: str, spec: _Attn2) -> str:
    return f'''def {op}(q, k, v, mask):
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    B, H, SQ, D = q.shape
    HKV = k.shape[1]
    SK = k.shape[2]
    bias = mask.float().unsqueeze(0).expand(H, SQ, SK).contiguous()
    o = torch.empty((B, H, SQ, D), dtype=q.dtype, device=q.device)
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(SQ, BLOCK_M), B * H)
    _attn2_fwd[grid](
        q, k, v, o, bias, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=False, WINDOW=0, DILATION=1, USE_BIAS=True, num_warps=4)
    return o
'''


def _wrap_varlen(op: str, spec: _Attn2) -> str:
    return f'''def {op}(q, k, v, cu_seqlens):
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    T, H, D = q.shape
    HKV = k.shape[1]
    o = torch.empty((T, H, D), dtype=q.dtype, device=q.device)
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_M = 64
    BLOCK_N = 64
    cu = cu_seqlens.tolist()
    for b in range(len(cu) - 1):
        s = cu[b]; e = cu[b + 1]
        L = e - s
        if L <= 0:
            continue
        qs = q[s:e].transpose(0, 1).unsqueeze(0).contiguous()
        ks = k[s:e].transpose(0, 1).unsqueeze(0).contiguous()
        vs = v[s:e].transpose(0, 1).unsqueeze(0).contiguous()
        os = torch.empty((1, H, L, D), dtype=q.dtype, device=q.device)
        grid = (triton.cdiv(L, BLOCK_M), H)
        _attn2_fwd[grid](
            qs, ks, vs, os, qs, sm_scale, H, HKV, L, L,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
            IS_CAUSAL={bool(spec.causal)}, WINDOW=0, DILATION=1, USE_BIAS=False, num_warps=4)
        o[s:e] = os[0].transpose(0, 1)
    return o
'''


def _wrap_cache(op: str, spec: _Attn2) -> str:
    return f'''def {op}(q, k_cache, v_cache, k_new, v_new):
    q = q.contiguous()
    k = torch.cat([k_cache, k_new], dim=2).contiguous()
    v = torch.cat([v_cache, v_new], dim=2).contiguous()
    B, H, SQ, D = q.shape
    HKV = k.shape[1]
    SK = k.shape[2]
    o = torch.empty((B, H, SQ, D), dtype=q.dtype, device=q.device)
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(SQ, BLOCK_M), B * H)
    _attn2_fwd[grid](
        q, k, v, o, q, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=True, WINDOW=0, DILATION=1, USE_BIAS=False, num_warps=4)
    return o
'''


def seed_source(op: str, dtype: str) -> str:
    spec = _SPECS[op]
    tldt = DTYPES[dtype][1]
    doc = (f'"""GENERATED breadth {op} seed ({dtype}). Naive but correct fused '
           f'flash-attention: one program per (query-block, batch*head) streams KV '
           f'blocks with an online (max, sum) softmax (fp32 math); {spec.family} '
           f'serving/inference variant, GQA via kv_head = head // group. The policy '
           f'fuses/tiles it. {tldt} store."""\n')
    src = doc + _SEED_HEADER + _FWD_KERNEL_SRC
    if spec.family == "relbias":
        return src + _T5_HELPER_SRC + _wrap_relbias(op, spec)
    if spec.family == "custommask":
        return src + _wrap_custommask(op, spec)
    if spec.family == "varlen":
        return src + _wrap_varlen(op, spec)
    if spec.family == "cache":
        return src + _wrap_cache(op, spec)
    return src + _wrap_dense(op, spec)


def op_names() -> list[str]:
    return list(OPS)
