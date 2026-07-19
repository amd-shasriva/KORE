"""Breadth FUSED FLASH-ATTENTION task-authoring engine (torch-baselined).

Widens the KORE suite with the *core LLM* attention kernels that the sibling
breadth engines (sequence models / conv / sort / train ops) never covered: the
hard, fused, online-/streaming-softmax flash-attention family. The naive form
materializes the full ``S x S`` score matrix (quadratic memory, memory-bound), so
a fused Triton tile that streams KV blocks with a running (max, sum) softmax and
never writes the scores has genuine headroom over the torch-eager baseline.

Unlike the vendor tasks (graded against AITER), these grade against the honest
torch reference: the correctness ORACLE is a torch fp32 attention (``ref_fn``,
softmax(QK^T*scale + bias/mask) V computed in fp32 then cast back) and the perf
BASELINE is the eager torch computation (``baseline_fn``).

Op families (all names prefixed ``attn_``)
------------------------------------------
  * core grid    - MHA / GQA / MQA  x  head_dim {64,128,256}  x  {causal, non-causal}
  * ALiBi        - linear-distance positional bias  b(i,j) = slope_h * (j - i)
  * softcap      - Gemma-2 logit soft-capping  s <- C * tanh(s / C)
  * sliding win. - local attention over the last ``W`` keys (W in {1024, 4096})
  * attn. sink   - gpt-oss style per-head sink logit in the softmax denominator
  * decode       - q_len == 1 against a long KV cache (MHA / GQA / MQA)
  * backward     - fused dQ/dK/dV for causal attention (MHA / GQA / MQA)
  * fp8          - e4m3 (OCP e4m3fn, CDNA4/gfx950) attention, bf16 output

Contract mirrors ``kore/tasks/breadth/seq.py`` (and ``kore/tasks/vendor_ops.py``)
so the shared ``_genops`` driver + the generator consume it unchanged:

    OPS / OP_DTYPES / SHAPES              module-level task catalog
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn fp32 oracle, baseline_fn torch, arity, entry_name,
        dtype_name, family=f"breadth_{op}", mutates_input)
    seed_source(op, dtype) -> str         a naive, COMPILING, correct online-softmax
        flash-attention Triton seed (defines ``def <op>(*inputs)``).

CORRECTNESS is paramount: every ``ref_fn`` computes in fp32 and casts back, and is
validated on CPU against an INDEPENDENT torch path (F.scaled_dot_product_attention
with a matching additive bias/scale, or a hand-written O(S^2) softmax for the
softcap/sink variants; backward via autograd through SDPA) at tight fp32 tol. The
Triton seeds are correct-but-naive flash tiles (block over queries, stream KV with
online softmax). torch/triton are imported lazily (registry discovery needs no GPU).
"""

from __future__ import annotations

from dataclasses import dataclass

from kore.tasks._genops import DTYPES, _parse_shape


# --------------------------------------------------------------------------- #
# per-op specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Attn:
    variant: str            # "mha" | "gqa" | "mqa"
    head_dim: int           # 64 | 128 | 256
    causal: bool
    kind: str = "fwd"       # "fwd" | "decode" | "bwd"
    window: int = 0         # 0 = full attention; else sliding-window width W
    alibi: bool = False
    softcap: float = 0.0    # 0 = disabled; else tanh soft-cap value C
    sink: bool = False
    fp8: bool = False


def _kv_heads(variant: str, H: int) -> int:
    """KV-head count for the attention variant (GQA uses a group of 4)."""
    if variant == "mha":
        return H
    if variant == "mqa":
        return 1
    return max(1, H // 4)


# --------------------------------------------------------------------------- #
# task catalog (exactly 36 ops)
# --------------------------------------------------------------------------- #
_SPECS: dict[str, _Attn] = {}


def _add(name: str, spec: _Attn) -> None:
    assert name.startswith("attn_"), name
    assert name not in _SPECS, name
    _SPECS[name] = spec


# --- core grid: {mha,gqa,mqa} x head_dim {64,128,256} x {causal,non-causal} (18)
for _variant in ("mha", "gqa", "mqa"):
    for _hd in (64, 128, 256):
        for _causal in (True, False):
            _suffix = "causal" if _causal else "noncausal"
            _add(f"attn_{_variant}_hd{_hd}_{_suffix}", _Attn(_variant, _hd, _causal))

# --- bias / feature variants (all causal, head_dim 128) (9)
_add("attn_alibi_mha_causal", _Attn("mha", 128, True, alibi=True))
_add("attn_alibi_gqa_causal", _Attn("gqa", 128, True, alibi=True))
_add("attn_softcap_mha_causal", _Attn("mha", 128, True, softcap=30.0))
_add("attn_softcap_gqa_causal", _Attn("gqa", 128, True, softcap=30.0))
_add("attn_swa1024_mha_causal", _Attn("mha", 128, True, window=1024))
_add("attn_swa1024_gqa_causal", _Attn("gqa", 128, True, window=1024))
_add("attn_swa4096_gqa_causal", _Attn("gqa", 128, True, window=4096))
_add("attn_sink_mha_causal", _Attn("mha", 128, True, sink=True))
_add("attn_sink_gqa_causal", _Attn("gqa", 128, True, sink=True))

# --- decode: q_len == 1 against a long KV cache (head_dim 128) (3)
_add("attn_decode_mha", _Attn("mha", 128, False, kind="decode"))
_add("attn_decode_gqa", _Attn("gqa", 128, False, kind="decode"))
_add("attn_decode_mqa", _Attn("mqa", 128, False, kind="decode"))

# --- backward dQ/dK/dV for causal attention (head_dim 128) (3)
_add("attn_bwd_mha_causal", _Attn("mha", 128, True, kind="bwd"))
_add("attn_bwd_gqa_causal", _Attn("gqa", 128, True, kind="bwd"))
_add("attn_bwd_mqa_causal", _Attn("mqa", 128, True, kind="bwd"))

# --- fp8 e4m3 attention (bf16 output) for core causal configs (3)
_add("attn_fp8_mha_hd128_causal", _Attn("mha", 128, True, fp8=True))
_add("attn_fp8_gqa_hd128_causal", _Attn("gqa", 128, True, fp8=True))
_add("attn_fp8_mha_hd64_causal", _Attn("mha", 64, True, fp8=True))

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
# realistic LLM attention shapes (B in {1,2,4,8}, heads {16,32}, GQA/MQA kv-heads,
# seqlen {512,2048,4096,...}, head_dim {64,128,256}, non-power-of-2 seqlen tail).
# Layout: q[B,H,SQ,D], k/v[B,HKV,SK,D] -> out[B,H,SQ,D]. Self-attention has SQ==SK;
# decode has SQ==1 against a long SK. Minimal shapes stay CPU-cheap for unit tests.
# --------------------------------------------------------------------------- #
def _shapes_for(spec: _Attn) -> dict:
    D = spec.head_dim
    v = spec.variant

    def mk(B, H, SQ, SK):
        return {"B": B, "H": H, "HKV": _kv_heads(v, H), "SQ": SQ, "SK": SK, "D": D}

    if spec.kind == "decode":
        return {
            "minimal": mk(1, 8, 1, 128),
            "primary": mk(4, 32, 1, 4096),
            "validation": [
                mk(1, 32, 1, 8192),
                mk(8, 16, 1, 2048),
                mk(2, 32, 1, 2047),      # non-pow2 KV tail
            ],
        }
    # forward / backward self-attention (SQ == SK)
    return {
        "minimal": mk(1, 8, 64, 64),
        "primary": mk(2, 32, 2048, 2048),
        "validation": [
            mk(4, 16, 512, 512),
            mk(1, 32, 4096, 4096),
            mk(2, 16, 2047, 2047),       # non-pow2 seqlen tail
        ],
    }


SHAPES: dict[str, dict] = {op: _shapes_for(_SPECS[op]) for op in OPS}


# --------------------------------------------------------------------------- #
# ALiBi slopes (the geometric sequence of the ALiBi paper)
# --------------------------------------------------------------------------- #
def _alibi_slopes(n_heads: int) -> list[float]:
    import math

    def pow2(n):
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return pow2(n_heads)
    closest = 2 ** int(math.floor(math.log2(n_heads)))
    slopes = pow2(closest)
    slopes += pow2(2 * closest)[0::2][: n_heads - closest]
    return slopes


# --------------------------------------------------------------------------- #
# fp32-capable attention core (shared by ref_fn / baseline_fn). Operates on the
# tensors AS GIVEN (ref_fn upcasts to fp32 first; baseline_fn keeps task dtype).
# --------------------------------------------------------------------------- #
def _attn_core(q, k, v, spec: _Attn, slopes=None, sink=None):
    import torch

    B, H, SQ, D = q.shape
    HKV = k.shape[1]
    SK = k.shape[2]
    group = H // HKV
    if group > 1:                                   # GQA / MQA broadcast of KV heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale      # [B,H,SQ,SK]

    if spec.softcap > 0.0:                          # Gemma-2 logit soft-cap
        c = spec.softcap
        scores = c * torch.tanh(scores / c)

    if spec.alibi:                                  # b(i,j) = slope_h * (j - i)
        i = torch.arange(SQ, device=q.device)
        j = torch.arange(SK, device=q.device)
        rel = (j[None, :] - i[:, None]).to(scores.dtype)       # [SQ,SK]
        scores = scores + slopes.view(1, H, 1, 1) * rel[None, None]

    if spec.causal or spec.window > 0:              # causal / sliding-window mask
        i = torch.arange(SQ, device=q.device)[:, None]
        j = torch.arange(SK, device=q.device)[None, :]
        qpos = (SK - SQ) + i                        # absolute query position
        allowed = torch.ones((SQ, SK), dtype=torch.bool, device=q.device)
        if spec.causal:
            allowed = allowed & (j <= qpos)
        if spec.window > 0:
            allowed = allowed & (qpos - j < spec.window)
        scores = scores.masked_fill(~allowed[None, None], torch.finfo(scores.dtype).min)

    if spec.sink:                                   # gpt-oss per-head sink logit
        col = sink.view(1, H, 1, 1).expand(B, H, SQ, 1).to(scores.dtype)
        aug = torch.cat([scores, col], dim=-1)      # extra key, V-contribution == 0
        p = torch.softmax(aug, dim=-1)[..., :SK]
    else:
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

    def _randn(shape_, device, seed, scale=1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape_, generator=g, device=device,
                            dtype=torch.float32) * scale).to(tdt)

    def get_inputs(shape, device="cuda", seed=0):
        B, H, HKV = shape["B"], shape["H"], shape["HKV"]
        SQ, SK, D = shape["SQ"], shape["SK"], shape["D"]
        q = _randn((B, H, SQ, D), device, seed)
        k = _randn((B, HKV, SK, D), device, seed + 1)
        v = _randn((B, HKV, SK, D), device, seed + 2)
        if spec.kind == "bwd":
            do = _randn((B, H, SQ, D), device, seed + 5)     # upstream gradient
            return (q, k, v, do)
        extra = []
        if spec.alibi:
            sl = torch.tensor(_alibi_slopes(H), dtype=torch.float32, device=device).to(tdt)
            extra.append(sl)
        if spec.sink:
            extra.append(_randn((H,), device, seed + 3))
        return (q, k, v, *extra)

    def _split(xs):
        q, k, v = xs[0], xs[1], xs[2]
        idx = 3
        slopes = sink = None
        if spec.alibi:
            slopes = xs[idx]; idx += 1
        if spec.sink:
            sink = xs[idx]; idx += 1
        return q, k, v, slopes, sink

    if spec.kind == "bwd":
        def ref_fn(q, k, v, do):
            qf = q.float().requires_grad_(True)
            kf = k.float().requires_grad_(True)
            vf = v.float().requires_grad_(True)
            o = _attn_core(qf, kf, vf, spec)
            dq, dk, dv = torch.autograd.grad(o, (qf, kf, vf), grad_outputs=do.float())
            return (dq.to(tdt), dk.to(tdt), dv.to(tdt))

        def baseline_fn(q, k, v, do):
            qf = q.detach().clone().requires_grad_(True)
            kf = k.detach().clone().requires_grad_(True)
            vf = v.detach().clone().requires_grad_(True)
            o = _attn_core(qf, kf, vf, spec)
            dq, dk, dv = torch.autograd.grad(o, (qf, kf, vf), grad_outputs=do)
            return (dq, dk, dv)

        arity = 4
    else:
        def ref_fn(*xs):
            q, k, v, slopes, sink = _split(xs)
            q, k, v = q.float(), k.float(), v.float()
            slopes = slopes.float() if slopes is not None else None
            sink = sink.float() if sink is not None else None
            return _attn_core(q, k, v, spec, slopes, sink).to(out_dtype)

        def baseline_fn(*xs):
            q, k, v, slopes, sink = _split(xs)
            cdt = torch.bfloat16 if spec.fp8 else q.dtype     # fp8 can't matmul: bf16
            q, k, v = q.to(cdt), k.to(cdt), v.to(cdt)
            slopes = slopes.to(cdt) if slopes is not None else None
            sink = sink.to(cdt) if sink is not None else None
            return _attn_core(q, k, v, spec, slopes, sink).to(out_dtype)

        arity = 3 + (1 if spec.alibi else 0) + (1 if spec.sink else 0)

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}", "mutates_input": False}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive (correct, compiling) Triton flash-attention seeds - the policy's start.
# Forward: one program per (query-block, batch*head); stream KV blocks with an
# online (max, sum) softmax; fp32 math; optional causal / sliding-window / ALiBi /
# soft-cap / sink; GQA via kv_head = head // group. Backward: recompute LSE then
# the standard dQ / dK / dV flash passes. Correct-but-naive; the policy tunes it.
# --------------------------------------------------------------------------- #
_SEED_HEADER = (
    "from __future__ import annotations\n"
    "import torch\n"
    "import triton\n"
    "import triton.language as tl\n\n\n"
)

_FWD_KERNEL_SRC = '''@triton.jit
def _attn_fwd(Q, K, V, O, Slopes, Sink, sm_scale, H, HKV, SQ, SK,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
              IS_CAUSAL: tl.constexpr, WINDOW: tl.constexpr,
              USE_ALIBI: tl.constexpr, SOFTCAP: tl.constexpr, USE_SINK: tl.constexpr):
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

    slope = 0.0
    if USE_ALIBI:
        slope = tl.load(Slopes + off_h).to(tl.float32)

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
        if SOFTCAP > 0.0:
            qk = SOFTCAP * tl.math.tanh(qk / SOFTCAP)
        if USE_ALIBI:
            qk = qk + slope * (n[None, :] - q_pos[:, None])
        keep = n_mask[None, :]
        if IS_CAUSAL:
            keep = keep & (n[None, :] <= q_pos[:, None])
        if WINDOW > 0:
            keep = keep & (q_pos[:, None] - n[None, :] < WINDOW)
        qk = tl.where(keep, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + (kv_base + n)[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_ij

    if USE_SINK:
        s = tl.load(Sink + off_h).to(tl.float32)
        l_i = l_i + tl.exp(s - m_i)

    acc = acc / l_i[:, None]
    tl.store(O + q_row[:, None] * HEAD_DIM + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=q_mask)


'''

_BWD_KERNEL_SRC = '''@triton.jit
def _attn_fwd_lse(Q, K, V, O, LSE, sm_scale, H, HKV, SQ, SK,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
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
    kv_base = (off_b * HKV + off_hkv) * SK
    hi = tl.minimum((start_m + 1) * BLOCK_M, SK)
    for start_n in range(0, hi, BLOCK_N):
        n = start_n + offs_n
        n_mask = n < SK
        k = tl.load(K + (kv_base + n)[None, :] * HEAD_DIM + offs_d[:, None],
                    mask=n_mask[None, :], other=0.0).to(tl.float32)
        qk = tl.dot(q, k) * sm_scale
        keep = n_mask[None, :] & (n[None, :] <= offs_m[:, None])
        qk = tl.where(keep, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + (kv_base + n)[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_ij
    tl.store(O + q_row[:, None] * HEAD_DIM + offs_d[None, :],
             (acc / l_i[:, None]).to(O.dtype.element_ty), mask=q_mask)
    tl.store(LSE + q_row, m_i + tl.log(l_i), mask=offs_m < SQ)


@triton.jit
def _attn_bwd_dq(Q, K, V, DO, LSE, Delta, DQ, sm_scale, H, HKV, SQ, SK,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
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
    m_mask = offs_m[:, None] < SQ
    q = tl.load(Q + q_row[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
    do = tl.load(DO + q_row[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
    lse = tl.load(LSE + q_row, mask=offs_m < SQ, other=0.0)
    delta = tl.load(Delta + q_row, mask=offs_m < SQ, other=0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    kv_base = (off_b * HKV + off_hkv) * SK
    hi = tl.minimum((start_m + 1) * BLOCK_M, SK)
    for start_n in range(0, hi, BLOCK_N):
        n = start_n + offs_n
        n_mask = n < SK
        k = tl.load(K + (kv_base + n)[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + (kv_base + n)[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(qk - lse[:, None])
        keep = n_mask[None, :] & (n[None, :] <= offs_m[:, None])
        p = tl.where(keep, p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds, k)
    tl.store(DQ + q_row[:, None] * HEAD_DIM + offs_d[None, :], dq * sm_scale, mask=m_mask)


@triton.jit
def _attn_bwd_dkdv(Q, K, V, DO, LSE, Delta, DK, DV, sm_scale, H, HKV, SQ, SK,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    start_n = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    group = H // HKV
    off_hkv = off_h // group
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    n_mask = offs_n[:, None] < SK
    kv_base = (off_b * HKV + off_hkv) * SK
    k = tl.load(K + (kv_base + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=n_mask, other=0.0).to(tl.float32)
    v = tl.load(V + (kv_base + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
                mask=n_mask, other=0.0).to(tl.float32)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    q_head = (off_b * H + off_h) * SQ
    lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    for start_m in range(lo, SQ, BLOCK_M):
        m = start_m + offs_m
        m_mask = m[:, None] < SQ
        q = tl.load(Q + (q_head + m)[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
        do = tl.load(DO + (q_head + m)[:, None] * HEAD_DIM + offs_d[None, :], mask=m_mask, other=0.0).to(tl.float32)
        lse = tl.load(LSE + q_head + m, mask=m < SQ, other=0.0)
        delta = tl.load(Delta + q_head + m, mask=m < SQ, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(qk - lse[:, None])
        keep = (offs_n[None, :] <= m[:, None]) & (offs_n[None, :] < SK) & (m[:, None] < SQ)
        p = tl.where(keep, p, 0.0)
        dv += tl.dot(tl.trans(p), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds), q)
    dk_row = (off_b * H + off_h) * SK + offs_n
    tl.store(DK + dk_row[:, None] * HEAD_DIM + offs_d[None, :], dk * sm_scale, mask=n_mask)
    tl.store(DV + dk_row[:, None] * HEAD_DIM + offs_d[None, :], dv, mask=n_mask)


'''


def _fwd_wrapper(op: str, spec: _Attn) -> str:
    out_dt = "torch.bfloat16" if spec.fp8 else "q.dtype"
    block_m = 16 if spec.kind == "decode" else 64
    params = "q, k, v"
    slopes_arg = "q"
    sink_arg = "q"
    if spec.alibi:
        params += ", slopes"
        slopes_arg = "slopes"
    if spec.sink:
        params += ", sink"
        sink_arg = "sink"
    return f'''def {op}({params}):
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    B, H, SQ, D = q.shape
    HKV = k.shape[1]
    SK = k.shape[2]
    o = torch.empty((B, H, SQ, D), dtype={out_dt}, device=q.device)
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_M = {block_m}
    BLOCK_N = 64
    grid = (triton.cdiv(SQ, BLOCK_M), B * H)
    _attn_fwd[grid](
        q, k, v, o, {slopes_arg}, {sink_arg}, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL={bool(spec.causal)}, WINDOW={int(spec.window)},
        USE_ALIBI={bool(spec.alibi)}, SOFTCAP={float(spec.softcap)}, USE_SINK={bool(spec.sink)},
        num_warps=4)
    return o
'''


def _bwd_wrapper(op: str, spec: _Attn) -> str:
    return f'''def {op}(q, k, v, do):
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); do = do.contiguous()
    B, H, SQ, D = q.shape
    HKV = k.shape[1]
    SK = k.shape[2]
    group = H // HKV
    sm_scale = 1.0 / (D ** 0.5)
    o = torch.empty((B, H, SQ, D), dtype=q.dtype, device=q.device)
    lse = torch.empty((B, H, SQ), dtype=torch.float32, device=q.device)
    BLOCK_M = 64
    BLOCK_N = 64
    _attn_fwd_lse[(triton.cdiv(SQ, BLOCK_M), B * H)](
        q, k, v, o, lse, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, num_warps=4)
    delta = (o.float() * do.float()).sum(-1).contiguous()
    dq = torch.zeros((B, H, SQ, D), dtype=torch.float32, device=q.device)
    dkf = torch.zeros((B, H, SK, D), dtype=torch.float32, device=q.device)
    dvf = torch.zeros((B, H, SK, D), dtype=torch.float32, device=q.device)
    _attn_bwd_dq[(triton.cdiv(SQ, BLOCK_M), B * H)](
        q, k, v, do, lse, delta, dq, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, num_warps=4)
    _attn_bwd_dkdv[(triton.cdiv(SK, BLOCK_N), B * H)](
        q, k, v, do, lse, delta, dkf, dvf, sm_scale, H, HKV, SQ, SK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, num_warps=4)
    dk = dkf.view(B, HKV, group, SK, D).sum(2).to(k.dtype)
    dv = dvf.view(B, HKV, group, SK, D).sum(2).to(v.dtype)
    return dq.to(q.dtype), dk, dv
'''


def seed_source(op: str, dtype: str) -> str:
    spec = _SPECS[op]
    tldt = DTYPES[dtype][1]
    if spec.kind == "bwd":
        doc = (f'"""GENERATED breadth {op} seed ({dtype}). Fused flash-attention BACKWARD '
               f'(dQ/dK/dV) for causal attention: recompute the online-softmax log-sum-exp, '
               f'then the standard flash dQ and dK/dV passes (GQA/MQA reduce over the kv '
               f'group). Naive but correct; the policy fuses/tiles it. {tldt} store."""\n')
        return doc + _SEED_HEADER + _BWD_KERNEL_SRC + _bwd_wrapper(op, spec)
    doc = (f'"""GENERATED breadth {op} seed ({dtype}). Fused flash-attention forward: one '
           f'program per (query-block, batch*head) streams KV blocks with an online '
           f'(max, sum) softmax (fp32 math), GQA via kv_head = head // group; optional '
           f'causal / sliding-window / ALiBi / soft-cap / sink. Naive but correct; the '
           f'policy tunes the tiling. {tldt} store."""\n')
    return doc + _SEED_HEADER + _FWD_KERNEL_SRC + _fwd_wrapper(op, spec)


def op_names() -> list[str]:
    return list(OPS)
