"""CPU-only tests for the breadth FUSED FLASH-ATTENTION authoring engine (attn_ext).

Every ``ref_fn`` is the exact fp32 attention oracle; each is checked against an
INDEPENDENT torch computation on a DIFFERENT code path:

  * MHA/GQA/MQA, causal/non-causal, ALiBi, sliding-window, decode, fp8  ->  a
    hand-written O(S^2) fp64 attention (einsum score matrix + masked_fill +
    softmax) - a separate formulation from ref_fn's (matmul + max-subtraction);
  * softcap  ->  the same fp64 O(S^2) path with the exact Gemma tanh cap;
  * attention sink  ->  the logsumexp-with-a-virtual-sink-key formulation (a
    distinct softmax code path from ref_fn's softmax-over-augmented-then-slice);
  * backward (dQ/dK/dV)  ->  autograd through ``F.scaled_dot_product_attention``
    (fp64), an entirely independent forward+backward vs ref_fn's autograd through
    the manual fp32 core.

Also asserts: the namespace ABI matches seq.py/vendor_ops.py, arity, dtype
preservation (bf16/fp16 pass-through; fp8 -> bf16 out; backward tuple), that each
seed COMPILES + defines a top-level ``def <op>`` and a Triton kernel, that the
shape catalog round-trips through ``parse_shape``, and the hard semantics hold
(causality, sliding-window locality, softcap saturation, learned-sink probability
mass, GQA == expanded-MHA, decode == full attention). All fp32/fp64 on CPU - the
seeds are only static-checked (no GPU / triton execution needed)."""

from __future__ import annotations

import ast
import math

import pytest
import torch
import torch.nn.functional as F

from kore.tasks._genops import DTYPES
from kore.tasks.breadth import attn_ext as AT

EXPECTED_OPS = 36


# --------------------------------------------------------------------------- #
# tiny CPU shapes per op (correct head structure from the spec, small seqlens)
# --------------------------------------------------------------------------- #
def _tiny(op: str) -> dict:
    s = AT._SPECS[op]
    H = 8
    hkv = AT._kv_heads(s.variant, H)
    if s.kind == "decode":
        return {"B": 1, "H": H, "HKV": hkv, "SQ": 1, "SK": 40, "D": 32}
    return {"B": 1, "H": H, "HKV": hkv, "SQ": 24, "SK": 24, "D": 32}


def _probe(op: str) -> dict:
    """A shape that ACTIVATES the op's feature (e.g. seqlen > window) yet stays
    CPU-cheap. Uses a non-power-of-2 head_dim (48) to stress the generic path."""
    s = AT._SPECS[op]
    v = s.variant
    D = 48
    if s.kind == "decode":
        H = 8
        return {"B": 2, "H": H, "HKV": AT._kv_heads(v, H), "SQ": 1, "SK": 72, "D": D}
    if s.window:
        H = 4 if v == "gqa" else 2
        S = s.window + 64 if s.window <= 1024 else 640   # activate 1024; cap 4096
        return {"B": 1, "H": H, "HKV": AT._kv_heads(v, H), "SQ": S, "SK": S, "D": D}
    H = 8
    return {"B": 2, "H": H, "HKV": AT._kv_heads(v, H), "SQ": 40, "SK": 40, "D": D}


def _expected_arity(op: str) -> int:
    s = AT._SPECS[op]
    if s.kind == "bwd":
        return 4                                          # q, k, v, dO
    return 3 + (1 if s.alibi else 0) + (1 if s.sink else 0)


# --------------------------------------------------------------------------- #
# independent fp64 oracles (distinct code paths from ref_fn)
# --------------------------------------------------------------------------- #
def _f64(t):
    return t.float().double()                             # fp8/bf16/fp16/fp32 -> fp64


def _mask(SQ, SK, causal, window):
    i = torch.arange(SQ)[:, None]
    j = torch.arange(SK)[None, :]
    qpos = (SK - SQ) + i
    allowed = torch.ones(SQ, SK, dtype=torch.bool)
    if causal:
        allowed = allowed & (j <= qpos)
    if window > 0:
        allowed = allowed & (qpos - j < window)
    return allowed


def _alibi64(H, SQ, SK):
    h = torch.arange(1, H + 1, dtype=torch.float64)
    slopes = torch.pow(torch.tensor(2.0, dtype=torch.float64), -8.0 * h / H)
    i = torch.arange(SQ)[:, None]
    j = torch.arange(SK)[None, :]
    rel = (j - i).double()                                # b(i,j) = slope * (j - i)
    return slopes[:, None, None] * rel[None]              # [H,SQ,SK]


def _naive_fwd(q, k, v, *, causal, scale, window=0, alibi=False, softcap=0.0, sink=None):
    """Hand-written O(S^2) fp64 attention, independent of ref_fn's formulation."""
    q, k, v = _f64(q), _f64(k), _f64(v)
    H, HKV = q.shape[1], k.shape[1]
    g = H // HKV
    if g > 1:
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
    SQ, SK = q.shape[2], k.shape[2]
    s = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if softcap > 0:
        s = softcap * torch.tanh(s / softcap)
    if alibi:
        s = s + _alibi64(H, SQ, SK)[None]
    if causal or window > 0:
        s = s.masked_fill(~_mask(SQ, SK, causal, window)[None, None], float("-inf"))
    if sink is not None:
        sk = _f64(sink).view(1, H, 1, 1).expand(q.shape[0], H, SQ, 1)
        lse = torch.logsumexp(torch.cat([s, sk], dim=-1), dim=-1, keepdim=True)
        p = torch.exp(s - lse)                            # sink steals mass, V-contrib 0
    else:
        p = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", p, v)


def _independent(op, xs):
    s = AT._SPECS[op]
    scale = 1.0 / math.sqrt(xs[0].shape[-1])
    if s.kind == "bwd":
        q, k, v, do = xs
        qf = _f64(q).detach().requires_grad_(True)
        kf = _f64(k).detach().requires_grad_(True)
        vf = _f64(v).detach().requires_grad_(True)
        g = q.shape[1] // k.shape[1]
        ke = kf.repeat_interleave(g, dim=1) if g > 1 else kf
        ve = vf.repeat_interleave(g, dim=1) if g > 1 else vf
        SQ, SK = q.shape[2], k.shape[2]
        m = None
        if s.causal:
            m = torch.zeros(SQ, SK, dtype=torch.float64).masked_fill(
                ~_mask(SQ, SK, True, 0), float("-inf"))
        o = F.scaled_dot_product_attention(qf, ke, ve, attn_mask=m, scale=scale)
        (o * _f64(do)).sum().backward()
        return (qf.grad, kf.grad, vf.grad)
    # forward / decode / fp8
    idx = 3
    sink = None
    if s.alibi:
        idx += 1                                          # slopes recomputed internally
    if s.sink:
        sink = xs[idx]
        idx += 1
    return _naive_fwd(xs[0], xs[1], xs[2], causal=s.causal, scale=scale,
                      window=s.window, alibi=s.alibi, softcap=s.softcap, sink=sink)


def _close(a, b, atol=2e-3, rtol=2e-3):
    return torch.allclose(a.double(), b.double(), atol=atol, rtol=rtol)


def _as_tuple(x):
    return x if isinstance(x, (tuple, list)) else (x,)


# --------------------------------------------------------------------------- #
# metadata / ABI surface
# --------------------------------------------------------------------------- #
def test_abi_present():
    assert isinstance(AT.OPS, list) and len(AT.OPS) == EXPECTED_OPS
    assert len(set(AT.OPS)) == EXPECTED_OPS               # unique names
    assert all(op.startswith("attn_") for op in AT.OPS)
    assert callable(AT.make_reference) and callable(AT.seed_source)
    assert callable(AT.op_names) and AT.op_names() == AT.OPS
    assert set(AT.OP_DTYPES) == set(AT.OPS)
    assert set(AT.SHAPES) == set(AT.OPS)


def test_op_family_coverage():
    """The core-LLM attention families this engine widens KORE with (MLA / paged /
    varlen are intentionally OUT of this pass)."""
    names = set(AT.OPS)
    for needle in ("mha", "gqa", "mqa", "alibi", "softcap", "swa", "sink",
                   "decode", "bwd", "fp8"):
        assert any(needle in n for n in names), f"missing family {needle}"
    kinds = {AT._SPECS[op].kind for op in AT.OPS}
    assert kinds == {"fwd", "decode", "bwd"}
    for excluded in ("mla", "paged", "varlen"):
        assert not any(excluded in n for n in names), f"{excluded} must be OUT"
    # head_dim sweep {64,128,256} and both causal + non-causal are present.
    assert all(any(f"hd{d}" in n for n in names) for d in (64, 128, 256))
    assert any("noncausal" in n for n in names)


def test_op_count_by_family():
    core = [o for o in AT.OPS if AT._SPECS[o].kind == "fwd" and not AT._SPECS[o].fp8
            and not AT._SPECS[o].alibi and AT._SPECS[o].softcap == 0.0
            and not AT._SPECS[o].sink and AT._SPECS[o].window == 0]
    fp8 = [o for o in AT.OPS if AT._SPECS[o].fp8]
    bwd = [o for o in AT.OPS if AT._SPECS[o].kind == "bwd"]
    dec = [o for o in AT.OPS if AT._SPECS[o].kind == "decode"]
    assert len(core) == 18                               # 3 variants x 3 hd x 2 causal
    assert len(fp8) == 3 and len(bwd) == 3 and len(dec) == 3
    assert 2 <= len(fp8) <= 3                             # "2-3 fp8 ops"


def test_ops_dtypes_shapes_consistent():
    for op in AT.OPS:
        dts = AT.OP_DTYPES[op]
        assert dts and AT.op_dtypes(op) == dts
        for d in dts:
            assert d in DTYPES, f"unknown dtype {d} for {op}"
        if AT._SPECS[op].fp8:
            assert dts == ["fp8"]
        else:
            assert dts == AT.DEFAULT_DTYPES == ["bf16", "fp16"]
        sh = AT.SHAPES[op]
        assert "minimal" in sh and "primary" in sh and "validation" in sh
        assert isinstance(sh["validation"], list) and sh["validation"]


@pytest.mark.parametrize("op", AT.OPS)
def test_namespace_contract(op):
    dt = AT.OP_DTYPES[op][0]
    ns = AT.make_reference(op, dt)
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family", "mutates_input"):
        assert k in ns, k
    assert ns["entry_name"] == op
    assert ns["dtype_name"] == dt
    assert ns["family"] == f"breadth_{op}"
    assert ns["mutates_input"] is False
    assert callable(ns["ref_fn"]) and callable(ns["baseline_fn"])
    assert ns[f"{op}_ref"] is ns["ref_fn"]


@pytest.mark.parametrize("op", AT.OPS)
def test_arity(op):
    dt = AT.OP_DTYPES[op][0]
    ns = AT.make_reference(op, dt)
    assert ns["arity"] == _expected_arity(op)
    inputs = ns["get_inputs"](_tiny(op), device="cpu", seed=0)
    assert isinstance(inputs, tuple) and len(inputs) == ns["arity"]


# --------------------------------------------------------------------------- #
# fp32 oracle correctness vs an INDEPENDENT torch compute (all 36 ops)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", AT.OPS)
def test_ref_matches_independent(op):
    s = AT._SPECS[op]
    dt = "fp8" if s.fp8 else "fp32"
    ns = AT.make_reference(op, dt)
    xs = ns["get_inputs"](_probe(op), device="cpu", seed=0)
    ref = _as_tuple(ns["ref_fn"](*xs))
    ind = _as_tuple(_independent(op, xs))
    assert len(ref) == len(ind)
    atol = 3e-2 if s.fp8 else 2e-3                        # fp8 -> bf16 output rounding
    for r, i in zip(ref, ind):
        assert r.shape == i.shape, f"{op}: {tuple(r.shape)} vs {tuple(i.shape)}"
        assert _close(r, i, atol=atol, rtol=atol), (
            f"{op}: max|diff|={(r.double() - i.double()).abs().max().item():.3e}")


@pytest.mark.parametrize("op", AT.OPS)
def test_baseline_matches_ref(op):
    """The torch eager baseline (fp32) agrees with the fp32 oracle (fp8 -> bf16)."""
    s = AT._SPECS[op]
    dt = "fp8" if s.fp8 else "fp32"
    ns = AT.make_reference(op, dt)
    xs = ns["get_inputs"](_tiny(op), device="cpu", seed=1)
    out = _as_tuple(ns["baseline_fn"](*xs))
    ref = _as_tuple(ns["ref_fn"](*xs))
    atol = 3e-2 if s.fp8 else 2e-3
    for o, r in zip(out, ref):
        assert o.shape == r.shape
        assert _close(o, r, atol=atol, rtol=atol), f"{op}: baseline != ref"


@pytest.mark.parametrize("op", AT.OPS)
def test_ref_preserves_output_dtype(op):
    """bf16/fp16 pass through; fp8 attention emits bf16; backward returns a tuple."""
    for dt in AT.OP_DTYPES[op]:
        ns = AT.make_reference(op, dt)
        xs = ns["get_inputs"](_tiny(op), device="cpu", seed=2)
        outs = _as_tuple(ns["ref_fn"](*xs))
        exp = torch.bfloat16 if dt == "fp8" else getattr(torch, DTYPES[dt][0])
        assert all(o.dtype == exp for o in outs), (op, dt, [o.dtype for o in outs])
        if AT._SPECS[op].kind == "bwd":
            assert len(outs) == 3                         # (dQ, dK, dV)


# --------------------------------------------------------------------------- #
# seed static checks (compiles + defines a top-level entry fn + a triton kernel)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", AT.OPS)
def test_seed_compiles_and_defines_entry(op):
    for dt in AT.op_dtypes(op):
        src = AT.seed_source(op, dt)
        compile(src, f"<{op}_{dt}_seed>", "exec")          # valid Python (COMPILING)
        tree = ast.parse(src)
        top = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        assert any(n.name == op for n in top), f"{op}: entry must be a top-level def"
        fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        need = "_attn_bwd_dq" if AT._SPECS[op].kind == "bwd" else "_attn_fwd"
        assert need in fns, f"{op}: seed must define the {need} triton kernel"
        assert "@triton.jit" in src and "online" in src.lower()


@pytest.mark.parametrize("op", AT.OPS)
def test_shapes_parse_roundtrip(op):
    ns = AT.make_reference(op, AT.op_dtypes(op)[0])
    parse = ns["parse_shape"]
    sh = AT.SHAPES[op]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        assert all(isinstance(x, int) for x in spec.values())
        assert spec["H"] % spec["HKV"] == 0               # valid GQA grouping
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert parse(s) == spec, (op, parse(s), spec)


# --------------------------------------------------------------------------- #
# hard semantic / numeric checks
# --------------------------------------------------------------------------- #
def test_causal_masks_the_future():
    """A causal op's early query outputs must not depend on future keys/values."""
    ns = AT.make_reference("attn_mha_hd128_causal", "fp32")
    q, k, v = ns["get_inputs"]({"B": 1, "H": 2, "HKV": 2, "SQ": 16, "SK": 16, "D": 64},
                               device="cpu", seed=0)
    t0 = 6
    k2, v2 = k.clone(), v.clone()
    k2[:, :, t0 + 1:] += 5.0
    v2[:, :, t0 + 1:] += 5.0
    y = ns["ref_fn"](q, k, v)
    y2 = ns["ref_fn"](q, k2, v2)
    assert _close(y[:, :, : t0 + 1], y2[:, :, : t0 + 1])
    assert not _close(y[:, :, t0 + 1:], y2[:, :, t0 + 1:])   # the future DID change


def test_noncausal_sees_the_future():
    ns = AT.make_reference("attn_mha_hd128_noncausal", "fp32")
    q, k, v = ns["get_inputs"]({"B": 1, "H": 2, "HKV": 2, "SQ": 16, "SK": 16, "D": 64},
                               device="cpu", seed=0)
    v2 = v.clone()
    v2[:, :, -1:] += 5.0
    assert not _close(ns["ref_fn"](q, k, v)[:, :, 0], ns["ref_fn"](q, k, v2)[:, :, 0])


def test_sliding_window_is_local():
    """Sliding-window attention ignores keys older than `window` (locality)."""
    op = "attn_swa1024_mha_causal"
    ns = AT.make_reference(op, "fp32")
    W = AT._SPECS[op].window
    S = W + 80
    q, k, v = ns["get_inputs"]({"B": 1, "H": 2, "HKV": 2, "SQ": S, "SK": S, "D": 64},
                               device="cpu", seed=0)
    assert _close(ns["ref_fn"](q, k, v), _independent(op, (q, k, v)))
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 0] += 9.0
    v2[:, :, 0] += 9.0
    y = ns["ref_fn"](q, k, v)
    y2 = ns["ref_fn"](q, k2, v2)
    assert _close(y[:, :, -1], y2[:, :, -1])                 # last query: key 0 is a no-op
    assert not _close(y[:, :, 0], y2[:, :, 0])               # first query DID see key 0


def test_softcap_saturates_large_logits():
    """With large logits the Gemma tanh cap bounds the pre-softmax scores; ref must
    match the exact-cap oracle AND differ from the uncapped attention."""
    op = "attn_softcap_mha_causal"
    ns = AT.make_reference(op, "fp32")
    cap = AT._SPECS[op].softcap
    D = 64
    scale = 1.0 / math.sqrt(D)
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, 2, 16, D, generator=g) * 8.0         # -> logit std >> cap
    k = torch.randn(2, 2, 16, D, generator=g) * 8.0
    v = torch.randn(2, 2, 16, D, generator=g)
    ref = ns["ref_fn"](q, k, v)
    capped = _naive_fwd(q, k, v, causal=True, scale=scale, softcap=cap)
    uncapped = _naive_fwd(q, k, v, causal=True, scale=scale, softcap=0.0)
    assert (q.double() @ k.double().transpose(-1, -2) * scale).std() > cap
    assert _close(ref, capped)
    assert not _close(ref, uncapped)                         # softcap actually bites


def test_attention_sink_absorbs_probability_mass():
    """A large learned sink logit steals softmax mass, shrinking the output; ref must
    match the logsumexp-with-virtual-sink oracle."""
    op = "attn_sink_mha_causal"
    ns = AT.make_reference(op, "fp32")
    shape = {"B": 2, "H": 4, "HKV": 4, "SQ": 12, "SK": 12, "D": 64}
    q, k, v, sink = ns["get_inputs"](shape, device="cpu", seed=0)
    assert _close(ns["ref_fn"](q, k, v, sink), _independent(op, (q, k, v, sink)))
    small = ns["ref_fn"](q, k, v, sink - 100.0)              # sink negligible
    big = ns["ref_fn"](q, k, v, sink + 100.0)                # sink dominates -> ~0 out
    assert big.abs().max() < small.abs().max()
    assert torch.allclose(big.double(), torch.zeros_like(big.double()), atol=1e-3)


def test_gqa_equals_expanded_mha():
    """GQA attention == dense MHA on the group-expanded K/V (the defining identity)."""
    ns = AT.make_reference("attn_gqa_hd128_causal", "fp32")
    H, HKV, D = 8, 2, 64
    q, k, v = ns["get_inputs"]({"B": 2, "H": H, "HKV": HKV, "SQ": 16, "SK": 16, "D": D},
                               device="cpu", seed=0)
    g = H // HKV
    ke, ve = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    mns = AT.make_reference("attn_mha_hd128_causal", "fp32")
    assert _close(ns["ref_fn"](q, k, v), mns["ref_fn"](q, ke, ve))


def test_ref_matches_sdpa_plain():
    """Cross-check a plain causal op against torch SDPA (a fully independent kernel)."""
    ns = AT.make_reference("attn_mha_hd128_causal", "fp32")
    q, k, v = ns["get_inputs"]({"B": 2, "H": 4, "HKV": 4, "SQ": 32, "SK": 32, "D": 64},
                               device="cpu", seed=3)
    sdpa = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)
    assert _close(ns["ref_fn"](q, k, v), sdpa)


def test_decode_is_full_attention_over_kv():
    """Decode (q_len == 1) attends the WHOLE KV cache (no causal masking)."""
    dec = AT.make_reference("attn_decode_gqa", "fp32")
    q, k, v = dec["get_inputs"]({"B": 2, "H": 8, "HKV": 2, "SQ": 1, "SK": 48, "D": 64},
                                device="cpu", seed=0)
    g = 4
    ke, ve = k.repeat_interleave(g, dim=1).double(), v.repeat_interleave(g, dim=1).double()
    full = F.scaled_dot_product_attention(q.double(), ke, ve, is_causal=False)
    assert _close(dec["ref_fn"](q, k, v), full)


def test_backward_matches_sdpa_autograd():
    """dQ/dK/dV from ref_fn autograd == autograd through independent SDPA (GQA)."""
    op = "attn_bwd_gqa_causal"
    ns = AT.make_reference(op, "fp32")
    xs = ns["get_inputs"]({"B": 2, "H": 8, "HKV": 2, "SQ": 24, "SK": 24, "D": 64},
                          device="cpu", seed=0)
    ref = ns["ref_fn"](*xs)
    ind = _independent(op, xs)
    assert len(ref) == len(ind) == 3
    for r, i in zip(ref, ind):
        assert r.shape == i.shape
        assert _close(r, i)
