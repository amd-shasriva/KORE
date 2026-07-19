"""CPU-only tests for the breadth SERVING/INFERENCE ATTENTION engine (attn2_ext).

Every ``ref_fn`` is the exact fp32 attention oracle for its serving variant; each
is checked against an INDEPENDENT torch computation on a DIFFERENT code path:

  * dense (chunked / cross / window / dilated / decode / fp8)  ->  a hand-written
    O(S^2) fp64 attention (einsum score matrix + masked_fill + softmax) - a
    separate formulation from ref_fn's (matmul + finfo-min mask + max-subtraction);
  * varlen  ->  a per-sequence loop of einsum attention over each cu_seqlens slice
    (asserting NO cross-sequence attention leaks);
  * cache-append  ->  attention over the CONCATENATED [cache, new] KV via
    F.scaled_dot_product_attention (a fully independent kernel);
  * relbias  ->  a T5 bucket computed by an O(S^2) PYTHON double-loop (distinct from
    ref_fn's vectorized bucket + gather);
  * custommask  ->  the same additive float mask added on the einsum path.

Also asserts: the namespace ABI matches attn_ext.py / seq.py / vendor_ops.py, arity,
dtype preservation (bf16/fp16 pass-through; fp8 -> bf16 out), that each seed COMPILES
+ defines a top-level ``def <op>`` and a Triton kernel, that the shape catalog
round-trips through ``parse_shape``, the RESERVED families (mla / paged / latent) are
absent, and the hard serving semantics hold (varlen per-sequence isolation, chunked
causal alignment, cross full-encoder attention, cache == concat-full, window locality,
dilated striding, GQA == expanded MHA, decode == full attention, T5 bias bites,
custom mask bites). All fp32/fp64 on CPU - the seeds are only static-checked."""

from __future__ import annotations

import ast
import math
from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from kore.tasks._genops import DTYPES
from kore.tasks.breadth import attn2_ext as AT

EXPECTED_OPS = 32


# --------------------------------------------------------------------------- #
# tiny / probe CPU shapes per family (correct head structure, small seqlens)
# --------------------------------------------------------------------------- #
def _tiny(op: str) -> dict:
    s = AT._SPECS[op]
    fam = s.family
    H = 8
    hk = AT._kv_heads(s.variant, H)
    D = 32
    if fam == "varlen":
        return {"B": 2, "H": H, "HKV": hk, "S": 20, "D": D}
    if fam == "cache":
        return {"B": 1, "H": H, "HKV": hk, "SKctx": 24, "SQ": 8, "D": D}
    if fam == "decode":
        return {"B": 1, "H": H, "HKV": hk, "SQ": 1, "SK": 40, "D": D}
    if fam == "cross":
        return {"B": 1, "H": H, "HKV": hk, "SQ": (1 if s.step else 12), "SK": 20, "D": D}
    if fam == "chunked":
        return {"B": 1, "H": H, "HKV": hk, "SQ": 12, "SK": 28, "D": D}
    return {"B": 1, "H": H, "HKV": hk, "SQ": 24, "SK": 24, "D": D}


def _probe(op: str) -> dict:
    """A shape that ACTIVATES the op's feature yet stays CPU-cheap; non-pow2 D=48."""
    s = AT._SPECS[op]
    fam = s.family
    v = s.variant
    D = 48

    def hk(H):
        return AT._kv_heads(v, H)

    if fam == "varlen":
        return {"B": 3, "H": 4, "HKV": hk(4), "S": 33, "D": D}
    if fam == "cache":
        return {"B": 2, "H": 4, "HKV": hk(4), "SKctx": 40, "SQ": 13, "D": D}
    if fam == "decode":
        return {"B": 2, "H": 8, "HKV": hk(8), "SQ": 1, "SK": 52, "D": D}
    if fam == "cross":
        return {"B": 2, "H": 4, "HKV": hk(4), "SQ": (1 if s.step else 17), "SK": 29, "D": D}
    if fam == "chunked":
        return {"B": 2, "H": 4, "HKV": hk(4), "SQ": 15, "SK": 37, "D": D}
    if fam == "window":
        H = 2
        S = s.window + 24                       # activate window (S > W)
        return {"B": 1, "H": H, "HKV": hk(H), "SQ": S, "SK": S, "D": D}
    if fam == "dilated":
        return {"B": 2, "H": 4, "HKV": hk(4), "SQ": 33, "SK": 33, "D": D}
    return {"B": 2, "H": 4, "HKV": hk(4), "SQ": 31, "SK": 31, "D": D}


def _expected_arity(op: str) -> int:
    fam = AT._SPECS[op].family
    if fam == "cache":
        return 5
    if fam in ("varlen", "relbias", "custommask"):
        return 4
    return 3


# --------------------------------------------------------------------------- #
# independent fp64 oracles (distinct code paths from ref_fn)
# --------------------------------------------------------------------------- #
def _f64(t):
    return t.float().double()                             # fp8/bf16/fp16/fp32 -> fp64


def _allowed(SQ, SK, causal, window, dilation):
    i = torch.arange(SQ)[:, None]
    j = torch.arange(SK)[None, :]
    qpos = (SK - SQ) + i
    a = torch.ones(SQ, SK, dtype=torch.bool)
    if causal:
        a = a & (j <= qpos)
    if window > 0:
        a = a & (qpos - j < window)
    if dilation > 1:
        a = a & (j <= qpos) & (((qpos - j) % dilation) == 0)
    return a


def _naive_fwd(q, k, v, *, causal, scale, window=0, dilation=1, bias=None, add_mask=None):
    """Hand-written O(S^2) fp64 attention, independent of ref_fn's formulation."""
    q, k, v = _f64(q), _f64(k), _f64(v)
    H, HKV = q.shape[1], k.shape[1]
    g = H // HKV
    if g > 1:
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
    SQ, SK = q.shape[2], k.shape[2]
    s = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if bias is not None:
        s = s + bias.double()
    if add_mask is not None:
        s = s + add_mask.double()
    if causal or window > 0 or dilation > 1:
        s = s.masked_fill(~_allowed(SQ, SK, causal, window, dilation)[None, None], float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", p, v)


def _t5_bias_loop(bt, SQ, SK, num_buckets=AT._T5_NUM_BUCKETS, max_distance=AT._T5_MAX_DIST):
    """T5 bucket via an O(S^2) python double-loop (independent of the vectorized path)."""
    bt = _f64(bt)
    H = bt.shape[0]
    bias = torch.zeros(H, SQ, SK, dtype=torch.float64)
    max_exact = num_buckets // 2
    for i in range(SQ):
        qpos = (SK - SQ) + i
        for j in range(SK):
            n = max(0, qpos - j)
            if n < max_exact:
                bucket = n
            else:
                val = max_exact + int(math.log(max(1, n) / max_exact)
                                      / math.log(max_distance / max_exact) * (num_buckets - max_exact))
                bucket = min(val, num_buckets - 1)
            for h in range(H):
                bias[h, i, j] = bt[h, bucket]
    return bias


def _independent(op, xs):
    s = AT._SPECS[op]
    fam = s.family
    scale = 1.0 / math.sqrt(xs[0].shape[-1])
    if fam == "varlen":
        q, k, v, cu = xs
        cul = cu.tolist()
        out = torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float64)
        for b in range(len(cul) - 1):
            a, e = cul[b], cul[b + 1]
            qs = q[a:e].transpose(0, 1).unsqueeze(0)
            ks = k[a:e].transpose(0, 1).unsqueeze(0)
            vs = v[a:e].transpose(0, 1).unsqueeze(0)
            o = _naive_fwd(qs, ks, vs, causal=s.causal, scale=scale)
            out[a:e] = o[0].transpose(0, 1)
        return out
    if fam == "cache":
        q, kc, vc, kn, vn = xs
        k = torch.cat([kc, kn], dim=2)
        v = torch.cat([vc, vn], dim=2)
        return _naive_fwd(q, k, v, causal=True, scale=scale)
    if fam == "relbias":
        q, k, v, bt = xs
        bias = _t5_bias_loop(bt, q.shape[2], k.shape[2])
        return _naive_fwd(q, k, v, causal=True, scale=scale, bias=bias)
    if fam == "custommask":
        q, k, v, mask = xs
        return _naive_fwd(q, k, v, causal=False, scale=scale, add_mask=mask)
    # dense: chunked / cross / window / dilated / decode / fp8
    q, k, v = xs[0], xs[1], xs[2]
    return _naive_fwd(q, k, v, causal=s.causal, scale=scale,
                      window=s.window, dilation=s.dilation)


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
    assert all(op.startswith("attn2_") for op in AT.OPS)
    assert callable(AT.make_reference) and callable(AT.seed_source)
    assert callable(AT.op_names) and AT.op_names() == AT.OPS
    assert set(AT.OP_DTYPES) == set(AT.OPS)
    assert set(AT.SHAPES) == set(AT.OPS)
    assert AT.DEFAULT_DTYPES == ["bf16", "fp16"]


def test_reserved_families_absent():
    """MLA / paged / latent are RESERVED held-out eval - must not appear anywhere."""
    for op in AT.OPS:
        low = op.lower()
        for bad in ("mla", "paged", "latent"):
            assert bad not in low, (op, bad)
        ns = AT.make_reference(op, AT.OP_DTYPES[op][0])
        fl = ns["family"].lower()
        assert not any(bad in fl for bad in ("mla", "paged", "latent")), ns["family"]


def test_family_coverage():
    fams = Counter(AT._SPECS[o].family for o in AT.OPS)
    assert set(fams) == {"varlen", "chunked", "cross", "cache", "window",
                         "dilated", "decode", "relbias", "custommask", "fp8"}
    assert fams["varlen"] == 4 and fams["chunked"] == 3 and fams["cross"] == 4
    assert fams["cache"] == 3 and fams["window"] == 4 and fams["dilated"] == 2
    assert fams["decode"] == 6 and fams["relbias"] == 2 and fams["custommask"] == 1
    assert fams["fp8"] == 3
    # head_dim sweep {64,128,256} present in decode
    hds = {AT._SPECS[o].head_dim for o in AT.OPS if AT._SPECS[o].family == "decode"}
    assert hds == {64, 128, 256}
    # both causal + non-causal, GQA and MQA present
    assert any(AT._SPECS[o].causal for o in AT.OPS)
    assert any(not AT._SPECS[o].causal for o in AT.OPS)
    assert any(AT._SPECS[o].variant == "gqa" for o in AT.OPS)
    assert any(AT._SPECS[o].variant == "mqa" for o in AT.OPS)


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
# fp32 oracle correctness vs an INDEPENDENT torch compute (all 32 ops)
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
    """The torch eager baseline agrees with the fp32 oracle (fp8 -> bf16)."""
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
    """bf16/fp16 pass through; fp8 attention emits bf16."""
    for dt in AT.OP_DTYPES[op]:
        ns = AT.make_reference(op, dt)
        xs = ns["get_inputs"](_tiny(op), device="cpu", seed=2)
        outs = _as_tuple(ns["ref_fn"](*xs))
        exp = torch.bfloat16 if dt == "fp8" else getattr(torch, DTYPES[dt][0])
        assert all(o.dtype == exp for o in outs), (op, dt, [o.dtype for o in outs])


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
        assert "_attn2_fwd" in fns, f"{op}: seed must define the _attn2_fwd triton kernel"
        assert "@triton.jit" in src and "online" in src.lower()


@pytest.mark.parametrize("op", AT.OPS)
def test_shapes_parse_roundtrip(op):
    ns = AT.make_reference(op, AT.op_dtypes(op)[0])
    parse = ns["parse_shape"]
    sh = AT.SHAPES[op]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        assert all(isinstance(x, int) for x in spec.values())
        if "H" in spec and "HKV" in spec:
            assert spec["H"] % spec["HKV"] == 0            # valid GQA grouping
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert parse(s) == spec, (op, parse(s), spec)


# --------------------------------------------------------------------------- #
# hard serving semantics
# --------------------------------------------------------------------------- #
def test_varlen_is_per_sequence():
    """Varlen attends WITHIN each cu_seqlens slice (per-sequence) and NEVER across."""
    op = "attn2_varlen_mha_causal"
    ns = AT.make_reference(op, "fp32")
    xs = ns["get_inputs"](_tiny(op), device="cpu", seed=0)
    q, k, v, cu = xs
    ref = ns["ref_fn"](*xs)
    cul = cu.tolist()
    scale = 1.0 / math.sqrt(q.shape[-1])
    for b in range(len(cul) - 1):
        s, e = cul[b], cul[b + 1]
        L = e - s
        qs = q[s:e].transpose(0, 1).unsqueeze(0).double()
        ks = k[s:e].transpose(0, 1).unsqueeze(0).double()
        vs = v[s:e].transpose(0, 1).unsqueeze(0).double()
        m = torch.zeros(L, L, dtype=torch.float64).masked_fill(
            ~_allowed(L, L, True, 0, 1), float("-inf"))
        o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=m, scale=scale)
        assert _close(ref[s:e], o[0].transpose(0, 1))
    # cross-sequence isolation: perturbing sequence 0's values leaves sequence 1 intact
    v2 = v.clone()
    v2[:cul[1]] += 5.0
    ref2 = ns["ref_fn"](q, k, v2, cu)
    assert not _close(ref[:cul[1]], ref2[:cul[1]])         # seq 0 changed
    assert _close(ref[cul[1]:], ref2[cul[1]:])             # seq 1 unaffected (no leak)


def test_chunked_causal_alignment():
    """A query chunk attends the growing KV up to its absolute position (SK-SQ)+i."""
    op = "attn2_chunked_mha_causal"
    ns = AT.make_reference(op, "fp32")
    q, k, v = ns["get_inputs"]({"B": 1, "H": 2, "HKV": 2, "SQ": 8, "SK": 20, "D": 32},
                               device="cpu", seed=0)
    ref = ns["ref_fn"](q, k, v)
    assert _close(ref, _independent(op, (q, k, v)))
    # query 0 sits at absolute pos 12 -> keys >= 13 are future; only later queries see them
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 13:] += 5.0
    v2[:, :, 13:] += 5.0
    y2 = ns["ref_fn"](q, k2, v2)
    assert _close(ref[:, :, 0], y2[:, :, 0])               # first query unaffected
    assert not _close(ref[:, :, -1], y2[:, :, -1])         # last query (pos 19) affected


def test_cross_attends_all_encoder():
    """Cross-attention (non-causal) lets a decoder query see EVERY encoder position."""
    op = "attn2_cross_mha_prefill"
    ns = AT.make_reference(op, "fp32")
    q, k, v = ns["get_inputs"]({"B": 1, "H": 2, "HKV": 2, "SQ": 6, "SK": 10, "D": 32},
                               device="cpu", seed=0)
    ref = ns["ref_fn"](q, k, v)
    sdpa = F.scaled_dot_product_attention(q.double(), k.double(), v.double(),
                                          scale=1.0 / math.sqrt(q.shape[-1]))
    assert _close(ref, sdpa)                               # full (no causal mask)
    v2 = v.clone()
    v2[:, :, -1] += 5.0
    assert not _close(ref[:, :, 0], ns["ref_fn"](q, k, v2)[:, :, 0])   # sees last enc pos


def test_cacheappend_equals_concat_full():
    """Cache-append == full attention over the concatenated [cache, new] KV."""
    op = "attn2_cacheappend_gqa"
    ns = AT.make_reference(op, "fp32")
    xs = ns["get_inputs"]({"B": 1, "H": 8, "HKV": 2, "SKctx": 20, "SQ": 6, "D": 32},
                          device="cpu", seed=0)
    q, kc, vc, kn, vn = xs
    ref = ns["ref_fn"](*xs)
    k = torch.cat([kc, kn], dim=2)
    v = torch.cat([vc, vn], dim=2)
    g = q.shape[1] // k.shape[1]
    ke, ve = k.repeat_interleave(g, 1).double(), v.repeat_interleave(g, 1).double()
    SQ, SK = q.shape[2], k.shape[2]
    m = torch.zeros(SQ, SK, dtype=torch.float64).masked_fill(
        ~_allowed(SQ, SK, True, 0, 1), float("-inf"))
    full = F.scaled_dot_product_attention(q.double(), ke, ve, attn_mask=m,
                                          scale=1.0 / math.sqrt(q.shape[-1]))
    assert _close(ref, full)


def test_sliding_window_is_local():
    """Windowed attention ignores keys older than `window` (locality)."""
    op = "attn2_window256_mha_causal"
    ns = AT.make_reference(op, "fp32")
    W = AT._SPECS[op].window
    S = W + 40
    q, k, v = ns["get_inputs"]({"B": 1, "H": 2, "HKV": 2, "SQ": S, "SK": S, "D": 32},
                               device="cpu", seed=0)
    assert _close(ns["ref_fn"](q, k, v), _independent(op, (q, k, v)))
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 0] += 9.0
    v2[:, :, 0] += 9.0
    y = ns["ref_fn"](q, k, v)
    y2 = ns["ref_fn"](q, k2, v2)
    assert _close(y[:, :, -1], y2[:, :, -1])               # last query: key 0 out of window
    assert not _close(y[:, :, 0], y2[:, :, 0])             # first query DID see key 0


def test_dilated_is_strided():
    """Strided/dilated attention keeps only every d-th past key."""
    op = "attn2_dilated_mha_causal"
    ns = AT.make_reference(op, "fp32")
    d = AT._SPECS[op].dilation
    S = 24
    q, k, v = ns["get_inputs"]({"B": 1, "H": 2, "HKV": 2, "SQ": S, "SK": S, "D": 32},
                               device="cpu", seed=0)
    ref = ns["ref_fn"](q, k, v)
    assert _close(ref, _independent(op, (q, k, v)))
    qpos = S - 1
    for idx, expect_change in ((qpos - 1, False), (qpos - d, True)):   # off- vs on-stride
        k2, v2 = k.clone(), v.clone()
        k2[:, :, idx] += 9.0
        v2[:, :, idx] += 9.0
        changed = not _close(ref[:, :, -1], ns["ref_fn"](q, k2, v2)[:, :, -1])
        assert changed == expect_change, (idx, expect_change)


def test_gqa_equals_expanded_mha():
    """GQA attention == dense MHA on the group-expanded K/V (the defining identity)."""
    ns = AT.make_reference("attn2_chunked_gqa_causal", "fp32")
    H, HKV, D = 8, 2, 32
    q, k, v = ns["get_inputs"]({"B": 2, "H": H, "HKV": HKV, "SQ": 10, "SK": 18, "D": D},
                               device="cpu", seed=0)
    g = H // HKV
    ke, ve = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    mns = AT.make_reference("attn2_chunked_mha_causal", "fp32")
    assert _close(ns["ref_fn"](q, k, v), mns["ref_fn"](q, ke, ve))


def test_decode_is_full_attention_over_kv():
    """Decode (q_len == 1) attends the WHOLE KV cache (no causal masking)."""
    dec = AT.make_reference("attn2_decode_gqa_hd128", "fp32")
    q, k, v = dec["get_inputs"]({"B": 2, "H": 8, "HKV": 2, "SQ": 1, "SK": 48, "D": 64},
                                device="cpu", seed=0)
    g = 4
    ke, ve = k.repeat_interleave(g, 1).double(), v.repeat_interleave(g, 1).double()
    full = F.scaled_dot_product_attention(q.double(), ke, ve, is_causal=False,
                                          scale=1.0 / math.sqrt(q.shape[-1]))
    assert _close(dec["ref_fn"](q, k, v), full)


def test_relbias_matches_and_bites():
    """T5 relative bias matches the python-loop bucket oracle AND changes attention."""
    op = "attn2_relbias_t5_mha_causal"
    ns = AT.make_reference(op, "fp32")
    xs = ns["get_inputs"]({"B": 1, "H": 4, "HKV": 4, "SQ": 20, "SK": 20, "D": 32},
                          device="cpu", seed=0)
    ref = ns["ref_fn"](*xs)
    assert _close(ref, _independent(op, xs))
    q, k, v, bt = xs
    zero = ns["ref_fn"](q, k, v, torch.zeros_like(bt))
    plain = _naive_fwd(q, k, v, causal=True, scale=1.0 / math.sqrt(q.shape[-1]))
    assert _close(zero, plain)                             # zero table -> plain causal
    assert not _close(ref, plain)                          # learned bias bites


def test_custommask_matches_and_bites():
    """A custom additive mask matches the einsum oracle AND changes the attention."""
    op = "attn2_custommask_mha"
    ns = AT.make_reference(op, "fp32")
    xs = ns["get_inputs"]({"B": 1, "H": 4, "HKV": 4, "SQ": 16, "SK": 16, "D": 32},
                          device="cpu", seed=0)
    ref = ns["ref_fn"](*xs)
    assert _close(ref, _independent(op, xs))
    q, k, v, mask = xs
    zero = ns["ref_fn"](q, k, v, torch.zeros_like(mask))
    plain = _naive_fwd(q, k, v, causal=False, scale=1.0 / math.sqrt(q.shape[-1]))
    assert _close(zero, plain)                             # zero mask -> plain full attn
    assert not _close(ref, plain)                          # the mask bites


def test_fp8_output_is_bf16():
    """fp8 (e4m3) attention accumulates in fp32 and emits bf16."""
    for op in [o for o in AT.OPS if AT._SPECS[o].fp8]:
        ns = AT.make_reference(op, "fp8")
        xs = ns["get_inputs"](_tiny(op), device="cpu", seed=0)
        out = ns["ref_fn"](*xs)
        assert out.dtype == torch.bfloat16, op
        assert _close(out, _independent(op, xs), atol=3e-2, rtol=3e-2)
