"""CPU-only tests for the breadth SEQUENCE-MODEL + CONV1D authoring engine.

Every ``ref_fn`` is checked against an INDEPENDENT torch computation on a DIFFERENT
code path than the eager recurrence it wraps (cumulative scans vs a manual prefix
loop; the gated scan vs its O(L^2) closed form; the Mamba-1 SSM vs an einsum
discretization; linear attention vs the QUADRATIC attention-matrix dual; the causal
conv vs an im2col-free shifted-accumulate), so a wrong oracle is caught with
certainty. Also asserts arity, that each seed compiles + defines its entry, the
shape catalog round-trips through ``parse_shape``, the namespace ABI matches
vendor_ops.py, and causality/segment-reset semantics hold. All fp32/fp64 on CPU
(no GPU / triton execution needed - the seed is only static-checked)."""

from __future__ import annotations

import ast

import pytest
import torch
import torch.nn.functional as F

from kore.tasks._genops import DTYPES
from kore.tasks.breadth import seq as SQ

DTYPE_NAMES = ("bf16", "fp16", "fp32")

# --------------------------------------------------------------------------- #
# tiny CPU shapes + expected arity per op
# --------------------------------------------------------------------------- #
_SMALL = {
    "cumsum": {"B": 1, "D": 3, "L": 10},
    "cumprod": {"B": 2, "D": 3, "L": 9},
    "assoc_scan_segmented": {"B": 2, "D": 3, "L": 10},
    "selective_scan": {"B": 2, "L": 8, "D": 4, "N": 8},
    "ssd_chunk_scan": {"B": 2, "L": 8, "D": 4, "N": 8},
    "linear_attention": {"B": 2, "H": 2, "L": 6, "Dh": 4},
    "causal_conv1d": {"B": 2, "D": 3, "L": 10, "K": 4},
}
_ARITY = {
    "cumsum": 1, "cumprod": 1, "assoc_scan_segmented": 2,
    "selective_scan": 6, "ssd_chunk_scan": 4, "linear_attention": 3,
    "causal_conv1d": 3,
}


# --------------------------------------------------------------------------- #
# independent fp64 oracles (distinct code paths from the eager recurrence)
# --------------------------------------------------------------------------- #
def _ind_cumsum(x):
    x = x.double()
    out = torch.zeros_like(x)
    acc = torch.zeros(x.shape[:-1], dtype=x.dtype)
    for t in range(x.shape[-1]):
        acc = acc + x[..., t]
        out[..., t] = acc
    return out


def _ind_cumprod(x):
    x = x.double()
    out = torch.zeros_like(x)
    acc = torch.ones(x.shape[:-1], dtype=x.dtype)
    for t in range(x.shape[-1]):
        acc = acc * x[..., t]
        out[..., t] = acc
    return out


def _ind_assoc(a, b):
    # Direct (O(L^2)) closed form of h_t = a_t*h_{t-1}+b_t, h_{-1}=0:
    #   h_t = sum_{s<=t} (prod_{r=s+1..t} a_r) * b_s
    a, b = a.double(), b.double()
    B, D, L = a.shape
    out = torch.zeros_like(a)
    for t in range(L):
        acc = torch.zeros(B, D, dtype=a.dtype)
        for s in range(t + 1):
            coef = torch.ones(B, D, dtype=a.dtype)
            for r in range(s + 1, t + 1):
                coef = coef * a[:, :, r]
            acc = acc + coef * b[:, :, s]
        out[:, :, t] = acc
    return out


def _ind_selective_scan(u, delta, A, B_, C, D_):
    # Mamba-1 selective_scan_ref via an einsum discretization (delta_softplus=True).
    u, delta, A, B_, C, D_ = (t.double() for t in (u, delta, A, B_, C, D_))
    Bs, L, D = u.shape
    dt = F.softplus(delta)
    dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))          # [B,L,D,N]
    dBu = torch.einsum("bld,bln,bld->bldn", dt, B_, u)           # [B,L,D,N]
    h = torch.zeros(Bs, D, A.shape[1], dtype=u.dtype)
    ys = []
    for t in range(L):
        h = dA[:, t] * h + dBu[:, t]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, 1) + torch.einsum("bld,d->bld", u, D_)


def _ind_ssd(x, a, B_, C):
    x, a, B_, C = (t.double() for t in (x, a, B_, C))
    Bs, L, D = x.shape
    h = torch.zeros(Bs, D, B_.shape[-1], dtype=x.dtype)
    ys = []
    for t in range(L):
        h = a[:, t, None, None] * h + torch.einsum("bd,bn->bdn", x[:, t], B_[:, t])
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, 1)


def _ind_linear_attention(q, k, v):
    # Quadratic (attention-matrix) dual of the recurrent linear-attention state:
    #   y_t = sum_{s<=t} (phi(q_t) . phi(k_s)) v_s ;  phi = elu + 1
    q, k, v = (t.double() for t in (q, k, v))
    pq = F.elu(q) + 1.0
    pk = F.elu(k) + 1.0
    scores = torch.einsum("bhtd,bhsd->bhts", pq, pk)             # [B,H,L,L]
    L = q.shape[2]
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
    scores = torch.where(causal, scores, torch.zeros_like(scores))
    return torch.einsum("bhts,bhse->bhte", scores, v)


def _ind_causal_conv1d(x, weight, bias):
    # im2col-free shifted accumulate: y[t] = bias + sum_k w[k]*x[t-(K-1)+k], x=0 for t<0.
    x, weight, bias = x.double(), weight.double(), bias.double()
    B, D, L = x.shape
    K = weight.shape[1]
    out = torch.zeros(B, D, L, dtype=x.dtype)
    for t in range(L):
        acc = bias.view(1, D).expand(B, D).clone()
        for kk in range(K):
            idx = t - (K - 1) + kk
            if idx >= 0:
                acc = acc + weight[:, kk].view(1, D) * x[:, :, idx]
        out[:, :, t] = acc
    return out


def _independent(op, inputs):
    if op == "cumsum":
        return _ind_cumsum(*inputs)
    if op == "cumprod":
        return _ind_cumprod(*inputs)
    if op == "assoc_scan_segmented":
        return _ind_assoc(*inputs)
    if op == "selective_scan":
        return _ind_selective_scan(*inputs)
    if op == "ssd_chunk_scan":
        return _ind_ssd(*inputs)
    if op == "linear_attention":
        return _ind_linear_attention(*inputs)
    if op == "causal_conv1d":
        return _ind_causal_conv1d(*inputs)
    raise AssertionError(f"no independent oracle for {op!r}")


def _close(a, b, atol=1e-4, rtol=1e-3):
    return torch.allclose(a.double(), b.double(), atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- #
# metadata / ABI surface
# --------------------------------------------------------------------------- #
def test_abi_present():
    assert isinstance(SQ.OPS, list) and len(SQ.OPS) == 7
    assert callable(SQ.make_reference) and callable(SQ.seed_source)
    assert set(SQ.OP_DTYPES) == set(SQ.OPS)
    assert set(SQ.SHAPES) == set(SQ.OPS)
    assert set(_SMALL) == set(SQ.OPS) == set(_ARITY)


def test_ops_dtypes_shapes_consistent():
    for op in SQ.OPS:
        assert SQ.OP_DTYPES[op], f"empty dtype sweep for {op}"
        assert SQ.op_dtypes(op) == SQ.OP_DTYPES[op]
        for d in SQ.OP_DTYPES[op]:
            assert d in DTYPE_NAMES, f"unknown dtype {d} for {op}"
        sh = SQ.SHAPES[op]
        assert "minimal" in sh and "primary" in sh and "validation" in sh
        assert isinstance(sh["validation"], list) and sh["validation"]
    assert SQ.op_dtypes("cumsum") == SQ.DEFAULT_DTYPES


@pytest.mark.parametrize("op", SQ.OPS)
def test_namespace_contract(op):
    ns = SQ.make_reference(op, "bf16")
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family", "mutates_input"):
        assert k in ns, k
    assert ns["entry_name"] == op
    assert ns["dtype_name"] == "bf16"
    assert ns["family"] == f"breadth_{op}"
    assert ns["mutates_input"] is False
    assert callable(ns["ref_fn"]) and callable(ns["baseline_fn"])
    assert ns[f"{op}_ref"] is ns["ref_fn"]


@pytest.mark.parametrize("op", SQ.OPS)
def test_arity(op):
    ns = SQ.make_reference(op, "fp32")
    assert ns["arity"] == _ARITY[op]
    inputs = ns["get_inputs"](_SMALL[op], device="cpu", seed=0)
    assert isinstance(inputs, tuple)
    assert len(inputs) == ns["arity"]


# --------------------------------------------------------------------------- #
# fp32 oracle correctness vs an INDEPENDENT torch compute
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", SQ.OPS)
def test_ref_matches_independent(op):
    ns = SQ.make_reference(op, "fp32")
    inputs = ns["get_inputs"](_SMALL[op], device="cpu", seed=0)
    ref = ns["ref_fn"](*inputs)
    ind = _independent(op, inputs)
    assert ref.shape == ind.shape, f"{op}: {tuple(ref.shape)} vs {tuple(ind.shape)}"
    assert _close(ref, ind), (
        f"{op}: max|diff|={(ref.double() - ind.double()).abs().max().item():.3e}")


@pytest.mark.parametrize("op", SQ.OPS)
def test_baseline_matches_ref(op):
    """The torch eager baseline (fp32) agrees with the fp32 oracle (same math)."""
    ns = SQ.make_reference(op, "fp32")
    inputs = ns["get_inputs"](_SMALL[op], device="cpu", seed=1)
    out = ns["baseline_fn"](*inputs)
    ref = ns["ref_fn"](*inputs)
    assert out.shape == ref.shape
    assert _close(out, ref)


@pytest.mark.parametrize("op", SQ.OPS)
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_ref_preserves_input_dtype(op, dtype):
    ns = SQ.make_reference(op, dtype)
    inputs = ns["get_inputs"](_SMALL[op], device="cpu", seed=2)
    out = ns["ref_fn"](*inputs)
    tdt = getattr(torch, DTYPES[dtype][0])
    outs = out if isinstance(out, (tuple, list)) else (out,)
    assert all(o.dtype == tdt for o in outs)


# --------------------------------------------------------------------------- #
# seed static checks (compiles + defines a top-level entry fn)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", SQ.OPS)
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_seed_compiles_and_defines_entry(op, dtype):
    src = SQ.seed_source(op, dtype)
    compile(src, f"<{op}_{dtype}_seed>", "exec")          # valid Python (COMPILING seed)
    tree = ast.parse(src)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert op in funcs, f"{op} seed must define def {op}(...)"
    assert any(isinstance(n, ast.FunctionDef) and n.name == op for n in tree.body), (
        f"{op} entry must be a top-level def")


@pytest.mark.parametrize("op", SQ.OPS)
def test_shapes_parse_roundtrip(op):
    ns = SQ.make_reference(op, "fp32")
    parse = ns["parse_shape"]
    sh = SQ.SHAPES[op]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert parse(s) == spec, (op, parse(s), spec)


# --------------------------------------------------------------------------- #
# op-specific numeric / semantic checks
# --------------------------------------------------------------------------- #
def test_cumsum_last_equals_row_sum():
    ns = SQ.make_reference("cumsum", "fp32")
    (x,) = ns["get_inputs"]({"B": 2, "D": 3, "L": 17}, device="cpu", seed=0)
    out = ns["ref_fn"](x)
    assert _close(out[..., -1], x.float().sum(-1))
    assert _close(out, torch.cumsum(x.float(), dim=-1))


def test_selective_scan_matches_mamba_ref_larger():
    """Bigger shape cross-check of the Mamba-1 core against the einsum discretization."""
    ns = SQ.make_reference("selective_scan", "fp32")
    inputs = ns["get_inputs"]({"B": 2, "L": 24, "D": 6, "N": 16}, device="cpu", seed=3)
    ref = ns["ref_fn"](*inputs)
    ind = _ind_selective_scan(*inputs)
    assert ref.shape == (2, 24, 6)
    assert _close(ref, ind)


def test_assoc_scan_segment_reset():
    """A zero gate resets the state (segment boundary): h_p == b_p, and values after
    the reset do NOT depend on inputs before it."""
    ns = SQ.make_reference("assoc_scan_segmented", "fp32")
    B, D, L, p = 1, 2, 8, 4
    g = torch.Generator().manual_seed(0)
    a = torch.sigmoid(torch.randn(B, D, L, generator=g))
    b = torch.randn(B, D, L, generator=g)
    a[:, :, p] = 0.0                                  # reset at t=p
    out = ns["ref_fn"](a, b)
    assert _close(out[:, :, p], b[:, :, p])           # h_p == b_p
    # Perturb only b BEFORE the reset -> outputs at t>=p are unchanged.
    b2 = b.clone()
    b2[:, :, :p] += 5.0
    out2 = ns["ref_fn"](a, b2)
    assert _close(out[:, :, p:], out2[:, :, p:])


def test_linear_attention_is_causal():
    """y_t must not depend on v_{s>t} (strict causality)."""
    ns = SQ.make_reference("linear_attention", "fp32")
    q, k, v = ns["get_inputs"]({"B": 1, "H": 2, "L": 7, "Dh": 4}, device="cpu", seed=0)
    t0 = 3
    v2 = v.clone()
    v2[:, :, t0 + 1:] += 3.0                          # perturb the future
    y = ns["ref_fn"](q, k, v)
    y2 = ns["ref_fn"](q, k, v2)
    assert _close(y[:, :, : t0 + 1], y2[:, :, : t0 + 1])
    assert not _close(y[:, :, t0 + 1:], y2[:, :, t0 + 1:])   # future DID change


def test_causal_conv1d_is_causal():
    """y_t must not depend on x_{s>t} (left-causal conv)."""
    ns = SQ.make_reference("causal_conv1d", "fp32")
    x, w, b = ns["get_inputs"]({"B": 1, "D": 3, "L": 12, "K": 4}, device="cpu", seed=0)
    t0 = 5
    x2 = x.clone()
    x2[:, :, t0 + 1:] += 4.0                          # perturb the future
    y = ns["ref_fn"](x, w, b)
    y2 = ns["ref_fn"](x2, w, b)
    assert _close(y[:, :, : t0 + 1], y2[:, :, : t0 + 1])


def test_causal_conv1d_matches_conv1d():
    """Matches F.conv1d with an explicit left pad (groups=D depthwise)."""
    ns = SQ.make_reference("causal_conv1d", "fp32")
    x, w, b = ns["get_inputs"]({"B": 2, "D": 5, "L": 16, "K": 4}, device="cpu", seed=1)
    out = ns["ref_fn"](x, w, b)
    K, D = w.shape[1], x.shape[1]
    exp = F.conv1d(F.pad(x.float(), (K - 1, 0)), w.float()[:, None, :], b.float(), groups=D)
    assert out.shape == x.shape
    assert _close(out, exp)
