"""CPU-only tests for the breadth SAMPLING / LOGIT-PROCESSOR / SPECULATIVE-DECODE /
ROPE task engine.

Every ``ref_fn`` is cross-checked against an INDEPENDENT torch computation on a
DIFFERENT code path than the manual formula it wraps (torch.softmax /
torch.searchsorted inverse-CDF / the HF shift-by-one nucleus rule / a complex-number
RoPE rotation via torch.polar / a boolean-matrix-power ancestor closure / explicit
per-row loops), at a tight fp32 tolerance AND on an EXTREME-magnitude input (huge
logits) that a naive non-max-subtracted softmax would fail. Also asserts the ABI
surface (28 ``smp_`` ops, bf16/fp16/fp32 sweep), the namespace contract, that every
seed parses + compiles + defines its entry, the shape catalog round-trips through
``parse_shape`` and is realistic (batch {1,64,256}, vocab {32000,128256}, non-pow2
tail), arity matches ``get_inputs``, ref preserves the input dtype (int64 for the
index samplers), determinism, and op-specific invariants (distributions sum to 1 and
are sparser than the full softmax, top-k keeps exactly k, the rejection accept rule
and a valid renormalized residual, categorical == inverse-CDF, Gumbel-max == argmax,
tree mask == ancestor reachability, RoPE is norm-preserving and pos-0 is identity).
All fp32/fp64 on CPU (no GPU / triton runtime)."""

from __future__ import annotations

import ast
import math

import pytest
import torch
import torch.nn.functional as F

from kore.tasks._genops import DTYPES
from kore.tasks.breadth import sample_ext as S

DTYPE_NAMES = ("bf16", "fp16", "fp32")

# expected arity per op (independent of make_reference, so a wrong arity is caught)
_ARITY = {
    "smp_temperature": 1, "smp_topk_mask_k20": 1, "smp_topk_mask_k50": 1,
    "smp_topp_renorm": 1, "smp_minp_mask": 1, "smp_typical_mask": 1,
    "smp_repetition_penalty": 2, "smp_presence_penalty": 2, "smp_frequency_penalty": 2,
    "smp_logit_bias": 2, "smp_no_repeat_ngram": 2, "smp_topk_topp": 1,
    "smp_categorical_sample": 2, "smp_gumbel_max": 2, "smp_topp_sample": 2,
    "smp_topk_sample": 2,
    "smp_spec_accept": 4, "smp_spec_residual": 2, "smp_spec_bonus_token": 2,
    "smp_tree_attn_mask": 1, "smp_verify_prefix": 2,
    "smp_rope_linear_pi": 2, "smp_rope_ntk": 2, "smp_rope_dynamic_ntk": 2,
    "smp_rope_yarn": 2, "smp_rope_partial": 2, "smp_rope_2d": 3, "smp_rope_llama3": 2,
}

_INT_OPS = frozenset({
    "smp_categorical_sample", "smp_gumbel_max", "smp_topp_sample", "smp_topk_sample",
    "smp_spec_bonus_token", "smp_verify_prefix",
})
_MASKED_OPS = frozenset({"smp_topk_mask_k20", "smp_topk_mask_k50"})
_BOOL_OPS = frozenset({"smp_spec_accept"})
_DIST_OPS = frozenset({
    "smp_temperature", "smp_topp_renorm", "smp_minp_mask", "smp_typical_mask",
    "smp_topk_topp", "smp_spec_residual",
})
_ROPE_OPS = frozenset({
    "smp_rope_linear_pi", "smp_rope_ntk", "smp_rope_dynamic_ntk", "smp_rope_yarn",
    "smp_rope_partial", "smp_rope_2d", "smp_rope_llama3",
})
_FORBIDDEN = ("mla", "paged", "latent")


# --------------------------------------------------------------------------- #
# metadata / ABI surface
# --------------------------------------------------------------------------- #
def test_ops_and_metadata():
    assert isinstance(S.OPS, tuple) and len(S.OPS) == 28
    assert len(set(S.OPS)) == 28
    assert all(op.startswith("smp_") for op in S.OPS)
    assert all(sub not in op for op in S.OPS for sub in _FORBIDDEN)
    assert set(S.OPS) == set(_ARITY)
    assert S.DEFAULT_DTYPES == ("bf16", "fp16", "fp32")
    for op in S.OPS:
        assert op in S.OP_DTYPES and op in S.SHAPES
        assert S.op_dtypes(op) == S.OP_DTYPES[op]
        assert all(dt in DTYPE_NAMES for dt in S.op_dtypes(op))
    assert S.op_names() == list(S.OPS)


@pytest.mark.parametrize("op", S.OPS)
def test_namespace_contract(op):
    ns = S.make_reference(op, "fp32")
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family", "mutates_input"):
        assert k in ns, k
    assert ns["entry_name"] == op
    assert ns["dtype_name"] == "fp32"
    assert ns["family"] == f"breadth_{op}"
    assert ns["mutates_input"] is False
    assert ns["arity"] == _ARITY[op]
    assert callable(ns["ref_fn"]) and callable(ns["baseline_fn"])
    assert ns[f"{op}_ref"] is ns["ref_fn"]


@pytest.mark.parametrize("op", S.OPS)
def test_seed_parses_compiles_and_defines_entry(op):
    for dtype in S.op_dtypes(op):
        src = S.seed_source(op, dtype)
        tree = ast.parse(src)                              # valid Python
        compile(src, f"<seed:{op}:{dtype}>", "exec")       # compiles to bytecode
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert op in funcs, (op, dtype, funcs)
        assert any(isinstance(n, ast.FunctionDef) and n.name == op
                   for n in tree.body), f"{op} entry must be a top-level def"


@pytest.mark.parametrize("op", S.OPS)
def test_shapes_parse_roundtrip(op):
    ns = S.make_reference(op, "fp32")
    parse = ns["parse_shape"]
    sh = S.SHAPES[op]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert parse(s) == spec, (op, parse(s), spec)


def test_shape_catalog_is_realistic():
    """Vocab ops sweep batch {1,64,256} x vocab {32000,128256} + a non-pow2 tail;
    RoPE ops sweep head_dim {64,128} + a non-pow2 96."""
    vocab_op = "smp_temperature"
    Ms, Vs, nonpow2 = set(), set(), False
    for spec in [S.SHAPES[vocab_op]["primary"], *S.SHAPES[vocab_op]["validation"]]:
        Ms.add(spec["M"])
        Vs.add(spec["V"])
        if spec["V"] & (spec["V"] - 1) != 0:
            nonpow2 = True
    assert {1, 64, 256} <= Ms, Ms
    assert {32000, 128256} <= Vs, Vs
    assert nonpow2

    Ds, rope_nonpow2 = set(), False
    for spec in [S.SHAPES["smp_rope_ntk"]["primary"], *S.SHAPES["smp_rope_ntk"]["validation"]]:
        Ds.add(spec["D"])
        if spec["D"] & (spec["D"] - 1) != 0:
            rope_nonpow2 = True
    assert {64, 128} <= Ds, Ds
    assert rope_nonpow2


@pytest.mark.parametrize("op", S.OPS)
def test_arity_matches_get_inputs(op):
    ns = S.make_reference(op, "fp32")
    inputs = ns["get_inputs"](S.SHAPES[op]["minimal"], device="cpu", seed=0)
    assert isinstance(inputs, tuple)
    assert len(inputs) == ns["arity"] == _ARITY[op]


@pytest.mark.parametrize("op", S.OPS)
@pytest.mark.parametrize("dtype", ("bf16", "fp16"))
def test_ref_preserves_input_dtype(op, dtype):
    ns = S.make_reference(op, dtype)
    inputs = ns["get_inputs"](S.SHAPES[op]["minimal"], device="cpu", seed=1)
    out = ns["ref_fn"](*inputs)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    if op in _INT_OPS:
        assert all(o.dtype == torch.int64 for o in outs)
    else:
        tdt = getattr(torch, DTYPES[dtype][0])
        assert all(o.dtype == tdt for o in outs)


def test_baseline_matches_reference_shapes():
    for op in S.OPS:
        ns = S.make_reference(op, "fp32")
        inputs = ns["get_inputs"](S.SHAPES[op]["minimal"], device="cpu", seed=3)
        r = ns["ref_fn"](*inputs)
        b = ns["baseline_fn"](*inputs)
        rs = r if isinstance(r, (tuple, list)) else (r,)
        bs = b if isinstance(b, (tuple, list)) else (b,)
        assert len(rs) == len(bs), (op, "tuple len")
        for ro, bo in zip(rs, bs):
            assert ro.shape == bo.shape, (op, tuple(ro.shape), tuple(bo.shape))


# --------------------------------------------------------------------------- #
# input builders (deterministic) + independent oracles
# --------------------------------------------------------------------------- #
def _logits(M, N, seed, scale=1.5, offset=0.0):
    g = torch.Generator().manual_seed(seed)
    return offset + scale * torch.randn(M, N, generator=g, dtype=torch.float32)


def _uni(M, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(M, generator=g, dtype=torch.float32)


def _probs(M, N, seed, scale=1.5):
    return torch.softmax(_logits(M, N, seed, scale), dim=-1)


def _counts(M, N, seed, hi=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, hi, (M, N), generator=g, dtype=torch.int64)


def _seen(M, N, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(M, N, generator=g, dtype=torch.float32) < 0.5).to(torch.float32)


def _idvec(M, hi, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, hi, (M,), generator=g, dtype=torch.int64)


def _gumbel(M, N, seed):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(M, N, generator=g, dtype=torch.float32).clamp_(1e-9, 1.0)
    return -torch.log(-torch.log(u))


def _pos(M, seed, hi=2048):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, hi, (M,), generator=g, dtype=torch.int64).to(torch.float32)


def _parents(M, T, seed):
    g = torch.Generator().manual_seed(seed)
    par = torch.full((M, T), -1, dtype=torch.int64)
    for j in range(1, T):
        par[:, j] = torch.randint(0, j, (M,), generator=g, dtype=torch.int64)
    return par


def _nucleus_hf(p, thresh):
    """The HF shift-by-one nucleus rule (an INDEPENDENT path vs the exclusive-prefix
    scatter used by the ref)."""
    sp, si = torch.sort(p, dim=-1, descending=True)
    remove = sp.cumsum(-1) > thresh
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    keep = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, si, ~remove)
    masked = torch.where(keep, p, torch.zeros_like(p))
    return masked / masked.sum(-1, keepdim=True)


def _rope_cplx(x, inv, pos, mscale=1.0):
    """Apply RoPE via COMPLEX multiplication (torch.polar) - a different code path
    than the ref's real rotate-half formula."""
    half = x.shape[-1] // 2
    ang = pos.view(-1, 1).double() * inv.view(1, -1).double()
    z = torch.complex(x[..., :half].double(), x[..., half:].double())
    rot = torch.polar(torch.ones_like(ang), ang)
    zr = z * rot
    return torch.cat([zr.real * mscale, zr.imag * mscale], dim=-1)


def _inv_full(op, D):
    half = D // 2
    i = torch.arange(half, dtype=torch.float64)
    theta = S.ROPE_THETA
    if op == "smp_rope_linear_pi":
        return (theta ** (-(2.0 * i) / D)) / S.ROPE_SCALE, 1.0
    if op == "smp_rope_ntk":
        th = theta * (S.ROPE_SCALE ** (D / (D - 2)))
        return th ** (-(2.0 * i) / D), 1.0
    if op == "smp_rope_dynamic_ntk":
        bf = (S.ROPE_SCALE * S.ROPE_DYN_SEQ_LEN / S.ROPE_ORIG_MAX) - (S.ROPE_SCALE - 1)
        th = theta * (bf ** (D / (D - 2)))
        return th ** (-(2.0 * i) / D), 1.0
    if op == "smp_rope_llama3":
        inv0 = theta ** (-(2.0 * i) / D)
        wl = 2 * math.pi / inv0
        low_wl = S.LLAMA3_OLD_CTX / S.LLAMA3_LOW_FREQ
        high_wl = S.LLAMA3_OLD_CTX / S.LLAMA3_HIGH_FREQ
        inv_low = inv0 / S.LLAMA3_FACTOR
        smooth = (S.LLAMA3_OLD_CTX / wl - S.LLAMA3_LOW_FREQ) / (S.LLAMA3_HIGH_FREQ - S.LLAMA3_LOW_FREQ)
        inv_sm = (1 - smooth) * inv0 / S.LLAMA3_FACTOR + smooth * inv0
        inv = torch.where(wl > low_wl, inv_low, torch.where(wl < high_wl, inv0, inv_sm))
        return inv, 1.0
    if op == "smp_rope_yarn":
        freq = theta ** ((2.0 * i) / D)
        inv_extra = 1.0 / freq
        inv_inter = 1.0 / (S.ROPE_SCALE * freq)

        def corr(nr):
            return (D * math.log(S.ROPE_ORIG_MAX / (nr * 2 * math.pi))) / (2 * math.log(theta))

        low = max(math.floor(corr(S.YARN_BETA_FAST)), 0)
        high = min(math.ceil(corr(S.YARN_BETA_SLOW)), half - 1)
        if high == low:
            high = low + 0.001
        ramp = torch.clamp((torch.arange(half, dtype=torch.float64) - low) / (high - low), 0, 1)
        inv_mask = 1.0 - ramp
        inv = inv_inter * (1 - inv_mask) + inv_extra * inv_mask
        mscale = 0.1 * math.log(S.ROPE_SCALE) + 1.0
        return inv, mscale
    raise AssertionError(op)


def _case(op, mode):
    """Return (inputs, expected, tol). ``expected`` comes from an INDEPENDENT torch
    path; ``mode`` in {'normal','extreme'}."""
    extreme = mode == "extreme"
    scale = 1e3 if extreme else 1.5
    tol = 1e-2 if extreme else 3e-3
    M, N = 6, 64

    if op == "smp_temperature":
        x = _logits(M, N, 0, scale)
        return (x,), torch.softmax(x / S.TEMP, dim=-1), tol
    if op in S.TOPK_MASK_SIZES:
        k = S.TOPK_MASK_SIZES[op]
        x = _logits(M, N, 0, scale)
        thr = torch.sort(x, dim=-1, descending=True).values[:, k - 1:k]
        exp = torch.where(x >= thr, x, torch.full_like(x, float("-inf")))
        return (x,), exp, tol
    if op == "smp_topp_renorm":
        x = _logits(M, N, 0, scale)
        return (x,), _nucleus_hf(torch.softmax(x, -1), S.TOPP_P), tol
    if op == "smp_minp_mask":
        x = _logits(M, N, 0, scale)
        p = torch.softmax(x, -1)
        keep = p >= S.MIN_P * p.amax(-1, keepdim=True)
        masked = torch.where(keep, p, torch.zeros_like(p))
        return (x,), masked / masked.sum(-1, keepdim=True), tol
    if op == "smp_typical_mask":
        x = _logits(M, N, 0, scale)
        logp = F.log_softmax(x.double(), -1)
        p = logp.exp()
        H = -(p * logp).sum(-1, keepdim=True)
        dev = ((-logp) - H).abs()
        si = dev.argsort(-1)
        p_sorted = p.gather(-1, si)
        excl = p_sorted.cumsum(-1) - p_sorted
        keep = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, si, excl <= S.TYPICAL_MASS)
        masked = torch.where(keep, p, torch.zeros_like(p))
        return (x,), masked / masked.sum(-1, keepdim=True), tol
    if op == "smp_repetition_penalty":
        x = _logits(M, N, 0, scale)
        seen = _seen(M, N, 1)
        pen = torch.where(x > 0, x / S.REP_PENALTY, x * S.REP_PENALTY)
        return (x, seen), torch.where(seen > 0, pen, x), tol
    if op == "smp_presence_penalty":
        x = _logits(M, N, 0, scale)
        c = _counts(M, N, 1)
        return (x, c), x - S.PRESENCE_PENALTY * (c > 0).to(torch.float32), tol
    if op == "smp_frequency_penalty":
        x = _logits(M, N, 0, scale)
        c = _counts(M, N, 1)
        return (x, c), x - S.FREQ_PENALTY * c.to(torch.float32), tol
    if op == "smp_logit_bias":
        x = _logits(M, N, 0, scale)
        bias = _logits(M, N, 1, 1.0)
        return (x, bias), x + bias, tol
    if op == "smp_topk_topp":
        k = S.TOPK_SAMPLE_K
        x = _logits(M, N, 0, scale)
        thr = torch.sort(x, dim=-1, descending=True).values[:, k - 1:k]
        xk = torch.where(x >= thr, x, torch.full_like(x, float("-inf")))
        return (x,), _nucleus_hf(torch.softmax(xk, -1), S.TOPP_P), tol

    if op == "smp_categorical_sample":
        x = _logits(M, N, 0, scale)
        u = _uni(M, 1)
        cdf = torch.softmax(x.double(), -1).cumsum(-1)
        exp = torch.searchsorted(cdf, u.double().view(-1, 1), right=True).squeeze(-1).clamp_(max=N - 1)
        return (x, u), exp, tol
    if op == "smp_gumbel_max":
        x = _logits(M, N, 0, scale)
        g = _gumbel(M, N, 1)
        return (x, g), (x + g).argmax(-1), tol
    if op == "smp_topp_sample":
        x = _logits(M, N, 0, scale)
        u = _uni(M, 1)
        pp = _nucleus_hf(torch.softmax(x.double(), -1), S.TOPP_P)
        exp = torch.searchsorted(pp.cumsum(-1), u.double().view(-1, 1), right=True).squeeze(-1).clamp_(max=N - 1)
        return (x, u), exp, tol
    if op == "smp_topk_sample":
        k = S.TOPK_SAMPLE_K
        x = _logits(M, N, 0, scale)
        u = _uni(M, 1)
        thr = torch.sort(x, dim=-1, descending=True).values[:, k - 1:k]
        xk = torch.where(x >= thr, x, torch.full_like(x, float("-inf")))
        cdf = torch.softmax(xk.double(), -1).cumsum(-1)
        exp = torch.searchsorted(cdf, u.double().view(-1, 1), right=True).squeeze(-1).clamp_(max=N - 1)
        return (x, u), exp, tol

    if op == "smp_spec_accept":
        q = _probs(M, N, 0, scale)
        p = _probs(M, N, 1, scale)
        d = _idvec(M, N, 2)
        u = _uni(M, 3)
        di = d.view(-1, 1)
        ratio = p.double().gather(-1, di).squeeze(-1) / q.double().gather(-1, di).squeeze(-1)
        exp = (u.double() <= torch.clamp(ratio, max=1.0)).to(torch.float32)
        return (q, p, d, u), exp, tol
    if op == "smp_spec_residual":
        q = _probs(M, N, 0, scale)
        p = _probs(M, N, 1, scale)
        resid = torch.clamp(p.double() - q.double(), min=0.0)
        exp = resid / resid.sum(-1, keepdim=True).clamp_min(1e-20)
        return (q, p), exp, tol
    if op == "smp_spec_bonus_token":
        p = _probs(M, N, 0, scale)
        u = _uni(M, 1)
        cdf = p.double().cumsum(-1)
        exp = torch.searchsorted(cdf, u.double().view(-1, 1), right=True).squeeze(-1).clamp_(max=N - 1)
        return (p, u), exp, tol
    if op == "smp_tree_attn_mask":
        T = 8
        parent = _parents(4, T, 0)
        Mt = parent.shape[0]
        exp = torch.zeros((Mt, T, T), dtype=torch.float32)
        for m in range(Mt):                          # ancestor closure via boolean matrix powers
            A = torch.zeros((T, T), dtype=torch.float64)
            for i in range(T):
                pa = parent[m, i].item()
                if pa >= 0:
                    A[i, pa] = 1.0
            reach = torch.eye(T, dtype=torch.float64)
            cur = torch.eye(T, dtype=torch.float64)
            for _ in range(T):
                cur = (cur @ A).clamp(max=1.0)
                reach = (reach + cur).clamp(max=1.0)
            exp[m] = reach.to(torch.float32)
        return (parent,), exp, tol
    if op == "smp_verify_prefix":
        K = 6
        g = torch.Generator().manual_seed(0)
        draft = torch.randint(0, 20, (M, K), generator=g, dtype=torch.int64)
        same = torch.rand((M, K), generator=g) < 0.6
        target = torch.where(same, draft, torch.randint(0, 20, (M, K), generator=g, dtype=torch.int64))
        exp = torch.zeros(M, dtype=torch.int64)
        for i in range(M):
            c = 0
            for j in range(K):
                if draft[i, j].item() == target[i, j].item():
                    c += 1
                else:
                    break
            exp[i] = c
        return (draft, target), exp, tol

    if op in _ROPE_OPS:
        D = 64
        x = _logits(M, D, 0, scale)
        if op == "smp_rope_2d":
            ph, pw = _pos(M, 1), _pos(M, 2)
            Dh = D // 2
            half = Dh // 2
            i = torch.arange(half, dtype=torch.float64)
            inv = S.ROPE_THETA ** (-(2.0 * i) / Dh)
            oh = _rope_cplx(x[:, :Dh], inv, ph)
            ow = _rope_cplx(x[:, Dh:], inv, pw)
            return (x, ph, pw), torch.cat([oh, ow], -1), tol
        if op == "smp_rope_partial":
            pos = _pos(M, 1)
            rot = int(D * S.ROT_PCT)
            rot -= rot % 2
            half = rot // 2
            i = torch.arange(half, dtype=torch.float64)
            inv = S.ROPE_THETA ** (-(2.0 * i) / rot)
            orot = _rope_cplx(x[:, :rot], inv, pos)
            return (x, pos), torch.cat([orot, x[:, rot:].double()], -1), tol
        pos = _pos(M, 1)
        inv, mscale = _inv_full(op, D)
        return (x, pos), _rope_cplx(x, inv, pos, mscale), tol

    raise AssertionError(f"no case for {op!r}")


def _agree(out, exp, tol):
    o, e = out.double(), exp.double()
    assert torch.isfinite(o).all(), "ref produced non-finite values (unstable!)"
    denom = 1.0 + e.abs().max().item()
    diff = (o - e).abs().max().item()
    return diff <= tol * denom, diff, denom


_BATTERY = tuple(op for op in S.OPS if op != "smp_no_repeat_ngram")


# --------------------------------------------------------------------------- #
# fp32 oracle correctness vs an INDEPENDENT torch compute (normal + EXTREME)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", _BATTERY)
@pytest.mark.parametrize("mode", ("normal", "extreme"))
def test_ref_matches_independent(op, mode):
    ns = S.make_reference(op, "fp32")
    inputs, expected, tol = _case(op, mode)
    ref = ns["ref_fn"](*inputs)

    if op in _INT_OPS:
        assert torch.equal(ref, expected), (op, mode, ref[:8].tolist(), expected[:8].tolist())
        return
    if op in _BOOL_OPS:
        assert torch.equal(ref.bool(), expected.bool()), (op, mode)
        return
    if op in _MASKED_OPS:
        rk, ek = torch.isfinite(ref), torch.isfinite(expected)
        assert torch.equal(rk, ek), (op, mode, "keep-set mismatch")
        assert torch.allclose(ref[rk].double(), expected[ek].double(), atol=tol, rtol=tol)
        return
    assert ref.shape == expected.shape, (op, mode, tuple(ref.shape), tuple(expected.shape))
    ok, diff, denom = _agree(ref, expected, tol)
    assert ok, f"{op} [{mode}]: max|diff|={diff:.3e} > tol*{denom:.3e}"


@pytest.mark.parametrize("op", S.OPS)
def test_determinism(op):
    ns = S.make_reference(op, "fp32")
    inputs = ns["get_inputs"](S.SHAPES[op]["minimal"], device="cpu", seed=7)
    a = ns["ref_fn"](*inputs)
    b = ns["ref_fn"](*inputs)
    outs_a = a if isinstance(a, tuple) else (a,)
    outs_b = b if isinstance(b, tuple) else (b,)
    for oa, ob in zip(outs_a, outs_b):
        assert torch.equal(oa, ob), op


def test_extreme_softmax_is_stable_where_naive_overflows():
    """temperature-softmax of 1e4-scale logits stays a valid distribution (a naive
    exp() would be inf/nan)."""
    x = 1e4 * _logits(4, 512, 0, 1.0)
    out = S.make_reference("smp_temperature", "fp32")["ref_fn"](x).double()
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(-1), torch.ones(4).double(), atol=1e-5)
    assert (torch.exp((x / S.TEMP).double()) == float("inf")).any()   # naive really overflows


# --------------------------------------------------------------------------- #
# op-specific numeric / semantic invariants
# --------------------------------------------------------------------------- #
def _ref(op, *inputs):
    return S.make_reference(op, "fp32")["ref_fn"](*inputs)


@pytest.mark.parametrize("op", sorted(_DIST_OPS))
def test_distribution_ops_sum_to_one_and_nonneg(op):
    inputs, _, _ = _case(op, "normal")
    out = _ref(op, *inputs).double()
    assert torch.all(out >= -1e-6)
    assert torch.allclose(out.sum(-1), torch.ones(out.shape[0], dtype=torch.float64), atol=1e-5)


def test_topk_mask_keeps_exactly_k():
    for op, k in S.TOPK_MASK_SIZES.items():
        x = _logits(5, 100, 0, 2.0)
        out = _ref(op, x)
        keep = torch.isfinite(out)
        assert torch.all(keep.sum(-1) == k), op
        assert torch.allclose(out[keep], x[keep])          # kept values are unchanged logits
        # the kept set is exactly the true top-k
        true_top = torch.topk(x, k, dim=-1).indices
        got = torch.sort(torch.nonzero(keep)[:, 1].view(5, k), dim=-1).values
        assert torch.equal(got, torch.sort(true_top, dim=-1).values)


def test_topp_and_minp_and_typical_are_sparser_than_softmax():
    x = _logits(7, 64, 0, 2.0)
    full = (torch.softmax(x, -1) > 0).sum(-1)
    for op in ("smp_topp_renorm", "smp_minp_mask", "smp_typical_mask", "smp_topk_topp"):
        out = _ref(op, x).double()
        assert torch.allclose(out.sum(-1), torch.ones(7, dtype=torch.float64), atol=1e-5)
        assert torch.all((out > 0).sum(-1) <= full)


def test_minp_keeps_exactly_the_above_threshold_set():
    x = _logits(6, 80, 0, 2.0)
    p = torch.softmax(x, -1)
    keep = p >= S.MIN_P * p.amax(-1, keepdim=True)
    out = _ref("smp_minp_mask", x).double()
    assert torch.equal(out > 0, keep)                       # support == above-threshold set


def test_categorical_is_inverse_cdf_of_the_uniform():
    x = _logits(64, 50, 0, 2.0)
    u = _uni(64, 1)
    idx = _ref("smp_categorical_sample", x, u)
    cdf = torch.softmax(x.double(), -1).cumsum(-1)
    # the uniform lies in (cdf[idx-1], cdf[idx]]  (right-continuous inverse CDF)
    lo = torch.where(idx > 0, cdf.gather(-1, (idx - 1).clamp_min(0).view(-1, 1)).squeeze(-1),
                     torch.zeros(64, dtype=torch.float64))
    hi = cdf.gather(-1, idx.view(-1, 1)).squeeze(-1)
    assert torch.all(u.double() <= hi + 1e-9)
    assert torch.all(u.double() > lo - 1e-9)


def test_gumbel_max_matches_argmax_of_perturbed_logits():
    x = _logits(32, 64, 0, 2.0)
    g = _gumbel(32, 64, 1)
    idx = _ref("smp_gumbel_max", x, g)
    assert torch.equal(idx, (x + g).argmax(-1))


def test_topp_and_topk_samplers_pick_a_kept_token():
    x = _logits(32, 64, 0, 2.0)
    u = _uni(32, 1)
    # top-p sampler must return a token inside the nucleus
    pp = _nucleus_hf(torch.softmax(x, -1), S.TOPP_P)
    idx = _ref("smp_topp_sample", x, u)
    assert torch.all(pp.gather(-1, idx.view(-1, 1)).squeeze(-1) > 0)
    # top-k sampler must return one of the top-k logits
    k = S.TOPK_SAMPLE_K
    top = torch.topk(x, k, dim=-1).indices
    idxk = _ref("smp_topk_sample", x, u)
    assert torch.all((idxk.view(-1, 1) == top).any(-1))


def test_spec_accept_rule_and_always_accepts_when_target_dominates():
    q = _probs(32, 40, 0, 2.0)
    p = _probs(32, 40, 1, 2.0)
    d = _idvec(32, 40, 2)
    u = _uni(32, 3)
    acc = _ref("smp_spec_accept", q, p, d, u).double()
    di = d.view(-1, 1)
    ratio = p.double().gather(-1, di).squeeze(-1) / q.double().gather(-1, di).squeeze(-1)
    assert torch.equal(acc.bool(), (u.double() <= torch.clamp(ratio, max=1.0)))
    # if p[d] >= q[d] (ratio>=1) the token is accepted for every u in [0,1)
    dominate = ratio >= 1.0
    assert torch.all(acc[dominate] == 1.0)


def test_spec_residual_is_a_valid_distribution_supported_where_target_exceeds_draft():
    q = _probs(16, 48, 0, 2.0)
    p = _probs(16, 48, 1, 2.0)
    r = _ref("smp_spec_residual", q, p).double()
    assert torch.all(r >= -1e-9)
    assert torch.allclose(r.sum(-1), torch.ones(16, dtype=torch.float64), atol=1e-5)
    assert torch.equal(r > 0, (p.double() - q.double()) > 0)


def test_tree_attn_mask_is_ancestor_reachability():
    T = 8
    parent = _parents(3, T, 0)
    mask = _ref("smp_tree_attn_mask", parent).double()
    Mt = parent.shape[0]
    for m in range(Mt):
        for i in range(T):
            assert mask[m, i, i] == 1.0                      # attends to itself
            anc = set()
            a = parent[m, i].item()
            while a >= 0:
                anc.add(a)
                a = parent[m, a].item()
            for j in range(T):
                want = 1.0 if (j == i or j in anc) else 0.0
                assert mask[m, i, j].item() == want, (m, i, j)
            assert torch.all(mask[m, i, i + 1:] == 0.0)       # only lower-triangular (ancestors precede)


def test_verify_prefix_counts_leading_matches():
    draft = torch.tensor([[3, 3, 3, 9], [1, 2, 3, 4], [5, 6, 7, 8]])
    target = torch.tensor([[3, 3, 0, 9], [1, 2, 3, 4], [0, 6, 7, 8]])
    out = _ref("smp_verify_prefix", draft, target)
    assert out.tolist() == [2, 4, 0]
    assert out.dtype == torch.int64


def test_no_repeat_ngram_blocks_the_repeated_continuation():
    # prev tokens: ... [7, 8] appeared and was followed by 5; the current suffix is
    # also [7, 8], so token 5 must be blocked (n=3 -> match the last 2 tokens).
    V = 16
    x = torch.zeros(1, V, dtype=torch.float32)
    prev = torch.tensor([[7, 8, 5, 1, 2, 7, 8]], dtype=torch.int64)   # suffix (7,8); (7,8)->5 earlier
    out = _ref("smp_no_repeat_ngram", x, prev)
    assert out[0, 5].item() == float("-inf")                 # the repeated continuation is banned
    kept = [v for v in range(V) if v != 5]
    assert torch.all(torch.isfinite(out[0, kept]))           # nothing else is touched


def test_no_repeat_ngram_matches_independent_bruteforce():
    V, L = 40, 12
    g = torch.Generator().manual_seed(4)
    x = torch.randn(5, V, generator=g, dtype=torch.float32)
    prev = torch.randint(0, 6, (5, L), generator=g, dtype=torch.int64)   # small alphabet -> real repeats
    out = _ref("smp_no_repeat_ngram", x, prev).double()
    n = S.NGRAM_N
    exp = x.double().clone()
    for i in range(5):
        row = prev[i].tolist()
        suffix = row[L - n + 1:L]
        for j in range(0, L - n + 1):
            if row[j:j + n - 1] == suffix:
                exp[i, row[j + n - 1]] = float("-inf")
    assert torch.equal(torch.isfinite(out), torch.isfinite(exp))
    fin = torch.isfinite(out)
    assert torch.allclose(out[fin], exp[fin])


def test_rope_is_norm_preserving_and_pos0_is_identity():
    D = 64
    x = _logits(6, D, 0, 2.0)
    zero = torch.zeros(6, dtype=torch.float32)
    for op in ("smp_rope_linear_pi", "smp_rope_ntk", "smp_rope_dynamic_ntk",
               "smp_rope_partial", "smp_rope_llama3"):
        pos = _pos(6, 1)
        out = _ref(op, x, pos).double()
        assert torch.allclose(out.norm(dim=-1), x.double().norm(dim=-1), atol=1e-3, rtol=1e-3), op
        out0 = _ref(op, x, zero).double()
        assert torch.allclose(out0, x.double(), atol=1e-4), (op, "pos0 must be identity")


def test_rope_2d_norm_preserving_and_pos0_identity():
    D = 64
    x = _logits(6, D, 0, 2.0)
    zero = torch.zeros(6, dtype=torch.float32)
    out = _ref("smp_rope_2d", x, _pos(6, 1), _pos(6, 2)).double()
    assert torch.allclose(out.norm(dim=-1), x.double().norm(dim=-1), atol=1e-3, rtol=1e-3)
    out0 = _ref("smp_rope_2d", x, zero, zero).double()
    assert torch.allclose(out0, x.double(), atol=1e-4)


def test_rope_yarn_scales_by_mscale_and_pos0_is_mscale_times_identity():
    D = 64
    x = _logits(6, D, 0, 2.0)
    mscale = 0.1 * math.log(S.ROPE_SCALE) + 1.0
    out0 = _ref("smp_rope_yarn", x, torch.zeros(6), ).double()
    assert torch.allclose(out0, x.double() * mscale, atol=1e-3)       # pos0 -> x * mscale
    out = _ref("smp_rope_yarn", x, _pos(6, 1)).double()
    assert torch.allclose(out.norm(dim=-1), x.double().norm(dim=-1) * mscale, atol=1e-2, rtol=1e-3)
