"""CPU-only tests for the breadth REDUCTION / NORMALIZATION task engine.

Every ``ref_fn`` is cross-checked against an INDEPENDENT torch computation on a
DIFFERENT code path than the manual stable formula it wraps (torch.softmax /
torch.log_softmax / F.cross_entropy / torch.var / torch.logsumexp / autograd /
Categorical.entropy / manual fp64 scans), at a tight fp32 tolerance AND on an
EXTREME-MAGNITUDE input (huge logits / large means) that a naive
non-max-subtracted / E[x^2]-E[x]^2 implementation would fail - so numerical
STABILITY is proven, not assumed. Also asserts the ABI surface (45 ``red_`` ops,
bf16/fp16/fp32 sweep), the namespace contract, that every seed parses + compiles +
defines its entry, the shape catalog round-trips through ``parse_shape``, arity
matches ``get_inputs``, ref preserves the input dtype, and op-specific numeric /
semantic invariants (softmax sums to 1, layer-norm zero-mean/unit-var, top-k
descending, argmax first-occurrence tie rule, cumulative-max monotonicity,
divergence non-negativity, ...). All fp32/fp64 on CPU (no GPU / triton runtime).
"""

from __future__ import annotations

import ast
import math

import pytest
import torch
import torch.nn.functional as F

from kore.tasks._genops import DTYPES
from kore.tasks.breadth import reduce_ext as R

DTYPE_NAMES = ("bf16", "fp16", "fp32")

# expected arity per op (independent of make_reference, so a wrong arity is caught)
_ARITY = {
    "red_softmax": 1, "red_log_softmax": 1, "red_softmax_temp": 1,
    "red_online_softmax": 1, "red_softmax_dim0": 1, "red_gumbel_softmax": 2,
    "red_softmax_bwd": 2, "red_log_softmax_bwd": 2,
    "red_logsumexp": 1, "red_logsumexp_dim0": 1, "red_entropy": 1,
    "red_logcumsumexp": 1,
    "red_var": 1, "red_var_unbiased": 1, "red_std": 1, "red_welford": 1,
    "red_rms": 1, "red_rmsnorm": 1, "red_layernorm": 1, "red_running_stats": 1,
    "red_cross_entropy": 2, "red_cross_entropy_bwd": 2, "red_label_smoothing_ce": 2,
    "red_z_loss": 1, "red_cross_entropy_zloss": 2, "red_focal_loss": 2,
    "red_soft_cross_entropy": 2, "red_bce_with_logits": 2,
    "red_kl_div": 2, "red_js_div": 2,
    "red_topk2": 1, "red_topk8": 1, "red_topk50": 1, "red_topk256": 1,
    "red_topp_renorm": 1, "red_argmax": 1, "red_argmin": 1,
    "red_cummax": 1, "red_cummin": 1,
    "red_norm_l1": 1, "red_norm_l2": 1, "red_norm_linf": 1, "red_l2_normalize": 1,
    "red_pairwise_dist": 1, "red_cosine_sim": 1,
}
# note: pairwise/cosine take 2 tensors -> arity 2 (fix below)
_ARITY["red_pairwise_dist"] = 2
_ARITY["red_cosine_sim"] = 2

_INT_OPS = frozenset({"red_argmax", "red_argmin"})
_TUPLE_OPS = frozenset({"red_welford", "red_running_stats"})
_VARFAM = frozenset({
    "red_var", "red_var_unbiased", "red_std", "red_welford",
    "red_layernorm", "red_running_stats",
})
_DIM0 = frozenset({"red_softmax_dim0", "red_logsumexp_dim0", "red_running_stats"})


# --------------------------------------------------------------------------- #
# metadata / ABI surface
# --------------------------------------------------------------------------- #
def test_ops_and_metadata():
    assert isinstance(R.OPS, tuple) and len(R.OPS) == 45
    assert len(set(R.OPS)) == 45
    assert all(op.startswith("red_") for op in R.OPS)
    assert set(R.OPS) == set(_ARITY)
    assert R.DEFAULT_DTYPES == ("bf16", "fp16", "fp32")
    for op in R.OPS:
        assert op in R.OP_DTYPES and op in R.SHAPES
        assert R.op_dtypes(op) == R.OP_DTYPES[op]
        assert all(dt in DTYPE_NAMES for dt in R.op_dtypes(op))
    assert R.op_names() == list(R.OPS)


@pytest.mark.parametrize("op", R.OPS)
def test_namespace_contract(op):
    ns = R.make_reference(op, "fp32")
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


@pytest.mark.parametrize("op", R.OPS)
def test_seed_parses_compiles_and_defines_entry(op):
    for dtype in R.op_dtypes(op):
        src = R.seed_source(op, dtype)
        tree = ast.parse(src)                              # valid Python
        compile(src, f"<seed:{op}:{dtype}>", "exec")       # compiles to bytecode
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert op in funcs, (op, dtype, funcs)
        assert any(isinstance(n, ast.FunctionDef) and n.name == op
                   for n in tree.body), f"{op} entry must be a top-level def"


@pytest.mark.parametrize("op", R.OPS)
def test_shapes_parse_roundtrip(op):
    ns = R.make_reference(op, "fp32")
    parse = ns["parse_shape"]
    sh = R.SHAPES[op]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert parse(s) == spec, (op, parse(s), spec)


@pytest.mark.parametrize("op", R.OPS)
def test_shape_catalog_is_realistic(op):
    """Rows in {4096, 16384}; wide N/V incl. large-vocab (>=32768) and a non-pow2 tail."""
    sh = R.SHAPES[op]
    assert set(sh) >= {"minimal", "primary", "validation"}
    assert isinstance(sh["validation"], list) and sh["validation"]
    widths, rows, has_nonpow2 = [], [], False
    for spec in [sh["primary"], *sh["validation"]]:
        rows.append(spec["M"])
        w = spec.get("N", spec.get("V"))
        widths.append(w)
        if w & (w - 1) != 0:
            has_nonpow2 = True
    assert max(widths) >= 32768, (op, widths)       # a genuinely wide reduction
    assert 4096 in rows and 16384 in rows, (op, rows)
    assert has_nonpow2, (op, widths)                # non-pow2 tail present


@pytest.mark.parametrize("op", R.OPS)
def test_arity_matches_get_inputs(op):
    ns = R.make_reference(op, "fp32")
    inputs = ns["get_inputs"](R.SHAPES[op]["minimal"], device="cpu", seed=0)
    assert isinstance(inputs, tuple)
    assert len(inputs) == ns["arity"] == _ARITY[op]


@pytest.mark.parametrize("op", R.OPS)
@pytest.mark.parametrize("dtype", ("bf16", "fp16"))
def test_ref_preserves_input_dtype(op, dtype):
    ns = R.make_reference(op, dtype)
    inputs = ns["get_inputs"](R.SHAPES[op]["minimal"], device="cpu", seed=1)
    out = ns["ref_fn"](*inputs)
    outs = out if isinstance(out, (tuple, list)) else (out,)
    if op in _INT_OPS:
        assert all(o.dtype == torch.int64 for o in outs)
    else:
        tdt = getattr(torch, DTYPES[dtype][0])
        assert all(o.dtype == tdt for o in outs)


def test_baseline_matches_reference_shapes():
    """The torch baseline (native dtype) agrees in SHAPE with the fp32 oracle on
    every op (same math; baseline is the timed 'production' bar)."""
    for op in R.OPS:
        ns = R.make_reference(op, "fp32")
        inputs = ns["get_inputs"](R.SHAPES[op]["minimal"], device="cpu", seed=3)
        r = ns["ref_fn"](*inputs)
        b = ns["baseline_fn"](*inputs)
        rs = r if isinstance(r, (tuple, list)) else (r,)
        bs = b if isinstance(b, (tuple, list)) else (b,)
        assert len(rs) == len(bs), (op, "tuple len")
        for ro, bo in zip(rs, bs):
            assert ro.shape == bo.shape, (op, tuple(ro.shape), tuple(bo.shape))


# --------------------------------------------------------------------------- #
# input builders + independent oracles for the correctness battery
# --------------------------------------------------------------------------- #
def _logits(M, N, seed, scale, offset=0.0):
    g = torch.Generator().manual_seed(seed)
    return offset + scale * torch.randn(M, N, generator=g, dtype=torch.float32)


def _targets(M, V, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (M,), generator=g, dtype=torch.int64)


def _rand01(M, N, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(M, N, generator=g, dtype=torch.float32)


def _gumbel(M, N, seed):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(M, N, generator=g, dtype=torch.float32).clamp_(1e-9, 1.0)
    return -torch.log(-torch.log(u))


def _mn(op):
    if op in R.TOPK_SIZES:
        return 6, R.TOPK_SIZES[op] + 21           # N > k, non-pow2
    if op in R._VOCAB_OPS:
        return 6, 51
    if op in _DIM0:
        return 8, 5
    return 6, 41


def _manual_scan(x, kind):
    """Independent fp64 sequential scan (a DIFFERENT path than the torch builtin ref)."""
    xd = x.double()
    M, N = xd.shape
    out = torch.empty_like(xd)
    for i in range(M):
        if kind == "logcumsumexp":
            rm, rs = -math.inf, 0.0
            for j in range(N):
                v = xd[i, j].item()
                nm = max(rm, v)
                rs = rs * math.exp(rm - nm) + math.exp(v - nm)
                rm = nm
                out[i, j] = rm + math.log(rs)
        elif kind == "cummax":
            r = -math.inf
            for j in range(N):
                r = max(r, xd[i, j].item())
                out[i, j] = r
        else:  # cummin
            r = math.inf
            for j in range(N):
                r = min(r, xd[i, j].item())
                out[i, j] = r
    return out


def _case(op, mode):
    """Return (inputs, expected, tol). ``expected`` comes from an INDEPENDENT torch
    path; ``mode`` in {'normal','extreme'} (extreme drives huge magnitudes)."""
    varfam = op in _VARFAM
    if mode == "extreme":
        scale, offset = (1.0, 1e3) if varfam else (1e3, 0.0)
        tol = 1e-2
    else:
        scale, offset = 1.5, 0.0
        tol = 3e-3
    M, N = _mn(op)

    if op in ("red_softmax", "red_online_softmax"):
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.softmax(x, -1), tol
    if op == "red_log_softmax":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.log_softmax(x, -1), tol
    if op == "red_softmax_temp":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.softmax(x / R.TEMP, -1), tol
    if op == "red_softmax_dim0":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.softmax(x, 0), tol
    if op == "red_gumbel_softmax":
        x = _logits(M, N, 0, scale, offset)
        gum = _gumbel(M, N, 1)
        return (x, gum), torch.softmax((x + gum) / R.GUMBEL_TAU, -1), tol

    if op == "red_softmax_bwd":
        x = _logits(M, N, 0, scale, offset)
        dy = _logits(M, N, 1, 1e2 if mode == "extreme" else 1.0)
        y = torch.softmax(x, -1)
        xr = x.clone().requires_grad_(True)
        torch.softmax(xr, -1).backward(dy)
        return (y, dy), xr.grad.detach(), tol
    if op == "red_log_softmax_bwd":
        x = _logits(M, N, 0, scale, offset)
        dy = _logits(M, N, 1, 1e2 if mode == "extreme" else 1.0)
        y = torch.log_softmax(x, -1)
        xr = x.clone().requires_grad_(True)
        torch.log_softmax(xr, -1).backward(dy)
        return (y, dy), xr.grad.detach(), tol

    if op == "red_logsumexp":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.logsumexp(x, -1), tol
    if op == "red_logsumexp_dim0":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.logsumexp(x, 0), tol
    if op == "red_entropy":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.distributions.Categorical(logits=x).entropy(), tol
    if op == "red_logcumsumexp":
        x = _logits(M, N, 0, scale, offset)
        return (x,), _manual_scan(x, "logcumsumexp"), tol

    if op == "red_var":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.var(x, dim=-1, unbiased=False), tol
    if op == "red_var_unbiased":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.var(x, dim=-1, unbiased=True), tol
    if op == "red_std":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.std(x, dim=-1, unbiased=True), tol
    if op == "red_welford":
        x = _logits(M, N, 0, scale, offset)
        return (x,), (x.mean(-1), x.var(-1, unbiased=False)), tol
    if op == "red_rms":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.linalg.vector_norm(x, ord=2, dim=-1) / (N ** 0.5), tol
    if op == "red_rmsnorm":
        x = _logits(M, N, 0, scale, offset)
        if hasattr(F, "rms_norm"):
            exp = F.rms_norm(x, (N,), eps=R.RMS_EPS)
        else:
            exp = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + R.RMS_EPS)
        return (x,), exp, tol
    if op == "red_layernorm":
        x = _logits(M, N, 0, scale, offset)
        return (x,), F.layer_norm(x, (N,), eps=R.LN_EPS), tol
    if op == "red_running_stats":
        x = _logits(M, N, 0, scale, offset)
        return (x,), (x.mean(0), x.var(0, unbiased=False)), tol

    if op == "red_cross_entropy":
        x = _logits(M, N, 0, scale, offset)
        t = _targets(M, N, 2)
        return (x, t), F.cross_entropy(x, t, reduction="none"), tol
    if op == "red_cross_entropy_bwd":
        x = _logits(M, N, 0, scale, offset)
        t = _targets(M, N, 2)
        xr = x.clone().requires_grad_(True)
        F.cross_entropy(xr, t, reduction="sum").backward()
        return (x, t), xr.grad.detach(), tol
    if op == "red_label_smoothing_ce":
        x = _logits(M, N, 0, scale, offset)
        t = _targets(M, N, 2)
        return (x, t), F.cross_entropy(x, t, reduction="none", label_smoothing=R.LS_EPS), tol
    if op == "red_z_loss":
        x = _logits(M, N, 0, scale, offset)
        return (x,), R.ZLOSS_COEF * torch.logsumexp(x, -1) ** 2, tol
    if op == "red_cross_entropy_zloss":
        x = _logits(M, N, 0, scale, offset)
        t = _targets(M, N, 2)
        exp = F.cross_entropy(x, t, reduction="none") + R.ZLOSS_COEF * torch.logsumexp(x, -1) ** 2
        return (x, t), exp, tol
    if op == "red_focal_loss":
        x = _logits(M, N, 0, scale, offset)
        t = _targets(M, N, 2)
        logpt = F.log_softmax(x, -1).gather(-1, t.view(-1, 1)).squeeze(-1)  # stable path
        pt = torch.exp(logpt)
        return (x, t), -((1.0 - pt) ** R.FOCAL_GAMMA) * logpt, tol
    if op == "red_soft_cross_entropy":
        x = _logits(M, N, 0, scale, offset)
        q = torch.softmax(_logits(M, N, 1, 1.5), -1)
        return (x, q), -(q * F.log_softmax(x, -1)).sum(-1), tol
    if op == "red_bce_with_logits":
        x = _logits(M, N, 0, scale, offset)
        z = _rand01(M, N, 1)
        return (x, z), F.binary_cross_entropy_with_logits(x, z, reduction="none").mean(-1), tol

    if op == "red_kl_div":
        lp = _logits(M, N, 0, scale, offset)
        lq = _logits(M, N, 1, scale, offset)
        exp = F.kl_div(F.log_softmax(lq, -1), F.softmax(lp, -1), reduction="none").sum(-1)
        return (lp, lq), exp, tol
    if op == "red_js_div":
        lp = _logits(M, N, 0, scale, offset)
        lq = _logits(M, N, 1, scale, offset)
        logp = F.log_softmax(lp.double(), -1)
        logq = F.log_softmax(lq.double(), -1)
        p, q = logp.exp(), logq.exp()
        logm = torch.log(0.5 * (p + q))
        z = torch.zeros_like(p)
        kl_pm = torch.where(p > 0, p * (logp - logm), z).sum(-1)
        kl_qm = torch.where(q > 0, q * (logq - logm), z).sum(-1)
        return (lp, lq), 0.5 * kl_pm + 0.5 * kl_qm, tol

    if op in R.TOPK_SIZES:
        k = R.TOPK_SIZES[op]
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.topk(x, k, dim=-1).values, tol
    if op == "red_topp_renorm":
        x = _logits(M, N, 0, scale, offset)
        probs = torch.softmax(x, -1)
        sp, si = torch.sort(probs, dim=-1, descending=True)
        remove = sp.cumsum(-1) > R.TOPP_P
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        keep = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, si, ~remove)
        masked = torch.where(keep, probs, torch.zeros_like(probs))
        return (x,), masked / masked.sum(-1, keepdim=True), tol
    if op == "red_argmax":
        x = _logits(M, N, 0, scale, offset)
        return (x,), x.argmax(-1), tol
    if op == "red_argmin":
        x = _logits(M, N, 0, scale, offset)
        return (x,), x.argmin(-1), tol
    if op == "red_cummax":
        x = _logits(M, N, 0, scale, offset)
        return (x,), _manual_scan(x, "cummax"), tol
    if op == "red_cummin":
        x = _logits(M, N, 0, scale, offset)
        return (x,), _manual_scan(x, "cummin"), tol

    if op == "red_norm_l1":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.linalg.vector_norm(x, ord=1, dim=-1), tol
    if op == "red_norm_l2":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.linalg.vector_norm(x, ord=2, dim=-1), tol
    if op == "red_norm_linf":
        x = _logits(M, N, 0, scale, offset)
        return (x,), torch.linalg.vector_norm(x, ord=float("inf"), dim=-1), tol
    if op == "red_l2_normalize":
        x = _logits(M, N, 0, scale, offset)
        return (x,), F.normalize(x, p=2.0, dim=-1, eps=R.NORM_EPS), tol
    if op == "red_pairwise_dist":
        a = _logits(M, N, 0, scale, offset)
        b = _logits(M, N, 1, scale, offset)
        return (a, b), torch.linalg.vector_norm(a - b, ord=2, dim=-1), tol
    if op == "red_cosine_sim":
        a = _logits(M, N, 0, scale, offset)
        b = _logits(M, N, 1, scale, offset)
        return (a, b), F.cosine_similarity(a, b, dim=-1, eps=R.COS_EPS), tol

    raise AssertionError(f"no case for {op!r}")


def _agree(out, exp, tol):
    o, e = out.double(), exp.double()
    assert torch.isfinite(o).all(), "ref produced non-finite values (unstable!)"
    denom = 1.0 + e.abs().max().item()
    diff = (o - e).abs().max().item()
    return diff <= tol * denom, diff, denom


# --------------------------------------------------------------------------- #
# fp32 oracle correctness vs an INDEPENDENT torch compute (normal + EXTREME)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", R.OPS)
@pytest.mark.parametrize("mode", ("normal", "extreme"))
def test_ref_matches_independent(op, mode):
    ns = R.make_reference(op, "fp32")
    inputs, expected, tol = _case(op, mode)
    ref = ns["ref_fn"](*inputs)
    if op in _INT_OPS:                       # index outputs: exact match (no ties for randn)
        assert torch.equal(ref, expected), (op, mode)
        return
    refs = ref if isinstance(ref, tuple) else (ref,)
    exps = expected if isinstance(expected, tuple) else (expected,)
    assert len(refs) == len(exps)
    for r, e in zip(refs, exps):
        assert r.shape == e.shape, (op, mode, tuple(r.shape), tuple(e.shape))
        ok, diff, denom = _agree(r, e, tol)
        assert ok, f"{op} [{mode}]: max|diff|={diff:.3e} > tol*{denom:.3e}"


def test_naive_variance_would_be_unstable_but_ref_is_not():
    """Sanity: on a large-mean input the NAIVE E[x^2]-E[x]^2 (fp32) is badly wrong,
    while our centered ref_fn matches torch.var - i.e. the stability is real."""
    x = 1e4 + torch.randn(4, 4096, dtype=torch.float32)
    ref = R.make_reference("red_var", "fp32")["ref_fn"](x)
    true = torch.var(x, dim=-1, unbiased=False)
    naive = (x * x).mean(-1) - x.mean(-1) ** 2          # catastrophic cancellation in fp32
    # the centered ref matches torch.var closely; the naive path does NOT (it is
    # dominated by fp32 rounding of two ~1e8 quantities that nearly cancel).
    assert torch.allclose(ref.double(), true.double(), atol=1e-2, rtol=1e-3)
    assert not torch.allclose(naive.double(), true.double(), atol=1e-1, rtol=1e-2)


# --------------------------------------------------------------------------- #
# op-specific numeric / semantic invariants
# --------------------------------------------------------------------------- #
def _ref(op, *inputs):
    return R.make_reference(op, "fp32")["ref_fn"](*inputs)


def test_softmax_is_a_distribution():
    x = _logits(5, 37, 0, 3.0)
    out = _ref("red_softmax", x)
    assert torch.allclose(out.sum(-1), torch.ones(5), atol=1e-5)
    assert torch.all(out >= 0)


def test_log_softmax_exp_sums_to_one():
    x = _logits(5, 37, 0, 3.0)
    out = _ref("red_log_softmax", x)
    assert torch.allclose(out.exp().sum(-1), torch.ones(5), atol=1e-5)


def test_softmax_dim0_normalizes_over_rows():
    x = _logits(9, 4, 0, 2.0)
    out = _ref("red_softmax_dim0", x)
    assert torch.allclose(out.sum(0), torch.ones(4), atol=1e-5)


def test_layernorm_zero_mean_unit_var():
    x = _logits(6, 128, 0, 2.0)
    out = _ref("red_layernorm", x).double()
    assert out.mean(-1).abs().max().item() < 1e-4
    assert (out.var(-1, unbiased=False) - 1.0).abs().max().item() < 1e-2


def test_rmsnorm_has_unit_rms():
    x = _logits(6, 128, 0, 2.0)
    out = _ref("red_rmsnorm", x).double()
    rms = out.pow(2).mean(-1).sqrt()
    assert (rms - 1.0).abs().max().item() < 1e-2


def test_l2_normalize_has_unit_norm():
    x = _logits(6, 128, 0, 2.0)
    out = _ref("red_l2_normalize", x).double()
    assert (torch.linalg.vector_norm(out, ord=2, dim=-1) - 1.0).abs().max().item() < 1e-4


def test_welford_matches_torch():
    x = _logits(6, 100, 0, 2.0)
    mean, var = _ref("red_welford", x)
    assert torch.allclose(mean.double(), x.mean(-1).double(), atol=1e-4)
    assert torch.allclose(var.double(), x.var(-1, unbiased=False).double(), atol=1e-4)


def test_var_unbiased_is_bessel_corrected():
    x = _logits(4, 64, 0, 2.0)
    v = _ref("red_var", x).double()
    vu = _ref("red_var_unbiased", x).double()
    assert torch.allclose(vu, v * (64.0 / 63.0), atol=1e-4, rtol=1e-4)


def test_topk_descending_and_matches_sorted():
    x = _logits(5, 200, 0, 2.0)
    out = _ref("red_topk8", x)
    assert out.shape == (5, 8)
    assert torch.all(out[:, 1:] <= out[:, :-1] + 1e-6)            # descending
    exp = torch.sort(x, dim=-1, descending=True).values[:, :8]
    assert torch.allclose(out.double(), exp.double(), atol=1e-5)


def test_topp_renorm_is_a_distribution_and_sparser():
    x = _logits(7, 64, 0, 2.0)
    out = _ref("red_topp_renorm", x).double()
    assert torch.allclose(out.sum(-1), torch.ones(7, dtype=torch.float64), atol=1e-5)
    assert torch.all(out >= 0)
    assert torch.all((out > 0).sum(-1) <= (torch.softmax(x, -1) > 0).sum(-1))


def test_argmax_argmin_first_occurrence_tie_rule():
    x = torch.tensor([[1.0, 5.0, 5.0, 2.0, 5.0],
                      [-3.0, -3.0, 0.0, -3.0, 1.0]])
    amax = _ref("red_argmax", x)
    amin = _ref("red_argmin", x)
    assert amax.tolist() == [1, 4]        # first index of the max value (5 -> idx 1; 1 -> idx 4)
    assert amin.tolist() == [0, 0]        # first index of the min value (-3 -> idx 0)
    assert amax.dtype == torch.int64 and amin.dtype == torch.int64


def test_cummax_monotone_and_cummin_matches_row_extreme():
    x = _logits(4, 50, 0, 2.0)
    cmax = _ref("red_cummax", x).double()
    cmin = _ref("red_cummin", x).double()
    assert torch.all(cmax[:, 1:] >= cmax[:, :-1])
    assert torch.all(cmin[:, 1:] <= cmin[:, :-1])
    assert torch.allclose(cmax[:, -1], x.amax(-1).double())
    assert torch.allclose(cmin[:, -1], x.amin(-1).double())


def test_cross_entropy_matches_functional():
    x = _logits(6, 128, 0, 2.0)
    t = _targets(6, 128, 1)
    out = _ref("red_cross_entropy", x, t).double()
    exp = F.cross_entropy(x, t, reduction="none").double()
    assert torch.allclose(out, exp, atol=1e-4, rtol=1e-4)


def test_cross_entropy_bwd_is_softmax_minus_onehot():
    x = _logits(6, 96, 0, 2.0)
    t = _targets(6, 96, 1)
    grad = _ref("red_cross_entropy_bwd", x, t).double()
    p = torch.softmax(x, -1).double()
    p[torch.arange(6), t] -= 1.0
    assert torch.allclose(grad, p, atol=1e-5)
    # row-sum of the gradient is ~0 (probabilities sum to 1, minus one onehot)
    assert grad.sum(-1).abs().max().item() < 1e-4


def test_softmax_bwd_row_sum_is_zero():
    x = _logits(6, 80, 0, 2.0)
    y = torch.softmax(x, -1)
    dy = _logits(6, 80, 1, 1.0)
    dx = _ref("red_softmax_bwd", y, dy).double()
    assert dx.sum(-1).abs().max().item() < 1e-4          # sum_j y_j(dy_j - mean) == 0


def test_kl_and_js_nonnegative_and_zero_on_equal():
    lp = _logits(5, 64, 0, 2.0)
    lq = _logits(5, 64, 1, 2.0)
    kl = _ref("red_kl_div", lp, lq).double()
    js = _ref("red_js_div", lp, lq).double()
    assert torch.all(kl >= -1e-5) and torch.all(js >= -1e-5)
    assert _ref("red_kl_div", lp, lp).abs().max().item() < 1e-5     # KL(p||p) == 0
    assert _ref("red_js_div", lp, lp).abs().max().item() < 1e-5     # JS(p,p) == 0
    js_sym = _ref("red_js_div", lq, lp).double()
    assert torch.allclose(js, js_sym, atol=1e-5)                    # JS is symmetric


def test_entropy_within_bounds():
    x = _logits(5, 64, 0, 2.0)
    h = _ref("red_entropy", x).double()
    assert torch.all(h >= -1e-5) and torch.all(h <= math.log(64) + 1e-4)
    # a near-uniform row has entropy ~ log(N)
    hu = _ref("red_entropy", torch.zeros(1, 64)).item()
    assert abs(hu - math.log(64)) < 1e-4


def test_logsumexp_dim0_and_running_stats_reduce_over_rows():
    x = _logits(10, 7, 0, 2.0)
    lse0 = _ref("red_logsumexp_dim0", x).double()
    assert torch.allclose(lse0, torch.logsumexp(x, 0).double(), atol=1e-4)
    mean, var = _ref("red_running_stats", x)
    assert torch.allclose(mean.double(), x.mean(0).double(), atol=1e-4)
    assert torch.allclose(var.double(), x.var(0, unbiased=False).double(), atol=1e-4)


def test_extreme_softmax_is_stable_where_naive_overflows():
    """softmax of 1e4-scale logits: the ref stays a valid distribution (a naive
    exp() would be inf/nan)."""
    x = 1e4 * _logits(4, 512, 0, 1.0)
    out = _ref("red_softmax", x).double()
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(-1), torch.ones(4).double(), atol=1e-5)
    assert (torch.exp(x.double()) == float("inf")).any()     # naive really overflows
