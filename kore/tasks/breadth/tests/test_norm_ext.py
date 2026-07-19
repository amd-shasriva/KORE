"""CPU-only tests for the breadth normalization-frontier authoring engine.

Every ``ref_fn`` oracle is checked against an INDEPENDENT torch computation (a
different code path than the oracle itself):
  * forward / fused / stats norms vs ``torch.nn.functional`` (layer_norm /
    group_norm / instance_norm / batch_norm / normalize) or a hand-rolled fp32
    formula distinct from the oracle's,
  * the BACKWARD oracles (torch autograd on the fp32 forward) vs a HAND-DERIVED
    analytic backward - so the dx / dweight / dbias reductions are proven exactly
    right against two independent derivations,
  * the norm+quant oracles: the per-token scale is recomputed independently
    (tight) and the dequantized output reconstructs the normed row (SNR bound).

Plus the ABI / arity / seed-compiles / seeds-load / shapes-parse / dtype-
preservation / mutates_input contract. All fp32 on CPU (no GPU / triton launch).
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from kore.tasks.breadth import norm_ext as N

# --------------------------------------------------------------------------- #
# op groupings
# --------------------------------------------------------------------------- #
_BWD_KINDS = {"rmsnorm_bwd", "layernorm_bwd", "layernorm_nobias_bwd",
              "groupnorm_bwd", "l2norm_bwd"}
_QUANT_KINDS = {"rmsnorm_quant", "layernorm_quant", "add_rmsnorm_quant"}

BWD_OPS = [op for op in N.OPS if N.KIND[op] in _BWD_KINDS]
QUANT_OPS = [op for op in N.OPS if N.KIND[op] in _QUANT_KINDS]
FWD_OPS = [op for op in N.OPS if N.KIND[op] not in _BWD_KINDS | _QUANT_KINDS]

DTYPE_NAMES = ("bf16", "fp16", "fp32", "fp8", "int8")
_ATOL, _RTOL = 1e-4, 1e-3
_BWD_ATOL, _BWD_RTOL = 2e-3, 2e-3


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clone(inp):
    return tuple(t.clone() if torch.is_tensor(t) else t for t in inp)


def _tup(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _small(op: str) -> dict:
    """Tiny CPU shape per op-kind (logic is generic over the pinned hidden size)."""
    kind = N.KIND[op]
    if kind in ("groupnorm", "groupnorm_stats", "groupnorm_silu", "groupnorm_bwd"):
        return {"M": 6, "N": 64}                    # C=64, 32 groups -> width 2
    if kind in ("instancenorm", "batchnorm", "batchnorm_stats"):
        return {"N": 2, "C": 8, "L": 10}
    if kind in ("qk_rmsnorm", "qk_layernorm"):
        return {"B": 1, "S": 3, "H": 2, "D": 16}
    if kind == "rmsnorm_swiglu":
        return {"M": 8, "N": 16}                     # input width 16 -> H=8
    if kind == "weightnorm":
        return {"M": 6, "N": 20}
    return {"M": 8, "N": 20}                          # non-pow2 hidden


def _snr_db(o, r):
    o, r = o.float(), r.float()
    noise = (o - r).norm().item()
    sig = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(sig / noise) if sig > 0 else -999.0


def _close(a, b, atol=_ATOL, rtol=_RTOL):
    return torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)


def _assert_tuples_close(got, exp, atol=_ATOL, rtol=_RTOL, label=""):
    g, e = _tup(got), _tup(exp)
    assert len(g) == len(e), f"{label}: {len(g)} vs {len(e)} outputs"
    for i, (a, b) in enumerate(zip(g, e)):
        assert a.shape == b.shape, f"{label}[{i}] shape {a.shape} vs {b.shape}"
        md = (a.float() - b.float()).abs().max().item()
        assert _close(a, b, atol, rtol), f"{label}[{i}] max|diff|={md:.3e}"


# independent fp32 primitives (division form / F.* - distinct from the oracle) ----
def _rms_n(x, w, eps=N.EPS):
    xf = x.float()
    ms = (xf * xf).sum(-1, keepdim=True) / xf.shape[-1]
    return xf / torch.sqrt(ms + eps) * w.float()


def _ln_n(x, w, b, eps=N.EPS):
    return F.layer_norm(x.float(), (x.shape[-1],), w.float(),
                        None if b is None else b.float(), eps)


def _has_bias(op):
    return N._SPECS[op].get("has_bias", True)


# --------------------------------------------------------------------------- #
# INDEPENDENT forward oracle (returns a tuple of fp32 tensors)
# --------------------------------------------------------------------------- #
def _independent_fwd(op, inp):
    kind = N.KIND[op]
    x = inp[0]
    if kind == "rmsnorm":
        return (_rms_n(x, inp[1]),)
    if kind == "layernorm":
        b = inp[2] if _has_bias(op) else None
        return (_ln_n(x, inp[1], b),)
    if kind == "groupnorm":
        return (F.group_norm(x.float(), N.NUM_GROUPS, inp[1].float(), inp[2].float(), N.EPS),)
    if kind == "instancenorm":
        return (F.instance_norm(x.float(), weight=inp[1].float(), bias=inp[2].float(), eps=N.EPS),)
    if kind == "batchnorm":
        return (F.batch_norm(x.float(), None, None, inp[1].float(), inp[2].float(), True, 0.0, N.EPS),)
    if kind == "l2norm":
        return (F.normalize(x.float(), p=2.0, dim=-1, eps=N.L2_EPS),)
    if kind == "weightnorm":
        v, g = x.float(), inp[1].float()
        n = torch.sqrt(torch.einsum("mn,mn->m", v, v)).unsqueeze(-1)
        return (g * v / n,)
    if kind == "rmsnorm_stats":
        xf = x.float()
        r = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + N.EPS)
        return (_rms_n(x, inp[1]), r.squeeze(-1))
    if kind == "layernorm_stats":
        xf = x.float()
        mean = xf.mean(-1)
        r = torch.rsqrt(xf.var(-1, unbiased=False) + N.EPS)
        return (_ln_n(x, inp[1], inp[2]), mean, r)
    if kind == "groupnorm_stats":
        M, C = x.shape
        xg = x.float().reshape(M, N.NUM_GROUPS, C // N.NUM_GROUPS)
        mean = xg.mean(-1)
        r = torch.rsqrt(xg.var(-1, unbiased=False) + N.EPS)
        y = F.group_norm(x.float(), N.NUM_GROUPS, inp[1].float(), inp[2].float(), N.EPS)
        return (y, mean, r)
    if kind == "batchnorm_stats":
        xf = x.float()
        mean = xf.mean(dim=(0, 2))
        r = torch.rsqrt(xf.var(dim=(0, 2), unbiased=False) + N.EPS)
        y = F.batch_norm(xf, None, None, inp[1].float(), inp[2].float(), True, 0.0, N.EPS)
        return (y, mean, r)
    if kind == "groupnorm_silu":
        return (F.silu(F.group_norm(x.float(), N.NUM_GROUPS, inp[1].float(), inp[2].float(), N.EPS)),)
    if kind == "add_rmsnorm":
        added = x.float() + inp[1].float()
        return (_rms_n(added, inp[2]), added)
    if kind == "add_layernorm":
        added = x.float() + inp[1].float()
        return (_ln_n(added, inp[2], inp[3]), added)
    if kind == "rmsnorm_swiglu":
        normed = _rms_n(x, inp[1])
        h = normed.shape[-1] // 2
        return (F.silu(normed[:, :h]) * normed[:, h:],)
    if kind == "rmsnorm_gated":
        return (_rms_n(x, inp[1]) * F.silu(inp[2].float()),)
    if kind == "qk_rmsnorm":
        return (_rms_n(inp[0], inp[2]), _rms_n(inp[1], inp[3]))
    if kind == "qk_layernorm":
        return (_ln_n(inp[0], inp[2], inp[4]), _ln_n(inp[1], inp[3], inp[5]))
    if kind == "dropout_rmsnorm":
        return (_rms_n(x, inp[1]) * inp[2].float() * N.INV_KEEP,)
    if kind == "dropout_layernorm":
        return (_ln_n(x, inp[1], inp[2]) * inp[3].float() * N.INV_KEEP,)
    raise AssertionError(f"no independent forward oracle for {op!r} ({kind})")


# --------------------------------------------------------------------------- #
# INDEPENDENT backward (hand-derived analytic; l2 uses autograd F.normalize)
# --------------------------------------------------------------------------- #
def _independent_bwd(op, inp):
    kind = N.KIND[op]
    eps = N.EPS
    if kind == "rmsnorm_bwd":
        x, w, dy = (t.float() for t in inp)
        Nd = x.shape[-1]
        r = torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
        g = dy * w
        S = (g * x).sum(-1, keepdim=True)
        dx = r * g - (r ** 3 / Nd) * x * S
        dw = (dy * (x * r)).sum(0)
        return (dx, dw)
    if kind in ("layernorm_bwd", "layernorm_nobias_bwd"):
        x, w, dy = (t.float() for t in inp)
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        r = torch.rsqrt(var + eps)
        xhat = (x - mean) * r
        g = dy * w
        dx = r * (g - g.mean(-1, keepdim=True) - xhat * (g * xhat).mean(-1, keepdim=True))
        dw = (dy * xhat).sum(0)
        if kind == "layernorm_bwd":
            return (dx, dw, dy.sum(0))
        return (dx, dw)
    if kind == "groupnorm_bwd":
        x, w, dy = (t.float() for t in inp)
        M, C = x.shape
        G, Wd = N.NUM_GROUPS, C // N.NUM_GROUPS
        xg = x.reshape(M, G, Wd)
        mean = xg.mean(-1, keepdim=True)
        var = (xg - mean).pow(2).mean(-1, keepdim=True)
        r = torch.rsqrt(var + eps)
        xhat = (xg - mean) * r
        gg = dy.reshape(M, G, Wd) * w.reshape(1, G, Wd)
        dxg = r * (gg - gg.mean(-1, keepdim=True) - xhat * (gg * xhat).mean(-1, keepdim=True))
        dx = dxg.reshape(M, C)
        dw = (dy * xhat.reshape(M, C)).sum(0)
        return (dx, dw, dy.sum(0))
    if kind == "l2norm_bwd":
        x, dy = inp
        xf = x.float().detach().requires_grad_(True)
        F.normalize(xf, p=2.0, dim=-1, eps=N.L2_EPS).backward(dy.float())
        return (xf.grad.detach(),)
    raise AssertionError(f"no independent backward for {op!r}")


def _quant_normed(op, inp):
    kind = N.KIND[op]
    if kind == "rmsnorm_quant":
        return _rms_n(inp[0], inp[1])
    if kind == "layernorm_quant":
        return _ln_n(inp[0], inp[1], inp[2])
    if kind == "add_rmsnorm_quant":
        return _rms_n(inp[0].float() + inp[1].float(), inp[2])
    raise AssertionError(op)


# =========================================================================== #
# ABI surface
# =========================================================================== #
def test_op_count_and_prefix():
    assert isinstance(N.OPS, list)
    assert len(N.OPS) == 45, f"expected 45 ops, got {len(N.OPS)}"
    assert len(set(N.OPS)) == len(N.OPS), "duplicate op names"
    assert all(op.startswith("norm_") for op in N.OPS)


def test_abi_present():
    assert callable(N.make_reference) and callable(N.seed_source) and callable(N.op_dtypes)
    assert set(N.OP_DTYPES) == set(N.OPS)
    assert set(N.SHAPES) == set(N.OPS)


def test_category_coverage():
    kinds = set(N.KIND.values())
    for k in ("rmsnorm", "layernorm", "groupnorm", "instancenorm", "batchnorm",
              "l2norm", "weightnorm", "rmsnorm_bwd", "layernorm_bwd", "groupnorm_bwd",
              "rmsnorm_quant", "layernorm_quant", "add_rmsnorm", "add_layernorm",
              "rmsnorm_swiglu", "qk_rmsnorm", "dropout_rmsnorm"):
        assert k in kinds, f"missing category {k}"
    # fp8 AND int8 quant-out variants both present
    assert any(N.OP_DTYPES[op] == ["fp8"] for op in QUANT_OPS)
    assert any(N.OP_DTYPES[op] == ["int8"] for op in QUANT_OPS)
    # a few fp32-enabled ops, and hidden-size variants for the core norms
    assert any("fp32" in dts for dts in N.OP_DTYPES.values())
    for n in (2048, 4096, 8192, 16384):
        assert f"norm_rmsnorm_h{n}" in N.OPS and f"norm_layernorm_h{n}" in N.OPS


def test_torch_imported_lazily():
    """Registry discovery must be GPU-free: torch imported only inside make_reference."""
    tree = ast.parse(inspect.getsource(N))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("torch") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch")


def test_op_dtypes_consistent():
    for op in N.OPS:
        dts = N.OP_DTYPES[op]
        assert dts and all(d in DTYPE_NAMES for d in dts)
        if N.KIND[op] in _QUANT_KINDS:
            assert dts in (["fp8"], ["int8"])
        else:
            assert dts[:2] == ["bf16", "fp16"]


@pytest.mark.parametrize("op", N.OPS)
def test_shapes_catalog(op):
    sh = N.SHAPES[op]
    assert {"minimal", "primary", "validation"} <= set(sh)
    assert isinstance(sh["validation"], list) and sh["validation"]


@pytest.mark.parametrize("op", N.OPS)
def test_shapes_parse_roundtrip(op):
    ns = N.make_reference(op, N.OP_DTYPES[op][0])
    ps = ns["parse_shape"]
    sh = N.SHAPES[op]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert ps(s) == spec


def test_hidden_pins():
    for op, n in N.HIDDEN.items():
        for spec in [N.SHAPES[op]["minimal"], N.SHAPES[op]["primary"], *N.SHAPES[op]["validation"]]:
            assert spec["N"] == n


def test_groupnorm_shapes_divisible():
    for op in N.OPS:
        if N.KIND[op] in ("groupnorm", "groupnorm_stats", "groupnorm_silu", "groupnorm_bwd"):
            for spec in [N.SHAPES[op]["minimal"], N.SHAPES[op]["primary"], *N.SHAPES[op]["validation"]]:
                assert spec["N"] % N.NUM_GROUPS == 0, f"{op}: N={spec['N']} not divisible by {N.NUM_GROUPS}"


# =========================================================================== #
# namespace / arity / mutates_input / dtype preservation
# =========================================================================== #
@pytest.mark.parametrize("op", N.OPS)
def test_namespace_contract(op):
    dt = N.OP_DTYPES[op][0]
    ns = N.make_reference(op, dt)
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family", "mutates_input"):
        assert k in ns, f"{op} missing ns key {k!r}"
    assert ns["entry_name"] == op
    assert ns["dtype_name"] == dt
    assert ns["family"] == f"breadth_{op}"
    assert isinstance(ns["arity"], int) and ns["arity"] > 0
    assert ns[f"{op}_ref"] is ns["ref_fn"]
    assert ns["mutates_input"] is False              # none of the norm ops are in-place


@pytest.mark.parametrize("op", N.OPS)
def test_arity_matches_get_inputs(op):
    ns = N.make_reference(op, N.OP_DTYPES[op][0])
    inp = ns["get_inputs"](_small(op), device="cpu", seed=0)
    assert len(inp) == ns["arity"]


@pytest.mark.parametrize("op", N.OPS)
def test_get_inputs_dtype(op):
    dt = N.OP_DTYPES[op][0]
    ns = N.make_reference(op, dt)
    inp = ns["get_inputs"](_small(op), device="cpu", seed=0)
    exp = getattr(torch, N.input_dtype(op, dt))
    for t in inp:
        if torch.is_tensor(t) and t.is_floating_point():
            assert t.dtype == exp, f"{op}: input dtype {t.dtype} != {exp}"


# =========================================================================== #
# seeds: compile + define entry, and load as a module (triton binds the kernel)
# =========================================================================== #
@pytest.mark.parametrize("op", N.OPS)
def test_seed_compiles_and_defines_entry(op):
    for dt in N.OP_DTYPES[op]:
        src = N.seed_source(op, dt)
        compile(src, f"<{op}:{dt}>", "exec")
        funcs = {n.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)}
        assert op in funcs, f"seed for {op}/{dt} must define def {op}(...)"


@pytest.mark.parametrize("op", N.OPS)
def test_seed_loads_as_module(op, tmp_path):
    pytest.importorskip("triton")
    dt = N.OP_DTYPES[op][0]
    path = tmp_path / f"seed_{op}.py"
    path.write_text(N.seed_source(op, dt))
    spec = importlib.util.spec_from_file_location(f"norm_seed_{op}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(getattr(mod, op))


# =========================================================================== #
# CORRECTNESS: ref_fn vs INDEPENDENT torch (fp32, tight tol)
# =========================================================================== #
@pytest.mark.parametrize("op", FWD_OPS)
def test_forward_ref_matches_independent(op):
    ns = N.make_reference(op, "fp32")
    inp = ns["get_inputs"](_small(op), device="cpu", seed=0)
    got = ns["ref_fn"](*_clone(inp))
    exp = _independent_fwd(op, inp)
    _assert_tuples_close(got, exp, label=op)


@pytest.mark.parametrize("op", FWD_OPS)
def test_forward_dtype_cast_back(op):
    """The oracle casts its primary output back to the task dtype (bf16 here)."""
    dt = "bf16"
    ns = N.make_reference(op, dt)
    inp = ns["get_inputs"](_small(op), device="cpu", seed=1)
    got = _tup(ns["ref_fn"](*_clone(inp)))
    assert got[0].dtype == torch.bfloat16
    # fp32 reference recomputation within bf16 rounding
    exp = _independent_fwd(op, inp)
    _assert_tuples_close(got[:1], exp[:1], atol=3e-2, rtol=3e-2, label=f"{op}(bf16)")


@pytest.mark.parametrize("op", BWD_OPS)
def test_backward_oracle_matches_analytic(op):
    """The autograd oracle == a hand-derived analytic backward (both fp32)."""
    ns = N.make_reference(op, "fp32")
    inp = ns["get_inputs"](_small(op), device="cpu", seed=0)
    got = ns["ref_fn"](*_clone(inp))
    exp = _independent_bwd(op, inp)
    _assert_tuples_close(got, exp, atol=_BWD_ATOL, rtol=_BWD_RTOL, label=op)


def test_backward_reduction_shapes():
    """dweight/dbias reduce over the token axis -> per-feature vectors."""
    ns = N.make_reference("norm_layernorm_bwd", "fp32")
    inp = ns["get_inputs"]({"M": 8, "N": 20}, device="cpu", seed=0)
    dx, dw, db = ns["ref_fn"](*_clone(inp))
    assert dx.shape == (8, 20) and dw.shape == (20,) and db.shape == (20,)
    # db = sum over tokens of dy  (independent of x)
    assert _close(db, inp[2].float().sum(0), atol=_BWD_ATOL, rtol=_BWD_RTOL)


@pytest.mark.parametrize("op", BWD_OPS)
def test_backward_baseline_matches_oracle(op):
    ns = N.make_reference(op, "fp32")
    inp = ns["get_inputs"](_small(op), device="cpu", seed=2)
    _assert_tuples_close(ns["baseline_fn"](*_clone(inp)), ns["ref_fn"](*_clone(inp)),
                         atol=_BWD_ATOL, rtol=_BWD_RTOL, label=op)


# =========================================================================== #
# CORRECTNESS: norm + output quantization (scale exact + reconstruction SNR)
# =========================================================================== #
@pytest.mark.parametrize("op", QUANT_OPS)
def test_quant_ref(op):
    dt = N.OP_DTYPES[op][0]                          # "fp8" | "int8"
    ns = N.make_reference(op, dt)
    inp = ns["get_inputs"](_small(op), device="cpu", seed=0)
    out = _tup(ns["ref_fn"](*_clone(inp)))
    q, scale = out[0], out[1]
    qmax = N.FP8_MAX if dt == "fp8" else N.INT8_MAX

    assert q.dtype == (torch.float8_e4m3fn if dt == "fp8" else torch.int8)
    assert scale.dtype == torch.float32
    normed = _quant_normed(op, inp)                  # fp32 [M, Nd]
    assert q.shape == normed.shape and scale.shape == normed.shape[:1]

    # per-token scale recomputed independently (the hard, normalization-dependent bit)
    amax = normed.abs().amax(-1)
    scale_ind = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    assert _close(scale, scale_ind, atol=1e-4, rtol=1e-3), "quant scale mismatch"

    # dequant reconstructs the normed row, and codes stay in range
    deq = q.float() * scale.unsqueeze(-1)
    assert q.float().abs().max().item() <= qmax + 1e-3
    snr = _snr_db(deq, normed)
    assert snr > (22.0 if dt == "fp8" else 34.0), f"{op}: reconstruction SNR {snr:.1f} dB"

    if N.KIND[op] == "add_rmsnorm_quant":            # third output = new residual
        added_ind = inp[0].float() + inp[1].float()
        assert _close(out[2], added_ind, atol=3e-2, rtol=3e-2)


# =========================================================================== #
# baseline_fn runs and structurally matches the oracle (forward/fused)
# =========================================================================== #
@pytest.mark.parametrize("op", FWD_OPS)
def test_baseline_matches_ref(op):
    ns = N.make_reference(op, "fp32")
    inp = ns["get_inputs"](_small(op), device="cpu", seed=3)
    _assert_tuples_close(ns["baseline_fn"](*_clone(inp)), ns["ref_fn"](*_clone(inp)),
                         atol=1e-3, rtol=1e-3, label=f"baseline:{op}")


# =========================================================================== #
# fused semantics: dropout determinism + residual passthrough
# =========================================================================== #
@pytest.mark.parametrize("op", ["norm_dropout_rmsnorm", "norm_dropout_layernorm"])
def test_dropout_deterministic_and_active(op):
    ns = N.make_reference(op, "fp32")
    i1 = ns["get_inputs"](_small(op), device="cpu", seed=7)
    i2 = ns["get_inputs"](_small(op), device="cpu", seed=7)
    mask1 = i1[-1]
    assert torch.equal(mask1, i2[-1]), "mask must be deterministic for a fixed seed"
    assert set(mask1.unique().tolist()) <= {0.0, 1.0}
    assert (mask1 == 0).any(), "dropout mask should zero some elements"
    # y is exactly zero wherever the mask dropped, nonzero-scaled where it kept
    y = ns["ref_fn"](*_clone(i1))
    assert torch.equal((y == 0), (mask1 == 0)) or (y[mask1 == 0].abs().max() == 0)


@pytest.mark.parametrize("op", ["norm_add_rmsnorm", "norm_add_layernorm"])
def test_add_norm_returns_new_residual(op):
    ns = N.make_reference(op, "fp32")
    inp = ns["get_inputs"](_small(op), device="cpu", seed=0)
    y, added = ns["ref_fn"](*_clone(inp))
    assert _close(added, inp[0].float() + inp[1].float())
    assert y.shape == added.shape


def test_qk_norm_per_head():
    """QK-norm normalizes over the head-dim D independently per (b, s, h)."""
    ns = N.make_reference("norm_qk_rmsnorm", "fp32")
    inp = ns["get_inputs"]({"B": 1, "S": 3, "H": 2, "D": 16}, device="cpu", seed=0)
    qn, kn = ns["ref_fn"](*_clone(inp))
    assert qn.shape == inp[0].shape and kn.shape == inp[1].shape
    # each head row has unit-RMS up to the weight (check with weight = 1)
    q = inp[0].float()
    rms_q = _rms_n(q, torch.ones(16))
    assert _close(qn, rms_q * inp[2].float())
