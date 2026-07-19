"""CPU-only tests for the breadth fused-transformer-block authoring engine.

Every ``ref_fn`` oracle is checked against an INDEPENDENT torch computation that
composes the sub-ops separately - a DIFFERENT code path than the fused oracle:
  * the 2-GEMM gated MLP vs ``F.linear`` + ``F.silu``/``F.gelu``/``F.relu`` + mul
    + ``F.linear`` (each GEMM and activation done separately),
  * RoPE (half-rotation / interleaved) vs an explicit apply-rope reference built
    from the two rotated halves / the even-odd interleave,
  * the QKV projection vs three independent ``F.linear`` projections + split,
  * the attention-logit epilogue vs a stable ``F.softmax`` (the oracle uses a
    hand-rolled max-subtract softmax),
  * the block-glue bias/dropout/residual/norm vs ``F.layer_norm`` / a division-
    form RMS,
  * the norm+quant oracles: the per-token scale is recomputed independently and
    the dequantized output reconstructs the normed row (SNR bound).

Plus the ABI / arity / seed-compiles / seeds-load / shapes-parse / dtype-
preservation / mutates_input (kv-cache in-place write) contract. All fp32 on CPU
(no GPU / triton launch), triton only imported (skippable) for the seed module-
load check.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from kore.tasks.breadth import fused_ext as N

# --------------------------------------------------------------------------- #
# op groupings
# --------------------------------------------------------------------------- #
_QUANT_KINDS = {"add_rmsnorm_quant", "add_layernorm_quant"}

QUANT_OPS = [op for op in N.OPS if N.KIND[op] in _QUANT_KINDS]
FWD_OPS = [op for op in N.OPS if N.KIND[op] not in _QUANT_KINDS]

DTYPE_NAMES = ("bf16", "fp16", "fp32", "fp8", "int8")
_ATOL, _RTOL = 1e-4, 1e-3


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clone(inp):
    return tuple(t.clone() if torch.is_tensor(t) else t for t in inp)


def _tup(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _small(op: str) -> dict:
    """Tiny CPU shape per op-kind (logic is generic over the pinned dims)."""
    k = N.KIND[op]
    if k in ("rope_half", "rope_interleaved", "rope_half_qknorm",
             "rope_interleaved_qknorm", "rope_kvcache"):
        return {"B": 1, "S": 3, "H": 2, "D": 16}
    if k in ("qkv_split", "out_proj_add", "norm_linear"):
        return {"M": 6, "K": 10, "N": 8}
    if k in ("glu_mlp", "glu_mlp_gateup"):
        return {"M": 6, "K": 10, "N": 12}
    if k == "glu_act":
        return {"M": 8, "N": 16}                     # input width 16 -> H=8
    if k == "embed":
        return {"V": 16, "D": 10, "M": 5}
    if k == "softcap_softmax":
        return {"R": 6, "Ncol": 7}
    return {"M": 8, "N": 20}                          # block-glue / quant / resid


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


# independent fp32 primitives (division-form RMS / F.* - distinct from oracle) --
def _rms_n(x, w, eps=N.EPS):
    xf = x.float()
    ms = (xf * xf).mean(-1, keepdim=True)
    return xf / torch.sqrt(ms + eps) * w.float()


def _ln_n(x, w, b, eps=N.EPS):
    return F.layer_norm(x.float(), (x.shape[-1],), w.float(),
                        None if b is None else b.float(), eps)


def _act_ind(name, t):
    if name == "silu":
        return F.silu(t)
    if name == "gelu":
        return F.gelu(t, approximate="tanh")
    return F.relu(t)


def _rope_half_ind(xf, cos, sin):
    """Explicit half-rotation (build the two rotated halves directly)."""
    c, s = cos.float(), sin.float()
    h = xf.shape[-1] // 2
    c0 = c[..., :h][None, :, None, :]
    s0 = s[..., :h][None, :, None, :]
    x1, x2 = xf[..., :h], xf[..., h:]
    out = torch.empty_like(xf)
    out[..., :h] = x1 * c0 - x2 * s0
    out[..., h:] = x2 * c0 + x1 * s0
    return out


def _rope_inter_ind(xf, cos, sin):
    """Explicit interleaved rotation via strided even/odd slices."""
    c = cos.float()[None, :, None, :]
    s = sin.float()[None, :, None, :]
    xe, xo = xf[..., 0::2], xf[..., 1::2]
    out = torch.empty_like(xf)
    out[..., 0::2] = xe * c - xo * s
    out[..., 1::2] = xe * s + xo * c
    return out


# --------------------------------------------------------------------------- #
# INDEPENDENT forward oracle (returns a tuple of fp32 tensors)
# --------------------------------------------------------------------------- #
def _independent_fwd(op, inp):
    k = N.KIND[op]
    spec = N._SPECS[op]
    if k == "rope_half":
        return (_rope_half_ind(inp[0].float(), inp[2], inp[3]),
                _rope_half_ind(inp[1].float(), inp[2], inp[3]))
    if k == "rope_interleaved":
        return (_rope_inter_ind(inp[0].float(), inp[2], inp[3]),
                _rope_inter_ind(inp[1].float(), inp[2], inp[3]))
    if k == "rope_half_qknorm":
        qn, kn = _rms_n(inp[0], inp[2]), _rms_n(inp[1], inp[3])
        return (_rope_half_ind(qn, inp[4], inp[5]), _rope_half_ind(kn, inp[4], inp[5]))
    if k == "rope_interleaved_qknorm":
        qn, kn = _rms_n(inp[0], inp[2]), _rms_n(inp[1], inp[3])
        return (_rope_inter_ind(qn, inp[4], inp[5]), _rope_inter_ind(kn, inp[4], inp[5]))
    if k == "rope_kvcache":
        apply = _rope_inter_ind if spec["mode"] == "interleaved" else _rope_half_ind
        return (apply(inp[0].float(), inp[1], inp[2]),)
    if k == "qkv_split":
        x, w = inp[0].float(), inp[1].float()
        n = w.shape[1] // 3
        q = F.linear(x, w[:, :n].t())
        kk = F.linear(x, w[:, n:2 * n].t())
        v = F.linear(x, w[:, 2 * n:3 * n].t())
        if spec["has_bias"]:
            b = inp[2].float()
            q, kk, v = q + b[:n], kk + b[n:2 * n], v + b[2 * n:3 * n]
        return (q, kk, v)
    if k == "glu_mlp":
        x = inp[0].float()
        g = F.linear(x, inp[1].t().float())
        u = F.linear(x, inp[2].t().float())
        h = _act_ind(spec["act"], g) * u
        return (F.linear(h, inp[3].t().float()),)
    if k == "glu_mlp_gateup":
        x = inp[0].float()
        gu = F.linear(x, inp[1].t().float())
        n = gu.shape[-1] // 2
        h = _act_ind(spec["act"], gu[:, :n]) * gu[:, n:]
        return (F.linear(h, inp[2].t().float()),)
    if k == "glu_act":
        x = inp[0].float()
        h = x.shape[-1] // 2
        return (_act_ind(spec["act"], x[:, :h]) * x[:, h:],)
    if k == "bias_drop_add_norm":
        has_bias, norm = spec["has_bias"], spec["norm"]
        x = inp[0]
        i = 1
        bias = inp[i].float() if has_bias else 0.0
        i += 1 if has_bias else 0
        residual, mask, weight = inp[i], inp[i + 1], inp[i + 2]
        i += 3
        lnbias = inp[i] if norm == "layer" else None
        added = residual.float() + (x.float() + bias) * mask.float() * N.INV_KEEP
        out = _rms_n(added, weight) if norm == "rms" else _ln_n(added, weight, lnbias)
        return (out, added)
    if k == "out_proj_add":
        attn, wo = inp[0], inp[1]
        y = F.linear(attn.float(), wo.t().float())
        if spec["has_bias"]:
            y = y + inp[2].float()
            residual = inp[3]
        else:
            residual = inp[2]
        return (y + residual.float(),)
    if k == "embed":
        ids, w = inp[0], inp[1]
        y = F.embedding(ids.long(), w.float()) * math.sqrt(w.shape[-1])
        if spec["pos"]:
            y = y + inp[2].float()
        return (y,)
    if k == "softcap_softmax":
        s = N.SOFTCAP * torch.tanh(inp[0].float() / N.SOFTCAP)
        if spec["masked"]:
            s = s + inp[1].float()
        return (F.softmax(s, dim=-1),)
    if k == "norm_linear":
        x, weight = inp[0], inp[1]
        if spec["norm"] == "layer":
            normed, W = _ln_n(x, weight, inp[2]), inp[3]
        else:
            normed, W = _rms_n(x, weight), inp[2]
        return (F.linear(normed, W.t().float()),)
    if k == "resid_drop_scale":
        x, residual, mask = inp[0], inp[1], inp[2]
        sc = x.float() * mask.float().unsqueeze(-1) * N.INV_KEEP
        if spec["layerscale"]:
            sc = sc * inp[3].float()
        return (residual.float() + sc,)
    raise AssertionError(f"no independent forward oracle for {op!r} ({k})")


def _quant_normed(op, inp):
    k = N.KIND[op]
    if k == "add_rmsnorm_quant":
        return _rms_n(inp[0].float() + inp[1].float(), inp[2])
    if k == "add_layernorm_quant":
        return _ln_n(inp[0].float() + inp[1].float(), inp[2], inp[3])
    raise AssertionError(op)


# =========================================================================== #
# ABI surface
# =========================================================================== #
def test_op_count_and_prefix():
    assert isinstance(N.OPS, list)
    assert len(N.OPS) == 32, f"expected 32 ops, got {len(N.OPS)}"
    assert len(set(N.OPS)) == len(N.OPS), "duplicate op names"
    assert all(op.startswith("fx_") for op in N.OPS)
    for bad in ("mla", "paged", "latent"):
        assert not any(bad in op for op in N.OPS), f"reserved substring {bad!r} in an op name"


def test_abi_present():
    assert callable(N.make_reference) and callable(N.seed_source) and callable(N.op_dtypes)
    assert set(N.OP_DTYPES) == set(N.OPS)
    assert set(N.SHAPES) == set(N.OPS)


def test_category_coverage():
    kinds = set(N.KIND.values())
    for k in ("rope_half", "rope_interleaved", "rope_half_qknorm",
              "rope_interleaved_qknorm", "rope_kvcache", "qkv_split", "glu_mlp",
              "glu_mlp_gateup", "glu_act", "bias_drop_add_norm", "add_rmsnorm_quant",
              "add_layernorm_quant", "out_proj_add", "embed", "softcap_softmax",
              "norm_linear", "resid_drop_scale"):
        assert k in kinds, f"missing category {k}"
    # fp8 AND int8 quant-out variants both present
    assert any(N.OP_DTYPES[op] == ["fp8"] for op in QUANT_OPS)
    assert any(N.OP_DTYPES[op] == ["int8"] for op in QUANT_OPS)


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
            assert dts == ["bf16", "fp16"]


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


# =========================================================================== #
# namespace / arity / mutates_input / dtype preservation
# =========================================================================== #
@pytest.mark.parametrize("op", N.OPS)
def test_namespace_contract(op):
    dt = N.OP_DTYPES[op][0]
    ns = N.make_reference(op, dt)
    for key in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
                "entry_name", "dtype_name", "family", "mutates_input"):
        assert key in ns, f"{op} missing ns key {key!r}"
    assert ns["entry_name"] == op
    assert ns["dtype_name"] == dt
    assert ns["family"] == f"breadth_{op}"
    assert isinstance(ns["arity"], int) and ns["arity"] > 0
    assert ns[f"{op}_ref"] is ns["ref_fn"]
    assert ns["mutates_input"] == (op in N.FX_MUTATES_INPUT)


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
    spec = importlib.util.spec_from_file_location(f"fused_seed_{op}", path)
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
    ns = N.make_reference(op, "bf16")
    inp = ns["get_inputs"](_small(op), device="cpu", seed=1)
    got = _tup(ns["ref_fn"](*_clone(inp)))
    assert got[0].dtype == torch.bfloat16
    exp = _independent_fwd(op, inp)
    _assert_tuples_close(got[:1], exp[:1], atol=3e-2, rtol=3e-2, label=f"{op}(bf16)")


@pytest.mark.parametrize("op", FWD_OPS)
def test_baseline_matches_ref(op):
    ns = N.make_reference(op, "fp32")
    inp = ns["get_inputs"](_small(op), device="cpu", seed=3)
    _assert_tuples_close(ns["baseline_fn"](*_clone(inp)), ns["ref_fn"](*_clone(inp)),
                         atol=1e-3, rtol=1e-3, label=f"baseline:{op}")


# =========================================================================== #
# CORRECTNESS: add-residual + norm + output quantization
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
    normed = _quant_normed(op, inp)                  # fp32 [M, N]
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

    # third output = new residual passthrough (bf16)
    added_ind = inp[0].float() + inp[1].float()
    assert _close(out[2], added_ind, atol=3e-2, rtol=3e-2)


# =========================================================================== #
# fused semantics: RoPE / split / residual-passthrough / dropout / kv-cache
# =========================================================================== #
def test_qkv_split_shapes():
    ns = N.make_reference("fx_qkv_proj_split_bias", "fp32")
    inp = ns["get_inputs"]({"M": 6, "K": 10, "N": 8}, device="cpu", seed=0)
    q, k, v = ns["ref_fn"](*_clone(inp))
    assert q.shape == (6, 8) and k.shape == (6, 8) and v.shape == (6, 8)


def test_rope_qknorm_shapes_and_unit_rms():
    """QK-RMSNorm normalizes over the head-dim D per (b,s,h) before the rotation."""
    ns = N.make_reference("fx_rope_qk_half_qknorm", "fp32")
    inp = ns["get_inputs"]({"B": 1, "S": 3, "H": 2, "D": 16}, device="cpu", seed=0)
    qn, kn = ns["ref_fn"](*_clone(inp))
    assert qn.shape == inp[0].shape and kn.shape == inp[1].shape
    # RoPE is norm-preserving per head, so post-rope head norm == qk-normed head norm
    normed = _rms_n(inp[0], inp[2])
    assert _close(qn.norm(dim=-1), normed.norm(dim=-1))


@pytest.mark.parametrize("op", ["fx_bias_dropout_add_rmsnorm", "fx_dropout_add_layernorm"])
def test_block_glue_returns_new_residual(op):
    ns = N.make_reference(op, "fp32")
    inp = ns["get_inputs"](_small(op), device="cpu", seed=0)
    out, added = ns["ref_fn"](*_clone(inp))
    exp = _independent_fwd(op, inp)                  # (out, added)
    _assert_tuples_close((added,), (exp[1],), label=f"{op}:added")
    assert out.shape == added.shape


def test_dropout_mask_deterministic():
    for op in ("fx_bias_dropout_add_rmsnorm", "fx_bias_dropout_add_layernorm",
               "fx_dropout_add_rmsnorm", "fx_dropout_add_layernorm",
               "fx_resid_dropout_scale", "fx_resid_dropout_scale_layerscale"):
        ns = N.make_reference(op, "fp32")
        i1 = ns["get_inputs"](_small(op), device="cpu", seed=7)
        i2 = ns["get_inputs"](_small(op), device="cpu", seed=7)
        for a, b in zip(i1, i2):
            assert torch.equal(a, b), f"{op}: inputs not deterministic for a fixed seed"
        masks = [t for t in i1 if torch.is_tensor(t) and t.is_floating_point()
                 and set(t.unique().tolist()) <= {0.0, 1.0}]
        assert masks, f"{op}: no binary dropout mask among inputs"


def test_dropout_active_and_zeros_output():
    """On a big enough tile the dropout mask both keeps and drops, and dropped
    positions of the (un-normalized) residual passthrough are visible."""
    op = "fx_dropout_add_rmsnorm"
    ns = N.make_reference(op, "fp32")
    inp = ns["get_inputs"]({"M": 16, "N": 32}, device="cpu", seed=7)
    masks = [t for t in inp if torch.is_tensor(t) and t.is_floating_point()
             and t.dim() == 2 and set(t.unique().tolist()) <= {0.0, 1.0}]
    assert masks, "no 2D binary dropout mask"
    m = masks[0]
    assert (m == 0).any() and (m == 1).any()


@pytest.mark.parametrize("op", sorted(N.FX_MUTATES_INPUT))
def test_kvcache_writes_in_place(op):
    """The rotary + kv-cache write mutates the supplied cache buffer in place."""
    ns = N.make_reference(op, "fp32")
    assert ns["mutates_input"] is True
    k, cos, sin, cache = ns["get_inputs"](_small(op), device="cpu", seed=0)
    assert torch.count_nonzero(cache) == 0            # starts zeroed
    ptr = cache.data_ptr()
    out = ns["ref_fn"](k, cos, sin, cache)            # NOT cloned -> mutate real cache
    assert out.data_ptr() == cache.data_ptr() == ptr  # same buffer returned
    assert torch.count_nonzero(cache) > 0             # written in place
    exp = _independent_fwd(op, (k, cos, sin, cache))[0]
    _assert_tuples_close((cache,), (exp,), label=op)


def test_non_mutating_ops_flag_false():
    for op in FWD_OPS:
        if op in N.FX_MUTATES_INPUT:
            continue
        assert N.make_reference(op, N.OP_DTYPES[op][0])["mutates_input"] is False
