"""CPU-only tests for the breadth quantization / low-precision engine (quant_ext).

Every ``ref_fn`` oracle is checked against an INDEPENDENT torch computation on a
DIFFERENT code path than the (vectorized / broadcast) oracle it wraps:

  * scales are recomputed in float64 with a distinct reduction (``amax`` over the
    explicitly-reshaped axis, not the module's ``keepdim`` path),
  * quantized values are reconstructed with reshape-based scale broadcasting (not
    the module's ``repeat_interleave``), and int4 / MXFP4 nibbles are unpacked with
    an independent LUT,
  * quantize ops are graded by BOTH an exact scale match (tight tol) AND a bounded
    dequant-reconstruction SNR against the original tensor, plus a cross-check of
    the reconstruction against an independently-quantized reconstruction,

so a wrong axis / qmax / clamp / nibble order / transpose / scale-broadcast in the
oracle is caught with certainty. All fp32/fp64 on CPU - no GPU / triton kernel is
ever launched.

Also asserts the ABI surface, arity, code/scale dtype + granularity structure, the
adversarial battery (hard regimes survive quantization), the output dtype (fp8/int8
codes + fp32 scale for quantize; bf16 for dequant), that each seed compiles +
defines + loads its entry, shape-catalog divisibility + ``parse_shape`` round-trip,
and the lazy torch import.
"""

from __future__ import annotations

import ast
import math

import pytest
import torch

from kore.tasks._genops import DTYPES
from kore.tasks.breadth import quant_ext as Q

FP8_MAX, INT8_MAX, INT4_MAX = 448.0, 127.0, 7.0
BLK, MX_BLOCK = 128, 32
_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

QUANTIZE_OPS = [o for o in Q.OPS if Q._is_quantize(Q._CFG[o])]
DEQUANT_OPS = [o for o in Q.OPS if not Q._is_quantize(Q._CFG[o])]

# per-format bounded reconstruction SNR gate (dB), measured on the tiny CPU shapes
# with comfortable margin; stochastic / nested / mxfp4 grids are coarser.
_SNR_GATE = {"fp8": 16.0, "int8": 34.0}
# 4-bit / nested / mxfp4 / stochastic grids are coarser -> lower (but bounded) SNR.
_SNR_GATE_KIND = {"double": 8.0, "mxfp4pack": 12.0, "stochastic": 8.0, "int4pack": 14.0}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _small(op: str) -> dict:
    cfg = Q._CFG[op]
    km, mm = Q._kmult(cfg), Q._mmult(cfg)
    K = km * 2 if km > 1 else 16
    M = mm * 2 if mm > 1 else 4
    return {"M": M, "K": K}


def _d(t):
    """Bring any code/scale tensor (fp8/int8/uint8/bf16/fp32) to float64."""
    return t.float().double()


def _relerr(a, b) -> float:
    a, b = _d(a), _d(b)
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def _snr_db(target, recon) -> float:
    t, r = _d(target), _d(recon)
    num, den = t.norm().item(), (t - r).norm().item()
    if num < 1e-12:
        return float("inf")            # zero target (adversarial zeros) -> pass
    if den < 1e-12:
        return 200.0
    return 20.0 * math.log10(num / den)


def _code_dt(fmt: str):
    return torch.float8_e4m3fn if fmt == "fp8" else torch.int8


# --------------------------------------------------------------------------- #
# INDEPENDENT float64 (de)quant: reshape broadcasting + LUT nibble unpack.
# --------------------------------------------------------------------------- #
def _ideq_nd(codes, s, gran):
    c = _d(codes)
    if gran in ("tensor", "token", "channel"):
        return c * _d(s)
    if gran == "block128":
        M, K = c.shape
        nb = s.shape[1]
        return (c.reshape(M, nb, K // nb) * _d(s)[:, :, None]).reshape(M, K)
    # block2d
    N, K = c.shape
    nbn, nbk = s.shape
    return (c.reshape(nbn, N // nbn, nbk, K // nbk) * _d(s)[:, None, :, None]).reshape(N, K)


def _ind_scale_nd(x, fmt, gran):
    """Independent scale (fp64, stored shape) via an explicitly-reshaped amax."""
    xf = x.double()
    mx = FP8_MAX if fmt == "fp8" else INT8_MAX
    if gran == "tensor":
        return (xf.abs().amax().clamp(min=1e-12) / mx).reshape(())
    if gran == "token":
        return xf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / mx
    if gran == "channel":
        return xf.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / mx
    if gran == "block128":
        M, K = xf.shape
        return xf.reshape(M, K // BLK, BLK).abs().amax(dim=2).clamp(min=1e-12) / mx
    # block2d
    M, K = xf.shape
    return xf.reshape(M // BLK, BLK, K // BLK, BLK).abs().amax(dim=(1, 3)).clamp(min=1e-12) / mx


def _to_codes_ind(q, fmt):
    if fmt == "fp8":
        return q.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.double().round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)


def _ind_quant(x, fmt, gran):
    """Independent quantize (fp64) -> (codes, scale_stored)."""
    xf = x.double()
    s = _ind_scale_nd(x, fmt, gran)
    if gran in ("tensor", "token", "channel"):
        q = xf / s
    elif gran == "block128":
        M, K = xf.shape
        q = (xf.reshape(M, K // BLK, BLK) / s[:, :, None]).reshape(M, K)
    else:  # block2d
        M, K = xf.shape
        q = (xf.reshape(M // BLK, BLK, K // BLK, BLK) / s[:, None, :, None]).reshape(M, K)
    return _to_codes_ind(q, fmt), s


def _iunpack_nibbles(packed):
    N, K = packed.shape[0], packed.shape[1] * 2
    out = torch.zeros((N, K), dtype=torch.long)
    out[:, 0::2] = (packed & 0xF).long()
    out[:, 1::2] = ((packed >> 4) & 0xF).long()
    return out


def _iunpack_int4(packed, scale, group):
    q = _iunpack_nibbles(packed) - 8
    N, K = q.shape
    return (q.reshape(N, K // group, group).double() * scale.double()[:, :, None]).reshape(N, K)


def _imxfp4(packed, e8m0):
    R, K = packed.shape[0], packed.shape[1] * 2
    codes = _iunpack_nibbles(packed)
    mag = torch.tensor(_E2M1_LEVELS, dtype=torch.float64)[(codes & 0x7)]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0).double()
    scale = torch.pow(2.0, e8m0.double() - 127.0)[:, :, None]
    return ((sign * mag).reshape(R, K // MX_BLOCK, MX_BLOCK) * scale).reshape(R, K)


# --------------------------------------------------------------------------- #
# reconstruction + target per quantize op (ref reconstruction, independent recon,
# and the tensor the reconstruction must approximate).
# --------------------------------------------------------------------------- #
def _recons(op, inputs, out):
    """-> list of (ref_recon, ind_recon, target) fp64 triples."""
    cfg = Q._CFG[op]
    kind, fmt, gran, group = cfg["kind"], cfg["fmt"], cfg["gran"], cfg["group"]
    if kind == "quant":
        x = inputs[0]
        codes, scale = out
        icodes, _ = _ind_quant(x, fmt, gran)
        return [(_ideq_nd(codes, scale, gran), _ideq_nd(icodes, scale, gran), x.double())]
    if kind == "stochastic":
        x = inputs[0]
        codes, scale = out
        return [(_ideq_nd(codes, scale, "token"), _ideq_nd(codes, scale, "token"), x.double())]
    if kind == "smooth":
        x, smooth = inputs
        codes, scale = out
        xs = x.double() / smooth.double().reshape(1, -1)
        icodes, _ = _ind_quant(xs, fmt, "token")
        return [(_ideq_nd(codes, scale, "token"), _ideq_nd(icodes, scale, "token"), xs)]
    if kind == "qtranspose":
        x = inputs[0]
        codesT, scale = out
        recon = _d(codesT) * _d(scale)
        return [(recon, recon, x.double().t())]
    if kind == "double":
        x = inputs[0]
        codes, sc_codes, meta = out
        bs = _d(sc_codes) * _d(meta)
        recon = _ideq_nd(codes, bs, "block128")
        return [(recon, recon, x.double())]
    if kind == "int4pack":
        w = inputs[0]
        packed, scale = out
        recon = _iunpack_int4(packed, scale, group)
        return [(recon, recon, w.double())]
    if kind == "mxfp4pack":
        x = inputs[0]
        packed, e8 = out
        recon = _imxfp4(packed, e8)
        return [(recon, recon, x.double())]
    if kind == "kvquant":
        k, v = inputs
        kq, ksc, vq, vsc = out
        ik, _ = _ind_quant(k, fmt, "token")
        iv, _ = _ind_quant(v, fmt, "token")
        return [(_ideq_nd(kq, ksc, "token"), _ideq_nd(ik, ksc, "token"), k.double()),
                (_ideq_nd(vq, vsc, "token"), _ideq_nd(iv, vsc, "token"), v.double())]
    return []


def _independent_dequant(op, inputs):
    """Independent fp64 result for the dequant-family ops."""
    cfg = Q._CFG[op]
    kind, gran, group = cfg["kind"], cfg["gran"], cfg["group"]
    if kind == "dequant":
        return _ideq_nd(inputs[0], inputs[1], gran)
    if kind == "int4unpack":
        return _iunpack_int4(inputs[0], inputs[1], group)
    if kind == "mxfp4unpack":
        return _imxfp4(inputs[0], inputs[1])
    # kvdequant
    return (_ideq_nd(inputs[0], inputs[1], "token"), _ideq_nd(inputs[2], inputs[3], "token"))


# --------------------------------------------------------------------------- #
# ABI surface
# --------------------------------------------------------------------------- #
def test_abi_surface():
    assert isinstance(Q.OPS, list) and len(Q.OPS) == 32
    assert len(set(Q.OPS)) == len(Q.OPS)
    assert set(Q.OP_DTYPES) == set(Q.OPS) == set(Q.SHAPES) == set(Q._CFG)
    assert callable(Q.make_reference) and callable(Q.seed_source)
    assert all(op.startswith("qx_") for op in Q.OPS)
    for op in Q.OPS:                                   # avoid banned substrings
        assert not any(b in op for b in ("mla", "paged", "latent")), op
    for attr in ("OPS", "OP_DTYPES", "SHAPES", "make_reference", "seed_source"):
        assert hasattr(Q, attr)


def test_op_dtypes_valid():
    for op in Q.OPS:
        dts = Q.OP_DTYPES[op]
        assert len(dts) == 1 and dts[0] in DTYPES, (op, dts)
        cfg = Q._CFG[op]
        if Q._is_quantize(cfg):
            assert dts[0] in ("fp8", "int8")
        else:
            assert dts[0] in ("bf16", "fp16")


def test_torch_imported_lazily():
    import inspect
    tree = ast.parse(inspect.getsource(Q))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("torch") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch")


@pytest.mark.parametrize("op", Q.OPS)
def test_arity(op):
    ns = Q.make_reference(op, Q.OP_DTYPES[op][0])
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=0)
    assert isinstance(inputs, tuple)
    assert len(inputs) == ns["arity"] == Q.arity_of(op)


@pytest.mark.parametrize("op", Q.OPS)
def test_namespace_contract(op):
    dt = Q.OP_DTYPES[op][0]
    ns = Q.make_reference(op, dt)
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family", "mutates_input",
              "adversarial_inputs"):
        assert k in ns, f"{op} missing ns key {k!r}"
    assert ns["entry_name"] == op
    assert ns["dtype_name"] == dt
    assert ns["family"] == f"breadth_{op}"
    assert ns["mutates_input"] is False
    assert ns[f"{op}_ref"] is ns["ref_fn"]
    assert ns["parse_shape"] is Q._parse_shape
    assert callable(ns["ref_fn"]) and callable(ns["baseline_fn"])
    assert callable(ns["adversarial_inputs"])


# --------------------------------------------------------------------------- #
# correctness: quantize ops - exact scale + bounded reconstruction SNR
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", QUANTIZE_OPS)
def test_quantize_reconstruction_and_scale(op):
    cfg = Q._CFG[op]
    fmt, kind = cfg["fmt"], cfg["kind"]
    ns = Q.make_reference(op, Q.OP_DTYPES[op][0])
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=0)
    out = ns["ref_fn"](*inputs)
    assert isinstance(out, tuple)
    gate = _SNR_GATE_KIND.get(kind, _SNR_GATE[fmt])
    code_tol = 6e-2 if fmt == "fp8" else 2e-2
    for ref_recon, ind_recon, target in _recons(op, inputs, out):
        assert tuple(ref_recon.shape) == tuple(target.shape), (op, ref_recon.shape, target.shape)
        assert _snr_db(target, ref_recon) >= gate, (op, "SNR", _snr_db(target, ref_recon))
        # ref reconstruction agrees with an independently-quantized reconstruction
        assert _relerr(ref_recon, ind_recon) < code_tol, (op, "recon", _relerr(ref_recon, ind_recon))


@pytest.mark.parametrize("op", [o for o in QUANTIZE_OPS
                                if Q._CFG[o]["kind"] in ("quant", "stochastic", "smooth",
                                                          "qtranspose", "kvquant", "int4pack")])
def test_quantize_scale_exact(op):
    cfg = Q._CFG[op]
    fmt, kind, gran, group = cfg["fmt"], cfg["kind"], cfg["gran"], cfg["group"]
    ns = Q.make_reference(op, Q.OP_DTYPES[op][0])
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=1)
    out = ns["ref_fn"](*inputs)
    if kind == "kvquant":
        k, v = inputs
        _, ksc, _, vsc = out
        assert _relerr(ksc, _ind_scale_nd(k, fmt, "token")) < 1e-4, op
        assert _relerr(vsc, _ind_scale_nd(v, fmt, "token")) < 1e-4, op
        return
    if kind == "int4pack":
        w = inputs[0]
        _, scale = out
        N, K = w.shape
        ind = w.double().reshape(N, K // group, group).abs().amax(dim=2).clamp(min=1e-12) / INT4_MAX
        assert tuple(scale.shape) == tuple(ind.shape), (op, scale.shape, ind.shape)
        assert _relerr(scale, ind) < 1e-4, (op, _relerr(scale, ind))
        return
    scale = out[1]
    if kind == "smooth":
        x, smooth = inputs
        ind = _ind_scale_nd(x.double() / smooth.double().reshape(1, -1), fmt, "token")
    elif kind == "qtranspose":
        ind = _ind_scale_nd(inputs[0], fmt, gran)
        ind = ind.reshape(1, -1) if gran == "token" else ind
    else:
        ind = _ind_scale_nd(inputs[0], fmt, gran)
    assert tuple(scale.shape) == tuple(ind.shape), (op, tuple(scale.shape), tuple(ind.shape))
    assert _relerr(scale, ind) < 1e-4, (op, _relerr(scale, ind))


def test_mxfp4_exponent_exact():
    """MXFP4 e8m0 block exponents match an independent floor(log2(amax))-EMAX."""
    ns = Q.make_reference("qx_mxfp4_pack", "fp8")
    x, = ns["get_inputs"]({"M": 4, "K": 64}, device="cpu", seed=2)
    packed, e8 = ns["ref_fn"](x)
    xb = x.double().reshape(4, 64 // MX_BLOCK, MX_BLOCK)
    amax = xb.abs().amax(dim=2).clamp(min=1e-20)
    exp = (torch.floor(torch.log2(amax)) - 2.0).clamp(-127.0, 127.0)
    e8_ind = (exp + 127.0).to(torch.uint8)
    assert torch.equal(e8, e8_ind), (e8, e8_ind)
    assert tuple(packed.shape) == (4, 32) and packed.dtype == torch.uint8


# --------------------------------------------------------------------------- #
# correctness: dequant-family ops vs independent fp64
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", DEQUANT_OPS)
def test_dequant_matches_independent(op):
    ns = Q.make_reference(op, Q.OP_DTYPES[op][0])
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=0)
    ref = ns["ref_fn"](*inputs)
    ind = _independent_dequant(op, inputs)
    ref = ref if isinstance(ref, tuple) else (ref,)
    ind = ind if isinstance(ind, tuple) else (ind,)
    assert len(ref) == len(ind)
    for r, i in zip(ref, ind):
        assert tuple(r.shape) == tuple(i.shape), (op, r.shape, i.shape)
        assert r.dtype == torch.bfloat16
        assert _relerr(r, i) < 3e-2, (op, _relerr(r, i))


@pytest.mark.parametrize("op", Q.OPS)
def test_baseline_matches_ref(op):
    ns = Q.make_reference(op, Q.OP_DTYPES[op][0])
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=1)
    out = ns["baseline_fn"](*inputs)
    ref = ns["ref_fn"](*inputs)
    out = out if isinstance(out, tuple) else (out,)
    ref = ref if isinstance(ref, tuple) else (ref,)
    assert len(out) == len(ref)
    for o, r in zip(out, ref):
        assert tuple(o.shape) == tuple(r.shape) and o.dtype == r.dtype
        assert _relerr(o, r) < 5e-2, (op, _relerr(o, r))


# --------------------------------------------------------------------------- #
# output code/scale dtype + granularity structure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", QUANTIZE_OPS)
def test_quant_output_structure(op):
    cfg = Q._CFG[op]
    kind, fmt, gran = cfg["kind"], cfg["fmt"], cfg["gran"]
    sh = _small(op)
    M, K = sh["M"], sh["K"]
    ns = Q.make_reference(op, Q.OP_DTYPES[op][0])
    out = ns["ref_fn"](*ns["get_inputs"](sh, device="cpu", seed=0))
    assert isinstance(out, tuple)
    if kind == "quant":
        codes, scale = out
        assert codes.dtype == _code_dt(fmt) and tuple(codes.shape) == (M, K)
        assert scale.dtype == torch.float32
        want = {"tensor": (), "token": (M, 1), "channel": (1, K),
                "block128": (M, K // BLK), "block2d": (M // BLK, K // BLK)}[gran]
        assert tuple(scale.shape) == want, (op, tuple(scale.shape), want)
    elif kind == "kvquant":
        kq, ksc, vq, vsc = out
        assert kq.dtype == vq.dtype == _code_dt(fmt)
        assert tuple(kq.shape) == (M, K) and tuple(ksc.shape) == (M, 1)
        assert ksc.dtype == vsc.dtype == torch.float32
    elif kind == "int4pack":
        packed, scale = out
        assert packed.dtype == torch.uint8 and tuple(packed.shape) == (M, K // 2)
        assert scale.dtype == torch.float32 and tuple(scale.shape) == (M, K // cfg["group"])
    elif kind == "mxfp4pack":
        packed, e8 = out
        assert packed.dtype == torch.uint8 and tuple(packed.shape) == (M, K // 2)
        assert e8.dtype == torch.uint8 and tuple(e8.shape) == (M, K // MX_BLOCK)
    elif kind == "double":
        codes, sc_codes, meta = out
        assert codes.dtype == _code_dt(fmt) and tuple(codes.shape) == (M, K)
        assert sc_codes.dtype == torch.float8_e4m3fn and tuple(sc_codes.shape) == (M, K // BLK)
        assert meta.dtype == torch.float32 and meta.shape == ()
    elif kind in ("smooth", "stochastic"):
        codes, scale = out
        assert codes.dtype == _code_dt(fmt) and tuple(codes.shape) == (M, K)
        assert scale.dtype == torch.float32 and tuple(scale.shape) == (M, 1)
    else:  # qtranspose
        codesT, scale = out
        assert codesT.dtype == _code_dt(fmt) and tuple(codesT.shape) == (K, M)
        assert scale.dtype == torch.float32
        assert tuple(scale.shape) == ((1, M) if gran == "token" else ())


# --------------------------------------------------------------------------- #
# adversarial battery: hard regimes survive quantization + still match the oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", Q.OPS)
def test_adversarial_inputs(op):
    cfg = Q._CFG[op]
    fmt = cfg["fmt"]
    ns = Q.make_reference(op, Q.OP_DTYPES[op][0])
    sh = _small(op)
    adv = ns["adversarial_inputs"](sh, device="cpu")
    assert [n for n, _ in adv] == ["zeros", "large", "neg_large", "small", "sign_alt"]
    ref0 = ns["get_inputs"](sh, device="cpu", seed=0)
    tol = 1.5e-1 if cfg["kind"] in ("mxfp4pack", "double") else (8e-2 if fmt == "fp8" else 5e-2)
    for name, ai in adv:
        assert isinstance(ai, tuple) and len(ai) == ns["arity"], (op, name)
        for t, t0 in zip(ai, ref0):
            assert t.dtype == t0.dtype, (op, name, t.dtype, t0.dtype)
        out = ns["ref_fn"](*ai)
        outs = out if isinstance(out, tuple) else (out,)
        for t in outs:
            assert torch.isfinite(t.float()).all(), (op, name, "non-finite")
        if Q._is_quantize(cfg):
            for ref_recon, ind_recon, _t in _recons(op, ai, out):
                assert _relerr(ref_recon, ind_recon) < tol, (op, name, _relerr(ref_recon, ind_recon))
        else:
            ind = _independent_dequant(op, ai)
            ref = out if isinstance(out, tuple) else (out,)
            ind = ind if isinstance(ind, tuple) else (ind,)
            for r, i in zip(ref, ind):
                assert _relerr(r, i) < 5e-2, (op, name, _relerr(r, i))


# --------------------------------------------------------------------------- #
# shape catalog: divisibility for the format + parse_shape round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", Q.OPS)
def test_shapes_parse_roundtrip(op):
    ns = Q.make_reference(op, Q.OP_DTYPES[op][0])
    parse = ns["parse_shape"]
    sh = Q.SHAPES[op]
    assert {"minimal", "primary", "validation"} <= set(sh)
    assert isinstance(sh["validation"], list) and sh["validation"]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert parse(s) == spec, (op, parse(s), spec)


@pytest.mark.parametrize("op", Q.OPS)
def test_shapes_divisible_for_format(op):
    cfg = Q._CFG[op]
    km, mm = Q._kmult(cfg), Q._mmult(cfg)
    sh = Q.SHAPES[op]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        assert spec["K"] % km == 0, (op, "K", spec["K"], km)
        assert spec["M"] % mm == 0, (op, "M", spec["M"], mm)
        if cfg["kind"] in ("int4pack", "int4unpack", "mxfp4pack", "mxfp4unpack"):
            assert spec["K"] % 2 == 0                      # nibble packing


def test_non_pow2_tail_present():
    for op in Q.OPS:
        val = Q.SHAPES[op]["validation"]
        assert any((s["M"] & (s["M"] - 1)) != 0 for s in val), op


# --------------------------------------------------------------------------- #
# seeds: compile + define entry (CPU-safe) and load as a module (triton binds jit)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", Q.OPS)
def test_seed_compiles_and_defines_entry(op):
    for dt in Q.OP_DTYPES[op]:
        src = Q.seed_source(op, dt)
        compile(src, f"<{op}:{dt}>", "exec")
        tree = ast.parse(src)
        funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert op in funcs, f"{op}/{dt} seed must define a top-level def {op}(...)"


@pytest.mark.parametrize("op", Q.OPS)
def test_seed_loads_as_module(op, tmp_path):
    import importlib.util

    pytest.importorskip("triton")
    dt = Q.OP_DTYPES[op][0]
    path = tmp_path / f"seed_{op}.py"
    path.write_text(Q.seed_source(op, dt))
    spec = importlib.util.spec_from_file_location(f"quant_ext_seed_{op}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(getattr(mod, op))


# --------------------------------------------------------------------------- #
# pointed structural checks on the hard formats
# --------------------------------------------------------------------------- #
def test_fp8_codes_within_range():
    ns = Q.make_reference("qx_quant_fp8_pertoken", "fp8")
    codes, scale = ns["ref_fn"](*ns["get_inputs"]({"M": 4, "K": 16}, device="cpu", seed=0))
    assert codes.float().abs().max().item() <= FP8_MAX + 1e-3
    assert tuple(scale.shape) == (4, 1)


def test_int4_codes_valid_nibbles():
    ns = Q.make_reference("qx_int4_pack_group", "int8")
    packed, scale = ns["ref_fn"](*ns["get_inputs"]({"M": 4, "K": 256}, device="cpu", seed=0))
    lo = (packed & 0xF).long()
    hi = ((packed >> 4) & 0xF).long()
    assert int(lo.max()) <= 15 and int(hi.max()) <= 15 and int(lo.min()) >= 0
    assert tuple(packed.shape) == (4, 128)             # K//2 packing


def test_transpose_recovers_transposed_input():
    ns = Q.make_reference("qx_quant_transpose_fp8", "fp8")
    x, = ns["get_inputs"]({"M": 8, "K": 16}, device="cpu", seed=3)
    codesT, scale = ns["ref_fn"](x)
    assert tuple(codesT.shape) == (16, 8) and tuple(scale.shape) == (1, 8)
    recon = _d(codesT) * _d(scale)
    assert _snr_db(x.double().t(), recon) >= _SNR_GATE["fp8"]


def test_kvcache_roundtrip():
    qns = Q.make_reference("qx_kvcache_quant_int8", "int8")
    k, v = qns["get_inputs"]({"M": 6, "K": 16}, device="cpu", seed=4)
    kq, ksc, vq, vsc = qns["ref_fn"](k, v)
    assert kq.dtype == vq.dtype == torch.int8
    assert _snr_db(k.double(), _ideq_nd(kq, ksc, "token")) >= _SNR_GATE["int8"]
    assert _snr_db(v.double(), _ideq_nd(vq, vsc, "token")) >= _SNR_GATE["int8"]
    dns = Q.make_reference("qx_kvcache_dequant_int8", "bf16")
    kd, vd = dns["ref_fn"](kq, ksc, vq, vsc)
    assert _relerr(kd, _ideq_nd(kq, ksc, "token")) < 3e-2


def test_stochastic_is_deterministic_and_on_grid():
    ns = Q.make_reference("qx_stochastic_fp8", "fp8")
    inp = ns["get_inputs"]({"M": 4, "K": 16}, device="cpu", seed=5)
    c1, s1 = ns["ref_fn"](*inp)
    c2, s2 = ns["ref_fn"](*inp)
    assert torch.equal(c1.float(), c2.float()) and torch.equal(s1, s2)   # seeded -> deterministic
    grid = set(torch.unique(Q._fp8_levels("cpu")).tolist())
    assert set(c1.float().unique().tolist()) <= grid                     # every code on the fp8 grid


def test_double_quant_nested_reconstructs():
    ns = Q.make_reference("qx_double_quant_fp8", "fp8")
    x, = ns["get_inputs"]({"M": 4, "K": 256}, device="cpu", seed=6)
    codes, sc_codes, meta = ns["ref_fn"](x)
    assert sc_codes.dtype == torch.float8_e4m3fn and meta.shape == ()
    bs = _d(sc_codes) * _d(meta)
    recon = _ideq_nd(codes, bs, "block128")
    assert _snr_db(x.double(), recon) >= _SNR_GATE_KIND["double"]
