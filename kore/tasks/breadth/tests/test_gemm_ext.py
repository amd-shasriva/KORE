"""CPU-only tests for the breadth quantized/mixed-precision GEMM engine (gemm_ext).

Every ``ref_fn`` oracle is checked against an INDEPENDENT torch computation on a
DIFFERENT code path than the (vectorized ``@`` / broadcast) oracle it wraps:

  * operands are re-dequantized in float64 with reshape-based scale broadcasting
    (not the module's ``repeat_interleave``), and int4/MXFP4/MXFP8 codes are
    unpacked with an independent LUT/loop,
  * the matmul is an ``einsum`` (plain), a per-token loop (grouped), or a
    transposed ``einsum`` (batched / dA / dB),
  * gelu-tanh is the explicit ``0.5 x (1+tanh(...))`` polynomial (not F.gelu),

so a wrong dequant / transpose / epilogue / scale-broadcast in the oracle is
caught with certainty. The quantized candidate + reference share the SAME codes,
so this measures the matmul/epilogue fidelity, not the (shared) quant error.

Also asserts the ABI surface, arity, quant code/scale dtype + granularity
structure, the adversarial battery (hard regimes survive quantization), output
dtype (bf16 for every quantized op), that each seed compiles + defines + loads its
entry, shape-catalog divisibility + ``parse_shape`` round-trip, and the lazy torch
import. All fp32/fp64 on CPU - no GPU / triton kernel is ever launched.
"""

from __future__ import annotations

import ast
import math

import pytest
import torch

from kore.tasks._genops import DTYPES
from kore.tasks.breadth import gemm_ext as G

_TORCH_DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
             "fp8": torch.float8_e4m3fn, "int8": torch.int8}


# --------------------------------------------------------------------------- #
# tiny CPU shapes per op (honour the op's K/N divisibility)
# --------------------------------------------------------------------------- #
def _small(op: str) -> dict:
    cfg = G._CFG[op]
    fam = cfg["fam"]
    km, nm = G._kmult(cfg), G._nmult(cfg)
    K = km * max(1, math.ceil(8 / km))
    N = nm * max(1, math.ceil(6 / nm))
    if fam == "batched":
        return {"B": 2, "M": 4, "N": N, "K": K}
    if fam == "grouped":
        return {"E": 2, "M": 6, "N": N, "K": K}
    return {"M": 4, "N": N, "K": K}


def _relerr(a, b) -> float:
    a, b = a.double(), b.double()
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


# --------------------------------------------------------------------------- #
# INDEPENDENT float64 dequant (reshape broadcasting; distinct from the module)
# --------------------------------------------------------------------------- #
def _ind_deq8_a(codes, s, gran):
    c = codes.double()
    if gran == "block128":
        M, K = c.shape
        nb = s.shape[1]
        return (c.reshape(M, nb, K // nb) * s.double()[:, :, None]).reshape(M, K)
    return c * s.double()


def _ind_deq8_w(codes, s, gran):
    c = codes.double()
    if gran == "block128":
        N, K = c.shape
        nbn, nbk = s.shape
        return (c.reshape(nbn, N // nbn, nbk, K // nbk)
                * s.double()[:, None, :, None]).reshape(N, K)
    return c * s.double()


def _unpack_nibbles(packed):
    N, K = packed.shape[0], packed.shape[1] * 2
    lo = (packed & 0xF).long()
    hi = ((packed >> 4) & 0xF).long()
    out = torch.zeros((N, K), dtype=torch.long)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


def _ind_int4c(packed, scale):
    q = _unpack_nibbles(packed) - 8
    return q.double() * scale.double()


def _ind_int4gs(packed, scale, group):
    q = _unpack_nibbles(packed) - 8
    N, K = q.shape
    return (q.reshape(N, K // group, group).double()
            * scale.double()[:, :, None]).reshape(N, K)


def _ind_int4ga(packed, scale, zero, group):
    code = _unpack_nibbles(packed)
    N, K = code.shape
    code = code.reshape(N, K // group, group).double()
    val = (code - zero.double()[:, :, None]) * scale.double()[:, :, None]
    return val.reshape(N, K)


def _ind_e2m1(codes):
    mag_lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    mag = torch.tensor(mag_lut, dtype=torch.float64)[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0).double()
    return sign * mag


def _ind_mxfp4(packed, e8m0):
    R, K = packed.shape[0], packed.shape[1] * 2
    codes = torch.zeros((R, K), dtype=torch.long)
    codes[:, 0::2] = (packed & 0xF).long()
    codes[:, 1::2] = ((packed >> 4) & 0xF).long()
    vals = _ind_e2m1(codes).reshape(R, K // 32, 32)
    scale = torch.pow(2.0, e8m0.double() - 127.0)[:, :, None]
    return (vals * scale).reshape(R, K)


def _ind_mxfp8(codes, e8m0):
    R, K = codes.shape
    scale = torch.pow(2.0, e8m0.double() - 127.0)[:, :, None]
    return (codes.double().reshape(R, K // 32, 32) * scale).reshape(R, K)


def _ind_A(cfg, inp, i):
    aq = cfg["aq"]
    if aq in ("bf16", "fp16"):
        return inp[i].double(), i + 1
    if aq in ("fp8", "int8"):
        return _ind_deq8_a(inp[i], inp[i + 1], cfg["asg"]), i + 2
    if aq == "mxfp4":
        return _ind_mxfp4(inp[i], inp[i + 1]), i + 2
    return _ind_mxfp8(inp[i], inp[i + 1]), i + 2


def _ind_W(cfg, inp, i):
    wq = cfg["wq"]
    if wq in ("bf16", "fp16"):
        return inp[i].double(), i + 1
    if wq in ("fp8", "int8"):
        return _ind_deq8_w(inp[i], inp[i + 1], cfg["wsg"]), i + 2
    if wq == "int4c":
        return _ind_int4c(inp[i], inp[i + 1]), i + 2
    if wq == "int4gs":
        return _ind_int4gs(inp[i], inp[i + 1], cfg["group"]), i + 2
    if wq == "int4ga":
        return _ind_int4ga(inp[i], inp[i + 1], inp[i + 2], cfg["group"]), i + 3
    if wq == "mxfp4":
        return _ind_mxfp4(inp[i], inp[i + 1]), i + 2
    return _ind_mxfp8(inp[i], inp[i + 1]), i + 2


def _ind_act(y, act):
    if act == "gelu":  # explicit tanh-gelu polynomial (distinct from F.gelu)
        return 0.5 * y * (1.0 + torch.tanh(0.7978845608028654 * (y + 0.044715 * y ** 3)))
    if act == "relu":
        return torch.clamp(y, min=0.0)
    if act == "silu":
        return y * torch.sigmoid(y)
    return y


def _independent(op, inputs):
    cfg = G._CFG[op]
    fam = cfg["fam"]
    if fam == "dgrad":
        dy, w = inputs
        return torch.einsum("mn,nk->mk", dy.double(), w.double())
    if fam == "wgrad":
        dy, a = inputs
        return torch.einsum("mn,mk->nk", dy.double(), a.double())
    if fam == "batched":
        if G._is_quant(cfg):
            aq, asc, wq, wsc = inputs
            A = aq.double() * asc.double()
            W = wq.double() * wsc.double()
        else:
            A, W = inputs[0].double(), inputs[1].double()
        return torch.einsum("bmk,bnk->bmn", A, W)
    if fam == "grouped":
        if G._is_quant(cfg):
            aq, asc, wq, wsc, eids = inputs
            A = aq.double() * asc.double()
            W = wq.double() * wsc.double()
        else:
            A, W, eids = inputs[0].double(), inputs[1].double(), inputs[2]
        M, N = A.shape[0], W.shape[1]
        out = torch.zeros(M, N, dtype=torch.float64)
        for m in range(M):
            out[m] = A[m] @ W[int(eids[m])].t()
        return out
    # plain
    i = 0
    A, i = _ind_A(cfg, inputs, i)
    W, i = _ind_W(cfg, inputs, i)
    y = torch.einsum("mk,nk->mn", A, W)
    if G._has_bias(cfg):
        y = y + inputs[i].double().reshape(1, -1)
        i += 1
    y = _ind_act(y, G._act_of(cfg["ep"]))
    if cfg["ep"] == "residual":
        y = y + inputs[i].double()
        i += 1
    if cfg["ep"] == "requant":
        osc = inputs[i].double()
        i += 1
        yq = (y / osc).clamp(-448.0, 448.0).to(torch.float32).to(torch.float8_e4m3fn)
        y = yq.double() * osc
    return y


# --------------------------------------------------------------------------- #
# ABI surface
# --------------------------------------------------------------------------- #
def test_abi_surface():
    assert isinstance(G.OPS, list) and len(G.OPS) == 45
    assert len(set(G.OPS)) == len(G.OPS)                       # no duplicates
    assert set(G.OP_DTYPES) == set(G.OPS) == set(G.SHAPES) == set(G._CFG)
    assert callable(G.make_reference) and callable(G.seed_source)
    assert all(op.startswith("gemm_") for op in G.OPS)
    for attr in ("OPS", "OP_DTYPES", "SHAPES", "make_reference", "seed_source"):
        assert hasattr(G, attr)


def test_op_dtypes_valid():
    for op in G.OPS:
        dts = G.OP_DTYPES[op]
        assert len(dts) == 1 and dts[0] in DTYPES, (op, dts)
        cfg = G._CFG[op]
        if G._is_quant(cfg):
            assert dts[0] in ("fp8", "int8", "bf16", "fp16")
        else:
            assert dts[0] in ("bf16", "fp16")


def test_torch_imported_lazily():
    """Registry discovery must be GPU-free: no top-level torch import (the ``import
    torch`` inside helper bodies + seed STRINGS is ignored - AST top level only)."""
    import inspect
    tree = ast.parse(inspect.getsource(G))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("torch") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch")


@pytest.mark.parametrize("op", G.OPS)
def test_arity(op):
    ns = G.make_reference(op, G.OP_DTYPES[op][0])
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=0)
    assert isinstance(inputs, tuple)
    assert len(inputs) == ns["arity"] == G.arity_of(op)


@pytest.mark.parametrize("op", G.OPS)
def test_namespace_contract(op):
    dt = G.OP_DTYPES[op][0]
    ns = G.make_reference(op, dt)
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family", "mutates_input"):
        assert k in ns, f"{op} missing ns key {k!r}"
    assert ns["entry_name"] == op
    assert ns["dtype_name"] == dt
    assert ns["family"] == f"breadth_{op}"
    assert ns["mutates_input"] is False
    assert ns[f"{op}_ref"] is ns["ref_fn"]
    assert ns["parse_shape"] is G._parse_shape
    assert callable(ns["ref_fn"]) and callable(ns["baseline_fn"])
    # quantized ops MUST author adversarial_inputs (generic float fills are invalid
    # for fp8/int8/int4 structured inputs).
    if G._is_quant(G._CFG[op]):
        assert callable(ns.get("adversarial_inputs"))


# --------------------------------------------------------------------------- #
# correctness: ref_fn oracle vs INDEPENDENT torch (the crux of the suite)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", G.OPS)
def test_ref_matches_independent(op):
    quant = G._is_quant(G._CFG[op])
    dt = G.OP_DTYPES[op][0] if quant else "fp32"        # fp32 for a tight dense check
    ns = G.make_reference(op, dt)
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=0)
    ref = ns["ref_fn"](*inputs)
    ind = _independent(op, inputs)
    assert tuple(ref.shape) == tuple(ind.shape), (op, tuple(ref.shape), tuple(ind.shape))
    tol = 3e-2 if quant else 2e-4                        # bf16-out (quant) vs fp32-out
    assert _relerr(ref, ind) < tol, (op, _relerr(ref, ind))


@pytest.mark.parametrize("op", G.OPS)
def test_baseline_matches_ref(op):
    """The torch eager production baseline agrees with the fp32 oracle (same math,
    bf16-materialized matmul for the quantized ops)."""
    dt = G.OP_DTYPES[op][0]
    ns = G.make_reference(op, dt)
    inputs = ns["get_inputs"](_small(op), device="cpu", seed=1)
    out = ns["baseline_fn"](*inputs)
    ref = ns["ref_fn"](*inputs)
    assert tuple(out.shape) == tuple(ref.shape)
    assert out.dtype == ref.dtype
    assert _relerr(out, ref) < 5e-2, (op, _relerr(out, ref))


@pytest.mark.parametrize("op", G.OPS)
def test_ref_output_dtype(op):
    """Quantized ops emit bf16 (fp32-accumulate -> bf16 vendor convention); dense ops
    preserve the task float dtype."""
    dt = G.OP_DTYPES[op][0]
    ns = G.make_reference(op, dt)
    out = ns["ref_fn"](*ns["get_inputs"](_small(op), device="cpu", seed=0))
    exp = torch.bfloat16 if G._is_quant(G._CFG[op]) else _TORCH_DT[dt]
    assert out.dtype == exp, (op, out.dtype, exp)


# --------------------------------------------------------------------------- #
# quantized input code/scale dtype + granularity structure
# --------------------------------------------------------------------------- #
def _code_dt(fmt):
    if fmt == "fp8":
        return torch.float8_e4m3fn
    if fmt == "int8":
        return torch.int8
    if fmt == "mxfp8":
        return torch.float8_e4m3fn
    return torch.uint8            # int4 packed / mxfp4 packed


@pytest.mark.parametrize("op", [o for o in G.OPS if G._CFG[o]["fam"] == "plain"
                                and G._is_quant(G._CFG[o])])
def test_quant_input_structure(op):
    cfg = G._CFG[op]
    ns = G.make_reference(op, G.OP_DTYPES[op][0])
    sh = _small(op)
    inp = ns["get_inputs"](sh, device="cpu", seed=0)
    M, N, K = sh["M"], sh["N"], sh["K"]
    i = 0
    # ---- A operand
    if cfg["aq"] in ("fp8", "int8"):
        codes, s = inp[i], inp[i + 1]; i += 2
        assert codes.dtype == _code_dt(cfg["aq"]) and codes.shape == (M, K)
        assert s.dtype == torch.float32
        want = {"tensor": (), "row": (M, 1), "block128": (M, K // 128)}[cfg["asg"]]
        assert tuple(s.shape) == want, (op, "A scale", tuple(s.shape), want)
    elif cfg["aq"] in ("mxfp4", "mxfp8"):
        codes, e8 = inp[i], inp[i + 1]; i += 2
        assert codes.dtype == _code_dt(cfg["aq"])
        assert e8.dtype == torch.uint8 and tuple(e8.shape) == (M, K // 32)
    else:
        assert inp[i].dtype in (torch.bfloat16, torch.float16); i += 1
    # ---- W operand
    wq = cfg["wq"]
    if wq in ("fp8", "int8"):
        codes, s = inp[i], inp[i + 1]; i += 2
        assert codes.dtype == _code_dt(wq) and codes.shape == (N, K)
        assert s.dtype == torch.float32
        want = {"tensor": (), "channel": (N, 1), "block128": (N // 128, K // 128)}[cfg["wsg"]]
        assert tuple(s.shape) == want, (op, "W scale", tuple(s.shape), want)
    elif wq in ("int4c", "int4gs", "int4ga"):
        packed, s = inp[i], inp[i + 1]
        assert packed.dtype == torch.uint8 and packed.shape == (N, K // 2)
        assert s.dtype == torch.float32
        if wq == "int4ga":
            assert inp[i + 2].dtype == torch.uint8           # zero-point codes
    elif wq == "mxfp4":
        packed, e8 = inp[i], inp[i + 1]
        assert packed.dtype == torch.uint8 and packed.shape == (N, K // 2)
        assert e8.dtype == torch.uint8 and tuple(e8.shape) == (N, K // 32)
    elif wq == "mxfp8":
        codes, e8 = inp[i], inp[i + 1]
        assert codes.dtype == torch.float8_e4m3fn and codes.shape == (N, K)
        assert e8.dtype == torch.uint8


# --------------------------------------------------------------------------- #
# adversarial battery: hard regimes survive quantization + still match the oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", [o for o in G.OPS if G._is_quant(G._CFG[o])])
def test_adversarial_inputs(op):
    ns = G.make_reference(op, G.OP_DTYPES[op][0])
    sh = _small(op)
    adv = ns["adversarial_inputs"](sh, device="cpu")
    assert [n for n, _ in adv] == ["zeros", "large", "neg_large", "small", "sign_alt"]
    ref0_inp = ns["get_inputs"](sh, device="cpu", seed=0)
    for name, ai in adv:
        assert isinstance(ai, tuple) and len(ai) == ns["arity"], (op, name)
        # quantized code/scale dtypes preserved under every hard regime
        for t, t0 in zip(ai, ref0_inp):
            assert t.dtype == t0.dtype, (op, name, t.dtype, t0.dtype)
        ref = ns["ref_fn"](*ai)
        ind = _independent(op, ai)
        assert torch.isfinite(ref.float()).all(), (op, name, "non-finite ref")
        assert _relerr(ref, ind) < 5e-2, (op, name, _relerr(ref, ind))


# --------------------------------------------------------------------------- #
# shape catalog: divisibility for the format + parse_shape round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", G.OPS)
def test_shapes_parse_roundtrip(op):
    ns = G.make_reference(op, G.OP_DTYPES[op][0])
    parse = ns["parse_shape"]
    sh = G.SHAPES[op]
    assert {"minimal", "primary", "validation"} <= set(sh)
    assert isinstance(sh["validation"], list) and sh["validation"]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert parse(s) == spec, (op, parse(s), spec)


@pytest.mark.parametrize("op", G.OPS)
def test_shapes_divisible_for_format(op):
    cfg = G._CFG[op]
    km, nm = G._kmult(cfg), G._nmult(cfg)
    sh = G.SHAPES[op]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        assert spec["K"] % km == 0, (op, "K", spec["K"], km)
        assert spec["N"] % nm == 0, (op, "N", spec["N"], nm)
        if cfg["wq"] in ("int4c", "int4gs", "int4ga", "mxfp4") or "mxfp4" in (cfg["aq"],):
            assert spec["K"] % 2 == 0                          # nibble packing


def test_non_pow2_tail_present():
    """Each regime's validation carries a genuinely non-power-of-2 M (mask stress)."""
    for op in G.OPS:
        val = G.SHAPES[op]["validation"]
        assert any((s["M"] & (s["M"] - 1)) != 0 for s in val), op


# --------------------------------------------------------------------------- #
# seeds: compile + define entry (CPU-safe) and load as a module (triton binds jit)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", G.OPS)
def test_seed_compiles_and_defines_entry(op):
    for dt in G.OP_DTYPES[op]:
        src = G.seed_source(op, dt)
        compile(src, f"<{op}:{dt}>", "exec")
        tree = ast.parse(src)
        funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert op in funcs, f"{op}/{dt} seed must define a top-level def {op}(...)"


@pytest.mark.parametrize("op", G.OPS)
def test_seed_loads_as_module(op, tmp_path):
    """Stronger check: import the seed from a real file so every @triton.jit binds +
    the entry resolves (no kernel is LAUNCHED, so this stays CPU-safe)."""
    import importlib.util

    pytest.importorskip("triton")
    dt = G.OP_DTYPES[op][0]
    path = tmp_path / f"seed_{op}.py"
    path.write_text(G.seed_source(op, dt))
    spec = importlib.util.spec_from_file_location(f"gemm_ext_seed_{op}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(getattr(mod, op))


# --------------------------------------------------------------------------- #
# a couple of pointed structural checks on the hard formats
# --------------------------------------------------------------------------- #
def test_fp8_codes_within_range():
    ns = G.make_reference("gemm_fp8_rowwise", "fp8")
    aq, asc, wq, wsc = ns["get_inputs"]({"M": 4, "N": 6, "K": 16}, device="cpu", seed=0)
    assert aq.float().abs().max().item() <= 448.0 + 1e-3
    assert asc.shape == (4, 1) and wsc.shape == (6, 1)      # per-token / per-channel


def test_int4_codes_within_range():
    ns = G.make_reference("gemm_int4_sym_channel", "bf16")
    a, wpk, wsc = ns["get_inputs"]({"M": 4, "N": 6, "K": 16}, device="cpu", seed=0)
    lo = (wpk & 0xF).long()
    hi = ((wpk >> 4) & 0xF).long()
    assert int(lo.max()) <= 15 and int(hi.max()) <= 15      # valid 4-bit codes
    assert wpk.shape == (6, 8)                              # K//2 packing


def test_mxfp4_block_structure():
    ns = G.make_reference("gemm_mxfp4", "bf16")
    apk, ae8, wpk, we8 = ns["get_inputs"]({"M": 4, "N": 6, "K": 64}, device="cpu", seed=0)
    assert apk.shape == (4, 32) and ae8.shape == (4, 2)     # K//2 packed, K//32 e8m0
    assert wpk.shape == (6, 32) and we8.shape == (6, 2)


def test_grouped_variable_m_matches_per_token():
    ns = G.make_reference("gemm_fp8_grouped", "fp8")
    inp = ns["get_inputs"]({"E": 3, "M": 12, "N": 6, "K": 16}, device="cpu", seed=3)
    ref = ns["ref_fn"](*inp).double()
    aq, asc, wq, wsc, eids = inp
    A = aq.double() * asc.double()
    W = wq.double() * wsc.double()
    dense = torch.stack([A[m] @ W[int(eids[m])].t() for m in range(A.shape[0])])
    assert _relerr(ref, dense) < 3e-2
