"""CPU-only tests for the breadth conv/pooling/resize authoring engine.

Every ``ref_fn`` is checked against an INDEPENDENT torch computation (a different
code path than the ``F.*`` baseline it wraps), so a wrong oracle is caught with
certainty. Also asserts arity, the seed compiles + defines its entry, the shape
catalog round-trips through ``parse_shape``, and the namespace ABI matches
vendor_ops.py. All fp32 on CPU (no GPU / triton execution needed)."""

from __future__ import annotations

import ast

import pytest
import torch
import torch.nn.functional as F

from kore.tasks.breadth import conv as C

# --------------------------------------------------------------------------- #
# tiny CPU shapes + expected arity per op
# --------------------------------------------------------------------------- #
_SMALL = {
    "conv2d_nchw": {"N": 2, "Cin": 4, "H": 8, "W": 8, "Cout": 6, "K": 3},
    "dilated_conv2d": {"N": 2, "Cin": 4, "H": 10, "W": 10, "Cout": 6, "K": 3},
    "depthwise_conv2d": {"N": 2, "C": 5, "H": 8, "W": 8, "K": 3},
    "maxpool2d": {"N": 2, "C": 3, "H": 8, "W": 8},
    "avgpool2d": {"N": 2, "C": 3, "H": 8, "W": 8},
    "adaptive_avgpool2d": {"N": 2, "C": 3, "H": 14, "W": 14},
    "global_avgpool": {"N": 2, "C": 3, "H": 8, "W": 8},
    "interpolate_bilinear": {"N": 2, "C": 3, "H": 8, "W": 8},
    "interpolate_nearest": {"N": 2, "C": 3, "H": 8, "W": 8},
}
_ARITY = {
    "conv2d_nchw": 3, "dilated_conv2d": 3, "depthwise_conv2d": 3,
    "maxpool2d": 1, "avgpool2d": 1, "adaptive_avgpool2d": 1,
    "global_avgpool": 1, "interpolate_bilinear": 1, "interpolate_nearest": 1,
}


# --------------------------------------------------------------------------- #
# independent torch oracles (distinct code paths from the F.* baselines)
# --------------------------------------------------------------------------- #
def _ind_conv(x, w, b, stride, pad, dil, groups):
    """im2col-free conv via shifted-slice accumulate (no F.conv2d)."""
    N, Cin, H, W = x.shape
    Cout, KH, KW = w.shape[0], w.shape[2], w.shape[3]
    OH = (H + 2 * pad - dil * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * pad - dil * (KW - 1) - 1) // stride + 1
    xp = F.pad(x, (pad, pad, pad, pad))
    out = torch.zeros((N, Cout, OH, OW), dtype=x.dtype)
    for kh in range(KH):
        for kw in range(KW):
            sl_h = slice(kh * dil, kh * dil + (OH - 1) * stride + 1, stride)
            sl_w = slice(kw * dil, kw * dil + (OW - 1) * stride + 1, stride)
            patch = xp[:, :, sl_h, sl_w]                      # [N, Cin, OH, OW]
            if groups == 1:
                out += torch.einsum("ncij,oc->noij", patch, w[:, :, kh, kw])
            else:                                             # depthwise (groups==Cin)
                out += patch * w[:, 0, kh, kw].view(1, -1, 1, 1)
    return out + b.view(1, -1, 1, 1)


def _ind_pool2x(x, mode):
    N, Cn, H, W = x.shape
    OH, OW = H // 2, W // 2
    xr = x[:, :, :2 * OH, :2 * OW].reshape(N, Cn, OH, 2, OW, 2)
    return xr.amax(dim=(3, 5)) if mode == "max" else xr.mean(dim=(3, 5))


def _ind_adaptive(x, out=7):
    N, Cn, H, W = x.shape
    kh, kw = H // out, W // out
    return x.reshape(N, Cn, out, kh, out, kw).mean(dim=(3, 5))


def _ind_global(x):
    N, Cn, H, W = x.shape
    return x.reshape(N, Cn, -1).sum(dim=2) / (H * W)


def _ind_bilinear2x(x):
    N, Cn, H, W = x.shape
    OH, OW = 2 * H, 2 * W
    oh = torch.arange(OH, dtype=torch.float32)
    ow = torch.arange(OW, dtype=torch.float32)
    sh = torch.clamp((oh + 0.5) * 0.5 - 0.5, min=0.0)
    sw = torch.clamp((ow + 0.5) * 0.5 - 0.5, min=0.0)
    h0 = sh.floor().long(); h1 = torch.clamp(h0 + 1, max=H - 1)
    w0 = sw.floor().long(); w1 = torch.clamp(w0 + 1, max=W - 1)
    lh = (sh - h0.float()).view(1, 1, OH, 1)
    lw = (sw - w0.float()).view(1, 1, 1, OW)
    v00 = x[:, :, h0][:, :, :, w0]
    v01 = x[:, :, h0][:, :, :, w1]
    v10 = x[:, :, h1][:, :, :, w0]
    v11 = x[:, :, h1][:, :, :, w1]
    top = v00 * (1 - lw) + v01 * lw
    bot = v10 * (1 - lw) + v11 * lw
    return top * (1 - lh) + bot * lh


def _ind_nearest2x(x):
    return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


def _independent(op, inputs):
    x = inputs[0]
    if op == "conv2d_nchw":
        return _ind_conv(x, inputs[1], inputs[2], 1, 1, 1, groups=1)
    if op == "dilated_conv2d":
        return _ind_conv(x, inputs[1], inputs[2], 1, 2, 2, groups=1)
    if op == "depthwise_conv2d":
        return _ind_conv(x, inputs[1], inputs[2], 1, 1, 1, groups=x.shape[1])
    if op == "maxpool2d":
        return _ind_pool2x(x, "max")
    if op == "avgpool2d":
        return _ind_pool2x(x, "avg")
    if op == "adaptive_avgpool2d":
        return _ind_adaptive(x, C.ADAPTIVE_OUT)
    if op == "global_avgpool":
        return _ind_global(x)
    if op == "interpolate_bilinear":
        return _ind_bilinear2x(x)
    if op == "interpolate_nearest":
        return _ind_nearest2x(x)
    raise AssertionError(f"no independent oracle for {op!r}")


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_abi_present():
    assert isinstance(C.OPS, list) and len(C.OPS) == 9
    assert callable(C.make_reference) and callable(C.seed_source)
    assert set(C.OP_DTYPES) == set(C.OPS)
    assert set(C.SHAPES) == set(C.OPS)


def test_ops_dtypes_shapes_consistent():
    for op in C.OPS:
        assert C.OP_DTYPES[op], f"empty dtype sweep for {op}"
        for d in C.OP_DTYPES[op]:
            assert d in DTYPE_NAMES, f"unknown dtype {d} for {op}"
        sh = C.SHAPES[op]
        assert "minimal" in sh and "primary" in sh and "validation" in sh
        assert isinstance(sh["validation"], list) and sh["validation"]


DTYPE_NAMES = ("bf16", "fp16", "fp32")


@pytest.mark.parametrize("op", C.OPS)
def test_ref_matches_independent(op):
    ns = C.make_reference(op, "fp32")
    inputs = ns["get_inputs"](_SMALL[op], device="cpu", seed=0)
    ref = ns["ref_fn"](*inputs)
    ind = _independent(op, inputs)
    assert ref.shape == ind.shape, f"{op}: {ref.shape} vs {ind.shape}"
    assert torch.allclose(ref.float(), ind.float(), atol=1e-4, rtol=1e-3), (
        f"{op}: max|diff|={ (ref.float() - ind.float()).abs().max().item() }")


@pytest.mark.parametrize("op", C.OPS)
def test_arity(op):
    ns = C.make_reference(op, "fp32")
    assert ns["arity"] == _ARITY[op]
    inputs = ns["get_inputs"](_SMALL[op], device="cpu", seed=0)
    assert len(inputs) == ns["arity"]


@pytest.mark.parametrize("op", C.OPS)
def test_baseline_matches_ref(op):
    """The F.* baseline (fp32 CPU) agrees with the fp32 oracle."""
    ns = C.make_reference(op, "fp32")
    inputs = ns["get_inputs"](_SMALL[op], device="cpu", seed=1)
    out = ns["baseline_fn"](*inputs)
    ref = ns["ref_fn"](*inputs)
    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("op", C.OPS)
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_seed_compiles_and_defines_entry(op, dtype):
    src = C.seed_source(op, dtype)
    compile(src, f"<{op}_{dtype}_seed>", "exec")   # valid Python (COMPILING seed)
    tree = ast.parse(src)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert op in funcs, f"{op} seed must define def {op}(...)"


@pytest.mark.parametrize("op", C.OPS)
def test_shapes_parse_roundtrip(op):
    ns = C.make_reference(op, "fp32")
    parse = ns["parse_shape"]
    sh = C.SHAPES[op]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert parse(s) == spec


@pytest.mark.parametrize("op", C.OPS)
def test_namespace_contract(op):
    ns = C.make_reference(op, "bf16")
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family"):
        assert k in ns
    assert ns["entry_name"] == op
    assert ns["dtype_name"] == "bf16"
    assert ns["family"] == f"breadth_{op}"
    assert ns[f"{op}_ref"] is ns["ref_fn"]


def test_adaptive_shapes_divisible():
    """The naive adaptive seed tiles fixed windows, so every catalog shape must be
    divisible by the fixed 7x7 output for the seed to equal the oracle."""
    for spec in [C.SHAPES["adaptive_avgpool2d"]["minimal"],
                 C.SHAPES["adaptive_avgpool2d"]["primary"],
                 *C.SHAPES["adaptive_avgpool2d"]["validation"]]:
        assert spec["H"] % C.ADAPTIVE_OUT == 0
        assert spec["W"] % C.ADAPTIVE_OUT == 0


def test_pool_shapes_even():
    """2x2 stride-2 pooling seeds assume even H,W (windows tile exactly)."""
    for op in ("maxpool2d", "avgpool2d"):
        for spec in [C.SHAPES[op]["minimal"], C.SHAPES[op]["primary"],
                     *C.SHAPES[op]["validation"]]:
            assert spec["H"] % 2 == 0 and spec["W"] % 2 == 0
