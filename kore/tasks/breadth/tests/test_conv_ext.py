"""CPU-only tests for the breadth convolution-frontier authoring engine (conv_ext).

Every ``ref_fn`` oracle is checked against an INDEPENDENT torch computation via a
DIFFERENT code path than the ``F.*`` baseline it wraps, so a wrong oracle is caught
with certainty:

  * conv2d / grouped / depthwise / dilated / strided / NHWC / fused-act / bn-fold
        -> im2col (F.unfold) + grouped einsum matmul (NOT F.conv2d),
  * depthwise-separable                 -> two unfold+einsum passes,
  * conv1d causal / non-causal          -> shifted-slice accumulate,
  * conv3d                              -> 3D shifted-slice accumulate,
  * transposed conv2d                   -> explicit scatter-add (deconv definition),
  * dgrad                               -> fold-adjoint of the im2col forward,
  * wgrad                               -> unfold + batched outer-product,
  * im2col / col2im                     -> explicit gather / overlap-add loops,
  * Winograd input/filter transforms    -> explicit B^T d B / G g G^T scalar loops.

Plus the ABI / arity / seed-compiles(+loads) / shapes-parse / dtype-preservation /
lazy-import / no-collision-with-conv.py contract. All fp32 on CPU (no GPU / triton
execution needed for the numeric checks).
"""

from __future__ import annotations

import ast
import math

import pytest
import torch
import torch.nn.functional as F

from kore.tasks.breadth import conv_ext as C

DTYPE_NAMES = ("bf16", "fp16", "fp8")
_TORCH_DT = {
    "bf16": torch.bfloat16, "fp16": torch.float16,
    "fp8": torch.float8_e4m3fn, "fp32": torch.float32,
}


def _close(a, b, atol=1e-4, rtol=1e-4):
    return torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- #
# tiny CPU shapes per op (derived from the module config; NOT the big catalog)
# --------------------------------------------------------------------------- #
def _small_shape(op: str) -> dict:
    cfg = C._CFG[op]
    fam = cfg["family"]
    if fam == "conv2d":
        gk = cfg["groups"]
        if gk == "depthwise":
            return {"N": 2, "C": 6, "H": 12, "W": 12}
        if gk != "one":
            g = int(gk)
            c = g * 2 if g * 2 >= 8 else 8
            return {"N": 2, "Cin": c, "H": 10, "W": 10, "Cout": c}
        return {"N": 2, "Cin": 4, "H": 12, "W": 12, "Cout": 6}
    if fam == "separable2d":
        return {"N": 2, "C": 4, "H": 12, "W": 12, "Cout": 6}
    if fam == "conv1d":
        return {"N": 2, "C": 5, "L": 20} if cfg["depthwise"] else \
            {"N": 2, "Cin": 4, "L": 20, "Cout": 6}
    if fam == "conv3d":
        return {"N": 1, "Cin": 2, "D": 6, "H": 8, "W": 8, "Cout": 4}
    if fam == "transpose2d":
        return {"N": 2, "Cin": 4, "H": 6, "W": 6, "Cout": 5}
    if fam in ("dgrad2d", "wgrad2d"):
        return {"N": 2, "Cin": 4, "H": 10, "W": 10, "Cout": 6}
    if fam in ("im2col", "col2im"):
        return {"N": 2, "C": 3, "H": 8, "W": 8}
    if fam == "winograd_input":
        return {"N": 3, "C": 5}
    if fam == "winograd_filter":
        return {"Cout": 4, "Cin": 3}
    raise AssertionError(op)


def _expected_arity(op: str) -> int:
    cfg = C._CFG[op]
    fam = cfg["family"]
    if fam == "conv2d":
        return 5 if cfg["bn"] else 3
    if fam == "separable2d":
        return 4
    if fam in ("conv1d", "conv3d", "transpose2d", "dgrad2d", "wgrad2d"):
        return 3
    return 1  # im2col / col2im / winograd


# --------------------------------------------------------------------------- #
# INDEPENDENT torch oracles (different code paths than the F.* baselines)
# --------------------------------------------------------------------------- #
def _ind_conv2d(cfg, inputs):
    x, w, b = inputs[0], inputs[1], inputs[2]
    rest = inputs[3:]
    K, S, D = cfg["K"], cfg["S"], cfg["D"]
    P = C._same_pad(K, D)
    layout, act, bn = cfg["layout"], cfg["act"], cfg["bn"]
    xf = x.float()
    if layout == "nhwc":
        xf = xf.permute(0, 3, 1, 2).contiguous()
    N, Cin, H, W = xf.shape
    Cout, Cin_g = w.shape[0], w.shape[1]
    G = Cin // Cin_g
    Cout_g = Cout // G
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    cols = F.unfold(xf, (K, K), dilation=D, padding=P, stride=S)          # (N, Cin*K*K, L)
    cols = cols.reshape(N, G, Cin_g, K * K, OH * OW)
    wr = w.float().reshape(G, Cout_g, Cin_g, K * K)
    out = torch.einsum("gock,ngckl->ngol", wr, cols).reshape(N, Cout, OH, OW)
    out = out + b.float().view(1, -1, 1, 1)
    if bn:
        out = out * rest[0].float().view(1, -1, 1, 1) + rest[1].float().view(1, -1, 1, 1)
    if act == "relu":
        out = F.relu(out)
    elif act == "silu":
        out = F.silu(out)
    if layout == "nhwc":
        out = out.permute(0, 2, 3, 1).contiguous()
    return out


def _ind_separable(cfg, inputs):
    x, dw, pw, b = inputs
    K = cfg["K"]
    P = C._same_pad(K, 1)
    xf = x.float()
    N, Cn, H, W = xf.shape
    cols = F.unfold(xf, (K, K), padding=P, stride=1).reshape(N, Cn, K * K, H * W)
    wr = dw.float().reshape(Cn, K * K)
    t = torch.einsum("ck,nckl->ncl", wr, cols)                            # depthwise (N,Cn,L)
    pwm = pw.float().reshape(pw.shape[0], Cn)
    out = torch.einsum("oc,ncl->nol", pwm, t).reshape(N, pw.shape[0], H, W)
    return out + b.float().view(1, -1, 1, 1)


def _ind_conv1d(cfg, inputs):
    x, w, b = inputs
    K, causal = cfg["K"], cfg["causal"]
    xf = x.float()
    N, Cin, Ll = xf.shape
    Cout, Cin_g = w.shape[0], w.shape[1]
    G = Cin // Cin_g
    Cout_g = Cout // G
    xp = F.pad(xf, (K - 1, 0)) if causal else F.pad(xf, ((K - 1) // 2, (K - 1) // 2))
    out = torch.zeros(N, Cout, Ll)
    for k in range(K):
        patch = xp[:, :, k:k + Ll].reshape(N, G, Cin_g, Ll)
        wk = w.float()[:, :, k].reshape(G, Cout_g, Cin_g)
        out = out + torch.einsum("goc,ngcl->ngol", wk, patch).reshape(N, Cout, Ll)
    return out + b.float().view(1, -1, 1)


def _ind_conv3d(cfg, inputs):
    x, w, b = inputs
    K, S = cfg["K"], cfg["S"]
    P = C._same_pad(K, 1)
    xf = x.float()
    N, Cin, Dd, H, W = xf.shape
    OD = (Dd + 2 * P - (K - 1) - 1) // S + 1
    OH = (H + 2 * P - (K - 1) - 1) // S + 1
    OW = (W + 2 * P - (K - 1) - 1) // S + 1
    xp = F.pad(xf, (P, P, P, P, P, P))
    out = torch.zeros(N, w.shape[0], OD, OH, OW)
    for kd in range(K):
        for kh in range(K):
            for kw in range(K):
                patch = xp[:, :, kd:kd + (OD - 1) * S + 1:S,
                           kh:kh + (OH - 1) * S + 1:S, kw:kw + (OW - 1) * S + 1:S]
                out = out + torch.einsum("ncdhw,oc->nodhw", patch, w.float()[:, :, kd, kh, kw])
    return out + b.float().view(1, -1, 1, 1, 1)


def _ind_transpose2d(cfg, inputs):
    x, w, b = inputs
    K, S, P = cfg["K"], cfg["S"], cfg["P"]
    xf = x.float()
    N, Cin, H, W = xf.shape
    Cout = w.shape[1]
    OH = (H - 1) * S - 2 * P + (K - 1) + 1
    OW = (W - 1) * S - 2 * P + (K - 1) + 1
    yf = torch.zeros(N, Cout, OH + 2 * P, OW + 2 * P)
    for kh in range(K):
        for kw in range(K):
            contrib = torch.einsum("ncij,co->noij", xf, w.float()[:, :, kh, kw])
            yf[:, :, kh:kh + (H - 1) * S + 1:S, kw:kw + (W - 1) * S + 1:S] += contrib
    y = yf[:, :, P:P + OH, P:P + OW]
    return y + b.float().view(1, -1, 1, 1)


def _ind_dgrad(cfg, inputs):
    x, grad_y, w = inputs
    K, S, D = cfg["K"], cfg["S"], cfg["D"]
    P = C._same_pad(K, D)
    N, Cin, H, W = x.shape
    Cout = w.shape[0]
    L = grad_y.shape[2] * grad_y.shape[3]
    gy_mat = grad_y.float().reshape(N, Cout, L)
    wmat = w.float().reshape(Cout, Cin * K * K)
    gcol = torch.matmul(wmat.t(), gy_mat)                                 # (N, Cin*K*K, L)
    return F.fold(gcol, (H, W), (K, K), dilation=D, padding=P, stride=S)


def _ind_wgrad(cfg, inputs):
    x, grad_y, w = inputs
    K, S, D = cfg["K"], cfg["S"], cfg["D"]
    P = C._same_pad(K, D)
    N, Cin, H, W = x.shape
    Cout = w.shape[0]
    OH, OW = grad_y.shape[2], grad_y.shape[3]
    cols = F.unfold(x.float(), (K, K), dilation=D, padding=P, stride=S)   # (N, Cin*K*K, L)
    gy_mat = grad_y.float().reshape(N, Cout, OH * OW)
    return torch.einsum("nol,nml->om", gy_mat, cols).reshape(Cout, Cin, K, K)


def _ind_im2col(cfg, inputs):
    x = inputs[0].float()
    K, S, D = cfg["K"], cfg["S"], cfg["D"]
    P = C._same_pad(K, D)
    N, Cn, H, W = x.shape
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    xp = F.pad(x, (P, P, P, P))
    out = torch.zeros(N, Cn * K * K, OH * OW)
    for c in range(Cn):
        for kh in range(K):
            for kw in range(K):
                row = (c * K + kh) * K + kw
                patch = xp[:, c, kh * D:kh * D + (OH - 1) * S + 1:S,
                           kw * D:kw * D + (OW - 1) * S + 1:S]
                out[:, row, :] = patch.reshape(N, OH * OW)
    return out


def _ind_col2im(cfg, inputs):
    cols = inputs[0].float()
    K, S, D = cfg["K"], cfg["S"], cfg["D"]
    P = C._same_pad(K, D)
    N, ROWS, L = cols.shape
    Cn = ROWS // (K * K)
    H = int(round(math.sqrt(L)))
    W = H
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    yp = torch.zeros(N, Cn, H + 2 * P, W + 2 * P)
    for c in range(Cn):
        for kh in range(K):
            for kw in range(K):
                row = (c * K + kh) * K + kw
                patch = cols[:, row, :].reshape(N, OH, OW)
                yp[:, c, kh * D:kh * D + (OH - 1) * S + 1:S,
                   kw * D:kw * D + (OW - 1) * S + 1:S] += patch
    return yp[:, :, P:P + H, P:P + W]


def _ind_wino_input(inputs):
    d = inputs[0].float()
    Bt = torch.tensor(C._WINO_BT)
    N, Cc = d.shape[0], d.shape[1]
    V = torch.zeros(N, Cc, 4, 4)
    for i in range(4):
        for j in range(4):
            s = torch.zeros(N, Cc)
            for k in range(4):
                for l in range(4):
                    coef = Bt[i, k] * Bt[j, l]
                    if coef != 0:
                        s = s + coef * d[:, :, k, l]
            V[:, :, i, j] = s
    return V


def _ind_wino_filter(inputs):
    g = inputs[0].float()
    G = torch.tensor(C._WINO_G)
    Co, Ci = g.shape[0], g.shape[1]
    U = torch.zeros(Co, Ci, 4, 4)
    for i in range(4):
        for j in range(4):
            s = torch.zeros(Co, Ci)
            for p in range(3):
                for q in range(3):
                    coef = G[i, p] * G[j, q]
                    if coef != 0:
                        s = s + coef * g[:, :, p, q]
            U[:, :, i, j] = s
    return U


def _independent(op, inputs):
    cfg = C._CFG[op]
    fam = cfg["family"]
    dispatch = {
        "conv2d": _ind_conv2d, "separable2d": _ind_separable, "conv1d": _ind_conv1d,
        "conv3d": _ind_conv3d, "transpose2d": _ind_transpose2d,
        "dgrad2d": _ind_dgrad, "wgrad2d": _ind_wgrad,
        "im2col": _ind_im2col, "col2im": _ind_col2im,
    }
    if fam in dispatch:
        return dispatch[fam](cfg, inputs)
    if fam == "winograd_input":
        return _ind_wino_input(inputs)
    if fam == "winograd_filter":
        return _ind_wino_filter(inputs)
    raise AssertionError(f"no independent oracle for {op!r}")


# --------------------------------------------------------------------------- #
# ABI surface
# --------------------------------------------------------------------------- #
def test_abi_surface():
    assert isinstance(C.OPS, list) and len(C.OPS) == 50
    assert len(set(C.OPS)) == len(C.OPS)                          # no duplicates
    assert set(C.OP_DTYPES) == set(C.OPS) == set(C.SHAPES) == set(C._CFG)
    assert callable(C.make_reference) and callable(C.seed_source)
    for attr in ("OPS", "OP_DTYPES", "SHAPES", "make_reference", "seed_source"):
        assert hasattr(C, attr)


def test_all_ops_cv_prefixed_and_disjoint_from_conv():
    from kore.tasks.breadth import conv as base
    assert all(op.startswith("cv_") for op in C.OPS)
    assert set(C.OPS).isdisjoint(set(base.OPS))                   # no genb_ collision


def test_op_dtypes_valid_and_fp8_only_for_pointwise():
    for op in C.OPS:
        dts = C.OP_DTYPES[op]
        assert dts and all(d in DTYPE_NAMES for d in dts)
        assert "bf16" in dts and "fp16" in dts
    for op in C.OPS:
        if "fp8" in C.OP_DTYPES[op]:
            assert op in C._FP8_OPS
    for op in C._FP8_OPS:
        assert "fp8" in C.OP_DTYPES[op]


def test_torch_imported_lazily():
    """Registry discovery must be GPU-free: torch imported INSIDE the paths, never at
    module scope (the `import torch` inside seed STRINGS is ignored - AST top level only)."""
    import inspect
    tree = ast.parse(inspect.getsource(C))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("torch") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("torch")


# --------------------------------------------------------------------------- #
# correctness: ref_fn oracle vs INDEPENDENT torch (the crux of the suite)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", C.OPS)
def test_ref_matches_independent(op):
    ns = C.make_reference(op, "fp32")
    inputs = ns["get_inputs"](_small_shape(op), device="cpu", seed=0)
    ref = ns["ref_fn"](*inputs)
    ind = _independent(op, inputs)
    assert ref.shape == ind.shape, f"{op}: {tuple(ref.shape)} vs {tuple(ind.shape)}"
    md = (ref.float() - ind.float()).abs().max().item()
    assert _close(ref, ind), f"{op}: max|diff|={md:.3e}"


@pytest.mark.parametrize("op", C.OPS)
def test_baseline_matches_ref(op):
    """The torch F.* production baseline (fp32 CPU) agrees with the fp32 oracle."""
    ns = C.make_reference(op, "fp32")
    inputs = ns["get_inputs"](_small_shape(op), device="cpu", seed=1)
    out = ns["baseline_fn"](*inputs)
    ref = ns["ref_fn"](*inputs)
    assert out.shape == ref.shape
    assert _close(out, ref)


# --------------------------------------------------------------------------- #
# arity / namespace / dtype / shapes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", C.OPS)
def test_arity(op):
    ns = C.make_reference(op, "fp32")
    assert ns["arity"] == _expected_arity(op)
    inputs = ns["get_inputs"](_small_shape(op), device="cpu", seed=0)
    assert len(inputs) == ns["arity"]


@pytest.mark.parametrize("op", C.OPS)
def test_namespace_contract(op):
    ns = C.make_reference(op, "bf16")
    for k in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn", "arity",
              "entry_name", "dtype_name", "family", "mutates_input"):
        assert k in ns, f"{op} missing ns key {k!r}"
    assert ns["entry_name"] == op
    assert ns["dtype_name"] == "bf16"
    assert ns["family"] == f"breadth_{op}"
    assert ns["mutates_input"] is False
    assert ns[f"{op}_ref"] is ns["ref_fn"]
    assert ns["parse_shape"] is C._parse_shape


@pytest.mark.parametrize("op", C.OPS)
def test_get_inputs_and_ref_dtype_preserved(op):
    for dt in C.OP_DTYPES[op]:
        ns = C.make_reference(op, dt)
        inputs = ns["get_inputs"](_small_shape(op), device="cpu", seed=0)
        want = _TORCH_DT[dt]
        for t in inputs:
            if torch.is_tensor(t) and t.is_floating_point():
                assert t.dtype == want, f"{op}/{dt}: input dtype {t.dtype}"
        out = ns["ref_fn"](*inputs)
        assert out.dtype == want, f"{op}/{dt}: output dtype {out.dtype}"


@pytest.mark.parametrize("op", C.OPS)
def test_shapes_parse_roundtrip(op):
    ns = C.make_reference(op, "fp32")
    parse = ns["parse_shape"]
    sh = C.SHAPES[op]
    assert {"minimal", "primary", "validation"} <= set(sh)
    assert isinstance(sh["validation"], list) and sh["validation"]
    for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
        s = ",".join(f"{k}={v}" for k, v in spec.items())
        assert parse(s) == spec


def test_grouped_and_depthwise_shapes_divisible():
    """Catalog channels must be divisible by the op's groups (grouped) / equal to C
    (depthwise) so weight (Cout, Cin//G, K, K) is well-formed at every shape."""
    for op in C.OPS:
        cfg = C._CFG[op]
        if cfg["family"] != "conv2d":
            continue
        gk = cfg["groups"]
        sh = C.SHAPES[op]
        for spec in [sh["minimal"], sh["primary"], *sh["validation"]]:
            if gk == "depthwise":
                assert "C" in spec
            elif gk != "one":
                g = int(gk)
                assert spec["Cin"] % g == 0 and spec["Cout"] % g == 0


def test_col2im_shapes_square():
    """The col2im seed recovers H=W=isqrt(L); every catalog shape must be square."""
    for op in C.OPS:
        if C._CFG[op]["family"] == "col2im":
            for spec in [C.SHAPES[op]["minimal"], C.SHAPES[op]["primary"],
                         *C.SHAPES[op]["validation"]]:
                assert spec["H"] == spec["W"]


# --------------------------------------------------------------------------- #
# seeds: parse + compile + define entry (CPU-safe) and load as a module (triton)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", C.OPS)
def test_seed_compiles_and_defines_entry(op):
    for dt in C.OP_DTYPES[op]:
        src = C.seed_source(op, dt)
        compile(src, f"<{op}:{dt}>", "exec")                     # syntactically compiles
        tree = ast.parse(src)
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert op in funcs, f"{op}/{dt} seed must define def {op}(...)"


@pytest.mark.parametrize("op", C.OPS)
def test_seed_loads_as_module(op, tmp_path):
    """Stronger 'compiles' check: import the seed from a real file so @triton.jit binds
    each kernel + the entry (no kernel is LAUNCHED, so this stays CPU-safe)."""
    import importlib.util

    pytest.importorskip("triton")
    dt = C.OP_DTYPES[op][0]
    path = tmp_path / f"seed_{op}.py"
    path.write_text(C.seed_source(op, dt))
    spec = importlib.util.spec_from_file_location(f"conv_ext_seed_{op}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(getattr(mod, op))


# --------------------------------------------------------------------------- #
# Winograd: exactness of the transform pair (oracle vs explicit + baseline)
# --------------------------------------------------------------------------- #
def test_winograd_input_transform_exact():
    ns = C.make_reference("cv_winograd_input_transform_f2x2", "fp32")
    d = ns["get_inputs"]({"N": 4, "C": 3}, device="cpu", seed=0)
    ref = ns["ref_fn"](*d)
    assert _close(ref, _ind_wino_input(d), atol=1e-5, rtol=1e-5)
    assert _close(ref, ns["baseline_fn"](*d), atol=1e-5, rtol=1e-5)


def test_winograd_filter_transform_exact():
    ns = C.make_reference("cv_winograd_filter_transform_f2x2", "fp32")
    g = ns["get_inputs"]({"Cout": 4, "Cin": 3}, device="cpu", seed=0)
    ref = ns["ref_fn"](*g)
    assert _close(ref, _ind_wino_filter(g), atol=1e-5, rtol=1e-5)
    assert _close(ref, ns["baseline_fn"](*g), atol=1e-5, rtol=1e-5)


def test_winograd_conv_identity():
    """A full Winograd F(2x2,3x3) conv assembled from the two transforms reproduces a
    direct 3x3 stride-1 conv - the transforms are a valid (exact) conv factorization."""
    torch.manual_seed(0)
    Cin, Cout = 2, 3
    x = torch.randn(1, Cin, 4, 4)                    # one 4x4 input tile -> 2x2 output
    g = torch.randn(Cout, Cin, 3, 3)
    Bt = torch.tensor(C._WINO_BT)
    Gm = torch.tensor(C._WINO_G)
    At = torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0]])   # 2x4 output transform
    V = torch.einsum("ik,nckl,jl->ncij", Bt, x, Bt)                     # (1,Cin,4,4)
    U = torch.einsum("ip,copq,jq->coij", Gm, g, Gm)                     # (Cout,Cin,4,4)
    M = torch.einsum("ncij,ocij->noij", V, U)                           # elementwise mul + sum Cin
    Y = torch.einsum("ai,noij,bj->noab", At, M, At)                     # (1,Cout,2,2)
    Y_direct = F.conv2d(x, g)                                           # valid 3x3 -> (1,Cout,2,2)
    assert _close(Y, Y_direct, atol=1e-4, rtol=1e-4)
