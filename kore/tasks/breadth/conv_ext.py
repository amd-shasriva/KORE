"""Breadth CONVOLUTION-FRONTIER task-authoring engine (torch-baselined).

Widens the KORE suite with the *hard* convolution operator families that dominate
vision / video / audio / sequence-model inference and training but that the
vendor-baselined core (norms / activations / GEMM / attention) and the sibling
``conv.py`` engine (which only covers a naive 3x3 / depthwise / dilated conv plus
pooling / resize) never covered. Every op here is a genuine "hard for a GPU"
kernel with real headroom over a naive direct impl: implicit-GEMM / im2col-fused
convs where data layout, channel blocking and im2col-free tiling separate a naive
loop from a cuDNN/MIOpen-optimal kernel. The policy learns to optimize the naive
seed toward implicit-GEMM / Winograd.

All op names are prefixed ``cv_`` and are DISJOINT from ``conv.py`` (which owns
``conv2d_nchw`` / ``depthwise_conv2d`` / ``dilated_conv2d`` / the pools / interp) -
the two engines coexist in the ``genb_`` registry with zero collision.

Contract mirrors ``conv.py`` / ``vendor_ops.py`` so the shared ``_genops`` driver +
the ``generate_breadth`` generator consume it unchanged:

    OPS / OP_DTYPES / SHAPES              module-level task catalog
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn fp32 oracle, baseline_fn torch F.*, arity, entry_name,
        dtype_name, family=f"breadth_{op}", mutates_input=False)
    seed_source(op, dtype) -> str         a naive, COMPILING, correct Triton seed
        (defines ``def <op>(*inputs)``) - the policy's starting point.

Op families (all ``cv_``-prefixed)
----------------------------------
  * conv2d NCHW  : 1x1 / 3x3 / 5x5 / 7x7 x stride{1,2} x dilation{1,2} (distinct
        pointwise / strided / dilated kernels).
  * conv2d NHWC  : the channels-last layout variants (layout is a first-class axis
        on MI350X - a distinct, hard kernel).
  * depthwise / grouped / depthwise-separable (dw 3x3 + pw 1x1 fused).
  * conv1d       : causal + non-causal short convs (audio / SSM), incl. the Mamba
        depthwise causal conv.
  * conv3d       : small 3x3x3 / 1x1x1 volumetric convs.
  * transposed conv2d (deconv / upsampling).
  * fused conv2d + bias + activation (relu / silu) and conv2d + batchnorm-fold
        (inference-time affine epilogue).
  * conv backward: dInput (dgrad) + dWeight (wgrad) for the core configs (the hard
        training kernels).
  * im2col / col2im primitives + a Winograd F(2x2, 3x3) transform pair (exact
        rational transforms, so the fp32 oracle is exact up to reassociation).

CORRECTNESS is paramount: every ``ref_fn`` computes in fp32 (via torch
``F.conv*`` / ``torch.nn.grad.conv2d_*`` / ``F.unfold`` / ``F.fold``) and casts back
to the task dtype, and is validated on CPU against an INDEPENDENT torch computation
via a DIFFERENT code path (im2col+matmul vs ``F.conv2d``, fold-adjoint dgrad,
scatter-add transposed conv, explicit Winograd matmul) at tight fp32 tolerance -
see tests/test_conv_ext.py. torch/triton are imported lazily inside the GPU paths
so registry discovery never needs a GPU.
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# --------------------------------------------------------------------------- #
# Winograd F(2x2, 3x3) transform matrices (exact rationals).
#   input transform : V = B^T d B          (B^T is 4x4, entries in {-1,0,1})
#   filter transform: U = G  g G^T         (G is 4x3, entries in {0,+-1/2,1})
# --------------------------------------------------------------------------- #
_WINO_BT = [
    [1.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, -1.0],
]
_WINO_G = [
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.0, 0.0, 1.0],
]


def _same_pad(K: int, D: int) -> int:
    """`same` padding for an odd kernel K at dilation D (integer for odd K)."""
    return D * (K - 1) // 2


# --------------------------------------------------------------------------- #
# task catalog: op -> config (single source of truth; OPS/SHAPES derive from it)
# --------------------------------------------------------------------------- #
def _c2(layout, K, S, D, groups, act, bn):
    return {"family": "conv2d", "layout": layout, "K": K, "S": S, "D": D,
            "groups": groups, "act": act, "bn": bn}


_CFG: dict[str, dict] = {
    # ---- conv2d NCHW: kernel {1,3,5,7} x stride {1,2} x dilation {1,2} -------
    "cv_conv2d_1x1_s1":        _c2("nchw", 1, 1, 1, "one", "none", False),
    "cv_conv2d_1x1_s2":        _c2("nchw", 1, 2, 1, "one", "none", False),
    "cv_conv2d_3x3_s1_d1":     _c2("nchw", 3, 1, 1, "one", "none", False),
    "cv_conv2d_3x3_s2_d1":     _c2("nchw", 3, 2, 1, "one", "none", False),
    "cv_conv2d_3x3_s1_d2":     _c2("nchw", 3, 1, 2, "one", "none", False),
    "cv_conv2d_5x5_s1_d1":     _c2("nchw", 5, 1, 1, "one", "none", False),
    "cv_conv2d_5x5_s2_d1":     _c2("nchw", 5, 2, 1, "one", "none", False),
    "cv_conv2d_5x5_s1_d2":     _c2("nchw", 5, 1, 2, "one", "none", False),
    "cv_conv2d_7x7_s1_d1":     _c2("nchw", 7, 1, 1, "one", "none", False),
    "cv_conv2d_7x7_s2_d1":     _c2("nchw", 7, 2, 1, "one", "none", False),
    # ---- conv2d NHWC (channels-last layout) ---------------------------------
    "cv_conv2d_nhwc_1x1_s1":   _c2("nhwc", 1, 1, 1, "one", "none", False),
    "cv_conv2d_nhwc_3x3_s1_d1": _c2("nhwc", 3, 1, 1, "one", "none", False),
    "cv_conv2d_nhwc_3x3_s2_d1": _c2("nhwc", 3, 2, 1, "one", "none", False),
    "cv_conv2d_nhwc_3x3_s1_d2": _c2("nhwc", 3, 1, 2, "one", "none", False),
    "cv_conv2d_nhwc_5x5_s1_d1": _c2("nhwc", 5, 1, 1, "one", "none", False),
    # ---- depthwise / grouped ------------------------------------------------
    "cv_depthwise_conv2d_3x3_s1": _c2("nchw", 3, 1, 1, "depthwise", "none", False),
    "cv_depthwise_conv2d_3x3_s2": _c2("nchw", 3, 2, 1, "depthwise", "none", False),
    "cv_depthwise_conv2d_5x5_s1": _c2("nchw", 5, 1, 1, "depthwise", "none", False),
    "cv_depthwise_conv2d_7x7_s1": _c2("nchw", 7, 1, 1, "depthwise", "none", False),
    "cv_grouped_conv2d_3x3_g2":  _c2("nchw", 3, 1, 1, 2, "none", False),
    "cv_grouped_conv2d_3x3_g4":  _c2("nchw", 3, 1, 1, 4, "none", False),
    "cv_grouped_conv2d_3x3_g32": _c2("nchw", 3, 1, 1, 32, "none", False),
    # ---- fused bias + activation / batchnorm-fold ---------------------------
    "cv_conv2d_3x3_relu":      _c2("nchw", 3, 1, 1, "one", "relu", False),
    "cv_conv2d_3x3_silu":      _c2("nchw", 3, 1, 1, "one", "silu", False),
    "cv_conv2d_1x1_relu":      _c2("nchw", 1, 1, 1, "one", "relu", False),
    "cv_conv2d_1x1_silu":      _c2("nchw", 1, 1, 1, "one", "silu", False),
    "cv_conv2d_bn_fold_3x3":   _c2("nchw", 3, 1, 1, "one", "none", True),
    "cv_conv2d_bn_relu_3x3":   _c2("nchw", 3, 1, 1, "one", "relu", True),
    # ---- depthwise-separable (dw 3x3 + pw 1x1 fused) ------------------------
    "cv_separable_conv2d_3x3": {"family": "separable2d", "K": 3},
    # ---- conv1d (causal + non-causal short convs) ---------------------------
    "cv_conv1d_k3":            {"family": "conv1d", "K": 3, "causal": False, "depthwise": False},
    "cv_conv1d_k5":            {"family": "conv1d", "K": 5, "causal": False, "depthwise": False},
    "cv_conv1d_k7":            {"family": "conv1d", "K": 7, "causal": False, "depthwise": False},
    "cv_conv1d_causal_k3":     {"family": "conv1d", "K": 3, "causal": True, "depthwise": False},
    "cv_conv1d_causal_k5":     {"family": "conv1d", "K": 5, "causal": True, "depthwise": False},
    "cv_conv1d_causal_k7":     {"family": "conv1d", "K": 7, "causal": True, "depthwise": False},
    "cv_dw_conv1d_causal_k4":  {"family": "conv1d", "K": 4, "causal": True, "depthwise": True},
    # ---- conv3d (small volumetric) ------------------------------------------
    "cv_conv3d_3x3x3_s1":      {"family": "conv3d", "K": 3, "S": 1},
    "cv_conv3d_3x3x3_s2":      {"family": "conv3d", "K": 3, "S": 2},
    "cv_conv3d_1x1x1_s1":      {"family": "conv3d", "K": 1, "S": 1},
    # ---- transposed conv2d (deconv / upsample) ------------------------------
    "cv_conv_transpose2d_k2_s2": {"family": "transpose2d", "K": 2, "S": 2, "P": 0},
    "cv_conv_transpose2d_k4_s2": {"family": "transpose2d", "K": 4, "S": 2, "P": 1},
    # ---- conv backward (the hard training kernels) --------------------------
    "cv_conv2d_dgrad_3x3":     {"family": "dgrad2d", "K": 3, "S": 1, "D": 1},
    "cv_conv2d_wgrad_3x3":     {"family": "wgrad2d", "K": 3, "S": 1, "D": 1},
    "cv_conv2d_dgrad_1x1":     {"family": "dgrad2d", "K": 1, "S": 1, "D": 1},
    "cv_conv2d_wgrad_1x1":     {"family": "wgrad2d", "K": 1, "S": 1, "D": 1},
    # ---- im2col / col2im primitives -----------------------------------------
    "cv_im2col_3x3_s1":        {"family": "im2col", "K": 3, "S": 1, "D": 1},
    "cv_im2col_3x3_s2":        {"family": "im2col", "K": 3, "S": 2, "D": 1},
    "cv_col2im_3x3_s1":        {"family": "col2im", "K": 3, "S": 1, "D": 1},
    # ---- Winograd F(2x2, 3x3) transform pair --------------------------------
    "cv_winograd_input_transform_f2x2":  {"family": "winograd_input"},
    "cv_winograd_filter_transform_f2x2": {"family": "winograd_filter"},
}

OPS: list[str] = list(_CFG)

# bf16/fp16 sweep (matches the vendor + sibling-breadth default); the fp32 oracle
# casts back. fp8 (OCP e4m3fn / CDNA4) added for the pointwise (GEMM-like) 1x1
# convs where fp8 conv is realistic on MI350X.
_FP8_OPS = ("cv_conv2d_1x1_s1", "cv_conv2d_nhwc_1x1_s1")
OP_DTYPES: dict[str, list[str]] = {
    op: (["bf16", "fp16", "fp8"] if op in _FP8_OPS else ["bf16", "fp16"])
    for op in OPS
}


# --------------------------------------------------------------------------- #
# realistic shape catalogs (N{1,8}, C{64,256,512}, spatial{14..224} / seq{1k..4k};
# validation carries a NON-power-of-2 tail). Never executed on CPU (tests use
# tiny shapes); only round-tripped through parse_shape + serialized to task.yaml.
# --------------------------------------------------------------------------- #
_S_CONV2D = {
    "minimal": {"N": 1, "Cin": 64, "H": 32, "W": 32, "Cout": 64},
    "primary": {"N": 8, "Cin": 128, "H": 56, "W": 56, "Cout": 128},
    "validation": [
        {"N": 4, "Cin": 256, "H": 28, "W": 28, "Cout": 256},
        {"N": 1, "Cin": 512, "H": 14, "W": 14, "Cout": 512},
        {"N": 2, "Cin": 192, "H": 30, "W": 30, "Cout": 96},   # non-pow2 tail
    ],
}
_S_GROUPED = {  # Cin==Cout, both divisible by 32 (covers g2/g4/g32)
    "minimal": {"N": 1, "Cin": 64, "H": 32, "W": 32, "Cout": 64},
    "primary": {"N": 8, "Cin": 128, "H": 56, "W": 56, "Cout": 128},
    "validation": [
        {"N": 4, "Cin": 256, "H": 28, "W": 28, "Cout": 256},
        {"N": 1, "Cin": 512, "H": 14, "W": 14, "Cout": 512},
        {"N": 2, "Cin": 96, "H": 30, "W": 30, "Cout": 96},    # non-pow2 tail
    ],
}
_S_DW = {
    "minimal": {"N": 1, "C": 64, "H": 32, "W": 32},
    "primary": {"N": 8, "C": 128, "H": 56, "W": 56},
    "validation": [
        {"N": 4, "C": 256, "H": 28, "W": 28},
        {"N": 1, "C": 512, "H": 14, "W": 14},
        {"N": 2, "C": 96, "H": 30, "W": 30},                  # non-pow2 tail
    ],
}
_S_SEP = {
    "minimal": {"N": 1, "C": 64, "H": 32, "W": 32, "Cout": 128},
    "primary": {"N": 8, "C": 128, "H": 56, "W": 56, "Cout": 256},
    "validation": [
        {"N": 4, "C": 256, "H": 28, "W": 28, "Cout": 256},
        {"N": 1, "C": 512, "H": 14, "W": 14, "Cout": 512},
        {"N": 2, "C": 96, "H": 30, "W": 30, "Cout": 192},     # non-pow2 tail
    ],
}
_S_CONV1D = {
    "minimal": {"N": 1, "Cin": 64, "L": 2048, "Cout": 64},
    "primary": {"N": 8, "Cin": 128, "L": 2048, "Cout": 128},
    "validation": [
        {"N": 4, "Cin": 256, "L": 4096, "Cout": 256},
        {"N": 1, "Cin": 512, "L": 1024, "Cout": 512},
        {"N": 2, "Cin": 96, "L": 3000, "Cout": 96},           # non-pow2 tail
    ],
}
_S_DWCONV1D = {
    "minimal": {"N": 1, "C": 64, "L": 2048},
    "primary": {"N": 8, "C": 256, "L": 2048},
    "validation": [
        {"N": 4, "C": 512, "L": 4096},
        {"N": 1, "C": 128, "L": 1024},
        {"N": 2, "C": 96, "L": 3000},                         # non-pow2 tail
    ],
}
_S_CONV3D = {
    "minimal": {"N": 1, "Cin": 16, "D": 8, "H": 16, "W": 16, "Cout": 16},
    "primary": {"N": 2, "Cin": 32, "D": 8, "H": 28, "W": 28, "Cout": 32},
    "validation": [
        {"N": 1, "Cin": 64, "D": 16, "H": 28, "W": 28, "Cout": 64},
        {"N": 2, "Cin": 32, "D": 4, "H": 56, "W": 56, "Cout": 32},
        {"N": 1, "Cin": 48, "D": 6, "H": 30, "W": 30, "Cout": 48},  # non-pow2 tail
    ],
}
_S_CONVT = {
    "minimal": {"N": 1, "Cin": 64, "H": 16, "W": 16, "Cout": 64},
    "primary": {"N": 8, "Cin": 128, "H": 28, "W": 28, "Cout": 64},
    "validation": [
        {"N": 4, "Cin": 256, "H": 14, "W": 14, "Cout": 128},
        {"N": 1, "Cin": 512, "H": 8, "W": 8, "Cout": 256},
        {"N": 2, "Cin": 96, "H": 30, "W": 30, "Cout": 48},    # non-pow2 tail
    ],
}
_S_BWD = {
    "minimal": {"N": 1, "Cin": 64, "H": 32, "W": 32, "Cout": 64},
    "primary": {"N": 8, "Cin": 128, "H": 28, "W": 28, "Cout": 128},
    "validation": [
        {"N": 4, "Cin": 128, "H": 28, "W": 28, "Cout": 128},
        {"N": 1, "Cin": 256, "H": 14, "W": 14, "Cout": 256},
        {"N": 2, "Cin": 96, "H": 30, "W": 30, "Cout": 96},    # non-pow2 tail
    ],
}
_S_IM2COL = {
    "minimal": {"N": 1, "C": 64, "H": 32, "W": 32},
    "primary": {"N": 8, "C": 128, "H": 56, "W": 56},
    "validation": [
        {"N": 4, "C": 256, "H": 28, "W": 28},
        {"N": 1, "C": 512, "H": 14, "W": 14},
        {"N": 2, "C": 96, "H": 30, "W": 30},                  # non-pow2 tail
    ],
}
_S_WINO_IN = {   # N == number of 4x4 tiles
    "minimal": {"N": 64, "C": 64},
    "primary": {"N": 512, "C": 128},
    "validation": [
        {"N": 256, "C": 256},
        {"N": 128, "C": 512},
        {"N": 196, "C": 96},                                  # non-pow2 tail
    ],
}
_S_WINO_FILT = {
    "minimal": {"Cout": 64, "Cin": 64},
    "primary": {"Cout": 256, "Cin": 128},
    "validation": [
        {"Cout": 512, "Cin": 256},
        {"Cout": 128, "Cin": 512},
        {"Cout": 96, "Cin": 192},                             # non-pow2 tail
    ],
}


def _shapes_for(op: str) -> dict:
    cfg = _CFG[op]
    fam = cfg["family"]
    if fam == "conv2d":
        if cfg["groups"] == "depthwise":
            return _S_DW
        if cfg["groups"] != "one":
            return _S_GROUPED
        return _S_CONV2D
    if fam == "separable2d":
        return _S_SEP
    if fam == "conv1d":
        return _S_DWCONV1D if cfg["depthwise"] else _S_CONV1D
    if fam == "conv3d":
        return _S_CONV3D
    if fam == "transpose2d":
        return _S_CONVT
    if fam in ("dgrad2d", "wgrad2d"):
        return _S_BWD
    if fam in ("im2col", "col2im"):
        return _S_IM2COL
    if fam == "winograd_input":
        return _S_WINO_IN
    if fam == "winograd_filter":
        return _S_WINO_FILT
    raise ValueError(f"no shapes for {op!r}")


SHAPES: dict[str, dict] = {op: _shapes_for(op) for op in OPS}


# --------------------------------------------------------------------------- #
# reference.py namespace (torch fp32 oracle + torch F.* production baseline)
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch
    import torch.nn.functional as F

    cfg = _CFG[op]
    fam = cfg["family"]
    tdt = getattr(torch, DTYPES[dtype][0])
    is_fp8 = dtype == "fp8"

    def _randn(shape, device, seed, scale=1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device,
                            dtype=torch.float32) * scale).to(tdt)

    if fam == "conv2d":
        K, S, D = cfg["K"], cfg["S"], cfg["D"]
        P = _same_pad(K, D)
        layout, act, bn, gk = cfg["layout"], cfg["act"], cfg["bn"], cfg["groups"]

        def _groups(Cin, Cout):
            if gk == "one":
                return 1
            if gk == "depthwise":
                return Cin
            return int(gk)

        def get_inputs(shape, device="cuda", seed=0):
            N, H, W = shape["N"], shape["H"], shape["W"]
            if gk == "depthwise":
                Cin = Cout = shape["C"]
            else:
                Cin, Cout = shape["Cin"], shape["Cout"]
            G = _groups(Cin, Cout)
            fan_in = (Cin // G) * K * K
            xshape = (N, Cin, H, W) if layout == "nchw" else (N, H, W, Cin)
            x = _randn(xshape, device, seed)
            w = _randn((Cout, Cin // G, K, K), device, seed + 1, scale=1.0 / (fan_in ** 0.5))
            b = _randn((Cout,), device, seed + 2, scale=0.1)
            if bn:
                g = torch.Generator(device=device).manual_seed(seed + 3)
                scale = (1.0 + 0.25 * torch.randn((Cout,), generator=g, device=device,
                                                  dtype=torch.float32)).to(tdt)
                shift = _randn((Cout,), device, seed + 4, scale=0.1)
                return (x, w, b, scale, shift)
            return (x, w, b)

        def _epilogue(y, rest, use_float):
            if bn:
                sc = rest[0].float() if use_float else rest[0]
                sf = rest[1].float() if use_float else rest[1]
                y = y * sc.view(1, -1, 1, 1) + sf.view(1, -1, 1, 1)
            if act == "relu":
                y = F.relu(y)
            elif act == "silu":
                y = F.silu(y)
            return y

        def ref_fn(x, w, b, *rest):
            if layout == "nchw":
                Cin = x.shape[1]
                xf = x.float()
            else:
                Cin = x.shape[3]
                xf = x.float().permute(0, 3, 1, 2).contiguous()
            G = Cin // w.shape[1]
            y = F.conv2d(xf, w.float(), b.float(), stride=S, padding=P, dilation=D, groups=G)
            y = _epilogue(y, rest, use_float=True)
            if layout == "nhwc":
                y = y.permute(0, 2, 3, 1).contiguous()
            return y.to(x.dtype)

        def baseline_fn(x, w, b, *rest):
            if is_fp8:                       # no native fp8 conv -> dequantize
                return ref_fn(x, w, b, *rest)
            if layout == "nchw":
                Cin = x.shape[1]
                xf = x
            else:
                Cin = x.shape[3]
                xf = x.permute(0, 3, 1, 2).contiguous()
            G = Cin // w.shape[1]
            y = F.conv2d(xf, w, b, stride=S, padding=P, dilation=D, groups=G)
            y = _epilogue(y, rest, use_float=False)
            if layout == "nhwc":
                y = y.permute(0, 2, 3, 1).contiguous()
            return y

        arity = 5 if bn else 3

    elif fam == "separable2d":
        K = cfg["K"]
        P = _same_pad(K, 1)

        def get_inputs(shape, device="cuda", seed=0):
            N, C, H, W, Cout = shape["N"], shape["C"], shape["H"], shape["W"], shape["Cout"]
            x = _randn((N, C, H, W), device, seed)
            dw = _randn((C, 1, K, K), device, seed + 1, scale=1.0 / (K * K) ** 0.5)
            pw = _randn((Cout, C, 1, 1), device, seed + 2, scale=1.0 / C ** 0.5)
            b = _randn((Cout,), device, seed + 3, scale=0.1)
            return (x, dw, pw, b)

        def ref_fn(x, dw, pw, b):
            C = x.shape[1]
            y = F.conv2d(x.float(), dw.float(), None, stride=1, padding=P, groups=C)
            y = F.conv2d(y, pw.float(), b.float(), stride=1, padding=0)
            return y.to(x.dtype)

        def baseline_fn(x, dw, pw, b):
            C = x.shape[1]
            y = F.conv2d(x, dw, None, stride=1, padding=P, groups=C)
            return F.conv2d(y, pw, b, stride=1, padding=0)

        arity = 4

    elif fam == "conv1d":
        K, causal, depthwise = cfg["K"], cfg["causal"], cfg["depthwise"]

        def get_inputs(shape, device="cuda", seed=0):
            N, L = shape["N"], shape["L"]
            if depthwise:
                Cin = Cout = shape["C"]
                w = _randn((Cin, 1, K), device, seed + 1, scale=1.0 / K ** 0.5)
            else:
                Cin, Cout = shape["Cin"], shape["Cout"]
                w = _randn((Cout, Cin, K), device, seed + 1, scale=1.0 / (Cin * K) ** 0.5)
            x = _randn((N, Cin, L), device, seed)
            b = _randn((Cout,), device, seed + 2, scale=0.1)
            return (x, w, b)

        def ref_fn(x, w, b):
            G = x.shape[1] // w.shape[1]
            xf = x.float()
            if causal:
                xf = F.pad(xf, (K - 1, 0))
                pad = 0
            else:
                pad = (K - 1) // 2
            y = F.conv1d(xf, w.float(), b.float(), stride=1, padding=pad, groups=G)
            return y.to(x.dtype)

        def baseline_fn(x, w, b):
            G = x.shape[1] // w.shape[1]
            xf = x
            if causal:
                xf = F.pad(xf, (K - 1, 0))
                pad = 0
            else:
                pad = (K - 1) // 2
            return F.conv1d(xf, w, b, stride=1, padding=pad, groups=G)

        arity = 3

    elif fam == "conv3d":
        K, S = cfg["K"], cfg["S"]
        P = _same_pad(K, 1)

        def get_inputs(shape, device="cuda", seed=0):
            N, Cin, Dd, H, W, Cout = (shape["N"], shape["Cin"], shape["D"],
                                      shape["H"], shape["W"], shape["Cout"])
            fan_in = Cin * K * K * K
            x = _randn((N, Cin, Dd, H, W), device, seed)
            w = _randn((Cout, Cin, K, K, K), device, seed + 1, scale=1.0 / fan_in ** 0.5)
            b = _randn((Cout,), device, seed + 2, scale=0.1)
            return (x, w, b)

        def ref_fn(x, w, b):
            return F.conv3d(x.float(), w.float(), b.float(), stride=S, padding=P).to(x.dtype)

        def baseline_fn(x, w, b):
            return F.conv3d(x, w, b, stride=S, padding=P)

        arity = 3

    elif fam == "transpose2d":
        K, S, P = cfg["K"], cfg["S"], cfg["P"]

        def get_inputs(shape, device="cuda", seed=0):
            N, Cin, H, W, Cout = (shape["N"], shape["Cin"], shape["H"],
                                  shape["W"], shape["Cout"])
            fan_in = Cout * K * K
            x = _randn((N, Cin, H, W), device, seed)
            w = _randn((Cin, Cout, K, K), device, seed + 1, scale=1.0 / fan_in ** 0.5)
            b = _randn((Cout,), device, seed + 2, scale=0.1)
            return (x, w, b)

        def ref_fn(x, w, b):
            return F.conv_transpose2d(x.float(), w.float(), b.float(),
                                      stride=S, padding=P).to(x.dtype)

        def baseline_fn(x, w, b):
            return F.conv_transpose2d(x, w, b, stride=S, padding=P)

        arity = 3

    elif fam in ("dgrad2d", "wgrad2d"):
        K, S, D = cfg["K"], cfg["S"], cfg["D"]
        P = _same_pad(K, D)

        def _out_hw(H, W):
            OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
            OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
            return OH, OW

        def get_inputs(shape, device="cuda", seed=0):
            N, Cin, H, W, Cout = (shape["N"], shape["Cin"], shape["H"],
                                  shape["W"], shape["Cout"])
            OH, OW = _out_hw(H, W)
            x = _randn((N, Cin, H, W), device, seed)
            w = _randn((Cout, Cin, K, K), device, seed + 1, scale=1.0 / (Cin * K * K) ** 0.5)
            grad_y = _randn((N, Cout, OH, OW), device, seed + 2)
            return (x, grad_y, w)

        if fam == "dgrad2d":
            def ref_fn(x, grad_y, w):
                gx = torch.nn.grad.conv2d_input(
                    tuple(x.shape), w.float(), grad_y.float(),
                    stride=S, padding=P, dilation=D, groups=1)
                return gx.to(x.dtype)

            def baseline_fn(x, grad_y, w):
                return torch.nn.grad.conv2d_input(
                    tuple(x.shape), w, grad_y, stride=S, padding=P, dilation=D, groups=1)
        else:
            def ref_fn(x, grad_y, w):
                gw = torch.nn.grad.conv2d_weight(
                    x.float(), tuple(w.shape), grad_y.float(),
                    stride=S, padding=P, dilation=D, groups=1)
                return gw.to(x.dtype)

            def baseline_fn(x, grad_y, w):
                return torch.nn.grad.conv2d_weight(
                    x, tuple(w.shape), grad_y, stride=S, padding=P, dilation=D, groups=1)

        arity = 3

    elif fam == "im2col":
        K, S, D = cfg["K"], cfg["S"], cfg["D"]
        P = _same_pad(K, D)

        def get_inputs(shape, device="cuda", seed=0):
            N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
            return (_randn((N, C, H, W), device, seed),)

        def ref_fn(x):
            cols = F.unfold(x.float(), (K, K), dilation=D, padding=P, stride=S)
            return cols.to(x.dtype)

        def baseline_fn(x):
            return F.unfold(x, (K, K), dilation=D, padding=P, stride=S)

        arity = 1

    elif fam == "col2im":
        K, S, D = cfg["K"], cfg["S"], cfg["D"]
        P = _same_pad(K, D)

        def get_inputs(shape, device="cuda", seed=0):
            N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
            OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
            OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
            return (_randn((N, C * K * K, OH * OW), device, seed),)

        def _hw(cols):
            L = cols.shape[2]
            H = int(round(L ** 0.5))
            return H, H

        def ref_fn(cols):
            H, W = _hw(cols)
            y = F.fold(cols.float(), (H, W), (K, K), dilation=D, padding=P, stride=S)
            return y.to(cols.dtype)

        def baseline_fn(cols):
            H, W = _hw(cols)
            return F.fold(cols, (H, W), (K, K), dilation=D, padding=P, stride=S)

        arity = 1

    elif fam == "winograd_input":
        Bt = torch.tensor(_WINO_BT, dtype=torch.float32)

        def get_inputs(shape, device="cuda", seed=0):
            N, C = shape["N"], shape["C"]
            return (_randn((N, C, 4, 4), device, seed),)

        def ref_fn(d):
            B = Bt.to(d.device)
            V = torch.einsum("ik,nckl,jl->ncij", B, d.float(), B)
            return V.to(d.dtype)

        def baseline_fn(d):
            B = Bt.to(d.device)
            tmp = torch.matmul(B, d.float())
            V = torch.matmul(tmp, B.transpose(0, 1))
            return V.to(d.dtype)

        arity = 1

    elif fam == "winograd_filter":
        G = torch.tensor(_WINO_G, dtype=torch.float32)

        def get_inputs(shape, device="cuda", seed=0):
            Cout, Cin = shape["Cout"], shape["Cin"]
            return (_randn((Cout, Cin, 3, 3), device, seed, scale=1.0 / 3.0),)

        def ref_fn(g):
            Gm = G.to(g.device)
            U = torch.einsum("ip,copq,jq->coij", Gm, g.float(), Gm)
            return U.to(g.dtype)

        def baseline_fn(g):
            Gm = G.to(g.device)
            tmp = torch.matmul(Gm, g.float())
            U = torch.matmul(tmp, Gm.transpose(0, 1))
            return U.to(g.dtype)

        arity = 1

    else:
        raise ValueError(f"unknown conv_ext family {fam!r} for op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}", "mutates_input": False}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive (correct, compiling) Triton seeds - the policy's starting point.
# --------------------------------------------------------------------------- #
_IMPORTS = "from __future__ import annotations\nimport torch, triton, triton.language as tl\n"
_ACT_BLOCK = {
    "none": "",
    "relu": "    acc = tl.maximum(acc, 0.0)\n",
    "silu": "    acc = acc * tl.sigmoid(acc)\n",
}


def _doc(op, dtype, text):
    return f'"""GENERATED breadth {op} seed ({dtype}). {text}"""\n'


_CONV2D_TMPL = '''{doc}{imports}

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, y_ptr, Cin, H, W, Cout, OH, OW, GIN, GOUT,
                 sxn, sxc, sxh, sxw, swo, swc, swh, sww, syn, syc, syh, syw{bnparams},
                 STRIDE: tl.constexpr, PAD: tl.constexpr, DIL: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    co = tmp % Cout
    n = tmp // Cout
    g = co // GOUT
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for ci in range(0, GIN):
        cin = g * GIN + ci
        for kh in range(0, KH):
            ih = oh * STRIDE - PAD + kh * DIL
            h_ok = (ih >= 0) & (ih < H)
            for kw in range(0, KW):
                iw = ow * STRIDE - PAD + kw * DIL
                m = ow_mask & h_ok & (iw >= 0) & (iw < W)
                xv = tl.load(x_ptr + n * sxn + cin * sxc + ih * sxh + iw * sxw,
                             mask=m, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + co * swo + ci * swc + kh * swh + kw * sww).to(tl.float32)
                acc += xv * wv
    acc += tl.load(b_ptr + co).to(tl.float32)
{bnblock}{actblock}    y_off = n * syn + co * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, acc.to({tldt}), mask=ow_mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor{extra_args}) -> torch.Tensor:
    {unpack}
    Cout = weight.shape[0]
    GIN = weight.shape[1]
    GROUPS = Cin // GIN
    GOUT = Cout // GROUPS
    KH, KW = weight.shape[2], weight.shape[3]
    STRIDE, PAD, DIL = {S}, {P}, {D}
    OH = (H + 2 * PAD - DIL * (KH - 1) - 1) // STRIDE + 1
    OW = (W + 2 * PAD - DIL * (KW - 1) - 1) // STRIDE + 1
    {yalloc}
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * Cout * OH,)
    _{op}_kernel[grid](x, weight, bias, y, Cin, H, W, Cout, OH, OW, GIN, GOUT,
                       {xstr},
                       weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
                       {ystr}{bncall},
                       STRIDE=STRIDE, PAD=PAD, DIL=DIL, KH=KH, KW=KW,
                       BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
'''


def _conv2d_seed(op, cfg, dtype, tldt):
    K, S, D = cfg["K"], cfg["S"], cfg["D"]
    P = _same_pad(K, D)
    bn, act, layout = cfg["bn"], cfg["act"], cfg["layout"]
    if bn:
        bnparams = ", scale_ptr, shift_ptr"
        bnblock = ("    sc = tl.load(scale_ptr + co).to(tl.float32)\n"
                   "    sf = tl.load(shift_ptr + co).to(tl.float32)\n"
                   "    acc = acc * sc + sf\n")
        extra_args = ", scale: torch.Tensor, shift: torch.Tensor"
        bncall = ", scale, shift"
    else:
        bnparams = bnblock = extra_args = bncall = ""
    if layout == "nchw":
        unpack = "N, Cin, H, W = x.shape"
        xstr = "x.stride(0), x.stride(1), x.stride(2), x.stride(3)"
        yalloc = "y = torch.empty((N, Cout, OH, OW), device=x.device, dtype=x.dtype)"
        ystr = "y.stride(0), y.stride(1), y.stride(2), y.stride(3)"
    else:
        unpack = "N, H, W, Cin = x.shape"
        xstr = "x.stride(0), x.stride(3), x.stride(1), x.stride(2)"
        yalloc = "y = torch.empty((N, OH, OW, Cout), device=x.device, dtype=x.dtype)"
        ystr = "y.stride(0), y.stride(3), y.stride(1), y.stride(2)"
    text = (f"Naive direct grouped conv2d ({layout}, K={K} S={S} D={D}) vs torch F.conv2d; "
            f"one program per (n, cout, oh), fp32 accumulate over (cin, kh, kw), output "
            f"width vectorized. Implicit-GEMM / channel-blocking headroom. {tldt} store.")
    return _CONV2D_TMPL.format(
        doc=_doc(op, dtype, text), imports=_IMPORTS, op=op, tldt=tldt,
        S=S, P=P, D=D, bnparams=bnparams, bnblock=bnblock, actblock=_ACT_BLOCK[act],
        extra_args=extra_args, bncall=bncall, unpack=unpack, xstr=xstr,
        yalloc=yalloc, ystr=ystr)


_SEPARABLE_TMPL = '''{doc}{imports}

@triton.jit
def _{op}_dw_kernel(x_ptr, w_ptr, t_ptr, C, H, W,
                    sxn, sxc, sxh, sxw, swo, swh, sww, stn, stc, sth, stw,
                    PAD: tl.constexpr, K: tl.constexpr, BLOCK_W: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % H
    tmp = pid // H
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_W)
    ow_mask = ow < W
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for kh in range(0, K):
        ih = oh - PAD + kh
        h_ok = (ih >= 0) & (ih < H)
        for kw in range(0, K):
            iw = ow - PAD + kw
            m = ow_mask & h_ok & (iw >= 0) & (iw < W)
            xv = tl.load(x_ptr + n * sxn + c * sxc + ih * sxh + iw * sxw,
                         mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + c * swo + kh * swh + kw * sww).to(tl.float32)
            acc += xv * wv
    tl.store(t_ptr + n * stn + c * stc + oh * sth + ow * stw, acc.to({tldt}), mask=ow_mask)


@triton.jit
def _{op}_pw_kernel(t_ptr, w_ptr, b_ptr, y_ptr, C, Cout, H, W,
                    stn, stc, sth, stw, swo, swc, syn, syc, syh, syw,
                    BLOCK_W: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % H
    tmp = pid // H
    co = tmp % Cout
    n = tmp // Cout
    ow = tl.arange(0, BLOCK_W)
    ow_mask = ow < W
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for c in range(0, C):
        tv = tl.load(t_ptr + n * stn + c * stc + oh * sth + ow * stw,
                     mask=ow_mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + co * swo + c * swc).to(tl.float32)
        acc += tv * wv
    acc += tl.load(b_ptr + co).to(tl.float32)
    tl.store(y_ptr + n * syn + co * syc + oh * syh + ow * syw, acc.to({tldt}), mask=ow_mask)


def {op}(x: torch.Tensor, dw: torch.Tensor, pw: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    Cout = pw.shape[0]
    PAD, K = {P}, {K}
    t = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)
    BLOCK_W = triton.next_power_of_2(W)
    _{op}_dw_kernel[(N * C * H,)](x, dw, t, C, H, W,
                                  x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                  dw.stride(0), dw.stride(2), dw.stride(3),
                                  t.stride(0), t.stride(1), t.stride(2), t.stride(3),
                                  PAD=PAD, K=K, BLOCK_W=BLOCK_W, num_warps=4)
    y = torch.empty((N, Cout, H, W), device=x.device, dtype=x.dtype)
    _{op}_pw_kernel[(N * Cout * H,)](t, pw, bias, y, C, Cout, H, W,
                                     t.stride(0), t.stride(1), t.stride(2), t.stride(3),
                                     pw.stride(0), pw.stride(1),
                                     y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                                     BLOCK_W=BLOCK_W, num_warps=4)
    return y
'''


def _separable_seed(op, cfg, dtype, tldt):
    K = cfg["K"]
    P = _same_pad(K, 1)
    text = ("Naive depthwise-separable conv2d = depthwise KxK (groups=C) then pointwise "
            "1x1, as two chained kernels vs torch two-conv baseline; the policy fuses the "
            f"two passes into one. {tldt} store.")
    return _SEPARABLE_TMPL.format(doc=_doc(op, dtype, text), imports=_IMPORTS,
                                  op=op, tldt=tldt, P=P, K=K)


_CONV1D_TMPL = '''{doc}{imports}

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, y_ptr, Cin, L, Cout, OL, GIN, GOUT,
                 sxn, sxc, sxl, swo, swc, swk, syn, syc, syl,
                 PADL: tl.constexpr, K: tl.constexpr, BLOCK_OL: tl.constexpr):
    row = tl.program_id(0)
    lblk = tl.program_id(1)
    co = row % Cout
    n = row // Cout
    g = co // GOUT
    ol = lblk * BLOCK_OL + tl.arange(0, BLOCK_OL)
    ol_mask = ol < OL
    acc = tl.zeros((BLOCK_OL,), dtype=tl.float32)
    for ci in range(0, GIN):
        cin = g * GIN + ci
        for k in range(0, K):
            il = ol - PADL + k
            m = ol_mask & (il >= 0) & (il < L)
            xv = tl.load(x_ptr + n * sxn + cin * sxc + il * sxl, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + co * swo + ci * swc + k * swk).to(tl.float32)
            acc += xv * wv
    acc += tl.load(b_ptr + co).to(tl.float32)
    tl.store(y_ptr + n * syn + co * syc + ol * syl, acc.to({tldt}), mask=ol_mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, Cin, L = x.shape
    Cout = weight.shape[0]
    GIN = weight.shape[1]
    GROUPS = Cin // GIN
    GOUT = Cout // GROUPS
    K = weight.shape[2]
    PADL = {PADL}
    OL = L
    y = torch.empty((N, Cout, OL), device=x.device, dtype=x.dtype)
    BLOCK_OL = 128
    grid = (N * Cout, triton.cdiv(OL, BLOCK_OL))
    _{op}_kernel[grid](x, weight, bias, y, Cin, L, Cout, OL, GIN, GOUT,
                       x.stride(0), x.stride(1), x.stride(2),
                       weight.stride(0), weight.stride(1), weight.stride(2),
                       y.stride(0), y.stride(1), y.stride(2),
                       PADL=PADL, K=K, BLOCK_OL=BLOCK_OL, num_warps=4)
    return y
'''


def _conv1d_seed(op, cfg, dtype, tldt):
    K, causal = cfg["K"], cfg["causal"]
    padl = (K - 1) if causal else (K - 1) // 2
    kind = "causal" if causal else "same"
    text = (f"Naive {kind} conv1d (K={K}, groups-generic) vs torch F.conv1d; one program "
            f"per (n, cout, L-block), fp32 accumulate over (cin, k). The audio / SSM short "
            f"conv - scan/fusion headroom. {tldt} store.")
    return _CONV1D_TMPL.format(doc=_doc(op, dtype, text), imports=_IMPORTS,
                               op=op, tldt=tldt, PADL=padl)


_CONV3D_TMPL = '''{doc}{imports}

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, y_ptr, Cin, Din, H, W, Cout, OD, OH, OW,
                 sxn, sxc, sxd, sxh, sxw, swo, swc, swd, swh, sww,
                 syn, syc, syd, syh, syw,
                 STRIDE: tl.constexpr, PAD: tl.constexpr,
                 KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    t = pid // OH
    od = t % OD
    t2 = t // OD
    co = t2 % Cout
    n = t2 // Cout
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for ci in range(0, Cin):
        for kd in range(0, KD):
            idp = od * STRIDE - PAD + kd
            d_ok = (idp >= 0) & (idp < Din)
            for kh in range(0, KH):
                ih = oh * STRIDE - PAD + kh
                h_ok = d_ok & (ih >= 0) & (ih < H)
                for kw in range(0, KW):
                    iw = ow * STRIDE - PAD + kw
                    m = ow_mask & h_ok & (iw >= 0) & (iw < W)
                    xv = tl.load(x_ptr + n * sxn + ci * sxc + idp * sxd + ih * sxh + iw * sxw,
                                 mask=m, other=0.0).to(tl.float32)
                    wv = tl.load(w_ptr + co * swo + ci * swc + kd * swd + kh * swh + kw * sww).to(tl.float32)
                    acc += xv * wv
    acc += tl.load(b_ptr + co).to(tl.float32)
    tl.store(y_ptr + n * syn + co * syc + od * syd + oh * syh + ow * syw,
             acc.to({tldt}), mask=ow_mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, Cin, Din, H, W = x.shape
    Cout = weight.shape[0]
    KD, KH, KW = weight.shape[2], weight.shape[3], weight.shape[4]
    STRIDE, PAD = {S}, {P}
    OD = (Din + 2 * PAD - (KD - 1) - 1) // STRIDE + 1
    OH = (H + 2 * PAD - (KH - 1) - 1) // STRIDE + 1
    OW = (W + 2 * PAD - (KW - 1) - 1) // STRIDE + 1
    y = torch.empty((N, Cout, OD, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * Cout * OD * OH,)
    _{op}_kernel[grid](x, weight, bias, y, Cin, Din, H, W, Cout, OD, OH, OW,
                       x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
                       weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
                       y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
                       STRIDE=STRIDE, PAD=PAD, KD=KD, KH=KH, KW=KW, BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
'''


def _conv3d_seed(op, cfg, dtype, tldt):
    K, S = cfg["K"], cfg["S"]
    P = _same_pad(K, 1)
    text = (f"Naive direct conv3d (K={K}x{K}x{K}, S={S}) vs torch F.conv3d; one program per "
            f"(n, cout, od, oh), fp32 accumulate over (cin, kd, kh, kw), output width "
            f"vectorized. Volumetric implicit-GEMM headroom. {tldt} store.")
    return _CONV3D_TMPL.format(doc=_doc(op, dtype, text), imports=_IMPORTS,
                               op=op, tldt=tldt, S=S, P=P)


_TRANSPOSE_TMPL = '''{doc}{imports}

@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, y_ptr, Cin, H, W, Cout, OH, OW,
                 sxn, sxc, sxh, sxw, swi, swo, swh, sww, syn, syc, syh, syw,
                 STRIDE: tl.constexpr, PAD: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    co = tmp % Cout
    n = tmp // Cout
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for cin in range(0, Cin):
        for kh in range(0, KH):
            numh = oh + PAD - kh
            ih = numh // STRIDE
            h_ok = (numh >= 0) & ((numh % STRIDE) == 0) & (ih < H)
            for kw in range(0, KW):
                numw = ow + PAD - kw
                iw = numw // STRIDE
                m = ow_mask & h_ok & (numw >= 0) & ((numw % STRIDE) == 0) & (iw < W)
                xv = tl.load(x_ptr + n * sxn + cin * sxc + ih * sxh + iw * sxw,
                             mask=m, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + cin * swi + co * swo + kh * swh + kw * sww).to(tl.float32)
                acc += xv * wv
    acc += tl.load(b_ptr + co).to(tl.float32)
    tl.store(y_ptr + n * syn + co * syc + oh * syh + ow * syw, acc.to({tldt}), mask=ow_mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, Cin, H, W = x.shape
    Cout = weight.shape[1]
    KH, KW = weight.shape[2], weight.shape[3]
    STRIDE, PAD = {S}, {P}
    OH = (H - 1) * STRIDE - 2 * PAD + (KH - 1) + 1
    OW = (W - 1) * STRIDE - 2 * PAD + (KW - 1) + 1
    y = torch.empty((N, Cout, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * Cout * OH,)
    _{op}_kernel[grid](x, weight, bias, y, Cin, H, W, Cout, OH, OW,
                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                       weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
                       y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                       STRIDE=STRIDE, PAD=PAD, KH=KH, KW=KW, BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
'''


def _transpose_seed(op, cfg, dtype, tldt):
    K, S, P = cfg["K"], cfg["S"], cfg["P"]
    text = (f"Naive transposed conv2d / deconv (K={K}, S={S}) vs torch F.conv_transpose2d; "
            f"gather form (scatter inverse) - one program per (n, cout, oh) with a "
            f"stride-modulo source map, fp32 accumulate. Upsampling headroom. {tldt} store.")
    return _TRANSPOSE_TMPL.format(doc=_doc(op, dtype, text), imports=_IMPORTS,
                                  op=op, tldt=tldt, S=S, P=P)


_DGRAD_TMPL = '''{doc}{imports}

@triton.jit
def _{op}_kernel(gy_ptr, w_ptr, gx_ptr, Cin, H, W, Cout, OH, OW,
                 sgn, sgc, sgh, sgw, swo, swc, swh, sww, sxn, sxc, sxh, sxw,
                 PAD: tl.constexpr, DIL: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_W: tl.constexpr):
    pid = tl.program_id(0)
    ih = pid % H
    tmp = pid // H
    cin = tmp % Cin
    n = tmp // Cin
    iw = tl.arange(0, BLOCK_W)
    iw_mask = iw < W
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for co in range(0, Cout):
        for kh in range(0, KH):
            oh = ih + PAD - kh * DIL
            oh_ok = (oh >= 0) & (oh < OH)
            for kw in range(0, KW):
                ow = iw + PAD - kw * DIL
                m = iw_mask & oh_ok & (ow >= 0) & (ow < OW)
                gv = tl.load(gy_ptr + n * sgn + co * sgc + oh * sgh + ow * sgw,
                             mask=m, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + co * swo + cin * swc + kh * swh + kw * sww).to(tl.float32)
                acc += gv * wv
    tl.store(gx_ptr + n * sxn + cin * sxc + ih * sxh + iw * sxw, acc.to({tldt}), mask=iw_mask)


def {op}(x: torch.Tensor, grad_y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    N, Cin, H, W = x.shape
    Cout = weight.shape[0]
    KH, KW = weight.shape[2], weight.shape[3]
    OH, OW = grad_y.shape[2], grad_y.shape[3]
    PAD, DIL = {P}, {D}
    gx = torch.empty((N, Cin, H, W), device=x.device, dtype=x.dtype)
    BLOCK_W = triton.next_power_of_2(W)
    grid = (N * Cin * H,)
    _{op}_kernel[grid](grad_y, weight, gx, Cin, H, W, Cout, OH, OW,
                       grad_y.stride(0), grad_y.stride(1), grad_y.stride(2), grad_y.stride(3),
                       weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
                       gx.stride(0), gx.stride(1), gx.stride(2), gx.stride(3),
                       PAD=PAD, DIL=DIL, KH=KH, KW=KW, BLOCK_W=BLOCK_W, num_warps=4)
    return gx
'''


def _dgrad_seed(op, cfg, dtype, tldt):
    K, D = cfg["K"], cfg["D"]
    P = _same_pad(K, D)
    text = (f"Naive conv2d dInput / dgrad (K={K}, stride 1) vs torch.nn.grad.conv2d_input; "
            f"one program per (n, cin, ih) accumulating grad_y * weight over (cout, kh, kw). "
            f"The hard training backward kernel. {tldt} store.")
    return _DGRAD_TMPL.format(doc=_doc(op, dtype, text), imports=_IMPORTS,
                              op=op, tldt=tldt, P=P, D=D)


_WGRAD_TMPL = '''{doc}{imports}

@triton.jit
def _{op}_kernel(x_ptr, gy_ptr, gw_ptr, N, Cin, H, W, Cout, OH, OW,
                 sxn, sxc, sxh, sxw, sgn, sgc, sgh, sgw, swo, swc, swh, sww,
                 PAD: tl.constexpr, DIL: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    cin = pid % Cin
    co = pid // Cin
    for kh in range(0, KH):
        for kw in range(0, KW):
            acc = 0.0
            for n in range(0, N):
                for oh in range(0, OH):
                    ow = tl.arange(0, BLOCK_OW)
                    ow_mask = ow < OW
                    ih = oh - PAD + kh * DIL
                    ih_ok = (ih >= 0) & (ih < H)
                    iw = ow - PAD + kw * DIL
                    m = ow_mask & ih_ok & (iw >= 0) & (iw < W)
                    gv = tl.load(gy_ptr + n * sgn + co * sgc + oh * sgh + ow * sgw,
                                 mask=ow_mask, other=0.0).to(tl.float32)
                    xv = tl.load(x_ptr + n * sxn + cin * sxc + ih * sxh + iw * sxw,
                                 mask=m, other=0.0).to(tl.float32)
                    acc += tl.sum(gv * xv, axis=0)
            tl.store(gw_ptr + co * swo + cin * swc + kh * swh + kw * sww, acc.to({tldt}))


def {op}(x: torch.Tensor, grad_y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    N, Cin, H, W = x.shape
    Cout = weight.shape[0]
    KH, KW = weight.shape[2], weight.shape[3]
    OH, OW = grad_y.shape[2], grad_y.shape[3]
    PAD, DIL = {P}, {D}
    gw = torch.empty((Cout, Cin, KH, KW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (Cout * Cin,)
    _{op}_kernel[grid](x, grad_y, gw, N, Cin, H, W, Cout, OH, OW,
                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                       grad_y.stride(0), grad_y.stride(1), grad_y.stride(2), grad_y.stride(3),
                       gw.stride(0), gw.stride(1), gw.stride(2), gw.stride(3),
                       PAD=PAD, DIL=DIL, KH=KH, KW=KW, BLOCK_OW=BLOCK_OW, num_warps=4)
    return gw
'''


def _wgrad_seed(op, cfg, dtype, tldt):
    K, D = cfg["K"], cfg["D"]
    P = _same_pad(K, D)
    text = (f"Naive conv2d dWeight / wgrad (K={K}, stride 1) vs torch.nn.grad.conv2d_weight; "
            f"one program per (cout, cin) reducing grad_y * x over (n, oh, ow) per tap. "
            f"The hard training backward kernel. {tldt} store.")
    return _WGRAD_TMPL.format(doc=_doc(op, dtype, text), imports=_IMPORTS,
                              op=op, tldt=tldt, P=P, D=D)


_IM2COL_TMPL = '''{doc}{imports}

@triton.jit
def _{op}_kernel(x_ptr, y_ptr, C, H, W, OH, OW, ROWS, LTOT,
                 sxn, sxc, sxh, sxw, syn, syr, syl,
                 STRIDE: tl.constexpr, PAD: tl.constexpr, DIL: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_L: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    row = pid0 % ROWS
    n = pid0 // ROWS
    kw = row % KW
    r2 = row // KW
    kh = r2 % KH
    cin = r2 // KH
    l = pid1 * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = l < LTOT
    oh = l // OW
    ow = l % OW
    ih = oh * STRIDE - PAD + kh * DIL
    iw = ow * STRIDE - PAD + kw * DIL
    m = l_mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    xv = tl.load(x_ptr + n * sxn + cin * sxc + ih * sxh + iw * sxw, mask=m, other=0.0)
    tl.store(y_ptr + n * syn + row * syr + l * syl, xv.to({tldt}), mask=l_mask)


def {op}(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    KH, KW = {K}, {K}
    STRIDE, PAD, DIL = {S}, {P}, {D}
    OH = (H + 2 * PAD - DIL * (KH - 1) - 1) // STRIDE + 1
    OW = (W + 2 * PAD - DIL * (KW - 1) - 1) // STRIDE + 1
    ROWS = C * KH * KW
    LTOT = OH * OW
    y = torch.empty((N, ROWS, LTOT), device=x.device, dtype=x.dtype)
    BLOCK_L = 128
    grid = (N * ROWS, triton.cdiv(LTOT, BLOCK_L))
    _{op}_kernel[grid](x, y, C, H, W, OH, OW, ROWS, LTOT,
                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                       y.stride(0), y.stride(1), y.stride(2),
                       STRIDE=STRIDE, PAD=PAD, DIL=DIL, KH=KH, KW=KW,
                       BLOCK_L=BLOCK_L, num_warps=4)
    return y
'''


def _im2col_seed(op, cfg, dtype, tldt):
    K, S, D = cfg["K"], cfg["S"], cfg["D"]
    P = _same_pad(K, D)
    text = (f"Naive im2col / unfold (K={K}, S={S}) vs torch F.unfold; one program per "
            f"(n, cin*kh*kw row, L-block) gathering the padded patch. The implicit-GEMM "
            f"lowering primitive. {tldt} store.")
    return _IM2COL_TMPL.format(doc=_doc(op, dtype, text), imports=_IMPORTS,
                               op=op, tldt=tldt, K=K, S=S, P=P, D=D)


_COL2IM_TMPL = '''{doc}{imports}
import math


@triton.jit
def _{op}_kernel(c_ptr, y_ptr, C, H, W, OH, OW, ROWS,
                 scn, scr, scl, syn, syc, syh, syw,
                 PAD: tl.constexpr, DIL: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_W: tl.constexpr):
    pid = tl.program_id(0)
    ih = pid % H
    tmp = pid // H
    c = tmp % C
    n = tmp // C
    iw = tl.arange(0, BLOCK_W)
    iw_mask = iw < W
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for kh in range(0, KH):
        oh = ih + PAD - kh * DIL
        oh_ok = (oh >= 0) & (oh < OH)
        for kw in range(0, KW):
            ow = iw + PAD - kw * DIL
            m = iw_mask & oh_ok & (ow >= 0) & (ow < OW)
            row = (c * KH + kh) * KW + kw
            l = oh * OW + ow
            cv = tl.load(c_ptr + n * scn + row * scr + l * scl, mask=m, other=0.0).to(tl.float32)
            acc += cv
    tl.store(y_ptr + n * syn + c * syc + ih * syh + iw * syw, acc.to({tldt}), mask=iw_mask)


def {op}(cols: torch.Tensor) -> torch.Tensor:
    N, ROWS, LTOT = cols.shape
    KH, KW = {K}, {K}
    PAD, DIL = {P}, {D}
    C = ROWS // (KH * KW)
    H = int(round(math.sqrt(LTOT)))
    W = H
    OH = (H + 2 * PAD - DIL * (KH - 1) - 1) + 1
    OW = (W + 2 * PAD - DIL * (KW - 1) - 1) + 1
    y = torch.empty((N, C, H, W), device=cols.device, dtype=cols.dtype)
    BLOCK_W = triton.next_power_of_2(W)
    grid = (N * C * H,)
    _{op}_kernel[grid](cols, y, C, H, W, OH, OW, ROWS,
                       cols.stride(0), cols.stride(1), cols.stride(2),
                       y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                       PAD=PAD, DIL=DIL, KH=KH, KW=KW, BLOCK_W=BLOCK_W, num_warps=4)
    return y
'''


def _col2im_seed(op, cfg, dtype, tldt):
    K, S, D = cfg["K"], cfg["S"], cfg["D"]
    P = _same_pad(K, D)
    text = (f"Naive col2im / fold (K={K}, stride 1, square) vs torch F.fold; one program per "
            f"(n, c, ih) overlap-adding the columns that map to it. The im2col adjoint. "
            f"{tldt} store.")
    return _COL2IM_TMPL.format(doc=_doc(op, dtype, text), imports=_IMPORTS,
                               op=op, tldt=tldt, K=K, P=P, D=D)


# --- Winograd unrolled transform seeds (exprs generated from the exact matrices) ---
def _lin_expr(coeffs_vars):
    """Build a Triton fp32 linear-combination expression from (coeff, var) pairs
    (coeff != 0). Emits exact float literals so the seed matches the oracle math."""
    out = ""
    for c, var in coeffs_vars:
        if c == 0:
            continue
        neg = c < 0
        mag = abs(c)
        term = var if mag == 1 else f"{mag:.10g} * {var}"
        if out == "":
            out = ("-" + term) if neg else term
        else:
            out += (" - " if neg else " + ") + term
    return out or "0.0"


_WINO_KERNEL_TMPL = '''{doc}{imports}

@triton.jit
def _{op}_kernel({inp}_ptr, y_ptr, NROW, s0, s1, s2, s3, o0, o1, o2, o3):
    pid = tl.program_id(0)
    j = pid % NROW
    i = pid // NROW
    base = i * s0 + j * s1
    obase = i * o0 + j * o1
{loads}
{stores}


def {op}({inp}: torch.Tensor) -> torch.Tensor:
    A, B, IH, IW = {inp}.shape
    y = torch.empty((A, B, 4, 4), device={inp}.device, dtype={inp}.dtype)
    grid = (A * B,)
    _{op}_kernel[grid]({inp}, y, B,
                       {inp}.stride(0), {inp}.stride(1), {inp}.stride(2), {inp}.stride(3),
                       y.stride(0), y.stride(1), y.stride(2), y.stride(3), num_warps=1)
    return y
'''


def _winograd_seed(op, cfg, dtype, tldt):
    fam = cfg["family"]
    if fam == "winograd_input":
        rows_in, cols_in = 4, 4            # V[i,j] = sum_kl Bt[i,k] Bt[j,l] d[k,l]
        left, right = _WINO_BT, _WINO_BT
        inp = "d"
        text = ("Naive Winograd F(2x2,3x3) INPUT transform V = Bt.d.B (exact integer "
                f"transform, unrolled per 4x4 tile) vs the batched-matmul oracle. {tldt} store.")
    else:
        rows_in, cols_in = 3, 3
        left, right = _WINO_G, _WINO_G     # 4x3; U[i,j] = sum_pq G[i,p] G[j,q] g[p,q]
        inp = "g"
        text = ("Naive Winograd F(2x2,3x3) FILTER transform U = G.g.Gt (exact rational "
                f"transform, unrolled per 3x3 filter) vs the batched-matmul oracle. {tldt} store.")
    var = "d" if fam == "winograd_input" else "g"
    loads = []
    for r in range(rows_in):
        for c in range(cols_in):
            loads.append(f"    {var}{r}{c} = tl.load({inp}_ptr + base + {r} * s2 + {c} * s3).to(tl.float32)")
    stores = []
    for i in range(4):
        for jj in range(4):
            pairs = []
            for p in range(rows_in):
                for q in range(cols_in):
                    coeff = left[i][p] * right[jj][q]
                    pairs.append((coeff, f"{var}{p}{q}"))
            expr = _lin_expr(pairs)
            stores.append(f"    u{i}{jj} = {expr}")
            stores.append(f"    tl.store(y_ptr + obase + {i} * o2 + {jj} * o3, (u{i}{jj}).to({tldt}))")
    return _WINO_KERNEL_TMPL.format(
        doc=_doc(op, dtype, text), imports=_IMPORTS, op=op, inp=inp,
        loads="\n".join(loads), stores="\n".join(stores))


def seed_source(op: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]
    cfg = _CFG[op]
    fam = cfg["family"]
    if fam == "conv2d":
        return _conv2d_seed(op, cfg, dtype, tldt)
    if fam == "separable2d":
        return _separable_seed(op, cfg, dtype, tldt)
    if fam == "conv1d":
        return _conv1d_seed(op, cfg, dtype, tldt)
    if fam == "conv3d":
        return _conv3d_seed(op, cfg, dtype, tldt)
    if fam == "transpose2d":
        return _transpose_seed(op, cfg, dtype, tldt)
    if fam == "dgrad2d":
        return _dgrad_seed(op, cfg, dtype, tldt)
    if fam == "wgrad2d":
        return _wgrad_seed(op, cfg, dtype, tldt)
    if fam == "im2col":
        return _im2col_seed(op, cfg, dtype, tldt)
    if fam == "col2im":
        return _col2im_seed(op, cfg, dtype, tldt)
    if fam in ("winograd_input", "winograd_filter"):
        return _winograd_seed(op, cfg, dtype, tldt)
    raise ValueError(f"unknown conv_ext op {op!r}")
