"""Breadth conv/pooling/resize task-authoring engine (torch-baselined).

Widens the KORE suite with the vision/CNN operator families that the vendor-
baselined core (norms / activations / GEMM / attention) never covered:
convolution (standard / depthwise / dilated), pooling (max / avg / adaptive /
global) and spatial resize (bilinear / nearest upsample). Unlike the vendor tasks
(graded against AITER), these grade against the honest torch ``F.*`` production
path - the framework op AMD's ROCm stack actually dispatches for these shapes.

Contract mirrors ``kore/tasks/vendor_ops.py`` so the shared ``_genops`` driver
applies unchanged:

    OPS / OP_DTYPES / SHAPES              module-level task catalog
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn fp32 oracle, baseline_fn torch, arity, entry_name,
        dtype_name, family=f"breadth_{op}")
    seed_source(op, dtype) -> str         a naive, COMPILING, correct Triton seed
        (defines ``def <op>(*inputs)``) - the policy's starting point.

CORRECTNESS is paramount: every ``ref_fn`` computes in fp32 and casts back, and
is validated on CPU against an INDEPENDENT torch computation (see tests). torch
imported lazily inside make_reference so registry discovery never needs a GPU.
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape

# --------------------------------------------------------------------------- #
# task catalog
# --------------------------------------------------------------------------- #
OPS: list[str] = [
    "conv2d_nchw",
    "depthwise_conv2d",
    "dilated_conv2d",
    "maxpool2d",
    "avgpool2d",
    "adaptive_avgpool2d",
    "global_avgpool",
    "interpolate_bilinear",
    "interpolate_nearest",
]

# bf16/fp16 sweep (matches the vendor default); the fp32 oracle casts back.
OP_DTYPES: dict[str, list[str]] = {op: ["bf16", "fp16"] for op in OPS}

# Adaptive-avg-pool fixed output (ResNet pre-FC 7x7). Shapes below keep H,W
# divisible by this so the naive seed's fixed-window tiling equals the exact
# adaptive result (the oracle uses F.adaptive_avg_pool2d and is always correct).
ADAPTIVE_OUT = 7

# conv geometry per op (kernel size comes from the weight tensor).
_CONV_CFG: dict[str, dict[str, int]] = {
    "conv2d_nchw": {"stride": 1, "pad": 1, "dil": 1},       # 3x3 same conv
    "dilated_conv2d": {"stride": 1, "pad": 2, "dil": 2},    # 3x3 dilation-2 same conv
}

# Realistic CNN shapes (N=1-8, C=64-512, H=W=14-224, 3x3 kernels).
_CONV_SHAPES = {
    "minimal": {"N": 1, "Cin": 64, "H": 32, "W": 32, "Cout": 64, "K": 3},
    "primary": {"N": 8, "Cin": 128, "H": 56, "W": 56, "Cout": 128, "K": 3},
    "validation": [
        {"N": 4, "Cin": 256, "H": 28, "W": 28, "Cout": 256, "K": 3},
        {"N": 1, "Cin": 512, "H": 14, "W": 14, "Cout": 512, "K": 3},
        {"N": 2, "Cin": 64, "H": 112, "W": 112, "Cout": 128, "K": 3},
    ],
}
_DILATED_SHAPES = {
    "minimal": {"N": 1, "Cin": 64, "H": 32, "W": 32, "Cout": 64, "K": 3},
    "primary": {"N": 8, "Cin": 128, "H": 56, "W": 56, "Cout": 128, "K": 3},
    "validation": [
        {"N": 4, "Cin": 256, "H": 28, "W": 28, "Cout": 256, "K": 3},
        {"N": 1, "Cin": 512, "H": 14, "W": 14, "Cout": 512, "K": 3},
        {"N": 2, "Cin": 128, "H": 56, "W": 56, "Cout": 128, "K": 3},
    ],
}
_DW_SHAPES = {  # depthwise: Cout == Cin == C
    "minimal": {"N": 1, "C": 64, "H": 32, "W": 32, "K": 3},
    "primary": {"N": 8, "C": 128, "H": 56, "W": 56, "K": 3},
    "validation": [
        {"N": 4, "C": 256, "H": 28, "W": 28, "K": 3},
        {"N": 1, "C": 512, "H": 14, "W": 14, "K": 3},
        {"N": 2, "C": 96, "H": 112, "W": 112, "K": 3},
    ],
}
_POOL_SHAPES = {  # H,W even so the 2x2/stride-2 windows tile exactly
    "minimal": {"N": 1, "C": 64, "H": 32, "W": 32},
    "primary": {"N": 8, "C": 256, "H": 56, "W": 56},
    "validation": [
        {"N": 4, "C": 128, "H": 112, "W": 112},
        {"N": 1, "C": 512, "H": 28, "W": 28},
        {"N": 2, "C": 64, "H": 224, "W": 224},
    ],
}
_ADAPTIVE_SHAPES = {  # H,W divisible by ADAPTIVE_OUT (7): 28/56/112/224
    "minimal": {"N": 1, "C": 64, "H": 28, "W": 28},
    "primary": {"N": 8, "C": 256, "H": 56, "W": 56},
    "validation": [
        {"N": 4, "C": 128, "H": 112, "W": 112},
        {"N": 1, "C": 512, "H": 28, "W": 28},
        {"N": 2, "C": 64, "H": 224, "W": 224},
    ],
}
_GLOBAL_SHAPES = {
    "minimal": {"N": 1, "C": 64, "H": 32, "W": 32},
    "primary": {"N": 8, "C": 512, "H": 56, "W": 56},
    "validation": [
        {"N": 4, "C": 256, "H": 28, "W": 28},
        {"N": 1, "C": 512, "H": 7, "W": 7},
        {"N": 2, "C": 128, "H": 112, "W": 112},
    ],
}
_INTERP_SHAPES = {
    "minimal": {"N": 1, "C": 64, "H": 32, "W": 32},
    "primary": {"N": 8, "C": 128, "H": 56, "W": 56},
    "validation": [
        {"N": 4, "C": 256, "H": 28, "W": 28},
        {"N": 1, "C": 64, "H": 112, "W": 112},
        {"N": 2, "C": 128, "H": 64, "W": 64},
    ],
}

SHAPES: dict[str, dict] = {
    "conv2d_nchw": _CONV_SHAPES,
    "depthwise_conv2d": _DW_SHAPES,
    "dilated_conv2d": _DILATED_SHAPES,
    "maxpool2d": _POOL_SHAPES,
    "avgpool2d": _POOL_SHAPES,
    "adaptive_avgpool2d": _ADAPTIVE_SHAPES,
    "global_avgpool": _GLOBAL_SHAPES,
    "interpolate_bilinear": _INTERP_SHAPES,
    "interpolate_nearest": _INTERP_SHAPES,
}


# --------------------------------------------------------------------------- #
# reference.py namespace (torch fp32 oracle + torch F.* production baseline)
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch
    import torch.nn.functional as F

    tdt = getattr(torch, DTYPES[dtype][0])

    def _randn(shape, device, seed, scale=1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        return (torch.randn(shape, generator=g, device=device,
                            dtype=torch.float32) * scale).to(tdt)

    if op in ("conv2d_nchw", "dilated_conv2d"):
        cfg = _CONV_CFG[op]
        S, P, D = cfg["stride"], cfg["pad"], cfg["dil"]

        def get_inputs(shape, device="cuda", seed=0):
            N, Cin, H, W = shape["N"], shape["Cin"], shape["H"], shape["W"]
            Cout, K = shape["Cout"], shape["K"]
            fan_in = Cin * K * K   # 1/sqrt(fan_in) weight scale -> output ~ O(1)
            x = _randn((N, Cin, H, W), device, seed)
            w = _randn((Cout, Cin, K, K), device, seed + 1, scale=1.0 / (fan_in ** 0.5))
            b = _randn((Cout,), device, seed + 2, scale=0.1)
            return (x, w, b)

        def ref_fn(x, w, b):
            return F.conv2d(x.float(), w.float(), b.float(),
                            stride=S, padding=P, dilation=D).to(x.dtype)

        def baseline_fn(x, w, b):
            return F.conv2d(x, w, b, stride=S, padding=P, dilation=D)

        arity = 3

    elif op == "depthwise_conv2d":
        def get_inputs(shape, device="cuda", seed=0):
            N, C, H, W, K = shape["N"], shape["C"], shape["H"], shape["W"], shape["K"]
            fan_in = K * K
            x = _randn((N, C, H, W), device, seed)
            w = _randn((C, 1, K, K), device, seed + 1, scale=1.0 / (fan_in ** 0.5))
            b = _randn((C,), device, seed + 2, scale=0.1)
            return (x, w, b)

        def ref_fn(x, w, b):
            C = x.shape[1]
            return F.conv2d(x.float(), w.float(), b.float(),
                            stride=1, padding=1, groups=C).to(x.dtype)

        def baseline_fn(x, w, b):
            C = x.shape[1]
            return F.conv2d(x, w, b, stride=1, padding=1, groups=C)

        arity = 3

    elif op in ("maxpool2d", "avgpool2d"):
        def get_inputs(shape, device="cuda", seed=0):
            N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
            return (_randn((N, C, H, W), device, seed),)

        if op == "maxpool2d":
            def ref_fn(x):
                return F.max_pool2d(x.float(), 2).to(x.dtype)

            def baseline_fn(x):
                return F.max_pool2d(x, 2)
        else:
            def ref_fn(x):
                return F.avg_pool2d(x.float(), 2).to(x.dtype)

            def baseline_fn(x):
                return F.avg_pool2d(x, 2)

        arity = 1

    elif op == "adaptive_avgpool2d":
        out = ADAPTIVE_OUT

        def get_inputs(shape, device="cuda", seed=0):
            N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
            return (_randn((N, C, H, W), device, seed),)

        def ref_fn(x):
            return F.adaptive_avg_pool2d(x.float(), (out, out)).to(x.dtype)

        def baseline_fn(x):
            return F.adaptive_avg_pool2d(x, (out, out))

        arity = 1

    elif op == "global_avgpool":
        def get_inputs(shape, device="cuda", seed=0):
            N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
            return (_randn((N, C, H, W), device, seed),)

        def ref_fn(x):
            return x.float().mean(dim=(2, 3)).to(x.dtype)   # [N, C]

        def baseline_fn(x):
            return F.adaptive_avg_pool2d(x, 1).flatten(1)    # [N, C]

        arity = 1

    elif op in ("interpolate_bilinear", "interpolate_nearest"):
        mode = "bilinear" if op == "interpolate_bilinear" else "nearest"

        def get_inputs(shape, device="cuda", seed=0):
            N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
            return (_randn((N, C, H, W), device, seed),)

        if mode == "bilinear":
            def ref_fn(x):
                return F.interpolate(x.float(), scale_factor=2, mode="bilinear",
                                     align_corners=False).to(x.dtype)

            def baseline_fn(x):
                return F.interpolate(x, scale_factor=2, mode="bilinear",
                                     align_corners=False)
        else:
            def ref_fn(x):
                return F.interpolate(x.float(), scale_factor=2, mode="nearest").to(x.dtype)

            def baseline_fn(x):
                return F.interpolate(x, scale_factor=2, mode="nearest")

        arity = 1

    else:
        raise ValueError(f"unknown breadth op {op!r}")

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity, "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}", "mutates_input": False}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive (correct, compiling) Triton seeds - the policy's starting point.
# --------------------------------------------------------------------------- #
_STD_CONV_SEED = '''"""GENERATED breadth {op} seed ({dtype}) vs torch F.conv2d.
Naive direct convolution: one program per (n, cout, oh) output row; fp32 accumulate
over (cin, kh, kw) with the output width vectorized. Correct-but-slow starting point
the KORE policy optimizes against the torch baseline. {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _{op}_kernel(x_ptr, w_ptr, b_ptr, y_ptr, Cin, H, W, Cout, OH, OW,
                 sxn, sxc, sxh, sxw, swo, swc, swh, sww, syn, syc, syh, syw,
                 STRIDE: tl.constexpr, PAD: tl.constexpr, DIL: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    co = tmp % Cout
    n = tmp // Cout
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for ci in range(0, Cin):
        for kh in range(0, KH):
            ih = oh * STRIDE - PAD + kh * DIL
            h_ok = (ih >= 0) & (ih < H)
            for kw in range(0, KW):
                iw = ow * STRIDE - PAD + kw * DIL
                m = ow_mask & h_ok & (iw >= 0) & (iw < W)
                xv = tl.load(x_ptr + n * sxn + ci * sxc + ih * sxh + iw * sxw,
                             mask=m, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + co * swo + ci * swc + kh * swh + kw * sww).to(tl.float32)
                acc += xv * wv
    acc += tl.load(b_ptr + co).to(tl.float32)
    y_off = n * syn + co * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, acc.to({tldt}), mask=ow_mask)


def {op}(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape
    STRIDE, PAD, DIL = {stride}, {pad}, {dil}
    OH = (H + 2 * PAD - DIL * (KH - 1) - 1) // STRIDE + 1
    OW = (W + 2 * PAD - DIL * (KW - 1) - 1) // STRIDE + 1
    y = torch.empty((N, Cout, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * Cout * OH,)
    _{op}_kernel[grid](x, weight, bias, y, Cin, H, W, Cout, OH, OW,
                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                       weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
                       y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                       STRIDE=STRIDE, PAD=PAD, DIL=DIL, KH=KH, KW=KW,
                       BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
'''

_DEPTHWISE_CONV_SEED = '''"""GENERATED breadth depthwise_conv2d seed ({dtype}) vs torch F.conv2d(groups=C).
Naive depthwise conv: one program per (n, c, oh); each output channel convolves only
its own input channel over (kh, kw). fp32 accumulate, output width vectorized, {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _depthwise_conv2d_kernel(x_ptr, w_ptr, b_ptr, y_ptr, C, H, W, OH, OW,
                             sxn, sxc, sxh, sxw, swo, swh, sww, syn, syc, syh, syw,
                             STRIDE: tl.constexpr, PAD: tl.constexpr, DIL: tl.constexpr,
                             KH: tl.constexpr, KW: tl.constexpr, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for kh in range(0, KH):
        ih = oh * STRIDE - PAD + kh * DIL
        h_ok = (ih >= 0) & (ih < H)
        for kw in range(0, KW):
            iw = ow * STRIDE - PAD + kw * DIL
            m = ow_mask & h_ok & (iw >= 0) & (iw < W)
            xv = tl.load(x_ptr + n * sxn + c * sxc + ih * sxh + iw * sxw,
                         mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + c * swo + kh * swh + kw * sww).to(tl.float32)
            acc += xv * wv
    acc += tl.load(b_ptr + c).to(tl.float32)
    y_off = n * syn + c * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, acc.to({tldt}), mask=ow_mask)


def depthwise_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    KH, KW = weight.shape[2], weight.shape[3]
    STRIDE, PAD, DIL = 1, 1, 1
    OH = (H + 2 * PAD - DIL * (KH - 1) - 1) // STRIDE + 1
    OW = (W + 2 * PAD - DIL * (KW - 1) - 1) // STRIDE + 1
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * C * OH,)
    _depthwise_conv2d_kernel[grid](x, weight, bias, y, C, H, W, OH, OW,
                                   x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                   weight.stride(0), weight.stride(2), weight.stride(3),
                                   y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                                   STRIDE=STRIDE, PAD=PAD, DIL=DIL, KH=KH, KW=KW,
                                   BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
'''

_MAXPOOL_SEED = '''"""GENERATED breadth maxpool2d seed ({dtype}) vs torch F.max_pool2d(2).
2x2 stride-2 max pool: one program per (n, c, oh), max over the 2x2 window across a
vectorized output width, {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _maxpool2d_kernel(x_ptr, y_ptr, C, H, W, OH, OW,
                      sxn, sxc, sxh, sxw, syn, syc, syh, syw, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32) - 1e38
    for kh in range(0, 2):
        ih = oh * 2 + kh
        for kw in range(0, 2):
            iw = ow * 2 + kw
            m = ow_mask & (ih < H) & (iw < W)
            xv = tl.load(x_ptr + n * sxn + c * sxc + ih * sxh + iw * sxw,
                         mask=m, other=-1e38).to(tl.float32)
            acc = tl.maximum(acc, xv)
    y_off = n * syn + c * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, acc.to({tldt}), mask=ow_mask)


def maxpool2d(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    OH = (H - 2) // 2 + 1
    OW = (W - 2) // 2 + 1
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * C * OH,)
    _maxpool2d_kernel[grid](x, y, C, H, W, OH, OW,
                            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                            BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
'''

_AVGPOOL_SEED = '''"""GENERATED breadth avgpool2d seed ({dtype}) vs torch F.avg_pool2d(2).
2x2 stride-2 average pool: one program per (n, c, oh), mean over the 2x2 window
(divisor 4, no padding) across a vectorized output width, {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _avgpool2d_kernel(x_ptr, y_ptr, C, H, W, OH, OW,
                      sxn, sxc, sxh, sxw, syn, syc, syh, syw, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for kh in range(0, 2):
        ih = oh * 2 + kh
        for kw in range(0, 2):
            iw = ow * 2 + kw
            m = ow_mask & (ih < H) & (iw < W)
            xv = tl.load(x_ptr + n * sxn + c * sxc + ih * sxh + iw * sxw,
                         mask=m, other=0.0).to(tl.float32)
            acc += xv
    y_off = n * syn + c * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, (acc / 4.0).to({tldt}), mask=ow_mask)


def avgpool2d(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    OH = (H - 2) // 2 + 1
    OW = (W - 2) // 2 + 1
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * C * OH,)
    _avgpool2d_kernel[grid](x, y, C, H, W, OH, OW,
                            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                            BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
'''

_ADAPTIVE_SEED = '''"""GENERATED breadth adaptive_avgpool2d seed ({dtype}) vs torch F.adaptive_avg_pool2d.
Output is a fixed 7x7 grid; with H,W divisible by 7 each output cell averages a
contiguous (H//7)x(W//7) window (== the exact adaptive result). One program per
(n, c, oh), output width vectorized, {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _adaptive_avgpool2d_kernel(x_ptr, y_ptr, C, H, W, OH, OW, KH, KW,
                               sxn, sxc, sxh, sxw, syn, syc, syh, syw, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    for a in range(0, KH):
        ih = oh * KH + a
        for b in range(0, KW):
            iw = ow * KW + b
            m = ow_mask & (ih < H) & (iw < W)
            xv = tl.load(x_ptr + n * sxn + c * sxc + ih * sxh + iw * sxw,
                         mask=m, other=0.0).to(tl.float32)
            acc += xv
    acc = acc / (KH * KW).to(tl.float32)
    y_off = n * syn + c * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, acc.to({tldt}), mask=ow_mask)


def adaptive_avgpool2d(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    OH, OW = 7, 7
    KH = H // OH
    KW = W // OW
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * C * OH,)
    _adaptive_avgpool2d_kernel[grid](x, y, C, H, W, OH, OW, KH, KW,
                                     x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                     y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                                     BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
'''

_GLOBAL_SEED = '''"""GENERATED breadth global_avgpool seed ({dtype}) vs torch global mean.
Global average over spatial dims: one program per (n, c) row reduces all H*W elements
in BLOCK-wide chunks (fp32 accumulate), output [N, C], {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _global_avgpool_kernel(x_ptr, y_ptr, HW, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, HW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HW
        xv = tl.load(x_ptr + row * HW + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(xv, axis=0)
    tl.store(y_ptr + row, (acc / HW.to(tl.float32)).to({tldt}))


def global_avgpool(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    HW = H * W
    y = torch.empty((N, C), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    _global_avgpool_kernel[grid](x.contiguous(), y, HW, BLOCK=1024, num_warps=4)
    return y
'''

_BILINEAR_SEED = '''"""GENERATED breadth interpolate_bilinear seed ({dtype}) vs torch F.interpolate.
2x bilinear upsample, align_corners=False: source coord = 0.5*(dst+0.5)-0.5 (clamped
>=0), 4-neighbor weighted blend. One program per (n, c, oh), output width vectorized,
{tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _interpolate_bilinear_kernel(x_ptr, y_ptr, C, H, W, OH, OW,
                                 sxn, sxc, sxh, sxw, syn, syc, syh, syw, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    fh = (oh + 0.5) * 0.5 - 0.5
    fh = tl.maximum(fh, 0.0)
    h0 = fh.to(tl.int32)
    h1 = tl.minimum(h0 + 1, H - 1)
    lh = fh - h0.to(tl.float32)
    fw = (ow.to(tl.float32) + 0.5) * 0.5 - 0.5
    fw = tl.maximum(fw, 0.0)
    w0 = fw.to(tl.int32)
    w1 = tl.minimum(w0 + 1, W - 1)
    lw = fw - w0.to(tl.float32)
    base = n * sxn + c * sxc
    v00 = tl.load(x_ptr + base + h0 * sxh + w0 * sxw, mask=ow_mask, other=0.0).to(tl.float32)
    v01 = tl.load(x_ptr + base + h0 * sxh + w1 * sxw, mask=ow_mask, other=0.0).to(tl.float32)
    v10 = tl.load(x_ptr + base + h1 * sxh + w0 * sxw, mask=ow_mask, other=0.0).to(tl.float32)
    v11 = tl.load(x_ptr + base + h1 * sxh + w1 * sxw, mask=ow_mask, other=0.0).to(tl.float32)
    top = v00 * (1.0 - lw) + v01 * lw
    bot = v10 * (1.0 - lw) + v11 * lw
    out = top * (1.0 - lh) + bot * lh
    y_off = n * syn + c * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, out.to({tldt}), mask=ow_mask)


def interpolate_bilinear(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    OH, OW = 2 * H, 2 * W
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * C * OH,)
    _interpolate_bilinear_kernel[grid](x, y, C, H, W, OH, OW,
                                       x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                       y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                                       BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
'''

_NEAREST_SEED = '''"""GENERATED breadth interpolate_nearest seed ({dtype}) vs torch F.interpolate.
2x nearest upsample: source index = dst // 2. One program per (n, c, oh), output width
vectorized gather, {tldt} store."""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _interpolate_nearest_kernel(x_ptr, y_ptr, C, H, W, OH, OW,
                                sxn, sxc, sxh, sxw, syn, syc, syh, syw, BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    c = tmp % C
    n = tmp // C
    ow = tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW
    ih = oh // 2
    iw = ow // 2
    base = n * sxn + c * sxc
    xv = tl.load(x_ptr + base + ih * sxh + iw * sxw, mask=ow_mask, other=0.0)
    y_off = n * syn + c * syc + oh * syh + ow * syw
    tl.store(y_ptr + y_off, xv.to({tldt}), mask=ow_mask)


def interpolate_nearest(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    OH, OW = 2 * H, 2 * W
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * C * OH,)
    _interpolate_nearest_kernel[grid](x, y, C, H, W, OH, OW,
                                      x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                                      y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                                      BLOCK_OW=BLOCK_OW, num_warps=4)
    return y
'''


def seed_source(op: str, dtype: str) -> str:
    tldt = DTYPES[dtype][1]
    if op in ("conv2d_nchw", "dilated_conv2d"):
        cfg = _CONV_CFG[op]
        return _STD_CONV_SEED.format(op=op, dtype=dtype, tldt=tldt,
                                     stride=cfg["stride"], pad=cfg["pad"], dil=cfg["dil"])
    if op == "depthwise_conv2d":
        return _DEPTHWISE_CONV_SEED.format(dtype=dtype, tldt=tldt)
    if op == "maxpool2d":
        return _MAXPOOL_SEED.format(dtype=dtype, tldt=tldt)
    if op == "avgpool2d":
        return _AVGPOOL_SEED.format(dtype=dtype, tldt=tldt)
    if op == "adaptive_avgpool2d":
        return _ADAPTIVE_SEED.format(dtype=dtype, tldt=tldt)
    if op == "global_avgpool":
        return _GLOBAL_SEED.format(dtype=dtype, tldt=tldt)
    if op == "interpolate_bilinear":
        return _BILINEAR_SEED.format(dtype=dtype, tldt=tldt)
    if op == "interpolate_nearest":
        return _NEAREST_SEED.format(dtype=dtype, tldt=tldt)
    raise ValueError(f"unknown breadth op {op!r}")
