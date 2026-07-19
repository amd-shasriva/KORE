"""Breadth QUANTIZATION / LOW-PRECISION frontier task-authoring engine.

Widens the KORE suite with the HARD *quantization machinery* kernels that flank
every quantized GEMM on MI350X / gfx950 (CDNA4): the dynamic activation / weight
quantizers, KV-cache compressors, weight packers and (de)quantizers that turn
bf16/fp16 tensors into fp8-e4m3 / int8 / int4 / mxfp4 codes + scales (and back).
These are the real serving/training-in-low-precision kernels - amax reductions
fused with a scaled round-and-pack, not memory-bound casts - and a fused Triton
kernel (block amax + reciprocal-scale + clamp + pack kept in registers) beats the
naive multi-pass torch path by a wide margin, so every op is a genuine "hard for
a GPU" kernel.

Coverage (every op name prefixed ``qx_``; 32 distinct ops):
  * Dynamic quant to fp8-e4m3 / int8 : per-tensor / per-token(row) / per-channel
        (col) / block-128 (amax -> scale -> quant; returns codes + scale).
  * Dequant fp8/int8 -> bf16 at the matching scale granularity.
  * KV-cache : quantize k,v per-token to fp8/int8 (+ scales) and dequant back.
  * int4 group-sym weight pack (nibble) + unpack, MXFP4 (e2m1 + e8m0/32) pack +
        unpack.
  * SmoothQuant : per-channel activation smoothing scale then per-token quant.
  * Block-wise 2D 128x128 fp8/int8 (DeepSeek-V3 weight) quant + dequant.
  * Double / nested quant (bitsandbytes) : quantize the block scales too.
  * Stochastic-rounding quant to fp8/int8 (seeded, deterministic via a noise input).
  * Fused quant + transpose (quantize while transposing for the next GEMM).

Contract mirrors ``kore/tasks/breadth/gemm_ext.py`` so the shared ``_genops``
driver + the breadth generator consume it unchanged:

    OPS / OP_DTYPES / SHAPES              module-level task catalog
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn EXACT fp32 quant oracle -> codes+scale tuple (quantize)
        or bf16 (dequant), baseline_fn torch eager path, arity, entry_name,
        dtype_name, family=f"breadth_{op}", mutates_input=False, adversarial_inputs).
    seed_source(op, dtype) -> str         a naive, COMPILING, CORRECT Triton seed
        (host amax/pack + a tiled elementwise quantize/dequantize kernel).

CORRECTNESS is paramount: ``ref_fn`` computes the EXACT fp32 quant math -
``scale = amax/qmax`` over the correct axis/block, ``q = round(x/scale)`` clamped
to the format range - and for a quantize op returns ``(codes, scale)`` (the
gate measures dequant-reconstruction SNR AND exact scale match). Every oracle is
validated on CPU against an INDEPENDENT torch computation (reshape-broadcast
scales, LUT nibble unpack, distinct reductions) at tight tolerance - see
tests/test_quant_ext.py.

SAFETY / SELF-CONTAINED: we deliberately do NOT import ``kore.tasks.aiter_ref``
(it touches ``torch.cuda`` at import). The OCP fp8-e4m3 max (448.0) is defined
locally and every reference is a pure, CPU-importable torch computation (torch is
imported lazily inside the bodies so registry discovery never needs a GPU).
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape  # noqa: F401  (DTYPES re-exported for parity)

# --------------------------------------------------------------------------- #
# Local quant constants (self-contained; NO aiter_ref -> no torch.cuda touch).
# --------------------------------------------------------------------------- #
FP8_MAX = 448.0                     # OCP float8_e4m3fn max finite (gfx950/CDNA4)
INT8_MAX = 127.0                    # symmetric int8 range
INT4_MIN, INT4_MAX = -8, 7          # symmetric int4 signed range
BLK = 128                           # DeepSeek-V3 block-scale group (1x128 / 128x128)
MX_BLOCK = 32                       # OCP microscaling group along K
E2M1_MAX = 6.0                      # max |value| in e2m1
E2M1_EMAX = 2                       # exponent of the e2m1 max normal (6.0 = 1.5*2^2)
_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   # e2m1 magnitudes (idx 0..7)
_E2M1_MIDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]      # round-to-nearest cuts
_IN_DT = "bfloat16"                 # activation/weight "in" dtype (label bf16)

_FP8_LEVELS_CACHE = None            # lazily-built sorted fp8-e4m3 level grid


# --------------------------------------------------------------------------- #
# Task catalog: op -> config dict (single source of truth). Fields:
#   kind : quant | dequant | kvquant | kvdequant | int4pack | int4unpack |
#          mxfp4pack | mxfp4unpack | smooth | double | stochastic | qtranspose
#   fmt  : "fp8" | "int8"  (quant format / expected code format)
#   gran : "tensor" | "token" | "channel" | "block128" | "block2d" | "block32"
#   group: int4 group size.
# --------------------------------------------------------------------------- #
def _c(kind: str, fmt: str = "fp8", gran: str = "token", group: int = 128) -> dict:
    return {"kind": kind, "fmt": fmt, "gran": gran, "group": group}


_CFG: dict[str, dict] = {
    # ---- dynamic activation quant to fp8 / int8 (per-tensor/token/channel/block)
    "qx_quant_fp8_pertensor":   _c("quant", "fp8", "tensor"),
    "qx_quant_fp8_pertoken":    _c("quant", "fp8", "token"),
    "qx_quant_fp8_perchannel":  _c("quant", "fp8", "channel"),
    "qx_quant_fp8_block128":    _c("quant", "fp8", "block128"),
    "qx_quant_int8_pertensor":  _c("quant", "int8", "tensor"),
    "qx_quant_int8_pertoken":   _c("quant", "int8", "token"),
    "qx_quant_int8_perchannel": _c("quant", "int8", "channel"),
    "qx_quant_int8_block128":   _c("quant", "int8", "block128"),
    # ---- dequant fp8 / int8 -> bf16 at matching granularity ------------------
    "qx_dequant_fp8_pertoken":   _c("dequant", "fp8", "token"),
    "qx_dequant_fp8_block128":   _c("dequant", "fp8", "block128"),
    "qx_dequant_int8_perchannel": _c("dequant", "int8", "channel"),
    "qx_dequant_int8_block128":  _c("dequant", "int8", "block128"),
    # ---- KV-cache quant / dequant (per-token) --------------------------------
    "qx_kvcache_quant_fp8":    _c("kvquant", "fp8", "token"),
    "qx_kvcache_quant_int8":   _c("kvquant", "int8", "token"),
    "qx_kvcache_dequant_fp8":  _c("kvdequant", "fp8", "token"),
    "qx_kvcache_dequant_int8": _c("kvdequant", "int8", "token"),
    # ---- int4 / mxfp4 weight pack + unpack -----------------------------------
    "qx_int4_pack_group":   _c("int4pack", "int8", "group", 128),
    "qx_int4_unpack_group": _c("int4unpack", "int8", "group", 128),
    "qx_mxfp4_pack":        _c("mxfp4pack", "fp8", "block32"),
    "qx_mxfp4_unpack":      _c("mxfp4unpack", "fp8", "block32"),
    # ---- SmoothQuant-style per-channel smoothing + quant ---------------------
    "qx_smoothquant_fp8":  _c("smooth", "fp8", "token"),
    "qx_smoothquant_int8": _c("smooth", "int8", "token"),
    # ---- block-wise 2D 128x128 (DeepSeek-V3 weight) --------------------------
    "qx_quant_fp8_block2d":   _c("quant", "fp8", "block2d"),
    "qx_dequant_fp8_block2d": _c("dequant", "fp8", "block2d"),
    "qx_quant_int8_block2d":  _c("quant", "int8", "block2d"),
    # ---- double / nested quant (bitsandbytes: quantize the scales too) -------
    "qx_double_quant_fp8":  _c("double", "fp8", "block128"),
    "qx_double_quant_int8": _c("double", "int8", "block128"),
    # ---- stochastic-rounding quant (seeded, deterministic) -------------------
    "qx_stochastic_fp8":  _c("stochastic", "fp8", "token"),
    "qx_stochastic_int8": _c("stochastic", "int8", "token"),
    # ---- fused quant + transpose (quantize while transposing) ----------------
    "qx_quant_transpose_fp8":           _c("qtranspose", "fp8", "token"),
    "qx_quant_transpose_int8":          _c("qtranspose", "int8", "token"),
    "qx_quant_transpose_fp8_pertensor": _c("qtranspose", "fp8", "tensor"),
}


def _dt_of(cfg: dict) -> str:
    """OP_DTYPES label (must be a DTYPES key): bf16 for the float-out (dequant/
    unpack) ops; the quant format (fp8/int8) for the quant-out ops; int4 pack ->
    int8, mxfp4 pack -> fp8 (nearest DTYPES quant label)."""
    kind = cfg["kind"]
    if kind in ("dequant", "kvdequant", "int4unpack", "mxfp4unpack"):
        return _IN_DT_LABEL
    if kind == "int4pack":
        return "int8"
    if kind == "mxfp4pack":
        return "fp8"
    return cfg["fmt"]


_IN_DT_LABEL = "bf16"

OPS: list[str] = list(_CFG)
OP_DTYPES: dict[str, list[str]] = {op: [_dt_of(_CFG[op])] for op in OPS}


# --------------------------------------------------------------------------- #
# small config predicates (module-level, pure)
# --------------------------------------------------------------------------- #
def _is_quantize(cfg: dict) -> bool:
    """True when the op EMITS quantized codes (oracle returns a tuple)."""
    return cfg["kind"] in ("quant", "kvquant", "int4pack", "mxfp4pack", "smooth",
                           "double", "stochastic", "qtranspose")


def _arg_names(cfg: dict) -> list[str]:
    return {
        "quant": ["x"],
        "dequant": ["codes", "scale"],
        "kvquant": ["k", "v"],
        "kvdequant": ["kq", "ksc", "vq", "vsc"],
        "int4pack": ["w"],
        "int4unpack": ["packed", "scale"],
        "mxfp4pack": ["x"],
        "mxfp4unpack": ["packed", "e8m0"],
        "smooth": ["x", "smooth"],
        "double": ["x"],
        "stochastic": ["x", "noise"],
        "qtranspose": ["x"],
    }[cfg["kind"]]


def arity_of(op: str) -> int:
    return len(_arg_names(_CFG[op]))


# --------------------------------------------------------------------------- #
# Shape catalog: realistic 2D [M(tokens), K(hidden)] shapes; K/M snapped to the
# format divisibility with a non-power-of-2 validation tail (mask stress). Never
# executed on CPU (tests use tiny shapes); only round-tripped through parse_shape.
# --------------------------------------------------------------------------- #
def _kmult(cfg: dict) -> int:
    g = cfg["gran"]
    if g in ("block128", "block2d"):
        return BLK
    if g == "block32":
        return MX_BLOCK
    if cfg["kind"] in ("int4pack", "int4unpack"):
        return cfg["group"]
    return 1


def _mmult(cfg: dict) -> int:
    return BLK if cfg["gran"] == "block2d" else 1


def _snap(v: int, m: int) -> int:
    return v if m <= 1 else max(m, (v // m) * m)


def _shapes_for(cfg: dict) -> dict:
    km, mm = _kmult(cfg), _mmult(cfg)
    tail_m = (mm * 33) if mm > 1 else 8193          # non-power-of-2, format-divisible
    f = lambda m, k: {"M": _snap(m, mm), "K": _snap(k, km)}
    return {
        "minimal": f(256, 256),
        "primary": f(4096, 4096),
        "validation": [f(2048, 4096), f(8192, 2048), {"M": tail_m, "K": _snap(4096, km)}],
    }


SHAPES: dict[str, dict] = {op: _shapes_for(_CFG[op]) for op in OPS}


# --------------------------------------------------------------------------- #
# QUANT ENCODE / DECODE helpers (float <-> codes + scales). Pure; lazy torch.
# --------------------------------------------------------------------------- #
def _qmax(fmt: str) -> float:
    return FP8_MAX if fmt == "fp8" else INT8_MAX


def _to_codes(q, fmt: str):
    import torch
    if fmt == "fp8":
        return q.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)


def _codes_f(codes):
    return codes.float()


def _quant_nd(xf, fmt: str, gran: str):
    """Symmetric quant of xf[M,K] at granularity ``gran`` -> (codes, scale) with
    scale shape () / [M,1] / [1,K] / [M,K//128] / [M//128,K//128]."""
    import torch
    mx = _qmax(fmt)
    if gran == "tensor":
        s = (xf.abs().amax().clamp(min=1e-12) / mx)
        q, s_store = xf / s, s.reshape(())
    elif gran == "token":
        s = (xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / mx)
        q, s_store = xf / s, s
    elif gran == "channel":
        s = (xf.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / mx)
        q, s_store = xf / s, s
    elif gran == "block128":
        M, K = xf.shape
        xb = xf.reshape(M, K // BLK, BLK)
        sb = (xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / mx)
        q, s_store = (xb / sb).reshape(M, K), sb.squeeze(-1)
    else:  # block2d (128x128)
        M, K = xf.shape
        xb = xf.reshape(M // BLK, BLK, K // BLK, BLK)
        sb = (xb.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / mx)
        q, s_store = (xb / sb).reshape(M, K), sb.squeeze(1).squeeze(-1)
    return (_to_codes(q, fmt), s_store.to(torch.float32).contiguous())


def _dequant_nd(codes, s, gran: str):
    c = _codes_f(codes)
    if gran in ("tensor", "token", "channel"):
        return c * s.float()
    if gran == "block128":
        return c * s.float().repeat_interleave(BLK, dim=1)
    return c * s.float().repeat_interleave(BLK, 0).repeat_interleave(BLK, 1)  # block2d


def _pack_int4_group(w, group: int):
    """Symmetric group-wise int4. w[N,K] -> (packed[N,K//2] uint8, scale[N,K//group]);
    w ~= (code-8)*scale."""
    import torch
    N, K = w.shape
    wb = w.float().reshape(N, K // group, group)
    amax = wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    scale = amax / INT4_MAX
    q = torch.round(wb / scale).clamp(INT4_MIN, INT4_MAX).reshape(N, K).to(torch.int32)
    code = (q + 8).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return (packed, scale.squeeze(-1).to(torch.float32).contiguous())


def _unpack_int4_group(packed, scale, group: int):
    import torch
    N, K = packed.shape[0], packed.shape[1] * 2
    q = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    q[:, 0::2] = (packed & 0xF).to(torch.int32) - 8
    q[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32) - 8
    return q.float() * scale.float().repeat_interleave(group, dim=1)


def _e2m1_codes(v):
    import torch
    sign = (v < 0).to(torch.uint8)
    mids = torch.tensor(_E2M1_MIDS, dtype=torch.float32, device=v.device)
    idx = torch.bucketize(v.abs(), mids).to(torch.uint8)   # 0..7
    return (sign << 3) | idx


def _quant_mxfp4(x):
    """OCP MXFP4: x[R,K] -> (packed[R,K//2] uint8, e8m0[R,K//32] uint8)."""
    import torch
    R, K = x.shape
    xb = x.float().reshape(R, K // MX_BLOCK, MX_BLOCK)
    amax = xb.abs().amax(dim=2, keepdim=True).clamp(min=1e-20)
    exp = (torch.floor(torch.log2(amax)) - float(E2M1_EMAX)).clamp(-127.0, 127.0)
    e8m0 = (exp + 127.0).to(torch.uint8)
    scale = torch.exp2(exp)
    xq = (xb / scale).clamp(-E2M1_MAX, E2M1_MAX).reshape(R, K)
    codes = _e2m1_codes(xq)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return (packed, e8m0.reshape(R, K // MX_BLOCK).contiguous())


def _dequant_mxfp4(packed, e8m0):
    import torch
    R, K = packed.shape[0], packed.shape[1] * 2
    codes = torch.empty((R, K), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=packed.device)
    mag = levels[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    scale = torch.exp2(e8m0.float() - 127.0).repeat_interleave(MX_BLOCK, dim=1)
    return (sign * mag) * scale


def _fp8_levels(device):
    """Sorted finite fp8-e4m3 level grid (253 values), built once via byte view."""
    import torch
    global _FP8_LEVELS_CACHE
    if _FP8_LEVELS_CACHE is None:
        b = torch.arange(256, dtype=torch.uint8)
        lv = b.view(torch.float8_e4m3fn).float()
        _FP8_LEVELS_CACHE = torch.unique(lv[torch.isfinite(lv)])
    return _FP8_LEVELS_CACHE.to(device)


def _stoch_round_fp8(v, noise):
    """Stochastic-round v (already in [-448,448]) onto the fp8 grid: pick the upper
    neighbour with probability (v-lo)/(hi-lo), else the lower neighbour."""
    import torch
    lv = _fp8_levels(v.device)
    hi_idx = torch.searchsorted(lv, v, right=False).clamp(max=lv.numel() - 1)
    lo_idx = (hi_idx - 1).clamp(min=0)
    hi, lo = lv[hi_idx], lv[lo_idx]
    denom = hi - lo
    p = torch.where(denom > 0, (v - lo) / denom, torch.zeros_like(v))
    return torch.where(noise < p, hi, lo).to(torch.float8_e4m3fn)


def _smooth_quant(xf, smooth, fmt: str):
    """SmoothQuant: divide activations by a per-channel smoothing factor, then a
    per-token symmetric quant. Reconstruction recovers x/smooth."""
    xs = xf / smooth.reshape(1, -1)
    return _quant_nd(xs, fmt, "token")


def _double_quant(xf, fmt: str):
    """bitsandbytes nested quant: block-128 quantize xf, then quantize the fp32
    block scales to fp8 with a per-tensor meta-scale. -> (codes, scale_codes, meta)."""
    import torch
    codes, bscale = _quant_nd(xf, fmt, "block128")            # bscale [M, K//128] fp32
    meta = (bscale.abs().amax().clamp(min=1e-12) / FP8_MAX)
    sc_codes = (bscale / meta).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (codes, sc_codes, meta.reshape(()).to(torch.float32))


def _stoch_quant(xf, noise, fmt: str):
    """Seeded stochastic-rounding per-token quant (noise ~ U[0,1) supplied as input
    so the kernel is deterministic + independently reproducible)."""
    import torch
    mx = _qmax(fmt)
    s = (xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / mx)
    v = xf / s
    if fmt == "int8":
        codes = torch.floor(v + noise).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    else:
        codes = _stoch_round_fp8(v.clamp(-FP8_MAX, FP8_MAX), noise)
    return (codes, s.to(torch.float32))


def _quant_transpose(xf, fmt: str, gran: str):
    """Fused quant + transpose: quantize xf[M,K] then emit codes^T[K,M] with the
    scale laid out to broadcast over the (now-leading) K axis. Recovers x^T."""
    codes, s = _quant_nd(xf, fmt, gran)
    codesT = codes.t().contiguous()
    s_store = s.reshape(1, -1) if gran == "token" else s              # [M,1]->[1,M]
    return (codesT, s_store)


# --------------------------------------------------------------------------- #
# float source tensors for get_inputs (randn) + the adversarial battery.
# --------------------------------------------------------------------------- #
def _src(mode: str, shape, g, device, sc: float):
    import torch
    if mode == "randn":
        return torch.randn(shape, generator=g, device=device, dtype=torch.float32) * sc
    if mode == "zeros":
        return torch.zeros(shape, device=device, dtype=torch.float32)
    if mode == "large":
        return torch.full(shape, 1e3, device=device, dtype=torch.float32)
    if mode == "neg_large":
        return torch.full(shape, -1e3, device=device, dtype=torch.float32)
    if mode == "small":
        return torch.full(shape, 1e-3, device=device, dtype=torch.float32)
    # sign_alt: alternating +-1
    n = 1
    for d in shape:
        n *= d
    return ((torch.arange(n, device=device) % 2) * 2 - 1).to(torch.float32).reshape(shape)


# --------------------------------------------------------------------------- #
# reference.py namespace (exact fp32 quant oracle + torch eager production path).
# torch imported lazily so registry discovery is GPU-free.
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch

    if op not in _CFG:
        raise ValueError(f"unknown breadth quant op {op!r}")
    cfg = _CFG[op]
    kind, fmt, gran, group = cfg["kind"], cfg["fmt"], cfg["gran"], cfg["group"]
    override = (dtype == "fp32")
    out_f = torch.float32 if override else (torch.float16 if dtype == "fp16" else torch.bfloat16)
    in_dt = torch.bfloat16

    def _gen(shape, device, seed, mode):
        g = torch.Generator(device=device).manual_seed(seed)
        M, K = shape["M"], shape["K"]
        if kind in ("quant", "double", "qtranspose", "mxfp4pack", "int4pack"):
            return (_src(mode, (M, K), g, device, 1.0).to(in_dt),)
        if kind == "smooth":
            x = _src(mode, (M, K), g, device, 1.0).to(in_dt)
            smooth = (torch.rand((K,), generator=g, device=device) * 1.5 + 0.5).to(in_dt)
            return (x, smooth)
        if kind == "stochastic":
            x = _src(mode, (M, K), g, device, 1.0).to(in_dt)
            noise = torch.rand((M, K), generator=g, device=device, dtype=torch.float32)
            return (x, noise)
        if kind == "kvquant":
            k = _src(mode, (M, K), g, device, 1.0).to(in_dt)
            v = _src(mode, (M, K), g, device, 0.9).to(in_dt)
            return (k, v)
        # ---- dequant-family: build structured (quantized) inputs -------------
        if kind == "dequant":
            codes, s = _quant_nd(_src(mode, (M, K), g, device, 1.0), fmt, gran)
            return (codes, s)
        if kind == "int4unpack":
            packed, s = _pack_int4_group(_src(mode, (M, K), g, device, 1.0), group)
            return (packed, s)
        if kind == "mxfp4unpack":
            packed, e8 = _quant_mxfp4(_src(mode, (M, K), g, device, 1.0))
            return (packed, e8)
        # kvdequant
        kq, ksc = _quant_nd(_src(mode, (M, K), g, device, 1.0), fmt, "token")
        vq, vsc = _quant_nd(_src(mode, (M, K), g, device, 0.9), fmt, "token")
        return (kq, ksc, vq, vsc)

    def _forward(inputs):
        if kind == "quant":
            return _quant_nd(inputs[0].float(), fmt, gran)
        if kind == "dequant":
            return _dequant_nd(inputs[0], inputs[1], gran).to(out_f)
        if kind == "kvquant":
            kq, ksc = _quant_nd(inputs[0].float(), fmt, "token")
            vq, vsc = _quant_nd(inputs[1].float(), fmt, "token")
            return (kq, ksc, vq, vsc)
        if kind == "kvdequant":
            k = _dequant_nd(inputs[0], inputs[1], "token").to(out_f)
            v = _dequant_nd(inputs[2], inputs[3], "token").to(out_f)
            return (k, v)
        if kind == "int4pack":
            return _pack_int4_group(inputs[0].float(), group)
        if kind == "int4unpack":
            return _unpack_int4_group(inputs[0], inputs[1], group).to(out_f)
        if kind == "mxfp4pack":
            return _quant_mxfp4(inputs[0].float())
        if kind == "mxfp4unpack":
            return _dequant_mxfp4(inputs[0], inputs[1]).to(out_f)
        if kind == "smooth":
            return _smooth_quant(inputs[0].float(), inputs[1].float(), fmt)
        if kind == "double":
            return _double_quant(inputs[0].float(), fmt)
        if kind == "stochastic":
            return _stoch_quant(inputs[0].float(), inputs[1], fmt)
        # qtranspose
        return _quant_transpose(inputs[0].float(), fmt, gran)

    def get_inputs(shape, device="cuda", seed=0):
        return _gen(shape, device, seed, "randn")

    def ref_fn(*inputs):
        return _forward(inputs)

    def baseline_fn(*inputs):
        # torch eager production path materialises the "in" dtype (bf16) before the
        # quant math, exactly like the shipped serving quantizers; the reduction is
        # otherwise identical to the fp32 oracle (same codes + scale).
        return _forward(inputs)

    def adversarial_inputs(shape, device="cuda"):
        return [(m, _gen(shape, device, 0, m))
                for m in ("zeros", "large", "neg_large", "small", "sign_alt")]

    ns = {"parse_shape": _parse_shape, "get_inputs": get_inputs, "ref_fn": ref_fn,
          "baseline_fn": baseline_fn, "arity": arity_of(op), "entry_name": op,
          "dtype_name": dtype, "family": f"breadth_{op}", "mutates_input": False,
          "adversarial_inputs": adversarial_inputs}
    ns[f"{op}_ref"] = ref_fn
    return ns


# --------------------------------------------------------------------------- #
# Naive (correct, compiling) Triton seeds - the policy's starting point.
# host amax / nibble-pack (torch) + a tiled elementwise quantize/dequantize
# kernel; the policy fuses the reduction + scaled round-and-pack into ONE kernel.
# --------------------------------------------------------------------------- #
_QHELP = '''

FP8_MAX = 448.0
INT8_MAX = 127.0
BLK = 128
MX_BLOCK = 32
E2M1_MAX = 6.0
E2M1_EMAX = 2
_E2M1_MIDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@triton.jit
def _qx_q_kernel(x_ptr, inv_ptr, o_ptr, n, LO, HI, ROUND: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + offs, mask=m, other=0.0).to(tl.float32)
    v = x * inv
    if ROUND:
        v = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
    v = tl.minimum(tl.maximum(v, LO), HI)
    tl.store(o_ptr + offs, v.to(o_ptr.dtype.element_ty), mask=m)


@triton.jit
def _qx_dq_kernel(c_ptr, s_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    c = tl.load(c_ptr + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs, (c * s).to(o_ptr.dtype.element_ty), mask=m)


def _q_map(xf, inv, lo, hi, do_round, out_dtype):
    xf, inv = xf.contiguous(), inv.contiguous()
    o = torch.empty(xf.shape, device=xf.device, dtype=out_dtype)
    n = xf.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _qx_q_kernel[grid](xf, inv, o, n, lo, hi, ROUND=(1 if do_round else 0), BLOCK=BLOCK)
    return o


def _dq_map(codes, scale_full, out_dtype):
    codes, scale_full = codes.contiguous(), scale_full.contiguous()
    o = torch.empty(codes.shape, device=codes.device, dtype=out_dtype)
    n = codes.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _qx_dq_kernel[grid](codes, scale_full, o, n, BLOCK=BLOCK)
    return o


def _e2m1_codes(v):
    sign = (v < 0).to(torch.uint8)
    mids = torch.tensor(_E2M1_MIDS, dtype=torch.float32, device=v.device)
    idx = torch.bucketize(v.abs(), mids).to(torch.uint8)
    return (sign << 3) | idx


def _e2m1_levels_lut(device):
    return torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=device)


def _fp8_levels(device):
    b = torch.arange(256, dtype=torch.uint8, device=device)
    lv = b.view(torch.float8_e4m3fn).float()
    return torch.unique(lv[torch.isfinite(lv)])
'''


def _qz_lines(v: str, fmt: str, gran: str) -> list[str]:
    """Body lines producing ``codes`` + ``scale`` from float tensor ``v`` at ``gran``."""
    mx = "FP8_MAX" if fmt == "fp8" else "INT8_MAX"
    outdt = "torch.float8_e4m3fn" if fmt == "fp8" else "torch.int8"
    rnd = "False" if fmt == "fp8" else "True"
    lo, hi = f"(-{mx})", mx
    qm = f"_q_map(xf, inv, {lo}, {hi}, {rnd}, {outdt})"
    L = [f"    xf = {v}.float()"]
    if gran == "tensor":
        L += ["    amax = xf.abs().amax().clamp(min=1e-12)",
              f"    scale = (amax / {mx}).reshape(())",
              "    inv = (1.0 / scale).expand_as(xf)",
              f"    codes = {qm}",
              "    scale = scale.to(torch.float32)"]
    elif gran == "token":
        L += ["    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)",
              f"    scale = amax / {mx}",
              "    inv = (1.0 / scale).expand_as(xf)",
              f"    codes = {qm}",
              "    scale = scale.to(torch.float32)"]
    elif gran == "channel":
        L += ["    amax = xf.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)",
              f"    scale = amax / {mx}",
              "    inv = (1.0 / scale).expand_as(xf)",
              f"    codes = {qm}",
              "    scale = scale.to(torch.float32)"]
    elif gran == "block128":
        L += ["    M, K = xf.shape",
              "    xb = xf.reshape(M, K // BLK, BLK)",
              f"    sb = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / {mx}",
              "    inv = (1.0 / sb).expand(M, K // BLK, BLK).reshape(M, K)",
              f"    codes = {qm}",
              "    scale = sb.squeeze(-1).to(torch.float32)"]
    else:  # block2d
        L += ["    M, K = xf.shape",
              "    xb = xf.reshape(M // BLK, BLK, K // BLK, BLK)",
              f"    sb = xb.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / {mx}",
              "    inv = (1.0 / sb).expand(M // BLK, BLK, K // BLK, BLK).reshape(M, K)",
              f"    codes = {qm}",
              "    scale = sb.squeeze(1).squeeze(-1).to(torch.float32)"]
    return L


def _dq_scale_full(gran: str) -> str:
    if gran in ("token", "channel"):
        return "    sf = scale.float().expand(codes.shape[0], codes.shape[1])"
    if gran == "tensor":
        return "    sf = scale.float().reshape(1, 1).expand(codes.shape[0], codes.shape[1])"
    if gran == "block128":
        return "    sf = scale.float().repeat_interleave(BLK, dim=1)"
    return "    sf = scale.float().repeat_interleave(BLK, 0).repeat_interleave(BLK, 1)"


def _seed_entry(cfg: dict, op: str) -> str:
    kind, fmt, gran, group = cfg["kind"], cfg["fmt"], cfg["gran"], cfg["group"]
    mx = "FP8_MAX" if fmt == "fp8" else "INT8_MAX"
    outdt = "torch.float8_e4m3fn" if fmt == "fp8" else "torch.int8"
    rnd = "False" if fmt == "fp8" else "True"
    lo, hi = f"(-{mx})", mx
    args = ", ".join(_arg_names(cfg))
    L = [f"def {op}({args}):"]
    if kind == "quant":
        L += _qz_lines("x", fmt, gran) + ["    return codes, scale"]
    elif kind == "dequant":
        L += [_dq_scale_full(gran), "    return _dq_map(codes, sf, torch.bfloat16)"]
    elif kind == "kvquant":
        L += [f"    ks = k.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / {mx}",
              f"    kq = _q_map(k.float(), (1.0 / ks).expand_as(k.float()), {lo}, {hi}, {rnd}, {outdt})",
              f"    vs = v.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / {mx}",
              f"    vq = _q_map(v.float(), (1.0 / vs).expand_as(v.float()), {lo}, {hi}, {rnd}, {outdt})",
              "    return kq, ks.to(torch.float32), vq, vs.to(torch.float32)"]
    elif kind == "kvdequant":
        L += ["    ksf = ksc.float().expand(kq.shape[0], kq.shape[1])",
              "    vsf = vsc.float().expand(vq.shape[0], vq.shape[1])",
              "    return (_dq_map(kq, ksf, torch.bfloat16), _dq_map(vq, vsf, torch.bfloat16))"]
    elif kind == "int4pack":
        L += [f"    wf = w.float(); N, K = wf.shape; g = {group}",
              "    wb = wf.reshape(N, K // g, g)",
              "    sb = wb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0",
              "    inv = (1.0 / sb).expand(N, K // g, g).reshape(N, K)",
              "    qf = _q_map(wf, inv, -8.0, 7.0, True, torch.float32)",
              "    code = (qf.to(torch.int32) + 8).to(torch.uint8)",
              "    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()",
              "    return packed, sb.squeeze(-1).to(torch.float32)"]
    elif kind == "int4unpack":
        L += [f"    N, Kp = packed.shape; K = Kp * 2; g = {group}",
              "    q = torch.empty((N, K), dtype=torch.int32, device=packed.device)",
              "    q[:, 0::2] = (packed & 0xF).to(torch.int32) - 8",
              "    q[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32) - 8",
              "    sf = scale.float().repeat_interleave(g, dim=1)",
              "    return _dq_map(q.to(torch.float32), sf, torch.bfloat16)"]
    elif kind == "mxfp4pack":
        L += ["    xf = x.float(); R, K = xf.shape",
              "    xb = xf.reshape(R, K // MX_BLOCK, MX_BLOCK)",
              "    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-20)",
              "    exp = (torch.floor(torch.log2(amax)) - E2M1_EMAX).clamp(-127.0, 127.0)",
              "    e8m0 = (exp + 127.0).to(torch.uint8)",
              "    inv = (1.0 / torch.exp2(exp)).expand(R, K // MX_BLOCK, MX_BLOCK).reshape(R, K)",
              "    xqf = _q_map(xf, inv, -E2M1_MAX, E2M1_MAX, False, torch.float32)",
              "    codes = _e2m1_codes(xqf)",
              "    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()",
              "    return packed, e8m0.reshape(R, K // MX_BLOCK).contiguous()"]
    elif kind == "mxfp4unpack":
        L += ["    R, Kp = packed.shape; K = Kp * 2",
              "    codes = torch.empty((R, K), dtype=torch.uint8, device=packed.device)",
              "    codes[:, 0::2] = packed & 0xF",
              "    codes[:, 1::2] = (packed >> 4) & 0xF",
              "    levels = _e2m1_levels_lut(packed.device)",
              "    mag = levels[(codes & 0x7).long()]",
              "    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)",
              "    sf = torch.exp2(e8m0.float() - 127.0).repeat_interleave(MX_BLOCK, dim=1)",
              "    return _dq_map(sign * mag, sf, torch.bfloat16)"]
    elif kind == "smooth":
        L += [f"    xf = x.float() / smooth.float().reshape(1, -1)",
              "    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)",
              f"    scale = amax / {mx}",
              f"    codes = _q_map(xf, (1.0 / scale).expand_as(xf), {lo}, {hi}, {rnd}, {outdt})",
              "    return codes, scale.to(torch.float32)"]
    elif kind == "double":
        L += ["    xf = x.float(); M, K = xf.shape",
              "    xb = xf.reshape(M, K // BLK, BLK)",
              f"    sb = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / {mx}",
              "    inv = (1.0 / sb).expand(M, K // BLK, BLK).reshape(M, K)",
              f"    codes = _q_map(xf, inv, {lo}, {hi}, {rnd}, {outdt})",
              "    bscale = sb.squeeze(-1).to(torch.float32)",
              "    meta = (bscale.abs().amax().clamp(min=1e-12) / FP8_MAX).reshape(())",
              "    sc_inv = (1.0 / meta).expand_as(bscale)",
              "    sc_codes = _q_map(bscale, sc_inv, -FP8_MAX, FP8_MAX, False, torch.float8_e4m3fn)",
              "    return codes, sc_codes, meta.to(torch.float32)"]
    elif kind == "stochastic" and fmt == "int8":
        L += ["    xf = x.float()",
              "    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)",
              "    scale = amax / 127.0",
              "    v = _q_map(xf, (1.0 / scale).expand_as(xf), -1e30, 1e30, False, torch.float32)",
              "    codes = torch.floor(v + noise).clamp(-127.0, 127.0).to(torch.int8)",
              "    return codes, scale.to(torch.float32)"]
    elif kind == "stochastic":  # fp8
        L += ["    xf = x.float()",
              "    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)",
              "    scale = amax / 448.0",
              "    v = _q_map(xf, (1.0 / scale).expand_as(xf), -FP8_MAX, FP8_MAX, False, torch.float32)",
              "    lv = _fp8_levels(xf.device)",
              "    hi_idx = torch.searchsorted(lv, v, right=False).clamp(max=lv.numel() - 1)",
              "    lo_idx = (hi_idx - 1).clamp(min=0)",
              "    hi = lv[hi_idx]; lo = lv[lo_idx]; denom = hi - lo",
              "    p = torch.where(denom > 0, (v - lo) / denom, torch.zeros_like(v))",
              "    codes = torch.where(noise < p, hi, lo).to(torch.float8_e4m3fn)",
              "    return codes, scale.to(torch.float32)"]
    else:  # qtranspose
        L += _qz_lines("x", fmt, gran)
        L.append("    codesT = codes.t().contiguous()")
        if gran == "token":
            L.append("    scale = scale.reshape(1, -1)")
        L.append("    return codesT, scale")
    return "\n".join(L) + "\n"


def seed_source(op: str, dtype: str) -> str:
    if op not in _CFG:
        raise ValueError(f"unknown breadth quant op {op!r}")
    cfg = _CFG[op]
    doc = (f'"""GENERATED breadth quant seed: {op} ({dtype}). Naive host amax /\n'
           f'nibble-pack + a tiled elementwise quantize/dequantize kernel - a correct,\n'
           f'COMPILING starting point the KORE policy fuses into one fused quant kernel."""\n')
    imports = ("from __future__ import annotations\n"
               "import torch\n"
               "import triton\n"
               "import triton.language as tl\n")
    return doc + imports + _QHELP + "\n\n" + _seed_entry(cfg, op)


def op_names() -> list[str]:
    return list(OPS)
