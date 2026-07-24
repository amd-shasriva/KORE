"""Breadth QUANTIZED / MIXED-PRECISION GEMM frontier task-authoring engine.

Widens the KORE suite with the HARD matmul kernels that ARE the LLM compute
backbone and that carry the largest optimization headroom on MI350X / gfx950
(CDNA4): quantized and mixed-precision GEMM with fused epilogues. A naive
dequantize-then-hipBLASLt path leaves a fused Triton kernel (dequant + tiled
tl.dot + epilogue kept in registers) far ahead, so every op here is a genuine
"hard for a GPU" kernel - no memory-bound trivia.

Coverage (every op name prefixed ``gemm_``; 45 distinct ops):
  * Quant formats  : fp8-e4m3, int8, int4 (packed sym / group sym / group asym
        AWQ), MXFP4 (block-32 e2m1+e8m0), MXFP8 (block-32 fp8), w4a16, w8a16,
        w8a8, w4a8 (mixed 4/8-bit).
  * Scale granularity (fp8 & int8): per-tensor / per-row(token) A / per-channel
        (col) W / block-128 (DeepSeek-V3 1x128 act, 128x128 weight).
  * Fused epilogues on fp8/int8/bf16 GEMM: +bias, +bias+gelu/relu/silu,
        +residual-add, +requant-to-fp8, +dequant-to-bf16.
  * Shape regimes (distinct kernels): square / tall-skinny / fat-K / skinny-K /
        batched GEMM / grouped GEMM (variable-M expert routing).
  * Dense bf16/fp16 GEMM (+epilogue) for contrast + GEMM backward dA / dB.

Contract mirrors ``kore/tasks/breadth/conv_ext.py`` (and ``vendor_ops.py``) so the
shared ``_genops`` driver + the breadth generator consume it unchanged:

    OPS / OP_DTYPES / SHAPES              module-level task catalog
    make_reference(op, dtype) -> dict     reference.py namespace (parse_shape,
        get_inputs, ref_fn EXACT fp32 dequant-matmul-epilogue oracle -> out dtype,
        baseline_fn torch eager production path, arity, entry_name, dtype_name,
        family=f"breadth_{op}", mutates_input=False, and adversarial_inputs for
        the quantized ops so the fp8/int8/int4 code + scale STRUCTURE survives).
    seed_source(op, dtype) -> str         a naive, COMPILING, CORRECT Triton seed
        (host dequant + a tiled tl.dot GEMM + epilogue) defining ``def <op>(...)``.

CORRECTNESS is paramount and follows the vendor convention: ``ref_fn`` dequantizes
the SAME quantized operands with their scales, matmuls in fp32, applies the
epilogue, and casts to the output dtype (bf16 for every quantized op; the task
dtype for the dense ops) - so the gate measures the kernel's accumulation +
epilogue fidelity, NOT the (shared) quantization error. Every oracle is validated
on CPU against an INDEPENDENT torch computation at tight fp32 tolerance - see
tests/test_gemm_ext.py.

SAFETY / SELF-CONTAINED: we deliberately do NOT import ``kore.tasks.aiter_ref`` (it
touches ``torch.cuda`` at import). The OCP fp8-e4m3 max (448.0) is defined locally
and every reference is a pure, CPU-importable torch computation (torch/triton are
imported lazily inside the GPU paths so registry discovery never needs a GPU).
"""

from __future__ import annotations

from kore.tasks._genops import DTYPES, _parse_shape  # noqa: F401  (DTYPES re-exported for parity)

# --------------------------------------------------------------------------- #
# Local quant constants (self-contained; NO aiter_ref -> no torch.cuda touch).
# --------------------------------------------------------------------------- #
FP8_MAX = 448.0                     # OCP float8_e4m3fn max finite (gfx950/CDNA4)
INT8_MAX = 127.0                    # symmetric int8 range
INT4_MIN, INT4_MAX = -8, 7          # symmetric int4 signed range
UINT4_MAX = 15                      # asymmetric int4 (AWQ zero-point) code range
BLK = 128                           # DeepSeek-V3 block-scale group (1x128 / 128x128)
MX_BLOCK = 32                       # OCP microscaling group along K
E2M1_MAX = 6.0                      # max |value| in e2m1
E2M1_EMAX = 2                       # exponent of the e2m1 max normal (6.0 = 1.5*2^2)
_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   # e2m1 magnitudes (idx 0..7)
_E2M1_MIDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]      # round-to-nearest cuts


# --------------------------------------------------------------------------- #
# Task catalog: op -> config dict (single source of truth). Fields:
#   fam   : "plain" | "batched" | "grouped" | "dgrad" | "wgrad"
#   aq/wq : activation / weight format ("bf16","fp16","fp8","int8","int4c",
#           "int4gs","int4ga","mxfp4","mxfp8")
#   asg/wsg: 8-bit scale granularity ("tensor","row"/"channel","block128","group")
#   group : int4 group size ; ep: fused epilogue ; regime: shape catalog key.
# --------------------------------------------------------------------------- #
def _auto_dt(aq: str, wq: str) -> str:
    """OP_DTYPES label (must be a DTYPES key): fp8/int8 for those formats, else the
    activation float dtype (bf16/fp16). Purely a label - quantized ops always emit
    a bf16 output regardless."""
    if aq == "fp8" or wq == "fp8":
        return "fp8"
    if aq == "int8" or wq == "int8":
        return "int8"
    if aq == "fp16" or wq == "fp16":
        return "fp16"
    return "bf16"


def _c(fam="plain", aq="bf16", wq="bf16", asg="tensor", wsg="tensor",
       group=0, ep="none", regime="square") -> dict:
    cfg = {"fam": fam, "aq": aq, "wq": wq, "asg": asg, "wsg": wsg,
           "group": group, "ep": ep, "regime": regime}
    cfg["dt"] = _auto_dt(aq, wq)
    return cfg


_CFG: dict[str, dict] = {
    # ---- fp8-e4m3 scale granularity (per-tensor / per-token+channel / block-128)
    "gemm_fp8_pertensor":   _c(aq="fp8", wq="fp8", asg="tensor", wsg="tensor"),
    "gemm_fp8_rowwise":     _c(aq="fp8", wq="fp8", asg="row", wsg="channel"),
    "gemm_fp8_channelwise": _c(aq="fp8", wq="fp8", asg="tensor", wsg="channel"),
    "gemm_fp8_block128":    _c(aq="fp8", wq="fp8", asg="block128", wsg="block128"),
    # ---- int8 scale granularity ---------------------------------------------
    "gemm_int8_pertensor":   _c(aq="int8", wq="int8", asg="tensor", wsg="tensor"),
    "gemm_int8_rowwise":     _c(aq="int8", wq="int8", asg="row", wsg="channel"),
    "gemm_int8_channelwise": _c(aq="int8", wq="int8", asg="tensor", wsg="channel"),
    "gemm_int8_block128":    _c(aq="int8", wq="int8", asg="block128", wsg="block128"),
    # ---- 4-bit + mixed-precision formats ------------------------------------
    "gemm_int4_sym_channel": _c(aq="bf16", wq="int4c", wsg="channel"),
    "gemm_int4_asym_group":  _c(aq="fp16", wq="int4ga", wsg="group", group=128),
    "gemm_int4_sym_group":   _c(aq="bf16", wq="int4gs", wsg="group", group=128),
    "gemm_w4a8":             _c(aq="fp8", wq="int4c", asg="row", wsg="channel"),
    "gemm_w8a16":            _c(aq="bf16", wq="int8", wsg="channel"),
    "gemm_mxfp4":            _c(aq="mxfp4", wq="mxfp4"),
    "gemm_mxfp4_weight":     _c(aq="bf16", wq="mxfp4"),
    "gemm_mxfp8":            _c(aq="mxfp8", wq="mxfp8"),
    # ---- fused epilogues on fp8 / int8 / (int4) GEMM ------------------------
    "gemm_fp8_bias":         _c(aq="fp8", wq="fp8", asg="row", wsg="channel", ep="bias"),
    "gemm_fp8_bias_gelu":    _c(aq="fp8", wq="fp8", asg="row", wsg="channel", ep="bias_gelu"),
    "gemm_fp8_bias_relu":    _c(aq="fp8", wq="fp8", asg="row", wsg="channel", ep="bias_relu"),
    "gemm_fp8_bias_silu":    _c(aq="fp8", wq="fp8", asg="row", wsg="channel", ep="bias_silu"),
    "gemm_fp8_residual":     _c(aq="fp8", wq="fp8", asg="row", wsg="channel", ep="residual"),
    "gemm_fp8_requant":      _c(aq="fp8", wq="fp8", asg="row", wsg="channel", ep="requant"),
    "gemm_int8_bias_gelu":   _c(aq="int8", wq="int8", asg="row", wsg="channel", ep="bias_gelu"),
    "gemm_int8_bias_relu":   _c(aq="int8", wq="int8", asg="row", wsg="channel", ep="bias_relu"),
    "gemm_int8_bias_silu":   _c(aq="int8", wq="int8", asg="row", wsg="channel", ep="bias_silu"),
    "gemm_int8_dequant_bf16": _c(aq="int8", wq="int8", asg="row", wsg="channel", ep="bias"),
    "gemm_int8_residual":    _c(aq="int8", wq="int8", asg="row", wsg="channel", ep="residual"),
    "gemm_w4a16_bias_gelu":  _c(aq="fp16", wq="int4ga", wsg="group", group=128, ep="bias_gelu"),
    # ---- shape regimes (distinct kernels) -----------------------------------
    "gemm_fp8_tall_skinny":  _c(aq="fp8", wq="fp8", asg="row", wsg="channel", regime="tall_skinny"),
    "gemm_fp8_fat_k":        _c(aq="fp8", wq="fp8", asg="row", wsg="channel", regime="fat_k"),
    "gemm_fp8_skinny_k":     _c(aq="fp8", wq="fp8", asg="row", wsg="channel", regime="skinny_k"),
    "gemm_fp8_batched":      _c(fam="batched", aq="fp8", wq="fp8", asg="row", wsg="channel", regime="batched"),
    "gemm_fp8_grouped":      _c(fam="grouped", aq="fp8", wq="fp8", asg="row", wsg="channel", regime="grouped"),
    "gemm_bf16_grouped":     _c(fam="grouped", aq="bf16", wq="bf16", regime="grouped"),
    "gemm_bf16_batched":     _c(fam="batched", aq="bf16", wq="bf16", regime="batched"),
    # ---- dense bf16/fp16 (+epilogue) contrast + backward --------------------
    "gemm_bf16":             _c(aq="bf16", wq="bf16"),
    "gemm_fp16":             _c(aq="fp16", wq="fp16"),
    "gemm_bf16_bias":        _c(aq="bf16", wq="bf16", ep="bias"),
    "gemm_bf16_bias_gelu":   _c(aq="bf16", wq="bf16", ep="bias_gelu"),
    "gemm_bf16_bias_relu":   _c(aq="bf16", wq="bf16", ep="bias_relu"),
    "gemm_bf16_bias_silu":   _c(aq="bf16", wq="bf16", ep="bias_silu"),
    "gemm_fp16_bias_gelu":   _c(aq="fp16", wq="fp16", ep="bias_gelu"),
    "gemm_bf16_residual":    _c(aq="bf16", wq="bf16", ep="residual"),
    "gemm_backward_da":      _c(fam="dgrad", aq="bf16", wq="bf16"),
    "gemm_backward_db":      _c(fam="wgrad", aq="bf16", wq="bf16"),
}

OPS: list[str] = list(_CFG)
OP_DTYPES: dict[str, list[str]] = {op: [_CFG[op]["dt"]] for op in OPS}


# --------------------------------------------------------------------------- #
# small config predicates (module-level, pure)
# --------------------------------------------------------------------------- #
def _is_quant(cfg: dict) -> bool:
    return cfg["aq"] not in ("bf16", "fp16") or cfg["wq"] not in ("bf16", "fp16")


def _has_bias(cfg: dict) -> bool:
    return cfg["ep"] in ("bias", "bias_gelu", "bias_relu", "bias_silu", "requant")


def _act_of(ep: str) -> str:
    if ep.endswith("gelu"):
        return "gelu"
    if ep.endswith("relu"):
        return "relu"
    if ep.endswith("silu"):
        return "silu"
    return "none"


def _a_names(cfg: dict) -> list[str]:
    aq = cfg["aq"]
    if aq in ("bf16", "fp16"):
        return ["a"]
    if aq in ("fp8", "int8"):
        return ["aq", "asc"]
    if aq == "mxfp4":
        return ["apk", "ae8"]
    if aq == "mxfp8":
        return ["aq", "ae8"]
    raise ValueError(f"bad aq {aq!r}")


def _w_names(cfg: dict) -> list[str]:
    wq = cfg["wq"]
    if wq in ("bf16", "fp16"):
        return ["w"]
    if wq in ("fp8", "int8"):
        return ["wq", "wsc"]
    if wq in ("int4c", "int4gs"):
        return ["wpk", "wsc"]
    if wq == "int4ga":
        return ["wpk", "wsc", "wz"]
    if wq == "mxfp4":
        return ["wpk", "we8"]
    if wq == "mxfp8":
        return ["wq", "we8"]
    raise ValueError(f"bad wq {wq!r}")


def _arg_names(cfg: dict) -> list[str]:
    fam = cfg["fam"]
    if fam == "dgrad":
        return ["dy", "w"]
    if fam == "wgrad":
        return ["dy", "a"]
    if fam == "batched":
        return ["aq", "asc", "wq", "wsc"] if _is_quant(cfg) else ["a", "w"]
    if fam == "grouped":
        return ["aq", "asc", "wq", "wsc", "eids"] if _is_quant(cfg) else ["a", "w", "eids"]
    names = _a_names(cfg) + _w_names(cfg)
    if _has_bias(cfg):
        names.append("bias")
    if cfg["ep"] == "residual":
        names.append("res")
    if cfg["ep"] == "requant":
        names.append("osc")
    return names


def arity_of(op: str) -> int:
    return len(_arg_names(_CFG[op]))


# --------------------------------------------------------------------------- #
# Shape catalog: realistic LLM GEMM shapes per regime (K/N snapped to the format
# divisibility; a non-power-of-2 tail stresses masking). Never executed on CPU
# (tests use tiny shapes); only round-tripped through parse_shape.
# --------------------------------------------------------------------------- #
_REGIME_BASE = {
    "square": {"minimal": (256, 256, 256), "primary": (4096, 4096, 4096),
               "validation": [(2048, 4096, 4096), (8192, 2048, 4096), (8193, 4096, 4064)]},
    "tall_skinny": {"minimal": (512, 128, 128), "primary": (16384, 1024, 1024),
                    "validation": [(8192, 2048, 1024), (16384, 512, 2048), (16385, 1024, 1024)]},
    "fat_k": {"minimal": (128, 128, 512), "primary": (1024, 1024, 16384),
              "validation": [(2048, 1024, 8192), (1024, 2048, 16384), (1025, 1024, 16384)]},
    "skinny_k": {"minimal": (256, 256, 128), "primary": (8192, 8192, 512),
                 "validation": [(4096, 8192, 512), (8192, 4096, 256), (8193, 8192, 512)]},
}


def _kmult(cfg: dict) -> int:
    if cfg["asg"] == "block128" or cfg["wsg"] == "block128":
        return 128
    if "mxfp4" in (cfg["aq"], cfg["wq"]) or "mxfp8" in (cfg["aq"], cfg["wq"]):
        return 32
    if cfg["wq"] in ("int4gs", "int4ga"):
        return cfg["group"]
    if cfg["wq"] == "int4c":
        return 2
    return 1


def _nmult(cfg: dict) -> int:
    return 128 if cfg["wsg"] == "block128" else 1


def _snap(v: int, m: int) -> int:
    return v if m <= 1 else max(m, (v // m) * m)


def _shapes_for(cfg: dict) -> dict:
    km, nm, fam = _kmult(cfg), _nmult(cfg), cfg["fam"]
    if fam == "batched":
        base = {"minimal": (2, 64, 128, 128), "primary": (8, 512, 4096, 4096),
                "validation": [(16, 256, 4096, 4096), (4, 512, 8192, 4096), (8, 513, 4096, 4096)]}
        f = lambda t: {"B": t[0], "M": t[1], "N": _snap(t[2], nm), "K": _snap(t[3], km)}
        return {"minimal": f(base["minimal"]), "primary": f(base["primary"]),
                "validation": [f(t) for t in base["validation"]]}
    if fam == "grouped":
        base = {"minimal": (2, 64, 128, 128), "primary": (8, 4096, 4096, 4096),
                "validation": [(16, 8192, 4096, 4096), (8, 4096, 2048, 4096), (8, 8193, 4096, 4096)]}
        f = lambda t: {"E": t[0], "M": t[1], "N": _snap(t[2], nm), "K": _snap(t[3], km)}
        return {"minimal": f(base["minimal"]), "primary": f(base["primary"]),
                "validation": [f(t) for t in base["validation"]]}
    reg = "square" if fam in ("dgrad", "wgrad") else cfg["regime"]
    b = _REGIME_BASE[reg]
    f = lambda t: {"M": t[0], "N": _snap(t[1], nm), "K": _snap(t[2], km)}
    return {"minimal": f(b["minimal"]), "primary": f(b["primary"]),
            "validation": [f(t) for t in b["validation"]]}


SHAPES: dict[str, dict] = {op: _shapes_for(_CFG[op]) for op in OPS}


# --------------------------------------------------------------------------- #
# QUANT ENCODE helpers (float -> codes + scales). Used by get_inputs.
# --------------------------------------------------------------------------- #
def _to_codes(q, fmt: str):
    import torch
    if fmt == "fp8":
        return q.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)


def _quant8_a(x, fmt: str, asg: str):
    """8-bit (fp8/int8) activation quant. -> (codes, scale) with scale granularity
    () per-tensor / [M,1] per-row / [M,K//128] block-128."""
    import torch
    mx = FP8_MAX if fmt == "fp8" else INT8_MAX
    xf = x.float()
    if asg == "tensor":
        s = (xf.abs().amax().clamp(min=1e-12) / mx)
        q, s = xf / s, s.reshape(())
    elif asg == "row":
        s = (xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / mx)
        q = xf / s
    else:  # block128
        M, K = xf.shape
        xb = xf.reshape(M, K // BLK, BLK)
        s = (xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / mx)
        q, s = (xb / s).reshape(M, K), s.squeeze(-1)
    return (_to_codes(q, fmt), s.to(torch.float32).contiguous())


def _quant8_w(x, fmt: str, wsg: str):
    """8-bit weight quant. -> (codes, scale) with () per-tensor / [N,1] per-channel /
    [N//128,K//128] 128x128 block."""
    import torch
    mx = FP8_MAX if fmt == "fp8" else INT8_MAX
    xf = x.float()
    if wsg == "tensor":
        s = (xf.abs().amax().clamp(min=1e-12) / mx)
        q, s = xf / s, s.reshape(())
    elif wsg == "channel":
        s = (xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / mx)
        q = xf / s
    else:  # block128 (128x128)
        N, K = xf.shape
        wb = xf.reshape(N // BLK, BLK, K // BLK, BLK)
        s = (wb.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / mx)
        q, s = (wb / s).reshape(N, K), s.squeeze(1).squeeze(-1)
    return (_to_codes(q, fmt), s.to(torch.float32).contiguous())


def _quant_rowwise_fp8(x):
    """Per-last-dim-row symmetric fp8 (works for [M,K], [B,M,K], [E,N,K])."""
    import torch
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    xq = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (xq, scale)


def _quant_int4_sym_channel(w):
    """Symmetric per-output-channel int4. w[N,K] -> (packed[N,K//2] uint8, scale[N,1]);
    w ~= (code-8)*scale[n]."""
    import torch
    amax = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / INT4_MAX
    q = torch.round(w.float() / scale).clamp(INT4_MIN, INT4_MAX).to(torch.int32)
    code = (q + 8).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return (packed, scale.to(torch.float32).contiguous())


def _quant_int4_sym_group(w, group: int):
    """Symmetric group-wise int4. -> (packed[N,K//2], scale[N,K//group])."""
    import torch
    N, K = w.shape
    wb = w.float().reshape(N, K // group, group)
    amax = wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    scale = amax / INT4_MAX
    q = torch.round(wb / scale).clamp(INT4_MIN, INT4_MAX).reshape(N, K).to(torch.int32)
    code = (q + 8).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return (packed, scale.squeeze(-1).to(torch.float32).contiguous())


def _quant_int4_asym_group(w, group: int):
    """Asymmetric (zero-point) group-wise int4 (AWQ/GPTQ). -> (packed[N,K//2],
    scale[N,K//group], zero[N,K//group] uint8); w ~= (code - zero)*scale."""
    import torch
    N, K = w.shape
    wb = w.float().reshape(N, K // group, group)
    wmin = wb.amin(dim=2, keepdim=True)
    wmax = wb.amax(dim=2, keepdim=True)
    scale = ((wmax - wmin) / float(UINT4_MAX)).clamp(min=1e-12)
    zero = torch.round(-wmin / scale).clamp(0, UINT4_MAX)
    code = torch.round(wb / scale + zero).clamp(0, UINT4_MAX).reshape(N, K).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return (packed, scale.squeeze(-1).to(torch.float32).contiguous(),
            zero.squeeze(-1).to(torch.uint8).contiguous())


def _e2m1_codes(v):
    import torch
    sign = (v < 0).to(torch.uint8)
    mids = torch.tensor(_E2M1_MIDS, dtype=torch.float32, device=v.device)
    idx = torch.bucketize(v.abs(), mids).to(torch.uint8)   # 0..7
    return (sign << 3) | idx


def _quant_mxfp4(x):
    """OCP MXFP4: x[R,K] -> (packed[R,K//2] uint8, e8m0[R,K//32] uint8). Per 32-K
    group: shared exponent floor(log2(amax))-EMAX (biased +127), e2m1 codes."""
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


def _quant_mxfp8(x):
    """OCP MXFP8 (e4m3 elements + e8m0/32 block scale). x[R,K] -> (codes fp8[R,K],
    e8m0[R,K//32]); scale is the power-of-2 2^(ceil(log2(amax/FP8_MAX)))."""
    import torch
    R, K = x.shape
    xb = x.float().reshape(R, K // MX_BLOCK, MX_BLOCK)
    amax = xb.abs().amax(dim=2, keepdim=True).clamp(min=1e-20)
    exp = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(-127.0, 127.0)
    e8m0 = (exp + 127.0).to(torch.uint8)
    scale = torch.exp2(exp)
    xq = (xb / scale).clamp(-FP8_MAX, FP8_MAX).reshape(R, K).to(torch.float8_e4m3fn)
    return (xq, e8m0.reshape(R, K // MX_BLOCK).contiguous())


# --------------------------------------------------------------------------- #
# DEQUANT helpers (codes + scales -> fp32 dense operand). Scale applied ONCE.
# --------------------------------------------------------------------------- #
def _deq_a8(codes, s, gran: str):
    c = codes.float()
    if gran == "block128":
        return c * s.float().repeat_interleave(BLK, dim=1)
    return c * s.float()                       # () or [M,1] broadcast


def _deq_w8(codes, s, gran: str):
    c = codes.float()
    if gran == "block128":
        return c * s.float().repeat_interleave(BLK, 0).repeat_interleave(BLK, 1)
    return c * s.float()                        # () or [N,1] broadcast


def _deq_int4c(packed, scale):
    import torch
    N, K = packed.shape[0], packed.shape[1] * 2
    q = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    q[:, 0::2] = (packed & 0xF).to(torch.int32) - 8
    q[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32) - 8
    return q.float() * scale.float()


def _deq_int4gs(packed, scale, group: int):
    import torch
    N, K = packed.shape[0], packed.shape[1] * 2
    q = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    q[:, 0::2] = (packed & 0xF).to(torch.int32) - 8
    q[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32) - 8
    return q.float() * scale.float().repeat_interleave(group, dim=1)


def _deq_int4ga(packed, scale, zero, group: int):
    import torch
    N, K = packed.shape[0], packed.shape[1] * 2
    codes = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    codes[:, 0::2] = (packed & 0xF).to(torch.int32)
    codes[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32)
    z = zero.to(torch.int32).repeat_interleave(group, dim=1)
    s = scale.float().repeat_interleave(group, dim=1)
    return (codes.float() - z.float()) * s


def _deq_mxfp4(packed, e8m0):
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


def _deq_mxfp8(codes, e8m0):
    import torch
    scale = torch.exp2(e8m0.float() - 127.0).repeat_interleave(MX_BLOCK, dim=1)
    return codes.float() * scale


# --------------------------------------------------------------------------- #
# operand readers (positional; consume 1..3 inputs) -> fp32 [., K]
# --------------------------------------------------------------------------- #
def _read_A(cfg, inp, i):
    aq = cfg["aq"]
    if aq in ("bf16", "fp16"):
        return inp[i].float(), i + 1
    if aq in ("fp8", "int8"):
        return _deq_a8(inp[i], inp[i + 1], cfg["asg"]), i + 2
    if aq == "mxfp4":
        return _deq_mxfp4(inp[i], inp[i + 1]), i + 2
    return _deq_mxfp8(inp[i], inp[i + 1]), i + 2   # mxfp8


def _read_W(cfg, inp, i):
    wq = cfg["wq"]
    if wq in ("bf16", "fp16"):
        return inp[i].float(), i + 1
    if wq in ("fp8", "int8"):
        return _deq_w8(inp[i], inp[i + 1], cfg["wsg"]), i + 2
    if wq == "int4c":
        return _deq_int4c(inp[i], inp[i + 1]), i + 2
    if wq == "int4gs":
        return _deq_int4gs(inp[i], inp[i + 1], cfg["group"]), i + 2
    if wq == "int4ga":
        return _deq_int4ga(inp[i], inp[i + 1], inp[i + 2], cfg["group"]), i + 3
    if wq == "mxfp4":
        return _deq_mxfp4(inp[i], inp[i + 1]), i + 2
    return _deq_mxfp8(inp[i], inp[i + 1]), i + 2   # mxfp8


def _act(y, name: str):
    import torch.nn.functional as F
    if name == "gelu":
        return F.gelu(y, approximate="tanh")
    if name == "relu":
        return F.relu(y)
    if name == "silu":
        return F.silu(y)
    return y


def _grouped_ref(A, W, eids):
    """Variable-M grouped GEMM: out[m] = A[m] @ W[eids[m]].T (fp32). 0-token experts
    are simply never visited (the MoE load-imbalance edge)."""
    import torch
    M, K = A.shape
    E, N, _ = W.shape
    out = torch.zeros((M, N), device=A.device, dtype=torch.float32)
    e = eids.long()
    for ex in range(E):
        idx = (e == ex).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        out[idx] = A[idx] @ W[ex].t()
    return out


def _src(mode: str, shape, g, device, sc: float):
    """Float source tensor for get_inputs (randn) or the adversarial battery (hard
    regimes that must survive quantization + the scale structure)."""
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
    # sign_alt: alternating +-1 (parity built in int64 to avoid low-precision overflow)
    n = 1
    for d in shape:
        n *= d
    return ((torch.arange(n, device=device) % 2) * 2 - 1).to(torch.float32).reshape(shape)


# --------------------------------------------------------------------------- #
# reference.py namespace (exact fp32 dequant-matmul-epilogue oracle + torch eager
# production baseline). torch imported lazily so registry discovery is GPU-free.
# --------------------------------------------------------------------------- #
def make_reference(op: str, dtype: str) -> dict:
    import torch

    if op not in _CFG:
        raise ValueError(f"unknown breadth GEMM op {op!r}")
    cfg = _CFG[op]
    fam = cfg["fam"]
    quant = _is_quant(cfg)
    override = (dtype == "fp32")

    def _fdt(fmt):
        if override:
            return torch.float32
        return torch.float16 if fmt == "fp16" else torch.bfloat16

    out_dt = (torch.float32 if override else torch.bfloat16) if quant else _fdt(cfg["aq"])

    def _q_a(Af):
        aq = cfg["aq"]
        if aq in ("bf16", "fp16"):
            return (Af.to(_fdt(aq)),)
        if aq in ("fp8", "int8"):
            return _quant8_a(Af, aq, cfg["asg"])
        if aq == "mxfp4":
            return _quant_mxfp4(Af)
        return _quant_mxfp8(Af)

    def _q_w(Wf):
        wq = cfg["wq"]
        if wq in ("bf16", "fp16"):
            return (Wf.to(_fdt(wq)),)
        if wq in ("fp8", "int8"):
            return _quant8_w(Wf, wq, cfg["wsg"])
        if wq == "int4c":
            return _quant_int4_sym_channel(Wf)
        if wq == "int4gs":
            return _quant_int4_sym_group(Wf, cfg["group"])
        if wq == "int4ga":
            return _quant_int4_asym_group(Wf, cfg["group"])
        if wq == "mxfp4":
            return _quant_mxfp4(Wf)
        return _quant_mxfp8(Wf)

    def _gen(shape, device, seed, mode):
        g = torch.Generator(device=device).manual_seed(seed)
        if fam == "batched":
            B, M, N, K = shape["B"], shape["M"], shape["N"], shape["K"]
            sc = 1.0 / (K ** 0.5)
            Af = _src(mode, (B, M, K), g, device, sc)
            Wf = _src(mode, (B, N, K), g, device, sc)
            if quant:
                aq, asc = _quant_rowwise_fp8(Af)
                wq, wsc = _quant_rowwise_fp8(Wf)
                return (aq, asc, wq, wsc)
            return (Af.to(_fdt(cfg["aq"])), Wf.to(_fdt(cfg["wq"])))
        if fam == "grouped":
            E, M, N, K = shape["E"], shape["M"], shape["N"], shape["K"]
            sc = 1.0 / (K ** 0.5)
            Af = _src(mode, (M, K), g, device, sc)
            Wf = _src(mode, (E, N, K), g, device, sc)
            eids = torch.randint(0, E, (M,), generator=g, device=device).to(torch.int32)
            if quant:
                aq, asc = _quant_rowwise_fp8(Af)
                wq, wsc = _quant_rowwise_fp8(Wf)
                return (aq, asc, wq, wsc, eids)
            return (Af.to(_fdt(cfg["aq"])), Wf.to(_fdt(cfg["wq"])), eids)
        if fam in ("dgrad", "wgrad"):
            M, N, K = shape["M"], shape["N"], shape["K"]
            sc = 1.0 / ((N if fam == "dgrad" else M) ** 0.5)
            dy = _src(mode, (M, N), g, device, sc)
            other = _src(mode, (N, K) if fam == "dgrad" else (M, K), g, device, sc)
            return (dy.to(_fdt("bf16")), other.to(_fdt("bf16")))
        # plain
        M, N, K = shape["M"], shape["N"], shape["K"]
        sc = 1.0 / (K ** 0.5)
        Af = _src(mode, (M, K), g, device, sc)
        Wf = _src(mode, (N, K), g, device, sc)
        parts = list(_q_a(Af)) + list(_q_w(Wf))
        if _has_bias(cfg):
            parts.append((torch.randn((N,), generator=g, device=device,
                                      dtype=torch.float32) * sc).to(out_dt))
        if cfg["ep"] == "residual":
            parts.append((torch.randn((M, N), generator=g, device=device,
                                      dtype=torch.float32) * sc).to(out_dt))
        if cfg["ep"] == "requant":
            rms_a = Af.float().pow(2).mean().clamp(min=1e-12).sqrt()
            rms_w = Wf.float().pow(2).mean().clamp(min=1e-12).sqrt()
            osc = (rms_a * rms_w * (K ** 0.5) / FP8_MAX).clamp(min=1e-8)
            parts.append(osc.to(torch.float32).reshape(()))
        return tuple(parts)

    def _forward(inputs, base):
        if fam == "batched":
            if quant:
                aq, asc, wq, wsc = inputs
                A = aq.float() * asc.float()
                W = wq.float() * wsc.float()
                mm_dt = torch.bfloat16
            else:
                A, W = inputs[0].float(), inputs[1].float()
                mm_dt = out_dt
            if base:
                y = torch.bmm(A.to(mm_dt), W.to(mm_dt).transpose(1, 2)).float()
            else:
                y = torch.bmm(A, W.transpose(1, 2))
            return y.to(out_dt)
        if fam == "grouped":
            if quant:
                aq, asc, wq, wsc, eids = inputs
                A = aq.float() * asc.float()
                W = wq.float() * wsc.float()
            else:
                A, W, eids = inputs[0].float(), inputs[1].float(), inputs[2]
            return _grouped_ref(A, W, eids).to(out_dt)
        if fam == "dgrad":
            dy, w = inputs
            return (dy.float() @ w.float()).to(out_dt)          # dA = dY @ W
        if fam == "wgrad":
            dy, a = inputs
            return (dy.float().t() @ a.float()).to(out_dt)      # dB = dY^T @ A
        # plain
        i = 0
        A, i = _read_A(cfg, inputs, i)
        W, i = _read_W(cfg, inputs, i)
        if base:
            mm_dt = torch.bfloat16 if quant else out_dt
            y = torch.matmul(A.to(mm_dt), W.to(mm_dt).t().contiguous()).float()
        else:
            y = A @ W.t()
        if _has_bias(cfg):
            y = y + inputs[i].float().reshape(1, -1)
            i += 1
        y = _act(y, _act_of(cfg["ep"]))
        if cfg["ep"] == "residual":
            y = y + inputs[i].float()
            i += 1
        if cfg["ep"] == "requant":
            osc = inputs[i].float()
            i += 1
            y = (y / osc).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * osc
        return y.to(out_dt)

    def get_inputs(shape, device="cuda", seed=0):
        return _gen(shape, device, seed, "randn")

    def ref_fn(*inputs):
        return _forward(inputs, base=False)

    def baseline_fn(*inputs):
        return _forward(inputs, base=True)

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
# host dequant (torch) + a tiled tl.dot GEMM + host epilogue; the policy fuses the
# dequant + matmul + epilogue into ONE quantized-GEMM kernel.
# --------------------------------------------------------------------------- #
_QHELP = '''

FP8_MAX = 448.0
_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _dq_a8(codes, s, gran):
    c = codes.float()
    if gran == "block128":
        return c * s.float().repeat_interleave(128, dim=1)
    return c * s.float()


def _dq_w8(codes, s, gran):
    c = codes.float()
    if gran == "block128":
        return c * s.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return c * s.float()


def _dq_int4c(packed, scale):
    N, K = packed.shape[0], packed.shape[1] * 2
    q = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    q[:, 0::2] = (packed & 0xF).to(torch.int32) - 8
    q[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32) - 8
    return q.float() * scale.float()


def _dq_int4gs(packed, scale, group):
    N, K = packed.shape[0], packed.shape[1] * 2
    q = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    q[:, 0::2] = (packed & 0xF).to(torch.int32) - 8
    q[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32) - 8
    return q.float() * scale.float().repeat_interleave(group, dim=1)


def _dq_int4ga(packed, scale, zero, group):
    N, K = packed.shape[0], packed.shape[1] * 2
    codes = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    codes[:, 0::2] = (packed & 0xF).to(torch.int32)
    codes[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32)
    z = zero.to(torch.int32).repeat_interleave(group, dim=1)
    s = scale.float().repeat_interleave(group, dim=1)
    return (codes.float() - z.float()) * s


def _dq_mxfp4(packed, e8m0):
    R, K = packed.shape[0], packed.shape[1] * 2
    codes = torch.empty((R, K), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=packed.device)
    mag = levels[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    scale = torch.exp2(e8m0.float() - 127.0).repeat_interleave(32, dim=1)
    return (sign * mag) * scale


def _dq_mxfp8(codes, e8m0):
    scale = torch.exp2(e8m0.float() - 127.0).repeat_interleave(32, dim=1)
    return codes.float() * scale
'''

_GEMM_BLOCK = '''

@triton.jit
def _mm_nt_kernel(a_ptr, b_ptr, c_ptr, Mr, N, K,
                  sam, sak, sbn, sbk, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    a_ptrs = a_ptr + offm[:, None] * sam + offk[None, :] * sak
    b_ptrs = b_ptr + offn[:, None] * sbn + offk[None, :] * sbk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        km = offk[None, :] < (K - k0)
        a = tl.load(a_ptrs, mask=(offm[:, None] < Mr) & km, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offn[:, None] < N) & km, other=0.0).to(tl.float32)
        acc += tl.dot(a, tl.trans(b))
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    cmask = (offm[:, None] < Mr) & (offn[None, :] < N)
    tl.store(c_ptr + offm[:, None] * scm + offn[None, :] * scn,
             acc.to(c_ptr.dtype.element_ty), mask=cmask)


def _mm_nt(a, b):
    """a [m,K], b [N,K] -> a @ b.T (fp32 accumulate via tl.dot); out dtype = a.dtype."""
    m, K = a.shape
    N = b.shape[0]
    c = torch.empty((m, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(m, BM), triton.cdiv(N, BN))
    _mm_nt_kernel[grid](a, b, c, m, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
    return c


def _grouped_mm(x, w, expert_ids):
    """Per-expert grouped GEMM: out[m] = x[m] @ w[expert_ids[m]].T (naive: one GEMM
    launch per non-empty expert -- the bar a fused variable-M grouped kernel beats)."""
    M, K = x.shape
    E, N, _ = w.shape
    out = torch.zeros((M, N), device=x.device, dtype=x.dtype)
    eids = expert_ids.to(torch.long)
    for e in range(E):
        idx = (eids == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        ye = _mm_nt(x.index_select(0, idx).contiguous(), w[e].contiguous())
        out.index_copy_(0, idx, ye)
    return out
'''

_ACT_BLOCK = '''

@triton.jit
def _activation_kernel(y_ptr, n_elements,
                       GELU: tl.constexpr, RELU: tl.constexpr, SILU: tl.constexpr,
                       BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if GELU:
        z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        x = 0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * z) - 1.0))
    elif RELU:
        x = tl.maximum(x, 0.0)
    elif SILU:
        x = x * tl.sigmoid(x)
    tl.store(y_ptr + offs, x, mask=mask)


def _activate(y, kind):
    n_elements = y.numel()
    BLOCK = 1024
    _activation_kernel[(triton.cdiv(n_elements, BLOCK),)](
        y, n_elements, GELU=kind == "gelu", RELU=kind == "relu",
        SILU=kind == "silu", BLOCK=BLOCK)
    return y
'''


def _seed_A(cfg) -> str:
    aq = cfg["aq"]
    if aq in ("bf16", "fp16"):
        return "a"
    if aq in ("fp8", "int8"):
        return f'_dq_a8(aq, asc, "{cfg["asg"]}").to(torch.bfloat16)'
    if aq == "mxfp4":
        return "_dq_mxfp4(apk, ae8).to(torch.bfloat16)"
    return "_dq_mxfp8(aq, ae8).to(torch.bfloat16)"


def _seed_W(cfg) -> str:
    wq = cfg["wq"]
    if wq in ("bf16", "fp16"):
        return "w"
    if wq in ("fp8", "int8"):
        return f'_dq_w8(wq, wsc, "{cfg["wsg"]}").to(torch.bfloat16)'
    if wq == "int4c":
        return "_dq_int4c(wpk, wsc).to(torch.bfloat16)"
    if wq == "int4gs":
        return f"_dq_int4gs(wpk, wsc, {cfg['group']}).to(torch.bfloat16)"
    if wq == "int4ga":
        return f"_dq_int4ga(wpk, wsc, wz, {cfg['group']}).to(torch.bfloat16)"
    if wq == "mxfp4":
        return "_dq_mxfp4(wpk, we8).to(torch.bfloat16)"
    return "_dq_mxfp8(wq, we8).to(torch.bfloat16)"


def _seed_entry(cfg, op) -> str:
    fam = cfg["fam"]
    quant = _is_quant(cfg)
    names = ", ".join(_arg_names(cfg))
    L = [f"def {op}({names}):"]
    if fam == "dgrad":
        L.append("    return _mm_nt(dy, w.t().contiguous())")
        return "\n".join(L) + "\n"
    if fam == "wgrad":
        L.append("    return _mm_nt(dy.t().contiguous(), a.t().contiguous())")
        return "\n".join(L) + "\n"
    if fam == "batched":
        if quant:
            L += ["    B = aq.shape[0]", "    outs = []", "    for b in range(B):",
                  '        ab = _dq_a8(aq[b], asc[b], "row").to(torch.bfloat16)',
                  '        wb = _dq_w8(wq[b], wsc[b], "channel").to(torch.bfloat16)',
                  "        outs.append(_mm_nt(ab, wb))",
                  "    return torch.stack(outs).to(torch.bfloat16)"]
        else:
            L += ["    B = a.shape[0]",
                  "    outs = [_mm_nt(a[b].contiguous(), w[b].contiguous()) for b in range(B)]",
                  "    return torch.stack(outs)"]
        return "\n".join(L) + "\n"
    if fam == "grouped":
        if quant:
            L += ['    a = _dq_a8(aq, asc, "row").to(torch.bfloat16)',
                  "    E = wq.shape[0]",
                  '    w = torch.stack([_dq_w8(wq[e], wsc[e], "channel") for e in range(E)]).to(torch.bfloat16)',
                  "    return _grouped_mm(a, w, eids)"]
        else:
            L += ["    return _grouped_mm(a, w, eids)"]
        return "\n".join(L) + "\n"
    # plain
    L.append(f"    a = {_seed_A(cfg)}")
    L.append(f"    w = {_seed_W(cfg)}")
    L.append("    c = _mm_nt(a, w)")
    L.append("    y = c.float()")
    if _has_bias(cfg):
        L.append("    y = y + bias.float().reshape(1, -1)")
    act = _act_of(cfg["ep"])
    if act == "gelu":
        L.append('    y = _activate(y, "gelu")')
    elif act == "relu":
        L.append('    y = _activate(y, "relu")')
    elif act == "silu":
        L.append('    y = _activate(y, "silu")')
    if cfg["ep"] == "residual":
        L.append("    y = y + res.float()")
    if cfg["ep"] == "requant":
        L.append("    y = (y / osc.float()).clamp(-448.0, 448.0)"
                 ".to(torch.float8_e4m3fn).float() * osc.float()")
    out = "torch.bfloat16" if quant else "a.dtype"
    L.append(f"    return y.to({out})")
    return "\n".join(L) + "\n"


def seed_source(op: str, dtype: str) -> str:
    if op not in _CFG:
        raise ValueError(f"unknown breadth GEMM op {op!r}")
    cfg = _CFG[op]
    doc = (f'"""GENERATED breadth GEMM seed: {op} ({dtype}). Naive host dequant + a\n'
           f'tiled tl.dot GEMM (fp32 accumulate) + epilogue - a correct, COMPILING\n'
           f'starting point the KORE policy fuses into one quantized-GEMM kernel."""\n')
    imports = ("from __future__ import annotations\n"
               "import torch\n"
               "import triton\n"
               "import triton.language as tl\n")
    act_block = _ACT_BLOCK if _act_of(cfg["ep"]) != "none" else ""
    return doc + imports + _QHELP + _GEMM_BLOCK + act_block + "\n\n" + _seed_entry(cfg, op)


def op_names() -> list[str]:
    return list(OPS)
