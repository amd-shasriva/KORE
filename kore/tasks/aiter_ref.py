"""Shared AITER baseline helpers for KORE tasks.

The whole point of the AMD-correct tasks: the *performance baseline* is the
kernel the production serving stack actually calls (AITER), not unfused torch.
This module centralizes the thin AITER wrappers + the fp8 quantization helpers
so each task's driver measures the honest bar.

Import-safe: AITER (and torch) are imported lazily inside the wrappers so that
`kore tasks` / registry discovery never require a GPU or the aiter runtime.

fp8 e4m3 encoding is ARCH-DEPENDENT and auto-selected (see :func:`_fp8_e4m3_dtype`):
  * gfx950 / CDNA4 (MI350X / MI355X, THIS node): **OCP** ``torch.float8_e4m3fn``
    (range +/-448, standard bias) -- the native format the CDNA4 matrix cores and
    AITER/hipBLASLt use. This is the KORE target hardware, so it is the default.
  * gfx942 / CDNA3 (MI300X / MI325X): the **FNUZ** ``torch.float8_e4m3fnuz``
    (range +/-240) -- CDNA3's fp8; using OCP there mismatches AITER/hipBLASLt.
Override with ``KORE_FP8_ENCODING=ocp|fnuz``. The reference oracle AND the candidate
kernel both consume ``FP8_DTYPE``, so the quant is self-consistent per arch.

Version-robustness + honest labeling (P0): AITER moved its ops from the top level
(``aiter.rms_norm``) into submodules (``aiter.ops.rmsnorm.rms_norm``) in newer
releases, and its gluon kernels require triton >= 3.6 which not every stack has.
Every wrapper therefore resolves the op via :func:`_aiter_fn` (top level OR
``aiter.ops.*``), falls back to the torch framework path when AITER is
unavailable, and emits a one-time :func:`_mark_baseline` sentinel so the P0
harness can label each check-(a) baseline ``aiter_vendor`` / ``hipblaslt_vendor``
/ ``framework``.
"""

from __future__ import annotations

import os

import torch


def _fp8_e4m3_dtype():
    """e4m3 fp8 dtype for the active arch (import-safe; no CUDA init required).

    OCP ``e4m3fn`` on gfx950/CDNA4 (MI350X/MI355X) and newer; FNUZ ``e4m3fnuz`` on
    gfx942/CDNA3 (MI300X/MI325X) / gfx90a. Order: ``KORE_FP8_ENCODING`` override,
    then the running GPU's gfx target, then OCP (the KORE target hardware).
    """
    enc = os.environ.get("KORE_FP8_ENCODING", "").strip().lower()
    if enc in ("ocp", "fn", "e4m3fn"):
        return torch.float8_e4m3fn
    if enc in ("fnuz", "e4m3fnuz"):
        return torch.float8_e4m3fnuz
    try:  # pragma: no cover - hardware dependent
        if torch.cuda.is_available():
            arch = (torch.cuda.get_device_properties(0).gcnArchName or "").lower()
            if "gfx942" in arch or "gfx90a" in arch or "gfx908" in arch:
                return torch.float8_e4m3fnuz   # CDNA2/CDNA3 -> FNUZ
            return torch.float8_e4m3fn         # gfx950/CDNA4+ -> OCP
    except Exception:
        pass
    return torch.float8_e4m3fn  # default: OCP (KORE target = gfx950/CDNA4)


# Arch-selected e4m3 fp8 dtype + its finite max (gfx950 OCP: 448.0; gfx942 FNUZ: 240.0).
FP8_DTYPE = _fp8_e4m3_dtype()
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


# AITER moved its ops from top-level (``aiter.rms_norm``) into submodules
# (``aiter.ops.rmsnorm.rms_norm``) in newer releases. Resolve version-robustly:
# try top-level first (old API), then the known ``aiter.ops.*`` submodules.
_AITER_OP_MODULES = (
    "ops.rmsnorm", "ops.norm", "ops.activation", "ops.gemm_op_a8w8",
    "ops.gemm_op", "ops.rope", "ops.quant", "ops.mha", "ops.attention",
    "ops.paged_attn", "ops.moe", "ops.moe_op", "ops.topk", "ops",
)


def _aiter_fn(name: str):
    """Return AITER callable ``name`` from top-level or an ``aiter.ops.*`` submodule."""
    import importlib

    import aiter

    fn = getattr(aiter, name, None)
    if callable(fn):
        return fn
    for sub in _AITER_OP_MODULES:
        try:
            mod = importlib.import_module(f"aiter.{sub}")
        except Exception:  # noqa: BLE001 - submodule may be absent/broken in some builds
            continue
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise AttributeError(f"aiter has no callable '{name}' (checked top-level + ops.*)")


_MARKED_BASELINE: set = set()


def _mark_baseline(kind: str) -> None:
    """Emit a one-time sentinel identifying which baseline implementation was used.

    ``kind`` is ``aiter_vendor`` (real AITER production kernel), ``hipblaslt_vendor``
    (torch.matmul/bmm -> hipBLASLt, the production dense-GEMM library), or
    ``framework`` (torch fused op used because AITER has no standalone kernel or its
    kernels are unavailable in this stack). The P0 harness parses the LAST such line
    from the ``--impl reference`` bench output to honestly label check-(a) baselines.
    Printed once per process to stderr so it never pollutes the driver's ``median_ms``.
    """
    if kind in _MARKED_BASELINE:
        return
    _MARKED_BASELINE.add(kind)
    try:
        import sys
        print(f"KORE_BASELINE_IMPL:{kind}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


# --- RMSNorm family -------------------------------------------------------
def aiter_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Production RMSNorm baseline: AITER CK ``rms_norm`` if its kernels load, else
    the torch framework RMSNorm (``F.rms_norm``), which on ROCm is a real fused
    kernel - the documented framework production bar when AITER is unavailable."""
    try:
        out = _aiter_fn("rms_norm")(x, weight, eps)
        _mark_baseline("aiter_vendor")
        return out
    except Exception:  # noqa: BLE001 - aiter absent / gluon triton mismatch
        _mark_baseline("framework")
        import torch.nn.functional as F
        return F.rms_norm(x, (x.shape[-1],), weight, eps)


def aiter_fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
):
    """AITER fused add + RMSNorm, matching in-place CU semantics.

    ``aiter.fused_add_rms_norm_cu(input, residual_in, weight, epsilon)`` mutates
    both tensors in place (returns None):
      * ``input``        <- RMSNorm(input + residual_in) * weight
      * ``residual_in``  <- input + residual_in   (the new residual)

    We operate on the passed tensors directly (caller owns cloning for a fair
    benchmark) and return ``(normed, new_residual)`` = ``(x, residual)``.
    """
    try:
        _aiter_fn("fused_add_rms_norm_cu")(x, residual, weight, eps)
        _mark_baseline("aiter_vendor")
        return x, residual
    except Exception:  # noqa: BLE001 - fall back to the torch framework path
        _mark_baseline("framework")
        import torch.nn.functional as F
        new_res = x + residual
        y = F.rms_norm(new_res, (new_res.shape[-1],), weight, eps)
        return y, new_res


# --- Gated MLP activations ------------------------------------------------
def aiter_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """AITER ``silu_and_mul(out, input)`` (in-place into out).

    Input is (M, 2*inter); returns SiLU(x[:, :inter]) * x[:, inter:] as (M, inter).
    """
    inter = x.shape[-1] // 2
    try:
        out = torch.empty((*x.shape[:-1], inter), dtype=x.dtype, device=x.device)
        _aiter_fn("silu_and_mul")(out, x)
        _mark_baseline("aiter_vendor")
        return out
    except Exception:  # noqa: BLE001 - torch framework SiLU-gate fallback
        _mark_baseline("framework")
        import torch.nn.functional as F
        return F.silu(x[..., :inter]) * x[..., inter:]


def aiter_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    """AITER ``gelu_tanh_and_mul(out, input)`` (in-place into out): GeGLU.

    Input is (M, 2*inter); returns GELU-tanh(x[:, :inter]) * x[:, inter:] as
    (M, inter) - the LLM-standard gated activation. Falls back to the torch
    framework GeGLU when AITER is unavailable.
    """
    inter = x.shape[-1] // 2
    try:
        out = torch.empty((*x.shape[:-1], inter), dtype=x.dtype, device=x.device)
        _aiter_fn("gelu_tanh_and_mul")(out, x)
        _mark_baseline("aiter_vendor")
        return out
    except Exception:  # noqa: BLE001 - torch framework GeGLU fallback
        _mark_baseline("framework")
        import torch.nn.functional as F
        return F.gelu(x[..., :inter], approximate="tanh") * x[..., inter:]


# --- fp8 GEMM -------------------------------------------------------------
def per_tensor_quant_fp8(x: torch.Tensor):
    """Per-tensor symmetric quantization to the arch-selected fp8 e4m3 (``FP8_DTYPE``:
    OCP e4m3fn on gfx950/CDNA4, FNUZ e4m3fnuz on gfx942/CDNA3).

    Returns ``(xq, scale)`` where ``scale`` is a scalar fp32 tensor and
    ``x ≈ xq.float() * scale``.
    """
    amax = x.abs().max().clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(torch.float32)
    xq = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq, scale.reshape(())


def aiter_gemm_a8w8(
    xq: torch.Tensor,
    wq: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """AITER fp8 GEMM: ``aiter.gemm_a8w8(XQ, WQ, x_scale, w_scale, dtype=...)``.

    Layout (CK): XQ [M, K], WQ [N, K] (computes ``X @ W^T``), x_scale [M, 1],
    w_scale [1, N], both fp32. Returns [M, N] in ``out_dtype``.
    """
    out = _aiter_fn("gemm_a8w8")(xq, wq, x_scale, w_scale, dtype=out_dtype)
    _mark_baseline("aiter_vendor")
    return out


def aiter_gemm_a8w8_blockscale(
    xq: torch.Tensor,
    wq: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """AITER block-scaled fp8 GEMM: ``aiter.gemm_a8w8_blockscale(XQ, WQ, x_scale,
    w_scale, dtype=...)`` (the DeepSeek-style 1x128 / 128x128 block-quant path).

    Layout (CK): XQ [M, K] fp8, WQ [N, K] fp8 (computes ``X @ W^T``); x_scale is
    the per-(1x128)-block activation scale [M, K//128] and w_scale the
    per-(128x128)-block weight scale [N//128, K//128], both fp32. Returns [M, N]
    in ``out_dtype``. Falls back to a torch block-dequant matmul (framework) when
    AITER's block-scale kernel is unavailable.

    fp8 codes are the arch-selected OCP e4m3 (``FP8_DTYPE``) on gfx950/CDNA4.
    """
    try:
        out = _aiter_fn("gemm_a8w8_blockscale")(xq, wq, x_scale, w_scale, dtype=out_dtype)
        _mark_baseline("aiter_vendor")
        return out
    except Exception:  # noqa: BLE001 - block-scale kernel absent -> torch dequant
        _mark_baseline("framework")
        # Generic block-dequant fallback: broadcast each block scale back over its
        # 128-wide tile, dequantize, then a hipBLASLt bf16 matmul of X @ W^T.
        block = 128
        xf = xq.float()
        wf = wq.float()
        xs = x_scale.float().repeat_interleave(block, dim=1)[:, : xf.shape[1]]
        ws = w_scale.float()
        ws = ws.repeat_interleave(block, dim=0)[: wf.shape[0], :]
        ws = ws.repeat_interleave(block, dim=1)[:, : wf.shape[1]]
        x_deq = xf * xs
        w_deq = wf * ws
        return torch.matmul(x_deq, w_deq.t()).to(out_dtype)


def aiter_gemm_int8_a8w8(
    xq: torch.Tensor,
    wq: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """AITER int8 W8A8 GEMM: reuses the ``gemm_a8w8`` CK path with **int8** XQ/WQ
    and fp32 per-row / per-col scales (the classic SmoothQuant / W8A8 int8 serving
    path, distinct from the fp8 :func:`aiter_gemm_a8w8`).

    Layout (CK): XQ [M, K] int8, WQ [N, K] int8 (computes ``X @ W^T``), x_scale
    [M, 1], w_scale [1, N], both fp32; int32 accumulation. Returns [M, N] in
    ``out_dtype``. Falls back to a torch int8-dequant matmul (framework) when the
    AITER int8 kernel is unavailable.
    """
    try:
        out = _aiter_fn("gemm_a8w8")(xq, wq, x_scale, w_scale, dtype=out_dtype)
        _mark_baseline("aiter_vendor")
        return out
    except Exception:  # noqa: BLE001 - int8 gemm_a8w8 unavailable -> torch dequant
        _mark_baseline("framework")
        # int32 accumulate on the integer codes, then apply row/col fp32 scales.
        acc = torch.matmul(xq.to(torch.int32).float(), wq.to(torch.int32).float().t())
        out = acc * x_scale.float().reshape(-1, 1) * w_scale.float().reshape(1, -1)
        return out.to(out_dtype)


def aiter_gemm_a4w4(
    xq: torch.Tensor,
    wq: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """AITER fp4 (MXFP4/NVFP4) GEMM: ``aiter.gemm_a4w4`` (or the block-scale
    variant ``aiter.gemm_a4w4_blockscale``) -- the 4-bit weight+activation path.

    Layout (CK): A [M, K//2] f4x2 (two fp4 codes packed per byte), B [N, K//2]
    f4x2 (computes ``A @ B^T``); A_scale [M, K//block_size] and B_scale
    [N, K//block_size] are the microscale block scales (MXFP4: block_size=32
    e8m0-padded; NVFP4: block_size=16 e4m3-padded). Returns [M, N] in
    ``out_dtype``.

    ``gemm_a4w4`` returns a padded output ``(out, out_padded)`` on some builds;
    we normalize to the [M, N] tensor. Falls back to ``gemm_a4w4_blockscale``
    (out-param form) and finally, if neither 4-bit kernel is available, tags
    ``framework`` and raises -- there is no torch fp4 dtype to dequantize packed
    f4x2 codes without the kernel's exact packing, so no numeric torch reference
    is offered for the 4-bit path (documented blocker).
    """
    M = xq.shape[0]
    N = wq.shape[0]
    # Preferred: the high-level gemm_a4w4 dispatcher.
    try:
        out = _aiter_fn("gemm_a4w4")(xq, wq, x_scale, w_scale, dtype=out_dtype)
        if isinstance(out, (tuple, list)):
            out = out[0]
        _mark_baseline("aiter_vendor")
        return out[:M, :N] if out.dim() == 2 else out
    except Exception:  # noqa: BLE001 - try the explicit block-scale out-param form
        pass
    try:
        out = torch.empty((M, N), dtype=out_dtype, device=xq.device)
        _aiter_fn("gemm_a4w4_blockscale")(xq, wq, x_scale, w_scale, out)
        _mark_baseline("aiter_vendor")
        return out
    except Exception as e:  # noqa: BLE001 - no 4-bit kernel + no torch fp4 dtype
        _mark_baseline("framework")
        raise NotImplementedError(
            "aiter_gemm_a4w4: neither aiter.gemm_a4w4 nor "
            "aiter.gemm_a4w4_blockscale is available, and packed f4x2 codes have "
            "no torch dtype to dequantize -- no framework fallback for the 4-bit "
            "path."
        ) from e


# --- Batched / grouped GEMM ----------------------------------------------
def aiter_batched_gemm_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """AITER batched bf16 GEMM: ``aiter.batched_gemm_bf16(A, B, out)`` (in-place).

    Layout (CK): A [B, M, K], B [B, N, K] (so it computes ``A @ B^T`` per batch),
    out [B, M, N] bf16, fp32 accumulation. Falls back to ``torch.bmm`` (which on
    ROCm dispatches to hipBLASLt batched GEMM) when AITER is unavailable.
    """
    B, M, _ = a.shape
    N = b.shape[1]
    try:
        out = torch.empty((B, M, N), dtype=torch.bfloat16, device=a.device)
        _aiter_fn("batched_gemm_bf16")(a, b, out)
        _mark_baseline("aiter_vendor")
        return out
    except Exception:  # noqa: BLE001 - torch batched matmul fallback (hipBLASLt)
        _mark_baseline("hipblaslt_vendor")
        return torch.bmm(a, b.transpose(1, 2))


# --- Dense bf16 GEMM ------------------------------------------------------
def hipblaslt_gemm_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Production dense bf16 GEMM baseline: ``torch.matmul(A, B)``.

    On ROCm, ``torch.matmul`` for bf16 dense matmul dispatches straight to
    **hipBLASLt** (the vendor tuned GEMM library that the serving stack uses),
    so this *is* the real production bar - not an unfused torch loop. A[M,K] @
    B[K,N] -> [M,N], fp32 accumulate, bf16 output.
    """
    _mark_baseline("hipblaslt_vendor")
    return torch.matmul(a, b)


# --- LayerNorm ------------------------------------------------------------
def aiter_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """AITER CK LayerNorm: ``aiter.layer_norm(input, weight, bias, epsilon)``.

    2D row LayerNorm over the last dim (mean + variance subtraction), affine
    with weight+bias. Falls back to the torch framework LayerNorm when AITER is
    unavailable. Returns a tensor of the same shape/dtype.
    """
    try:
        out = _aiter_fn("layer_norm")(x, weight, bias, eps)
        _mark_baseline("aiter_vendor")
        return out
    except Exception:  # noqa: BLE001 - torch framework LayerNorm fallback
        _mark_baseline("framework")
        import torch.nn.functional as F
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def aiter_layer_norm_noaffine_ok(x, weight, bias, eps: float) -> torch.Tensor:
    """LayerNorm wrapper used by the vendor tasks; delegates to
    :func:`aiter_layer_norm` (instrumented, with a torch framework fallback)."""
    return aiter_layer_norm(x, weight, bias, eps)


# --- Softmax --------------------------------------------------------------
def torch_softmax_lastdim(x: torch.Tensor) -> torch.Tensor:
    """Production row-softmax baseline.

    Newer AITER ships a standalone Triton row-softmax
    (``aiter.ops.triton.softmax.softmax``), which is the honest vendor production
    op when available (2D ``[n_rows, n_cols]`` row-softmax over the last dim). We
    try it first and tag ``aiter_vendor``. When it is unavailable (older AITER,
    triton mismatch, or a >2D input the Triton kernel does not accept) we fall
    back to the torch framework path: on ROCm ``torch.softmax`` lowers to a fused
    MIOpen/rocm softmax kernel -- the documented framework production baseline.
    """
    try:
        if x.dim() != 2:
            raise ValueError("aiter triton softmax expects a 2D [rows, cols] input")
        import importlib
        _sm = importlib.import_module("aiter.ops.triton.softmax")
        out = _sm.softmax(x)
        _mark_baseline("aiter_vendor")
        return out
    except Exception:  # noqa: BLE001 - aiter/triton unavailable or non-2D input
        _mark_baseline("framework")
        return torch.softmax(x, dim=-1)


# --- GELU (tanh approximation) -------------------------------------------
def torch_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """Production tanh-approx GELU baseline: ``F.gelu(x, approximate='tanh')``.

    AITER only ships *gated* GELU (``gelu_and_mul`` / ``gelu_tanh_and_mul``), not
    a standalone elementwise GELU activation, so the honest production op is the
    framework path: ``torch.nn.functional.gelu`` lowers to a fused rocm
    elementwise kernel. Documented as the framework production baseline.
    """
    import torch.nn.functional as F

    _mark_baseline("framework")
    return F.gelu(x, approximate="tanh")


# --- RoPE (rotary position embedding) ------------------------------------
def aiter_rope_neox(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """AITER RoPE: ``aiter.rope_fwd`` (NEOX-style, full head-dim rotation).

    ``x`` is (S, B, H, D); ``freqs`` is (S, 1, 1, D//2) of rotation *angles*
    (the op computes cos/sin internally). Call convention:
    ``rope_fwd(input, freqs, rotate_style=0 (NEOX), reuse_freqs_front_part=True,
    nope_first=False)`` -> rotated tensor (S, B, H, D). This is the vendor HIP
    rope kernel used by the serving stack.
    """
    out = _aiter_fn("rope_fwd")(x, freqs, 0, True, False, False)
    _mark_baseline("aiter_vendor")
    return out


# --- Dynamic per-token fp8 quantization ----------------------------------
def aiter_dynamic_per_token_quant(x: torch.Tensor):
    """AITER dynamic per-token (rowwise) fp8 quant to the arch-selected e4m3
    (``FP8_DTYPE``: OCP e4m3fn on gfx950/CDNA4, FNUZ e4m3fnuz on gfx942/CDNA3).

    ``aiter.dynamic_per_token_scaled_quant(out, input, scales)`` writes the fp8
    codes into ``out`` [M,N] and the per-row fp32 scales into ``scales`` [M,1]
    in place (returns None). ``x ≈ out.float() * scales``. This is the vendor
    quant kernel the serving stack calls for W8A8 / fp8 activation quant.
    """
    M, N = x.shape
    out = torch.empty((M, N), dtype=FP8_DTYPE, device=x.device)
    scales = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    _aiter_fn("dynamic_per_token_scaled_quant")(out, x, scales)
    _mark_baseline("aiter_vendor")
    return out, scales
