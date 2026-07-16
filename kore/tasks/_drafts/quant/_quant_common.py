"""Shared quant encode helpers + fp32 dequant-matmul oracles + verifier driver for
the DRAFT quantized-GEMM tasks.

STAGING NOTE (safety): this module and every task under
``kore/tasks/_drafts/quant/`` are STAGED, not live. The registry discovers tasks via
``kore/tasks/*/task.yaml`` (ONE directory level, see ``registry._discover``), so a task
nested at ``kore/tasks/_drafts/quant/<id>/task.yaml`` is THREE levels deep and is NOT
auto-discovered -- no run/campaign can pick these up until a human promotes them.

Promotion (per task, after on-gfx950 verification): move
``kore/tasks/_drafts/quant/<id>/`` to ``kore/tasks/<id>/`` AND copy this file to
``kore/tasks/_quant_common.py`` (each task's thin ``driver.py`` imports ``_quant_common``
from its parent directory, mirroring how the live tasks share ``kore.tasks.aiter_ref``).
See VERIFICATION_CHECKLIST.md.

This centralizes the two things that must be correct in ONE place:
  * The QUANT ENCODE helpers (float -> codes + scales) used by every ``get_inputs``, and
    the fp32 DEQUANT-MATMUL ORACLES (codes + scales -> fp32 matmul -> out dtype) used by
    every ``reference_output``. The oracle applies each scale EXACTLY ONCE, in fp32; it
    IS the correctness ground truth for every drafted task. The candidate + reference
    share the SAME quantized inputs (same rounding), so the SNR gate measures the
    kernel's matmul/epilogue fidelity, NOT the quantization error.
  * :func:`driver_main` -- the KernelForge verifier contract (multi-trial reseeded
    correctness printing ``SNR`` / ``allclose`` / ``max_diff``, plus cold-cache
    CUDA-event median timing with post-timing anti-hack re-verification), identical in
    spirit to ``kore.tasks._genops.driver_main`` / the live per-task drivers, so we do
    not duplicate the ~140-line driver 8 times.

fp8 is arch-selected via the LIVE ``kore.tasks.aiter_ref`` (single source of truth):
OCP ``float8_e4m3fn`` (max 448) on gfx950/CDNA4, FNUZ ``float8_e4m3fnuz`` (max 240) on
gfx942/CDNA3. The candidate + oracle both consume ``FP8_DTYPE`` so the quant is
self-consistent per arch.

reference.py contract (each drafted task implements):
    parse_shape(s) -> dict
    get_inputs(shape, device="cuda", seed=0) -> tuple            # positional entry args
    reference_output(shape, inputs) -> torch.Tensor              # fp32 oracle -> out dtype
    candidate_output(fn, shape, inputs) -> torch.Tensor          # invoke candidate entry
    baseline_output(shape, inputs) -> torch.Tensor               # REAL vendor op
    ENTRY: str                                                   # candidate attr name
    ATOL: float, RTOL: float                                     # allclose tolerances
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os

import torch

# --------------------------------------------------------------------------- #
# arch-selected constants (from the LIVE aiter_ref, so drafts never fork the
# OCP-vs-FNUZ decision). Resolved lazily so a torch-only context can import this.
# --------------------------------------------------------------------------- #
INT8_MAX = 127.0
INT4_MIN, INT4_MAX = -8, 7          # symmetric int4 signed range
UINT4_MAX = 15                      # asymmetric int4 unsigned code range
BLK = 128                           # DeepSeek-V3 block-scale group (1x128 act, 128x128 w)
MX_BLOCK = 32                       # OCP microscaling group along K
E2M1_MAX = 6.0                      # max representable |value| in e2m1
E2M1_EMAX = 2                       # exponent of the e2m1 max normal (6.0 = 1.5 * 2^2)
_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   # e2m1 magnitudes (idx 0..7)
_E2M1_MIDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]      # round-to-nearest boundaries


def fp8_dtype_max():
    """(FP8_DTYPE, FP8_MAX) for the active arch, from the live aiter_ref module."""
    from kore.tasks.aiter_ref import FP8_DTYPE, FP8_MAX
    return FP8_DTYPE, FP8_MAX


# --------------------------------------------------------------------------- #
# QUANT ENCODE helpers (float -> codes + scales). Used by get_inputs.
# --------------------------------------------------------------------------- #
def quant_per_tensor_fp8(x: torch.Tensor):
    """Per-tensor symmetric fp8. Returns (xq fp8, scale scalar fp32)."""
    fp8, fmax = fp8_dtype_max()
    amax = x.float().abs().max().clamp(min=1e-12)
    scale = (amax / fmax).to(torch.float32)
    xq = (x.float() / scale).clamp(-fmax, fmax).to(fp8)
    return xq, scale.reshape(())


def quant_rowwise_fp8(x: torch.Tensor):
    """Per-row (per-token / per-output-channel) symmetric fp8. x[R,K] ->
    (xq fp8 [R,K], scale fp32 [R,1]); ``x ~= xq.float() * scale``."""
    fp8, fmax = fp8_dtype_max()
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / fmax).to(torch.float32)
    xq = (x.float() / scale).clamp(-fmax, fmax).to(fp8)
    return xq, scale


def quant_rowwise_int8(x: torch.Tensor):
    """Per-row symmetric int8. x[R,K] -> (xq int8 [R,K], scale fp32 [R,1])."""
    amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / INT8_MAX).to(torch.float32)
    xq = (x.float() / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return xq, scale


def quant_1x128_fp8(x: torch.Tensor):
    """Per-1x128 (per-token-group) fp8. x[M,K] -> (xq fp8 [M,K], xs fp32 [M,K//128])."""
    fp8, fmax = fp8_dtype_max()
    M, K = x.shape
    xb = x.float().view(M, K // BLK, BLK)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    s = amax / fmax
    xq = (xb / s).clamp(-fmax, fmax).to(fp8).view(M, K)
    return xq, s.squeeze(-1).to(torch.float32).contiguous()


def quant_128x128_fp8(w: torch.Tensor):
    """Per-128x128 block fp8. w[N,K] -> (wq fp8 [N,K], ws fp32 [N//128,K//128])."""
    fp8, fmax = fp8_dtype_max()
    N, K = w.shape
    wb = w.float().view(N // BLK, BLK, K // BLK, BLK)
    amax = wb.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-8)
    s = amax / fmax
    wq = (wb / s).clamp(-fmax, fmax).to(fp8).view(N, K)
    return wq, s.squeeze(1).squeeze(-1).to(torch.float32).contiguous()


def _e2m1_quant_codes(v: torch.Tensor) -> torch.Tensor:
    """Scaled values -> e2m1 codes 0..15 (round-to-nearest magnitude + sign bit)."""
    sign = (v < 0).to(torch.uint8)
    mids = torch.tensor(_E2M1_MIDS, dtype=torch.float32, device=v.device)
    idx = torch.bucketize(v.abs(), mids).to(torch.uint8)   # 0..7
    return (sign << 3) | idx


def _e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    """e2m1 codes 0..15 -> fp32 values (LUT for magnitude, bit-3 for sign)."""
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=code.device)
    mag = levels[(code & 0x7).long()]
    sign = torch.where((code & 0x8) != 0, -1.0, 1.0)
    return sign * mag


def quant_pack_mxfp4(x: torch.Tensor):
    """OCP MXFP4: x[R,K] fp32 -> (packed[R,K//2] uint8, e8m0[R,K//32] uint8).

    Per 32-wide K group: shared exponent ``floor(log2(amax)) - E2M1_EMAX`` stored biased
    (+127) as E8M0; codes = round-to-nearest e2m1 of ``x / 2^exp`` (clamped to +/-6).
    Two nibbles/byte along K (even-K low nibble, odd-K high nibble)."""
    R, K = x.shape
    assert K % MX_BLOCK == 0, "K must be a multiple of 32 for MXFP4 blocks"
    xb = x.float().reshape(R, K // MX_BLOCK, MX_BLOCK)
    amax = xb.abs().amax(dim=2, keepdim=True).clamp(min=1e-20)
    exp = (torch.floor(torch.log2(amax)) - float(E2M1_EMAX)).clamp(-127.0, 127.0)
    e8m0 = (exp + 127.0).to(torch.uint8)
    scale = torch.exp2(exp)
    xq = (xb / scale).clamp(-E2M1_MAX, E2M1_MAX).reshape(R, K)
    codes = _e2m1_quant_codes(xq)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, e8m0.reshape(R, K // MX_BLOCK).contiguous()


def quant_pack_int4_perchannel(w: torch.Tensor):
    """Symmetric per-output-channel int4. w[N,K] -> (packed[N,K//2] uint8, scale[N,1]).

    codes 0..15 <-> values -8..7 (code - 8); ``w ~= (code - 8) * scale[n]``."""
    amax = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (amax / INT4_MAX).to(torch.float32)
    q = torch.round(w.float() / scale).clamp(INT4_MIN, INT4_MAX).to(torch.int32)
    code = (q + 8).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, scale


def quant_pack_int4_group_asym(w: torch.Tensor, group: int):
    """Asymmetric (zero-point) group-wise int4 (AWQ/GPTQ). w[N,K] ->
    (packed[N,K//2] uint8, scale[N,K//group] fp32, zero[N,K//group] uint8).

    Per group g of ``group`` consecutive K: ``scale = (wmax - wmin) / 15``,
    ``zero = round(-wmin / scale)`` (a 0..15 code), ``code = round(w/scale + zero)``,
    so ``w ~= (code - zero) * scale``. Zero-point application is the classic bug this
    scheme exists to exercise."""
    N, K = w.shape
    assert K % group == 0, "K must be a multiple of the int4 group size"
    wb = w.float().reshape(N, K // group, group)
    wmin = wb.amin(dim=2, keepdim=True)
    wmax = wb.amax(dim=2, keepdim=True)
    scale = ((wmax - wmin) / float(UINT4_MAX)).clamp(min=1e-12)
    zero = torch.round(-wmin / scale).clamp(0, UINT4_MAX)
    code = torch.round(wb / scale + zero).clamp(0, UINT4_MAX).to(torch.uint8).reshape(N, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return (packed,
            scale.squeeze(-1).to(torch.float32).contiguous(),
            zero.squeeze(-1).to(torch.uint8).contiguous())


# --------------------------------------------------------------------------- #
# DEQUANT helpers (codes + scales -> fp32 dense operand). Used by oracles +
# by the vendor "materialize to bf16 + hipBLASLt" baselines.
# --------------------------------------------------------------------------- #
def unpack_dequant_mxfp4(packed: torch.Tensor, e8m0: torch.Tensor, K: int) -> torch.Tensor:
    """(packed[R,K//2], e8m0[R,K//32]) -> deq[R,K] fp32."""
    R = packed.shape[0]
    codes = torch.empty((R, K), dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    vals = _e2m1_decode(codes)
    scale = torch.exp2(e8m0.to(torch.float32) - 127.0).repeat_interleave(MX_BLOCK, dim=1)
    return vals * scale


def unpack_dequant_int4_perchannel(packed: torch.Tensor, scale: torch.Tensor,
                                   K: int) -> torch.Tensor:
    """(packed[N,K//2], scale[N,1]) -> deq[N,K] fp32; value = (code - 8) * scale[n]."""
    N = packed.shape[0]
    q = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    q[:, 0::2] = (packed & 0xF).to(torch.int32) - 8
    q[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32) - 8
    return q.float() * scale.float()


def unpack_dequant_int4_group_asym(packed: torch.Tensor, scale: torch.Tensor,
                                   zero: torch.Tensor, K: int, group: int) -> torch.Tensor:
    """(packed[N,K//2], scale[N,K//g], zero[N,K//g]) -> deq[N,K] fp32;
    value = (code - zero[n,g]) * scale[n,g], g = k // group."""
    N = packed.shape[0]
    codes = torch.empty((N, K), dtype=torch.int32, device=packed.device)
    codes[:, 0::2] = (packed & 0xF).to(torch.int32)
    codes[:, 1::2] = ((packed >> 4) & 0xF).to(torch.int32)
    z = zero.to(torch.int32).repeat_interleave(group, dim=1)
    s = scale.float().repeat_interleave(group, dim=1)
    return (codes.float() - z.float()) * s


# --------------------------------------------------------------------------- #
# fp32 DEQUANT-MATMUL ORACLES (the correctness ground truth). Scale applied once.
# --------------------------------------------------------------------------- #
def matmul_a8w8_fp32(xq, wq, x_scale, w_scale) -> torch.Tensor:
    """8-bit (fp8/int8) A and W with per-row activation scale + per-channel weight
    scale. XQ[M,K], WQ[N,K], x_scale[M,1], w_scale[1,N] (or [N,1]/[N]). Covers
    per-tensor (all rows/cols equal), per-token, per-channel, int8. -> bf16 [M,N]."""
    a_deq = xq.float() * x_scale.float()                       # [M,K] * [M,1]
    w_deq = wq.float() * w_scale.float().reshape(-1, 1)        # [N,K] * [N,1]
    return (a_deq @ w_deq.t()).to(torch.bfloat16)


def matmul_blockscale_fp32(xq, wq, xs, ws) -> torch.Tensor:
    """DeepSeek block-scaled fp8. XQ[M,K] with xs[M,K//128] (1x128), WQ[N,K] with
    ws[N//128,K//128] (128x128). Per-128-K-group dequant, then matmul -> bf16 [M,N]."""
    M, K = xq.shape
    N = wq.shape[0]
    xd = (xq.view(M, K // BLK, BLK).float() * xs[:, :, None]).view(M, K)
    wd = (wq.view(N // BLK, BLK, K // BLK, BLK).float() * ws[:, None, :, None]).view(N, K)
    return (xd @ wd.t()).to(torch.bfloat16)


def matmul_mxfp4_a4w4_fp32(a_packed, a_e8m0, w_packed, w_e8m0, K) -> torch.Tensor:
    """MXFP4 both operands (A and W in e2m1 + e8m0/32). -> bf16 [M,N]."""
    a_deq = unpack_dequant_mxfp4(a_packed, a_e8m0, K)          # [M,K]
    w_deq = unpack_dequant_mxfp4(w_packed, w_e8m0, K)          # [N,K]
    return (a_deq @ w_deq.t()).to(torch.bfloat16)


def matmul_w4a16_group_fp32(a, w_packed, scale, zero, K, group) -> torch.Tensor:
    """int4 group-wise asymmetric weight, bf16 activation. -> bf16 [M,N]."""
    w_deq = unpack_dequant_int4_group_asym(w_packed, scale, zero, K, group)  # [N,K]
    return (a.float() @ w_deq.t()).to(torch.bfloat16)


def matmul_w4a8_fp32(xq, x_scale, w_packed, w_scale, K) -> torch.Tensor:
    """fp8 per-token activation, int4 per-channel symmetric weight. -> bf16 [M,N]."""
    a_deq = xq.float() * x_scale.float()                       # [M,K]
    w_deq = unpack_dequant_int4_perchannel(w_packed, w_scale, K)   # [N,K]
    return (a_deq @ w_deq.t()).to(torch.bfloat16)


def gemm_fp8_requant_fp32(xq, wq, x_scale, w_scale, bias, out_scale) -> torch.Tensor:
    """fp8 a8w8 GEMM (per-token A, per-channel W) with a fused bias + static-scale fp8
    REQUANT epilogue. Returns the DEQUANTIZED bf16 view of the fp8 output so the SNR
    gate measures matmul + epilogue fidelity (the fp8 output rounding is shared by
    candidate + reference). Y_fp8 = requant(A_deq @ W_deq^T + bias, out_scale)."""
    fp8, fmax = fp8_dtype_max()
    acc = (xq.float() * x_scale.float()) @ (wq.float() * w_scale.float().reshape(-1, 1)).t()
    acc = acc + bias.float().reshape(1, -1)
    yq = (acc / out_scale.float()).clamp(-fmax, fmax).to(fp8)   # fp8 output codes
    return (yq.float() * out_scale.float()).to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# Verifier driver (correctness + cold-cache bench + post-timing anti-hack).
# Identical in spirit to the live per-task drivers / _genops.driver_main.
# --------------------------------------------------------------------------- #
def _snr_db(out, ref_out) -> float:
    o, r = out.float(), ref_out.float()
    noise = (o - r).norm().item()
    signal = r.norm().item()
    if noise == 0:
        return 999.0
    return 20.0 * math.log10(signal / noise) if signal > 0 else -999.0


def _num_correct_trials() -> int:
    """KernelBench-fidelity: >=5 reseeded correctness trials (env-overridable)."""
    try:
        return max(5, int(os.environ.get("KORE_CORRECTNESS_TRIALS", "5")))
    except ValueError:
        return 5


def _bench_cold() -> bool:
    return os.environ.get("KORE_BENCH_COLD", "1") != "0"


_L2_SCRATCH = None


def _flush_l2(device: str = "cuda") -> None:
    """Evict the GPU last-level cache between timed iters (cold-cache, KernelBench)."""
    global _L2_SCRATCH
    if _L2_SCRATCH is None:
        _L2_SCRATCH = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    _L2_SCRATCH.zero_()


def _time_fn(fn, warmup: int, iters: int) -> int:
    cold = _bench_cold()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if cold:
            _flush_l2()
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    for t in times:
        print(f"wall_ms: {t:.4f}")
    print(f"median_ms: {times[len(times) // 2]:.4f}")
    return 0


def _load_candidate(task_dir: str, entry: str):
    # Cache the module so a stateful kernel's globals persist from the bench timing
    # loop into the post-timing re-verification (anti invocation-count timing hack).
    if getattr(_load_candidate, "_mod", None) is None:
        path = os.path.join(task_dir, "kernel.py")
        spec = importlib.util.spec_from_file_location("candidate_kernel", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _load_candidate._mod = mod
    return getattr(_load_candidate._mod, entry)


def _run_correctness(ref, task_dir, shape) -> int:
    fn = _load_candidate(task_dir, ref.ENTRY)
    atol = getattr(ref, "ATOL", 5e-1)
    rtol = getattr(ref, "RTOL", 5e-2)
    worst, maxd, ok = 999.0, 0.0, True
    for s in range(_num_correct_trials()):
        inputs = ref.get_inputs(shape, device="cuda", seed=s)
        r = ref.reference_output(shape, inputs)
        try:
            o = ref.candidate_output(fn, shape, inputs)
        except Exception as e:  # noqa: BLE001
            print("SNR: -999.00 dB"); print("allclose: False"); print("max_diff: inf")
            print(f"CANDIDATE_ERROR: {type(e).__name__}: {e}")
            return 0
        torch.cuda.synchronize()
        worst = min(worst, _snr_db(o, r))
        maxd = max(maxd, (o.float() - r.float()).abs().max().item())
        ok = ok and torch.allclose(o.float(), r.float(), atol=atol, rtol=rtol)
    print(f"SNR: {worst:.2f} dB"); print(f"allclose: {ok}"); print(f"max_diff: {maxd:.6f}")
    return 0


def _run_bench(ref, task_dir, shape, impl, warmup, iters) -> int:
    inputs = ref.get_inputs(shape, device="cuda", seed=0)
    if impl == "reference":
        fn = lambda: ref.baseline_output(shape, inputs)          # REAL vendor op
    elif impl == "torch":
        fn = lambda: ref.reference_output(shape, inputs)         # fp32 oracle
    else:
        cand = _load_candidate(task_dir, ref.ENTRY)
        fn = lambda: ref.candidate_output(cand, shape, inputs)
    return _time_fn(fn, warmup, iters)


def driver_main(ref, task_dir: str, argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="default")
    p.add_argument("--mode", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--impl", default="candidate", choices=["candidate", "reference", "torch"])
    a = p.parse_args(argv)
    shape = ref.parse_shape(a.shape)
    if a.bench_mode:
        rc = _run_bench(ref, task_dir, shape, a.impl, a.warmup, a.iters)
        # Post-timing anti-hack correctness re-verification on the cached candidate.
        if a.impl == "candidate":
            _run_correctness(ref, task_dir, shape)
        return rc
    return _run_correctness(ref, task_dir, shape)
