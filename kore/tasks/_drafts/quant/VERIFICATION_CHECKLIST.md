# Quantized-GEMM DRAFT tasks: on-gfx950 verification checklist

STATUS: STAGED DRAFTS. Not live. Do NOT promote any task until the boxes below are
checked on a real gfx950 / MI350X (CDNA4) node with the installed AITER / hipBLASLt.

These tasks are integrity-critical: a wrong reference oracle (a scale applied on the wrong
axis, a mis-indexed block/group, a dropped zero-point) would poison the KORE verified-data
moat. Everything here has been CPU-verified for REFERENCE CORRECTNESS ONLY (the fp32
dequant-matmul math). NOTHING here has been GPU-verified: the Triton seeds have not been
compiled, the AITER / hipBLASLt vendor baselines have not been run, and no timing has been
measured.

## 0. Why these are safe while staged

- They live at `kore/tasks/_drafts/quant/<id>/task.yaml` (THREE dir levels under
  `kore/tasks/`). The registry discovers tasks with `TASKS_DIR.glob("*/task.yaml")` (ONE
  level, see `kore/tasks/registry.py::_discover`), so these are NOT auto-discovered and
  cannot enter any train/eval run or the running campaign. Verified on-node:
  `registry.task_ids()` returns 251 tasks and contains NONE of the 8 draft ids.
- All 8 operations contain `gemm`, so they classify as the trainable `gemm` family (not
  the held-out `mla` / `paged_attention` families). Verified via `operator_family` (all 8
  return `gemm`; `HELDOUT_FAMILIES = ("mla", "paged_attention")`).
- No draft id collides with a live task id (verified against the 251 live ids). The ids
  are deliberately distinct from the live `gemm_fp8_a8w8` (per-tensor), `gemm_mxfp4`
  (weight-only), `gemm_w4a16` (per-channel symmetric), `genv_gemm_a8w8_int8` (per-tensor
  generated) -- each draft is a genuinely different quant scheme (see the table).

## 1. What has already been verified (CPU, reference correctness)

- `~/kore-venv/bin/python kore/tasks/_drafts/quant/_cpu_sanity_check.py` -> ALL PASS.
  Each task's `reference.reference_output` was cross-checked on CPU against a fully
  INDEPENDENT fp32 dequant-matmul (a DIFFERENT code path: `torch.einsum` with the scales
  applied on the accumulator, `repeat_interleave` scale expansion, an ARITHMETIC e2m1
  decode instead of the reference LUT, and an explicit zero-point). Agreement was
  bf16-exact (SNR 999 dB, max_diff 0.000000, all `allclose=True`) on tiny primary shapes
  AND extra edges: multi-block block-scale indexing (2x2), multi-group int4 (2 groups),
  non-pow2 tails, and a transposed (non-contiguous) activation. This explicitly checks
  per-token vs per-tensor vs block vs group scaling and the zero-point / mxfp4-e8m0 math.
- K-alignment GUARDS verified: `get_inputs` with the illegal `K=4095` RAISES for
  `gemm_fp8_a8w8_blockscale` (K%128), `gemm_mxfp4_a4w4` (K%32), `gemm_w4a16_g128` (K%128),
  and `gemm_w4a8_fp8` (K odd / nibble packing).
- `python -m py_compile` succeeds on every drafted `.py` (syntax only; NOT a Triton compile).
- No U+2010..U+2015 / U+2212 dashes anywhere under `_drafts/quant/` (ASCII only, verified
  `grep -rnP`).
- fp8 arch resolved on-node: `aiter_ref.FP8_DTYPE == torch.float8_e4m3fn`, `FP8_MAX == 448.0`
  (OCP, gfx950/CDNA4). The oracle + candidate both consume this.

## 2. Global gfx950 verification (applies to EVERY task)

For each `<id>`, run from the repo root with the KORE venv active:

```
cp kore/tasks/_drafts/quant/<id>/seed_triton.py kore/tasks/_drafts/quant/<id>/kernel.py
~/kore-venv/bin/python kore/tasks/_drafts/quant/<id>/driver.py --shape "<primary>"          # correctness
~/kore-venv/bin/python kore/tasks/_drafts/quant/<id>/driver.py --bench-mode --impl reference --shape "<primary>"  # vendor baseline runs + times
~/kore-venv/bin/python kore/tasks/_drafts/quant/<id>/driver.py --bench-mode --impl candidate --shape "<primary>"  # seed times + anti-hack re-verify
rm kore/tasks/_drafts/quant/<id>/kernel.py
```

Per task, confirm ALL of:

- [ ] SEED COMPILES: the Triton `seed_triton.py` compiles on gfx950 (no LDS-over-160KiB,
      no unsupported dtype, no non-pow2 `tl.arange`) and produces output. fp8 tasks: the
      up-convert `.to(tl.float32)` of `float8_e4m3fn` operands must compile; the requant
      task must compile the direct fp8 STORE (`acc.to(c_ptr.dtype.element_ty)`).
- [ ] SEED CORRECT: correctness run prints `allclose: True` and `SNR` >= the task's
      `snr_threshold` (25 dB) across all >=5 reseeded trials AND all `shapes` (minimal,
      primary, every validation entry, including the `K=4095` legal-tail shape on the
      per-tensor / per-token / int8 / requant tasks, and the `TA:1` transposed-A shapes).
- [ ] VENDOR SYMBOL EXISTS + RUNS: `--impl reference` runs without error, i.e. the exact
      `comparison_baseline` is present in the installed AITER / hipBLASLt and accepts this
      task's layout/dtype (see per-task FLAGS).
- [ ] BASELINE == ORACLE (numerics): time-independent, confirm the vendor `reference`
      output matches `reference_output` within the SNR gate (guards against a vendor scale
      convention that differs from the oracle -- see per-task FLAGS).
- [ ] BENCH RUNS: both `--impl reference` and `--impl candidate` print `median_ms`, and the
      post-timing anti-hack re-verification prints `allclose: True`.
- [ ] fp8 arch: on gfx950 `aiter_ref.FP8_DTYPE` resolves to OCP `float8_e4m3fn` (max 448),
      NOT FNUZ. (fp8 tasks: 1, 2, 3, 5-is-int8, 8; and the fp8 activation in 7.)

## 3. Shared-driver promotion requirement

Every task's `driver.py` is a thin wrapper that imports `_quant_common` from its PARENT
directory, and each `reference.py` imports the shared oracles / quant helpers from the same
`_quant_common`. When promoting a task from `kore/tasks/_drafts/quant/<id>/` to the live
`kore/tasks/<id>/`:

- [ ] Copy `kore/tasks/_drafts/quant/_quant_common.py` to `kore/tasks/_quant_common.py`
      (once), so the promoted `driver.py` (parent dir now `kore/tasks/`) and `reference.py`
      still resolve it. (This mirrors how the live tasks share `kore/tasks/aiter_ref.py`.)
      Alternatively inline `_quant_common` into each task.
- [ ] Re-run the task's driver from the promoted location to confirm the imports resolve.
- [ ] Choose the final `snr_threshold`: 25 dB is the family floor (fp8/MX bar). The
      bf16-activation weight-only tasks (`gemm_w4a16_g128`) and the near-exact `gemm_int8_a8w8`
      will land well above 30 dB; raise their gate to 30 dB if desired after measuring.

## 4. Per-task status + FLAGS (read before promoting each)

Legend: baseline = the `comparison_baseline`; oracle = CPU-verified fp32 ground truth;
FLAG = a specific thing to confirm on gfx950 beyond the global checks.

| # | task_id | dtype | oracle (CPU-verified) | vendor baseline | key FLAG to verify on gfx950 |
|---|---------|-------|-----------------------|-----------------|------------------------------|
| 1 | gemm_fp8_a8w8_pertoken | fp8 e4m3fn | dequant-matmul, x_scale[M,1] per-token, w_scale[1,N] per-channel | `aiter.gemm_a8w8` | confirm CK `gemm_a8w8` accepts DISTINCT per-row x_scale + per-col w_scale (dynamic per-token, not just per-tensor) and fp8 OCP. The `TA:1` shape passes a NON-CONTIGUOUS xq: the candidate must be stride-correct; the vendor baseline may require `xq.contiguous()` (bench path only -- add if it errors, document as a copy) |
| 2 | gemm_fp8_a8w8_pertensor | fp8 e4m3fn | dequant-matmul, scalar scales broadcast to [M,1]/[1,N] | `aiter.gemm_a8w8` | none beyond global (this is the per-tensor special case of #1; complements the live `gemm_fp8_a8w8`) |
| 3 | gemm_fp8_a8w8_blockscale | fp8 e4m3fn | blockwise dequant-matmul, xs[M,K//128] (1x128), ws[N//128,K//128] (128x128) | `aiter.gemm_a8w8_blockscale` | HIGH-PRIORITY: confirm the symbol exists and its signature/scale-layout EXACTLY matches (1x128 activation groups, 128x128 weight blocks, `dtype=` kw). If the installed AITER uses a different block-scale entry (e.g. `gemm_a8w8_blockscale_bpreshuffle` with pre-shuffled weights), bind that in `baseline_output` and pre-shuffle OUTSIDE the timed region. If the layout differs, keep the oracle and adjust the vendor CALL (never the oracle) |
| 4 | gemm_mxfp4_a4w4 | mxfp4 (e2m1+e8m0/32) | dequant-matmul of BOTH operands in mxfp4 | bf16 dequant + hipBLASLt (`hipblaslt_gemm_bf16`) | HIGH-PRIORITY: the baseline is a REAL vendor lib (hipBLASLt via `torch.matmul`) but is a PERF-ONLY bar; the NATIVE fp4 path is `aiter.gemm_a4w4` (1x32 e8m0). Confirm `aiter.gemm_a4w4` exists + its exact packing/scale-shuffle (`e8m0_shuffle`) layout, then switch `baseline_output` to it (pre-shuffle outside timing) so the bar is the native MX-MFMA kernel. Confirm the e2m1/e8m0 encoding here matches the OCP MX spec the hardware decodes. K%32 guard confirmed on CPU |
| 5 | gemm_int8_a8w8 | int8 | dequant-matmul, per-token x_scale + per-channel w_scale | `aiter.gemm_a8w8` | confirm CK `gemm_a8w8` int8 path (`dtypes.i8`) accepts int8 XQ/WQ + fp32 scales. int8 is near-exact so SNR lands high; the 25 dB gate leaves margin (consider raising to 30 dB after measuring) |
| 6 | gemm_w4a16_g128 | int4 group asym (bf16 act) | grouped dequant-matmul, (code - zero[n,g]) * scale[n,g], group=128 | bf16 dequant + hipBLASLt | the group-wise ASYMMETRIC (zero-point) 4-bit weight is materialized to bf16 for the hipBLASLt bar (no direct grouped-w4a16 AITER symbol is wired here; this is the AWQ/GPTQ layout). Confirm the seed's `BLOCK_K == group` (=128) assumption holds for all validation shapes (all use group=128). `TA:1` passes a non-contiguous activation. Complements the live per-channel SYMMETRIC `gemm_w4a16` |
| 7 | gemm_w4a8_fp8 | int4 weight + fp8 act | dequant-matmul, fp8 per-token act * int4 per-channel weight | bf16 dequant + hipBLASLt | mixed-precision (4-bit weight, 8-bit fp8 activation, QServe/QoQ style). No single AITER w4a8 symbol is wired -> the bar is dequant-both-to-bf16 + hipBLASLt (perf-only). If an `aiter` w4a8 / mixed a8w4 kernel exists on the node, bind it as the vendor baseline. Confirm the two scales stay on their axes (activation per-row, weight per-col) -- the oracle pins this |
| 8 | gemm_fp8_requant_epilogue | fp8 e4m3fn | dequant-matmul + bias, fp8 requant with static out_scale, compared DEQUANTIZED | `aiter.gemm_a8w8` + unfused torch requant | confirm the fused Triton fp8 STORE path compiles + saturates correctly on gfx950 (`float8_e4m3fn`, clamp to +/-448 before cast, no inf). The baseline is the REAL vendor fp8 GEMM followed by an UNFUSED bias+requant (the multi-kernel bar the fused epilogue must beat). `out_scale` is a static per-tensor calibration computed in `get_inputs`; confirm it is treated as a fixed input, never recomputed from the candidate output |

## 5. Requested families -> drafted coverage (for reviewer sanity)

- fp8 a8w8 per-tensor: #2 (complements live `gemm_fp8_a8w8`).
- fp8 a8w8 per-token (dynamic, per-channel weight): #1 (the serving-critical path).
- fp8 a8w8 block-scaled (DeepSeek 1x128 / 128x128): #3.
- mxfp4 (OCP microscaling e2m1 + e8m0/32), gfx950-native: #4 (a4w4, both operands;
  complements the live weight-only `gemm_mxfp4`).
- int8 a8w8: #5.
- w4a16 (int4 weight / bf16 act): #6 (group-wise asymmetric AWQ/GPTQ; complements the
  live per-channel symmetric `gemm_w4a16`).
- w4a8: #7 (int4 weight + fp8 activation).
- fp8 GEMM with fused dequant/requant epilogue: #8.
- Shape coverage across the family: decode M=1, tails M=17 / M=4097, giant K=28672 /
  giant N=28672, non-pow2 K=4095 (LEGAL tail on the no-pack tasks; ILLEGAL-and-guarded on
  the block/MX/int4 tasks), transposed A (A^T, non-contiguous) on #1 and #6. B^T is the
  inherent layout of every task (W is stored [N,K], so the contraction is `A @ W^T`).

## 6. If a FLAG fails

Reference correctness is paramount. If a vendor baseline does not match the oracle on
gfx950, do NOT weaken the oracle to match the vendor. The oracle is the fp32 ground truth
(CPU-verified against an independent implementation that applies each scale exactly once).
Instead: (a) fix the vendor CALL (args / layout / scale-shuffle / pre-shuffle / dtype) so
it computes the same thing, or (b) if no vendor kernel computes this variant, keep the
oracle and set the perf baseline to the closest real vendor kernel (a dequant-to-bf16
hipBLASLt matmul is a REAL vendor-library bar, documented as perf-only -- as tasks #4, #6,
#7 already do), or (c) hold the task back. Update this checklist with the resolution
before promoting. Never claim GPU verification that has not been run.
