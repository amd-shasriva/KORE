# MoE DRAFT tasks: on-gfx950 verification checklist

STATUS: STAGED DRAFTS. Not live. Do NOT promote any task until the boxes below are
checked on a real gfx950 / MI350X (CDNA4) node with the installed AITER / hipBLASLt.

These tasks are integrity-critical: a wrong reference oracle would poison the KORE
verified-data moat. Everything here has been CPU-verified for REFERENCE CORRECTNESS ONLY
(the fp32 oracle math + the routing/dispatch/combine logic). NOTHING here has been
GPU-verified: the Triton seeds have not been compiled, the AITER vendor baselines have not
been run, and no timing has been measured.

## 0. Why these are safe while staged

- They live at `kore/tasks/_drafts/moe/<id>/task.yaml` (THREE dir levels under
  `kore/tasks/`). The registry discovers tasks with `TASKS_DIR.glob("*/task.yaml")` (ONE
  level, see `kore/tasks/registry.py::_discover`), so these are NOT auto-discovered and
  cannot enter any train/eval run or the running campaign. Verified: `registry.task_ids()`
  is unchanged at 251 and contains none of the 8 draft ids; none collide with a live id.
- Operation names all contain `moe` (routers also contain `topk`) and none contain
  `mla` / `latent` / `paged` / `attn`, so `operator_family()` returns the trainable
  `moe` (6 tasks) or `moe_router` (2 tasks) families -- NOT the held-out `mla` /
  `paged_attention` families. Verified via `operator_family` (6x `moe`, 2x `moe_router`).

## 1. What has already been verified (CPU, reference correctness)

- `python kore/tasks/_drafts/moe/_cpu_sanity_check.py` -> ALL PASS. Each task's
  `reference.reference_output` was cross-checked on CPU against a fully INDEPENDENT torch
  implementation (a DIFFERENT code path: per-(token,slot) gather + `bmm` instead of the
  reference's per-expert group loop; manual `exp`/`argsort` softmax instead of
  `torch.softmax`+`torch.topk`; an explicit per-token python loop for the DeepSeek-V3
  biased grouped router). Agreement was bf16-exact (SNR ~inf to 141 dB, all
  `allclose=True`), including non-power-of-2 dims, higher top-k, the 0-token last-expert
  edge, and single-token decode (M=1).
- Two integrity checks passed EXPLICITLY (MoE dispatch is error-prone):
  - token->expert assignment (permute): `sort_idx` is a permutation, it groups every
    expert's tokens into one contiguous ascending-expert block, the gather is exact, and
    the last expert receives 0 tokens.
  - weighted combine (`moe_sum`): equals an explicit per-slot `sum_k w[m,k]*y[m,k,:]`.
- `python -m py_compile` succeeds on every drafted `.py` (syntax only; NOT a Triton compile).
- No U+2012..U+2015 dashes anywhere under `_drafts/moe/` (ASCII only).

## 2. Global gfx950 verification (applies to EVERY task)

For each `<id>`, run from the repo root with the KORE venv active:

```
cp kore/tasks/_drafts/moe/<id>/seed_triton.py kore/tasks/_drafts/moe/<id>/kernel.py
python kore/tasks/_drafts/moe/<id>/driver.py --shape "<primary shape>"                            # correctness
python kore/tasks/_drafts/moe/<id>/driver.py --bench-mode --impl reference --shape "<primary>"     # vendor baseline runs + times
python kore/tasks/_drafts/moe/<id>/driver.py --bench-mode --impl candidate --shape "<primary>"     # seed times + anti-hack re-verify
rm kore/tasks/_drafts/moe/<id>/kernel.py
```

Per task, confirm ALL of:

- [ ] SEED COMPILES: the Triton `seed_triton.py` compiles on gfx950 (no LDS-over-160KiB,
      no unsupported dtype, no non-pow2 `tl.arange`) and produces output.
- [ ] SEED CORRECT: correctness run prints `allclose: True` and `SNR` >= the task's
      `snr_threshold` (25 dB) across all >=5 reseeded trials AND all `shapes` (minimal,
      primary, every validation entry).
- [ ] VENDOR SYMBOL EXISTS + RUNS: `--impl reference` runs without error, i.e. the exact
      `comparison_baseline` symbol is present in the installed AITER and accepts this
      task's layout/dtype. (Confirmed present at draft time via the live MoE tasks:
      `aiter.fused_moe.fused_moe`, `aiter.topk_softmax`, `aiter.gemm_a8w8`,
      `aiter.batched_gemm_bf16`, `aiter.ops.shuffle.shuffle_weight`. Re-confirm on node.)
- [ ] BASELINE == ORACLE (numerics): time-independent, confirm the vendor `reference`
      output matches `reference_output` within the SNR gate (guards against a vendor
      convention that differs from the oracle -- see per-task FLAGS below).
- [ ] BENCH RUNS: both `--impl reference` and `--impl candidate` print `median_ms`, and
      the post-timing anti-hack re-verification prints `allclose: True`.
- [ ] fp8 arch: on gfx950 `kore.tasks.aiter_ref.FP8_DTYPE` resolves to OCP
      `float8_e4m3fn` (max 448), NOT FNUZ. (fp8 task `moe_grouped_gemm_fp8` only.)

## 3. Shared-driver promotion requirement

Every task's `driver.py` is a thin wrapper that imports `_moe_common` from its PARENT
directory, and each `reference.py` imports the shared fp32 oracles / routing / fp8 quant /
vendor wrappers from the same `_moe_common`. When promoting a task from
`kore/tasks/_drafts/moe/<id>/` to the live `kore/tasks/<id>/`:

- [ ] Copy `kore/tasks/_drafts/moe/_moe_common.py` to `kore/tasks/_moe_common.py` (once),
      so the promoted `driver.py` (parent dir now `kore/tasks/`) and `reference.py` still
      resolve it. (This mirrors how the live MoE tasks share `kore.tasks.aiter_ref_attn`.)
      Alternatively inline `_moe_common` into each task.
- [ ] Re-run the task's driver from the promoted location to confirm the imports resolve.
- [ ] Add a matching `driver.py` correctness/bench contract if the promoted task should use
      the per-task driver style of the existing live MoE tasks (or keep the shared driver).

## 4. Per-task status + FLAGS (read before promoting each)

Legend: family = `operator_family()`; baseline = the `comparison_baseline` vendor symbol;
FLAG = a specific thing to confirm on gfx950 beyond the global checks.

| # | task_id | dtype | family | oracle (CPU-verified) | vendor baseline | key FLAG to verify on gfx950 |
|---|---------|-------|--------|-----------------------|-----------------|------------------------------|
| 1 | moe_gelu_bf16 | bf16 | moe | top-k GeGLU (tanh-GELU gate*up) MLP + weighted combine | `aiter.fused_moe.fused_moe(ActivationType.Gelu, QuantType.No)` | HIGH-PRIORITY FLAG: confirm `ActivationType.Gelu` exists AND that aiter's gated GELU is the tanh approximation (the oracle uses `F.gelu(approximate="tanh")`). If aiter ships erf-GELU or a different gate, the baseline will not equal the oracle -- keep the oracle (the task IS tanh-GeGLU) and either point the baseline at the exact matching aiter activation or hold the task. Also confirm `shuffle_weight(layout=(16,16))` accepts these `[E,2I,D]`/`[E,D,I]` shapes. |
| 2 | moe_batched_gemm_bf16 | bf16 | moe | fp32 batched A@B^T | `aiter.batched_gemm_bf16` (torch.bmm->hipBLASLt fallback) | confirm `batched_gemm_bf16(A[E,m,K], B[E,N,K], out)` computes A@B^T per batch into a `[E,m,N]` bf16 out; non-pow2 N=513 tail must mask correctly. Fallback to `torch.bmm` is built in, so `--impl reference` always runs. |
| 3 | moe_grouped_gemm_bf16 | bf16 | moe | fp32 per-expert segmented GEMM (top-1, 0-token last expert) | per-expert `torch.matmul` -> hipBLASLt | baseline always runs (dense hipBLASLt). Confirm the seed's per-(token,n-tile) grid does not exceed launch limits at M=8192 (consider promoting to a token-sorted tiled grouped GEMM). 0-token last expert is simply never visited (correct). |
| 4 | moe_topk_softmax_norenorm_bf16 | bf16 | moe_router | fp32 softmax -> top-k, NO renorm, dense [M,E] | `aiter.topk_softmax(..., renormalize=False)` | HIGH-PRIORITY FLAG: confirm the installed `aiter.topk_softmax` honors `renormalize=False` (returns the RAW softmax probs of the selected experts). Some builds always renormalize; if so the vendor output will not equal this (un-renormalized) oracle -- keep the oracle and either find the un-renormalized vendor path or document the baseline as the renormalized variant / hold. (The live `topk_softmax_bf16` covers the renormalized case with the same symbol.) |
| 5 | moe_biased_grouped_topk_bf16 | bf16 | moe_router | DeepSeek-V3 sigmoid + bias grouped top-k, dense [M,E] | `aiter.biased_grouped_topk` (oracle fallback) | HIGH-PRIORITY FLAG: `aiter.biased_grouped_topk`'s exact signature/semantics are version-dependent (arg order; sigmoid vs softmax gate; correction-bias handling; `num_expert_group`/`topk_group` names; `renormalize`; `routed_scaling_factor`). The wrapper tries one common form and otherwise FALLS BACK to the verified oracle (labeled `framework`). On node: confirm the real signature, confirm its result matches the oracle (sigmoid gate, bias used for SELECTION only, group top-2-sum, weights = renormalized sigmoid scores), and wire it in. If aiter's variant differs, KEEP the oracle (DSV3-correct, CPU cross-checked) and document the baseline. Also confirm the seed's per-token group loops compile at E=256/n_groups=8. |
| 6 | moe_permute_bf16 | bf16 | moe | exact gather permuted[i]=hidden[sort_idx[i]] (ATOL=RTOL=0) | framework indexed gather (`hidden[sort_idx]`) | baseline always runs. This is the memory-bound dispatch copy; `sort_idx` is precomputed (the sort/`moe_sorting` is a separate op). Optionally back the baseline with `aiter.moe_sorting`+gather once its public permuted layout is confirmed. Correctness is EXACT (no arithmetic), so any dropped/duplicated row fails. |
| 7 | moe_sum_combine_bf16 | bf16 | moe | fp32 weighted reduce sum_k w*y over top-k | framework weighted reduce | baseline always runs. If wiring `aiter`/FlagGems `moe_sum`: note the vendor `moe_sum` is typically an UNWEIGHTED reduce over the top-k axis, so pre-multiply `y` by the router weights (or fold weights in) to match this WEIGHTED-combine oracle. |
| 8 | moe_grouped_gemm_fp8 | fp8_e4m3fn | moe | fp32-of-dequant per-expert a8w8 segmented GEMM (top-1, 0-token last expert) | per-expert `aiter.gemm_a8w8` (CK) | confirm `aiter.gemm_a8w8(xq[n,K], wq[N,K], x_scale[n,1], w_scale[1,N], dtype=bf16)` accepts the per-expert token slice with per-token activation scale + per-channel weight scale; confirm FP8_DTYPE is OCP `e4m3fn` (max 448) on gfx950. The fp8 rounding is SHARED by candidate + oracle (both dequantize the same fp8 codes), so the gate measures accumulation/scale-fold fidelity, not quant error. |

## 5. Requested families -> drafted coverage (for reviewer sanity)

- fused MoE MLP (gate+up SiLU/GELU fused expert compute): #1 (GeGLU tanh-GELU); the SiLU
  variant already exists live (`fused_moe_silu_bf16`).
- grouped / batched expert GEMM: #3 (jagged segmented, top-1, 0-token edge), #2 (balanced
  batched expert GEMM).
- top-k router (softmax + top-k select): #4 (top-k softmax, no renorm), #5 (DeepSeek-V3
  biased grouped top-k); the renormalized softmax router already exists live
  (`topk_softmax_bf16`).
- expert sorting / permutation (scatter tokens to experts): #6 (dispatch permute), plus the
  weighted combine counterpart #7 (`moe_sum`).
- MoE + fp8 quantized expert GEMM: #8 (per-expert a8w8 grouped GEMM, OCP e4m3fn).
- Production shapes covered across tasks: num_experts E in {8,16,32,64,256}, top_k in
  {1,2,6,8}, group routing (n_groups=8, topk_group=4 DSV3), hidden D in {4096,5120,7168,
  8192}, expert inter I in {768,1024,1408,8192}, tokens M in {1..8192}, plus the mandatory
  unbalanced jagged trace with a giant expert + a guaranteed 0-token last expert.
- Deliberately EXCLUDED: MLA / paged (reserved HELDOUT_FAMILIES -- none drafted); a FULL
  fp8 fused-MoE (`fused_moe` with a fp8 QuantType) -- its per-token/per-tensor scale-passing
  API is version-dependent, so fp8 is covered by the confirmed `aiter.gemm_a8w8` grouped
  expert GEMM (#8) instead. Add the full fp8 `fused_moe` later once its scale API is
  confirmed on-node.

## 6. If a FLAG fails

Reference correctness is paramount. If a vendor baseline does not match the oracle on
gfx950, do NOT weaken the oracle to match the vendor. The oracle is the fp32 ground truth
(CPU-verified against an independent implementation). Instead: (a) fix the vendor CALL
(args/layout/convention) so it computes the same thing, or (b) if no vendor kernel computes
this variant, keep the oracle and set the perf baseline to the closest dense vendor kernel
(documented as a perf-only bar, as #3/#6/#7 already do with hipBLASLt / framework), or
(c) hold the task back. Update this checklist with the resolution before promoting.
