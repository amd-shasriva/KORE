# Attention DRAFT tasks: on-gfx950 verification checklist

STATUS: STAGED DRAFTS. Not live. Do NOT promote any task until the boxes below are
checked on a real gfx950 / MI350X (CDNA4) node with the installed AITER / AOTriton.

These tasks are integrity-critical: a wrong reference oracle would poison the KORE
verified-data moat. Everything here has been CPU-verified for REFERENCE CORRECTNESS ONLY
(the fp32 oracle math). NOTHING here has been GPU-verified: the Triton seeds have not been
compiled, the AITER vendor baselines have not been run, and no timing has been measured.

## 0. Why these are safe while staged

- They live at `kore/tasks/_drafts/attention/<id>/task.yaml` (THREE dir levels under
  `kore/tasks/`). The registry discovers tasks with `TASKS_DIR.glob("*/task.yaml")` (ONE
  level, see `kore/tasks/registry.py::_discover`), so these are NOT auto-discovered and
  cannot enter any train/eval run or the running campaign. Verified:
  `registry.task_ids()` contains none of the 11 draft ids.
- Operation names all contain `attn` and none contain `mla` / `latent` / `paged`, so they
  classify as the trainable `attention` family (not the held-out `mla` /
  `paged_attention` families). Verified via `operator_family` / `is_heldout`.
- No draft id collides with a live task (the live `flash_attn_varlen_bf16` is a DIFFERENT,
  causal task; this draft set adds the NON-causal `flash_attn_varlen_noncausal_bf16`).

## 1. What has already been verified (CPU, reference correctness)

- `python kore/tasks/_drafts/attention/_cpu_sanity_check.py` -> ALL PASS. Each task's
  `reference.reference_output` was cross-checked on CPU against a fully INDEPENDENT
  brute-force attention (`torch.softmax` + `torch.einsum` with independently-constructed
  masks; the gpt-oss sink cross-checked via the equivalent concat-a-sink-column
  formulation). Agreement was bf16-exact (SNR ~inf to 107 dB, all `allclose=True`),
  including non-power-of-2 seqlen edges, `W >= Skv` (full window), `Sq == Skv` (ordinary
  causal), and the ragged length-1 / max-length edges.
- `python -m py_compile` succeeds on every drafted `.py` (syntax only; NOT a Triton compile).
- No U+2013 / U+2014 dashes anywhere under `_drafts/` (ASCII only).

## 2. Global gfx950 verification (applies to EVERY task)

For each `<id>`, run from the repo root with the KORE venv active:

```
cp kore/tasks/_drafts/attention/<id>/seed_triton.py kore/tasks/_drafts/attention/<id>/kernel.py
python kore/tasks/_drafts/attention/<id>/driver.py --shape "<primary shape>"          # correctness
python kore/tasks/_drafts/attention/<id>/driver.py --bench-mode --impl reference --shape "<primary>"   # vendor baseline runs + times
python kore/tasks/_drafts/attention/<id>/driver.py --bench-mode --impl candidate --shape "<primary>"   # seed times + anti-hack re-verify
rm kore/tasks/_drafts/attention/<id>/kernel.py
```

Per task, confirm ALL of:

- [ ] SEED COMPILES: the Triton `seed_triton.py` compiles on gfx950 (no LDS-over-160KiB,
      no unsupported dtype, no non-pow2 `tl.arange`) and produces output.
- [ ] SEED CORRECT: correctness run prints `allclose: True` and `SNR` >= the task's
      `snr_threshold` (25 dB) across all >=5 reseeded trials AND all `shapes` (minimal,
      primary, every validation entry).
- [ ] VENDOR SYMBOL EXISTS + RUNS: `--impl reference` runs without error, i.e. the exact
      `comparison_baseline` symbol is present in the installed AITER and accepts this
      task's layout/dtype. (API confirmed present at draft time: `aiter.flash_attn_func`
      has params `causal`, `window_size`, `sink_ptr`, `cu_seqlens_q/kv`;
      `aiter.flash_attn_varlen_func` exists. Re-confirm on the target node/version.)
- [ ] BASELINE == ORACLE (numerics): time-independent, confirm the vendor `reference`
      output matches `reference_output` within the SNR gate (guards against a vendor
      convention that differs from the oracle -- see per-task flags below).
- [ ] BENCH RUNS: both `--impl reference` and `--impl candidate` print `median_ms`, and
      the post-timing anti-hack re-verification prints `allclose: True`.
- [ ] fp8 arch: on gfx950 `kore.tasks.aiter_ref.FP8_DTYPE` resolves to OCP
      `float8_e4m3fn` (max 448), NOT FNUZ. (fp8 tasks only.)

## 3. Shared-driver promotion requirement

Every task's `driver.py` is a thin wrapper that imports `_attn_common` from its PARENT
directory, and each `reference.py` imports the shared fp32 oracle
(`sdpa_fp32` / `expand_kv` / `causal_mask` / `sliding_window_mask`) from the same
`_attn_common`. When promoting a task from `kore/tasks/_drafts/attention/<id>/` to the
live `kore/tasks/<id>/`:

- [ ] Copy `kore/tasks/_drafts/attention/_attn_common.py` to `kore/tasks/_attn_common.py`
      (once), so the promoted `driver.py` (parent dir now `kore/tasks/`) and `reference.py`
      still resolve it. (This mirrors how the live attention tasks share
      `kore/tasks/aiter_ref_attn.py`.) Alternatively inline `_attn_common` into each task.
- [ ] Re-run the task's driver from the promoted location to confirm the imports resolve.

## 4. Per-task status + FLAGS (read before promoting each)

Legend: baseline = the `comparison_baseline` vendor symbol; FLAG = a specific thing to
confirm on gfx950 beyond the global checks.

| # | task_id | dtype | oracle (CPU-verified) | vendor baseline | key FLAG to verify on gfx950 |
|---|---------|-------|-----------------------|-----------------|------------------------------|
| 1 | flash_attn_mha_prefill_bf16 | bf16 | causal MHA (H==KV) SDPA | `aiter.flash_attn_func(causal=True)` | none beyond global |
| 2 | flash_attn_noncausal_prefill_bf16 | bf16 | bidirectional GQA SDPA | `aiter.flash_attn_func(causal=False)` | none beyond global |
| 3 | flash_attn_mqa_prefill_bf16 | bf16 | causal SDPA, KV=1 broadcast | `aiter.flash_attn_func(causal=True)` | confirm aiter accepts Hkv=1 (MQA) |
| 4 | flash_attn_headdim_prefill_bf16 | bf16 | causal GQA SDPA, D in {64,128,192,256} | `aiter.flash_attn_func(causal=True)` | D=192 (non-pow2): seed pads to 256 + masks -- confirm it COMPILES and is correct; confirm aiter supports D=192 and D=256 (else drop those validation shapes) |
| 5 | flash_attn_noncausal_fp8 | fp8_e4m3fn | bidirectional SDPA on dequant fp8 | bf16 `aiter.flash_attn_func(causal=False)` on dequant | confirm FP8 OCP e4m3fn; the fp8 kernel wins on bandwidth, not the baseline math |
| 6 | flash_attn_mqa_decode_bf16 | bf16 | non-causal decode SDPA, KV=1 | `aiter.flash_attn_func(causal=False)` seq_q=1 | confirm aiter decode path accepts seq_q=1 + Hkv=1 |
| 7 | flash_attn_decode_fp8 | fp8_e4m3fn | non-causal decode SDPA on dequant fp8 | bf16 `aiter.flash_attn_func(causal=False)` seq_q=1 on dequant | confirm FP8 OCP e4m3fn; decode seq_q=1 path |
| 8 | flash_attn_varlen_noncausal_bf16 | bf16 | per-seq NON-causal SDPA (packed) | `aiter.flash_attn_varlen_func(causal=False)` | confirm varlen_func arg order + int32 cu_seqlens + non-causal path; NOTE a live CAUSAL `flash_attn_varlen_bf16` already exists (this is the distinct non-causal complement) |
| 9 | flash_attn_sliding_decode_bf16 | bf16 | windowed decode SDPA (last W keys) | dense `aiter.flash_attn_func(causal=False)` seq_q=1 | baseline is DENSE decode on purpose (perf-only bar; the windowed kernel beats it by reading only W keys). Correctness is the WINDOWED oracle. Optionally switch the baseline to `flash_attn_func(window_size=(W-1,0,0))` ONCE the window_size tuple semantics are confirmed |
| 10 | flash_attn_sink_prefill_bf16 | bf16 | causal GQA SDPA + per-head sink logit | `aiter.flash_attn_func(causal=True, sink_ptr=sinks)` | HIGH-PRIORITY FLAG: confirm the installed aiter `sink_ptr` semantics EXACTLY match the gpt-oss oracle (per-head additive sink logit in the softmax denominator, no value) AND the expected `sink_ptr` dtype/shape ([H], fp32?). If aiter's sink differs, the baseline will NOT equal the oracle -- in that case keep the oracle (it is the verified gpt-oss math) and switch the perf baseline to dense `flash_attn_func(causal=True)` |
| 11 | flash_attn_chunked_prefill_bf16 | bf16 | bottom-right causal GQA SDPA (Sq<Skv) | `aiter.flash_attn_func(causal=True)` Sq!=Skv | HIGH-PRIORITY FLAG: confirm the installed aiter uses BOTTOM-RIGHT causal alignment when Sq != Skv (last query attends the full context) to match the oracle. If aiter aligns top-left instead, either pass an explicit mask/`cu_seqlens` to force bottom-right or adjust the oracle's `q_offset` to match the vendor+model convention (KEEP oracle and vendor in agreement) |

## 5. Requested families -> drafted coverage (for reviewer sanity)

- flash prefill causal: #1 (MHA), #3 (MQA), #4 (head-dim edges); + existing live GQA prefill.
- flash prefill non-causal: #2 (bf16), #5 (fp8).
- flash decode: #6 (MQA bf16), #7 (fp8); + existing live GQA decode.
- varlen / packed: #8 (non-causal); + existing live causal varlen.
- sliding-window: #9 (decode); + existing live prefill SWA.
- GQA / MQA (grouped-query): #1 (MHA extreme), #3 (MQA prefill), #6 (MQA decode).
- attention-sink: #10 (gpt-oss per-head sink).
- chunked-prefill: #11 (bottom-right causal, Sq<Skv).
- fp8 where vendor supports: #5, #7.
- Deliberately EXCLUDED (reserved HELDOUT_FAMILIES): MLA / latent attention, paged-KV
  decode. None drafted.

## 6. If a FLAG fails

Reference correctness is paramount. If a vendor baseline does not match the oracle on
gfx950, do NOT weaken the oracle to match the vendor. The oracle is the fp32 ground truth
(CPU-verified against an independent implementation). Instead: (a) fix the vendor CALL
(args/layout/convention) so it computes the same thing, or (b) if no vendor kernel
computes this variant, keep the oracle and set the perf baseline to the closest dense
vendor kernel (documented as a perf-only bar, as tasks #9 already does), or (c) hold the
task back. Update this checklist with the resolution before promoting.
