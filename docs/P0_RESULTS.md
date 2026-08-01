# P0 roofline / Speed-of-Light validation — native gfx950 (MI350-class / CDNA4)

**Verdict: `INTEGRITY_ONLY`. All three preregistered checks FAIL under the v2 analysis.**

An earlier revision of this document reported `PARTIAL`, with check (b) as a "decisive PASS" at
R² = 0.978. That reading does not survive the controls this repository now implements. The stored
artifact `data/p0_study_final.json` is a v1-schema report; re-running the current adjudicator
(`kore.analysis.p0_sol.reanalyze_report`) on that same file returns `INTEGRITY_ONLY` with every
check failing. The numbers below are from that re-analysis, not from the stored verdict strings.

The physics model is retained, but its role is now stated correctly: **η is a bounded, PMC-free
shaping potential and a speed-of-light integrity ceiling. It is not a validated predictor of
speedup.** No operator family currently authorizes empirical residual shaping.

## Why the earlier reading was wrong

Both failing checks share the same defect: the quantity under test and the quantity it is being
tested against share the denominator `T_candidate`, which varies over orders of magnitude across
kernels. A high correlation or R² therefore follows from scale alone.

The regression target in check (b) was `residual = T_candidate − T_min` and the regressors were
`stall_frac · T_candidate` and `occupancy_deficit · T_candidate`. The v2 analysis adds three
controls, and all three are decisive:

| Control | Result |
| --- | --- |
| Named-term model, raw in-sample R² | **0.9783** (reproduces the previously reported headline exactly) |
| `T_candidate`-only predictor, raw in-sample R² | **0.9971** — *higher* than the named model |
| `T_candidate`-only, task-level cross-validated R² | **0.9970** vs the named model's 0.9471 |
| Preregistered normalized target, held-out task clusters | **R² = −0.4582** |

The preregistered primary target is the normalized gap `(T_candidate − T_min) / T_candidate`,
which removes the shared scale. On held-out task clusters it scores −0.458 — worse than predicting
the mean. Against a `T_candidate`-only baseline the increment is **−0.3365**, 95% CI
[−0.4880, +0.6078]. In-sample it reaches only 0.3140. `p = p_adjusted = 0.986`.

Check (a) fails the same way. η = `T_min / T_candidate` and speedup = `T_vendor / T_candidate`
again share `T_candidate`:

```
(a) eta predicts speedup   : rho = 0.5290   n = 114   95% CI [0.1762, 0.8229]
    T_candidate-only rho   :       0.7274
    increment over T_cand  :      -0.1984   95% CI [-0.4862, +0.0474]   p = 0.970   -> FAIL
(b) residual decomposition : normalized held-out CV R2 = -0.4582  n = 132, 15 task clusters
    increment over T_cand  :      -0.3365   95% CI [-0.4880, +0.6078]   p = 0.986   -> FAIL
(c) monotone-in-valley     : frac = 0.500   95% CI [0.3611, 0.6286]                 -> FAIL
DECISION: INTEGRITY_ONLY        model_fingerprint_status: legacy-unfingerprinted
shaping_evidence: disabled      authorized families: none
```

η is a *worse* predictor of speedup than the trivial `1/T_candidate` — that is, worse than the
statement "faster kernels are faster."

Reproduce:

```bash
PYTHONPATH=. python -c "
import json; from kore.analysis.p0_sol import reanalyze_report
print(json.dumps(reanalyze_report(json.load(open('data/p0_study_final.json'))), indent=2)[:4000])"
```

## Independent replication on measured peaks (`data/p0_study_calibrated.json`)

The re-analysis above adjudicates a *stored* v1 artifact whose peaks were datasheet numbers. The
obvious objection is that the model only fails because it was fed the wrong constants. It was
therefore re-measured end to end against **calibrated peaks from this device** (HBM 4.763 TB/s,
bf16 1.296 PF/s — 60% and 56% of datasheet), under a verified model fingerprint, on an otherwise
idle GPU:

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python -m kore.analysis.p0_sol \
  --tasks fused_add_rmsnorm_bf16,fused_moe_silu_bf16,gelu_tanh_bf16,gemm_bf16,gemm_fp8_a8w8,\
layernorm_bf16,quant_fp8_pertoken,rmsnorm_aiter,rope_bf16,silu_mul_bf16,softmax_bf16,\
topk_softmax_bf16,flash_attn_decode_bf16,flash_attn_prefill_bf16,paged_attn_decode_bf16 \
  --calibration data/calibration_v1.json \
  --expect-model-fingerprint sha256:a6e01795829dd9a1c11752e12ff84825241f1e7d1e752c47dd2d926f7b858c7a \
  --arch gfx950 --shapes-per-task 3 --reseeds 3 --bootstrap 1000 --permutations 1000 \
  --out data/p0_study_calibrated.json
```

```
(a) roofline beyond Tcand : rho = 0.6205   Tcand-only = 0.7291   delta = -0.1086   q = 0.277 -> FAIL
(b) normalized held-out   : R2  = 0.0557   Tcand-only  = -0.1987                   q = 0.004 -> FAIL
    raw in-sample         : named 0.9177 | Tcand-only 0.9814 | null median 0.9443 (p = 0.827)
(c) collection-order      : frac = 0.5185  27 pairs                                q = 0.500 -> FAIL
DECISION: INTEGRITY_ONLY        SHAPING: disabled; no family passed held-out evidence
```

Calibration helps and does not rescue. Check (a)'s deficit narrows from −0.198 to −0.109 once the
peaks are real, but η still trails the trivial `1/T_candidate`. The raw in-sample R² of 0.918
reproduces the old headline and remains *below* the denominator-preserving null's median of 0.944:
random regressors sharing the same denominator score higher, and the permutation test cannot reject
them (p = 0.83). Only under the normalized target does the named model finally beat its baseline
(+0.254, p = 0.001) — but at CV R² = 0.056 it explains essentially none of the variance, and its
95% CI [−0.327, +0.447] spans zero.

Two further findings sharpen the negative result:

- **Leave-one-family-out transfer is catastrophic.** Holding out a family and predicting it scores
  activation +0.124, gemm −0.224, reduction −4.39, norm −5.23, quant −5.43, positional −14.87. The
  fit is family-local; it does not generalize to an operator class it has not seen.
- **The model cannot express the operators that matter most.** Five of the fifteen requested
  operators — `fused_moe_silu`, `topk_softmax`, `flash_attn_decode`, `flash_attn_prefill`,
  `paged_attn_decode` — have no roofline at all and were dropped as *unsupported model*. What
  survives to be tested is ten memory- or compute-bound primitives. Attention and MoE, the shapes
  that dominate real LLM serving, are outside the model's domain entirely.

Both runs agree on `INTEGRITY_ONLY`, one from a stored artifact on datasheet peaks and one measured
fresh on calibrated peaks under a verified fingerprint. The verdict is not an artifact of stale
constants.

## What this does and does not invalidate

**Still valid.** The measurements themselves are sound: the peak calibration, the AITER baseline
timings, the PMC collection methodology, and the paired timing protocol are unaffected. The
speed-of-light ceiling remains usable as an *integrity* bound — a kernel timing faster than
`T_min` is physically impossible and is rejected — because that use requires only a conservative
lower bound on achievable time, not predictive validity.

**No longer claimable.** That counter-derived named terms explain the runtime residual; that η
predicts vendor-relative speedup; that the residual is a validated training signal. Any external
claim must be restated accordingly.

**Not contaminated.** Empirical per-family shaping was never active in training. It requires both
`physics_shaping_evidence_path` and `physics_shaping_evidence_fingerprint`, and `grpo_14b_full.json`
sets neither, so `_dense_profile_weight` returns 0.0. `reanalyze_report` independently reports
`shaping_evidence.status = "disabled"` with no authorized families, because a `legacy-unfingerprinted`
model can never authorize shaping. The shipped models are therefore not affected by this correction.

## Node & stack

- **Host:** 8× **gfx950** (AMD Instinct MI350-class, CDNA4), ROCm 7.2.3, `rocprofv3`. All GPU
  measurement runs on one device (datagen runs on a separate node).
- **Main stack:** `torch 2.10.0+rocm7.0` + `triton 3.6.0` (native gfx950).
- **AITER baseline:** AITER's kernels require `triton >= 3.6`. That constraint historically forced
  the vendor baseline into an isolated venv, and older revisions of this document describe that
  split. The main stack now ships `triton 3.6.0` and imports `aiter` directly
  (`aiter.jit.module_aiter_core` loads on gfx950), so the calibrated replication above ran the
  vendor baselines and the candidate timings in **one** environment. Confirm before trusting a
  vendor-anchored number: `python -c "import aiter, triton; print(triton.__version__)"`. If that
  import fails the wrappers degrade to torch, which the run records as `baseline_impl` rather than
  claiming a vendor win.

## Peak calibration (measured achievable, not datasheet)

On-device microbenchmarks (`kore.analysis.calibrate_peaks`, batched event timing):

| peak | datasheet | measured achievable | attained | method |
| --- | --- | --- | --- | --- |
| HBM bandwidth | 8.0 TB/s | **4.60 TB/s** | 57% | STREAM triad `a = b + q·c` (3·N·4 B traffic) |
| bf16 matrix | 2.5 PF/s | **1.27 PF/s** | 51% | 8192³ square matmul (`2N³` FLOPs, sustained) |
| fp8 matrix | 5.0 PF/s | *(datasheet)* | – | `torch._scaled_mm` unavailable on this stack |

> **Applying calibration.** Earlier revisions documented `KORE_PEAK_BF16` / `KORE_PEAK_HBM_BW` /
> `KORE_PEAK_FP8` environment overrides. Those were deliberately removed as invisible,
> unfingerprinted global calibration and are now a **silent no-op** — setting them changes nothing.
> Calibrated peaks must be supplied as a fingerprinted `kore.runtime-calibration.v1` document; see
> `kore/analysis/calibrate_peaks.py` and the `--calibration` path. Any figure produced without one
> is computed against **datasheet** peaks, which inflates `T_min` by roughly 1.74× (memory-bound)
> to 1.81× (compute-bound) and correspondingly roughly halves η.

Using an achievable rather than datasheet peak rescales η for every kernel of a dtype identically,
so it does not change any of the relationships tested above.

## Check (a) baselines — AITER production kernels

The reference for each operator is the real production kernel, tagged per operator in the JSON:

| operator | baseline | median speedup (seed→best vs vendor) |
| --- | --- | --- |
| rmsnorm / layernorm / fused_add_rmsnorm | **AITER CK** | 0.71 / 0.98 / 0.86× |
| silu_and_mul / rope | **AITER CK** | 0.64 / 0.33× |
| flash_attn_decode / prefill / paged_attn_decode | **AITER** FMHA / ROCm paged attn | 1.32 / 0.07 / 0.15× |
| fused_moe_silu / topk_softmax | **AITER** `fused_moe` / `topk_softmax` | 0.04 / 0.29× |
| gemm_bf16 | **hipBLASLt** (`torch.matmul`) | 0.50× |
| softmax / gelu_tanh | framework (torch — no standalone AITER op) | 0.51 / 0.89× |
| gemm_fp8_a8w8 / quant_fp8_pertoken | η-only (fp8 vendor path not built on this stack) | – |

Baseline composition: 10 AITER-vendor, 1 hipBLASLt, 2 framework. Every AITER baseline passes the
same torch-fp32 correctness oracle used for candidates.

**Read this table as the difficulty bar, not as a result.** These are offline schedule-mutation
kernels, not an RL policy. Only `flash_attn_decode` exceeds parity with the vendor library; the
rest lose, several by an order of magnitude. Beating AITER and hipBLASLt is the actual problem.

## Cross-family transfer

A leave-one-family-out experiment refits the named-term → residual map on all families but one and
predicts the held-out family. On the normalized target the held-out scores are strongly negative
for most families, so the residual is at best a **dense per-family** signal and does not transfer
zero-shot. Combined with the checks above, per-family fitting is a necessary but not sufficient
condition — no family currently reaches an authorizing verdict.

## PMC (gfx950)

gfx950/CDNA4 renamed the raw counters, so the original `SQ_*` list collected nothing. We use
rocprofv3 derived metrics (`OccupancyPercent`, `MemUnitStalled`, `MfmaUtil`, `GRBM_GUI_ACTIVE`),
parse the long-format `*_counter_collection.csv`, and select the longest-running compute kernel.

## Limitations

- **`minimal`-shape regime:** on tiny correctness shapes every kernel is launch/overhead-bound
  (η < 2%) and the roofline does not model launch cost; check (a) is reported on representative
  shapes only.
- **fp8 peak** is datasheet (no `_scaled_mm`); the two fp8 operators are η-only.
- **The stored artifact is v1-schema and `legacy-unfingerprinted`.** Its `checks.*.verdict` strings
  (`PASS`/`WEAK`) are not values the current code can emit and must not be quoted. A definitive
  re-run on-box with a fingerprinted calibration document is required before any figure derived
  from this study is published.

## How the physics enters training

- **Integrity ceiling (active).** `T_min` rejects physically impossible timings. This use is sound
  under a conservative lower bound and does not depend on predictive validity.
- **Shaping potential (structurally present, empirically unauthorized).** `Φ = η` is wired into
  GRPO credit as `F_t = γ·Φ(s_{t+1}) − Φ(s_t)`. Potential-based shaping preserves the ordering of
  returns regardless of whether Φ is predictive, so this is safe — but it should be described as a
  variance-reduction heuristic, not as a validated hardware signal.
- **Residual-descent reward (available, not validated).** `reward_mode="residual"` exists and is
  unit-tested as an alternative objective. Given the results above it should not be used as a
  primary objective without a fresh, fingerprinted study that passes the v2 checks.
