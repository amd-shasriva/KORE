# P0 (roofline / Speed-of-Light) results - native gfx950 (MI350-class / CDNA4)

Status: **AITER-backed, peak-calibrated, CI'd. Verdict PARTIAL - check (b) is a decisive PASS
(the load-bearing result), check (a) PASSES against the AITER gold baseline, check (c) is WEAK
(expected pre-RL).**

## Node & stack
- Host - 8× **gfx950** (AMD Instinct **MI350-class**, CDNA4), ROCm 7.2.3, `rocprofv3`. All GPU
  measurement on **one** device (a separate node runs datagen; never touched here).
- **Main stack:** `torch 2.10.0+rocm7.0` + `pytorch-triton-rocm 3.5.1` (native gfx950). Used for the
  roofline model, PMC, candidate kernels, and peak calibration.
- **AITER baseline stack (isolated):** AITER's kernels require `triton ≥ 3.6`, which the main stack
  does not ship. Rather than disturb the (datagen-shared) main venv, the AITER gold baseline is run
  from a **separate** venv - the *local* `torch 2.10.0+rocm7.0` wheel + `triton 3.6.0` + a separate
  AITER checkout built against it - so `aiter.ops.*` CK kernels JIT-compile and run on gfx950. This
  never modifies the main venv, the shared repo's runtime, or `~/aiter`.

## Roofline model
`T_min = max(W_flops/P_peak, Q_bytes/B_peak)`, `eta = T_min/T_measured ∈ (0,1]` (SOL attainment).
Peaks are overridable via `KORE_PEAK_{BF16,FP8,HBM_BW}` (see calibration below).

## Peak calibration (Phase 2 - measured achievable, not datasheet)
On-device microbenchmarks (`kore.analysis.calibrate_peaks`, CUDA-event batched timing):

| peak | datasheet | **measured achievable** | attained | method |
|---|---|---|---|---|
| HBM bandwidth | 8.0 TB/s | **4.60 TB/s** | 57% | STREAM triad `a = b + q·c` (3·N·4 B traffic) |
| bf16 matrix | 2.5 PF/s | **1.27 PF/s** | 51% | 8192³ square matmul (`2N³` FLOPs, sustained) |
| fp8 matrix | 5.0 PF/s | *(kept datasheet)* | - | `torch._scaled_mm` unavailable on this stack |

Measured peaks are piped through `KORE_PEAK_HBM_BW=4.599e12`, `KORE_PEAK_BF16=1.273e15` (the fp8
peak keeps the datasheet value). Using the *achievable* peak makes `eta` a defensible
"fraction-of-attainable-SOL"; because every kernel of a dtype divides by the same peak, this only
rescales absolute `eta` - the *relationships* the three checks test are unchanged (verified: check
(b) R² is 0.978 calibrated vs 0.99 uncalibrated; both PASS). Details in `data/calibration.json`.

## Final study (Phase 3 - 3 representative shapes × 3 reseeds, PMC, bootstrap CIs)
`data/p0_study_final.json`: every operator measured at **3 representative shapes** (primary +
validation_0 + validation_1; the tiny `minimal` correctness shape is *excluded* - it is
launch-overhead-bound, where the roofline model does not apply, see Limitations), each timing
**reseeded 3×** (median-of-medians, L2-flushed cold-cache), with **rocprofv3 PMC** and **1000×
bootstrap** 95% CIs. 132 correct kernel×shape points.

```
(a) eta predicts speedup   : rho = 0.529   n=114   95% CI [0.346, 0.701]   -> PASS
(b) residual decomposition : R^2 = 0.9783  n=132   95% CI [0.967, 0.989]   -> PASS  (load-bearing)
(c) monotone-in-valley     : frac= 0.525   pairs=59 95% CI [0.393, 0.646]  -> WEAK  (expected pre-RL)
VERDICT: PARTIAL
```

**Check (b) is the headline.** The runtime residual `(T_measured − T_min)` reconstructs from
counter-derived **named** terms - memory-stall time (`MemUnitStalled`) + occupancy-deficit time
(`1 − OccupancyPercent`) - with **R² = 0.978 (95% CI [0.967, 0.989])** across 132 kernels. The CI
lower bound stays well above 0.9: the "named gradient" the KORE paradigm descends is real and
measurable, not drowned by cross-terms.

**Check (a) PASSES against the AITER gold baseline** (`ρ = 0.529`, CI entirely positive
[0.346, 0.701]): kernels nearer the roofline attain higher speedup vs the *production* vendor.

## check (a) baselines - AITER gold, honestly labeled
The reference for each operator is now the real production kernel, tagged per operator in the JSON:

| operator | baseline | median speedup (seed→best vs vendor) |
|---|---|---|
| rmsnorm / layernorm / fused_add_rmsnorm | **AITER CK** (`rms_norm`, `layer_norm`, `fused_add_rms_norm_cu`) | 0.71 / 0.98 / 0.86× |
| silu_and_mul / rope | **AITER CK** (`silu_and_mul`, `rope_fwd`) | 0.64 / 0.33× |
| flash_attn_decode / prefill / paged_attn_decode | **AITER** FMHA / ROCm paged attn | 1.32 / 0.07 / 0.15× |
| fused_moe_silu / topk_softmax | **AITER** `fused_moe` / `topk_softmax` | 0.04 / 0.29× |
| gemm_bf16 | **hipBLASLt** (`torch.matmul`) | 0.50× |
| softmax / gelu_tanh | framework (torch - no standalone AITER op) | 0.51 / 0.89× |
| gemm_fp8_a8w8 / quant_fp8_pertoken | η-only (fp8 vendor path not built on this stack) | - |

Baseline composition (operators with a vendor speedup): **10 AITER-vendor, 1 hipBLASLt, 2 framework**.
Every AITER baseline passes the same torch-fp32 correctness oracle used for candidates.

## PMC (gfx950)
gfx950/CDNA4 renamed the raw counters, so the original `SQ_*` list collected nothing. We use
rocprofv3 **derived metrics** (`OccupancyPercent`, `MemUnitStalled`, `MfmaUtil`, `GRBM_GUI_ACTIVE`),
parse the long-format `*_counter_collection.csv`, and pick the longest-running compute kernel.

## Limitations
- **`minimal`-shape regime:** on the tiny correctness shapes (e.g. M=64,N=512) every kernel is
  launch/overhead-bound (`η < 2%`) and the roofline does not model launch cost; pooling them destroys
  the check-(a) correlation (ρ → 0). Check (a) is therefore reported on **representative** shapes only
  - an honest scope statement, not a tuning knob.
- **fp8 peak** is datasheet (no `_scaled_mm`); the two fp8 operators are η-only.
- **check (c)** trajectories are schedule-mutations of the seed, not an RL policy - they show the
  residual *moves* with schedule, not that a policy drives it monotonically. WEAK is the expected
  pre-RL reading.

## Reproduce
```bash
# calibrate peaks (main stack)
python -m kore.analysis.calibrate_peaks --out data/calibration.json
# AITER gold baseline needs triton>=3.6 in a SEPARATE venv (never the datagen-shared one):
#   python3.10 -m venv ~/kore-venv-aiter && \
#   ~/kore-venv-aiter/bin/pip install <local torch 2.10+rocm7.0 wheel> triton==3.6 pandas einops psutil ninja pybind11 && \
#   pip install -e <separate aiter checkout>
# final study (calibrated peaks + AITER baselines + PMC + CIs):
KORE_PEAK_HBM_BW=4.599e12 KORE_PEAK_BF16=1.273e15 PYTHONPATH=<aiter2> \
  python -m kore.analysis.p0_sol --shapes-per-task 3 --reseeds 3 --bootstrap 1000 \
    --warmup 5 --iters 20 --max-kernels-per-task 3 --out data/p0_study_final.json
python -m kore.analysis.plots --report data/p0_study_final.json --out figures/
```

## Downstream (built, not trained)
- **Residual-descent reward** (`kore.reward.physics`): scores a correct kernel by the fraction of the
  *named* residual it removes vs `T_min` (`ρ_phys = T_min/(T_min+N)`), delegating all
  hack/compile/correctness gating to the existing lexicographic reward; graceful η fallback when PMC
  is absent. Unit-tested.
- **Zero-shot generalization harness** (`kore.eval.generalization`): leakage-checked hold-out of whole
  operator families; offline eval of η + residual-descent on held-out families. Unit-tested.
- No RL/GRPO training has been run - this is a press-go state.
