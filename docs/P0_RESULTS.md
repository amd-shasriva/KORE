# P0 (roofline / Speed-of-Light) results - native gfx950 (MI350-class / CDNA4)

Status: **AITER-backed, peak-calibrated, CI'd. Verdict PARTIAL - check (b) is a decisive PASS
(the load-bearing result), check (a) PASSES against the AITER gold baseline, check (c) is WEAK
(expected pre-RL). All results here are OFFLINE; the live GRPO reward uses the PMC-free `η`, not the
validated `ρ` (see the Downstream section below).**

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

**But R²≈0.98 is IN-SAMPLE and does NOT transfer across operator families (OFFLINE crux).** The
leave-one-family-out experiment (`kore.analysis.residual_transfer`) refits the named-term → residual map
on all families but one and predicts the held-out family: pooled in-sample R² = 0.978, but **median
out-of-family R² = 0.107 (raw) / negative (normalized)**, and families are separable in residual space.
**Verdict: the residual value is operator-SPECIFIC, not a universal latent.** So the R²≈0.98 justifies a
dense *per-family* signal, not a claim that a single learned residual manifold generalizes zero-shot.
This is OFFLINE (schedule-mutation kernels, not an RL policy) - see [`kore/analysis`](../kore/analysis/README.md).

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

## Downstream (paradigm-v2) - the roofline potential is wired online as `η`; the named residual `ρ` is BUILT but DORMANT
The P0 `T_min` is now a live reward input, but with an important caveat: **the online potential is the
PMC-free `η`, not the R²≈0.98 named residual `ρ`.** The within-turn reward is the vendor-relative
**speedup** reward; the physics signal enters only as a potential-based-shaping (PBS) term on top.

- **Online potential is `η`, not `ρ` (the #1 open item).** The scalar potential `Φ(s)` is folded into
  the per-turn Kevin credit as `F_t = γ·Φ(s_{t+1}) − Φ(s_t)` (`kore.reward.whitebox` +
  `kore.reward.shaping`). `whitebox.phi_potential` *can* return the **named residual**
  `ρ = T_min/(T_min+N)`, `N = (stall_frac + occupancy_deficit)·T_meas` - the same decomposition that
  carries the check-(b) R²≈0.98 - **but only when rocprofv3 PMC counters are passed in.** The live
  rollout sites (`grpo._turn_phi(task, obs)` and the agentic `ToolExecutor` via
  `phi_potential(self.task, obs)`) call it **without a counter dict**, so `residual_descent_frac` takes
  the `η = T_min/T_meas` branch. Result: the live PBS potential is `η` (bounded, sane, but lower-contrast
  than `ρ`). Bringing `ρ` online - thread per-turn `KoreEnv.collect_counters` output into `phi_potential`,
  or run `reward_mode="residual"` with real per-candidate counters - is the **#1 open item**; the R²≈0.98
  result here remains an **offline** validation, not the live signal.
- **PBS invariance is approximate here, not a theorem.** The discounted shaping telescopes to a
  start-state constant `−Φ(s_0)` - the Ng-Harada-Russell result for the *vanilla expected policy
  gradient*. KORE's estimator is not that idealization: the `−w·Φ(s_t)` offset feeds GRPO's
  **std-normalized, group-relative, per-turn-as-sample** advantage (dividing by a σ that depends on the
  shifted returns), and the correct→incorrect boundary (`Φ=None` zeroes the term) leaves a small
  **bounded, action-dependent leak** of order `γ·w·Φ ≈ 0.4·0.15·1 ≈ 0.06`. So it is best described as an
  **approximate, expected-gradient-neutral state-dependent baseline** - it re-distributes existing
  terminal credit across turns (denser signal, variance reduction) without *adding* directional gradient,
  and the leak is benign but real, not "zero at any weight". It is wired identically into BOTH the
  single-process and distributed GRPO credit paths, and is live at `physics_shaping_weight=0.15` (with
  `reward_mode=speedup`, `credit_incorrect_turns=true` - `configs/grpo_14b_full.json`). Unit-tested
  (telescoping on the idealized gradient).
- **Within-turn reward is speedup** (`reward_mode=speedup`): the tier-3 correct-kernel signal
  is the high-contrast `1[correct]·log(T_base/T_cand)` vendor-relative reward; the physics term
  (currently `η`, see above) enters ONLY as the *shaping* potential, not as the base objective. The
  residual-descent reward (`kore.reward.physics`, `reward_mode=residual`) remains available and
  unit-tested, but is not the campaign's within-turn objective (and would itself be `η` online until
  per-candidate counters are threaded in).
- **Zero-shot generalization harness** (`kore.eval.generalization`): leakage-checked hold-out of whole
  operator families; offline eval of η + residual-descent on held-out families. Unit-tested.

**Honest scope (verdict unchanged).** The R²≈0.98 is the **in-sample** named-residual decomposition on
the 132-kernel P0 study (check (b)), and it does **not** transfer across families (median out-of-family
R² ≈ 0.11; see the crux above); `η` remains the **low-contrast** correlate (check (a) ρ=0.53,
check (c) WEAK) and the study verdict stays **PARTIAL**. Wiring the potential online is a *mechanism*
claim, NOT an end-to-end validation - and note the live potential is `η`, not `ρ`, so even the mechanism
does not yet carry the R²≈0.98 signal. Whether the online (`η`) PBS actually improves the policy is what
the live GRPO run will test; no end-to-end GRPO training-efficacy has been demonstrated yet.
