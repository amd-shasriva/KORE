# P0 roofline / Speed-of-Light validation — native gfx950 (MI350-class / CDNA4)

**Verdict: PARTIAL.** Check (b), the residual decomposition, is a decisive PASS and the load-bearing result; check (a) passes against the AITER production baseline; check (c) is weak, as expected before any RL. All results here are offline (schedule-mutation kernels, not an RL policy).

## Node & stack

- **Host:** 8× **gfx950** (AMD Instinct MI350-class, CDNA4), ROCm 7.2.3, `rocprofv3`. All GPU measurement runs on one device (datagen runs on a separate node).
- **Main stack:** `torch 2.10.0+rocm7.0` + `pytorch-triton-rocm 3.5.1` (native gfx950) — used for the roofline model, PMC, candidate kernels, and peak calibration.
- **AITER baseline stack (isolated):** AITER's kernels require `triton ≥ 3.6`, which the main stack does not ship, so the production baseline runs from a separate venv (local `torch 2.10.0+rocm7.0` + `triton 3.6.0` + a separate AITER checkout) so `aiter.ops.*` CK kernels JIT-compile on gfx950. This never modifies the main venv or the shared runtime.

## Roofline model

`T_min = max(W_flops/P_peak, Q_bytes/B_peak)`, `η = T_min/T_measured ∈ (0,1]` (SOL attainment). Peaks are overridable via `KORE_PEAK_{BF16,FP8,HBM_BW}`.

## Peak calibration (measured achievable, not datasheet)

On-device microbenchmarks (`kore.analysis.calibrate_peaks`, batched event timing):

| peak | datasheet | measured achievable | attained | method |
| --- | --- | --- | --- | --- |
| HBM bandwidth | 8.0 TB/s | **4.60 TB/s** | 57% | STREAM triad `a = b + q·c` (3·N·4 B traffic) |
| bf16 matrix | 2.5 PF/s | **1.27 PF/s** | 51% | 8192³ square matmul (`2N³` FLOPs, sustained) |
| fp8 matrix | 5.0 PF/s | *(datasheet)* | – | `torch._scaled_mm` unavailable on this stack |

Measured peaks are applied via `KORE_PEAK_HBM_BW=4.599e12`, `KORE_PEAK_BF16=1.273e15` (fp8 keeps the datasheet value). Using the achievable peak makes `η` a defensible fraction-of-attainable-SOL; because every kernel of a dtype divides by the same peak, this only rescales absolute `η` and leaves the tested relationships unchanged (check (b) R² is 0.978 calibrated vs 0.99 uncalibrated; both PASS). Details in `data/calibration.json`.

## Final study (3 representative shapes × 3 reseeds, PMC, bootstrap CIs)

`data/p0_study_final.json`: every operator is measured at 3 representative shapes (primary + validation_0 + validation_1; the tiny `minimal` correctness shape is excluded as launch-overhead-bound), each timing reseeded 3× (median-of-medians, L2-flushed cold-cache), with rocprofv3 PMC and 1000× bootstrap 95% CIs. 132 correct kernel×shape points.

```
(a) η predicts speedup     : ρ = 0.529    n=114   95% CI [0.346, 0.701]   -> PASS
(b) residual decomposition : R² = 0.9783  n=132   95% CI [0.967, 0.989]   -> PASS  (load-bearing)
(c) monotone-in-valley     : frac = 0.525 pairs=59 95% CI [0.393, 0.646]  -> WEAK  (expected pre-RL)
VERDICT: PARTIAL
```

**Check (b) is the headline.** The runtime residual `(T_measured − T_min)` reconstructs from counter-derived named terms — memory-stall time (`MemUnitStalled`) and occupancy-deficit time (`1 − OccupancyPercent`) — with **R² = 0.978 (95% CI [0.967, 0.989])** across 132 kernels. The CI lower bound stays well above 0.9: the named gradient KORE descends is real and measurable.

**The R² ≈ 0.98 is in-sample and is operator-specific.** A leave-one-family-out experiment (`kore.analysis.residual_transfer`) refits the named-term → residual map on all families but one and predicts the held-out family: pooled in-sample R² = 0.978, but median out-of-family R² = 0.107 (raw) / negative (normalized), and families are separable in residual space. The residual is therefore a **dense per-family** signal, not a single universal latent that transfers zero-shot — which is why KORE trains on the per-family residual rather than assuming cross-family transfer.

**Check (a) passes against the AITER production baseline** (`ρ = 0.529`, CI [0.346, 0.701]): kernels nearer the roofline attain higher speedup versus the production vendor.

## Check (a) baselines — AITER production kernels

The reference for each operator is the real production kernel, tagged per operator in the JSON:

| operator | baseline | median speedup (seed→best vs vendor) |
| --- | --- | --- |
| rmsnorm / layernorm / fused_add_rmsnorm | **AITER CK** (`rms_norm`, `layer_norm`, `fused_add_rms_norm_cu`) | 0.71 / 0.98 / 0.86× |
| silu_and_mul / rope | **AITER CK** (`silu_and_mul`, `rope_fwd`) | 0.64 / 0.33× |
| flash_attn_decode / prefill / paged_attn_decode | **AITER** FMHA / ROCm paged attn | 1.32 / 0.07 / 0.15× |
| fused_moe_silu / topk_softmax | **AITER** `fused_moe` / `topk_softmax` | 0.04 / 0.29× |
| gemm_bf16 | **hipBLASLt** (`torch.matmul`) | 0.50× |
| softmax / gelu_tanh | framework (torch — no standalone AITER op) | 0.51 / 0.89× |
| gemm_fp8_a8w8 / quant_fp8_pertoken | η-only (fp8 vendor path not built on this stack) | – |

Baseline composition (operators with a vendor speedup): 10 AITER-vendor, 1 hipBLASLt, 2 framework. Every AITER baseline passes the same torch-fp32 correctness oracle used for candidates.

## PMC (gfx950)

gfx950/CDNA4 renamed the raw counters, so the original `SQ_*` list collected nothing. We use rocprofv3 derived metrics (`OccupancyPercent`, `MemUnitStalled`, `MfmaUtil`, `GRBM_GUI_ACTIVE`), parse the long-format `*_counter_collection.csv`, and select the longest-running compute kernel.

## Limitations

- **`minimal`-shape regime:** on tiny correctness shapes (e.g. M=64, N=512) every kernel is launch/overhead-bound (`η < 2%`) and the roofline does not model launch cost; check (a) is therefore reported on representative shapes only.
- **fp8 peak** is datasheet (no `_scaled_mm`); the two fp8 operators are η-only.
- **Check (c)** trajectories are schedule-mutations of the seed, not an RL policy — they show the residual moves with schedule, not that a policy drives it monotonically. WEAK is the expected pre-RL reading.

## Reproduce

```bash
# calibrate peaks (main stack)
python -m kore.analysis.calibrate_peaks --out data/calibration.json

# AITER production baseline needs triton>=3.6 in a SEPARATE venv (never the datagen-shared one):
#   python3.10 -m venv ~/kore-venv-aiter && \
#   ~/kore-venv-aiter/bin/pip install <local torch 2.10+rocm7.0 wheel> triton==3.6 pandas einops psutil ninja pybind11 && \
#   pip install -e <separate aiter checkout>

# final study (calibrated peaks + AITER baselines + PMC + CIs):
KORE_PEAK_HBM_BW=4.599e12 KORE_PEAK_BF16=1.273e15 PYTHONPATH=<aiter2> \
  python -m kore.analysis.p0_sol --shapes-per-task 3 --reseeds 3 --bootstrap 1000 \
    --warmup 5 --iters 20 --max-kernels-per-task 3 --out data/p0_study_final.json
python -m kore.analysis.plots --report data/p0_study_final.json --out figures/
```

## How the physics enters training

The validated `T_min` is a live reward input. The within-turn reward is the vendor-relative **speedup** (`reward_mode=speedup`); the physics enters GRPO as a potential-based-shaping term on top:

- **Shaping potential.** The scalar potential `Φ(s)` is folded into per-turn credit as `F_t = γ·Φ(s_{t+1}) − Φ(s_t)` (`kore.reward.whitebox`, `kore.reward.shaping`). Online, `Φ = η = T_min/T_measured` (PMC-free, bounded). Its counter-grounded refinement, the named residual `ρ = T_min/(T_min+N)` with `N = (stall_frac + occupancy_deficit)·T_measured` — the same decomposition that carries the check-(b) R² ≈ 0.98 — is the validated target `Φ` approximates; supplying per-turn rocprofv3 counters (or `reward_mode="residual"` with per-candidate counters) makes `ρ` the live potential. The shaping is a state-dependent baseline that redistributes terminal credit across turns (denser signal, variance reduction) without changing the ranking of returns. It is wired identically into the single-process and distributed GRPO paths and is live at `physics_shaping_weight=0.15` (`configs/grpo_14b_full.json`).
- **Residual-descent reward.** `reward_mode="residual"` (`kore.reward.physics`) is available and unit-tested as an alternative within-turn objective.
- **Zero-shot generalization harness** (`kore.eval.generalization`): leakage-checked hold-out of whole operator families, with offline evaluation of η and residual-descent on the held-out families.

The R² ≈ 0.98 result is an offline validation of the physics signal; whether the online shaping improves the policy is measured by the GRPO run and its held-out evaluation.
