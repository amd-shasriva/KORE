# P0 roofline/residual validation — controlled reanalysis

Verdict: **INTEGRITY_ONLY.** The stored gfx950 measurements support using a
conservative roofline as an integrity bound. They do not support eta, the named
counter residual, or a profile score as a reward-shaping signal. Physics and
counter shaping are therefore disabled.

This is a CPU-only reanalysis of the existing 132-point
`data/p0_study_final.json`; it is not a new GPU run. The source report predates
model fingerprinting and identifies only `gfx950`, not the exact MI350X/MI355X
SKU and runtime calibration identity, so it cannot authorize online shaping.

## Why the earlier R²=.978 conclusion changed

The former primary regression used

```
residual_ms ~ stall_fraction * T_candidate
            + occupancy_deficit * T_candidate
```

The target and both regressors contain the same `T_candidate` scale. The
controlled analysis treats the resulting R² as a leakage diagnostic:

- Named raw in-sample R²: **0.97835**.
- `T_candidate`-only raw in-sample R²: **0.99713**.
- Denominator-preserving within-task permutation null median: **0.98083**.
- Raw null p-value: **0.7203**.

The named regressors neither beat the simple runtime baseline nor the null.
Consequently R²=.978 is not evidence for a causal or reward-useful residual
decomposition.

## Preregistered primary results

The primary target is the unitless normalized gap
`(T_candidate - T_min) / T_candidate`. Splits hold out whole task clusters;
confidence intervals resample task clusters; permutation controls keep every
candidate runtime fixed; primary and family tests use Benjamini–Hochberg FDR
correction.

Check (a), roofline attainment versus vendor speedup:

- Spearman eta/speedup: **0.5290**.
- Spearman `1/T_candidate`/speedup baseline: **0.7274**.
- Increment over the runtime-only baseline: **-0.1984**, task-bootstrap 95% CI
  **[-0.4862, 0.0474]**.
- Held-task log-model R²: eta **-0.0630**, runtime-only **0.5582**.
- Adjusted p-value: **0.9860**.
- Result: **FAIL**. The previously reported correlation is consistent with the
  shared candidate-time denominator and does not show incremental roofline value.

Check (b), counter residual:

- Normalized in-sample R²: **0.3140**.
- Normalized held-task-cluster R²: **-0.4582**.
- `T_candidate`-only normalized held-task R²: **-0.1217**.
- Increment over that baseline: **-0.3365**.
- Task-bootstrap 95% CI for held-task R²: **[-0.4880, 0.6078]**.
- Denominator-preserving null p-value and adjusted p-value: **0.9860**.
- Result: **FAIL**.

Normalized leave-family-out R² is also non-transferable: activation **-1.91**,
attention **0.396**, GEMM **0.171**, MoE **-348.35**, norm **-5.13**,
positional **-26.04**, quant **-5.05**, and reduction **-1.84**.

Check (c), collection-order trajectory control:

- Dominant-term decrease on flat-runtime pairs: **0.500** over 38 pairs.
- Task-bootstrap 95% CI: **[0.361, 0.629]**.
- Adjusted p-value: **0.9405**.
- Result: **FAIL**.

## Family shaping decision

No measured family passes the preregistered requirements. Activation, GEMM, MoE,
positional, quant, and reduction have fewer than three independent task
clusters. Attention has negative held-task R². Norm has a positive point
estimate, but its task-bootstrap interval crosses zero. All eight families are
unavailable for empirical shaping.

`configs/grpo_14b_full.json` therefore pins `physics_shaping_weight=0`,
`physics_live_counters=false`, and an explicit MI350X datasheet model
fingerprint. Counter data can still be logged for diagnosis. A future P0 report
may enable a family only when its own held-out evidence passes and its evidence
and physical-model fingerprints match the runtime configuration.

## Authoritative physical model

`kore.analysis.roofline` is the single implementation used by offline analysis,
online reward, integrity checks, reports, and GRPO. A `PhysicalModel` names an
exact architecture/SKU, encodes FLOP/s and byte/s units, records runtime
calibration metadata, validates all values, and has a stable SHA-256
fingerprint. There is no import-time active GPU or `KORE_PEAK_*` global state.

`kore.analysis.rooflines` is a compatibility facade over that implementation.
Unknown dtypes and unsupported operation models return unavailable. In
particular, attention, MoE, top-k, backward operations, unknown fused
operations, and low-precision formats whose scale/metadata traffic is not
modeled do not receive fabricated attainment.

Integrity uses a conservative vendor-upper-bound model. It always permits the
mandatory compute floor; it includes the HBM floor only when the observation
explicitly records verified cold-cache timing. Empirical calibration is never
treated as a hard physical maximum.

## Hardware work still required

Before any shaping re-enable or new GPU claim:

1. Recalibrate on an identified SKU with timestamp, device identity, firmware,
   ROCm/driver, library versions, clock/power state, and thermal state recorded
   in `kore.runtime-calibration.v1`.
2. Repeat HBM calibration across working-set sizes with verified cache eviction
   and audited byte accounting.
3. Measure sustained BF16, FP16, FP8, FP32, INT8, FP6, and FP4 paths separately;
   include output and scale/metadata traffic. The stored FP8 peak was not
   measured.
4. Validate each requested rocprofv3 counter and derived metric on that exact
   runtime, including its unit and dispatch aggregation. Do not combine
   quad-cycles, instructions, and MOPS.
5. Collect at least three independent task clusters per candidate family, plus
   held-out tasks and a leave-family-out replication. Re-run 1000 task-cluster
   bootstraps and 1000 denominator-preserving permutations.
6. Measure launch/dispatch overhead separately before extending the model to
   tiny shapes.

## Reproduce the controlled analysis

```bash
python -m kore.analysis.p0_sol \
  --reanalyze data/p0_study_final.json \
  --permutations 1000 --bootstrap 1000 \
  --out runs/p0_study_final_controlled.json

python -m kore.analysis.residual_transfer \
  --report data/p0_study_final.json \
  --permutations 1000 --bootstrap 1000
```
