# `kore/analysis` — physical model and P0 validation

`roofline.py` is the authoritative implementation. It provides validated
`HardwareSpec`, `PhysicalModel`, `WorkEstimate`, counter units, model
fingerprints, roofline evaluation, and integrity diagnostics. It never detects a
GPU or selects mutable peaks at import time.

`rooflines.py` is a deprecated compatibility facade. Its operator results and
FLOP/byte estimates delegate to `roofline.py`; it does not carry independent
physics.

## Availability rules

Callers select an exact SKU and optionally a
`kore.runtime-calibration.v1` file:

```python
from kore.analysis.roofline import (
    estimate_work, evaluate_roofline, make_physical_model,
)

model = make_physical_model(
    "mi350x",
    calibration=None,  # explicit vendor-datasheet model
    expected_fingerprint="sha256:...",
)
work = estimate_work("gemm", {"M": 4096, "N": 4096, "K": 4096}, "bf16")
result = evaluate_roofline(work, model) if work else None
```

Unknown dtypes, unsupported hardware paths, and operations without defensible
mandatory work return `None`. There is no generic operation fallback. Attention,
MoE, top-k, backward operations, unknown fusions, and low-precision formats with
unmodeled scale traffic are currently unavailable.

A calibration must identify architecture, exact SKU, calibration id, runtime
stack, HBM byte/s, and calibrated FLOP/s by dtype. Legacy `KORE_PEAK_*`
environment overrides are not accepted because they cannot be fingerprinted or
reproduced.

## Integrity versus shaping

The roofline has two distinct uses:

- Integrity rejection/pruning uses vendor upper bounds (or a higher measured
  bound), producing the smallest conservative runtime floor. The HBM component
  requires explicit cold-cache provenance; mandatory compute does not.
- Empirical reward shaping requires a matching, fingerprinted P0 evidence
  artifact and a family-specific held-out PASS. Diagnostics alone never enable
  shaping.

The stored P0 study passes neither the normalized held-task test nor any family
gate, so the current GRPO configuration is integrity-only. See
`docs/P0_RESULTS.md`.

## Leakage-controlled P0

`p0_sol.py` preregisters:

1. eta versus vendor speedup after a `T_candidate`-only baseline and a
   denominator-preserving numerator permutation;
2. normalized residual prediction on held-out task clusters, raw and normalized
   simple baselines, within-task feature permutations, leave-family-out scores,
   and task-cluster bootstrap intervals;
3. collection-order trajectory tests, avoiding outcome-based eta sorting;
4. Benjamini–Hochberg correction over primary and family hypotheses.

`residual_transfer.py` is now a report-only compatibility wrapper around the
canonical `p0_sol.reanalyze_report`; it has no duplicate OLS implementation.

CPU-only reproduction:

```bash
python -m kore.analysis.p0_sol \
  --reanalyze data/p0_study_final.json \
  --permutations 1000 --bootstrap 1000 \
  --out runs/p0_study_final_controlled.json

python -m kore.analysis.residual_transfer \
  --report data/p0_study_final.json \
  --permutations 1000 --bootstrap 1000
```

The existing measurement report is legacy-unfingerprinted and cannot authorize
shaping even if a statistic were to pass. A new GPU study must use an exact SKU
and fingerprint-safe runtime calibration.
