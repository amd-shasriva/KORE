# `kore/reward` — verified reward and physics integrity

The active objective is the verified vendor-speedup reward. The reward ladder is
strict:

```
hack < compile failure < incorrect < correct
```

`validate_reward_config` enforces the ladder with ordinary runtime exceptions,
including adverse format terms. The checks cannot disappear under `python -O`.
`RewardResult` rejects non-finite rewards and speedups.

## Physical model

Offline analysis, integrity checks, reward diagnostics, reports, and GRPO use the
same fingerprinted `kore.analysis.roofline.PhysicalModel`. Runtime configuration
selects:

- `physics_sku`
- optional `physics_calibration_path`
- optional pinned `physics_model_fingerprint`

There is no import-time active architecture and no process-global
`KORE_PEAK_*` calibration.

Unsupported operations, dtypes, physical paths, and counter sets are
unavailable. They do not receive zero or fabricated attainment.

## Integrity is not shaping

`roofline_gate` is an integrity control. It uses conservative vendor upper
bounds, not an achievable calibration. Mandatory compute can always contribute
to the floor; HBM traffic contributes only when
`Observation.cold_cache_verified` is true. A super-floor observation is rejected
to the hack tier. Missing physical information fails open.

Counter and roofline metrics remain available for logging and diagnosis, but
they become reward terms only with family-specific held-out evidence:

- `physics_shaping_evidence_path`
- `physics_shaping_evidence_fingerprint`
- matching physical-model fingerprint
- preregistered P0 family PASS

The current controlled P0 reanalysis has no passing family. Therefore
`configs/grpo_14b_full.json` sets `physics_shaping_weight=0` and
`physics_live_counters=false`. `reward_mode="residual"` falls back to the normal
speedup reward without passing evidence.

## Counter units

`profile_reward.py` and `whitebox.py` use the counter schema from the canonical
model:

- `SQ_WAIT_INST_*` values are quad-cycles.
- `SQ_INSTS_*` values are instruction counts.
- `*_MFMA_MOPS_*` values are 512-FMA work units.
- derived occupancy/stall/utilization values are percentages.

Quad-cycles are never added to or divided by instructions. MOPS are converted to
FLOPs only and are never included in an instruction count. MFMA issue work is not
added to `SQ_INSTS_VALU`, which already contains matrix instructions.

`stall_fraction` consequently requires the validated `MemUnitStalled`
percentage; raw wait plus instruction counters are insufficient. Exact HBM bytes
require both aggregate request counts and 32B/64B split counters.

## Evidence-backed potential

`FamilyShapingEvidence` stores the normalized held-out model, task/point counts,
task-bootstrap interval, adjusted p-value, physical-model fingerprint, report
fingerprint, and three coefficients. It recomputes the preregistered pass
criteria when loaded.

`phi_potential` returns `None` unless matching evidence passes and both validated
counter features are present. Defined potentials and PBS inputs must be finite
and lie in `[0, 1]`; gamma must be in `[0, 1]`; weights and rewards must be finite.
Undefined transitions are zero-contribution boundaries.

The potential-based shaping identity remains useful as a credit-allocation
mechanism, but KORE does not claim exact invariance under std-normalized,
group-relative GRPO. The empirical gate is required independently of that
identity.

## Main interfaces

```python
compute_reward(obs, source="", dtype="fp32", cfg=CONFIG)

compute_kernel_reward(
    obs, source, task,
    mode="speedup",
    model=physical_model,
    physics_config=grpo_config,
)

roofline_ceiling_violation_from_obs(
    task, obs, model=physical_model,
)

phi_potential(
    task, obs, counters,
    model=physical_model,
    evidence=family_evidence,
)
```

See `docs/P0_RESULTS.md` for the controlled scientific conclusion and the
hardware measurements required before shaping can be reconsidered.
