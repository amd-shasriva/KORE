# `scripts/`

The active production flow is data first, then 30B SFT, then multi-turn RL.
The scheduler is the execution boundary; use the SPUR controller address before
every `squeue` or `sbatch` call.

```bash
export SPUR_CONTROLLER_ADDR="http://crs-m2m-cpu-spur-005:6817"
sbatch scripts/spur_build_task_pool.sbatch
bash scripts/spur_data_driver.sh
```

`spur_data_driver.sh` intentionally stops before training and preserves
`runs/DATA_NOT_FINAL`. It waits for task-pool construction because flooding the
old registry would produce redundant trajectories; it also respects the
six-node QoS limit.

## Active data and evaluation entrypoints

| Script | Purpose |
| --- | --- |
| `build_task_pool.py` | Mine, screen, decontaminate, and index external tasks without mutating the registry. |
| `spur_build_task_pool.sbatch` | CPU scheduler entrypoint for the pool build. |
| `spur_data_driver.sh` | Drive task pool then saturated agentic datagen; never starts training. |
| `run_agent_kernel_arena.py` | Run AgentKernelArena in copied workspaces under its native scoring formula. |
| `spur_aka_1node.sbatch` | GPU-node entrypoint for AgentKernelArena. |
| `run_kernelbench_amd.py` | Materialize and score KernelBench tasks through KORE. |
| `launch_distributed.sh` | Launch one explicit FSDP stage inside an allocation. |
| `spur_pipeline_driver.sh` | Scheduler driver with `KORE_RECIPE=direct` as its production default. |

The model-side benchmark command defaults to
`Qwen/Qwen3-Coder-30B-A3B-Instruct`; AgentKernelArena discovery filters to
gfx950-compatible tasks before any GPU time is spent.

## Building the v5 SFT mixture

Seven stages, in this order. Each reads the previous stage's default output, so
running one out of order silently builds against a stale input:

| Order | Script | Writes |
| ---: | --- | --- |
| 1 | `v5_stage1_gather.py` | `runs/v5_build/stage1.pkl` — every mined record, deduplicated and thinned |
| 2 | `v5_stage2_translate.py` | the translate slice — verified twins re-posed as the shapes the arena asks |
| 3 | `v5_stage3_recover.py` | `runs/v5_build/` — ranked groups recovered as generation supervision |
| 4 | `v5_stage4_mixture.py` | `data/v5_sft.jsonl` — the mixture, with replay budgeted in tokens |
| 5 | `v5_split_eval.py` | `data/v5_eval.jsonl`, and rewrites the mixture without those rows |
| 6 | `v5_fix_truncated.py` | both files — drops cut-off targets and backfills the eval math group |
| 7 | `v5_verify.py` | nothing; runs the twelve correctness gates and a token-length pass |

`scripts/v5_verify_all.sh` runs the round-trip, the disjointness check, and both
verification passes together as the pre-launch gate.

`build_sft_v3_mixture.py` built the superseded v3 mixture and
`build_sft_v2_mixture.py` the one before it. Both are retained to reproduce
those artifacts; neither is on the path to the shipped 30B config.

## Legacy scripts

The operations registry is the lifecycle authority:
[`operations_registry.json`](operations_registry.json). Scripts marked
`lifecycle: deprecated` reject production execution unless
`KORE_ALLOW_DEPRECATED_DEV=1` is explicitly set. They are retained for
reproducible 14B investigations and tests, not as production alternatives.

In particular, these are not current stages: `spur_midtrain_*`,
`spur_dpo_1node.sbatch`, the 14B conductor/tmux wrappers, and the residual
builder. The old path needed a Base model for CPT and a Base/Instruct pair for
the residual merge; those prerequisites do not exist for the instruct-only 30B
target. DPO is also superseded by execution-rewarded multi-turn RL.

## Production SFT

The existing SFT sbatch wrapper still has 14B defaults. Invoke it explicitly
until it is migrated:

```bash
export SPUR_CONTROLLER_ADDR="http://crs-m2m-cpu-spur-005:6817"
sbatch scripts/spur_sft_1node.sbatch \
  configs/sft_coder30b_a3b.json - runs/sft_coder30b_a3b
```

`-` preserves the instruct model named in the config. See
[`docs/DISTRIBUTED.md`](../docs/DISTRIBUTED.md) for the exact configuration and
the single-checkpoint storage requirement.
