# `docs/` — deep-dive documentation

Long-form documentation that complements the per-package READMEs. Start with the [repository README](../README.md) for the overview.

| Doc | What it covers |
| --- | --- |
| [`DISTRIBUTED.md`](DISTRIBUTED.md) | FSDP sizing per model scale, the one-command `--full-ft` launch, the manual sharded launch, and per-stage full-FT configuration. Read this before running multi-GPU training. |
| [`DATASET_SPEC.md`](DATASET_SPEC.md) | Corpus design and the datagen record schemas (repair / ranked-group / win / agentic), the multi-capability SFT mix, and DPO pair construction — all on the vendor-relative speedup objective shared with GRPO. |
| [`KORE_BENCH_BLUEPRINT.md`](KORE_BENCH_BLUEPRINT.md) | The kernel task taxonomy, operator families, and the benchmark release plan. |
| [`P0_RESULTS.md`](P0_RESULTS.md) | The roofline validation study: the falsification checks, the R² ≈ 0.98 residual-decomposition result, the physics reward as a shaping potential, and cross-family transfer. |

## Objective alignment

SFT, DPO, and GRPO optimize the **same** objective. The SFT mix and DPO pairs are assembled on the vendor-relative **speedup** signal (`faster-correct > slower-correct > incorrect > non-compiling`), and GRPO's within-turn reward is that same speedup reward (`reward_mode=speedup`). The physics enters GRPO as a potential-based-shaping term (`physics_shaping_weight`) with potential `Φ = η = T_min/T_measured` online; its counter-grounded refinement `ρ` (R² ≈ 0.98 offline; see [`P0_RESULTS.md`](P0_RESULTS.md)) is the validated target that the shaping approximates. The shaping offset is fed into GRPO's std-normalized group-relative per-turn advantage as an expected-gradient-neutral, state-dependent baseline that densifies credit toward the roofline without changing the ranking of returns. This is a training-objective alignment; datagen generation itself is unchanged.

Related: the [`Kore-prelim-analysis`](../../Kore-prelim-analysis/) sibling repo is the self-contained P0 study (data + figures + reproduce steps); the [`papers/`](../../papers/) directory in the umbrella repo holds the annotated literature the methods draw on.
