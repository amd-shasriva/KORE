# `docs/` — deep-dive documentation

Long-form documentation that complements the per-package READMEs. Start with the [repository README](../README.md) for the overview.

| Doc | What it covers |
| --- | --- |
| [`DISTRIBUTED.md`](DISTRIBUTED.md) | FSDP sizing per model scale, the one-command `--full-ft` launch, the manual sharded launch, and per-stage full-FT configuration. Read this before running multi-GPU training. |
| [`DATASET_SPEC.md`](DATASET_SPEC.md) | Corpus design and the datagen record schemas (repair / ranked-group / win / agentic), the multi-capability SFT mix, and DPO pair construction — all on the vendor-relative speedup objective shared with GRPO. |
| [`KORE_BENCH_BLUEPRINT.md`](KORE_BENCH_BLUEPRINT.md) | The kernel task taxonomy, operator families, and the benchmark release plan. |
| [`P0_RESULTS.md`](P0_RESULTS.md) | The roofline validation study. Verdict `INTEGRITY_ONLY`: all three preregistered checks FAIL, so the roofline is usable as an integrity ceiling and a shaping potential but is **not** a validated predictor of speedup. Read this before quoting any physics number. |
| [`SFT_READINESS.md`](SFT_READINESS.md) | Stage-1 SFT launch readiness: the measured 14B lifecycle (steps, checkpoint, resume), the three fixed blockers, and the launch command. |
| [`E2E_SERVING_GATE.md`](E2E_SERVING_GATE.md) | How to provision a real SGLang/vLLM ROCm backend on gfx950, point the serving gate at it, and what it measured. |
| [`FRONTIER_CLAIM_PROTOCOL.md`](FRONTIER_CLAIM_PROTOCOL.md) | The offline, preregistered adjudication layer for a frontier model-vs-system claim. |
| [`GRPO_MIN_TRUSTWORTHY.md`](GRPO_MIN_TRUSTWORTHY.md) | The fail-closed `grpo_32b_min_trustworthy` profile contract — a semantic safety profile, not a 32B sizing claim. |

Midtrain builders should also read
[`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md) and use the adjacent
`source_metadata.schema.json` contract before rebuilding a production corpus.

## Objective alignment

SFT, DPO, and GRPO optimize the **same** objective. The SFT mix and DPO pairs are assembled on the vendor-relative **speedup** signal (`faster-correct > slower-correct > incorrect > non-compiling`), and GRPO's within-turn reward is that same speedup reward (`reward_mode=speedup`).

The physics enters GRPO only as a potential-based-shaping term (`physics_shaping_weight`) with potential `Φ = η = T_min/T_measured` online. The shaping offset is fed into GRPO's std-normalized group-relative per-turn advantage as a state-dependent baseline that densifies credit toward the roofline without changing the ranking of returns — which is why it is safe *regardless* of whether `Φ` predicts anything.

> **`ρ` is not a validated target.** An earlier revision of this page described the counter-grounded refinement `ρ` as "validated at R² ≈ 0.98 offline". That figure is a shared-denominator artifact and does not survive the repository's own v2 controls: a `T_candidate`-only predictor scores higher (0.997), and on the preregistered normalized target over held-out task clusters the named model scores −0.458. The current adjudicator returns `INTEGRITY_ONLY` and authorizes empirical shaping for **no** operator family. See [`P0_RESULTS.md`](P0_RESULTS.md). Treat `η` as a variance-reduction heuristic, not a hardware signal the shaping "approximates".

This is a training-objective alignment; datagen generation itself is unchanged.

Related: the [`Kore-prelim-analysis`](../../Kore-prelim-analysis/) sibling repo is the self-contained P0 study (data + figures + reproduce steps); the [`papers/`](../../papers/) directory in the umbrella repo holds the annotated literature the methods draw on.
