# `docs/` — deep-dive documentation

Long-form docs that complement the per-package READMEs. Start with the [repository README](../README.md) for the overview.

| Doc | What it covers |
| --- | --- |
| [`DISTRIBUTED.md`](DISTRIBUTED.md) | FSDP sizing per model scale, the one-command `--full-ft` launch, the manual sharded launch, and per-stage full-FT status. Read this before running multi-GPU training. |
| [`DATASET_SPEC.md`](DATASET_SPEC.md) | Corpus design and the datagen record schemas (repair / ranked-group / win / agentic), the multi-capability SFT mix, and DPO pair construction. |
| [`KORE_BENCH_BLUEPRINT.md`](KORE_BENCH_BLUEPRINT.md) | The kernel task taxonomy, operator families, and the benchmark release plan. |
| [`P0_RESULTS.md`](P0_RESULTS.md) | The roofline validation study: the three falsification checks, the R²≈0.98 residual-decomposition result, the physics reward, and the cross-family transfer crux. Canonical write-up of "is the paradigm real?". |

Related: the [`Kore-prelim-analysis`](../../Kore-prelim-analysis/) sibling repo is the self-contained P0 study (data + figures + reproduce steps); the [`papers/`](../../papers/) directory in the umbrella repo has the annotated literature the methods draw on.
