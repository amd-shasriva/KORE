# `docs/` - deep-dive documentation

Long-form docs that complement the per-package READMEs. Start with the [repository README](../README.md) for the overview.

| Doc | What it covers |
| --- | --- |
| [`DISTRIBUTED.md`](DISTRIBUTED.md) | FSDP sizing per model scale, the one-command `--full-ft` launch, the manual sharded launch, and per-stage full-FT status. Read this before running multi-GPU training. |
| [`DATASET_SPEC.md`](DATASET_SPEC.md) | Corpus design and the datagen record schemas (repair / ranked-group / win / agentic), the multi-capability SFT mix, and DPO pair construction - all on the **speedup objective** now shared with GRPO. |
| [`KORE_BENCH_BLUEPRINT.md`](KORE_BENCH_BLUEPRINT.md) | The kernel task taxonomy, operator families, and the benchmark release plan. |
| [`P0_RESULTS.md`](P0_RESULTS.md) | The roofline validation study: the three falsification checks, the R²≈0.98 residual-decomposition result, the physics reward (now wired ONLINE as a paradigm-v2 PBS potential in GRPO), and the cross-family transfer crux. Canonical write-up of "is the paradigm real?". |

**Objective alignment (paradigm-v2).** SFT/DPO and GRPO now optimize the **same** objective. The SFT
mix and DPO pairs are assembled on the vendor-relative **speedup** signal (`faster-correct >
slower-correct > incorrect > non-compiling`), and GRPO's within-turn reward is that same speedup reward
(`reward_mode=speedup`). The physics named-residual `ρ` enters GRPO ONLY as a *policy-invariant*
potential-based-shaping term (`physics_shaping_weight`), which by construction cannot change the optimal
policy - so it densifies credit without re-introducing the prior SFT/DPO-vs-GRPO objective mismatch.
This is a training-objective alignment; the **datagen generation itself is unchanged**.

Related: the [`Kore-prelim-analysis`](../../Kore-prelim-analysis/) sibling repo is the self-contained P0 study (data + figures + reproduce steps); the [`papers/`](../../papers/) directory in the umbrella repo has the annotated literature the methods draw on.
