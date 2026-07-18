# `docs/` - deep-dive documentation

Long-form docs that complement the per-package READMEs. Start with the [repository README](../README.md) for the overview.

| Doc | What it covers |
| --- | --- |
| [`DISTRIBUTED.md`](DISTRIBUTED.md) | FSDP sizing per model scale, the one-command `--full-ft` launch, the manual sharded launch, and per-stage full-FT status. Read this before running multi-GPU training. |
| [`DATASET_SPEC.md`](DATASET_SPEC.md) | Corpus design and the datagen record schemas (repair / ranked-group / win / agentic), the multi-capability SFT mix, and DPO pair construction - all on the **speedup objective** now shared with GRPO. |
| [`KORE_BENCH_BLUEPRINT.md`](KORE_BENCH_BLUEPRINT.md) | The kernel task taxonomy, operator families, and the benchmark release plan. |
| [`P0_RESULTS.md`](P0_RESULTS.md) | The roofline validation study: the three falsification checks, the R²≈0.98 residual-decomposition result, the physics reward (repositioned in paradigm-v2 as a PBS shaping potential in GRPO — but **online it is the PMC-free `η`**, since the validated `ρ` needs per-rollout PMC counters that are not yet threaded), and the cross-family transfer crux. Canonical write-up of "is the paradigm real?". |

**Objective alignment (paradigm-v2).** SFT/DPO and GRPO optimize the **same** objective. The SFT
mix and DPO pairs are assembled on the vendor-relative **speedup** signal (`faster-correct >
slower-correct > incorrect > non-compiling`), and GRPO's within-turn reward is that same speedup reward
(`reward_mode=speedup`). The physics enters GRPO ONLY as a potential-based-shaping term
(`physics_shaping_weight`) — and, stated honestly, **online it is the PMC-free `η = T_min/T_measured`,
not the validated named-residual `ρ`**: the rollout call sites invoke `phi_potential(task, obs)` without
a counter dict, so `ρ` (R²≈0.98 offline) is a target the online reward does not yet consume. Threading
per-rollout rocprofv3 counters to bring `ρ` online is the #1 open item. Its policy-invariance is
**approximate, not a theorem here**: the shaping offset is fed into GRPO's *std-normalized
group-relative per-turn advantage*, and the correct→incorrect boundary leaves a small bounded
action-dependent leak (≤~0.06). Read it as an expected-gradient-neutral **state-dependent baseline**
that densifies credit without re-introducing the prior SFT/DPO-vs-GRPO objective mismatch.
This is a training-objective alignment; the **datagen generation itself is unchanged**.

> **Live-run status.** The campaign is currently in the `midtrain` stage; the chain is
> `midtrain → build → sft → dpo → grpo → soup → eval`, so GRPO (where the paradigm-v2 levers actually
> activate) is still days away. It is a **non-curriculum** run, so at GRPO start it renders a fresh
> `data/full14b/launch/grpo.json` from `configs/grpo_14b_full.json`; the stale
> `data/full14b/launch/grpo_phase1_correctness.json` (Jul-10 curriculum artifact) is **not** this run's
> config. For an honest novelty/frontier and data-efficiency assessment, see the root
> [README](../README.md#novelty--frontier-status-honestly).

Related: the [`Kore-prelim-analysis`](../../Kore-prelim-analysis/) sibling repo is the self-contained P0 study (data + figures + reproduce steps); the [`papers/`](../../papers/) directory in the umbrella repo has the annotated literature the methods draw on.
