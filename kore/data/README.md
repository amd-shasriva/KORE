# `kore/data` — verified training-data transformations

The production data path is task pool → agentic trajectories → step-centric SFT
rows → v5 mixture (`scripts/v5_stage1_gather.py` through `v5_stage4_mixture.py`,
documented in [`DATAGEN_OVERVIEW.md`](../../DATAGEN_OVERVIEW.md)). It is designed
around a simple constraint: a teacher's text is not evidence that a kernel is
correct or fast. `KoreEnv` produces those labels by compile, oracle, and timing.

## Task pool

`kore/tasks/external.py` keeps mined tasks outside the registry. Adding them to
the registry would change its taxonomy digest and invalidate an in-flight split
manifest. Pool tasks instead use the same on-disk task ABI and the same
taxonomy, while a content-addressed index preserves their identity.

`scripts/build_task_pool.py`:

- ingests pinned KernelBook PyTorch modules and deterministic synthetic modules;
- rejects unsafe, nondeterministic, unclassifiable, too-small, duplicate, and
  held-out candidates;
- decontaminates against both KORE held-out tasks and KernelBench references;
- materializes only when requested, leaving the registry unchanged.

The benchmark reference gate is fail-closed by default: a missing KernelBench
reference is not permission to weaken the contamination check.

## Agentic data

Agentic trajectories preserve the tool observations a kernel improver actually
receives: build, correctness, timing, and profiling feedback. The data driver
uses the external pool before saturation because episode volume over a small
task set mainly makes duplicates. Measured throughput is 462–469 kept episodes
per node-hour at 100% keep rate; the scheduler driver caps itself at six nodes.

## Step-centric supervision

`step_centric.py` implements the Kernel-Smith-style conversion from trajectory
to local-improvement examples. A row is kept only when it:

- makes a broken kernel correct; or
- stays correct and improves a measured speedup by at least 5%.

It rejects broken revisions, regressions, missing measurements, no-op gains, and
speedups above the configured credibility cap. The 5% threshold is larger than
the normal timing noise: without it, the corpus would teach the model to chase
the clock rather than improve kernels.

## v5 mixture

`scripts/v5_stage3_recover.py` reuses `step_centric.py`'s `extract_steps` and
`extract_full_trajectory`, and `v5_emit.py`'s `flatten_history`, to turn agentic
trajectories into step-centric rows the same way the v3-era pipeline did.
`v5_stage4_mixture.py` then applies one gate to every source (task pool,
recovered trajectories, translated twins): nonempty messages, a token-length
cap, held-out task and family exclusion, and content-hash deduplication. No
source is exempt, which prevents a recovered file from bypassing the safeguards
applied to freshly generated trajectories.

The production model consumes the resulting SFT mixture, currently 206,000 rows
across six task shapes; see [`docs/DATASET_SPEC.md`](../../docs/DATASET_SPEC.md)
for composition and [`docs/SOURCE_PROVENANCE.md`](../../docs/SOURCE_PROVENANCE.md)
for where each row's content originates. Legacy DPO assemblers remain in the
package for the 14B research branch, but DPO is not part of the production
recipe; verifiable multi-turn execution rewards are the preferred signal.
