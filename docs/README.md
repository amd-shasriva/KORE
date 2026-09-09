# `docs/` — operational evidence

KORE's production recipe is **Qwen/Qwen3-Coder-30B-A3B-Instruct → SFT →
multi-turn RL** on AMD MI355X (`gfx950`). The model is a 30.5B-parameter MoE
with 3.3B active parameters (48 layers, 128 experts, 8 selected per token);
the active footprint is why it is viable for the on-device deliverable.

| Doc | What it establishes |
| --- | --- |
| [`DISTRIBUTED.md`](DISTRIBUTED.md) | The 30B SFT launch contract and storage constraint. |
| [`SFT_READINESS.md`](SFT_READINESS.md) | Why each SFT hyperparameter is what it is, and what the held-out evals watch for. |
| [`GRPO_READINESS.md`](GRPO_READINESS.md) | The reviewed 30B RL recipe, why it uses TRLOO, and its launch gate. |
| [`CLUSTER_OPERATIONS.md`](CLUSTER_OPERATIONS.md) | How a run is scheduled, supervised, and resumed on SPUR, and the scheduler gotchas behind each rule. |
| [`DATASET_SPEC.md`](DATASET_SPEC.md) | Data provenance, verification, task-pool and step-centric admission rules. |
| [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md) | Where the corpus content originates and how it is screened for contamination. |
| [`KORE_BENCH_BLUEPRINT.md`](KORE_BENCH_BLUEPRINT.md) | Task taxonomy and benchmark scope. |
| [`P0_RESULTS.md`](P0_RESULTS.md) | Why rooflines remain an integrity ceiling, not a validated speed predictor. |
| [`E2E_SERVING_GATE.md`](E2E_SERVING_GATE.md) | Serving-gate procedure and limits. |
| [`EVAL_RESULTS.md`](EVAL_RESULTS.md) | The failed 14B midtrain experiment that ruled out CPT on an instruct checkpoint. |
| [`FRONTIER_CLAIM_PROTOCOL.md`](FRONTIER_CLAIM_PROTOCOL.md) | Requirements for a model-vs-system claim. |

`evidence/` holds the run outputs a claim elsewhere in this tree rests on, so a
number can be traced to the artifact that produced it rather than to prose:

| Doc | What it establishes |
| --- | --- |
| [`evidence/RL_RUN_PROVENANCE.md`](evidence/RL_RUN_PROVENANCE.md) | The only multi-turn RL run to complete optimizer steps against the v5 SFT checkpoint: what ran, what it produced, and why the resolved config matches no committed file. |
| [`evidence/HARDWARE_VALIDATION.md`](evidence/HARDWARE_VALIDATION.md) | Which components had only ever run against scripted fakes, and what touching a GPU changed. |
| [`evidence/coverage_denominator.md`](evidence/coverage_denominator.md) | What `rocprofv3` actually measures for the coverage reward on real gfx950. |

The raw artifacts behind the RL document are in `evidence/rl_v5_frontier/`
(resolved config, checkpoint manifest, event log, and the four arena ledgers).

## Production decisions, with the failure modes they prevent

- **No production CPT or chat-vector merge.** No selected 30B-class Qwen offers
  a Base checkpoint: Qwen3-32B, Qwen3.6-35B-A3B, and
  Qwen3-Coder-30B-A3B are instruct-only. CPT needs a Base model, and the
  residual transfer needs a same-family Base/Instruct pair. More importantly,
  the 14B experiment in [`EVAL_RESULTS.md`](EVAL_RESULTS.md) showed that CPT on
  an instruct model destroyed instruction-following. `KORE_RECIPE=direct` is
  therefore the production default in `scripts/spur_pipeline_driver.sh`;
  `KORE_RECIPE=cpt` is a 14B-only legacy experiment.
- **No DPO production stage.** A kernel proposal can be compiled, checked, and
  timed. That execution signal is stronger than an offline preference label.
  The RL stage is multi-turn because each observation changes the next edit;
  plain GRPO is biased in that setting, which is why the project uses its
  multi-turn credit path rather than treating a transcript as one preference.
- **Data diversity precedes volume.** The registry contributes 1,501 trainable
  tasks (`registry.train_tasks()`: 1,546 registered, 45 held out — 43
  near-generalization probes plus the two `whole_family` members).
  `scripts/build_task_pool.py` adds screened external tasks without
  mutating registry manifests; the current pool is 14,859 plannable tasks and
  14,461 eligible tasks after held-out screening (13,570 external tasks; 398
  registry seeds excluded as contaminated).
- **The SFT mix teaches local improvement, not search imitation.**
  `kore/data/step_centric.py` retains correctness-preserving revisions that
  fix a kernel or improve measured speed by at least 5%; it drops regressions,
  no-ops, and suspicious speedups. The v5 build chain then deduplicates and
  re-screens every source against held-out ids and families:
  `scripts/v5_stage1_gather.py` → `scripts/v5_stage2_translate.py` →
  `scripts/v5_stage3_recover.py` → `scripts/v5_stage4_mixture.py` →
  `scripts/v5_split_eval.py` → `scripts/v5_fix_truncated.py` →
  `scripts/v5_verify.py`. `scripts/build_sft_v3_mixture.py` built the
  superseded v3 mixture and is retained only to reproduce it.

AgentKernelArena is the external AMD bar: on gfx950, published Claude Opus
means are 6.89x (`torch2hip`), 6.69x (`hip2hip`), and 2.13x
(`triton2triton`). `scripts/run_agent_kernel_arena.py` runs the benchmark's
declared compile, correctness, and performance commands in copied workspaces
and scores with its formula; `scripts/spur_aka_1node.sbatch` is the GPU-node
entrypoint. The local discovery filter reports 416 gfx950-runnable tasks from
the 426-task suite, measured with `run_agent_kernel_arena.py discover` against
AKA `b09f5eb`; the ten it drops declare a `required_arch` other than gfx950.

Kernel-Smith-235B's reported KernelBench Triton speedup is 3.70 versus 3.33 for
Claude-4.6-opus. Dr. Kernel-14B reports 31.6% single-pass and 47.8% with
test-time scaling on KernelBench L2, versus 28.6% for GPT-5 and 26.7% for
Claude-4.5-Sonnet. These are comparison bars, not KORE results.
