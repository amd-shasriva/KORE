# `docs/` — operational evidence

KORE's production recipe is **Qwen/Qwen3-Coder-30B-A3B-Instruct → SFT →
multi-turn RL** on AMD MI355X (`gfx950`). The model is a 30.5B-parameter MoE
with 3.3B active parameters (48 layers, 128 experts, 8 selected per token);
the active footprint is why it is viable for the on-device deliverable.

| Doc | What it establishes |
| --- | --- |
| [`DISTRIBUTED.md`](DISTRIBUTED.md) | The 30B SFT launch contract and storage constraint. |
| [`DATASET_SPEC.md`](DATASET_SPEC.md) | Data provenance, verification, task-pool and step-centric admission rules. |
| [`KORE_BENCH_BLUEPRINT.md`](KORE_BENCH_BLUEPRINT.md) | Task taxonomy and benchmark scope. |
| [`P0_RESULTS.md`](P0_RESULTS.md) | Why rooflines remain an integrity ceiling, not a validated speed predictor. |
| [`E2E_SERVING_GATE.md`](E2E_SERVING_GATE.md) | Serving-gate procedure and limits. |
| [`EVAL_RESULTS.md`](EVAL_RESULTS.md) | The failed 14B midtrain experiment that ruled out CPT on an instruct checkpoint. |
| [`FRONTIER_CLAIM_PROTOCOL.md`](FRONTIER_CLAIM_PROTOCOL.md) | Requirements for a model-vs-system claim. |

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
- **Data diversity precedes volume.** The registry contributes 1,289 trainable
  tasks. `scripts/build_task_pool.py` adds screened external tasks without
  mutating registry manifests; the current pool is 14,859 plannable tasks and
  14,461 eligible tasks after held-out screening (13,570 external tasks; 398
  registry seeds excluded as contaminated).
- **The SFT mix teaches local improvement, not search imitation.**
  `kore/data/step_centric.py` retains correctness-preserving revisions that
  fix a kernel or improve measured speed by at least 5%; it drops regressions,
  no-ops, and suspicious speedups. `scripts/build_sft_v3_mixture.py` then
  deduplicates and re-screens every source against held-out ids and families.

AgentKernelArena is the external AMD bar: on gfx950, published Claude Opus
means are 6.89x (`torch2hip`), 6.69x (`hip2hip`), and 2.13x
(`triton2triton`). `scripts/run_agent_kernel_arena.py` runs the benchmark's
declared compile, correctness, and performance commands in copied workspaces
and scores with its formula; `scripts/spur_aka_1node.sbatch` is the GPU-node
entrypoint. The local discovery filter reports 402 gfx950-runnable tasks from
the 412-task suite.

Kernel-Smith-235B's reported KernelBench Triton speedup is 3.70 versus 3.33 for
Claude-4.6-opus. Dr. Kernel-14B reports 31.6% single-pass and 47.8% with
test-time scaling on KernelBench L2, versus 28.6% for GPT-5 and 26.7% for
Claude-4.5-Sonnet. These are comparison bars, not KORE results.
