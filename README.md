# KORE

KORE trains a kernel-optimization agent for AMD MI355X (`gfx950` / CDNA4).
Every candidate is compiled, checked against its task oracle, and only then
timed. A fast wrong kernel receives no speed credit.

The production backbone is
`Qwen/Qwen3-Coder-30B-A3B-Instruct`: 30.5B parameters total, 3.3B active per
token, 48 layers, 128 experts, and 8 selected experts per token. It is chosen
for code and tool use, and its active MoE compute is about ten times lower than
a comparable 32B dense model while retaining comparable memory requirements.
The pinned, full-parameter SFT recipe is
[`configs/sft_coder30b_a3b.json`](configs/sft_coder30b_a3b.json).

## The production path

```mermaid
flowchart LR
  P[Task pool] --> D[Agentic datagen]
  D --> S[Step-centric decomposition]
  S --> M[v3 mixture]
  M --> F[30B MoE SFT]
  F --> E[AKA + KernelBench-AMD + vs_opus]
  E --> R[Multi-turn RL]
  R --> T[Test-time scaling]
```

1. `scripts/build_task_pool.py` mines external PyTorch modules without changing
   the content-addressed registry. The pool is screened against held-out KORE
   tasks and KernelBench, deduplicated, and materialized with the same task ABI.
   The current inventory is 14,859 plannable tasks and 14,461 eligible tasks:
   13,570 are external and 398 registry seeds are excluded by the held-out
   screen.
2. `scripts/spur_data_driver.sh` waits for the pool before it saturates
   datagen; task diversity, not raw episode count, limits non-redundant
   trajectories. The QoS cap is six concurrent nodes. Measured datagen is
   462–469 kept episodes per node-hour at 100% keep rate.
3. `kore/data/step_centric.py` turns a trajectory into independent revisions.
   It keeps only a correctness-preserving ≥5% speed gain or the revision that
   first fixes a broken kernel. This avoids supervising search flailing as if it
   were an optimization policy.
4. `scripts/build_sft_v3_mixture.py` joins v2, recovered rows, and decomposed
   AMD trajectories. Every source passes the same length, exact-content dedup,
   task-id, and held-out-family gates before it can enter
   `data/b05factory/sft/multicap_v3.jsonl`.
5. SFT starts from the vendor instruct checkpoint. There is no production
   continued-pretraining, chat-vector merge, or DPO stage.
6. Multi-turn RL consumes execution feedback. Kernel quality is directly
   verifiable by compile, correctness, and timing; preference pairs discard that
   information. The multi-turn path also avoids applying a plain GRPO estimator
   to a setting where its policy gradient is biased.

## Why the older recipe is not production

The former 14B path was `base → CPT → chat-vector merge → SFT → DPO → GRPO`.
It is retained only as experimental history. No eligible 30B-class Qwen has a
Base variant: Qwen3-32B, Qwen3.6-35B-A3B, and Qwen3-Coder-30B-A3B are all
instruct-only. CPT requires a Base model; the residual transfer requires a
same-family Base/Instruct pair.

This is not only a packaging constraint. The 14B experiment documented in
[`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md) found that CPT on an instruct
model improved the raw-code objective while destroying instruction-following and
kernel generation. `scripts/spur_pipeline_driver.sh` therefore defaults
`KORE_RECIPE=direct`; `KORE_RECIPE=cpt` is a legacy 14B-only branch.

## Measurement discipline

The registry has 1,334 tasks. 106 are vendor-lane tasks: 63 declare AITER and 4
declare hipBLASLt; runtime resolution adds 33 `gemm_fusion` hipBLASLt tasks and
6 gated activations backed by AITER. The remaining 92%, including all 1,052
generated breadth tasks, are torch-lane tasks. Vendor and torch speedups must
not be pooled: a torch-lane result does not establish a production-library win.

The production correctness oracle combines reseeded random trials, adversarial
fills when `KORE_VERIFIED_CORRECTNESS=1`, determinism, and post-timing
re-verification. Metamorphic checks are fail-closed for the 168 generated tasks
whose generator contracts prove a relation. Timing alternates candidate and
baseline under cold-cache, variance-gated measurement.

Roofline bounds remain an integrity ceiling and an optional shaping potential,
not a demonstrated speed predictor. The residual study was retracted as a
shared-denominator artifact; [`docs/P0_RESULTS.md`](docs/P0_RESULTS.md) records
the `INTEGRITY_ONLY` verdict.

## Evaluation

`scripts/run_agent_kernel_arena.py` evaluates AgentKernelArena in copied
workspaces using its compile → correctness → performance contract and its score.
The gfx950 filter admits 402 of the benchmark's 412 tasks. Published Claude
Opus comparison means are 6.89x (`torch2hip`), 6.69x (`hip2hip`), and 2.13x
(`triton2triton`); these are external bars, not KORE claims.

For KernelBench Triton, Kernel-Smith-235B reports 3.70 average speedup versus
3.33 for Claude-4.6-opus. Dr. Kernel-14B reports 31.6% single-pass and 47.8%
with test-time scaling on KernelBench L2, compared with GPT-5 at 28.6% and
Claude-4.5-Sonnet at 26.7%. KORE reports its own results through
KernelBench-AMD and paired `vs_opus` evaluation rather than importing those
numbers as its own.

## Operations

The data driver stops before training and leaves `runs/DATA_NOT_FINAL` in place
for human review. Do not remove that sentinel automatically.

```bash
export SPUR_CONTROLLER_ADDR="http://crs-m2m-cpu-spur-005:6817"
sbatch scripts/spur_build_task_pool.sbatch
bash scripts/spur_data_driver.sh
```

For the explicit production SFT invocation and the checkpoint storage
constraint, see [`docs/DISTRIBUTED.md`](docs/DISTRIBUTED.md). A 30B checkpoint
is about 488 GB; `save_total_limit` must remain 1 because rotation already peaks
near 976 GB on the shared volume.

## Release prerequisites

[`LICENSE`](LICENSE) declares the repository AMD-internal and not for external
release. [`THIRD_PARTY.md`](THIRD_PARTY.md) contains the required attribution.
Do not publish source, weights, a dataset, or a build artifact without a new
authorized release decision.

## Testing

```bash
PYTHONPATH=. /home/shasriva/kore-venv/bin/python -m pytest tests/ -q \
  -p no:warnings -k "not gpu"
```

## Documentation

- [`docs/README.md`](docs/README.md) — evidence and decisions
- [`docs/DISTRIBUTED.md`](docs/DISTRIBUTED.md) — 30B FSDP launch
- [`docs/DATASET_SPEC.md`](docs/DATASET_SPEC.md) — dataset contract
- [`kore/data/README.md`](kore/data/README.md) — data factory
- [`kore/eval/README.md`](kore/eval/README.md) — evaluation harnesses
