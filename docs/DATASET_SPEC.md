# KORE production data specification

KORE trains an AMD gfx950 kernel improver, not a text-only code generator. A
record is useful only when its correctness and performance assertions come from
execution through the task contract.

## Scope

The product recipe is instruct → SFT → multi-turn RL for
`Qwen/Qwen3-Coder-30B-A3B-Instruct`. The 14B midtrain and DPO corpora are
historical inputs to experiments, not stages in this recipe. CPT on an instruct
checkpoint destroyed instruction-following in the recorded 14B experiment, and
the 30B target has no Base sibling for either CPT or a residual merge.

## Task pool before trajectories

The static registry contains 1,334 tasks, of which 1,289 are trainable. It
cannot grow in place: the taxonomy digest guards split manifests, so adding a
directory would invalidate a campaign already in flight.

`scripts/build_task_pool.py` creates a separate pool through
`kore/tasks/external.py`. A candidate must be safe to execute, deterministic,
large enough to measure, classifiable to a non-held-out family, structurally
distinct, and decontaminated against KORE and KernelBench references. Pool tasks
reuse the trusted `task.yaml` / `reference.py` / seed / driver ABI but never
appear in `registry.all_tasks()`.

The current aggregate is 14,859 plannable / 14,461 eligible tasks. It includes
13,570 external tasks and excludes 398 registry seeds that fail held-out
screening.

## Agentic trajectories

Datagen writes the transcript plus the evidence needed to understand each
revision: tool calls, compile result, correctness result, timing result, and
best observed kernel. `scripts/spur_data_driver.sh` builds the pool first, then
fills available capacity without exceeding six QoS nodes. Measured throughput is
462–469 episodes per hour per node with a 100% keep rate.

The raw trajectory is not directly the SFT target. A long search contains failed
edits and regressions; teaching all of it would make eventual success look like
permission to flail.

## Step-centric SFT rows

`kore/data/step_centric.py` extracts each revision that is worth imitating:

- a transition from incorrect to correct; or
- a correct-to-correct revision with measured gain of at least 5%.

It rejects a broken child, a missing or nonpositive measurement, a gain under
the timing-noise floor, a regression, and a speedup above the anti-hack cap.
Each accepted row retains the transcript only through the accepted assistant
revision, making it a local-improvement sample.

## v3 mixture admission

`scripts/build_sft_v3_mixture.py` assembles:

1. the decontaminated v2 base;
2. step-centric AMD trajectories;
3. newly recovered rows from alternate SFT cuts.

Every source uses the same admission function: require messages, exclude
estimated content above 17,408 tokens, reject held-out task IDs and families,
and deduplicate message content. The result is written to
`data/b05factory/sft/multicap_v3.jsonl`; this path is cluster-only until the
data release process materializes it.

## Verification and baseline discipline

Correctness is gated before speed. The task driver uses the FP32 oracle across
declared shapes, plus adversarial and determinism checks when enabled. Timings
are cold-cache, paired, and variance-gated.

There are two baseline lanes. Of 1,334 registry tasks, 106 use a production
vendor baseline: 63 declare AITER, 4 declare hipBLASLt, runtime resolution adds
33 `gemm_fusion` hipBLASLt tasks and 6 gated activations using AITER. The
remaining 92%, including all 1,052 generated breadth tasks, use torch. A
torch-lane speedup is useful for training but is not evidence of beating a
production library.

## What is deliberately absent

DPO pairs are not a production source. Kernel quality is observed directly by
compile, oracle, and time; converting that evidence to a static preference loses
the feedback that the multi-turn RL policy can use. Similarly, a midtrain
corpus is not a fallback: no chosen 30B Qwen has the required Base model.
