# The v5 frontier RL run: what ran, and what it produced

The first and only multi-turn RL run to complete optimizer steps against the v5
SFT checkpoint. Everything in this document is derived from the artifacts in
`docs/evidence/rl_v5_frontier/`, which are the run's own outputs, not a
retelling.

This document exists because the run happened on infrastructure that is not
this cluster, launched from a resolved config that was never committed. Without
it the repository has no record that stage 3 produced anything, and the numbers
quoted elsewhere have no provenance.

## Summary

| | |
| --- | --- |
| Window | 2026-08-27 05:49:24Z to 2026-08-30 15:39:12Z (81.8 h) |
| Optimizer steps completed | 31 (step 0 through step 30) |
| Configured horizon | 1,000 steps; the run was stopped, not converged |
| Checkpoint used for every number below | `global_step` 30 |
| World size | 8 ranks, one MI355X (gfx950) each |
| Advantage estimator | `trloo` (`kore/policy/trloo.py`) |
| Benchmark | AgentKernelArena, 416 tasks, all four arms on the same commit |

## 1. The config that actually ran is not the config in `configs/`

This is the single most important fact in this document.

`docs/evidence/rl_v5_frontier/grpo_resolved_config.json` is the resolved config
the trainer received. It does not correspond to any committed file. The nearest
committed config is `configs/grpo_coder30b_a3b_trloo_burst.json`, and it differs
in the learning rate:

| Key | This run | `_trloo.json` | `_2node.json` | `_burst.json` |
| --- | --- | --- | --- | --- |
| `learning_rate` | **2e-06** | 1e-06 | 1e-06 | 1e-06 |
| `num_trajectories` | 16 | 16 | 16 | 16 |
| `num_turns` | 4 | 3 | 3 | 4 |
| `max_tool_turns` | 4 | 3 | 3 | 4 |
| `tasks_per_step` | 8 | 4 | 4 | 8 |
| `advantage_estimator` | trloo | trloo | trloo | trloo |

The `2e-06` is not an inference from the config file alone. Every one of the 33
`grpo_step_dist` events in `docs/evidence/rl_v5_frontier/grpo_events.jsonl.gz`
carries `"lr": 2e-06`, so this is what the optimizer used, on every step, on
every rank. `2e-06` is the learning rate of the legacy 14B recipe; no 30B config
in this repository sets it.

The run also wrote to `/mnt/vast/shasriva/runs/grpo_v5_frontier` from a base
model at `/mnt/vast/shasriva/models/sft_coder30b_a3b_v5`. Every committed config
targets `/shared_nfs`. Reproducing this run therefore needs both a config change
and a different storage layout; it is not a matter of pointing an existing
launcher at a new checkpoint.

The resulting checkpoint was later copied to
`/shared_nfs/shasriva/rl_checkpoint-30`, where it survives and is fetchable with
`scripts/fetch_weights.sh`. It verifies as 25 safetensors shards on a
`qwen3_moe` config of 48 layers and 128 experts, with `trainer_state.json`
reporting `global_step` 30. The SFT checkpoint it started from was not found
there, so the `/mnt/vast` paths above are a record of where the run ran rather
than a place to go looking.

## 2. What the run produced

All four arms ran the same 416 AgentKernelArena tasks through the same harness
with the same attempt budget. Ledgers are in `docs/evidence/rl_v5_frontier/`.

| Arm | Correct | Compiles | Speedup (geomean, own correct set) |
| --- | --- | --- | --- |
| Base Qwen3-Coder-30B-A3B | 28.6% | 49.0% | 0.841x |
| + supervised fine-tuning | 52.4% | 82.7% | 0.951x |
| + multi-turn RL (step 30) | **62.3%** | **93.8%** | 0.955x |
| Claude Opus | 67.3% | 78.8% | 1.617x |

Paired on identical task IDs, RL against the fine-tuned model: **63 tasks fixed,
22 broken**, McNemar z = 4.34 with continuity correction. The net of 41 is 9.9
points, which is the correctness gain.

Speed did not move. On the 186 tasks where both the fine-tuned and RL models
produced a correct, timed kernel, the geometric means are 0.940x and 0.878x, a
ratio of 0.934x with a paired t of -2.84 on the log ratio. RL is faster on 56 of
those 186.

## 3. The overlong mask discarded 908 training samples

The run set `max_response_length` 1024 and `overlong_buffer_len` 512, so the
mask dropped every response of 512 tokens or more, not merely those close to the
limit. This is half the generation range rather than its tail.

The cost is counted, not estimated. Each `grpo_step_dist` event carries
`n_overlong_masked`. Summing the surviving event for each of the 31 distinct
steps gives **908** masked samples, a mean of 29.3 per step, range 9 to 50.

Two steps, 4 and 23, appear twice in the log because they were retried after
preemption. Their pairs are (45, 22) and (27, 9). Summing every event instead
gives 980, and summing the first event per step gives 949; 908 is the figure
that counts each step once, using the attempt that actually contributed to the
gradient. Anyone recomputing this number should expect 908 and should not be
surprised by the other two.

The mask falls hardest on exactly the behaviour the speed objective needs. A
merely-correct kernel is short; a tiled, unrolled, shared-memory kernel is long.
This is the most likely explanation for section 2's flat speed result, though
the run was not repeated with the mask corrected, so it remains a diagnosis
rather than a demonstration.

## 4. What is in `docs/evidence/rl_v5_frontier/`

| File | What it is |
| --- | --- |
| `grpo_resolved_config.json` | The resolved config the trainer received |
| `ckpt30_trainer_state.json` | Checkpoint manifest at `global_step` 30 |
| `grpo_events.jsonl.gz` | 23,057 structured training events, the source for section 3 |
| `rl_ckpt30_results.json.gz` | Arena ledger, RL arm, 416 tasks |
| `rl_ckpt30_baseline.json.gz` | Arena ledger, fine-tuned arm, same tasks |
| `opus_results.json.gz` | Arena ledger, Opus arm, same tasks |

The base-model ledger lives outside this repository and is the one input to
section 2 that is not committed here.

## 5. Caveats

The run stopped at 31 of 1,000 configured steps. Nothing here should be read as
a converged result.

Opus was measured against its own baseline ledger, so its 1.617x speedup shares
a task set with the other arms but not a denominator. Treat the correctness and
compile columns as directly comparable and the Opus speedup column as
indicative.

The checkpoint itself is not in this repository and is not reachable from this
cluster.
