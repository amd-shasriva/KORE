# Multi-turn RL readiness for the 30B MoE

**Status: recipe reviewed and regression-tested; launch blocked on the SFT
checkpoint.** `configs/grpo_coder30b_a3b_trloo.json` is the reviewed 30B
recipe: it parses, validates, and requests no capability the code cannot
deliver (`tests/test_grpo_30b_moe_config.py`). It is not yet launchable
because its policy source is the SFT stage's output
(`runs/sft_coder30b_a3b`), and that stage is still training
([`SFT_READINESS.md`](SFT_READINESS.md), job `9229`). The blocker is narrow
and mechanical, not "no recipe exists."

The 14B DPO handoff is intentionally absent from this path. Kernel quality is
observed directly by execution; collapsing compile, correctness, and timing
evidence into preference pairs discards information multi-turn RL can use
directly. There is also no DPO stage between SFT and RL for this backbone at
all: `ref_checkpoint` in the 30B config points at the post-SFT policy itself,
not at a preference-tuned intermediate.

## The reviewed 30B recipe

Selected settings from `configs/grpo_coder30b_a3b_trloo.json` (full
justification lives in the config's own `_comment_*` fields; this table is a
map, not a duplicate):

| Setting | Value | Why |
| --- | --- | --- |
| `model_id` / `model_revision` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` @ `b2cff646...` | Same backbone and pinned commit as the SFT stage; the launcher rewrites `model_id` to the SFT checkpoint at launch via `--from-stage`. |
| `advantage_estimator` | `trloo` | See below; the reason this config exists. |
| `variance_floor` | `0.0` | Must be 0 under `trloo`: the AVSPO floor injects virtual samples into normalization statistics that TRLOO never computes, so a nonzero value is an inert, audit-flagged request. |
| `num_trajectories` / `num_turns` / `max_tool_turns` | 16 / 3 / 3 | Dr. Kernel's reported defaults; `num_turns` matches `max_tool_turns` because the agentic rollout's horizon *is* `max_tool_turns`. |
| `tasks_per_step` | 4 | Half the 14B recipe's 8: this backbone's per-rank memory is larger and every turn compiles and benches, so step cost is rollout-dominated. |
| `learning_rate` / `lr_scheduler_type` | `1e-6` / `constant` | Half the 14B recipe's `2e-6`: fewer active parameters per token move further per update, and this is the first RL run on an unevaluated SFT checkpoint. |
| `ref_anchor_coef` | `0.0` | See memory note below. |
| `dual_clip_c` | `3.0` | Floors the clipped surrogate for negative advantages so one off-policy sample cannot own the whole gradient; should not bind at `ppo_epochs=1`. |
| `rejection_sampling` / `rejection_aggregate` | `true` / `geometric` | Geometric aggregation is dominated by the weakest turn, so one strong turn cannot carry a trajectory broken elsewhere. |
| `fsdp_transformer_layer_cls` | `Qwen3MoeDecoderLayer` | Matches the SFT stage; wrapping the dense class would shard nothing. |
| `save_steps` / `save_total_limit` | 50 / 2 | Same checkpoint-rotation arithmetic as SFT, at GRPO's own step cost. |

## TRLOO: the reason this config exists

Plain GRPO centers each sample's advantage on its group's mean, and that mean
*contains* the sample. `kore/policy/trloo.py` states precisely why that
matters more here than in single-turn RL: `build_kevin_samples` pools `m`
trajectories x `n` turns into one group, so the baseline for turn `t` of
trajectory `i` also contains every *other* turn of trajectory `i`, including
turns causally downstream of the action being credited. That downstream
credit is exactly the long-horizon signal multi-turn RL is supposed to
capture, and the self-inclusive baseline subtracts a fraction of it.

Measured by exact enumeration over small MDPs (`tests/test_trloo_advantages.py`,
no sampling noise, cross-checked against a finite-difference gradient):

- On a symmetric two-turn MDP, pooled GRPO returns exactly `(M-1)/M` of the
  true gradient (a uniform, benign shrink). Adding GRPO's std normalization
  makes the error *grow* with group size instead: +41% at M=3, +91% at M=5.
- On ~1,930 random asymmetric two-turn MDPs, pooled GRPO's expected gradient
  points the wrong way in 7-10 of them, and 55-58 with std normalization.
  Worst case found: true gradient +0.171, GRPO -0.007, GRPO+std -0.098,
  **TRLOO +0.171**.
- TRLOO produced zero wrong-direction cases in every configuration tried.

TRLOO (`turn_loo_advantages`) fixes this by centering each sample on the mean
of *other* trajectories at the *same* turn index, which is independent of the
credited action and therefore unbiased. Two design choices in that module are
settled by the same measurement discipline, not by taste: TRLOO applies no std
normalization at all (a leave-one-out std reintroduces bias by a different
route, measured ratio to the true gradient from -15,107x to +503,846x over
1,443 cases), and a sample with no leave-one-out peer at its turn is *kept*
with baseline 0.0 rather than dropped (dropping is a measurable selection
bias, up to 4.6x error with sign flips, because whether a turn exists depends
on the policy's own earlier actions).

`group_sample_advantages` in `kore/policy/grpo.py` reads
`config.advantage_estimator` and calls `trloo.turn_loo_advantages` when it is
`"trloo"`. If any sample in a group is missing its `(trajectory_id, turn_id)`
key, the trainer raises `AdvantageKeyError` rather than silently falling back
to the pooled estimator, so a broken key can never train a different
estimator than the config asked for without saying so.

## Distributed topology: SHARD_GRAD_OP, not FULL_SHARD

Unlike SFT (`FULL_SHARD`, ZeRO-3, since SFT never generates), GRPO's FSDP
plugin (`build_fsdp_plugin`) uses `SHARD_GRAD_OP` (ZeRO-2): gradients and
optimizer state are sharded, but parameters stay resident and un-resharded
after the forward pass. This is the enabler for co-located online RL:
`FULL_SHARD` reshards parameters to a flat 1-D buffer between forwards, which
breaks `model.generate()` and would force an all-gather on every decode step.

The live policy is synced once per step into a plain, un-sharded per-rank
replica (`_summon_full_params_ctx` + `_sync_gen_replica`), and every rollout
generates *locally* on that replica with zero FSDP collectives. That is what
lets `synced_gpus` stay off and ragged agentic turn counts not deadlock the
group: the only cross-rank synchronization point is the per-roll
`_all_gather_object` that reconstructs each Kevin group's full reward
baseline, which tolerates a different sample count per rank by design.

## Checkpoint and resume mechanics

Checkpoints are staged, validated against a written manifest
(`kore_grpo_files`), and atomically renamed into place before rotation, so a
crash mid-save never presents a half-written directory as resumable
(`_publish_grpo_checkpoint`). `_grpo_save_total_limit` floors
`save_total_limit` at 2 in code, independent of what a config sets, because
retaining one checkpoint would let rotation delete the only resumable copy.

Resume restores optimizer state, LR schedule, RNG state, and the step counter
explicitly and fails closed on any missing piece
(`_restore_grpo_training_state`) rather than silently reinitializing the
optimizer. One historical bug this guards against, fixed in
`_load_full_optim_state`: `Optimizer.load_state_dict` casts every
floating-point state tensor to the *parameter's* dtype, and under FSDP mixed
precision the parameters are bf16 at load time, so a naive load silently
downcast the fp32 Adam moments to bf16 and every resumed run died on its first
optimizer step. The fix restores fp32 explicitly for every state tensor after
the sharded load.

## Memory: what `ref_anchor_coef=0.0` avoids

`ref_anchor_coef` is `0.0`, not the 14B recipe's `0.001`. Any value above zero
makes `grpo.py` load a second, frozen, full-weight reference replica per rank
for the k3 KL anchor (`_load_ref_model`; not FSDP-sharded, because it is only
ever used for no-grad reference-logprob forwards). At 30.5B bf16 that is
another ~61 GB per rank on top of the rollout replica, roughly ~122 GB per
rank before any training state, for a term that measured at ~0.1% of the loss
gradient at the 14B recipe's coefficient. Dropping KL entirely is standard
practice in reasoning RL rather than a shortcut here (DAPO removes it
explicitly): KL exists to keep a policy near its initial model, which is the
opposite of what a run that must acquire a capability it lacks wants.
Stability instead comes from TRLOO's dual-clip ratio clipping. The
rollout/training mismatch correction (`mismatch_weight`) stays off
(`mismatch_correction: false`) for the same reason: at `ppo_epochs=1` the
measured weight sits within about a percent of 1.0, so the correction would
state a capability the run does not need. Its diagnostic half
(`_grpo_step_mismatch_stat`) still runs unconditionally, so a weight that
drifts from 1.0 is visible as a bug report rather than silently corrected
away.

## Capability audit is a gate, not log decoration

`train_grpo` writes `capability_audit.json`
(`kore.policy.capabilities.audit_requested_capabilities`). A requested feature
with a missing artifact (a reference checkpoint, a value model, an Opus-score
cache) can otherwise fail open and train without the advertised component.
`tests/test_grpo_30b_moe_config.py` asserts the shipped 30B config carries no
`DECLARED`-scope finding (provable from the config text alone): TRLOO is
actually selected, the AVSPO floor it bypasses is zero, the coverage reward is
unarmed without a hardware receipt, and `physics_shaping_weight` stays zero on
the existing evidence in `docs/P0_RESULTS.md` (no authorized family).

## Legacy 14B invariant (regression pin only)

The block below is not the production recipe. It pins the historical 14B
config `tests/test_docs_contract.py` checks against, so that regression
target cannot silently go untracked while this page describes the 30B recipe
above it.

```jsonc
{
  "model_id": "Qwen/Qwen3-14B",
  "output_dir": "runs/grpo_14b_full",
  "num_trajectories": 16,
  "num_turns": 4,
  "tasks_per_step": 8,
  "total_steps": 2000,
  "reward_mode": "speedup",
  "agentic": true,
  "physics_shaping_weight": 0.0,
  "search_value_prior": false
}
```

## 30B RL launch gate

Do not submit the 30B job until every item below is true:

1. `runs/sft_coder30b_a3b` (or wherever `--from-stage` points) is a complete,
   consolidated SFT checkpoint; the current blocker.
2. `configs/grpo_coder30b_a3b_trloo.json`'s `capability_audit.json` carries no
   unapproved inert feature at actual launch time (the config-text-only
   `DECLARED` check above cannot see an `ARTIFACT`-scope gap, such as a
   `ref_checkpoint` that does not exist yet).
3. A measured 30B MoE memory snapshot from an actual rollout step, including
   the rollout replica and any reference replica. The legacy 14B figures
   (below) do not transfer to a MoE architecture at 2x the parameter count.
4. Distributed generation is exercised end to end on this backbone at least
   once (`SHARD_GRAD_OP` plus a resumed optimizer step), not inferred from the
   14B measurement.

## Durable distributed-RL constraints (from the 14B lineage)

These are architectural facts about the code path, not 30B measurements, and
they still bind:

### Resume has to preserve FP32 optimizer moments

Covered above under checkpoint mechanics; the fix is in `grpo.py` and applies
to every backbone that trains under FSDP.

### Generation requires a non-full-shard policy replica

Covered above under distributed topology.

### Memory is driven by replicated models, not just parameter count

The legacy 14B measurement peaked at 119.9 GB/GPU without the KL reference,
and about 147 GB/GPU with a three-replica estimate, on 288 GiB MI355X cards.
The absolute figure does not transfer to the 30B MoE, but the accounting
does: policy, generation replica, and any frozen reference are each
*replicated* per rank, while only gradients and optimizer state shard. More
ranks does not shrink the replicated floor.

## Regression contract

- `tests/test_grpo_30b_moe_config.py`: the 30B config parses, validates,
  requests no provably-inert capability, selects TRLOO, keeps the AVSPO floor
  at zero, agrees with the SFT stage on backbone/revision/FSDP wrap class, and
  matches its checkpoint-rotation bound to the SFT config's.
- `tests/test_trloo_advantages.py`: the exact-enumeration bias measurements
  cited above.
- `tests/test_docs_contract.py`: pins the legacy `jsonc` block against
  `configs/grpo_14b_full.json`, key for key.
