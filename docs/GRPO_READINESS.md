# Multi-turn RL readiness

**Verdict: not launchable as a 30B production stage yet.** Production policy is
instruct → SFT → multi-turn RL with compile, oracle, and timing evidence. The
checked-in `configs/grpo_14b_full.json` is an operational legacy experiment,
not a substitute for a reviewed 30B / TRLOO recipe. This page retains the
failure modes discovered under real distributed RL so the new recipe does not
repeat them.

The 14B DPO handoff is intentionally absent from the 30B path. Kernel quality is
observed directly by execution; collapsing compile, correctness, and timing
evidence into preference pairs discards information that multi-turn RL can use.
Plain GRPO also has a biased policy-gradient estimator for multi-turn episodes,
which is why the active RL work is moving to turn-level leave-one-out credit.

## Current configuration invariant

Until a coherent 30B RL configuration lands, the documented legacy config remains
mechanically pinned rather than silently untracked:

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

The block is intentionally legacy evidence: when a 30B TRLOO recipe is added,
replace this block and the contract-test target in the same change. Do not
delete the invariant.

## Durable distributed-RL constraints

### Resume has to preserve FP32 optimizer moments

The historical eight-rank run could save, discover, and load a checkpoint, then
die on the first resumed `opt.step()`. FSDP temporarily held parameters in bf16,
and `Optimizer.load_state_dict` downcast the saved FP32 Adam moments to the
parameter dtype. The mixed-dtype optimizer step then failed. The repair restores
Adam moments to FP32 in `_load_full_optim_state`.

This only appears after a real requeue. Unit tests without FSDP mixed precision
cannot establish that resume is safe. Every new RL recipe needs a save → requeue
→ resume → optimizer-step test on its actual sharding topology.

### Generation requires a non-full-shard policy replica

The agentic generation replica uses `SHARD_GRAD_OP` (ZeRO-2). `FULL_SHARD`
would all-gather parameters during generation and can deadlock with the rollout
topology. This is a topology constraint, not a performance preference: change
it only with a distributed generation proof.

### Memory is driven by replicated models

The legacy 14B measurement peaked at 119.9 GB/GPU without the KL reference; the
three-replica estimate was about 147 GB/GPU on 252 GiB MI355X cards. Its absolute
number does **not** transfer to 30B MoE, but the accounting does: policy,
generation replica, and any frozen reference are replicated, while gradients and
optimizer state shard. More ranks do not erase the replicated floor. Obtain a
fresh 30B snapshot before claiming capacity.

### Checkpoint rotation is an allocation constraint

Legacy GRPO checkpoints were 166 GB and temporarily reached roughly 500 GB
during rotation. The numeric figure is historical; the invariant is not. A
recipe must state checkpoint size, final-model size, `save_total_limit`, and the
rotation peak before it can be submitted to a shared filesystem.

### Capability audit is a gate, not log decoration

`train_grpo` writes `capability_audit.json`. A requested feature with a missing
artifact—such as a reference checkpoint, value model, or Opus-score cache—can
otherwise fail open and train without the advertised component. Read the audit
before accepting a run. For the new profiling rewards, P0 currently authorizes
no empirical residual-shaping family; roofline remains an integrity ceiling, not
a validated speedup predictor.

### Evaluation and held-out isolation remain mandatory

The task list must be frozen before distributed rollout, and task-pool
decontamination must remain separate from the registry split. Test-time search
can bank only verifier-confirmed kernels; it cannot contribute unverified
off-policy reward.

## 30B RL launch gate

Do not submit a 30B RL job until all of these are recorded in its own config and
run evidence:

1. exact SFT checkpoint and tokenizer identity;
2. TRLOO / multi-turn credit implementation and a CPU regression test;
3. distributed generation topology (`SHARD_GRAD_OP`) plus an actual resumed
   optimizer step;
4. 30B MoE memory snapshot including all replicas and KV cache;
5. checkpoint rotation arithmetic and output-volume reservation;
6. capability audit with no unapproved inert feature;
7. execution-reward, held-out, and AgentKernelArena/KernelBench-AMD evaluation
   plan.

The existing `scripts/spur_grpo_1node.sbatch` remains useful only to reproduce
the 14B experiment. It must not be invoked as a 30B production command without
a reviewed 30B config and starting checkpoint.

## Historical evidence retained

The prior 14B report demonstrated real kernel compilation, correctness checks,
timing rewards, optimizer steps, manifest validation, and distributed resume.
It also found skipped updates from collapsed reward groups, a missing reference
checkpoint that disabled retention, and expensive rank-zero search at step zero.
Those observations motivate the gates above, but they are not performance or
readiness claims for 30B.
