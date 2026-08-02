# Stage-3 GRPO launch readiness

**Verdict: NO-GO as shipped. CONDITIONAL GO with Blocker 1's patch applied
(it is applied in this working tree) and Blocker 2 answered.**

The GRPO path is in much better shape than "never run" suggested. On a real
8-rank launch against real 14B weights it froze the held-out shape lane, loaded
three full-weight model replicas per rank, synced the policy into the generation
replica, rolled out against the real `KoreEnv`, compiled and verified real
candidate kernels on GPU through the full adversarial battery, produced real
rewards, took real optimizer steps, and wrote a 166 GB checkpoint that validates
against its own manifest.

Then it hit the thing that only a real resume can find:

> **Every requeued GRPO job died on its first optimizer step after resuming.**

Stage 3 is a 2000-step online-RL run that *cannot* finish in one 23 h allocation
— the launcher says so itself — so every run after the first was guaranteed to
crash. That is Blocker 1. I reproduced it in 8 seconds on a tiny model, fixed it
in `_load_full_optim_state`, and re-verified both the reproduction and the real
14B resume. **`tests/test_grpo_checkpoint_resume.py` passed throughout**, before
and after, because it never runs under real FSDP mixed precision — which is
exactly why this review was asked for evidence from real artifacts.

Blocker 2 is not a defect but a planning question with a large number attached:
at the shipped rollout topology, 2000 steps is **tens of 23 h allocations** on a
QOS capped at 8 nodes for all users.

- **Verified on:** `master` @ `aeada9b`, plus the `_load_full_optim_state` fix this document adds
- **Hardware:** 8 × AMD Instinct MI350X (gfx950), 252 GiB HBM each — the full production topology
- **Stack:** Python 3.10.14, torch 2.10.0+rocm7.0, transformers 4.57.6, accelerate 1.14.0. `flash_attn` absent → SDPA, which GRPO pins deliberately.
- **Regression tests:** `tests/test_grpo_launch_readiness.py` — 34 pass, 1 skip (needs a visible accelerator), no xfail.

| # | Item | Verdict |
|---|---|---|
| 1 | Launches on real weights, real optimizer steps, real rollouts | **PASS** |
| 2 | Reward path end to end (rollout → GPU verify → reward) | **PASS** — unqualified |
| 3 | Checkpoint write and validation | **PASS** |
| 4 | **Resume** | **WAS BROKEN — fixed and re-verified** (Blocker 1) |
| 5 | Frozen held-out shape lane under multi-rank | **PASS** |
| 6 | Per-GPU memory | **PASS** — wide margin |
| 7 | Budget ledger counters move | **FAIL** — inert under the shipped config (Blocker 3) |
| 8 | Run length fits the cluster | **NO** — Blocker 2 |
| 9 | Stage-2 handoff | **BLOCKED** — `runs/dpo_14b_frontier` does not exist |

## What was fixed

| Blocker | Fix |
|---|---|
| 1 — every resumed run died on its first optimizer step | `kore/policy/grpo.py::_load_full_optim_state` now restores the Adam moments to fp32 after `load_state_dict` downcast them to the (BF16) param dtype |

Blockers 2, 3 and 4 are reported with patches rather than applied: 2 is a
capacity decision for the human, 3 is entangled with a config another agent owns,
and 4 depends on Stage 1 existing.

---

## Blockers

### Blocker 1 — every resumed GRPO run died on its first optimizer step (FIXED)

**Severity: launch-fatal, and specific to the case that matters most.** A fresh
GRPO run works. A *resumed* one crashes — and `scripts/spur_grpo_1node.sbatch`'s
own header says:

> EXPECT MULTIPLE JOBS: total_steps is 2000 with per-step rollouts and verified
> benching, which no 23 h allocation will finish.

Observed on the real 8-rank 14B run, resuming from a real 166 GB `checkpoint-2`.
Resume *discovery* and *restore* both reported success:

```
grpo distributed: resuming from checkpoint  path=runs/grpo_14b_verify/checkpoint-2 start_step=2 total_steps=4
grpo resume: restored training state       path=... optimizer=True scheduler=True
```

and then, on all 8 ranks, at the first `opt.step()`:

```
File "kore/policy/grpo.py", line 3394, in _train_grpo_distributed
    opt.step()
  ...
File "torch/optim/adam.py", line 627, in _multi_tensor_adam
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
RuntimeError: Tensors of the same index must be on the same device and the same
dtype except `step` tensors that can be CPU and float32/64 notwithstanding
```

**Mechanism, measured rather than inferred.** I reproduced it on a 2-rank tiny
Qwen3 through the *real* helpers and the *real* `configs/accelerate_fsdp_grpo.yaml`,
printing the dtype of every optimizer state tensor on each side:

```
[after a real step (the SAVE side)]
    PARAM        -> [('torch.float32', 'cuda:0')]
    exp_avg      -> [('torch.float32', 'cuda:0')]
    exp_avg_sq   -> [('torch.float32', 'cuda:0')]
[gathered full_osd] exp_avg -> torch.float32 cpu
[after _load_full_optim_state (the RESUME side)]
    PARAM        -> [('torch.bfloat16', 'cuda:0')]
    exp_avg      -> [('torch.bfloat16', 'cuda:0')]      <- silently downcast
    exp_avg_sq   -> [('torch.bfloat16', 'cuda:0')]      <- silently downcast
[REPRODUCED] opt.step() after resume raised: Tensors of the same index ...
```

`torch.optim.Optimizer.load_state_dict` casts every floating-point state tensor
to `param.dtype`. At the moment `_load_full_optim_state` runs, FSDP holds the
params in their **BF16 low-precision form**, so the fp32 Adam moments are cast to
bf16 — permanently. When the training step later runs, the params are back to
their fp32 master and the moments are stuck at bf16, so the foreach AdamW refuses
to group them.

The crash is the *lucky* part. The quiet part is worse: `exp_avg_sq` holds
squared gradients, whose dynamic range is precisely what bf16 does not have. Had
the dtype check been lenient, every resume would have silently corrupted the
second moment and the run would have looked healthy.

**The patch** (applied), in `_load_full_optim_state` immediately after
`inner.load_state_dict(sharded)`:

```python
for group in inner.param_groups:
    for param in group["params"]:
        entry = inner.state.get(param)
        if not entry:
            continue
        for key, value in list(entry.items()):
            if key == "step" or not torch.is_tensor(value):
                continue
            if value.is_floating_point() and value.dtype != torch.float32:
                entry[key] = value.to(dtype=torch.float32)
```

`step` is deliberately excluded — torch documents it as legitimately CPU-resident
and the error message exempts it explicitly.

**Verification after the patch**, same harness:

```
[after _load_full_optim_state (the RESUME side)]
    PARAM        -> [('torch.bfloat16', 'cuda:0')]
    exp_avg      -> [('torch.float32', 'cuda:0')]
    exp_avg_sq   -> [('torch.float32', 'cuda:0')]
[NOT REPRODUCED] opt.step() after resume succeeded unchanged
```

Regression test:
`test_restored_optimizer_moments_stay_fp32_under_mixed_precision`.

**Why the existing tests did not catch it.** `tests/test_grpo_checkpoint_resume.py`
(682 lines) passes before and after this patch. It exercises checkpoint
discovery, the manifest contract and the fail-closed paths on CPU, with no FSDP
and no mixed precision, so `param.dtype` is fp32 there and the downcast never
happens. The defect lives exactly in the gap between that test and a real
multi-rank resume.

---

### Blocker 2 — 2000 steps is tens of allocations at the shipped rollout topology

**Severity: the run cannot be scheduled as specified.** Measured on the real
8-rank run, at a deliberately tiny rollout topology:

| phase | wall time |
|---|---|
| startup: identity, shape freeze, 3 × 14B loads per rank | ≈ 3 min |
| step 0 rollout + **test-time search** + update | 8 min 20 s |
| step 1 rollout + update | 3 min 11 s |
| step 2 rollout (skipped — see F1) | 3 min 45 s |
| final consolidate + save | 21 s |

That is **~3.5 min/step** with `tasks_per_step: 1`, `num_trajectories: 8`
(one trajectory per rank), `num_turns: 2`, `max_response_length: 1024`.

The shipped config asks for `tasks_per_step: 8`, `num_trajectories: 16`
(two per rank) and `max_response_length: 16384`. Those multiply the rollout work
by roughly 8 × 2 = **16×** before accounting for longer generations:

| assumption | min/step | 2000 steps | 23 h allocations |
|---|---|---|---|
| 16×, generation length unchanged | 56 | 1,867 h | **82** |
| 16×, but rollouts overlap better at scale (÷2) | 28 | 933 h | **41** |
| 24× (longer responses at `max_response_length: 16384`) | 84 | 2,800 h | **122** |

Even the most generous reading is **~40 allocations**, on `amd-general-qos`,
which is capped at **8 nodes for the entire QOS** and is usually full. This is
not a defect in the code — it is a number nobody has written down, and the
launcher header's "EXPECT MULTIPLE JOBS" does not convey it.

**Treat this as an extrapolation with a wide error bar**, not a measurement: it
comes from a 3-step run at 1/16th of the rollout width. But the conclusion is
robust to a factor of two in either direction.

**The lever already exists and is switched off.** `GRPOConfig` has an adaptive
horizon:

```python
adaptive_steps: bool = False           # off by default (fixed total_steps)
min_steps: int = 100                   # never stop before this many steps
plateau_patience: int = 40             # stop after this many steps w/o improvement
```

`configs/grpo_14b_full.json` does not set `adaptive_steps`, so the run is a fixed
2000 steps with no early stop. **Proposed patch** (that config is owned by
another change, so this is a recommendation, not an edit):

```diff
   "total_steps": 2000,
+  "adaptive_steps": true,
+  "min_steps": 200,
+  "plateau_patience": 60,
```

With that, `total_steps` becomes a hard cap rather than a plan, and the run stops
when the gathered reward mean plateaus. The controller's decisions are derived
from cross-rank-gathered scores, so every rank breaks in lockstep — this is safe
under FSDP. **Whether to lower `total_steps` outright is a recipe decision for
the human.**

---

### Blocker 3 — the budget ledger never records anything under the shipped config

**Severity: no verifier-spend accounting on the one stage that spends it.**
The task asked me to confirm that `correctness_calls`, `fresh_timed_calls`,
`verifier_gpu_seconds`, `replay_hits` and `profiler_gpu_seconds` move. **They do
not.** No `budget_ledger.json` was written by a complete 3-step run that
performed dozens of real GPU verifications.

The whole ledger is gated on one flag:

```python
def _feature_runtime(config):
    return getattr(config, "_grpo_feature_runtime", None)     # set ONLY under strict

def _budget_ledger(config):
    runtime = _feature_runtime(config)
    return runtime.ledger if runtime is not None else None    # -> None
```

`_grpo_feature_runtime` is set in `train_grpo` only inside `if strict:`, i.e. only
when `strict_feature_validation` is true. Everything downstream —
`_record_generated_tokens`, `_record_optimizer_tokens`, `_record_groups`, the
cross-rank `BudgetLedgerV1.merge`, and `_write_grpo_foundations_state` (which
returns immediately when the runtime is `None`) — is a silent no-op.

`configs/grpo_14b_full.json` sets `strict_feature_validation: false`, and its
in-config comment argues that persuasively: strict mode's
`_UNSUPPORTED_STRICT_FEATURES` refuses `use_search`, `value_prefilter`,
`coevolve_mint`, `search_bnb`, `coevolve_regret_vs_opus` and `distillation`
outright, four of which are this recipe's whole point. **I agree with that
choice.** The problem is that budget accounting was bundled into the same flag as
feature strictness, so choosing the research recipe also silently turns off the
verifier-spend meter.

The verifier *work* absolutely happens — item 2 shows dozens of `oracle_report`
and `eval_done` events, including replay-cache hits. Only the *accounting* is
missing.

**Proposed patch** (in `kore/policy/grpo.py`, which I did not apply because it
interacts with the capability/strict design another change is actively
reworking): decouple the ledger from strictness by constructing a bare
`BudgetLedgerV1` when strict is off.

```diff
     strict = bool(getattr(config, "strict_feature_validation", False))
     if strict:
         ...
         setattr(config, "_grpo_feature_runtime", feature_runtime)
+    else:
+        # Budget accounting is not a strictness feature: the verifier spend is
+        # real either way, and a run with no ledger cannot answer "how much GPU
+        # time did reward evaluation cost". Attach a runtime that ONLY carries a
+        # ledger, so _record_* and _write_grpo_foundations_state work while the
+        # feature-phase assertions stay disabled.
+        setattr(config, "_grpo_feature_runtime", _ledger_only_runtime(config))
```

The narrower alternative — have `_budget_ledger` lazily create and cache a
`BudgetLedgerV1` on the config when no runtime exists — is a three-line change
and touches nothing else.

Regression test: `test_budget_ledgers_merge_across_ranks_by_summing` locks the
merge arithmetic that would otherwise under-report by `world_size`; the gating
itself is documented here rather than pinned by an xfail, because the fix belongs
to the change that owns the strict-mode design.

---

### Blocker 4 — the Stage-2 handoff does not exist

`scripts/spur_grpo_1node.sbatch` defaults `FROM_STAGE` to
`runs/dpo_14b_frontier`, which does not exist (Stage 1 is held, so Stage 2 has
not run). The resolver fails in milliseconds with a clear message before any rank
loads anything, which is correct behaviour and is regression-tested. It is
nonetheless a hard precondition.

The same applies to `ref_checkpoint: "runs/sft_14b_frontier"` — see F2, which is
the more dangerous of the two because it fails *open*.

---

## Evidence, item by item

### 1. GRPO launches on real weights and takes real optimizer steps — PASS

Real launcher (`scripts/launch_distributed.sh grpo`), real entrypoint, **all 8
MI350X**, real `Qwen/Qwen3-14B`, the shipped `configs/grpo_14b_full.json` with
only the rollout topology shrunk (`total_steps: 3`, `tasks_per_step: 1`,
`num_trajectories: 8`, `num_turns: 2`, `max_response_length: 1024`, two tasks).
Every production feature flag — `agentic`, `starpo_s`, `dynamic_sampling`,
`rc_grpo`, `sc_grpo`, `gtpo_codesim`, `coevolve`, `use_search`, `roofline_gate` —
was left on.

```
grpo distributed: starting  backend=fsdp world=8 num_trajectories=8 agentic=True
grpo distributed: AGENTIC tool-use rollouts ACTIVE on the sharded path
grpo(dist): policy->replica weight sync  synced=443 total=443
coevolve: frontier curriculum active (distributed)  menu_size=2 mint=True

[grpo/dist] step 0 kept=1/1 world=8 epochs=1 meanR=0.479 loss=0.0000
grpo_step_dist  step=0 n_groups=1 n_kept_groups=1 n_attempts=1 reward_mean=0.4791
                loss=0 global_tokens=3342 n_overlong_masked=5 gpu0_gb=97.55
[grpo/dist] step 1 kept=1/1 world=8 epochs=1 meanR=1.102 loss=0.5167
grpo_step_dist  step=1 ... reward_mean=1.102 loss=0.5167 global_tokens=6232
                n_overlong_masked=3 gpu0_gb=119.9
grpo(dist) step: dynamic-sampling exhausted - skipping  step=2 attempts=3   (all 8 ranks)
grpo_done  steps=3 mean_reward_last=1.102 out=runs/grpo_14b_verify backend=fsdp world=8
```

`synced=443 total=443` is the whole generation design working: all 443 parameter
tensors copied from the FSDP-sharded policy into the plain full-weight replica
under `summon_full_params`, so the rollout generates with zero FSDP collectives
and ragged decode cannot deadlock. Mean reward rose 0.479 → 1.102 across two
steps — not evidence of learning at this scale, but evidence the reward signal
reaches the objective.

**`loss=0.0000` at step 0 is expected, not a bug.** With `ppo_epochs: 1` the
importance ratio is `exp(logp − old_logp) = 1` on the first pass, so the scalar
surrogate reduces to the negative mean advantage, and advantages are mean-centred
over the group. The *gradient* is not zero — it is `−Σ Aᵢ ∇logpᵢ`, the ordinary
policy gradient. Step 1 reports `0.5167` because the replica and the sharded
policy compute log-probs through different kernels, so the ratio is no longer
exactly 1. Worth writing down, because "loss = 0" looks alarming in a log.

### 2. The reward path, end to end — PASS

This is the strongest part of the stage. A rollout produced a candidate kernel,
it was compiled and verified on GPU through the full adversarial battery, and it
yielded a reward:

```
env: eval_start   task=gen_add_mul_bf16 n_shapes=5 do_bench=False
env: oracle_report task=gen_add_mul_bf16 verified=True
     live_prongs=['random', 'adversarial', 'metamorphic', 'determinism']
     random_trials_per_shape=5 random_shapes=5 random_elements=616181760
     false_accept_bound=0 false_accept_bound_log10=-2.676e+04
     prong_states={'random':'pass','adversarial':'pass','metamorphic':'pass','determinism':'pass'}
env: eval_done    compiled=True correct=True snr_min=999 worst_speedup=1 cv_pct=1.188
agent.harness: agent_episode_done task=gen_add_mul_bf16 turns=5 success=True best_reward=0.3
```

Four things worth calling out, all observed:

- **The adversarial prong actively rejected a wrong kernel** rather than rubber-
  stamping everything:
  ```
  oracle_report verified=False live_prongs=['adversarial']
    prong_states={'random':'inconclusive','adversarial':'fail','metamorphic':'off','determinism':'off'}
  eval_done compiled=True correct=False snr_min=-999
  ```
- **Timed benching is real and low-variance**: `worst_speedup` came back as
  1.0, 0.5812, 0.9251 and 0.5595 across different candidates with `cv_pct`
  between 0.49% and 1.19%. Different kernels get different speedups — which is
  what gives the group its reward variance.
- **The replay cache hits**: re-evaluating an identical `source_sha` returned
  `cached=True` immediately. (The counter that should record this is inert —
  Blocker 3.)
- **Rewards spread across trajectories**: `best_reward` ranged 0.3 to 1.756
  within a step.

The co-evolution curriculum also ran, minted new tasks, and fed the distillation
sink:

```
coevolve(dist) step  step=0 archive_coverage=7 measured_tasks=1 mean_solve_rate=0.75
                     mint=True minted_materialized=6 minted_pool=6 opus_anchored=False
coevolve distilled win  task=gen_add_mul_bf16 speedup=1 count=1
```

### 3. Checkpoint write and validation — PASS

`checkpoint-2` written by `_save_grpo_checkpoint_distributed` on the 8-rank run:

| | |
|---|---|
| Size | **166 GB** |
| Contents | 13 fp32 safetensors shards + `optimizer.pt` + `scheduler.pt` + `rng_state.pth` + `trainer_state.json` + tokenizer |
| `trainer_state.json` | `{"global_step": 2, "optimizer_state_saved": true, "total_steps": 3, "world_size": 8}` |
| Manifest | 26 entries, all present on disk |

Validated with the real helpers against the real directory:

```
_read_grpo_trainer_state(checkpoint-2) -> valid
_find_grpo_resume_checkpoint(...)      -> runs/grpo_14b_verify/checkpoint-2
```

The consolidated optimizer state is internally consistent — 443 param entries,
`step`/`exp_avg`/`exp_avg_sq` all fp32 — so the **save** side is sound. It was
the load side that was wrong (Blocker 1).

The final `grpo_done` consolidation wrote a plain 13-shard HF checkpoint to the
output dir, which is what the certification/soup stages load.

### 4. Resume — WAS BROKEN, now works

See Blocker 1 for the failure and the fix. After the patch, the real 8-rank 14B
resume from the real 166 GB `checkpoint-2` was re-run end to end:

```
grpo distributed: resuming from checkpoint  path=runs/grpo_14b_verify/checkpoint-2
                  start_step=2 total_steps=4
grpo resume: restored training state  optimizer=True scheduler=True     (all 8 ranks)
[grpo/dist] step 2 kept=1/1 world=8 epochs=1 meanR=0.819 loss=0.0000
grpo_step_dist  step=2 world=8 reward_mean=0.8187 global_tokens=7004 gpu0_gb=112.3
```

That last pair of lines is the proof: **a real optimizer step completed after a
real resume**, at exactly the point where the unpatched code raised on all eight
ranks. `optimizer=True scheduler=True` means the consolidated Adam state and the
LR schedule were both scattered back, not silently reinitialised.

Discovery semantics are separately correct and regression-tested: newest-first
past a damaged directory, manifest-based completeness (a checkpoint that lost a
single file is rejected, not half-restored), and **fail-closed** — if checkpoint
directories exist but none is resumable it raises rather than silently restarting
at step 0. The error is broadcast as a value so all ranks fail together instead
of leaving seven blocked in a collective.

### 5. The frozen held-out shape lane — PASS under a real 8-rank launch

`freeze_training_shape_splits` ran on all 8 ranks before any rollout, and
produced exactly one lane:

```
runs/grpo_14b_verify/shape_splits/
    gen_add_mul_bf16.json
    gen_add_relu_bf16.json
    shape_split_index.json
```

Rank 0 is the single writer and every other rank waits on the barrier — with 8
ranks racing on one directory, a follower that enumerated it mid-write would
publish an index omitting the manifests the writer added afterwards, and
certification would later reject the lane. One writer removes the interleaving,
and that is what the run did.

Two operational notes for the real launch: the default task list is **1,289
tasks**, so rank 0 writes 1,289 manifests while seven ranks sit on a filesystem
barrier bounded at **900 s**. That barrier is loud on timeout, which is right. I
did not measure the 1,289-manifest freeze time — my run froze 2 — so **the one
untested question here is whether a cold NFS freeze of 1,289 manifests fits in
900 s.** On the cluster's shared `/home` that is worth checking before the first
launch; it is a cheap check (`python -m kore.tasks.shape_policy freeze`).

### 6. Memory — PASS, wide margin

Measured per-GPU peak, from `gpu_mem_snapshot()` on rank 0 (each rank reports its
own device):

| step | peak |
|---|---|
| 0 | **97.55 GB** |
| 1 | **119.9 GB** |

**This run had no KL-anchor reference loaded** (F2), so it holds two of the three
full-weight replicas, not three. Adding the frozen reference costs one more bf16
14B = **27.5 GiB**, putting the real production peak at roughly **147 GB/GPU
against 252 GiB**.

The arithmetic, which the regression test pins:

| term | per rank |
|---|---|
| policy params, bf16, **replicated** (SHARD_GRAD_OP never reshards) | 27.5 GiB |
| `gen_replica`, full weight, bf16 | 27.5 GiB |
| frozen KL reference, full weight, bf16 | 27.5 GiB |
| bf16 grads + fp32 master + 2 fp32 Adam moments, **sharded ÷ 8** | 24.0 GiB |
| **subtotal** | **106.5 GiB** |
| measured overhead (activations, KV cache during generation, fragmentation) | ~20–25 GiB |

This was the likeliest place to OOM and it does not. The margin comes from the
`÷ world` on the optimizer terms; note it does **not** improve with more ranks
beyond this, because 82.5 GiB of the 106.5 is replicated by design.

`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (which the launcher sets)
matters here — the rollout allocates and frees a KV cache every step against a
resident 82.5 GiB floor.

### 7. Budget ledger — FAIL, see Blocker 3

### 8. Capability audit is doing its job

`train_grpo` writes `capability_audit.json` unconditionally and logs loudly. On
my run it correctly identified both artifact gaps:

```
WARN policy.capabilities: requested GRPO capabilities are inert  count=2
  features=['coevolve_regret_vs_opus', 'ref_anchor']
  - ref_anchor [artifact] requested by ref_anchor_coef, ref_checkpoint:
    'runs/sft_14b_frontier' is not a directory on this filesystem; _load_ref_model
    prints a warning and trains with NO KL anchor, so the retention term is
    silently dropped
```

This is a genuinely good mechanism and it caught the exact thing I would
otherwise be reporting as an undetected silent failure. It is in another change's
area; I am confirming it works, not proposing anything.

---

## Non-blocking findings

**F1 — one step in three did no work, and this scales.** Step 2 was skipped on
all 8 ranks in lockstep:

```
grpo(dist) step: dynamic-sampling exhausted - skipping  step=2 attempts=3
```

`dynamic_sampling_refill` discards any group whose reward std is ≤ `min_std` and
refills, up to `max_attempts = 3 × target_groups`. All 3 attempts produced
collapsed groups, so the step paid a **full 3 min 45 s of 8-GPU rollout** and
applied no update. The lockstep is correct — the decision is derived from gathered
scores, so no rank desyncs — but the cost is real and it *increases* as the policy
gets good at a task and starts solving it the same way every time. At production
`tasks_per_step: 8` there are 24 attempts to find 8 live groups, which helps, but
the underlying dynamic is unchanged. Worth logging a running skip-rate; a run
where most steps skip is burning an allocation to learn nothing.

**F2 — the KL anchor fails OPEN.** `_load_ref_model` catches every load failure,
logs a warning, returns `None`, and training proceeds with `ref_anchor_coef` set
but no anchor at all. On my run that meant the entire retention term was absent —
`ref_checkpoint: "runs/sft_14b_frontier"` does not exist. The capability audit
does flag it (item 8), so this is not undetected, but a `WARN` in a log that also
carries thousands of rollout events is easy to miss. **Before launch, confirm the
log says the reference loaded** — or set `ref_anchor_coef: 0.0` to state honestly
that there is no anchor. This is the difference between RL-with-retention and
plain RL, and it also changes per-rank memory by 27.5 GiB.

**F3 — the test-time search fires on the very first step of every allocation.**
`search_every: 50` and `0 % 50 == 0`, so step 0 always triggers it. Measured
cost: roughly **4 minutes of rank-0 straggler time** with `search_budget: 32`,
while the other seven ranks sit at the next collective. That is well inside the
1800 s collective timeout, so it is safe — but the code comment claims it is "a
small fraction of a step", and at my topology it was more than half of step 0.
Because a requeued child restarts at its resumed step, this recurs whenever the
resume step is a multiple of 50.

**F4 — `n_overlong_masked` was 5 of the step-0 samples and 3 of step-1's.** DAPO
overlong filtering dropped generated responses that hit the cap. That is correct
behaviour (a truncated response is a biased gradient), and it was inflated by my
tiny `max_response_length: 1024`; at the production 16,384 it should be rare.
Flagged so that a high `n_overlong_masked` in the real log is recognised as a
signal that responses are hitting the cap, not as noise.

**F5 — GRPO checkpoints are 166 GB each and `save_steps` is 100.** With
`save_total_limit` floored at 2, steady state is 332 GB plus the final
consolidated 55 GiB. The transient during rotation is three at once, ≈ 500 GB.
**Provision ≥ 600 GB** on the GRPO output filesystem. Note this is *smaller* than
SFT/DPO's 221 GB checkpoints because GRPO's optimizer state is sharded and
consolidated differently.

**F6 — neither co-evolution artifact exists on either data root.** Neither
`data/full14b/` nor `data/b05factory/` holds `coevolve_wins.jsonl` or
`opus_scores.json`, on the cluster or locally. `coevolve_distill_path` is a
*sink* and is created on demand (my run wrote one), so that is fine.
`coevolve_opus_scores_path` is a *source*: `_build_opus_scores` fails safe and
returns `None`, leaving the regret-vs-Opus curriculum inert — which the
capability audit reports. This is inside the change that owns
`configs/grpo_14b_full.json`; reporting, not editing.

**F7 — `adaptive_steps` is off, so `total_steps: 2000` is a plan, not a cap.**
See Blocker 2.

**F8 — resource preflight reports `unresolved`, by design.** Same as SFT's F9 and
DPO's F6. The launcher correctly exports `KORE_RESOURCE_PREFLIGHT=report`. **Do
not set `strict`.**

---

## Launch command

Once Stage 2 lands at `runs/dpo_14b_frontier`:

```bash
export SPUR_CONTROLLER_ADDR=http://crs-m2m-cpu-spur-005:6817
export KORE_SPUR_CONTROLLER_ADDR=$SPUR_CONTROLLER_ADDR   # required: the run WILL requeue

cd /home/shasriva/Kore-RL/KORE

# The KL anchor fails OPEN (F2). Confirm the reference exists BEFORE launching.
ls runs/sft_14b_frontier/config.json runs/sft_14b_frontier/*.safetensors >/dev/null \
  || echo "WARNING: GRPO will train with NO KL anchor"

sbatch scripts/spur_grpo_1node.sbatch \
    configs/grpo_14b_full.json \
    runs/dpo_14b_frontier \
    runs/grpo_14b_frontier \
    data/b05factory
```

Then, in `runs/grpo-<jobid>.out`, check in this order:

1. `requested GRPO capabilities are inert` — read every entry. If `ref_anchor` is
   listed, the KL anchor is off (F2).
2. `frozen held-out shape lane published  tasks=1289` — and that it appeared
   within the 900 s barrier (item 5).
3. `grpo(dist): policy->replica weight sync  synced=443 total=443`.
4. `[grpo/dist] step 0 ...` — `loss=0.0000` at step 0 is expected (item 1).
5. On the **second** allocation: `grpo resume: restored training state
   optimizer=True scheduler=True`, followed by a real `[grpo/dist] step N`. That
   second line is the one Blocker 1 was eating.

Provision **≥ 600 GB** on the output filesystem (F5). Do not set
`KORE_RESOURCE_PREFLIGHT=strict` (F8).

**Before committing to 2000 steps, settle Blocker 2.** I would enable
`adaptive_steps` and treat 2000 as a cap.

---

## What I could not prove

1. **GRPO has never trained the actual Stage-2 output.** Every run used
   `Qwen/Qwen3-14B`. The architecture, parameter count, wrap class and tokenizer
   are identical and the directory-handoff path is regression-tested, but the
   rewards, solve rates and memory figures are from the *base* model.
2. **The production rollout topology was never run.** Everything is at
   `tasks_per_step: 1`, `num_trajectories: 8`, `num_turns: 2`,
   `max_response_length: 1024`, on 2 tasks out of 1,289. The shipped config
   *parses*, *resolves* and *starts*, and the memory arithmetic has a wide
   margin — but the 16–24× wall-time extrapolation in Blocker 2 rests on a
   3-step run and is the weakest number in this document.
3. **The 1,289-task shape freeze was not timed.** I froze 2 manifests. Whether
   1,289 fit inside the 900 s barrier on cold NFS is unmeasured, and is the one
   cheap check I would run before launching (item 5).
4. **Only 3 steps, one of which skipped.** Nothing here says whether this recipe
   learns, whether the KL anchor holds retention, or whether rewards stay
   non-degenerate over hundreds of steps.
5. **The KL-anchor path never executed.** `ref_checkpoint` pointed at a
   nonexistent directory throughout, so `_load_ref_model`'s success path, the
   third full-weight replica's memory, and the k3 anchor term in the loss are all
   unexercised. The 147 GB/GPU production estimate is 119.9 GB measured plus
   27.5 GiB arithmetic, not a measurement.
6. **The budget ledger's counters were never observed moving** — they cannot
   under the shipped config (Blocker 3). I verified the merge arithmetic by unit
   test and the underlying verifier work by its log events, but not the counters
   themselves.
7. **DeepSpeed ZeRO-3 (`sharding_backend: "deepspeed"`) is untested.** The
   shipped config uses `"auto"` → FSDP. The DeepSpeed branch of
   `_gather_full_optim_state` returns `None` and makes resume fail closed, which
   is at least honest.

---

## Reproducing this report

```bash
pytest tests/test_grpo_launch_readiness.py -q      # 34 pass, 1 skip, no xfail
pytest tests/test_grpo_checkpoint_resume.py -q     # the pre-existing unit tests
```

The GPU portions need 8 idle MI350X and ~600 GB of scratch. The config used was
`configs/grpo_14b_full.json` with `model_id=Qwen/Qwen3-14B`, `total_steps: 3`,
`save_steps: 2`, `tasks_per_step: 1`, `num_trajectories: 8`, `num_turns: 2`,
`max_prompt_length: 4096`, `max_response_length: 1024`, and
`tasks: ["gen_add_relu_bf16", "gen_add_mul_bf16"]`, into `runs/grpo_14b_verify`.

Blocker 1 reproduces in about 8 seconds on two GPUs without any 14B weights:

```bash
accelerate launch --config_file configs/accelerate_fsdp_grpo.yaml \
    --gpu_ids 0,1 --num_processes 2 .grpo_resume_repro.py
```

which builds a tiny real `Qwen3ForCausalLM`, drives the real
`_gather_full_optim_state` / `_load_full_optim_state`, prints the dtype of every
optimizer state tensor on both sides, and asserts that `opt.step()` after a
resume succeeds.
