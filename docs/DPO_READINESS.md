# Stage-2 DPO launch readiness

**Verdict: CONDITIONAL GO.** The DPO path works end to end on real 14B weights —
identity resolution, preference-pair masking, dataset load, FSDP construction,
real optimizer steps with sane preference metrics, a 221 GB checkpoint write, and
a real resume from it all execute on the current `master`. Three defects were
found; the two I own are **fixed and regression-tested**. The remaining condition
is external: **Stage-1 SFT has not run**, so the checkpoint this stage is supposed
to train and anchor to does not exist.

Because SFT is held, every run below used a **real, cached `Qwen/Qwen3-14B`** as
the stand-in base — the same architecture, the same `Qwen3DecoderLayer` FSDP wrap
class and the same 14,768,307,200 parameters as the midtrain output, and a real
14B directory in every respect that DPO touches. **No run trained the midtrain
checkpoint itself**, and no run wrote into `runs/dpo_14b_frontier`. What that
substitution cannot prove is called out in "What I could not prove" below.

- **Verified on:** `master` @ `aeada9b`, with the working-tree fixes this document adds
- **Hardware:** 8 × AMD Instinct MI350X (gfx950), 252 GiB HBM each; the DPO runs used HIP ordinals 0 and 1
- **Stack:** Python 3.10.14, torch 2.10.0+rocm7.0, transformers 4.57.6, trl 0.29.1, accelerate 1.14.0, datasets 3.6.0. `flash_attn` is **absent** → SDPA, which is what `dpo.py` pins anyway.
- **Regression tests:** `tests/test_dpo_launch_readiness.py` — 28 pass, no xfail. Three of them encoded the blocker list and are now ordinary regressions.

| # | Item | Verdict |
|---|---|---|
| 1 | Preference corpus exists, loads, and is TRL-shaped | **PASS** — unqualified |
| 2 | Preference-pair masking and the loss actually computed | **PASS** — unqualified |
| 3 | Model identity for the policy AND the frozen reference | **PASS** |
| 4 | Real multi-rank run: steps, checkpoint, resume | **PASS** — real 14B, real steps, 221 GB checkpoint, real resume |
| 5 | `save_total_limit` >= 2 reaches the trainer | **PASS** — verified by observing rotation |
| 6 | Step count, wall time, memory | **PASS with a corrected budget** (Blocker 1) |
| 7 | Iterative-DPO reference refresh | **PASS** — with a recipe disagreement to settle (Blocker 3) |
| 8 | Stage-1 handoff | **BLOCKED** — `runs/sft_14b_frontier` does not exist |

## What was fixed

| Blocker | Fix |
|---|---|
| 1 — the launcher's step and wall-time budget ignored preference weighting, understating the run by 73% | `scripts/spur_dpo_1node.sbatch` now derives its budget from the **weighted** 167,054 rows → 1,305 steps and a measured 13–19 h, and names `apply_pref_weights` as the reason |
| 2 — the shipped config relied on the `save_total_limit` dataclass default | `configs/dpo_14b_full.json` now sets `"save_total_limit": 2` explicitly with a justifying comment, matching midtrain and SFT |
| 3 — the launcher never said it runs ONE pass while the campaign defaults to 2 iterative rounds | `scripts/spur_dpo_1node.sbatch` header now states the recipe it implements and how an iterative round maps onto it |

---

## Blockers

### Blocker 1 — the run is 1,305 optimizer steps, not 755 (FIXED)

**Severity: allocation sizing.** `scripts/spur_dpo_1node.sbatch` sized the job
from the raw pair count:

```
# SINGLE NODE, no multi-node variant: 96,675 preference pairs at an effective
# batch of 128 pairs/step (2 x 8 x 8 ranks) for 1 epoch is ~755 optimizer steps.
# ... so budget roughly 8-11 h
```

But `kore/policy/dpo.py` consumes the corpus's per-pair `weight` by
**deterministic multiplicity** before the `Dataset` is built, and it is **on by
default** (`KORE_PREF_TRAIN_WEIGHTING`, default `"1"`):

```python
if w >= 1.0:
    out.extend(_strip(r) for _ in range(int(round(w))))
elif rng.random() < w:
    out.append(_strip(r))
```

Measured on the real corpus with the real function:

```
load_preference_jsonl -> 96,675 rows
apply_pref_weights(enabled=True)  -> 167,054 rows (1.728x)
apply_pref_weights(enabled=False) -> 96,675 rows
deterministic across calls (FSDP ranks must agree): True
```

At the shipped `2 × 8 × 8 = 128` pairs/step that is **1,305 optimizer steps**,
not 755 — a **73% underestimate** of a multi-hour stage. The determinism check
matters independently: the sub-1.0 branch is stochastic, and a seed that differed
across ranks would give each rank a different row list to shard.

Regression tests: `test_preference_weighting_multiplies_the_real_corpus_by_1_73x`,
`test_full_run_step_count_uses_the_WEIGHTED_row_count`,
`test_launcher_step_budget_accounts_for_preference_weighting`.

---

### Blocker 2 — the shipped config inherited its retention instead of stating it (FIXED)

**Severity: durability convention.** `kore/policy/configs.py` already carries the
rule, on `MidTrainConfig`:

> Note that 1 is NOT crash-safe on its own: the Trainer rotates the previous
> checkpoint out around the new save … **Every shipped launch config therefore
> sets >= 2 explicitly.**

`DPOConfig.save_total_limit` defaults to 2 and `build_trl_dpo_kwargs` reads it
(`getattr(config, "save_total_limit", 2)`), so the *behaviour* was already
correct — and I confirmed it on real artifacts rather than by reading:
`checkpoint-3` was rotated out when `checkpoint-8` landed, leaving exactly
`checkpoint-6` and `checkpoint-8`. But `configs/dpo_14b_full.json` did not say
so, which is the one thing the repo's own rule asks for: a later edit to the
dataclass default would silently lower a shipped launch's retention.

The config now sets it explicitly. Regression tests:
`test_dpo_save_total_limit_is_read_from_config_and_defaults_to_two`,
`test_shipped_dpo_config_states_its_retention_explicitly`.

---

### Blocker 3 — the launcher and the campaign implement different Stage-2 recipes (FIXED by documenting; the choice is the human's)

**Severity: recipe ambiguity.** `scripts/run_campaign.py:3301` defaults
`--dpo-rounds` to **2**, which routes to `_stage_dpo_iterative` — relabel
on-policy from the current checkpoint, aggregate DAgger-style, retrain with
`["ipo", "sft"]` against a **refreshed** frozen reference, fold DAgger repairs
back into SFT. `scripts/spur_dpo_1node.sbatch` runs `kore.policy.dpo` directly:
**one** non-iterative pass over the pre-built corpus with `["sigmoid", "sft"]`.

Both are defensible; they are not the same stage, and nothing said which one the
frontier run is. The launcher header now states that it implements the single
pass and that round *N* of the iterative recipe is this launcher re-invoked with
`FROM_STAGE` set to round *N−1*'s output. **The human still has to pick one.**

The reference-refresh logic itself is correct — see item 7.

---

## Evidence, item by item

### 1. The preference corpus exists, loads, and is TRL-shaped — PASS

Full streaming audit of all 96,675 rows of `data/b05factory/dpo/pairs.jsonl`
(1,093,321,730 B), on the cluster and again locally after a byte-identical copy:

| | |
|---|---|
| **Rows** | **96,675 — exactly the documented count** |
| Malformed JSON | 0 |
| Missing `prompt`/`chosen`/`rejected` | 0 |
| Empty prompt / chosen / rejected | 0 / 0 / 0 |
| Non-string message content | 0 |
| `prompt` not a list | 0 |
| `chosen` or `rejected` not a single-turn list | 0 |
| `chosen`/`rejected` role != `assistant` | 0 |
| `prompt` last role | `user` × 96,675 |
| `prompt` roles | `system` × 96,675, `user` × 96,675 |
| **Degenerate (`chosen == rejected`)** | **2** |

Every row is exactly the conversational shape `trl.DPOTrainer` consumes
natively: `prompt = [system, user]`, `chosen = rejected = [one assistant turn]`.
There is nothing in this corpus that would break TRL.

Extra columns present on every row — `weight`, `margin` (52,715 non-null),
`anchor`, `_provenance` (`kind: dpo_group`) — are read by
`load_preference_jsonl` and stripped to TRL's `{prompt, chosen, rejected}` schema
by `apply_pref_weights`, so nothing unexpected reaches the `Dataset`.

Weight distribution: 50,359 rows at exactly 1.0, 11,287 at 0.25, 35,029 above
1.0 (top of the tail: 8.0 × 208, 5.605 × 159), and **zero** at ≤ 0, so no row is
dropped outright.

The 2 degenerate rows are worth one sentence and no patch: for an identical pair
the implicit reward difference is identically zero, so they contribute a constant
`log 2` to the loss and an exactly **zero** gradient. They are wasted compute
(0.002% of the corpus), not a correctness problem.

### 2. Preference-pair masking and the loss actually computed — PASS

DPO has no template surgery to verify — the SFT stage's `{% generation %}`
injection has no analogue here. TRL 0.29.1 instead derives the completion by
**subtraction**, and that is the fragile step:

```python
prompt_ids        = apply_chat_template(prompt, add_generation_prompt=True)
prompt_chosen_ids = apply_chat_template(prompt + chosen)
output["chosen_ids"] = prompt_chosen_ids[len(prompt_ids):]      # <- the offset
```

If Qwen3's `add_generation_prompt=True` rendering were not a byte-prefix of the
full rendering, TRL only **logs a warning** and then slices at the wrong offset,
training on a completion that starts mid-sequence. Measured on **64 real corpus
rows**, running TRL's own function body:

```
prompt-prefix mismatches (TRL warns + mis-slices the completion): 0/64
prompt_ids   mean=922  max=928
chosen_ids   mean=814  max=2662
rejected_ids mean=808  max=2628
```

Then through the real `DataCollatorForPreference` and the real
`_compute_loss` shift (`shift_completion_mask = completion_mask[..., 1:]`;
`per_token_logps[shift_completion_mask == 0] = 0.0`), decoding what survives:

| side | tokens | in loss | masked | what the model actually trains on |
|---|---|---|---|---|
| chosen | 2,157 | 1,097 | 1,060 | `<think>\n\n</think>\n\nFULL_KERNEL:\n\`\`\`python …` → `… return o\n\`\`\`\n<\|im_end\|>\n` |
| rejected | 2,157 | 1,237 | 920 | same shape, different kernel body |

In both cases: the system prompt does **not** appear in the loss span, the user
prompt does **not**, `<|im_end|>` **is** in the loss (the stop token trains), and
`<|im_start|>assistant` **is** in the masked span. That is the correct DPO
objective on real token ids.

**`truncation_mode: "keep_end"` is load-bearing, and measurably so.** The prompt
is ~922 tokens of fixed system+task preamble. Truncating the same real pair at
`max_length: 512`:

| mode | completion tokens kept |
|---|---|
| `keep_start` (TRL's default) | **0** |
| `keep_end` (what `dpo.py` sets) | 372 |

TRL's default would spend the entire budget on the prompt and hand the trainer a
batch with an **empty loss mask**. `dpo.py` defaults to `keep_end` even when the
config omits the key, which is right.

### 3. Model identity for the policy AND the frozen reference — PASS

From the real 2-rank run:

```
policy.dpo: model identity resolved  role=policy model_id=Qwen/Qwen3-14B
  mode=development verify=metadata revision=40c069824f4251a91eefaf281ebe4c544efd3e18
  revision_pinned_at_load=True parameter_count=14768307200
```

DPO resolves identity for **both** models before the heavy imports, which matters
here more than in SFT because this stage loads **two** 14B models. The shipped
config pins no separate `ref_model_id`, so `_reference_identity` returns the
policy's identity object *by reference* — one checkpoint cannot disagree with
itself about its own pin. When the reference genuinely differs (iterative DPO),
it gets an independent identity with its own `ref_model_revision` pin, and a
local directory correctly yields `load_kwargs == {}` rather than laundering the
base model's Hub commit onto a trained checkpoint.

Mutable or malformed revisions (`main`, `v1.0`, `40c0698`, `refs/pr/1`) raise
`FloatingRevisionError` in both development and production mode.

### 4. DPO can actually start training — PASS

Real launcher (`scripts/launch_distributed.sh dpo <config>`), real entrypoint
(`accelerate launch -m kore.policy.dpo`), `GPU_IDS=0,1`, `HF_HUB_OFFLINE=1`, real
`Qwen/Qwen3-14B`, **64 real preference pairs** taken from the head of the real
corpus, `max_length: 4096`, `per_device_train_batch_size: 1`,
`gradient_accumulation_steps: 4`.

```
dpo: dataset loaded  n_pairs=64 beta=0.1 loss_type=['sigmoid','sft']
                     label_smoothing=0.1 fsdp=True ref_model_id=None
                     chosen_tok_mean=1736 chosen_tok_max=3587
dpo: preference weighting  pairs_in=64 pairs_effective=64 weighting=True

step 1  loss 1.771  grad_norm 60.41  rewards_acc 0.625  margins  0.0697  logps_chosen  -695.4  peak 136.2 GB
step 2  loss 1.974  grad_norm 64.17  rewards_acc 0.375  margins -0.0149  logps_chosen  -810.6  peak 148.1 GB
step 3  loss 1.660  grad_norm 89.33  rewards_acc 0.750  margins  0.1118  logps_chosen -1033.5  peak 154.7 GB
...
step 8  loss 1.681  grad_norm 78.52  rewards_acc 0.750  margins  0.1386  logps_chosen  -985.6
```

`mean_token_accuracy` held at **0.785–0.823** and `entropy` at 0.34–0.45
throughout — a real pretrained-model objective on real data, not a degenerate
one. The composite RPO loss behaves as designed: ≈ 0.69 from the smoothed sigmoid
term plus ≈ 1.0 from the `sft` NLL-on-chosen anchor.

The likelihood-displacement alarm `dpo.py` documents is **not** firing:
`rewards/chosen` oscillates around zero rather than trending negative while
`rewards/margins` grows, and `logps/chosen` is not in monotonic decline. Over 8
steps that is a smoke test, not a verdict on a 1,305-step run — the callback that
would catch it is wired and emitting.

Then:

- **`checkpoint-3` written: 221 GB** — 13 fp32 shards + `optimizer.bin` +
  `pytorch_model_fsdp.bin` + `scheduler.pt` + `rng_state_{0,1}.pth` +
  `trainer_state.json`
- **rotation observed:** at `checkpoint-8`, `checkpoint-3` was deleted and
  exactly `checkpoint-6` + `checkpoint-8` remained — `save_total_limit: 2`
  demonstrably reaches the trainer
- final `trainer.save_model`: 13 shards, **55 GiB** (fp32, for the same reason
  SFT's output is — accelerate's bf16 mixed precision keeps an fp32 master and
  `FULL_STATE_DICT` gathers it)
- **total output directory: 496 GB** after a clean 8-step finish

**Resume works, on the real artifacts.** Relaunched with `num_train_epochs: 2`:

```
dpo: resuming from checkpoint  ckpt=runs/dpo_14b_verify/checkpoint-8
step  9  loss 1.778  grad_norm 106.7  rewards_acc 0.75   margins  0.0967
step 10  loss 1.797  grad_norm  93.8  rewards_acc 0.375  margins -0.1341
step 11  loss 1.909  grad_norm 108.7  rewards_acc 0.5    margins -0.0637
step 12  loss 1.587  grad_norm  84.3  rewards_acc 0.75   margins  0.2017
```

The step counter continued from 8 and the cosine LR schedule resumed at the right
point rather than restarting.

**Checkpoint discovery, exercised against the real 221 GB directories** rather
than synthetic fixtures. With both checkpoints intact, `latest_checkpoint` picked
`checkpoint-8`. With `checkpoint-8/trainer_state.json` removed — exactly what a
crash during a 221 GB save leaves behind — it fell back to the complete
`checkpoint-6` instead of returning `None` and restarting from step 0. SFT's
Blocker-3 fix is live and DPO inherits it.

### 5. Step count, wall time and memory

**Step count.** 167,054 weighted rows ÷ 128 pairs/step = **1,305 optimizer
steps** for the single epoch the config requests. At `save_steps: 200` that is
6 periodic checkpoints plus the end-of-training one, each 221 GB.

**Wall time — extrapolated, not measured at production shape.** Measured
**9.3 s/step** at 2 ranks, micro-batch 1, ~29k tokens/step → **1,559 tok/s/GPU**.
The corpus is **673.8M tokens/epoch** (chosen + rejected, over the weighted
rows), so 8 ranks project to:

| assumption | wall time |
|---|---|
| token parity with the 2-rank measurement | 15.0 h |
| −15% for micro-batch 2 feeding the matrix cores better | 12.7 h |
| +25% for FSDP all-gather traffic at 8-way vs 2-way | 18.7 h |

Call it **13–19 hours**. That fits one 23 h allocation, but with as little as 4 h
of margin at the pessimistic end, so the launcher's self-requeue is load-bearing
rather than a formality. Treat this as a throughput extrapolation: the measured
batch had a padded sequence length of 2,158 against a corpus mean of 2,076, so
the sequence shape is close to like-for-like, but the rank count is not.

**Memory — measured.** Peak **160.0 GB/GPU** at 2-way FSDP, `max_length: 4096`,
micro-batch 1. Analytical persistent state for 14,768,307,200 params is
236,292,915,200 B (220.1 GiB), which the preflight reports; at 2-way sharding
that is 110 GiB/rank, leaving ~40 GiB for the two concatenated forward passes,
the reference forward and workspace. **At the production 8-way the sharded term
drops to 27.5 GiB/rank**, so the same activation envelope lands near
**70–90 GB/GPU against 252 GiB**. Wide margin.

That margin is wider than `DPOConfig`'s own comment assumes, and for a reason
worth writing down: the comment sizes the peak at "seq 16384", but **no sequence
in this corpus is anywhere near that** (item 6).

**Host memory.** `load_preference_jsonl` reads all 96,675 rows into Python
objects (1.20 GB peak RSS), but `Dataset.from_list` on the 167,054 weighted rows
peaks at **15.34 GB RSS per rank** — ≈ **123 GB across 8 ranks** at startup, on
top of the FSDP load. Fine on a 3 TB host; worth knowing, since a previous 14B
midtrain died of host-memory exhaustion.

### 6. `max_length: 16384` is 3.4× the longest real pair

Tokenized over a 2,500-pair random sample with the real Qwen3-14B tokenizer:

| | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| prompt | 922 | 921 | 927 | 929 | 931 |
| prompt+chosen | 2,076 | 2,037 | 3,096 | 3,670 | **4,802** |
| prompt+rejected | 1,958 | 1,962 | 2,980 | 3,672 | 4,656 |

**0 of 5,000 sequences exceed `max_length`.** The prompt is essentially fixed
width (the system+task preamble; p99 − p50 = 8 tokens), and the whole corpus fits
in under 5k tokens.

This is not a bug and I am not proposing a patch — TRL pads to the longest
sequence *in batch*, not to `max_length`, so the over-provisioning costs no
memory. It matters for two reasons: the `keep_end` truncation guard is insurance
that never fires on this corpus (it is still correct to keep), and any memory
estimate written against "seq 16384" is describing a shape that does not occur.

### 7. Iterative DPO refreshes the reference correctly — PASS

Traced through `kore/data/onpolicy.py::iterative_dpo` and
`run_campaign.py::_stage_dpo_iterative`, and pinned by a test that drives the
real loop with stub policies:

```
round 0: ref_model_id=None            -> base_ckpt = sft         -> ckpt-round-0
round 1: ref_model_id=ckpt-round-0    -> base_ckpt = ckpt-round-0 -> ckpt-round-1
round 2: ref_model_id=ckpt-round-1    -> base_ckpt = ckpt-round-1 -> ckpt-round-2
```

`prev_ckpt = rd.policy_ckpt` after every round, and `train_fn` sets **both**
`model_id` and `ref_model_id` to that checkpoint. So round *N* trains from, and
anchors to, round *N−1* — which is the iterative-DPO recipe, and specifically not
the failure mode where every round's KL term measures drift from the same stale
SFT policy.

Two details that are right and easy to get wrong: the loss switches to
`["ipo", "sft"]` for the on-policy rounds (bounded IPO so near-deterministic
on-policy pairs cannot push the reward gap to infinity, keeping the NLL anchor),
and `build_trl_dpo_kwargs` reconciles `loss_type`/`loss_weights` arity so a
per-round override that changes only one of the pair cannot raise inside
`DPOTrainer` and hard-stop the stage. I verified the reconciliation in all three
directions (composite with matching weights, composite with mismatched weights,
scalar with stale weights).

### 8. Stage-1 handoff — BLOCKED, and it fails correctly

`scripts/spur_dpo_1node.sbatch` defaults `FROM_STAGE` to
`runs/sft_14b_frontier`, which does not exist on the cluster (only
`runs/midtrain_14b_frontier` does). The resolver catches this in milliseconds,
before any rank loads anything:

```
FATAL launch-config resolution: dpo: --from-stage 'runs/sft_14b_frontier' is not
a loadable checkpoint: runs/sft_14b_frontier does not exist. The previous stage
must have finished and consolidated its weights before this one can train them.
```

That is the correct behaviour and it is regression-tested. It is also the one
thing standing between this stage and a launch.

---

## Non-blocking findings

**F1 — `dpo.py` reads the dataset after identity resolution, not before.**
`load_preference_jsonl` is called at line 288, after `model_identity_for_config`
and the preflight but *before* `from_pretrained`, so a bad `dataset_path` costs
the identity fingerprint rather than a full 14B load. In development mode that is
2 ms; in production mode the fingerprint tier SHA-256s all 8 shards (~15 s/rank).
The launcher's resolver already checks `dataset_path` up front, so this only
bites a direct `python -m kore.policy.dpo` invocation. Not worth a patch, but it
is the reason SFT got an explicit early check and DPO did not need one.

**F2 — the DPO checkpoint is fp32, like SFT's.** 13 shards, 55 GiB, versus the
28 GiB bf16 base. Same root cause as SFT's F1 (accelerate's bf16 mixed precision
keeps an fp32 master; `FULL_STATE_DICT` gathers it). Numerically fine — GRPO
loads with an explicit `torch_dtype` — but it doubles every inter-stage artifact.

**F3 — provision ≥ 800 GB on the DPO output filesystem.** With
`save_total_limit: 2`, the transient peak during rotation is three checkpoints at
once (3 × 221 GB ≈ 663 GB), settling to 2 × 221 GB + 55 GiB ≈ 497 GB. I measured
496 GB for a *finished* 8-step run, which matches. On my box a run that was
killed mid-rotation left 606 GB behind.

**F4 — `dataset_num_proc: 32` spawns 32 tokenizer workers per rank.** On 8 ranks
that is 256 processes racing at startup. It completed fine here (64 rows in 2 s),
and `cpus-per-task=128` gives it room, but it is worth knowing what the startup
spike looks like on a shared node.

**F5 — `_pair_token_stats` renders 512 rows through the chat template on every
rank at startup**, purely for the log line. Cheap (~2 s), but it is duplicated
work across all 8 ranks.

**F6 — resource preflight reports `unresolved`, by design.** Same as SFT's F9: it
cannot join DRM cards to HIP ordinals without `KORE_HIP_INVENTORY_JSON`, logs
`status=unresolved visible_gpus=0`, and warns. Harmless in `report` mode. **Do
not set `KORE_RESOURCE_PREFLIGHT=strict`** — the launcher correctly exports
`report`.

**F7 — the launcher execs bare `accelerate`.** `spur_dpo_1node.sbatch` already
puts the venv on `PATH`, so this is handled; noted because a manual
`launch_distributed.sh` invocation outside the sbatch will fail without it.

---

## Launch command

Once Stage-1 SFT lands at `runs/sft_14b_frontier`:

```bash
export SPUR_CONTROLLER_ADDR=http://crs-m2m-cpu-spur-005:6817
export KORE_SPUR_CONTROLLER_ADDR=$SPUR_CONTROLLER_ADDR   # lets the job requeue itself

cd /home/shasriva/Kore-RL/KORE

# The corpus must already be materialized (1,093,321,730 B, 96,675 lines).
wc -l data/b05factory/dpo/pairs.jsonl        # must print 96675

sbatch scripts/spur_dpo_1node.sbatch \
    configs/dpo_14b_full.json \
    runs/sft_14b_frontier \
    runs/dpo_14b_frontier
```

The launcher resolves the config itself — do not hand it a `*.resolved.json`.
Overriding `model_id` to the SFT output also anchors the frozen reference to the
SFT policy, because `ref_model_id` defaults to `model_id`; that is the intended
recipe and nothing extra is passed.

Expect, in `runs/dpo-<jobid>.out`:

```
[dpo-1node] resume_from=None output_dir=runs/dpo_14b_frontier
policy.dpo: model identity resolved  ... revision_pinned_at_load=False   (correct: a directory)
dpo: dataset loaded  n_pairs=96675 loss_type=['sigmoid','sft'] label_smoothing=0.1
dpo: preference weighting  pairs_in=96675 pairs_effective=167054 weighting=True
```

then **1,305 steps over 13–19 hours**. If `pairs_effective` is not ~167,054, the
weighting was disabled and the step count and budget above do not apply.

Before launching, confirm **≥ 800 GB free** on the output filesystem (F3). Watch
`rewards/chosen` and `logps/chosen` in the `dpo_step` events: if `logps/chosen`
trends down while `rewards/margins` grows, that is the likelihood displacement
that degenerated DPO v1 — stop, raise `beta`, lower the LR, or raise the `sft`
loss weight.

If the run dies, resubmitting the same command resumes from the newest checkpoint
that has a `trainer_state.json`, falling back past a half-written one.

---

## What I could not prove

Stated plainly, because the substitutions above are real limits and not
formalities:

1. **DPO has never trained the actual Stage-1 output.** Every run used
   `Qwen/Qwen3-14B` as a stand-in. The architecture, parameter count, wrap class,
   tokenizer and chat template are identical, and the directory-handoff path is
   separately regression-tested — but the loss values, reward accuracies and
   gradient norms above are from the *base* model, not from an SFT'd one.
   A model that has already been SFT'd on this corpus's chosen responses will
   start with much higher `logps/chosen`, so the numbers above are not a baseline
   to compare the real run against.
2. **No run used the midtrain checkpoint either.** It lives on cluster NFS
   (55 GiB) and the local box has no copy; pulling it would have cost more than
   it proved, given that `Qwen/Qwen3-14B` exercises the identical code path and
   the directory-vs-Hub distinction is already covered by tests.
3. **The production shape was never run.** All measurements are at 2 ranks,
   `max_length: 4096`, micro-batch 1. The 8-rank micro-batch-2 memory and
   throughput figures are extrapolations, flagged as such. The full config
   *parses* and *resolves* correctly, and the memory extrapolation has a very
   wide margin, but I did not put 8 GPUs under the real shape.
4. **Only 8 optimizer steps and 64 pairs.** Nothing here says anything about
   whether 1,305 steps of this recipe converges, or whether likelihood
   displacement appears at step 400. The alarms are wired; they have not been
   stress-tested.
5. **The iterative path was traced and unit-tested, never executed.** It needs a
   generation server, on-policy relabeling through `KoreEnv`, and two full DPO
   rounds — far beyond a verification budget.

---

## Reproducing this report

```bash
pytest tests/test_dpo_launch_readiness.py -q              # 28 pass, no xfail
pytest tests/test_dpo_launch_readiness.py -q -m release   # full 96,675-row corpus
```

The GPU portions (items 4 and 5) are not in the test suite: they need two idle
MI350X and ~700 GB of scratch. The configs used were
`configs/dpo_14b_full.json` with `model_id=Qwen/Qwen3-14B`,
`dataset_path` pointing at a 64-row head of the real corpus,
`max_length: 4096`, `per_device_train_batch_size: 1`,
`gradient_accumulation_steps: 4`, `save_steps: 3`, into
`runs/dpo_14b_verify` (deleted afterwards).
