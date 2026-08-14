# SFT readiness: Qwen3-Coder-30B-A3B production run

**Status: queued, waiting for a node.** The production SFT stage is Slurm job
`11215` (account `amd-primus`, QOS `amd-primus-qos`, a guaranteed,
non-preemptible pool; 3-day walltime), pending on `QOSGrpNodeLimit` because that
pool is capped at 16 nodes and holds 16. An earlier attempt, job `9229` on
`crsuse2-m2m-037`, trained 150 clean steps before its node died; this run starts
fresh from the pinned base rather than warm starting from it (see `model_id` in
docs/DISTRIBUTED.md). This document explains what
the shipped recipe (`configs/sft_coder30b_a3b.json`) trains, why each
non-obvious setting is what it is, and what the held-out evals are watching
for. [`DISTRIBUTED.md`](DISTRIBUTED.md) is the executable launch reference
(FSDP mechanics, the pinned config block, disk arithmetic, launch commands).

Do not resubmit this job. `scripts/spur_sft_1node.sbatch` takes a single-trainer
lock on `output_dir`; a second submission observes the lock and exits without
training, but there is no reason to test that while the run is live.

## Model and hardware

| | |
| --- | --- |
| Model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Revision | `b2cff646eb4bb1d68355c01b18ae02e7cf42d120` |
| Architecture | `Qwen3MoeForCausalLM`, 30.5B total / 3.3B active parameters |
| Routing | 128 experts, top-8 per token, all 48 layers routed, no shared expert |
| Hardware | 1 node, 8x MI355X (gfx950); 236 CPUs, 2.75 TB RAM |

There is no Base sibling of this model, and continued pretraining on the
Instruct checkpoint destroys instruction-following and executable-code output
(`docs/EVAL_RESULTS.md`, the 14B midtrain experiment: 29 of 33 parseable
responses were not valid Python, and correct kernels went 23/34 -> 0/34). SFT
therefore starts directly from the vendor Instruct checkpoint; the midtrain
and chat-vector paths in `kore/policy/configs.py` are 14B-only artifacts and do
not apply here.

## Run shape

206,000 training rows, held out from a 207,782-row build by content hash (899
rows moved to the eval split; 32 truncated math rows replaced from the clean
pool so every eval group is full size).

| | |
| --- | --- |
| Global batch | 128 (`per_device_train_batch_size` 2 x `gradient_accumulation_steps` 8 x 8 GPUs) |
| Optimizer steps / epoch | 1,609 |
| `num_train_epochs` | 1.0 |
| Warmup steps | 241 (`warmup_ratio` 0.15 x 1,609) |
| Checkpoints | every 50 steps, 32 over the run |
| Measured throughput | ~60 s/step, ~29 hours total |

One epoch, not two or three, because this checkpoint is a cold start for RL
rather than a finished model. Note that walltime is NOT the binding reason: at
~29 h/epoch a second epoch (~58 h) still fits inside the 3-day limit, so the
choice rests on what the artifact is for, not on the clock. The held-out eval below (not a guess)
decides whether a second epoch is worth scheduling later.

## Hyperparameters

| Setting | Value |
| --- | --- |
| `learning_rate` | 5e-6 |
| `lr_scheduler_type` | `cosine_with_min_lr`, `min_lr_rate` 0.1 |
| `warmup_ratio` | 0.15 |
| `weight_decay` | 0.0 |
| `max_grad_norm` | 0.5 |
| `adam_beta2` | 0.98 |
| `repair_loss_weight` | 1.0 (off; no row duplication) |
| `packing` | `false` |
| `group_by_length` | `true` |
| `use_lora` | `false` (full fine-tune) |
| `report_to` | `tensorboard` |
| `eval_steps` / `eval_on_start` | 200 / `true` |
| `per_device_eval_batch_size` | 1 |

The min-LR floor holds the schedule at 5e-7 (10% of peak) at step 1,609 instead
of decaying to exactly zero, so the checkpoint arrives as a warm start rather
than a converged endpoint with nothing left to continue from.

## The learning rate, and a correction

The shipped config's own comments used to justify 5e-6 partly by citing an
earlier project finding that learning rate 1e-5 "collapsed instruction
following." That citation is wrong and has been dropped from this document
rather than repeated.

The record it pointed at, `docs/EVAL_RESULTS.md`, evaluates a **14B dense**
model under **continued pretraining** on raw Triton source, cancelled at epoch
1.29. Its failure was output-surface destruction: 29 of 33 parseable
completions were not valid Python, and correct kernels went from 23/34 (base)
to 0/34. That run put gradient on every token of raw code; this one puts
gradient on the 38.6% of tokens inside assistant turns of a chat-formatted
corpus (see below). Different model, different objective, different loss
surface. It does not bound 1e-5 for this recipe, and 1e-5 has never actually
been tried here.

The honest justification is batch scaling. Nemotron-Cascade-2 trains this same
30B-total/3B-active architecture at learning rate 5e-5, but at a batch of
64 x 256k = 16.4M tokens per step against this run's 128 x ~2,379 = ~305k, a
53.5x larger batch.

| Scaling rule | Implied LR from 5e-5 at 53.5x batch |
| --- | --- |
| Square-root | 6.8e-6 |
| Linear | 9.4e-7 |
| **Shipped** | **5e-6 (0.73x the sqrt-scaled value)** |

5e-6 sits inside that band, below the square-root scaling point. The remaining
discount below the sqrt-scaled value is warranted for two reasons: Nemotron's
5e-5 starts from a Base model, while this run starts from Instruct (a smaller
distributional shift, tolerating less aggressive an LR); and this mixture's
replay share is 12.0% of tokens against the 25-30% plateau the forgetting
literature reports as sufficient, so the run is under-replayed and LR is the
compensating lever until more replay is available.

## Assistant-only loss

`kore/policy/sft.py` injects `{% generation %}` markers into the Qwen3 chat
template (`build_assistant_masked_template`) so TRL's `assistant_only_loss`
masks every system/user/tool token to `-100` and trains only on assistant
responses plus their `<|im_end|>` stop token. The masked template is asserted
render-identical to the base template before training starts
(`_verify_assistant_masking`), and TRL itself raises if any row ends up with no
assistant tokens.

This is verified working, not assumed: **38.6% of tokens are in the loss**
(mean 923 supervised tokens per row), so this epoch carries **~189M**
supervised tokens, not the ~490M raw tokens in the corpus. That smaller number
is the one that matters for "is one epoch enough" and for reasoning about the
LR/batch scaling above, both of which are about gradient signal, not raw
corpus size.

## Packing, and why it stays off

`packing` is `false`, and the padding argument for turning it on is a myth this
config corrects. Measured padding waste is **0.5%**, not the ~86% that naively
comparing the mixture's mean length (2,394 tokens) to `max_seq_length`
(17,408) implies. The reason the naive number is wrong is `group_by_length`:
it sorts rows into similar-length batches, and the collator pads to the
longest row **in that batch**, not to the sequence cap.

Packing's real potential win is different: the MoE block runs a Python
for-loop over the experts a token bank hit, and that loop's cost is
independent of sequence length. Packing ~7 documents into one sequence would
amortize the same ~6,144 loop iterations over 7x the useful work. That
requires a flash-attention backend to build the per-document block-diagonal
attention mask, and `flash_attn` is not installed in this venv. On SDPA, TRL's
block-diagonal packing silently falls back to a plain causal mask (it
force-enables padding-free batching under packing, which emits no
`attention_mask` at all), so packed documents would attend across each other
and cross-contaminate. `kore/policy/sft.py` enforces this at runtime
(`packing DISABLED -- attn backend is SDPA`), not just in the config comment.

## MoE load-balancing: deliberately off

The checkpoint ships `router_aux_loss_coef=0.001`, but `Qwen3MoeForCausalLM`
only adds that term when the forward is asked for router logits
(`output_router_logits=True`), which this run leaves `False`. That looks like
an oversight and is not one, for three independent reasons documented in
`kore/policy/sft.py`:

1. **It would add no gradient anyway.** Router logits are captured by a
   monkey-patch on `Qwen3MoeSparseMoeBlock.forward`, which sits inside the
   region wrapped by gradient checkpointing. With `use_reentrant=True` (set
   because the ROCm/SDPA runtime has no fixed saved-tensor count across
   forward and recompute, and non-reentrant checkpointing enforces one), the
   wrapped forward runs under `torch.no_grad()`, so every captured logit
   tensor is detached. Measured: router-gate gradient norms are byte-identical
   with the flag on and off. The aux term would still land in the *reported*
   loss as a constant +0.008, breaking comparability with earlier runs, and
   cost up to ~40 GB of transient VRAM at the longest length band
   (`load_balancing_loss_func` materializes an int64 one-hot over
   `num_layers x B x S x top_k x num_experts`).
2. **If the gradient were live, it would be 8x over-weighted.** The
   cross-entropy term is normalized by a global token count and survives
   FSDP/accumulation scaling exactly; the aux term is a per-micro-batch mean
   with no such denominator, so at `gradient_accumulation_steps=8` it would
   land at 8x the configured coefficient.
3. **Even correctly scaled, it would be the wrong loss.** Qwen3 was pretrained
   with a *global-batch* balancing loss. `transformers` computes the
   *micro-batch* variant, which at `per_device_train_batch_size=2` is
   effectively per-sequence. Qiu et al. (ACL 2025, the paper behind Qwen3's own
   choice of global-batch LBL) show that regime forces even domain-specific
   sequences to spread load across all experts, measurably suppressing the
   expert specialization this run wants on an 86%-kernel corpus.

What was missing was not pressure on the router but *visibility* into it.
Forward hooks on every `Qwen3MoeSparseMoeBlock`, gated on
`not torch.is_grad_enabled()` (so the reentrant-checkpoint backward recompute,
which runs *with* grad enabled, is not double-counted), accumulate per-layer
expert selection counts. They reset once per optimizer step, not per
micro-batch, so the reported load is the global-batch signal that actually
predicts specialization. Each logging step reports, per watched layer
(0, 24, 47): normalized load entropy, max-expert share, and dead-expert count,
with a collapse alarm when any watched layer's entropy falls below 0.85x its
step-0 value.

## Warmup: what it protects, corrected

`warmup_ratio` is 0.15 (241 steps), well above the 0.03 dense-model default.
The mechanism is router stability: the router is a small linear layer whose
top-8-of-128 selection decides which experts see gradient on a given token,
and a large early step can concentrate that selection onto a subset of experts
before the loss has said anything useful.

An earlier version of this reasoning claimed that experts the router stops
selecting "receive no gradient to recover with." That claim is wrong. The gate
is a softmax over all 128 logits, so every expert's *router logit* keeps
receiving gradient through the normalization whether or not that expert is
selected. What actually freezes when an expert stops being selected is that
expert's *FFN weights* (`gate_proj`/`up_proj`/`down_proj`), because no token
routed to it means no backward pass through it. For a 1,609-step
narrow-domain SFT, leaving most experts' FFN weights close to their pretrained
values is arguably desirable, not a failure mode: it limits how much of the
model's general capability this specialization run can overwrite. Warmup is
still the right lever against router collapse; it is no longer justified by a
mechanism that does not exist. (`tests/test_sft_readiness.py` still carries the
old, incorrect phrasing in a docstring; see the defects list in this repo
review.)

## Retention monitoring

A held-out slice of 899 rows is loaded as one dataset **per capability group**
(`load_eval_datasets`), so the trainer logs `eval_<group>_loss` separately
instead of one pooled number that could hide "kernels improved, chat
collapsed" inside "nothing moved." `eval_on_start=True` gives a baseline on the
untrained model:

| Group | Step-0 loss |
| --- | --- |
| `kernel_repair` | 0.392 |
| `kernel_optimize` | 0.561 |
| `instruction_following` | 0.633 |
| `kernel_generate` | 0.691 |
| `math` | 0.768 |
| `general_code` | 0.957 |
| `tool_use` | 1.152 |
| `chat` | 1.182 |

The alarm (`_ObsCallback._on_eval_log`) does not fire on an absolute loss
threshold. It fires on the *shape* of forgetting: any `kernel*` group's loss
falling more than 2% from its own baseline while any non-kernel group's loss
*climbs* more than 5% from its own baseline. Taught capabilities are supposed
to improve while retained ones drift a little; it is the two moving in
opposite directions that defines catastrophic forgetting, and alarming on that
shape is what makes it readable at step 200 instead of after the run ends.

## Divergence cannot hide in the log

`logging_nan_inf_filter` is set `False`, against the Hugging Face default of
`True`. With the default on, a NaN or Inf micro-batch loss is silently
replaced in the logged running average by the running mean, so a run that is
actively diverging reports a clean loss curve while the bad gradient still
lands in the weights. With it off, `_ObsCallback` logs a warning the moment a
non-finite `loss` or `grad_norm` appears, and a separate check flags a
grad-norm spike against the running median (not the mean, so one large step
cannot hide the next one behind it).

## Checkpoint disk footprint

A full checkpoint (bf16 weights plus fp32 optimizer state) is measured at
**456 GB**. `save_total_limit: 2` keeps two on disk at steady state
(~912 GB); because the trainer writes the new checkpoint before rotating the
old one out, the rotation window transiently holds three (~1.37 TB peak).
`save_only_model` stays `false`: dropping optimizer state would shrink the
write 8x but makes every future kill unrecoverable, and this run has a whole
node and a guaranteed QOS, not a disk-pressure problem to trade against. See
[`DISTRIBUTED.md`](DISTRIBUTED.md) for where those bytes live and the FSDP
mechanics of the write.

## Regression contract

- `tests/test_sft_readiness.py`: every key in `configs/sft_coder30b_a3b.json`
  is a recognized `SFTConfig` field or identity/preflight key; the MoE warmup
  and gradient-clip bands; packing stays off without flash-attention;
  checkpoint rotation stays bounded to 2-3 copies at `save_steps <= 50`.
- `tests/test_sft_launch_readiness.py`: the shared masking, model-identity, and
  FSDP-wiring machinery, exercised against `configs/sft_14b_full.json`.
  Currently **37 pass** in the default suite plus one `release`-marked
  full-corpus check; the count is pinned by `tests/test_docs_contract.py`, so
  it cannot drift without an explicit edit here. That suite predates the 30B
  recipe and is a regression guard on shared code paths (`kore/policy/sft.py`,
  `kore/policy/model_spec.py`), not a readiness claim about this run.
- `tests/test_docs_contract.py`: mechanically pins the JSON block in
  [`DISTRIBUTED.md`](DISTRIBUTED.md) to `configs/sft_coder30b_a3b.json`, key
  for key.
