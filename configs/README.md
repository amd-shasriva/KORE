# `configs/`

`sft_coder30b_a3b.json` is the production training recipe. It pins
`Qwen/Qwen3-Coder-30B-A3B-Instruct` to a Hub revision, uses full-parameter FSDP
with `Qwen3MoeDecoderLayer`, and starts from the instruct checkpoint rather than
a CPT artifact.

The operational constraints are encoded in the config comments because they are
easy to undo accidentally:

- `max_seq_length: 17408` matches the mixture admission limit.
- One epoch is **1,609 steps** (206,000 rows at a global batch of 128), not
  478. An earlier revision of this file said 478; the config's own
  `_comment_epochs_and_lr` records that figure as wrong, left over from a
  `repair_loss_weight`-inflated row count that no longer applies. One epoch
  does not fit a 23-hour allocation, which is why the launcher requests seven
  days instead; see [`docs/CLUSTER_OPERATIONS.md`](../docs/CLUSTER_OPERATIONS.md)
  for the measured walltime and the reasoning.
- A checkpoint is about 488 GB. `save_total_limit: 2`, not 1: the Trainer
  writes the new checkpoint before rotating the old one out, so normal
  rotation transiently holds three checkpoints (~1.46 TB) against 42 TB free
  on `/shared_nfs` (3.5%). Limit 1 was the riskier setting for a run that
  expects preemption, because its rotation window holds no complete
  checkpoint at all; limit 2 is what makes a kill mid-rotation resumable.
- The explicit MoE transformer layer avoids incorrect FSDP wrapping of experts.

`accelerate_fsdp.yaml` is the shared Trainer launcher configuration.
`accelerate_fsdp_grpo.yaml` is the GRPO topology; it uses `SHARD_GRAD_OP`
because generation repeatedly gathers parameters under `FULL_SHARD`.

The `*_14b_*.json` and `grpo_32b_min_trustworthy.json` files remain for
reproducible legacy experiments and their tests. They are not production launch
templates. In particular, the 14B midtrain, residual transfer, and DPO recipes
cannot be generalized to the instruct-only 30B target.

Use [`docs/DISTRIBUTED.md`](../docs/DISTRIBUTED.md) for the documented
production launch; do not infer a recommended command from a legacy config name.
