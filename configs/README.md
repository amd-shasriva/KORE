# `configs/`

`sft_coder30b_a3b.json` is the production training recipe. It pins
`Qwen/Qwen3-Coder-30B-A3B-Instruct` to a Hub revision, uses full-parameter FSDP
with `Qwen3MoeDecoderLayer`, and starts from the instruct checkpoint rather than
a CPT artifact.

The operational constraints are encoded in the config comments because they are
easy to undo accidentally:

- `max_seq_length: 17408` matches the mixture admission limit.
- One epoch is about 478 steps at 90–120 seconds per step and fits the
  23-hour allocation; a second epoch does not.
- A checkpoint is about 488 GB. `save_total_limit: 1` is mandatory: normal
  rotation already has two checkpoints on disk (~976 GB), while limit 2 would
  exceed the shared volume.
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
