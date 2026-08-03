# Distributed SFT for the production 30B MoE

Production SFT uses `configs/sft_coder30b_a3b.json` with PyTorch FSDP on eight
MI355X GPUs. The target is instruct-only, so the supported path is directly from
the pinned Hub model to SFT. Do not route it through the 14B midtrain, DPO, or
chat-vector launchers: they are legacy experiments and their artifacts are not
inputs to this recipe.

The MoE wrap class is explicit. Wrapping anything other than
`Qwen3MoeDecoderLayer` can shard its 128 experts at the wrong granularity and
either leave full layers resident or create pathological collectives.

## Shipped configuration

The block below is a transcription of the non-comment fields in
`configs/sft_coder30b_a3b.json`.

```json
{
  "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
  "model_revision": "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
  "dataset_path": "data/b05factory/sft/multicap_v2.jsonl",
  "output_dir": "runs/sft_coder30b_a3b",
  "use_lora": false,
  "distributed": true,
  "bf16": true,
  "gradient_checkpointing": true,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "max_seq_length": 17408,
  "num_train_epochs": 1.0,
  "learning_rate": 5e-06,
  "lr_scheduler_type": "cosine",
  "warmup_ratio": 0.03,
  "repair_loss_weight": 2.0,
  "save_steps": 400,
  "save_total_limit": 1,
  "fsdp": "full_shard auto_wrap",
  "fsdp_transformer_layer_cls": "Qwen3MoeDecoderLayer",
  "fsdp_cpu_offload": false,
  "dataloader_num_workers": 4,
  "dataloader_pin_memory": false,
  "dataloader_prefetch_factor": 2
}
```

The 17,408-token cap matches the mixture filter; raising it would add no
admitted data. Measured SFT is about 478 steps at 90–120 seconds per step, or
roughly 12–16 hours. A second epoch would cross the 23-hour walltime and force a
checkpoint resume, so it is deferred until evaluation says it is necessary.

`save_total_limit: 1` is not a durability preference. A 30.5B Adam checkpoint is
about 488 GB; rotation temporarily holds two (~976 GB), which fits the shared
volume's roughly 1,090 GB free space. Limit 2 would require about 1,464 GB and
fail mid-run. The saved checkpoint is therefore the single recoverable
generation.

## Launch boundary

`scripts/launch_distributed.sh` starts FSDP processes and must run inside an
allocated GPU node:

```bash
PYTHONPATH=. bash scripts/launch_distributed.sh sft \
  configs/sft_coder30b_a3b.json --dry-run
```

On SPUR, submit from a shell with the controller address set. The login node has
no controller, so omitting this variable makes a healthy controller look
unavailable:

```bash
export SPUR_CONTROLLER_ADDR="http://crs-m2m-cpu-spur-005:6817"
sbatch scripts/spur_sft_1node.sbatch \
  configs/sft_coder30b_a3b.json - runs/sft_coder30b_a3b
```

`-` tells the resolver to keep the config's `model_id`, preventing a legacy stage
checkpoint from replacing the instruct starting point. The sbatch wrapper still
has 14B defaults; production invocations must pass all three arguments until it
is migrated or retired.
