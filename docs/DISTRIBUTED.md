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
  "dataset_path": "/home/shasriva/Kore-RL/KORE/data/v5_sft.jsonl",
  "eval_dataset_path": "/home/shasriva/Kore-RL/KORE/data/v5_eval.jsonl",
  "eval_steps": 200,
  "eval_on_start": true,
  "per_device_eval_batch_size": 1,
  "output_dir": "/shared_nfs/shasriva/kore/runs/sft_coder30b_a3b_v5",
  "use_lora": false,
  "distributed": true,
  "bf16": true,
  "gradient_checkpointing": true,
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "max_seq_length": 17408,
  "num_train_epochs": 1.0,
  "learning_rate": 5e-06,
  "lr_scheduler_type": "cosine_with_min_lr",
  "lr_scheduler_kwargs": {"min_lr_rate": 0.1},
  "warmup_ratio": 0.15,
  "max_grad_norm": 0.5,
  "packing": false,
  "repair_loss_weight": 1.0,
  "adam_beta2": 0.98,
  "report_to": "tensorboard",
  "save_steps": 50,
  "save_total_limit": 2,
  "save_only_model": false,
  "fsdp": "full_shard auto_wrap",
  "fsdp_transformer_layer_cls": "Qwen3MoeDecoderLayer",
  "fsdp_cpu_offload": false,
  "dataloader_num_workers": 4,
  "dataloader_pin_memory": false,
  "dataloader_prefetch_factor": 2
}
```

The 17,408-token cap matches the mixture filter; raising it would add no
admitted data. One epoch is **1,613 optimizer steps** (206,586 rows / global batch
128). Earlier revisions of this page said 478 steps and 12–16 hours; both were
wrong — 478 by a factor of about four, and the walltime because no step time has
ever been measured on the real length distribution rather than on artificially
maximal sequences. Expect 26–36 hours and at least one preemption, and measure
`train_samples_per_second` over the first 50 steps before believing any estimate
here. A second epoch would cross the 23-hour walltime, so it is deferred until
the held-out eval says it is necessary.

`save_total_limit: 2` replaces an earlier `1`, and the reasoning inverted. A 30.5B
Adam checkpoint is about 488 GB and the Trainer writes the new one *before*
rotating the old one out, so limit 2 peaks near 1.46 TB. The old justification for
1 was that the volume had roughly 1,090 GB free and could not hold two; it
actually has 42 TB, making the peak 3.5% of free space. Limit 1 was the more
dangerous setting, because its rotation window contains no complete checkpoint at
all, and this run is expected to be preempted enough times to eventually be killed
inside one.

`eval_dataset_path` holds out 899 rows (0.43% of the mixture), removed from
training by content hash so the mixture's deliberate duplicates cannot leave a
copy behind. Rows are tagged by capability and loaded as one dataset per group, so
the log carries `eval_instruction_following_loss` separately from
`eval_kernel_generate_loss`. With `eval_on_start` the baseline is the untrained
model, and the trainer warns when kernel loss falls while retained capabilities
climb — the signature of catastrophic forgetting, which was previously invisible
until the run finished.

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
  configs/sft_coder30b_a3b.json - /shared_nfs/shasriva/kore/runs/sft_coder30b_a3b
```

`-` tells the resolver to keep the config's `model_id`, preventing a legacy stage
checkpoint from replacing the instruct starting point. The sbatch wrapper still
has 14B defaults; production invocations must pass all three arguments until it
is migrated or retired.
