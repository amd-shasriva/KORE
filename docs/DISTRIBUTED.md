# Distributed SFT for the production 30B MoE

Production SFT uses `configs/sft_coder30b_a3b.json` with PyTorch FSDP on eight
MI355X GPUs. The target is instruct-only, so the supported path is directly
from the pinned Hub model to SFT. Do not route it through the 14B midtrain,
DPO, or chat-vector launchers: they are legacy experiments and their artifacts
are not inputs to this recipe. See [`SFT_READINESS.md`](SFT_READINESS.md) for
why each hyperparameter is what it is; this page is the launch mechanics.

**A run is live.** Job `9229` on `crsuse2-m2m-037` (account `amd-primus`, QOS
`amd-primus-qos`, a guaranteed non-preemptible pool, 7-day walltime) is
training against this exact config. Do not `sbatch`, `scancel`, or `squeue`
against it; the commands below are the launch reference, not an instruction to
relaunch.

## Hardware and topology

| | |
| --- | --- |
| Node | 1, 8x MI355X (gfx950) |
| CPUs | 236 |
| Host RAM | 2.75 TB |
| Sharding | FSDP `full_shard` (ZeRO-3 equivalent): params, gradients, and optimizer state all sharded |

The MoE wrap class is explicit (`fsdp_transformer_layer_cls:
"Qwen3MoeDecoderLayer"`). Wrapping anything other than `Qwen3MoeDecoderLayer`
can shard the 128 experts at the wrong granularity, or leave whole layers
unsharded and out-of-memory; `kore/policy/configs.py::detect_transformer_layer_cls`
also derives this from the model id as a fallback, but the launch config names
it explicitly rather than depending on that inference alone.

## Shipped configuration

The block below is a transcription of the non-comment fields in
`configs/sft_coder30b_a3b.json`, and `tests/test_docs_contract.py` asserts it
key-for-key against the live file, so it cannot silently drift.

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
  "logging_dir": "/shared_nfs/shasriva/kore/runs/sft_coder30b_a3b_v5/tb",
  "save_steps": 50,
  "save_total_limit": 2,
  "save_only_model": false,
  "fsdp": "full_shard auto_wrap",
  "fsdp_transformer_layer_cls": "Qwen3MoeDecoderLayer",
  "fsdp_cpu_offload": false,
  "dataloader_num_workers": 2,
  "dataset_num_proc": 16,
  "dataloader_pin_memory": false,
  "dataloader_prefetch_factor": 2
}
```

`dataloader_num_workers` is 2, not the `DistributedMixin` default of 4: it was
lowered together with the sbatch's `--cpus-per-task` (128 -> 32), the change
that let the job be scheduled at all on a CPU-contended cluster. At 8 ranks x
2 workers that is 16 loader processes against 32 allocated CPUs, comfortable
headroom; the GPU-side MoE expert loop dominates step time, not host-side
batch assembly, so the lower worker count costs no measurable throughput.

`dataloader_pin_memory` is `false` and `dataloader_prefetch_factor` is 2 for a
host-memory reason documented in `kore/policy/configs.py`: pinned pages are
not reclaimable when the process needs memory elsewhere, and a
`FULL_STATE_DICT` checkpoint save gathers the whole model+optimizer state onto
the host at once. An earlier 14B run exhausted host RAM at step ~492 from the
combination; these defaults are sized to not repeat it.

## Accelerate / FSDP mechanics

`configs/accelerate_fsdp.yaml` drives `accelerate launch` for this run:

| Setting | Value | Why |
| --- | --- | --- |
| `fsdp_reshard_after_forward` | `FULL_SHARD` | ZeRO-3 equivalent: params, grads, and optimizer state are all sharded. Unlike GRPO's co-located-rollout path, SFT does no generation, so full resharding costs nothing it needs back. |
| `fsdp_auto_wrap_policy` | `TRANSFORMER_BASED_WRAP` | Wraps one `Qwen3MoeDecoderLayer` per FSDP unit. |
| `mixed_precision` | `bf16` | Compute in bf16; `accelerate`'s FSDP prepare step still upcasts every trainable flat-parameter to fp32 (DeepSpeed-ZeRO-style), so the optimizer steps on an fp32 master regardless. Loading the model in fp32 would only double load-time host memory for an identical result. |
| `fsdp_state_dict_type` | `FULL_STATE_DICT` | Consolidates a plain HF checkpoint so the next stage (GRPO) can load it with `from_pretrained`; a sharded state dict is only reloadable under an identical FSDP mesh. |
| `fsdp_activation_checkpointing` | `false` | Activation checkpointing is enabled on the model directly (`gradient_checkpointing_kwargs={"use_reentrant": True}`), not through the FSDP plugin. The plugin's external `checkpoint_wrapper` mismatches saved-tensor counts on an FSDP1/`use_orig_params` unit and raises `CheckpointError`. |
| `fsdp_offload_params` | `false` | Kept on-device; the model comfortably fits the 8-GPU mesh. |

Activation checkpointing uses **reentrant** checkpointing specifically because
the ROCm stack has no `flash_attn` wheel, so training runs on SDPA. SDPA
transparently switches between fused kernels (flash / mem-efficient / math)
depending on shape and free memory, and that choice can differ between the
checkpointed forward and its recompute. Non-reentrant checkpointing enforces a
fixed saved-tensor count across that boundary and raises `CheckpointError` on
the mismatch; reentrant checkpointing skips that check and tolerates the swap.
`kore/policy/configs.py::build_fsdp_kwargs` is the single source of truth for
this choice; `kore/policy/sft.py` matches it.

## Run shape and disk arithmetic

206,000 training rows at global batch 128 is **1,609 optimizer steps** for one
epoch (`ceil(206000/8 ranks) = 25,750` per rank; `ceil(25750/2 micro) = 12,875`
micro-batches; `12875 // 8 accumulation = 1,609`). Measured throughput is
~60 s/step, giving roughly 29 hours end to end; a prior estimate of 478 steps
and 12-16 hours on this page described an earlier, smaller mixture and no
longer applies.

A full checkpoint (bf16 weights + fp32 optimizer state, `FULL_STATE_DICT`
gathered to rank 0) is measured at **456 GB**. `save_total_limit: 2` holds two
at steady state (~912 GB); the trainer writes the new checkpoint *before*
rotating the old one out, so the rotation window transiently holds three
(~1.37 TB peak). `/shared_nfs` is the target for exactly this reason: the
model-relative volume this project has used before could not hold that peak.

`save_steps: 50` over 1,609 steps yields 32 checkpoints. That frequency is not
about preemption risk here (the job holds a guaranteed, non-preemptible QOS
allocation), it bounds how much work a genuine crash costs: at 50 steps and
~60 s/step, at most ~50 minutes of training is ever at risk between saves.

## Launch boundary

`scripts/launch_distributed.sh` starts FSDP processes and must run inside an
allocated GPU node:

```bash
PYTHONPATH=. bash scripts/launch_distributed.sh sft \
  configs/sft_coder30b_a3b.json --dry-run
```

The live run was started through `scripts/sft_supervise_v5.sh`, which wraps
`scripts/watch_and_resume.sh` around `scripts/spur_sft_1node.sbatch` with the
account/QOS pairing that actually schedules on this cluster:

```bash
export SPUR_CONTROLLER_ADDR="http://crs-m2m-cpu-spur-005:6817"
sbatch --account=amd-primus --qos=amd-primus-qos \
  scripts/spur_sft_1node.sbatch \
  configs/sft_coder30b_a3b.json - -
```

`-` for the model-source and output-dir arguments tells
`scripts/spur_resolve_launch_config.py` to leave the config's own `model_id`
and `output_dir` alone, which keeps the vendor Instruct checkpoint as the
starting point and matches the resumable `output_dir` above across a
requeue. The login node is not a scheduler controller; exporting
`SPUR_CONTROLLER_ADDR` before `sbatch`/`squeue` prevents a false
"controller down" diagnosis.

`scripts/spur_sft_1node.sbatch` takes an NFS-`mkdir`-based single-trainer lock
on `output_dir` before training starts, so a second job submitted against the
same config observes the lock, prints `KORE_LOCK_HELD=<jobid>`, and exits
without writing anything. This is what makes the recovery path (a fresh
`sbatch` after a kill, or a second queued backup job) safe: at most one
process ever trains into this `output_dir` at a time.
