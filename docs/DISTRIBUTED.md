# Distributed full fine-tuning (FSDP) for KORE

KORE's SFT and DPO stages support **real distributed full fine-tuning** via
PyTorch FSDP (`full_shard` == ZeRO-3 equivalent), so full-FT actually runs at
14B / 32B / 70B on 8× MI300-class GPUs (gfx942). This replaces the old
`device_map="auto"` shortcut, which only ever pipelines a single process across
GPUs and cannot train a 14B+ model.

- **LoRA is unchanged.** The distributed path is only for full-FT
  (`use_lora=false`). LoRA (and single-process / CPU) runs keep the exact legacy
  path, including merge-on-save.
- **FSDP only engages when both** `use_lora=false` **and** `distributed=true`
  (the launcher sets `distributed=true` for you). Otherwise `build_fsdp_kwargs`
  returns `{}` and nothing changes.

## How full-FT now runs

Launch with `accelerate` + the FSDP config (the launcher wraps this):

```bash
# SFT full-FT on 8 GPUs
scripts/launch_distributed.sh sft configs/sft_14b_full.json

# DPO full-FT on 8 GPUs
scripts/launch_distributed.sh dpo configs/dpo_14b_full.json --nproc 8
```

which expands to:

```bash
PYTHONPATH=<repo> accelerate launch \
    --config_file configs/accelerate_fsdp.yaml \
    -m kore.policy.sft configs/sft_14b_full.json
```

`kore.policy.sft` / `kore.policy.dpo` each expose a `__main__` entry that reads
the JSON config, defaults `distributed=true`, and calls `train_sft` /
`dpo.train`. Under FSDP the model is loaded **without** `device_map` (the two are
incompatible — accelerate/FSDP owns device placement); the HF `Trainer` wraps it
with `TrainingArguments(fsdp=..., fsdp_config=...)` built from the config.

### Example training config (`configs/sft_14b_full.json`)

```json
{
  "model_id": "Qwen/Qwen3-14B",
  "dataset_path": "data/sft/multicap.jsonl",
  "output_dir": "runs/sft_14b_full",
  "use_lora": false,
  "bf16": true,
  "gradient_checkpointing": true,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "max_seq_length": 16384,
  "fsdp": "full_shard auto_wrap",
  "fsdp_transformer_layer_cls": null,
  "fsdp_cpu_offload": false
}
```

`fsdp_transformer_layer_cls: null` auto-detects the decoder block from
`model_id` (Qwen3 → `Qwen3DecoderLayer`, DeepSeek-R1-Distill-Qwen → `Qwen2DecoderLayer`,
DeepSeek-R1-Distill-Llama → `LlamaDecoderLayer`).

## What the config → TrainingArguments wiring produces

`kore.policy.configs.build_fsdp_kwargs(config)` returns (for full-FT + distributed):

```python
{
  "fsdp": "full_shard auto_wrap",           # + " offload" if fsdp_cpu_offload
  "fsdp_config": {
    "transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
    "activation_checkpointing": True,       # from gradient_checkpointing
    "backward_prefetch": "backward_pre",
    "forward_prefetch": False,
    "use_orig_params": True,
    "sync_module_states": True,
    "cpu_ram_efficient_loading": True,
    "limit_all_gathers": True,
    "state_dict_type": "SHARDED_STATE_DICT",
  },
}
```

Under `full_shard`, activation checkpointing is routed through `fsdp_config`
(not `TrainingArguments.gradient_checkpointing`) to avoid a redundant AllGather
in the backward pass, so the trainer sets `gradient_checkpointing=False` in that
path and lets FSDP own it.

## Per-size memory guidance & accelerate configs

Rules of thumb for a full-FT run in bf16 with AdamW (~16 bytes/param of optimizer
+ master state, sharded across N ranks under FULL_SHARD):

| Model | GPUs | Sharding | Offload | Notes |
|-------|------|----------|---------|-------|
| 14B   | 8×MI300 | `full_shard auto_wrap` | none | Fits comfortably; activation checkpointing on. |
| 32B   | 8×MI300 | `full_shard auto_wrap` | optimizer/param CPU offload (or +nodes) | Tight without offload at long `max_seq_length`; enable offload or shrink seq len / grad-accum. |
| 70B   | 8×MI300 (min) → multi-node | `full_shard auto_wrap offload` | CPU param+grad+optimizer offload; prefer 2+ nodes | Single-node 8-GPU is borderline; multi-node is the safe path. Wrap class = `LlamaDecoderLayer`. |

### 14B — `configs/accelerate_fsdp.yaml` as shipped (no offload)

Use the shipped file directly:

```bash
scripts/launch_distributed.sh sft configs/sft_14b_full.json
```

Key `fsdp_config` bits: `fsdp_reshard_after_forward: FULL_SHARD`,
`fsdp_offload_params: false`, `fsdp_activation_checkpointing: true`,
`fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer`.

### 32B — full_shard + optimizer/param CPU offload

Copy the shipped yaml and flip offload on (or add nodes). Training config sets
`"fsdp_cpu_offload": true` (which appends `offload` to the `fsdp` string), and the
accelerate yaml:

```yaml
distributed_type: FSDP
mixed_precision: bf16
num_machines: 1
num_processes: 8
fsdp_config:
  fsdp_version: 1
  fsdp_reshard_after_forward: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer   # Qwen3-32B
  fsdp_offload_params: true            # <-- CPU offload params+grads+optim
  fsdp_activation_checkpointing: true
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: true
```

For `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` set
`fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer` (and either leave
`fsdp_transformer_layer_cls: null` in the JSON to auto-detect, or set it
explicitly).

### 70B — full_shard + CPU offload, multi-node preferred

`deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (Llama arch). Single 8×MI300 node is
borderline even with full offload; use 2+ nodes when possible. Training config:
`"model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"`, `"use_lora": false`,
`"fsdp_cpu_offload": true`, `"fsdp_transformer_layer_cls": "LlamaDecoderLayer"`.

Single-node (8 GPUs) accelerate yaml:

```yaml
distributed_type: FSDP
mixed_precision: bf16
num_machines: 1
num_processes: 8
fsdp_config:
  fsdp_version: 1
  fsdp_reshard_after_forward: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer   # 70B is Llama arch
  fsdp_offload_params: true
  fsdp_activation_checkpointing: true
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: true
```

Multi-node (e.g. 2 nodes × 8 GPUs = 16 ranks): set `num_machines: 2`,
`num_processes: 16`, and per-node `machine_rank` (0 and 1), plus
`main_process_ip` / `main_process_port` on the workers. Launch the same
`scripts/launch_distributed.sh sft <config.json> --nproc 16` on each node with the
node-specific `machine_rank`, or drive it through your cluster's `accelerate
launch --multi_gpu --machine_rank ...` wrapper.

## Checkpointing

`fsdp_state_dict_type: SHARDED_STATE_DICT` writes sharded checkpoints (scalable
for 32B/70B). The trainer's `save_model` / merge-on-save behavior is unchanged
for LoRA; for full-FT the sharded state dict is consolidated by the HF Trainer on
save. To reload into a single-file model for downstream stages (DPO/GRPO/soup),
point those stages at the produced `output_dir` as usual.

## Quick sanity checks (no real training)

```bash
# import-level + wiring unit tests (CPU only)
PYTHONPATH=. python -m pytest tests/test_distributed.py -q

# launcher dry-run (prints the accelerate command, does not train)
bash scripts/launch_distributed.sh sft configs/sft_14b_full.json --dry-run
```
