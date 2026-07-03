# Distributed full fine-tuning (FSDP) for KORE

KORE's training stages support **real distributed full fine-tuning** via
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

## Full-FT is ONE command: `run_campaign.py --full-ft`

Full fine-tuning at 14B / 32B / 70B is a **documented one-command path** — you do
**not** write a config or invoke `accelerate` yourself:

```bash
# Full best-in-world 14B run, single command. The campaign spawns the FSDP
# processes under the hood (LoRA is the default; --full-ft opts into full-FT).
PYTHONPATH=. python scripts/run_campaign.py --model Qwen/Qwen3-14B \
    --tasks rmsnorm_aiter,gemm_bf16,flash_attn_decode_bf16 \
    --teacher claude --full-ft
```

When you pass `--full-ft`, the campaign:

1. sets `distributed=true` on **every** training config (midtrain / sft / dpo / grpo); and
2. for each training stage whose `-m kore.policy.<stage> <config.json>` entry can
   read a JSON config, **shells out under the hood** to
   `scripts/launch_distributed.sh <stage> <resolved.json>`, which runs
   `accelerate launch --config_file configs/accelerate_fsdp.yaml`. The campaign
   renders `<resolved.json>` into `<data_root>/launch/` from the shipped internal
   template (`configs/<stage>_14b_full.json`) overlaid with the run's dynamic
   paths (model / dataset / output_dir) — these templates are **internal**, not
   something you author.

For 32B / 70B pass `--model Qwen/Qwen3-32B` (or the 70B id); the per-size
`accelerate` config (offload / multi-node) is described below. The one-command
`--full-ft` invocation is identical — only `--model` (and, for the biggest sizes,
the shipped `accelerate_fsdp.yaml` offload knobs) changes.

### <a name="full-ft-per-stage-status"></a>Full-FT per-stage status

The launcher accepts all four stages (`midtrain|sft|dpo|grpo`), and the campaign
routes each to the launcher **only if** that stage exposes a JSON `-m` entry
(detected via a `<stage>_config_from_dict` builder, so it flips on automatically
the moment the entry ships — no campaign change needed):

| Stage | JSON `-m` entry today | `--full-ft` behavior |
|-------|-----------------------|----------------------|
| `sft`  | ✅ `kore.policy.sft`  | Shells out to the FSDP launcher (real full-FT). |
| `dpo`  | ✅ `kore.policy.dpo`  | Shells out per pass/round to the FSDP launcher (IPO + refreshed ref travel in the JSON). |
| `midtrain` | ❌ (sibling-owned) | Runs **in-process with a LOUD warning** — `kore.policy.midtrain`'s `-m` entry parses `--flags`, not a JSON config, and its full-FT path currently defers, so no sharded checkpoint is produced. `distributed=true` is still set. |
| `grpo` | ❌ (sibling-owned) | Runs **in-process with a LOUD warning** — `kore.policy.grpo` has no `__main__` JSON entry, so it cannot shard a 14B. `distributed=true` is still set. |

This is deliberate: the campaign **never silently degrades**. For `midtrain`/`grpo`
it prints a loud warning naming the missing entry and pointing here, rather than
pretending to shard. When the sibling track ships
`kore.policy.grpo` / `kore.policy.midtrain` JSON `-m` entrypoints (each reading a
`<config.json>` and exposing `<stage>_config_from_dict`), those stages become
one-command full-FT automatically too.

Until then, to full-FT `midtrain`/`grpo` manually, run the sharded launcher
directly with the shipped config (this is the exact command the campaign will
issue once the entry lands):

```bash
scripts/launch_distributed.sh midtrain configs/midtrain_14b_full.json
scripts/launch_distributed.sh grpo     configs/grpo_14b_full.json
```

## How a launcher-driven stage runs

`scripts/launch_distributed.sh sft configs/sft_14b_full.json` expands to:

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
PYTHONPATH=. python -m pytest tests/test_distributed.py tests/test_campaign_wiring.py -q

# whole-campaign wiring preflight (no GPU/teacher; import-checks every symbol)
PYTHONPATH=. python scripts/run_campaign.py --dry-run --tasks rmsnorm_aiter,gemm_bf16

# launcher dry-run for any stage (prints the accelerate command, does not train)
bash scripts/launch_distributed.sh midtrain configs/midtrain_14b_full.json --dry-run
bash scripts/launch_distributed.sh sft      configs/sft_14b_full.json      --dry-run
bash scripts/launch_distributed.sh dpo      configs/dpo_14b_full.json      --dry-run
bash scripts/launch_distributed.sh grpo     configs/grpo_14b_full.json     --dry-run
```

The shipped internal full-FT config templates are
`configs/{midtrain,sft,dpo,grpo}_14b_full.json` (all `use_lora=false`,
`distributed=true`). The campaign overlays the run's model/dataset/output paths
onto these before launching, so you never edit them by hand.
