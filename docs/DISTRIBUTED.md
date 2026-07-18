# Distributed full fine-tuning (FSDP) for KORE

KORE's training stages support **real distributed full fine-tuning** via
PyTorch FSDP (`full_shard` == ZeRO-3 equivalent), so full-FT actually runs at
14B / 32B / 70B on **8x AMD Instinct MI350X GPUs (gfx950 / CDNA4, ~270 GB HBM3E
per GPU)**, which is the sole KORE target. This replaces the old
`device_map="auto"` shortcut, which only ever pipelines a single process across
GPUs and cannot train a 14B+ model.

- **LoRA is unchanged.** The distributed path is only for full-FT
  (`use_lora=false`). LoRA (and single-process / CPU) runs keep the exact legacy
  path, including merge-on-save.
- **FSDP only engages when both** `use_lora=false` **and** `distributed=true`
  (the launcher sets `distributed=true` for you). Otherwise `build_fsdp_kwargs`
  returns `{}` and nothing changes.

## Full-FT is ONE command: `run_campaign.py --full-ft`

Full fine-tuning at 14B / 32B / 70B is a **documented one-command path** - you do
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
   paths (model / dataset / output_dir) - these templates are **internal**, not
   something you author.

For 32B / 70B pass `--model Qwen/Qwen3-32B` (or the 70B id); the per-size
`accelerate` config (offload / multi-node) is described below. The one-command
`--full-ft` invocation is identical - only `--model` (and, for the biggest sizes,
the shipped `accelerate_fsdp.yaml` offload knobs) changes.

### <a name="full-ft-per-stage-status"></a>Full-FT per-stage status

The launcher accepts all four stages (`midtrain|sft|dpo|grpo`), and the campaign
routes each to the launcher **only if** that stage exposes a JSON `-m` entry
(detected via a `<stage>_config_from_dict` builder, so it flips on automatically
the moment the entry ships - no campaign change needed):

| Stage | JSON `-m` entry | `--full-ft` behavior |
|-------|-----------------|----------------------|
| `midtrain` | ✅ `kore.policy.midtrain` | Shells out to the FSDP launcher - **real full-parameter sharded** (ZeRO-3/FSDP) continued-pretrain. The base model / corpus / output_dir travel in the JSON. |
| `sft`  | ✅ `kore.policy.sft`  | Shells out to the FSDP launcher - **real full-parameter sharded** (ZeRO-3/FSDP). |
| `dpo`  | ✅ `kore.policy.dpo`  | Shells out per pass/round to the FSDP launcher - **full-parameter sharded** (IPO + refreshed ref travel in the JSON). |
| `grpo` | ✅ `kore.policy.grpo` | Shells out to the FSDP launcher - **full-parameter sharded** GRPO. The correctness→latency curriculum runs as **two launched full-parameter GRPO runs** (phase-1 checkpoint → phase-2 init); the train-split task ids + Kevin/anti-collapse levers travel in the JSON. **There is no LoRA shortcut for the RL stage under `--full-ft`.** |

So under `--full-ft` **all four training stages - `midtrain` / `sft` / `dpo` /
`grpo` - are full-parameter sharded via the one command**, live today.

This is deliberate: the campaign **never silently degrades**. Each training
stage's JSON `-m` entry is detected via its `<stage>_config_from_dict` builder, so
the campaign shells out to the FSDP launcher automatically; if an entry were ever
missing it would print a loud warning naming it (rather than pretending to shard)
and it **never** falls back to LoRA for a `--full-ft` run. `--lora` remains the
single-process LoRA bring-up path (including GRPO LoRA), which needs no launcher.

To full-FT `midtrain` by hand (the exact command the campaign issues under the
hood), run the sharded launcher directly with the shipped config:

```bash
scripts/launch_distributed.sh midtrain configs/midtrain_14b_full.json
```

## How a launcher-driven stage runs

`scripts/launch_distributed.sh sft configs/sft_14b_full.json` expands to:

```bash
PYTHONPATH=<repo> accelerate launch \
    --config_file configs/accelerate_fsdp.yaml \
    -m kore.policy.sft configs/sft_14b_full.json
```

`kore.policy.sft` / `kore.policy.dpo` / `kore.policy.grpo` each expose a
`__main__` entry that reads the JSON config (via `<stage>_config_from_dict`),
defaults `distributed=true`, and calls `train_sft` / `dpo.train` / `train_grpo`.
Under FSDP the model is loaded **without** `device_map` (the two are incompatible
- accelerate/FSDP owns device placement); the HF-`Trainer` stages wrap the model
with `TrainingArguments(fsdp=..., fsdp_config=...)` built from the config. The
Trainer stages (midtrain / sft / dpo) run `FULL_SHARD` (ZeRO-3): params, grads,
and optimizer state are all sharded, since they never call `generate()` in the
loop. GRPO is the exception: it runs `SHARD_GRAD_OP` (ZeRO-2, via
`accelerate_fsdp_grpo.yaml` and `build_fsdp_plugin`), which keeps params
replicated between forwards while still sharding grads and the optimizer. This is
required because `FULL_SHARD` reshards params after every forward, so the many
decode steps inside `model.generate()` would re-gather params each step and
deadlock. GRPO additionally rolls out against a full-weight local replica synced
once per step, so no FSDP collective runs during generation at all.

For GRPO the resolved JSON also carries the run's **train-split task ids** (so the
sharded run trains only on the TRAIN split, never the held-out generalization
family) and the **Kevin + anti-collapse levers** (`rc_grpo` / `variance_floor` /
`sc_grpo` / `gtpo_codesim` / `value_prefilter`, all on by default -
`configs/grpo_14b_full.json`).

### Identical per-turn credit: single-process == distributed

The single-process and distributed GRPO paths now compute **identical per-turn
credit**. Both `_one_group` (serial) and `_rollout_slice_distributed` (sharded)
feed their per-trajectory `(rewards, correct, infra, phis)` traces through the
*same* `build_kevin_samples(...)` call with the *same* paradigm-v2 levers read
from the config:

- **P0d** `credit_incorrect_turns` - an incorrect turn keeps its bounded shaped
  SNR-progress reward (below `correctness_weight`) instead of a hard zero, so the
  gradient is not flat across the not-yet-correct band; and
- **P0b** `physics_shaping_weight` - the roofline-attainment potential
  `Φ(s)` (`kore.reward.whitebox.phi_potential`; online the PMC-free `η`, not the
  named-residual `ρ`) is added as Ng-Harada-Russell PBS
  `F_t = γ·Φ(s_{t+1}) − Φ(s_t)` (`kore.reward.shaping`), *approximately* policy-
  invariant (an expected-gradient-neutral state-dependent baseline, not an exact
  at-any-weight theorem here).

Both travel in the resolved GRPO JSON (`credit_incorrect_turns=true`,
`physics_shaping_weight=0.15` in `configs/grpo_14b_full.json`), so switching
between the single-GPU LoRA bring-up and the sharded full-FT run does **not**
change the credit assignment - only the sharding does. (The sharded path applies
this credit per-rank on that rank's trajectory slice; the group-relative
advantage baseline is then computed over the all-gathered full group.)

### Example training config (`configs/sft_14b_full.json`, as actually shipped)

```json
{
  "model_id": "Qwen/Qwen3-14B",
  "dataset_path": "data/sft/multicap.jsonl",
  "output_dir": "runs/sft_14b_full",
  "use_lora": false,
  "distributed": true,
  "bf16": true,
  "gradient_checkpointing": true,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "max_seq_length": 16384,
  "num_train_epochs": 3.0,
  "learning_rate": 1e-05,
  "lr_scheduler_type": "cosine",
  "warmup_ratio": 0.03,
  "repair_loss_weight": 2.0,
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
    "backward_prefetch": "backward_pre",
    "forward_prefetch": True,                # overlaps the next layer's all-gather with compute
    "use_orig_params": True,
    "sync_module_states": True,
    "cpu_ram_efficient_loading": True,
    "limit_all_gathers": True,
    "state_dict_type": "FULL_STATE_DICT",
  },
}
```

There is deliberately **no** `activation_checkpointing` key in `fsdp_config`. Driving activation
checkpointing from the FSDP plugin's external `checkpoint_wrapper` on an FSDP1 +
`use_orig_params` unit mismatches the saved-tensor count between the forward and its recompute
(`torch.utils.checkpoint` raises `CheckpointError`). Instead each Trainer stage enables HF's own
layer-internal gradient checkpointing directly via `TrainingArguments.gradient_checkpointing=True`
with `gradient_checkpointing_kwargs={"use_reentrant": True}` - REENTRANT is deliberate: the ROCm
stack has no `flash_attn` wheel, so training runs on SDPA, whose fused-kernel choice can differ
between the checkpointed forward and its recompute (and across ranks); reentrant checkpointing
skips the saved-tensor-count check that non-reentrant enforces and would otherwise raise on that
swap.

`state_dict_type: FULL_STATE_DICT` (not a sharded state dict) is deliberate too: it makes
`trainer.save_model()` consolidate a plain HF checkpoint that the next stage loads directly via
`from_pretrained` (midtrain→sft→dpo→grpo→soup handoff), matching GRPO's own save path. At 14B the
rank-0 gather is cheap; `cpu_ram_efficient_loading` streams it for 32B/70B so the handoff still
works there. See "Checkpointing" below.

## Per-size memory guidance & accelerate configs

Rules of thumb for a full-FT run in bf16 with AdamW (~16 bytes/param of optimizer
+ master state, sharded across N ranks under FULL_SHARD):

All sizing below is for the KORE target: **8x MI350X (gfx950 / CDNA4, ~270 GB
HBM3E per GPU)**. The extra memory over the previous-gen 192 GB MI300 is decisive:
it moves 32B and even 70B full-FT into single-node-feasible territory without CPU
offload.

| Model | GPUs | Sharding | Offload | Notes (MI350X, ~270 GB/GPU) |
|-------|------|----------|---------|-------|
| 14B   | 8×MI350X | `full_shard auto_wrap` | none | Fits easily (~40 GB/rank); activation checkpointing on. |
| 32B   | 8×MI350X | `full_shard auto_wrap` | none | Fits without offload (~90 GB/rank); enable offload only to push a very long `max_seq_length`. |
| 70B   | 8×MI350X | `full_shard auto_wrap` | none (validate) or offload if tight | Sharded state ~140 GB/rank + activations => ~180-220 GB/rank at 16k seq, which fits in 270 GB, so single-node is feasible on MI350X (the 192 GB MI300 needed offload + multi-node). Validate at your `max_seq_length`, free any co-tenant GPU memory first, and use 2+ nodes for headroom. Wrap class = `LlamaDecoderLayer`. |

### 14B - `configs/accelerate_fsdp.yaml` as shipped (no offload)

Use the shipped file directly:

```bash
scripts/launch_distributed.sh sft configs/sft_14b_full.json
```

Key `fsdp_config` bits, as actually shipped: `fsdp_version: 1`, `fsdp_reshard_after_forward: FULL_SHARD`,
`fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`, `fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer`,
`fsdp_state_dict_type: FULL_STATE_DICT`, `fsdp_use_orig_params: true`, `fsdp_offload_params: false`.
That's the whole file - it does **not** set `fsdp_activation_checkpointing` (gradient checkpointing
is enabled at the `TrainingArguments` level instead, per the note above) or the
`cpu_ram_efficient_loading`/`sync_module_states`/prefetch keys (those are supplied by
`build_fsdp_kwargs` at the Python/`TrainingArguments` layer for the Trainer stages, not the
accelerate YAML).

### 32B - full_shard (offload optional on MI350X)

On 8x MI350X, 32B full-FT fits **without** CPU offload (~90 GB/rank). Offload is
only needed to push a very long `max_seq_length` (or to free headroom for a
co-tenant). To enable it, copy the shipped yaml and set `"fsdp_cpu_offload":
true` (which appends `offload` to the `fsdp` string); the accelerate yaml:

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
  fsdp_state_dict_type: FULL_STATE_DICT   # consolidated HF ckpt for cross-stage handoff (NOT sharded)
  fsdp_use_orig_params: true
```

(Do not add `fsdp_activation_checkpointing: true` here - gradient checkpointing is enabled at the
`TrainingArguments` level, not the FSDP plugin; see the note above. `cpu_ram_efficient_loading` /
`sync_module_states` are likewise supplied by `build_fsdp_kwargs`, not this YAML.)

For `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` set
`fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer` (and either leave
`fsdp_transformer_layer_cls: null` in the JSON to auto-detect, or set it
explicitly).

### 70B - full_shard + CPU offload, multi-node preferred

`deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (Llama arch). On 8x MI350X (~270
GB/GPU) a 70B full-FT fits **single-node** (~180-220 GB/rank at 16k seq) without
CPU offload in principle; validate at your target `max_seq_length` and free any
co-tenant GPU memory first. Use 2+ nodes for headroom. (The older 192 GB MI300
required offload and multi-node here.) Training config: `"model_id":
"deepseek-ai/DeepSeek-R1-Distill-Llama-70B"`, `"use_lora": false`,
`"fsdp_transformer_layer_cls": "LlamaDecoderLayer"` (add `"fsdp_cpu_offload":
true` only if a rank is tight at long sequence length).

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
  fsdp_state_dict_type: FULL_STATE_DICT   # consolidated HF ckpt for cross-stage handoff (NOT sharded)
  fsdp_use_orig_params: true
```

(Same caveat as the 32B block above: no `fsdp_activation_checkpointing` here; `cpu_ram_efficient_loading`/`sync_module_states` come from `build_fsdp_kwargs`.)

Multi-node (e.g. 2 nodes × 8 GPUs = 16 ranks): set `num_machines: 2`,
`num_processes: 16`, and per-node `machine_rank` (0 and 1), plus
`main_process_ip` / `main_process_port` on the workers. Launch the same
`scripts/launch_distributed.sh sft <config.json> --nproc 16` on each node with the
node-specific `machine_rank`, or drive it through your cluster's `accelerate
launch --multi_gpu --machine_rank ...` wrapper.

## Checkpointing

`fsdp_state_dict_type: FULL_STATE_DICT` is used everywhere (14B/32B/70B), not a sharded state
dict: the HF Trainer gathers the full model to rank 0 on save and writes a plain, consolidated HF
checkpoint - the same format `from_pretrained` expects, so the next stage (midtrain→sft→dpo→grpo→
soup) loads it exactly like a non-distributed run, with no FSDP-mesh-matching requirement on
reload. The trainer's `save_model` / merge-on-save behavior is unchanged for LoRA. The rank-0
gather is cheap at 14B; at 32B/70B `fsdp_cpu_ram_efficient_loading` streams it so the save (and the
next stage's load) still fit. To reload into a single-file model for downstream stages
(DPO/GRPO/soup), point those stages at the produced `output_dir` as usual.

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
