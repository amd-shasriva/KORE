# `configs/` — FSDP & per-stage full-FT recipes

The distributed launch config and the "locked" full-parameter recipes for each training stage. `scripts/run_campaign.py --full-ft` overlays the run's dynamic fields (model / dataset / output dir) onto these templates, writes the resolved config into `<data-root>/launch/`, and shells out to `scripts/launch_distributed.sh` → `accelerate launch`.

---

## Files

| File | Purpose |
| --- | --- |
| `accelerate_fsdp.yaml` | The FSDP launch config (8 ranks, ZeRO-3 for Trainer stages) |
| `midtrain_14b_full.json` | Stage-0 continued-pretraining recipe |
| `sft_14b_full.json` | Stage-1 SFT recipe |
| `dpo_14b_full.json` | Stage-2 DPO recipe |
| `grpo_14b_full.json` | Stage-3 GRPO recipe (all best-in-class levers) |

---

## FSDP config (`accelerate_fsdp.yaml`)

```yaml
distributed_type: FSDP
num_processes: 8                 # one rank per gfx942 GPU
mixed_precision: bf16
fsdp_config:
  fsdp_version: 1
  fsdp_reshard_after_forward: FULL_SHARD          # ZeRO-3 (params+grads+optim)
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer
  fsdp_state_dict_type: FULL_STATE_DICT           # consolidate a plain HF ckpt for cross-stage handoff
  fsdp_use_orig_params: true
  fsdp_offload_params: false                      # 14B: keep on GPU (set true for 32B/70B)
```

Scaling notes are inline: flip `fsdp_offload_params: true` for 32B, add nodes (`num_machines>1`) for 70B.

---

## Stage recipes (highlights)

**`sft_14b_full.json`** — full-FT, `max_seq_length=16384`, `num_train_epochs=3`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`, `repair_loss_weight=2.0`.

**`dpo_14b_full.json`** — `beta=0.1`, `loss_type="ipo"`, `max_length=16384`, `max_prompt_length=8192`, `learning_rate=5e-6`.

**`grpo_14b_full.json`** — the full paradigm, all levers on:

```jsonc
{
  "num_trajectories": 16, "num_turns": 4, "tasks_per_step": 8,
  "learning_rate": 2e-6, "max_grad_norm": 0.5, "ref_anchor_coef": 1e-3,
  "agentic": true, "starpo_s": true, "dynamic_sampling": true,
  "rc_grpo": true, "sc_grpo": true, "gtpo_codesim": true, "value_prefilter": true,
  "coevolve": true, "coevolve_include_vendor": true,     // open-ended curriculum
  "reward_mode": "residual", "physics_weight": 1.0,      // physics roofline reward
  "reward_phase": "all",
  "fsdp_version": 1, "zero_stage": 3, "synced_gpus": true, "cpu_offload": false,
  "bf16": true, "gradient_checkpointing": true
}
```

> GRPO distributed effectively uses ZeRO-2 (SHARD_GRAD_OP) at rollout time so `model.generate()` works under sharding — see [`kore/policy`](../kore/policy/README.md) FSDP notes.

Every field maps to a `GRPOConfig` dataclass field (`kore/policy/configs.py`). See [`scripts/README.md`](../scripts/README.md) for how these are launched and [`docs/DISTRIBUTED.md`](../docs/DISTRIBUTED.md) for sizing.
