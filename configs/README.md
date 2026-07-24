# `configs/` — FSDP and per-stage full fine-tuning recipes

The distributed launch configuration and the full-parameter recipes for each training stage. When `scripts/run_campaign.py` runs with `--full-ft`, it overlays the run's dynamic fields (model / dataset / output directory) onto the shipped template, writes the resolved config into `<data-root>/launch/`, and shells out to `scripts/launch_distributed.sh` → `accelerate launch`. See the root [training pipeline](../README.md#the-training-pipeline) for where these stages sit.

---

## Files

| File | Purpose |
| --- | --- |
| `accelerate_fsdp.yaml` | FSDP launch config for the Trainer stages (`midtrain` / `sft` / `dpo`): 8 ranks, `FULL_SHARD` (ZeRO-3) |
| `accelerate_fsdp_grpo.yaml` | FSDP launch config for `grpo`: 8 ranks, `SHARD_GRAD_OP` (ZeRO-2). GRPO's in-loop `model.generate()` deadlocks under `FULL_SHARD`'s per-decode re-gather, so params must stay replicated between forwards |
| `midtrain_14b_full.json` | Continued-pretraining recipe |
| `sft_14b_full.json` | SFT recipe |
| `dpo_14b_full.json` | DPO recipe |
| `grpo_14b_full.json` | GRPO recipe |

## How a resolved config is written (`_launch_distributed`)

The JSON templates are never edited by hand at run time. For each full-FT training stage (`midtrain` / `sft` / `dpo` / `grpo`), `run_campaign.py`'s `_launch_distributed` helper:

1. loads the shipped template `configs/<stage>_14b_full.json`;
2. overlays the run's dynamic overrides (`model_id`, `output_dir`, dataset paths, task ids, the anti-collapse and value-prefilter fields, …);
3. forces `distributed=true` and `use_lora=false`;
4. writes the result to `<data_root>/launch/<run_name>.json` (`run_name` defaults to the stage name); and
5. shells out to `scripts/launch_distributed.sh <stage> <resolved.json>`.

For GRPO, `run_name` depends on `--grpo-curriculum` (default on): a curriculum run writes `grpo_phase1_correctness.json` then `grpo_phase2_latency.json`; a non-curriculum run (`--no-grpo-curriculum`) writes a single `grpo.json`, rendered fresh from `configs/grpo_14b_full.json` each time the stage starts. A `launch/grpo_phase1_correctness.json` on disk is authoritative only for a curriculum-mode run — check the run's CLI flags or manifest rather than assuming a leftover file describes the live config.

---

## FSDP config (`accelerate_fsdp.yaml`)

```yaml
distributed_type: FSDP
num_processes: 8                 # one rank per gfx950 (MI350X) GPU
mixed_precision: bf16
fsdp_config:
  fsdp_version: 1
  fsdp_reshard_after_forward: FULL_SHARD          # ZeRO-3 (params + grads + optim)
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer
  fsdp_state_dict_type: FULL_STATE_DICT           # consolidate a plain HF ckpt for cross-stage handoff
  fsdp_use_orig_params: true
  fsdp_offload_params: false                      # 14B: keep on GPU (set true for 32B/70B)
```

`grpo` uses `accelerate_fsdp_grpo.yaml`, identical except that it pins `SHARD_GRAD_OP` (ZeRO-2) through both `fsdp_sharding_strategy` and `fsdp_reshard_after_forward` so params stay replicated between forwards and in-loop generation runs locally on each rank (grads and optimizer state are still sharded, so a 14B full-FT still fits on 8× MI350X). Scaling is documented inline in both files: set `fsdp_offload_params: true` for 32B, add nodes (`num_machines>1`) and switch the wrap class to `LlamaDecoderLayer` for the 70B.

---

## Stage recipes

**`midtrain_14b_full.json`** — continued pretraining on the general Triton/HIP corpus: `max_seq_length=8192`, `num_train_epochs=1`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`, `learning_rate=1e-5` (cosine, `warmup_ratio=0.05`), `general_replay_frac=0.3` (anti-forgetting blend).

**`sft_14b_full.json`** — full-FT: `max_seq_length=16384`, `num_train_epochs=3`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, `learning_rate=1e-5` (cosine, `warmup_ratio=0.03`), `repair_loss_weight=2.0`.

**`dpo_14b_full.json`** — `beta=0.1`, `loss_type=["sigmoid","sft"]` (a combined preference + SFT-on-chosen loss, `loss_weights=[1.0,1.0]`), `label_smoothing=0.1`, `truncation_mode="keep_end"`, `max_length=16384`, `learning_rate=5e-7` (cosine, `warmup_ratio=0.1`), `max_grad_norm=0.5`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`.

**`grpo_14b_full.json`** — the multi-turn agentic GRPO recipe (key fields):

```jsonc
{
  "num_trajectories": 16, "num_turns": 4, "tasks_per_step": 8,
  "learning_rate": 2e-6, "max_grad_norm": 0.5, "ref_anchor_coef": 1e-3,
  "temperature": 0.9, "max_prompt_length": 16384, "max_response_length": 16384,
  "agentic": true, "starpo_s": true, "dynamic_sampling": true,
  "rc_grpo": true, "variance_floor": 0.1, "sc_grpo": true,
  "gtpo_codesim": true, "value_prefilter": true,
  "coevolve": true, "coevolve_include_vendor": true,          // open-ended curriculum (SELECT)
  "coevolve_distill_path": "data/full14b/coevolve_wins.jsonl",
  "reward_phase": "all", "reward_mode": "speedup", "physics_weight": 1.0,
  "physics_sku": "mi350x", "physics_model_fingerprint": "sha256:54296...",
  "credit_incorrect_turns": true, "physics_shaping_weight": 0.0,
  "physics_shaping_evidence_path": null, "physics_shaping_evidence_fingerprint": null,
  "agentic_transform_tools": true,
  "use_search": true, "search_budget": 32, "search_every": 50,
  "search_bnb": true, "search_k_expand": 6, "search_max_depth": 6, "search_value_prior": true,
  "roofline_gate": true, "roofline_tol": 0.25, "physics_live_counters": false,
  "transform_discover": true,
  "coevolve_mint": true, "coevolve_mint_batch": 6,
  "coevolve_evolve_grammar": true, "coevolve_regret_vs_opus": true,
  "total_steps": 2000, "save_steps": 100,
  "fsdp_version": 1, "zero_stage": 3, "synced_gpus": true, "cpu_offload": false,
  "bf16": true, "gradient_checkpointing": true
}
```

> The `zero_stage: 3` and `synced_gpus: true` fields are superseded at launch. Distributed GRPO runs ZeRO-2 (`SHARD_GRAD_OP`, set authoritatively in `accelerate_fsdp_grpo.yaml`) and rolls out against a full-weight local replica synced once per step, so `model.generate()` never triggers an FSDP all-gather. See [`kore/policy`](../kore/policy/README.md) for the FSDP notes.

### GRPO credit, search, and curriculum fields

These fields configure GRPO's credit assignment, verified transform action space, test-time search, and open-ended curriculum. Each maps to a `GRPOConfig` dataclass field (`kore/policy/configs.py`), where the defaults are conservative (off) and this template turns them on. Every backing module is fail-safe: a runtime error degrades to a no-op rather than failing the run.

| Field | `grpo_14b_full.json` | Description |
| --- | --- | --- |
| `reward_mode` | `"speedup"` | Terminal / within-turn reward tier for a *correct* kernel. `"residual"` is evidence-gated and falls back to verified speedup unless the task family passes held-out P0 validation under the exact model fingerprint. |
| `credit_incorrect_turns` | `true` | Feed an incorrect turn's bounded shaped-progress reward (sub-threshold SNR + format signal, always `< correctness_weight`) into the per-turn return instead of zeroing it, densifying the gradient in the not-yet-correct band while keeping correctness lexicographically dominant. Consumed in `build_kevin_samples`. |
| `physics_shaping_weight` | `0.0` | Empirical physics shaping is disabled. A nonzero value is effective only with a pinned evidence artifact whose per-family normalized task-cluster CV, bootstrap CI, permutation test, and adjusted p-value all pass under the same physical-model fingerprint. |
| `agentic_transform_tools` | `true` | Expose the ε-typed transform calculus (`kore.transform`: real Triton rewrites, each exact `≡` or approximate `≈_ε` with an error budget and side conditions) to the agent as `list_transforms` / `apply_transform` tools — a verified, in-contract optimization action space alongside free-form editing. CPU-only source rewrites; the env still verifies the result. Consumed in `kore/agent/tools.py`. |
| `use_search` | `true` | Run AlphaKernel value-guided best-first search over the verified transform action space (`kore.search.propose.search_from_kernel`) as a throttled, off-policy search-then-distill hook (`_maybe_search_then_distill`). It runs after the on-policy gradient sample is built, banks faster verified kernels as distillation targets (`coevolve_distill_path`), and is never attributed to the on-policy update. |
| `search_budget` | `32` | Verifier-call cap per search invocation. |
| `search_every` | `50` | Run the search-then-distill hook once every N steps, on the step's single best group (rank 0), so the extra verifier budget stays bounded. |
| `coevolve_mint` | `true` | Let the `CoevolutionController` mint new correct-by-construction tasks (`kore.openended.minter`) beyond the registered menu, then materialize them with a materialize-time self-check (`kore.openended.materialize`) before they enter the curriculum. Complements the SELECT-only curriculum (`coevolve: true`). See [`kore/openended`](../kore/openended/README.md). |
| `coevolve_mint_batch` | `6` | Candidate tasks minted per round; each must pass the materialize self-check to be admitted. Inert while `coevolve_mint` is `false`. |

See [`scripts/README.md`](../scripts/README.md) for how these configs are launched and [`docs/DISTRIBUTED.md`](../docs/DISTRIBUTED.md) for sizing.
