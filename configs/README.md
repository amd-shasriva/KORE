# `configs/` - FSDP & per-stage full-FT recipes

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
num_processes: 8                 # one rank per gfx950 (MI350X) GPU
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

**`sft_14b_full.json`** - full-FT, `max_seq_length=16384`, `num_train_epochs=3`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`, `repair_loss_weight=2.0`.

**`dpo_14b_full.json`** - `beta=0.1`, `loss_type="ipo"`, `max_length=16384`, `max_prompt_length=8192`, `learning_rate=5e-6`.

**`grpo_14b_full.json`** - the full paradigm, all levers on:

```jsonc
{
  "num_trajectories": 16, "num_turns": 4, "tasks_per_step": 8,
  "learning_rate": 2e-6, "max_grad_norm": 0.5, "ref_anchor_coef": 1e-3,
  "agentic": true, "starpo_s": true, "dynamic_sampling": true,
  "rc_grpo": true, "sc_grpo": true, "gtpo_codesim": true, "value_prefilter": true,
  "coevolve": true, "coevolve_include_vendor": true,     // open-ended curriculum (SELECT)
  "reward_mode": "speedup", "physics_weight": 1.0,       // vendor-relative speedup reward
  "credit_incorrect_turns": true, "physics_shaping_weight": 0.15,  // paradigm-v2 credit (ON)
  "agentic_transform_tools": true,                       // paradigm-v2 verified transform tools (ON)
  "use_search": true, "search_budget": 16, "search_every": 50,     // paradigm-v2 test-time search (ON)
  "coevolve_mint": true, "coevolve_mint_batch": 6,       // paradigm-v2 open-ended minting (ON)
  "reward_phase": "all",
  "fsdp_version": 1, "zero_stage": 3, "synced_gpus": true, "cpu_offload": false,
  "bf16": true, "gradient_checkpointing": true
}
```

> Note: the `zero_stage: 3` and `synced_gpus: true` fields above are superseded at runtime. Distributed GRPO runs ZeRO-2 (`SHARD_GRAD_OP`, set authoritatively in `accelerate_fsdp_grpo.yaml`) and rolls out against a full-weight local replica synced once per step, so `model.generate()` never triggers an FSDP all-gather. See [`kore/policy`](../kore/policy/README.md) FSDP notes.

### Paradigm-v2 GRPO flags

Nine flags tune the paradigm-v2 credit assignment, verified transform action space, test-time search, and open-ended curriculum. Each maps to a `GRPOConfig` field (`kore/policy/configs.py`); the "flagship" column is the value shipped in `grpo_14b_full.json`. All nine are **ON** in the flagship, and every backing module is fail-safe (any runtime error degrades to a no-op).

| Flag | Flagship | State | What it does |
| --- | --- | --- | --- |
| `reward_mode` | `"speedup"` | **ON (active)** | Terminal/within-turn reward tier for a *correct* kernel. `"speedup"` = the high-contrast vendor-relative speedup (was `"residual"`, the compressed physics-residual credit). Ops with no roofline model fall back to speedup either way; anti-hack/correctness gating is identical. Consumed in `kore/policy/grpo.py`. |
| `credit_incorrect_turns` | `true` | **ON (active)** | Feed an **incorrect** turn's *shaped progress* reward (bounded sub-threshold SNR + format signal, always `< correctness_weight`) into the Kevin per-turn return instead of hard-zeroing it - densifies the gradient in the not-yet-correct band while keeping correctness lexicographically dominant. Consumed in `build_kevin_samples` (`kore/policy/grpo.py`). |
| `physics_shaping_weight` | `0.15` | **ON (active)** | Weight on potential-based shaping `F_t = γ·Φ(s_{t+1}) − Φ(s_t)` with `Φ =` roofline attainment `ρ` (`kore.reward.whitebox.phi_potential`). By the Ng-Harada-Russell theorem this is **policy-invariant at any weight** (no reward-hacking incentive), only densifying per-turn credit toward the roofline. `0.0` disables it. Consumed in `build_kevin_samples`. |
| `agentic_transform_tools` | `true` | **ON (fail-safe)** | Advertise the verified ε-typed transformation calculus (`kore.transform`: real Triton rewrites, each exact `≡` / approx `≈_ε` with an error budget + side conditions) to the agent as `list_transforms`/`apply_transform` tools - a **provably-in-contract** optimization action space on top of free-form editing. Pure CPU (source rewrites); the env still verifies the result, so it cannot bypass the SNR gate. Consumed in `kore/agent/tools.py`. |
| `use_search` | `true` | **ON (fail-safe)** | Run AlphaKernel value-guided best-first search over the **verified transform action space** (`kore.search.propose.search_from_kernel`, driven by the production `TransformProposePolicy`) with the env as a perfect simulator + roofline admissible bound. Wired as a **throttled, off-policy search-then-distill hook** (`_maybe_search_then_distill`, `kore/policy/grpo.py`): it runs AFTER the on-policy gradient sample is built from the model's own tokens, banks faster verified kernels as **distillation targets** (`coevolve_distill_path`), and is never attributed to the on-policy update - sound at any setting. Fully fail-safe (any error is a no-op). |
| `search_budget` | `16` | **ON (fail-safe)** | Hard verifier-call cap per search invocation for `use_search`. Together with `search_every` this bounds the extra bench budget over the run (one search per `search_every` steps × up to `search_budget` benches each). |
| `search_every` | `50` | **ON (fail-safe)** | Throttle: run the search-then-distill hook once every `search_every` steps, on the step's single best group only (rank-0), so the extra verifier budget stays bounded rather than multiplied across every rollout. |
| `coevolve_mint` | `true` | **ON (fail-safe)** | Let the `CoevolutionController` **mint** net-new correct-by-construction tasks (`kore.openended.minter`) beyond the registered menu (measured-roofline QD + learning-progress), then **materialize** them into runnable task dirs with a materialize-time self-check (`kore.openended.materialize`) before they enter the curriculum. Complements the SELECT-only curriculum (`coevolve: true`). See [`kore/openended`](../kore/openended/README.md). |
| `coevolve_mint_batch` | `6` | **ON (fail-safe)** | Number of candidate tasks the controller mints per minting round; each must pass the materialize self-check to be admitted. Inert while `coevolve_mint` is `false`. |

All nine flags are **active in this flagship run** and consumed in the GRPO training loop: the credit-assignment trio (`reward_mode`, `credit_incorrect_turns`, `physics_shaping_weight`) shapes the on-policy gradient; `agentic_transform_tools` hands the agent the verified transform action space; the search trio (`use_search`, `search_budget`, `search_every`) runs the throttled, off-policy search-then-distill hook *after* each on-policy gradient step (banking faster verified kernels as distillation targets); and the minting pair (`coevolve_mint`, `coevolve_mint_batch`) expands the curriculum with materialized, self-checked correct-by-construction tasks. Each is a `GRPOConfig` field (default off in `configs.py`) whose backing module is fail-safe + unit-tested, so any runtime error degrades to a no-op rather than corrupting the run.

Every field maps to a `GRPOConfig` dataclass field (`kore/policy/configs.py`). See [`scripts/README.md`](../scripts/README.md) for how these are launched and [`docs/DISTRIBUTED.md`](../docs/DISTRIBUTED.md) for sizing.
