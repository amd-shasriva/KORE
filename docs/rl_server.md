# KORE RL Server — verl + SGLang on an isolated ROCm box

This document describes the **production RL-training path** for KORE's Stage-3
multi-turn GRPO: a distributed [verl](https://github.com/volcengine/verl) trainer
with an [SGLang](https://github.com/sgl-project/sglang) rollout engine, running on
a **dedicated ROCm box in its own venv**, driven by KORE's config and the KORE
verified reward.

It is referenced from `scripts/run_campaign.py` (Stage-0/1/GRPO) and implemented
by `kore/policy/grpo.py` (`train_grpo(..., backend="verl")`,
`build_verl_grpo_config`, `kore_verl_reward`, `_train_grpo_verl`) plus
`scripts/launch_verl.sh`.

---

## 1. Why the RL trainer is a separate box/venv

KORE has two hard-to-reconcile environments:

| | Verified env (this repo) | RL trainer (verl + SGLang) |
|---|---|---|
| Purpose | Compile + validate + bench candidate kernels (the reward) | Roll out + optimize the policy |
| torch | ROCm **2.10** wheel (verifier/kernels qualified against it) | **2.11**, as required by the current verl/SGLang ROCm wheels |
| Footprint | Small, CPU/1-GPU, deterministic subprocess sandbox | 4–8+ GPU, FSDP/Megatron shards, SGLang servers |
| Failure isolation | A bad kernel must not crash the trainer | A trainer OOM must not corrupt the verifier |

These **could not be pip-installed together** (torch 2.11 vs the ROCm 2.10 pin).
So the verl trainer lives in `.venv-verl` on a dedicated box, and the verified
reward is invoked *from this repo* as verl's custom reward function. The
in-process backend (`backend="inprocess"`/`"fallback"`) is the single-box path
that avoids the split entirely; the verl backend is the multi-GPU scale-out path.

There is **no silent substitution**: `backend="verl"` raises an actionable error
if `verl` is not importable; `backend="auto"` logs and uses the in-process
backend; only `backend="verl"` (or `auto` on a provisioned box) runs verl.

---

## 2. Architecture

```
        ┌──────────────────────────────────────────────────────────────┐
        │  RL box (.venv-verl, ROCm, torch 2.11)                         │
        │                                                                │
        │   verl.trainer.main_ppo (FSDP2/Megatron actor + ref)           │
        │        │                     ▲                                 │
        │        │ prompts             │ token log-probs / weights sync   │
        │        ▼                     │                                 │
        │   SGLang rollout server (async, multi-turn, TP=tensor_parallel)│
        │        │ generated FULL_KERNEL text                            │
        │        ▼                                                        │
        │   custom_reward_function = kore.policy.grpo.kore_verl_reward    │
        └────────┼───────────────────────────────────────────────────────┘
                 │  parse_response -> KoreEnv.step -> compute_reward
                 ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  KORE verified env (kore.env.KoreEnv, kore.reward.compute_reward)│
        │  compile → 5-stage validation + SNR gate → on-box speedup vs    │
        │  the production baseline (AITER/hipBLASLt). Anti-hack scan.     │
        └──────────────────────────────────────────────────────────────┘
```

* **verl** owns the GRPO optimization: group-normalized advantages, the PPO/
  Clip-Higher surrogate, the KL-to-reference anchor, FSDP sharding, checkpointing.
* **SGLang** owns rollout generation (fast paged-attention serving, async
  multi-turn so the env feedback can be injected between assistant turns).
* **KORE** owns the *reward*: the same verified, anti-hackable
  `compute_reward(obs, kernel)` used by the in-process backend, wrapped as
  `kore_verl_reward(...)` so verl calls it per rollout.

---

## 3. How KORE's reward/env plug in as the verl reward fn

`build_verl_grpo_config` sets:

```json
"custom_reward_function": { "path": "<repo>/kore/policy/grpo.py", "name": "kore_verl_reward" }
```

verl calls `kore_verl_reward(data_source, solution_str, ground_truth, extra_info)`
for each rollout. The dataset rows (written by `_write_verl_dataset`) carry
`extra_info.task_id`, so the reward fn:

1. `get_task(task_id)` — resolves the verified task,
2. `parse_response(solution_str)["kernel"]` — extracts the FULL_KERNEL,
3. `KoreEnv(task).step(kernel, full_validation=True, multi_shape=True)` — compile
   + validate + bench on real silicon,
4. `compute_reward(obs, kernel, dtype=..., snr_threshold=...)` — the lexicographic
   reward (hack/compile-fail < incorrect < correct-slow < correct-fast).

This is **byte-for-byte the same reward** as the in-process backend — the only
difference is who drives the rollout (SGLang vs transformers `.generate`).

> Multi-turn note: verl's async rollout supports serial refinement
> (`multi_turn.enable`, `max_assistant_turns=num_turns`). Wire the per-turn
> verifier feedback (`build_turn_feedback`) as the interaction/tool response so
> the trajectory matches the Kevin serial-refinement recipe. The trajectory
> value uses best-correct-kernel scoring; see `kore/policy/grpo.py`.

---

## 4. GRPOConfig → verl config mapping

`build_verl_grpo_config(GRPOConfig)` (pure, unit-tested in
`tests/test_rl_core.py`) maps:

| GRPOConfig field | verl key | meaning |
|---|---|---|
| — | `algorithm.adv_estimator = "grpo"` | group-normalized advantages |
| `gamma` | `algorithm.gamma` | discounted multi-turn credit |
| `kl_coef` | `algorithm.kl_ctrl.kl_coef` | (reward-KL ctrl; off by default) |
| `num_trajectories` | `actor_rollout_ref.rollout.n` | group size *m* per prompt |
| `num_turns` | `...rollout.multi_turn.max_assistant_turns` (+ `.enable=True`) | serial refinement turns |
| `tensor_parallel_size` | `...rollout.tensor_model_parallel_size`, `trainer.n_gpus_per_node` | rollout TP |
| `temperature`, `top_p` | `...rollout.temperature`, `...rollout.top_p` | sampling |
| — | `...rollout.name = "sglang"`, `.mode = "async"` | SGLang async rollout |
| `ref_anchor_coef` (>0) | `...actor.use_kl_loss=True`, `...actor.kl_loss_coef` | retention KL anchor |
| — | `...actor.kl_loss_type = "low_var_kl"` | Schulman k3 (low-variance) KL |
| `clip_ratio_low`, `clip_ratio_high` | `...actor.clip_ratio_low/high` | DAPO Clip-Higher |
| — | `...actor.loss_agg_mode = "token-mean"` | DAPO length-debias |
| `learning_rate`, `warmup_ratio`, `lr_scheduler_type` | `...actor.optim.*` | optimizer |
| `use_lora` + `lora.*` | `...model.lora_rank/lora_alpha/target_modules` | LoRA vs full-FT (`lora_rank=0`) |
| `gradient_checkpointing` | `...model.enable_gradient_checkpointing` | activation memory |
| `max_prompt_length`, `max_response_length` | `data.max_prompt_length/max_response_length`, `...rollout.max_model_len` | lengths |
| `tasks_per_step` | `data.train_batch_size`, `...actor.ppo_mini_batch_size` | batch |
| `total_steps`, `save_steps` | `trainer.total_training_steps`, `trainer.save_freq` | schedule |
| `output_dir` | `trainer.default_local_dir`, `data.train_files` (`.../data/kore_train.parquet`) | outputs |
| (reward) | `custom_reward_function.{path,name}` | `kore_verl_reward` |

`_train_grpo_verl` writes the resolved config to `{output_dir}/verl_grpo_config.json`,
writes the task dataset to `{output_dir}/data/kore_train.parquet`, then runs
`python -m verl.trainer.main_ppo <hydra overrides>`.

---

## 5. Launching (14B / 32B / 70B)

Everything goes through `scripts/launch_verl.sh`, which provisions `.venv-verl`,
installs the ROCm torch + SGLang/vLLM wheels + verl, starts SGLang, and runs the
KORE-driven verl trainer.

### 14B — bring-up / single-node LoRA
```bash
bash scripts/launch_verl.sh \
  --model Qwen/Qwen3-14B \
  --tasks rmsnorm_aiter,gemm_bf16 \
  --out runs/grpo_14b --backend verl \
  --tp 2 --traj 8 --turns 3 --lora --steps 200
```

### 32B — primary GRPO (full-FT, single 8×MI325X node)
```bash
bash scripts/launch_verl.sh \
  --model Qwen/Qwen3-32B \
  --out runs/grpo_32b --backend verl \
  --tp 4 --traj 16 --turns 4 --steps 500
```

### 70B — LoRA scale target (multi-node; set nnodes in the verl config)
```bash
bash scripts/launch_verl.sh \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --out runs/grpo_70b --backend verl \
  --tp 8 --traj 16 --turns 4 --lora --steps 500
# For >1 node, raise trainer.nnodes and launch under Ray/torchrun per verl docs;
# SGLang rollout TP stays within a node, FSDP shards the actor across nodes.
```

Notes:
* Pin `--rocm` to the box's driver stack (`--rocm 6.3` → the matching torch wheel
  index). The exact SGLang/vLLM ROCm wheels are box-qualified — see the AMD
  ROCm wheel index / infinity-hub and adjust `SGLANG_ROCM_WHEEL`/`VLLM_ROCM_WHEEL`
  in `scripts/launch_verl.sh`.
* `--dry-run` prints the commands without executing (useful to inspect the
  resolved plan on a box without GPUs).
* `--no-install` skips the venv/wheel install when `.venv-verl` is already
  provisioned.

---

## 6. Relationship to the in-process backend

`backend="inprocess"` (alias `"fallback"`) runs the complete, GPU-proven
transformers+PEFT GRPO loop in `kore/policy/grpo.py::_train_grpo_inprocess`
(rollout against `KoreEnv`, Kevin per-turn credit, StarPO-S variance filtering,
micro-batched O(1-sample) backward, k3 KL anchor). It needs no separate server
and is the right choice for single-GPU / LoRA bring-up and CI smoke
(`scripts/grpo_smoke.py`). The verl backend is the scale-out path for full 32B/70B
runs. Both optimize the **same GRPO objective against the same verified reward**.
