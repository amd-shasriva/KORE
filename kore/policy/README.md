# `kore/policy` — the training stages

The five-stage policy curriculum that turns a base LLM into a kernel-optimization agent, plus the prompt/response contract, RL math, FSDP wiring, and optional vLLM serving. Every training stage saves a consolidated HF checkpoint, so the next stage loads it with `from_pretrained` — no FSDP mesh is required for cross-stage handoff.

---

## The curriculum

```mermaid
flowchart LR
  BASE[HF base model] --> MT["midtrain<br/>ROCm/HIP/Triton CPT"]
  MT -->|ckpt = SFT base| SFT["sft<br/>repair-weighted multi-cap, 3 ep"]
  BUILD["build data<br/>(from kore/data)"] --> SFT
  SFT -->|ckpt| DPO["dpo<br/>iterative on-policy + DAgger, IPO"]
  DPO -->|ckpt| GRPO["grpo<br/>multi-turn agentic RL"]
  SFT -.->|KL anchor| GRPO
  GRPO -->|ckpt| SOUP["soup<br/>WiSE-FT α-sweep"]
  BASE -.->|interpolation endpoint| SOUP
  SOUP --> FINAL[final checkpoint]
```

| File | Stage | Algorithm |
| --- | --- | --- |
| `midtrain.py` | Stage 0 | Continued pretraining on a ROCm/HIP/Triton corpus (`{"text"}` JSONL), plain-text completion |
| `sft.py` | Stage 1 | Repair-weighted multi-capability SFT (repair rows upsampled 2×) |
| `dpo.py` | Stage 2 | DPO on preference pairs; iterative rounds use IPO loss + refreshed reference |
| `grpo.py` | Stage 3 | Multi-turn agentic GRPO (see below) |
| `soup.py` | Stage 4 | WiSE-FT interpolation `θ = (1-α)·θ_base + α·θ_kore`, α-swept under a retention gate |
| `configs.py` | — | All stage config dataclasses + FSDP/DeepSpeed helpers (no torch import) |
| `format.py` | — | `SYSTEM_PROMPT`, transcript assembly, `parse_response`, turn feedback |
| `anticollapse.py` | — | Anti-collapse math (RC-GRPO, AVSPO, SC-GRPO, GTPO) |
| `dynamic.py` | — | `DynamicStepController` plateau early-stop |
| `coevolve_distill.py` | — | `DistillationSink` — write verified co-evolution wins to JSONL |
| `serve.py` | — | Optional vLLM-ROCm serving wrapper (not used by the native GRPO loop) |

---

## GRPO in depth

`grpo.py` is a native, in-process, multi-turn GRPO implementation (no external rollout server). It has a single-GPU fallback (`_train_grpo_fallback`) and an FSDP/DeepSpeed distributed path (`_train_grpo_distributed`).

```mermaid
sequenceDiagram
  participant Ctrl as CoevolutionController
  participant Pol as Policy
  participant Val as Value reranker
  participant Env as KoreEnv
  participant Rew as compute_kernel_reward
  loop dynamic_sampling_refill (until target non-degenerate groups)
    Ctrl->>Pol: next task (frontier or round-robin)
    loop G trajectories × n turns
      Pol->>Pol: generate N candidates
      Val->>Val: rank → keep top-k (value prefilter)
      Pol->>Env: step(kernel)
      Env->>Rew: Observation
      Rew-->>Pol: reward (apply_reward_phase)
    end
    Note over Pol: build_kevin_samples (per-turn Kevin credit + potential shaping, best-correct scoring)
  end
  Note over Pol: GTPO code-sim · AVSPO variance-floor · StarPO-S filter (SC-GRPO: serial path only)
  Pol->>Pol: on-policy PG (clip inert @ ppo_epochs=1) + k3 KL anchor to SFT ckpt
```

Techniques wired into the loop:

- **Kevin multi-turn credit** — each trajectory is scored by its *best correct* kernel, with per-turn discounted returns, correctness-gated.
- **StarPO-S** — keep the top-variance groups (echo-trap stabilization).
- **DAPO dynamic sampling** — oversample and refill until enough non-degenerate groups are collected.
- **GRPO objective** — the importance ratio is the **GSPO sequence ratio** `r = exp(logp − old_logp)` on **token-mean** log-probs (`token_mean_logprob`, DAPO length-debias), with **global token-mean** normalization across the kept batch, a **k3 KL anchor** (`ref_anchor_coef`, the only KL term) to the post-SFT checkpoint, and a **cross-rank** group-relative baseline on the distributed path (`distributed_group_advantages` over an `all_gather` of returns). The **clip-higher** bounds (`0.03 / 0.04`) are inert in the flagship: at `ppo_epochs=1` the loss is on-policy (`old_logp == new_logp` ⇒ `r ≈ 1`), so the asymmetric clip is defense-in-depth for any `ppo_epochs > 1`.
- **Anti-collapse ladder (active levers)** — in the distributed + agentic flagship the active rungs are **AVSPO variance-floor** (`avspo_advantages`, `variance_floor=0.1`), **GTPO code-similarity** partial rewards for all-fail groups, **StarPO-S** high-variance selection, and **DAPO dynamic-sampling** refill. **SC-GRPO and RC-GRPO are serial-path-only**: the SC-GRPO KL-weighting block runs only in the single-process loop (distributed samples carry no `sc_weight`), and RC-GRPO reward tokens are prepended only in the serial `_rollout`, not in `_rollout_agentic`, so under FSDP + agentic they are configured (`sc_grpo=true`, `rc_grpo=true`) but do not act.
- **Value prefilter** — generate N, bench only the top-k (~4× measurement efficiency). The ranker is an incremental (Ansor-like) learned cost model, trained from the run's own verified ranked groups when enough exist, else a hand-coded schedule heuristic (`kore/value/replay_train.py`, auto-trained pre-GRPO, fail-safe); see [`kore/value`](../value/README.md).
- **Co-evolution** — `CoevolutionController` replaces round-robin task selection with a learnability / regret / novelty frontier (see [`kore/openended`](../openended/README.md)).
- **Physics integrity, shaping disabled** — the flagship terminal reward is vendor-relative **speedup**. The fingerprinted MI350X model is available for conservative physical-impossibility rejection and search pruning. `physics_shaping_weight=0` and live physics counters are off because no family passes the normalized held-task P0 evidence gate. `phi_potential` returns a value only with a matching physical-model fingerprint and passing family evidence; see [`kore/reward`](../reward/README.md).
- **Identical single-process / distributed credit** — both paths reduce the effective physics-shaping weight to zero without a fingerprint-pinned evidence artifact, so neither can accidentally consume a diagnostic eta/counter value as reward.
- **Verified-transform action space + test-time search + open-ended minting** — `agentic_transform_tools=true` exposes the verified ε-typed transform calculus (`kore/transform`) to the agentic policy as first-class `list_transforms` / `apply_transform` tools (`kore/agent/tools.py`; `_rollout_agentic` builds the harness via `agent_tool_schemas(transforms=True)`), so the model proposes provably-in-contract rewrites rather than free-form edits. `use_search=true` runs AlphaKernel value-guided best-first search (`kore/search`) over that same calculus through the production `TransformProposePolicy` (`kore/search/propose.py`), wired as a throttled, fail-safe, off-policy search-then-distill hook (`_maybe_search_then_distill`, in both the serial and distributed loops): it fires once every `search_every=50` steps on the step's single best group, *after* the on-policy gradient is built, spends only `search_budget=16` verifier benches, and feeds only the distillation sink — so it never corrupts on-policy credit and any error is a no-op. `coevolve_mint=true` lets `CoevolutionController` mint net-new correct-by-construction tasks (`kore/openended/minter.py`) and materialize them into runnable on-disk task dirs (`kore/openended/materialize.py`, self-checked) — see [`kore/openended`](../openended/README.md).

---

## Key config defaults (`configs.py`)

**GRPOConfig** dataclass defaults are listed below. The full 14B campaign ([`configs/grpo_14b_full.json`](../../configs/README.md)) pins `model_id` to Qwen3-14B and enables the verified `reward_mode="speedup"` terminal objective, `agentic`, `coevolve`, `value_prefilter`, the anti-collapse rungs, and the discovery levers. Physics is pinned to the fingerprinted MI350X datasheet model for integrity, while empirical physics shaping is disabled (`physics_shaping_weight=0`, `physics_live_counters=false`) because no family passes the controlled P0 held-out evidence gate.

| Field | Default | Field | Default |
| --- | --- | --- | --- |
| `num_trajectories` | 16 | `num_turns` | 4 |
| `tasks_per_step` | 8 | `temperature` | 0.9 |
| `learning_rate` | 2e-6 | `ref_anchor_coef` | 1e-3 |
| `clip_ratio_low/high` | 0.03 / 0.04 | `max_response_length` | 16384 |
| `reward_mode` | `speedup` | `physics_weight` | 1.0 |
| `starpo_s` | true | `dynamic_sampling` | true |
| `coevolve` | false | `value_prefilter` | false |
| `agentic` | false | `synced_gpus` | true |
| `credit_incorrect_turns` | false | `physics_shaping_weight` | 0.0 |
| `use_search` | false | `search_budget` | 64 |
| `coevolve_mint` | false | `coevolve_mint_batch` | 8 |
| `search_every` | 25 | `agentic_transform_tools` | false |

The dataclass defaults use legacy Kevin credit with search and minting off; the flagship JSON overrides `credit_incorrect_turns=true`, keeps `physics_shaping_weight=0`, and enables the transformation/search/minting controls. A future nonzero physics weight additionally requires a fingerprint-pinned P0 evidence artifact with a PASS for each shaped family.

**SFT**: full-FT, `max_seq_length=16384`, `num_train_epochs=3`, `repair_loss_weight=2.0`. **DPO**: `beta=0.1`, iterative rounds use `loss_type="ipo"`. **Midtrain**: `general_replay_frac=0.30`, `max_seq_length=8192`. **Soup**: `alphas=(0.7, 0.8, 0.9)`, retention `epsilon=0.005`.

---

## Anti-collapse math (`anticollapse.py`)

GRPO's group-relative advantage degenerates when a whole group shares one reward: the std → 0 and every advantage collapses to ~0. The anti-collapse ladder is a set of pure, unit-testable rungs that keep a usable signal:

- **RC-GRPO** — prepend a `<|high_reward|>` / `<|low_reward|>` control token to a fraction of rollouts. The two conditioned modes have different mean rewards, so group variance has a guaranteed floor (`variance_floor` is the diagnostic).
- **AVSPO** — when a group's realized reward std is below `tau`, inject `k` mean-preserving virtual samples into the normalization statistics only (no policy-gradient term), raising the denominator and guaranteeing a variance floor (`avspo_advantages`).
- **SC-GRPO** — for partial-solve groups, re-score other turns against a correct kernel used as an in-context demo and weight each token's PG term by KL(teacher‖student), bounded (`scgrpo_weight_from_kl`).
- **GTPO** — for all-fail groups, assign a graded partial reward from code shingle-cosine similarity to the nearest correct kernel, so an all-fail group still carries a non-degenerate signal (`gtpo_codesim_shaping`).

`AVSPO`, `SC-GRPO`, and `GTPO` are KORE's internal labels for these rungs. As noted above, only AVSPO, GTPO, StarPO-S, and DAPO dynamic-sampling are active on the distributed + agentic flagship path.

---

## Distributed (FSDP) details

- **GRPO distributed uses ZeRO-2 (`SHARD_GRAD_OP`), not `FULL_SHARD`.** `FULL_SHARD` reshards params to a 1-D flat buffer between forwards, which breaks `model.generate()`. ZeRO-2 keeps params resident per rank (only grads + optimizer state are sharded).
- **Rollout runs on a full-weight replica.** Generating on the sharded policy deadlocks (per-layer all-gathers interleave with `generate`'s own collectives on ragged lengths), so each step syncs the policy weights into a plain per-rank replica (`summon_full_params` + tensor copy, no forward) and generates locally with zero FSDP collectives; `synced_gpus` is off on the distributed path.
- **Gradient checkpointing** is non-reentrant on the GRPO sharded path and reentrant for SFT/DPO/midtrain (the flash→SDPA downgrade breaks the non-reentrant tensor-count check).
- **SDPA, not `flash_attention_2`, for GRPO/DPO**: ROCm FlashAttention hard-faults on padded generate/logp batches.
- **Agentic rollouts run serial on the distributed path** — ragged per-trajectory turn counts would desync collectives.

See [`docs/DISTRIBUTED.md`](../../docs/DISTRIBUTED.md) for sizing and the one-command launch. See also [`kore/data`](../data/README.md) (produces the corpora), [`kore/reward`](../reward/README.md), and [`kore/eval`](../eval/README.md).
