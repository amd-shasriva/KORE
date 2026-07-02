"""Training config dataclasses for the KORE policy stages.

Hyperparameters follow the KORE plan (Sec 4.6) + the Kevin recipe:
  - Base models are reasoning+math+code, RL-trainable. 14B for bring-up/SFT,
    32B primary for GRPO, scaling to a 70B distill via LoRA.
  - Stage curriculum: repair-weighted SFT -> RFT + DPO -> multi-turn GRPO.
  - GRPO (Kevin): per-turn reward S = 0.3*1{correct} + (t_base/t_cand)*1{correct},
    discounted-sum credit gamma=0.4, per-turn-as-sample, m=16 traj x n=4 turns,
    KL=0, Clip-Higher (0.2 / 0.28), serial > parallel refinement.

These are plain dataclasses (no heavy imports) so they can be constructed and
inspected on CPU / in tests. The training entrypoints in ``sft.py`` / ``dpo.py``
/ ``grpo.py`` consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- Base model ids (reasoning+math+code, RL-trainable) ---
MODEL_14B = "Qwen/Qwen3-14B"                              # bring-up / SFT default
MODEL_32B = "Qwen/Qwen3-32B"                              # GRPO primary
MODEL_32B_R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # alt 32B primary
MODEL_70B = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"    # LoRA scale target


@dataclass
class LoRAConfig:
    """PEFT LoRA adapter config shared across stages."""

    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class SFTConfig:
    """Stage-1 repair-weighted SFT (also reused by RFT on self-gen samples)."""

    model_id: str = MODEL_14B
    dataset_path: str = ""                 # chat-format JSONL
    output_dir: str = "runs/sft"

    # Plan hyperparams.
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    num_train_epochs: float = 3.0          # 2-3 epochs
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 16384
    bf16: bool = True
    gradient_checkpointing: bool = True
    packing: bool = False

    # Repair weighting: up-weight repair (broken -> fixed) turns.
    repair_loss_weight: float = 2.0

    use_lora: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    seed: int = 0
    logging_steps: int = 10
    save_steps: int = 200
    report_to: str = "none"


@dataclass
class DPOConfig:
    """Stage-2 DPO on ranked preference pairs; reference = the SFT policy."""

    model_id: str = MODEL_14B              # start from the SFT checkpoint
    ref_model_id: Optional[str] = None     # defaults to model_id (frozen SFT)
    dataset_path: str = ""                 # {prompt, chosen, rejected} JSONL
    output_dir: str = "runs/dpo"

    beta: float = 0.1                      # plan: DPO beta = 0.1
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.03

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_length: int = 16384
    max_prompt_length: int = 8192
    bf16: bool = True
    gradient_checkpointing: bool = True

    use_lora: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    seed: int = 0
    logging_steps: int = 10
    save_steps: int = 200
    report_to: str = "none"


@dataclass
class GRPOConfig:
    """Stage-3 multi-turn GRPO (Kevin recipe) with anti-collapse ladder."""

    model_id: str = MODEL_32B              # GRPO primary
    output_dir: str = "runs/grpo"

    # --- Rollout shape (Kevin: m=16 trajectories x n=4 turns) ---
    num_trajectories: int = 16             # m: group size per task
    num_turns: int = 4                     # n: refinement turns per trajectory
    serial_refine: bool = True             # serial > parallel (Kevin)
    tasks_per_step: int = 8

    # --- Kevin per-turn reward + credit ---
    correctness_weight: float = 0.3        # S = 0.3*1{correct} + speedup*1{correct}
    gamma: float = 0.4                     # discounted-sum look-ahead across turns
    per_turn_as_sample: bool = True

    # --- GRPO objective ---
    kl_coef: float = 0.0                   # Kevin: KL = 0
    clip_ratio_low: float = 0.2            # Clip-Higher lower bound (1 - 0.2)
    clip_ratio_high: float = 0.28          # Clip-Higher upper bound (1 + 0.28)
    adv_eps: float = 1e-6                  # group-normalization epsilon

    # --- Optimization ---
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    total_steps: int = 500
    max_prompt_length: int = 16384
    max_response_length: int = 8192
    bf16: bool = True
    gradient_checkpointing: bool = True

    # --- Rollout sampling ---
    temperature: float = 1.0
    top_p: float = 1.0

    # --- Serving / parallelism ---
    tensor_parallel_size: int = 4
    rollout_backend: str = "vllm"          # "vllm" | "hf"

    use_lora: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # --- Anti-collapse ladder (see anticollapse.py) ---
    rc_grpo: bool = False                  # reward-conditioned rollouts (variance floor)
    rc_p_high: float = 0.5                 # fraction of <|high_reward|> tokens
    gtpo_turn_credit: bool = False         # turn-level credit assignment
    sc_grpo_allfail: bool = False          # all-fail diversity bonus
    sc_grpo_alpha: float = 0.1

    seed: int = 0
    logging_steps: int = 1
    save_steps: int = 50
    report_to: str = "none"
