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
class DistributedMixin:
    """FSDP / distributed full-FT knobs shared by SFT & DPO.

    These only take effect for **full fine-tuning** (``use_lora=False``) that was
    launched as a multi-process job (via ``scripts/launch_distributed.sh`` /
    ``accelerate launch``). A single-process run (CPU tests, single-GPU LoRA)
    ignores them entirely and keeps the legacy ``device_map="auto"`` path, so
    nothing changes for the LoRA recipe or for the CPU test suite.

    ``fsdp`` mirrors HF ``TrainingArguments.fsdp`` (e.g. ``"full_shard auto_wrap"``,
    ``"full_shard auto_wrap offload"``). ``fsdp_transformer_layer_cls`` names the
    decoder block to shard/wrap; when ``None`` it is auto-detected from
    ``model_id`` (Qwen3 / Qwen2 / Llama families — covers the 14B/32B/70B bases).
    """

    distributed: bool = False
    fsdp: str = "full_shard auto_wrap"
    fsdp_transformer_layer_cls: Optional[str] = None
    fsdp_cpu_offload: bool = False


@dataclass
class SFTConfig(DistributedMixin):
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
class DPOConfig(DistributedMixin):
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

    # --- GRPO objective (DAPO clip-higher + importance ratio + multi-epoch) ---
    kl_coef: float = 0.0                   # Kevin: KL = 0
    clip_ratio_low: float = 0.2            # Clip-Higher lower bound (1 - 0.2)
    clip_ratio_high: float = 0.28          # Clip-Higher upper bound (1 + 0.28)
    adv_eps: float = 1e-6                  # group-normalization epsilon
    ppo_epochs: int = 2                    # minibatch passes per rollout batch (reuse old_logp)

    # --- Optimization (Kevin recipe) ---
    learning_rate: float = 2e-6            # Kevin: 2e-6
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0
    max_grad_norm: float = 0.5             # Kevin: grad-norm clip 0.5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    total_steps: int = 500
    max_prompt_length: int = 16384
    max_response_length: int = 16384       # Kevin: 16384
    bf16: bool = True
    gradient_checkpointing: bool = True

    # --- Rollout sampling (Kevin recipe) ---
    temperature: float = 0.9               # Kevin: 0.9
    top_p: float = 1.0

    # --- Serving / parallelism ---
    tensor_parallel_size: int = 4
    rollout_backend: str = "vllm"          # "vllm" | "hf"

    use_lora: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # --- Anti-collapse ladder (see anticollapse.py) ---
    rc_grpo: bool = False                  # reward-conditioned rollouts (variance floor)
    rc_p_high: float = 0.5                 # fraction of <|high_reward|> tokens
    # AVSPO virtual-sample injection: when a group's reward std < variance_floor,
    # inject ``avspo_virtual_k`` virtual samples into the NORMALIZATION stats only
    # (no PG term) to guarantee a variance floor. 0.0 disables (pure GRPO).
    variance_floor: float = 0.0            # AVSPO tau trigger (0 disables)
    avspo_virtual_k: int = 2               # #virtual samples injected at +/- tau
    # Real SC-GRPO: for partial-solve groups, re-score other turns' tokens with a
    # correct kernel as an in-context demo (teacher) and weight the per-token PG
    # term by per-token KL(teacher||student). One extra forward per weighted sample.
    sc_grpo: bool = False
    sc_grpo_w_min: float = 0.5             # SC-GRPO multiplicative weight floor
    sc_grpo_w_max: float = 2.0             # SC-GRPO multiplicative weight ceiling
    # GTPO code-similarity shaping: for ALL-FAIL groups, give a graded partial
    # reward = normalized code shingle-cosine similarity to the nearest correct
    # kernel (or the seed reference), so an all-fail group still carries signal.
    gtpo_codesim: bool = False
    gtpo_codesim_scale: float = 0.3        # magnitude of the partial reward in [0, scale]

    # --- Kevin multi-turn credit (best-kernel scoring + CoT masking) ---
    kevin_best_kernel_scoring: bool = True  # trajectory value = best correct kernel
    cot_masking: bool = True                # drop prior-turn thinking from context

    # --- Retention: KL anchor to the post-SFT multi-capability checkpoint ---
    ref_checkpoint: Optional[str] = None    # defaults to model_id (the SFT ckpt)
    ref_anchor_coef: float = 1e-3           # KL-to-reference coef (chat/code retention)

    # --- StarPO-S stabilization + DAPO dynamic sampling (oversample-and-refill) ---
    starpo_s: bool = True
    starpo_min_std: float = 1e-3            # drop zero-variance (collapsed) groups
    starpo_keep_frac: float = 0.75          # keep top-variance fraction of groups
    dynamic_sampling: bool = True           # DAPO: refill non-degenerate groups (not drop-and-shrink)
    target_groups: Optional[int] = None     # #non-degenerate groups to collect (default: tasks_per_step)
    max_sampling_attempts: Optional[int] = None  # bound on oversampling (default: 3x target_groups)

    # --- Measurement efficiency: value-model bench prefilter ---
    value_prefilter: bool = False
    num_candidates_per_turn: int = 8        # generate N per turn, bench only the top-k
    value_prefilter_k: int = 4              # bench only top-k candidates by value model
    value_model_path: Optional[str] = None

    # --- Correctness -> latency curriculum (P1) ---
    # "correctness": mask the speed term (train correctness only); "latency":
    # full correctness+speed reward; "all": no masking. The campaign runs GRPO
    # twice (correctness phase, then latency phase) by flipping this flag.
    reward_phase: str = "all"

    # --- Agentic tool-use RL (ToolRL reward shaping) ---
    agentic: bool = False                   # rollouts drive build/test/bench/pmc tools
    tool_reward_weight: float = 0.2         # weight on ToolRL-style shaping term
    max_tool_turns: int = 8

    seed: int = 0
    logging_steps: int = 1
    save_steps: int = 50
    report_to: str = "none"


@dataclass
class MidTrainConfig:
    """Stage-0 continued pretraining on the ROCm/HIP/Triton corpus."""

    model_id: str = MODEL_14B
    corpus_path: str = "data/midtrain/corpus.jsonl"
    output_dir: str = "runs/midtrain"
    general_replay_frac: float = 0.15       # 10-15% general shards (strong-shift regime)
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.05
    max_seq_length: int = 8192
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_lora: bool = False                  # full-FT (large ROCm distribution shift)


@dataclass
class MultiCapSFTConfig(SFTConfig):
    """Stage-1 multi-capability SFT mixture (kernel + general + agentic).

    Fractions of the training mix (must sum ~1.0). ~45% general = the retention
    backbone; ~10% agentic tool-use trajectories = the orchestration skill.
    """

    frac_kernel_repair_opt: float = 0.35
    frac_kernel_qa: float = 0.10
    frac_agentic_tooluse: float = 0.10
    frac_general_code: float = 0.20
    frac_math_reasoning: float = 0.15
    frac_general_chat: float = 0.10
    use_lora: bool = False                  # full-FT, governed by replay + small LR
    num_train_epochs: float = 3.0


@dataclass
class SoupConfig:
    """Stage-4 base-ward model soup (WiSE-FT interpolation)."""

    base_model_id: str = MODEL_14B          # the instruct base to interpolate toward
    kore_checkpoint: str = "runs/grpo"
    output_dir: str = "runs/soup"
    alphas: tuple = (0.7, 0.8, 0.9)         # weight on the KORE specialist
    epsilon: float = 0.005                  # max tolerated general-metric regression


# --------------------------------------------------------------------------- #
# FSDP wiring helpers (pure — no torch/transformers, safe on CPU / in tests)
# --------------------------------------------------------------------------- #

# HF decoder-block class names by model family. Auto-wrap needs the exact class
# so FSDP shards one transformer layer per unit (the ZeRO-3-equivalent recipe).
def detect_transformer_layer_cls(model_id: str) -> str:
    """Best-effort map a HF ``model_id`` to its decoder layer class for FSDP wrap.

    Covers the KORE bases: Qwen3 (14B/32B), DeepSeek-R1-Distill-Qwen (32B ->
    Qwen2), DeepSeek-R1-Distill-Llama (70B -> Llama). Falls back to the Qwen3
    block (the bring-up default). ``llama`` is checked first because the 70B id
    contains ``Llama`` but not ``qwen``.
    """
    mid = (model_id or "").lower()
    if "llama" in mid:
        return "LlamaDecoderLayer"
    if "qwen3" in mid:
        return "Qwen3DecoderLayer"
    if "qwen2" in mid or "qwen" in mid:
        return "Qwen2DecoderLayer"
    if "mistral" in mid:
        return "MistralDecoderLayer"
    return "Qwen3DecoderLayer"


def fsdp_enabled(config) -> bool:
    """True iff this run should take the distributed FSDP full-FT path.

    FSDP is used only for full fine-tuning (``use_lora=False``) launched as a
    distributed job (``distributed=True``). LoRA and single-process runs keep the
    legacy ``device_map`` path unchanged.
    """
    return bool(getattr(config, "distributed", False)) and not bool(getattr(config, "use_lora", False))


def build_fsdp_kwargs(config) -> dict:
    """Translate a KORE config into HF ``TrainingArguments`` FSDP kwargs.

    Returns ``{}`` (i.e. *keep the current single-process / device_map path*)
    unless :func:`fsdp_enabled` is true. Otherwise returns::

        {"fsdp": "<sharding string>", "fsdp_config": { ... }}

    Notes:
      * Activation (gradient) checkpointing is routed through ``fsdp_config``
        rather than ``TrainingArguments.gradient_checkpointing`` — under
        ``full_shard`` the latter adds a redundant AllGather (HF warns about it).
      * ``cpu_ram_efficient_loading`` + ``sync_module_states`` let rank-0 stream
        the checkpoint and broadcast, which is what makes 32B/70B fit.
    """
    if not fsdp_enabled(config):
        return {}
    layer_cls = getattr(config, "fsdp_transformer_layer_cls", None) or detect_transformer_layer_cls(
        getattr(config, "model_id", "")
    )
    fsdp_str = getattr(config, "fsdp", "full_shard auto_wrap") or "full_shard auto_wrap"
    if getattr(config, "fsdp_cpu_offload", False) and "offload" not in fsdp_str:
        fsdp_str = f"{fsdp_str} offload"
    fsdp_config = {
        "transformer_layer_cls_to_wrap": [layer_cls],
        "activation_checkpointing": bool(getattr(config, "gradient_checkpointing", False)),
        "backward_prefetch": "backward_pre",
        "forward_prefetch": False,
        "use_orig_params": True,
        "sync_module_states": True,
        "cpu_ram_efficient_loading": True,
        "limit_all_gathers": True,
        "state_dict_type": "SHARDED_STATE_DICT",
    }
    if getattr(config, "fsdp_cpu_offload", False):
        fsdp_config["offload_params"] = True
    return {"fsdp": fsdp_str, "fsdp_config": fsdp_config}
