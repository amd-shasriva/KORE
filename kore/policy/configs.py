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
    """FSDP / distributed full-FT knobs shared by SFT, DPO & GRPO.

    These only take effect for **full fine-tuning** (``use_lora=False``) that was
    launched as a multi-process job (via ``scripts/launch_distributed.sh`` /
    ``accelerate launch``). A single-process run (CPU tests, single-GPU LoRA)
    ignores them entirely and keeps the legacy ``device_map="auto"`` path, so
    nothing changes for the LoRA recipe or for the CPU test suite.

    ``fsdp`` mirrors HF ``TrainingArguments.fsdp`` (e.g. ``"full_shard auto_wrap"``,
    ``"full_shard auto_wrap offload"``). ``fsdp_transformer_layer_cls`` names the
    decoder block to shard/wrap; when ``None`` it is auto-detected from
    ``model_id`` (Qwen3 / Qwen2 / Llama families - covers the 14B/32B/70B bases).
    """

    distributed: bool = False
    fsdp: str = "full_shard auto_wrap"
    fsdp_transformer_layer_cls: Optional[str] = None
    fsdp_cpu_offload: bool = False
    # Input-pipeline parallelism (audit THEME E / R2 perf): the HF default is 0 loader
    # workers + single-core tokenization, which starves 8 GPUs at 8k-16k seq. Sized for
    # the 384-core box (12/rank x 8 = 96 loader procs, well under 384 with room for the
    # trainers + preprocessing). These feed the Trainer stages (midtrain/sft/dpo);
    # GRPO's custom loop ignores them.
    dataloader_num_workers: int = 12
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 4
    dataset_num_proc: int = 32


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
    max_grad_norm: float = 1.0             # gradient-norm clip (HF default; now explicit)

    # THROUGHPUT (audit R2 perf): per_device batch 1 starved the MI350X matrix cores
    # (a 14B FSDP shard uses ~30GB of 288GB VRAM). Micro-batch 4 feeds the cores 4x
    # and cuts accumulation micro-steps 4x, at the SAME effective batch (4*4*8=128 =
    # the old 1*16*8) so the optimization recipe is unchanged. VRAM-safe with wide
    # margin even accounting for the bf16+fp32 LM-head logit term at seq 16384
    # (~117GB peak << 288GB). group_by_length (sft.py) minimises padding since packing
    # is off on the SDPA runtime.
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 16384
    bf16: bool = True
    gradient_checkpointing: bool = True
    packing: bool = False
    # Completion-only loss: mask prompt/user/system/tool tokens (label -100) and
    # train ONLY on assistant responses (+ their <|im_end|> stop). Implemented via
    # a {% generation %}-tagged chat template (built at train time, render-identical
    # to the base template) + TRL's assistant_only_loss. Standard SFT best-practice;
    # the base Qwen3 template lacks the marker, so sft.py injects it and self-verifies
    # (render-identity + non-empty masks) before training, and TRL raises if any
    # example ends up with no assistant tokens.
    assistant_only_loss: bool = True

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
    weight_decay: float = 0.0

    # THROUGHPUT (audit R2 perf): micro-batch 2 (was 1) at the SAME effective batch
    # (2*8*8=128 = the old 1*16*8). DPO concatenates chosen+rejected, so a micro-batch
    # of 2 processes 4 sequences/step -- 2x the matrix-core feeding -- while the
    # doubled logit term at seq 16384 stays VRAM-safe (~117GB peak << 288GB).
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    # NB: no ``max_prompt_length`` - trl>=0.29 removed it (the total ``max_length``
    # caps prompt+completion, and ``truncation_mode="keep_end"`` (dpo.py) keeps the
    # completion/kernel tail). GRPO keeps its own live ``max_prompt_length``.
    max_length: int = 16384
    bf16: bool = True
    gradient_checkpointing: bool = True

    use_lora: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    seed: int = 0
    logging_steps: int = 10
    save_steps: int = 200
    report_to: str = "none"


@dataclass
class GRPOConfig(DistributedMixin):
    """Stage-3 multi-turn GRPO (Kevin recipe) with anti-collapse ladder.

    Inherits :class:`DistributedMixin` (``distributed`` / ``fsdp`` /
    ``fsdp_transformer_layer_cls`` / ``fsdp_cpu_offload``) so a campaign can
    request full-FT GRPO under the FSDP launcher, exactly like SFT/DPO. The
    native single-process loop honors ``distributed`` by skipping
    ``device_map="auto"`` (accelerate/FSDP owns placement); LoRA / single-GPU /
    CPU runs keep the legacy ``device_map`` path untouched.
    """

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
    # NB: ``kl_coef`` was REMOVED - the only KL/anchor the native loop applies is
    # the k3 retention anchor ``ref_anchor_coef`` (see below); there was never a
    # second, separate KL coefficient, and the step log used to mislabel the
    # anchor as ``kl_coef``. Kevin's "KL = 0" is expressed by ``ref_anchor_coef``.
    # KORE's importance ratio is length-normalized (token-MEAN logprob), i.e. the
    # GSPO *sequence* ratio, not DAPO's per-token ratio - so DAPO's 0.2/0.28
    # token-level clip bounds never bind (a legit sequence ratio sits within ~1%
    # of 1.0). We therefore (a) run on-policy (ppo_epochs=1) so old_logp==new_logp
    # and the ratio is ~1 by construction (no off-policy drift to clip), and
    # (b) set sequence-scale clip bounds as defense-in-depth for any ppo_epochs>1.
    clip_ratio_low: float = 0.03           # GSPO sequence-ratio lower bound (was 0.2, inert)
    clip_ratio_high: float = 0.04          # asymmetric clip-higher, sequence scale
    adv_eps: float = 1e-6                  # group-normalization epsilon
    ppo_epochs: int = 1                    # on-policy (was 2): sequence-ratio clip can't
                                           # safely protect a 2nd off-policy epoch uncalibrated
    # DAPO Overlong Filtering: a response within overlong_buffer_len tokens of the
    # generation cap was (almost certainly) TRUNCATED; its per-token log-probs are
    # a noisy, biased gradient (the kernel is cut off mid-emit), so it is masked out
    # of the policy loss. Prevents length-hacking + truncation noise from stalling
    # the policy. Default on with a 512-token buffer (DAPO recipe).
    overlong_mask: bool = True
    overlong_buffer_len: int = 512

    # --- Dynamic training horizon (adaptive steps) --------------------------------
    # Instead of a fixed step count, keep training WHILE the monitored signal (the
    # rollout reward / held-out fast_p) is still climbing, and stop early once it
    # plateaus - so a run neither wastes compute after convergence nor stops before
    # the policy has actually moved. See kore.policy.dynamic.DynamicStepController.
    adaptive_steps: bool = False           # off by default (fixed total_steps)
    min_steps: int = 100                   # never stop before this many steps
    plateau_patience: int = 40             # stop after this many steps w/o improvement
    plateau_min_delta: float = 1e-3        # min reward gain that counts as improvement

    # --- Optimization (Kevin recipe) ---
    learning_rate: float = 2e-6            # Kevin: 2e-6
    lr_scheduler_type: str = "constant"    # WIRED: torch LambdaLR (constant|linear|cosine)
    warmup_ratio: float = 0.0              # WIRED: linear LR warmup over warmup_ratio*total_steps
    max_grad_norm: float = 0.5             # Kevin: grad-norm clip 0.5
    # NB: ``per_device_train_batch_size`` / ``gradient_accumulation_steps`` were
    # REMOVED - the native loop uses an O(1-sample) MICRO-BATCHED backward that
    # accumulates one grad term per rollout sample and steps once per PPO epoch,
    # so a per-device batch size + a separate accumulation count are meaningless
    # (accumulation is inherent, and the effective batch is the kept rollout set).
    total_steps: int = 500
    max_prompt_length: int = 16384         # WIRED: left-truncate the rendered prompt to this many tokens
    max_response_length: int = 16384       # Kevin: 16384
    bf16: bool = True                      # WIRED: bf16 vs fp32 model dtype (was hardcoded)
    gradient_checkpointing: bool = True

    # --- Rollout sampling (Kevin recipe) ---
    temperature: float = 0.9               # Kevin: 0.9
    top_p: float = 1.0                     # WIRED: passed to model.generate

    # NB: ``rollout_backend`` / ``tensor_parallel_size`` were REMOVED - KORE runs
    # ONE self-contained in-process transformers+PEFT loop on local AMD GPUs.
    # There is no vLLM rollout server and no tensor-parallel/distributed rollout
    # path in this loop, so those flags implied capabilities that don't exist.
    # (Distributed FULL-FT is handled by the FSDP fields from DistributedMixin.)

    # LoRA path removed from the GRPO trainer (full-parameter FT only); use_lora
    # defaults OFF and only gates FSDP selection. No `lora` config field: GRPO never
    # applies PEFT (the LoRAConfig type is still used by SFT/DPO/midtrain).
    use_lora: bool = False

    # --- Sharded FULL-PARAMETER distributed training (best-in-world RL) ---
    # These take effect ONLY for full-FT (``use_lora=False``) launched as a
    # multi-process job (``distributed=True`` via ``accelerate launch`` /
    # ``scripts/launch_distributed.sh grpo``). A single-process run (CPU tests,
    # single-GPU LoRA) ignores them entirely and keeps the legacy in-process path.
    #
    # ``sharding_backend`` selects how the POLICY (and frozen REFERENCE) are
    # sharded across ranks so no full replica ever lives on a single GPU:
    #   "fsdp"      -> torch FullyShardedDataParallel FULL_SHARD (ZeRO-3-equivalent);
    #   "deepspeed" -> DeepSpeed ZeRO-3 engine;
    #   "auto"      -> FSDP (the O(1-sample) micro-batched backward + ROCm-native
    #                  robustness make FSDP the default; see grpo.py docstring for
    #                  the ZeRO-3-vs-FSDP rationale). DeepSpeed ZeRO-3 is fully
    #                  wired and selectable via "deepspeed".
    sharding_backend: str = "auto"          # "auto" | "fsdp" | "deepspeed"
    fsdp_version: int = 1                    # torch FSDP version (1 = full_shard; 2 = FSDP2)
    zero_stage: int = 3                      # DeepSpeed ZeRO stage (3 = shard params+grads+optim)
    cpu_offload: bool = False                # offload params+optimizer to CPU (needed at 32B/70B)
    ds_config: Optional[str] = None          # explicit DeepSpeed JSON config path (overrides builder)
    # Generate in lockstep across ranks: REQUIRED under ZeRO-3/FSDP so ranks that
    # finish a rollout early keep issuing (dummy) forwards until every rank is done
    # - otherwise the collective per-forward all-gather deadlocks on ragged lengths.
    synced_gpus: bool = True

    # --- Anti-collapse ladder (see anticollapse.py) ---
    rc_grpo: bool = False                  # reward-conditioned rollouts (variance floor)
    rc_p_high: float = 0.5                 # fraction of <|high_reward|> tokens
    # AVSPO virtual-sample injection: when a group's reward std < variance_floor,
    # inject ``avspo_virtual_k`` virtual samples into the NORMALIZATION stats only
    # (no PG term) to guarantee a variance floor. 0.0 disables (pure GRPO).
    # AVSPO variance floor. Default 0.1 (not 0) as defense-in-depth against the
    # (r-mean)/(std+eps) blow-up: a group whose std is tiny but above starpo_min_std
    # would otherwise get ~1/std (huge) advantages and spike the gradient on a long
    # run. The floor bounds aug_std >= sqrt(k*tau^2/(n+k)) > 0. (The campaign already
    # sets 0.1 when anti-collapse is on; this makes standalone GRPO safe too.)
    variance_floor: float = 0.1            # AVSPO tau trigger (0 disables)
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

    # --- Open-ended verified co-evolution curriculum ---
    # When True, task selection is driven by the frontier proposer (learnability
    # p(1-p) + performance-headroom regret + MAP-Elites novelty) over the archive,
    # co-evolving the curriculum with the policy, instead of fixed round-robin.
    # Grounded to the registered task list (only runnable task_ids are proposed).
    coevolve: bool = False
    coevolve_batch: Optional[int] = None    # frontier proposals per refill (default: menu size, capped)
    coevolve_include_vendor: bool = True    # include vendor-baselined ops in the space
    # Distillation sink: append verified >=min_speedup winning kernels discovered
    # during co-evolution to this JSONL (WinRecords) for expert-iteration/RFT reuse.
    coevolve_distill_path: Optional[str] = None
    coevolve_distill_min_speedup: float = 1.0

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

    # --- Physics residual-descent reward (P0 paradigm) ---
    # "speedup" (default): classic vendor-relative reward. "residual": score a
    # correct kernel by the physics residual to the roofline T_min (dense, ABSOLUTE,
    # arch-normalized eta = T_min/T_measured) via kore.reward.physics -- the P0
    # paradigm signal that keeps intra-group advantage non-zero where the flat
    # speedup reward collapses. Anti-hack/correctness gating is identical; ops with
    # no roofline model transparently fall back to the speedup reward. physics_weight
    # scales the residual credit on the correct tier.
    reward_mode: str = "speedup"
    physics_weight: float = 1.0

    # --- Agentic tool-use RL (ToolRL reward shaping) ---
    agentic: bool = False                   # rollouts drive build/test/bench/pmc tools
    tool_reward_weight: float = 0.2         # weight on ToolRL-style shaping term
    max_tool_turns: int = 8

    seed: int = 0
    logging_steps: int = 1                  # WIRED: emit the per-step metrics event every N steps
    save_steps: int = 50                    # WIRED: write a periodic checkpoint every N steps
    report_to: str = "none"


@dataclass
class MidTrainConfig(DistributedMixin):
    """Stage-0 continued pretraining on the ROCm/HIP/Triton corpus.

    Inherits the FSDP/distributed full-FT knobs (:class:`DistributedMixin`) so the
    locked full-FT recipe shards across GPUs exactly like SFT/DPO/GRPO when the
    campaign shells it out under ``accelerate launch`` (``distributed=True`` +
    ``use_lora=False``). LoRA / single-GPU smoke runs ignore them.
    """

    model_id: str = MODEL_14B
    corpus_path: str = "data/midtrain/corpus.jsonl"
    output_dir: str = "runs/midtrain"
    # CPT replay: Triton/HIP is FAR from the base LM distribution, so the forgetting
    # literature (Ibrahim et al. 2024; DeepSeek-V2's 30% replay) says a large-shift
    # domain needs ~30-35% general replay + LR re-warm/re-decay to acquire the domain
    # WITHOUT wrecking general/chat ability. 0.30 (was 0.15).
    general_replay_frac: float = 0.30
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.05
    max_seq_length: int = 8192
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_lora: bool = False                  # full-FT (large ROCm distribution shift)

    # Batch / optimization / checkpoint knobs (previously hardcoded in midtrain.py).
    # Exposing them makes a launch JSON authoritative and - critically - bounds the
    # checkpoint count (``save_total_limit``) so a 14B full-FT CPT run cannot fill
    # disk (the same guard SFT/DPO already have).
    # THROUGHPUT (audit R2 perf): micro-batch 4 (was 1) feeds the MI350X matrix cores
    # 4x at the SAME effective batch (4*4*8=128 = the old 1*16*8). CPT chunks are
    # uniform seq 8192, so padding is negligible; VRAM-safe with wide margin (~75GB
    # peak << 288GB even with the LM-head logit term).
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    # packing MUST be False on the SDPA runtime: TRL bfd packing needs a flash-attn
    # backend to build the block-diagonal mask, and silently falls back to plain SDPA
    # (cross-document attention contamination) when flash-attn is absent -- which it is
    # on this ROCm stack. False = each doc is a correctly-masked padded example (audit
    # THEME B/C2). Re-enable only once flash_attention_2/flex_attention is wired.
    packing: bool = False
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 1              # a 14B full-FT ckpt is ~220GB w/ optimizer; cap to avoid disk-fill
    seed: int = 0


@dataclass
class MultiCapSFTConfig(SFTConfig):
    """Stage-1 multi-capability SFT mixture (kernel + general + agentic).

    Fractions of the training mix (must sum ~1.0). ~45% general = the retention
    backbone; ~10% agentic tool-use trajectories = the orchestration skill.
    """

    # Dual-capability SFT mix (research-tuned for a model that is SIMULTANEOUSLY a
    # kernel generator, a chat model, and an orchestrator). ~38% kernel / ~12%
    # agentic / ~50% general (chat+code+math). The large, co-mixed general plurality
    # (esp. chat 0.27) is what keeps IFEval/MT-Bench up while the kernel+agentic
    # slices specialize - the literature-backed guard against specialization collapse.
    frac_kernel_repair_opt: float = 0.28   # was 0.35
    frac_kernel_qa: float = 0.10
    frac_agentic_tooluse: float = 0.12     # was 0.10
    frac_general_code: float = 0.13        # was 0.20 (now REAL code instruct data)
    frac_math_reasoning: float = 0.10      # was 0.15 (now REAL CoT math data)
    frac_general_chat: float = 0.27        # was 0.10 -> the "talk to it" backbone
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
# FSDP wiring helpers (pure - no torch/transformers, safe on CPU / in tests)
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
      * Activation (gradient) checkpointing is enabled by the Trainer stage via
        HF's ``TrainingArguments.gradient_checkpointing`` +
        ``gradient_checkpointing_kwargs={"use_reentrant": True}`` (layer-internal,
        FSDP-safe) - NOT via ``fsdp_config``. The FSDP-plugin (external
        checkpoint_wrapper) path mismatches saved-tensor counts on an
        FSDP1/``use_orig_params`` unit and raises ``CheckpointError``. REENTRANT is
        deliberate: the ROCm stack has no ``flash_attn`` wheel so training runs on
        SDPA, whose fused-kernel choice can differ between the checkpointed forward
        and its recompute (and across ranks); reentrant checkpointing skips the
        saved-tensor-count check that non-reentrant enforces and would otherwise
        raise on that swap. (This is the single source of truth for the choice; the
        stage entrypoints match it.)
      * ``cpu_ram_efficient_loading`` + ``sync_module_states`` let rank-0 stream
        the checkpoint and broadcast, which is what makes 32B/70B fit.
      * fp32 MASTER weights are automatic: the model is loaded in bf16 (compute
        dtype), but with accelerate ``mixed_precision: bf16`` the FSDP prepare step
        UPCASTS every trainable flat-parameter to fp32 (accelerate mimics
        DeepSpeed ZeRO's fp32 partition - see ``Accelerator.prepare_model``), so the
        optimizer steps on an fp32 master while forward/backward run in bf16. Do NOT
        load the model in fp32 to "get" a master - it is already fp32; fp32 loading
        only doubles load-time host memory for an identical result. (Gradient
        reduce-scatter stays bf16, matching the DeepSpeed default.)
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
        # NOTE: activation checkpointing is intentionally NOT set here. Driving it
        # from the FSDP plugin (accelerate's external checkpoint_wrapper) on an
        # FSDP1 + use_orig_params unit mismatches the saved-tensor count between
        # forward and recompute (torch.utils.checkpoint CheckpointError "different
        # number of tensors ..."). Instead each Trainer stage enables HF's own
        # layer-internal gradient checkpointing (use_reentrant=True - SDPA-swap
        # robust; see build_fsdp_kwargs docstring), which wraps the decoder block's
        # forward and is FSDP-safe. See build_fsdp_kwargs docs.
        "backward_prefetch": "backward_pre",
        # forward_prefetch=True overlaps the NEXT layer's all-gather with the current
        # layer's compute in the forward pass -- a free throughput win on the 8-GPU
        # FSDP mesh where the model comfortably fits (audit R2 perf). Safe with the
        # huge 288GB VRAM headroom (the extra prefetched shard is tiny).
        "forward_prefetch": True,
        "use_orig_params": True,
        "sync_module_states": True,
        "cpu_ram_efficient_loading": True,
        "limit_all_gathers": True,
        # FULL_STATE_DICT so ``trainer.save_model()`` consolidates a plain HF
        # checkpoint that the NEXT stage loads with ``from_pretrained``
        # (midtrain->sft->dpo->grpo->soup handoff) and that serving can load -
        # matching GRPO's own save path. A sharded state dict is only reloadable
        # under an identical FSDP mesh, which the cross-stage handoff is not. At
        # 14B the rank-0 gather is cheap; for 32B/70B keep it consolidated too
        # (cpu_ram_efficient_loading streams it) so the handoff never breaks.
        "state_dict_type": "FULL_STATE_DICT",
    }
    if getattr(config, "fsdp_cpu_offload", False):
        fsdp_config["offload_params"] = True
    return {"fsdp": fsdp_str, "fsdp_config": fsdp_config}


def latest_checkpoint(output_dir) -> Optional[str]:
    """Latest resumable HF ``checkpoint-<step>`` dir in ``output_dir``, else None.

    Wired into the Trainer stages so ``trainer.train(resume_from_checkpoint=...)``
    auto-resumes a crashed mid-stage run (the campaign's stage-skip only resumes
    *completed* stages; without this a crash at step 40k of SFT or step N of a
    multi-hour run restarts from zero). Returns None (fresh run) when absent.
    """
    import glob
    import os
    ckpts = glob.glob(os.path.join(str(output_dir), "checkpoint-*"))
    ckpts = [c for c in ckpts if os.path.isdir(c)]
    if not ckpts:
        return None

    def _step(p: str) -> int:
        try:
            return int(os.path.basename(p).rsplit("-", 1)[1])
        except (ValueError, IndexError):
            return -1

    latest = max(ckpts, key=_step)
    # Only resume if the checkpoint actually has trainer state (not a half-written dir).
    if os.path.exists(os.path.join(latest, "trainer_state.json")):
        return latest
    return None


def preferred_attn_impl() -> str:
    """Attention backend for training model loads (``from_pretrained``).

    Prefer FlashAttention-2 when the ROCm ``flash_attn`` wheel is importable. This
    is not just a speed/memory win: SDPA transparently switches between fused
    kernels (flash / mem-efficient / math) depending on shape and free memory, and
    that choice can differ between the checkpointed forward and its recomputation
    (and across ranks), so the NON-REENTRANT activation checkpoint sees a DIFFERENT
    saved-tensor count and raises ``CheckpointError`` (observed intermittently on
    the 8-GPU FSDP full-FT path). FlashAttention-2 saves a FIXED tensor set on
    every forward/recompute and on every rank, which makes gradient checkpointing
    deterministic. Falls back to ``"sdpa"`` when the wheel is absent (e.g. CPU
    tests never reach a real model load anyway).
    """
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:  # noqa: BLE001 - any import problem -> safe SDPA fallback
        return "sdpa"


# --------------------------------------------------------------------------- #
# Sharded full-parameter GRPO wiring (pure - no torch/accelerate/deepspeed here,
# so ``kore.policy.configs`` still imports on CPU with NO heavy deps). The heavy
# accelerate/DeepSpeed plugin objects are built lazily inside ``kore.policy.grpo``.
# --------------------------------------------------------------------------- #
def grpo_distributed_enabled(config) -> bool:
    """True iff GRPO should take the sharded FULL-PARAMETER distributed path.

    Identical gate to :func:`fsdp_enabled`: the sharded path is used ONLY for full
    fine-tuning (``use_lora=False``) launched as a multi-process job
    (``distributed=True``). LoRA and single-process runs keep the legacy
    in-process ``device_map`` loop unchanged, so every CPU/LoRA test is untouched.
    """
    return fsdp_enabled(config)


def grpo_sharding_backend(config) -> str:
    """Resolve which sharded backend the distributed GRPO loop uses.

    Returns ``"none"`` when this is NOT a distributed full-FT run (keep the legacy
    in-process path). Otherwise honors ``config.sharding_backend``:
      * ``"deepspeed"`` -> DeepSpeed ZeRO-3;
      * ``"fsdp"``/``"fsdp2"`` -> torch FullyShardedDataParallel FULL_SHARD;
      * ``"auto"`` -> ``"fsdp"`` (the default). FSDP is chosen because KORE's
        O(1-sample) micro-batched backward (many per-sample ``backward()`` then one
        ``optimizer.step()``) maps cleanly onto FSDP's grad reduce-scatter +
        accumulate, whereas DeepSpeed's engine couples ``backward``/``step`` to a
        fixed accumulation counter; FSDP is also torch-native (guaranteed ROCm
        support, no compiled ops). DeepSpeed ZeRO-3 stays fully wired for anyone
        who sets ``sharding_backend="deepspeed"``.

    Uses ``importlib.util.find_spec`` (no import) so this stays torch/deepspeed
    free and safe to call on CPU / in tests.
    """
    if not grpo_distributed_enabled(config):
        return "none"
    want = (getattr(config, "sharding_backend", "auto") or "auto").lower()
    if want == "deepspeed":
        return "deepspeed"
    if want in ("fsdp", "fsdp2", "fsdp1"):
        return "fsdp"
    # "auto" (and any unknown value): prefer FSDP for the micro-batched RL loop.
    return "fsdp"


def build_deepspeed_config(config) -> dict:
    """Build a DeepSpeed ZeRO config dict for the sharded full-FT GRPO RL loop.

    ZeRO-3 shards params + grads + optimizer state across ranks and - critically
    for the online generate->train loop - GATHERS params per-forward, so
    ``model.generate`` (rollouts) works out of the box on the sharded engine (the
    property that makes ZeRO-3 the natural online-RL sharding choice, used by TRL).

    If ``config.ds_config`` points at a JSON file it is loaded and returned
    verbatim (full user control). Otherwise a ZeRO-``zero_stage`` config is
    synthesized from the KORE knobs (``bf16``, ``cpu_offload``, ``max_grad_norm``).
    Pure/JSON only - no torch/deepspeed import - so it is unit-testable on CPU.
    """
    import json as _json

    ds_path = getattr(config, "ds_config", None)
    if ds_path:
        with open(ds_path) as f:
            return _json.load(f)

    stage = int(getattr(config, "zero_stage", 3))
    offload = bool(getattr(config, "cpu_offload", False))
    bf16 = bool(getattr(config, "bf16", True))
    zero: dict = {
        "stage": stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
    }
    if stage == 3:
        zero.update({
            # gather the fp16/bf16 shards into a full state dict at save time so a
            # plain (un-sharded) checkpoint is written for soup/serve/eval.
            "stage3_gather_16bit_weights_on_model_save": True,
            "stage3_param_persistence_threshold": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_max_live_parameters": int(1e9),
            "stage3_max_reuse_distance": int(1e9),
        })
    if offload:
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    return {
        # the native loop owns micro-batching (one backward per rollout sample),
        # so DeepSpeed sees a micro-batch of 1 and no internal accumulation.
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": float(getattr(config, "max_grad_norm", 1.0)),
        "bf16": {"enabled": bf16},
        "fp16": {"enabled": False},
        "zero_optimization": zero,
    }
