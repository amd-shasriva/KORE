"""Discover and load KORE tasks from ``kore/tasks/<id>/task.yaml``.

Also defines the **train / held-out generalization split**. A whole operator
*family* (plus any arch-specific task) is deterministically RESERVED as held-out
so that training data-generation never sees it and eval measures generalization
to an unseen family - mirroring KernelBench/GEAK's train-vs-heldout discipline.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

from kore.tasks.base import Task

TASKS_DIR = Path(__file__).resolve().parent

# The PRIMARY arch we train on = the KORE target hardware, gfx950 / CDNA4 (AMD
# Instinct MI350X / MI355X). New records are tagged with this.
TRAIN_ARCH = "gfx950"

# Arches ACCEPTED into the train set. gfx942/CDNA3 is retained alongside gfx950 so
# that (a) tasks/records tagged with the previous-gen label are NOT retroactively
# held out when the primary arch advances to gfx950 (they are the same hardware
# lineage and run correctly on this CDNA4 node), and (b) a mid-flight campaign's
# already-generated gfx942-tagged data keeps training. A truly foreign arch
# (e.g. gfx1100 / NVIDIA) is still held out. Override via KORE_TRAIN_ARCHS.
import os as _os
TRAIN_ARCHS: frozenset = frozenset(
    _os.environ.get("KORE_TRAIN_ARCHS", "gfx950,gfx942").split(","))

# Whole operator families reserved for the held-out generalization set. Core
# attention (flash prefill/decode/varlen/fp8) TRAINS for product capability, but the
# structurally-distinct MLA (latent attention) and paged-KV decode families are
# reserved WHOLE -- so any generated/mined variant (not just the two seed task ids in
# HELDOUT_TASKS) is kept out of TRAIN by family, closing the last MLA/paged leakage
# path (audit R2). No core-attention task matches these (they lack "mla"/"paged").
HELDOUT_FAMILIES: tuple[str, ...] = ("mla", "paged_attention")

# Specific TASKS reserved for the held-out generalization eval (never trained on).
# The policy trains on core attention (prefill / decode / sliding-window / varlen /
# fp8) so the product model is strong at attention, but these structurally-distinct
# variants are withheld to measure TRUE generalization: paged-KV decode (a different
# KV-cache mechanism) and MLA (DeepSeek latent attention, the hardest novel variant).
# This is the "best product model AND a frontier generalization eval" split.
HELDOUT_TASKS: frozenset = frozenset({
    # (1) FAR generalization - structurally novel seeds (also held out WHOLE by
    #     HELDOUT_FAMILIES): paged-KV decode + MLA (DeepSeek latent attention).
    "paged_attn_decode_bf16",
    "mla_decode_bf16",
    # (2) NEAR generalization - a STRATIFIED, deterministic ~43-task breadth probe:
    #     3-4 genb_ tasks per major family, picked at even positions in each sorted
    #     family so the policy trains on the REST of the family and is tested on an
    #     UNSEEN member. Spans every op class + dtype (bf16/fp16/fp32/fp8/int8/int4),
    #     turning the generalization eval from a 2-point claim into a real one.
    "genb_attn2_cross_gqa_step_fp16",
    "genb_attn2_decode_mqa_hd256_bf16",
    "genb_attn2_varlen_gqa_causal_bf16",
    "genb_attn_fp8_mha_hd128_causal_fp8",
    "genb_attn_mha_hd128_noncausal_fp16",
    "genb_attn_mqa_hd64_causal_bf16",
    "genb_cv_conv2d_1x1_s2_fp16",
    "genb_cv_conv2d_7x7_s1_d1_fp16",
    "genb_cv_conv2d_nhwc_5x5_s1_d1_bf16",
    "genb_cv_depthwise_conv2d_5x5_s1_bf16",
    "genb_fx_embed_scale_bf16",
    "genb_fx_reglu_act_fp16",
    "genb_fx_rope_qk_half_qknorm_bf16",
    "genb_gemm_bf16_residual_bf16",
    "genb_gemm_fp8_channelwise_fp8",
    "genb_gemm_int4_asym_group_fp16",
    "genb_gemm_int8_pertensor_int8",
    "genb_moe_block_silu_k8_e256_bf16",
    "genb_moe_fused_moe_silu_fp16",
    "genb_moe_grouped_mlp_gelu_bf16",
    "genb_moe_sigmoid_topk_norenorm_fp16",
    "genb_norm_groupnorm_bf16",
    "genb_norm_layernorm_bwd_fp32",
    "genb_norm_layernorm_quant_fp8_fp8",
    "genb_norm_rmsnorm_h16384_bf16",
    "genb_qx_int4_unpack_group_bf16",
    "genb_qx_quant_fp8_block2d_fp8",
    "genb_qx_quant_int8_pertoken_int8",
    "genb_red_entropy_bf16",
    "genb_red_log_softmax_bwd_fp32",
    "genb_red_rms_bf16",
    "genb_red_topk256_bf16",
    "genb_smp_repetition_penalty_bf16",
    "genb_smp_rope_yarn_bf16",
    "genb_smp_topk_sample_bf16",
    "genb_ssm_gated_retention_c128_bf16",
    "genb_ssm_lightning_attn_bf16",
    "genb_ssm_mamba2_ssd_c128_n128_bf16",
    "genb_ssm_retnet_c64_bf16",
    "genb_tr_adamw_8bit_bf16",
    "genb_tr_foreach_sgd_bf16",
    "genb_tr_ls_ce_bwd_bf16",
    "genb_tr_rmsprop_centered_momentum_fp32",
})


def operator_family(task: Task) -> str:
    """Coarse operator family for a task (used for the generalization split)."""
    op = (getattr(task, "operation", None) or getattr(task, "task_id", "") or "").lower()
    # MLA (DeepSeek latent attention) and paged-KV decode are the held-out
    # generalization probes -- classify them as their OWN families (checked BEFORE the
    # generic "attn" catch, since "paged_attn_decode" also contains "attn") so any
    # variant record is grouped + held out by family, not just the two exact task ids
    # (audit R2: MLA/paged had no family class and fell through to a raw-string family).
    if "mla" in op or "latent_attn" in op or "latent_attention" in op:
        return "mla"
    if "paged" in op:
        return "paged_attention"
    if "attn" in op or "attention" in op:
        return "attention"
    if "topk" in op:
        return "moe_router"
    if "moe" in op:
        return "moe"
    if "rmsnorm" in op:
        return "rmsnorm"
    if "layernorm" in op:
        return "layernorm"
    if "gemm" in op or "matmul" in op:
        return "gemm"
    if "quant" in op:
        return "quant"
    if "rope" in op:
        return "rope"
    if "softmax" in op:
        return "softmax"
    if "gelu" in op or "silu" in op or "relu" in op:
        return "activation"
    return op or "other"


def is_heldout(task: Task) -> bool:
    """Held out if the task is individually reserved, its family is reserved, OR it
    targets a FOREIGN arch (outside TRAIN_ARCHS -- gfx950/gfx942 lineage)."""
    if getattr(task, "task_id", "") in HELDOUT_TASKS:
        return True
    if operator_family(task) in HELDOUT_FAMILIES:
        return True
    if (getattr(task, "gpu_target", None) or TRAIN_ARCH) not in TRAIN_ARCHS:
        return True
    return False


@lru_cache(maxsize=1)
def _discover() -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    for yml in sorted(TASKS_DIR.glob("*/task.yaml")):
        try:
            t = Task.from_dir(yml.parent)
            tasks[t.task_id] = t
        except Exception as e:  # noqa: BLE001
            print(f"[registry] skip {yml.parent.name}: {e}")
    return tasks


def all_tasks() -> list[Task]:
    return list(_discover().values())


def task_ids() -> list[str]:
    return list(_discover().keys())


def get_task(task_id: str) -> Task:
    tasks = _discover()
    if task_id not in tasks:
        raise KeyError(f"unknown task '{task_id}'; known: {sorted(tasks)}")
    return tasks[task_id]


# --------------------------------------------------------------------------- #
# Train / held-out generalization split
# --------------------------------------------------------------------------- #
def heldout_tasks() -> list[Task]:
    """Tasks RESERVED for held-out generalization eval (never seen in training).

    Deterministic and independent of any seed - the reserved set is a function of
    the operator family + arch alone, so training data-gen can safely exclude it.
    """
    return [t for t in all_tasks() if is_heldout(t)]


def train_tasks() -> list[Task]:
    """Tasks available for training data-generation (complement of held-out)."""
    return [t for t in all_tasks() if not is_heldout(t)]


def heldout_families() -> set[str]:
    """The operator families that actually appear in the held-out split."""
    return {operator_family(t) for t in heldout_tasks()}


def split_tasks(seed: int = 0) -> dict[str, object]:
    """Deterministic train/held-out split.

    The held-out set is FIXED (reserved operator families + arch-specific tasks)
    regardless of ``seed`` so training can never leak into it; ``seed`` only
    controls the reproducible ordering WITHIN each split (useful for sharding or
    cross-validation folds). Returns
    ``{"train": [...], "heldout": [...], "seed": seed}``.
    """
    train = sorted(train_tasks(), key=lambda t: t.task_id)
    held = sorted(heldout_tasks(), key=lambda t: t.task_id)
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(held)
    return {"train": train, "heldout": held, "seed": seed}
