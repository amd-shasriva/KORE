"""Stage-2 DPO on ranked preference pairs.

Trains against a frozen reference (the SFT checkpoint) with ``trl``'s
``DPOTrainer``. Plan hyperparameters (see ``configs.DPOConfig``): beta = 0.1,
ref = SFT, bf16, max_len 16384. Full-FT vs LoRA is governed by
``config.use_lora``; a LoRA adapter is merged into the base before saving.

Dataset is the *conversational* preference JSONL produced by
``kore.data.build_datasets.build_dpo``::

    {"prompt":   [ {"role": "system", ...}, {"role": "user", ...} ],
     "chosen":   [ {"role": "assistant", "content": "FULL_KERNEL:..."} ],
     "rejected": [ {"role": "assistant", "content": "FULL_KERNEL:..."} ]}

``trl.DPOTrainer`` consumes this conversational shape natively (it applies the
chat template per column). Legacy string rows (``prompt``/``chosen``/``rejected``
as plain text) are still accepted for backward compatibility. Heavy imports are
guarded.
"""

from __future__ import annotations

import json
from typing import Any

from kore.obs import get_logger, gpu_mem_snapshot
from kore.policy.configs import (
    DPOConfig,
    LoRAConfig,
    build_fsdp_kwargs,
)

log = get_logger("policy.dpo")


def _pair_token_stats(rows: list[dict], tokenizer, sample: int = 512) -> dict:
    """Best-effort chosen/rejected chat-token length stats (logging; read-only)."""
    try:
        def _n(msgs):
            return len(tokenizer.apply_chat_template(msgs, tokenize=True,
                                                     add_generation_prompt=False))

        ch, rj = [], []
        for r in rows[:sample]:
            try:
                ch.append(_n(r["prompt"] + r["chosen"]) if isinstance(r["chosen"], list)
                          else len(tokenizer(str(r["chosen"])).get("input_ids", [])))
                rj.append(_n(r["prompt"] + r["rejected"]) if isinstance(r["rejected"], list)
                          else len(tokenizer(str(r["rejected"])).get("input_ids", [])))
            except Exception:  # noqa: BLE001 - skip a row that won't render
                continue
        out = {"n_pairs": len(rows), "tok_sampled": len(ch)}
        if ch:
            out["chosen_tok_mean"] = round(sum(ch) / len(ch), 1)
            out["chosen_tok_max"] = max(ch)
        if rj:
            out["rejected_tok_mean"] = round(sum(rj) / len(rj), 1)
            out["rejected_tok_max"] = max(rj)
        return out
    except Exception as e:  # noqa: BLE001 - advisory only
        return {"n_pairs": len(rows), "tok_stats_error": repr(e)}


def build_trl_dpo_kwargs(config) -> dict:
    """Assemble the kwargs for ``trl.DPOConfig`` from a KORE :class:`DPOConfig`.

    PURE (imports no torch/trl) so the IPO / cDPO + FSDP config path is unit-
    testable on CPU. Two knobs counter deterministic-preference overfitting on
    hard negatives (a real risk once preferences come from on-policy relabeling):

      * ``loss_type`` — the trl DPO loss variant. ``"ipo"`` uses the IPO
        (bounded, MSE-style) objective which does not push the implicit reward gap
        to infinity on easy/near-deterministic pairs. Others: ``"sigmoid"``
        (vanilla DPO, default), ``"hinge"``, etc.
      * ``label_smoothing`` — conservative-DPO (cDPO): treats preference labels as
        soft (``1 - eps`` / ``eps``), so a noisy/mislabeled hard negative can't
        dominate the gradient.

    Both are read via ``getattr`` so a plain ``DPOConfig`` (which has neither
    field) still works — set them as attributes (per round, alongside a refreshed
    ``ref_model_id``) for iterative DPO. FSDP kwargs are merged for the
    distributed full-FT path; activation checkpointing uses HF's layer-internal
    REENTRANT path (robust to the intermittent flash_attention_2 -> SDPA per-worker
    downgrade; reentrant skips the saved-tensor-count check that NON-REENTRANT does
    and that raises CheckpointError when SDPA swaps kernels), NOT ``fsdp_config``
    (the external wrapper is the other source of the mismatch on FSDP1)."""
    kwargs = dict(
        output_dir=config.output_dir,
        beta=config.beta,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        bf16=config.bf16,
        gradient_checkpointing=bool(config.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": True},
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        seed=config.seed,
        report_to=config.report_to,
    )
    loss_type = getattr(config, "loss_type", None)
    if loss_type:
        kwargs["loss_type"] = str(loss_type)
    label_smoothing = getattr(config, "label_smoothing", None)
    if label_smoothing:
        kwargs["label_smoothing"] = float(label_smoothing)
    kwargs.update(build_fsdp_kwargs(config))
    return kwargs


def load_preference_jsonl(path: str) -> list[dict]:
    """Read preference rows: ``{"prompt", "chosen", "rejected"}`` per line.

    Consumes the CONVERSATIONAL DPO schema where ``prompt`` is a chat-message
    list and ``chosen`` / ``rejected`` are each a single-message assistant
    completion list (``[{"role": "assistant", "content": ...}]``). Legacy rows
    where any field is a plain string are passed through unchanged so trl's
    standard (non-conversational) path still works. Lines missing any of the
    three fields are skipped.
    """
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "prompt" in d and "chosen" in d and "rejected" in d:
                rows.append({
                    "prompt": d["prompt"],
                    "chosen": d["chosen"],
                    "rejected": d["rejected"],
                })
    return rows


def train(config: DPOConfig) -> dict:
    """Run DPO LoRA training from ``config.dataset_path`` (preference JSONL)."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    rows = load_preference_jsonl(config.dataset_path)
    if not rows:
        raise ValueError(f"no usable preference pairs in {config.dataset_path!r}")

    # Distributed full-FT (FSDP) vs the legacy single-process / LoRA path. FSDP is
    # only for full-FT (use_lora=False) launched distributed; LoRA and single
    # process keep the current path. DPO never set device_map, so nothing to strip
    # there — under FSDP the model is loaded plain and the Trainer wraps it.
    fsdp_kwargs = build_fsdp_kwargs(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("dpo: dataset loaded", dataset=config.dataset_path, model=config.model_id,
             use_lora=bool(config.use_lora), beta=config.beta, epochs=config.num_train_epochs,
             distributed=bool(config.distributed), fsdp=bool(fsdp_kwargs),
             ref_model_id=getattr(config, "ref_model_id", None),
             loss_type=getattr(config, "loss_type", None),
             label_smoothing=getattr(config, "label_smoothing", None),
             **_pair_token_stats(rows, tokenizer))

    dataset = Dataset.from_list(rows)

    dtype = torch.bfloat16 if config.bf16 else torch.float32
    # DPO uses SDPA, NOT flash_attention_2. DPO concatenates chosen+rejected into a
    # right-PADDED batch (unlike SFT's packed/varlen inputs), and the ROCm
    # FlashAttention-2 kernel hard-faults on that padded layout ("Memory access
    # fault by GPU node-N" at the first step). SDPA handles the padded batch
    # correctly; combined with reentrant gradient checkpointing it is also immune
    # to the saved-tensor-count check, and its mem-efficient backend stays O(seq)
    # in memory at long context.
    attn_impl = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=dtype,
                                                 attn_implementation=attn_impl)
    # Activation checkpointing (routed through fsdp_config) is INCOMPATIBLE with
    # the KV cache: the cache changes the tensor count between forward and
    # recompute -> torch.utils.checkpoint CheckpointError. HF's Trainer only
    # auto-disables use_cache when TrainingArguments.gradient_checkpointing is set,
    # which we turn OFF under FSDP (FSDP owns checkpointing), so disable it here.
    model.config.use_cache = False

    # Reference model. For full-FT we ALWAYS load an EXPLICIT reference (the frozen
    # ``ref_model_id`` for iterative DPO, else ``model_id`` — the SFT/prev-round
    # checkpoint). trl then FSDP-shards + freezes it via ``accelerator.prepare_model``.
    # We must NOT leave ref_model=None under full-FT: trl would then deep-copy the
    # policy into an UNSHARDED reference (a second full 14B on ONE GPU) -> ROCm
    # "Memory access fault" at the first step. With LoRA, trl uses the frozen base
    # (adapter-disabled) as the implicit reference, so we pass no ref_model there
    # (an explicit ref alongside a peft_config is unsupported by trl).
    ref_model = None
    if not config.use_lora:
        ref_id = config.ref_model_id or config.model_id
        ref_model = AutoModelForCausalLM.from_pretrained(ref_id, torch_dtype=dtype,
                                                         attn_implementation=attn_impl)

    peft_config = _peft_config(config) if config.use_lora else None

    trl_args = TRLDPOConfig(**build_trl_dpo_kwargs(config))

    # Lightweight per-log-step observability callback (guarded transformers import).
    from transformers import TrainerCallback

    class _ObsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs:  # skip eval/summary logs — train-step only
                return
            log.event("dpo_step", step=int(state.global_step), loss=logs.get("loss"),
                      lr=logs.get("learning_rate"),
                      epoch=round(float(state.epoch), 4) if state.epoch is not None else None,
                      grad_norm=logs.get("grad_norm"),
                      rewards_margin=logs.get("rewards/margins"),
                      rewards_acc=logs.get("rewards/accuracies"), **gpu_mem_snapshot())

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=trl_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[_ObsCallback()],
    )
    trainer.train()

    # Merge the LoRA adapter into the base before saving so downstream stages
    # (GRPO, soup) load a plain full model; full-FT saves weights directly.
    if config.use_lora:
        log.info("dpo: merging LoRA adapter into base", out=config.output_dir)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(config.output_dir)
    else:
        log.info("dpo: saving full-FT model", out=config.output_dir)
        trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    log.metric("dpo_done", out=config.output_dir, n_pairs=len(rows),
               merged_lora=bool(config.use_lora), **gpu_mem_snapshot())
    return {"stage": "dpo", "output_dir": config.output_dir, "n_pairs": len(rows)}


def _peft_config(config: Any):
    from peft import LoraConfig

    lc = config.lora
    return LoraConfig(
        r=lc.r,
        lora_alpha=lc.lora_alpha,
        lora_dropout=lc.lora_dropout,
        bias=lc.bias,
        task_type=lc.task_type,
        target_modules=lc.target_modules,
    )


# --------------------------------------------------------------------------- #
# Distributed entry: `python -m kore.policy.dpo <config.json>`
#
# Used by scripts/launch_distributed.sh under `accelerate launch`. Pure-stdlib
# JSON parsing (no torch at import time). The JSON is a flat map of DPOConfig
# fields (including "dataset_path"), with an optional nested "lora" object.
# --------------------------------------------------------------------------- #
def dpo_config_from_dict(d: dict) -> DPOConfig:
    """Build a :class:`DPOConfig` from a plain dict (nested ``lora`` supported).

    ``loss_type`` / ``label_smoothing`` (IPO / cDPO knobs, not fields on the base
    dataclass) are accepted here and attached as attributes so the JSON config
    path — and per-round iterative DPO — can select them without a schema change.
    """
    d = dict(d)
    lora = d.pop("lora", None)
    loss_type = d.pop("loss_type", None)
    label_smoothing = d.pop("label_smoothing", None)
    cfg = DPOConfig(**d)
    if lora is not None:
        cfg.lora = LoRAConfig(**lora)
    if loss_type is not None:
        cfg.loss_type = loss_type
    if label_smoothing is not None:
        cfg.label_smoothing = label_smoothing
    return cfg


def _main(argv: list | None = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m kore.policy.dpo <config.json>", file=sys.stderr)
        return 2
    raw = json.loads(open(argv[0]).read())
    # Launched via accelerate/FSDP -> default to the distributed full-FT path
    # unless the config explicitly opts out.
    raw.setdefault("distributed", True)
    cfg = dpo_config_from_dict(raw)
    if not cfg.dataset_path:
        print("error: no dataset_path in config", file=sys.stderr)
        return 2
    result = train(cfg)
    print(f"[dpo] -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
