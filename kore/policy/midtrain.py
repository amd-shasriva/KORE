"""Stage-0 mid-train: continued pretraining on the ROCm/HIP/Triton corpus.

This is the FIRST KORE training stage. It continues pretraining the base model
on the domain corpus assembled by :mod:`kore.data.midtrain_corpus` (plain-text
completion documents), so the policy enters Stage-1 SFT already fluent in the
ROCm/HIP/Triton/Composable-Kernel distribution. The locked recipe is full-FT
(``use_lora=False``) because the domain shift is large; LoRA is available for a
single-GPU smoke run.

Structure mirrors ``sft.py``: heavy imports (torch/transformers/trl) are guarded
inside the training function, the dataset is a plain ``{"text": ...}`` JSONL, and
the trained weights are saved to ``config.output_dir`` (LoRA is merged into the
base first so every downstream stage loads a plain full model).

Full-FT of a 14B needs a sharded multi-GPU launch (FSDP). The campaign shells out
to ``scripts/launch_distributed.sh midtrain <config.json>`` (→ ``accelerate launch
-m kore.policy.midtrain <config.json>``), exactly like sft/dpo; that runs
``_train_single_process`` on the FSDP path. LoRA / single-GPU smoke runs in-process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from kore.obs import get_logger, gpu_mem_snapshot
from kore.policy.configs import (
    MidTrainConfig,
    build_fsdp_kwargs,
    fsdp_enabled,
    preferred_attn_impl,
)

log = get_logger("policy.midtrain")


def load_midtrain_dataset(path: Path):
    """Load a plain-text corpus JSONL into an HF ``Dataset`` of ``{"text": ...}``.

    Accepts rows shaped ``{"text": ...}`` (the mid-train corpus format) or
    ``{"messages": [...]}`` (rendered to a single text field), so the trainer can
    consume either the corpus builder's output or a raw chat shard.
    """
    from datasets import Dataset

    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if isinstance(rec, dict) and isinstance(rec.get("text"), str):
            text = rec["text"]
        elif isinstance(rec, dict) and isinstance(rec.get("messages"), list):
            text = "\n\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in rec["messages"] if str(m.get("content", "")).strip()
            )
        elif isinstance(rec, str):
            text = rec
        else:
            continue
        text = text.strip()
        if text:
            rows.append({"text": text})
    return Dataset.from_list(rows)


def _train_single_process(config: MidTrainConfig, corpus_path: str) -> str:
    """Continued pretraining in the current process.

    Handles three regimes with one code path (mirrors ``sft.train_sft``):
      * **FSDP full-FT** — reached under ``accelerate launch`` (the campaign shells
        out here for ``--full-ft``). ``device_map`` is INCOMPATIBLE with FSDP
        (accelerate owns placement), so the model is loaded plain and the Trainer
        wraps it via ``fsdp``/``fsdp_config``.
      * **LoRA** / **single-GPU full-FT smoke** — keep the legacy
        ``device_map="auto"`` path unchanged.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    use_fsdp = fsdp_enabled(config)
    fsdp_kwargs = build_fsdp_kwargs(config)

    tok = AutoTokenizer.from_pretrained(config.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _attn_impl = preferred_attn_impl()
    model_kwargs = {"torch_dtype": torch.bfloat16,
                    "attn_implementation": _attn_impl}
    if not use_fsdp:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    # Activation checkpointing (routed through fsdp_config) is INCOMPATIBLE with
    # the KV cache: the cache changes the tensor count between forward and
    # recompute -> torch.utils.checkpoint CheckpointError. HF's Trainer only
    # auto-disables use_cache when TrainingArguments.gradient_checkpointing is set,
    # which we turn OFF under FSDP (FSDP owns checkpointing), so disable it here.
    model.config.use_cache = False

    ds = load_midtrain_dataset(Path(corpus_path))
    log.info("midtrain: corpus loaded", corpus=str(corpus_path), model=config.model_id,
             use_lora=bool(config.use_lora), epochs=config.num_train_epochs,
             distributed=bool(getattr(config, "distributed", False)), fsdp=bool(fsdp_kwargs),
             n_docs=len(ds), max_seq_length=config.max_seq_length)

    # Honor the recipe: full-FT (use_lora=False) or a LoRA adapter for smoke.
    peft_cfg = None
    if config.use_lora:
        from peft import LoraConfig
        peft_cfg = LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

    # Activation checkpointing via HF's layer-internal path with REENTRANT
    # checkpointing. Rationale: `attn_implementation="flash_attention_2"` can
    # intermittently downgrade to SDPA per-worker (a ROCm flash-availability race
    # across the 8 FSDP ranks), and SDPA switches fused kernels between the
    # checkpointed forward and its recomputation -> the NON-REENTRANT checkpoint's
    # saved-tensor-count check raises CheckpointError and kills the whole job.
    # Reentrant checkpointing does NOT perform that count check, so it is robust to
    # the backend swap. It is NOT routed through fsdp_config (that external wrapper
    # is the other source of the mismatch on FSDP1/use_orig_params).
    grad_ckpt = bool(config.gradient_checkpointing)

    # Packing safety guard (audit R2 / THEME B): TRL bfd packing needs a flash-attn
    # backend for the block-diagonal mask; on SDPA it silently falls back to a plain
    # causal mask (cross-document attention). Enforce the config invariant at runtime.
    _packing = bool(config.packing)
    if _packing and _attn_impl != "flash_attention_2":
        log.info("midtrain: packing DISABLED -- attn backend is SDPA (not "
                 "flash_attention_2); packing on SDPA cross-contaminates docs", attn=_attn_impl)
        _packing = False

    # Plain-text completion mode: SFTTrainer trains the LM objective over the
    # ``text`` field (no chat template / no completion-only masking).
    args = TRLSFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        max_length=config.max_seq_length,
        bf16=bool(config.bf16),
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        dataset_text_field="text",
        packing=_packing,
        dataloader_num_workers=getattr(config, "dataloader_num_workers", 8),
        dataloader_pin_memory=getattr(config, "dataloader_pin_memory", True),
        dataset_num_proc=getattr(config, "dataset_num_proc", 32),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,  # a 14B full-FT ckpt is ~220GB w/ optimizer; cap to avoid disk-fill
        report_to=[],
        **fsdp_kwargs,
    )

    class _ObsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs:  # train-step logs only
                return
            log.event("midtrain_step", step=int(state.global_step), loss=logs.get("loss"),
                      lr=logs.get("learning_rate"),
                      epoch=round(float(state.epoch), 4) if state.epoch is not None else None,
                      grad_norm=logs.get("grad_norm"), **gpu_mem_snapshot())

    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok,
                         callbacks=[_ObsCallback()])
    from kore.policy.configs import latest_checkpoint
    _resume = latest_checkpoint(config.output_dir)
    if _resume:
        log.info("midtrain: resuming from checkpoint", ckpt=_resume)
    trainer.train(resume_from_checkpoint=_resume)

    # Merge LoRA into the base before saving so downstream stages load a full
    # model; full-FT just saves the trained weights directly.
    if config.use_lora:
        log.info("midtrain: merging LoRA adapter into base", out=config.output_dir)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(config.output_dir)
    else:
        log.info("midtrain: saving full-FT model", out=config.output_dir)
        trainer.save_model(config.output_dir)
    tok.save_pretrained(config.output_dir)
    log.metric("midtrain_done", out=config.output_dir, n_docs=len(ds),
               merged_lora=bool(config.use_lora), **gpu_mem_snapshot())
    return config.output_dir


def train_midtrain(config: MidTrainConfig, corpus_path: Optional[str] = None) -> str:
    """Continued-pretrain the base on the ROCm/HIP/Triton corpus.

    Full-parameter FSDP sharding for a 14B is driven by the campaign, which shells
    out to ``scripts/launch_distributed.sh midtrain <config.json>`` (→
    ``accelerate launch -m kore.policy.midtrain <config.json>``). Under that
    launcher this function runs :func:`_train_single_process`, which takes the
    FSDP path (``device_map`` disabled, ``fsdp``/``fsdp_config`` set) whenever
    :func:`fsdp_enabled` holds. LoRA / single-GPU smoke runs in-process the same
    way. Returns the output checkpoint dir, which the campaign threads in as the
    SFT base.
    """
    corpus = str(corpus_path or config.corpus_path)
    if not Path(corpus).exists():
        raise FileNotFoundError(f"midtrain corpus not found: {corpus} "
                                "(build it with build_midtrain_corpus first)")
    return _train_single_process(config, corpus)


# --------------------------------------------------------------------------- #
# Distributed entry: `python -m kore.policy.midtrain <config.json>`
#
# Used by scripts/launch_distributed.sh under `accelerate launch` (the campaign
# shells out here for --full-ft, exactly like sft/dpo/grpo). Pure-stdlib JSON
# parsing (no torch at import time); the heavy trainer is only touched when
# _train_single_process() runs. The JSON is a flat map of MidTrainConfig fields
# (incl. the DistributedMixin FSDP knobs).
# --------------------------------------------------------------------------- #
def midtrain_config_from_dict(d: dict) -> MidTrainConfig:
    """Build a :class:`MidTrainConfig` from a plain JSON dict.

    Presence of this builder is what the campaign's ``_stage_supports_launcher``
    detects to route ``--full-ft`` midtrain through the FSDP launcher.
    """
    return MidTrainConfig(**dict(d))


def _main(argv: Optional[list[str]] = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m kore.policy.midtrain <config.json>", file=sys.stderr)
        return 2
    raw = json.loads(Path(argv[0]).read_text())
    # Launched via accelerate/FSDP -> default to distributed full-FT unless the
    # config explicitly opts out.
    raw.setdefault("distributed", True)
    cfg = midtrain_config_from_dict(raw)
    out = _train_single_process(cfg, cfg.corpus_path)
    print(f"[midtrain] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
