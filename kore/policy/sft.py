"""Repair-weighted supervised fine-tuning for the KORE cold start.

Trains on chat-formatted transcripts (repair transitions, verified wins, and
reasoning traces). Uses trl's SFTTrainer. Note: trl renamed ``max_seq_length`` ->
``max_length``; ``assistant_only_loss`` needs a ``{% generation %}`` marker in the
chat template, which Qwen3's template lacks, so we do not set it.

Full-FT vs LoRA is governed by ``config.use_lora`` (the locked KORE recipe is
full-FT). When LoRA is used the adapter is merged into the base before saving so
every downstream stage (DPO, GRPO, soup) can load a plain full model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from kore.obs import get_logger, gpu_mem_snapshot
from kore.policy.configs import (
    LoRAConfig,
    SFTConfig,
    build_fsdp_kwargs,
    fsdp_enabled,
    preferred_attn_impl,
)

log = get_logger("policy.sft")

# ``_source`` markers (see kore/data/build_datasets.py) that denote a repair
# (broken -> fixed) transition, which the plan up-weights during SFT.
REPAIR_SOURCES = ("kernel_repair_opt", "repair")


def _token_stats(ds, tok, sample: int = 512) -> dict:
    """Best-effort chat-token length stats over up to ``sample`` rows (logging).

    Read-only: renders each sampled row through the chat template to count
    tokens. Never raises — returns ``{}`` if the tokenizer/template can't render.
    """
    try:
        n = len(ds)
        idxs = range(min(n, sample))
        lengths = []
        for i in idxs:
            msgs = ds[i]["messages"]
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            lengths.append(len(ids))
        if not lengths:
            return {"n_rows": n}
        lengths.sort()
        p95 = lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))]
        return {"n_rows": n, "tok_sampled": len(lengths),
                "tok_min": lengths[0], "tok_max": lengths[-1],
                "tok_mean": round(sum(lengths) / len(lengths), 1), "tok_p95": p95}
    except Exception as e:  # noqa: BLE001 - stats are advisory, never fatal
        return {"n_rows": len(ds), "tok_stats_error": repr(e)}


def load_sft_dataset(path: Path, repair_weight: float = 1.0,
                     repair_sources: tuple[str, ...] = REPAIR_SOURCES):
    """Load a chat JSONL into an HF ``Dataset`` of ``{"messages": [...]}`` rows.

    ``repair_weight`` implements the plan's repair up-weighting. trl's
    ``SFTTrainer`` computes a token-mean loss and does not expose a per-example
    scalar loss weight without subclassing ``compute_loss`` (which would also
    fight its packing/collator path), so we approximate per-example weighting by
    integer up-sampling: a row whose ``_source`` is a repair marker is emitted
    ``round(repair_weight)`` times. This raises the effective gradient mass on
    repair transitions proportional to ``repair_weight`` while keeping trl's
    stock training path intact.
    """
    from datasets import Dataset

    factor = max(1, int(round(repair_weight)))
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        messages = rec["messages"] if isinstance(rec, dict) and "messages" in rec else rec
        row = {"messages": messages}
        rows.append(row)
        src = rec.get("_source") if isinstance(rec, dict) else None
        if factor > 1 and src in repair_sources:
            rows.extend([dict(row) for _ in range(factor - 1)])
    return Dataset.from_list(rows)


def train_sft(config: SFTConfig, dataset_path: Path, tasks: Optional[list[str]] = None) -> str:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    # Distributed full-FT (FSDP) vs the legacy single-process / LoRA path.
    # Under FSDP, device_map is INCOMPATIBLE (accelerate/FSDP owns placement), so
    # we must load the model plain and let the Trainer wrap it. Only full-FT
    # (use_lora=False) launched distributed takes this path; everything else
    # (LoRA, single-GPU, CPU tests) keeps device_map="auto" exactly as before.
    use_fsdp = fsdp_enabled(config)
    fsdp_kwargs = build_fsdp_kwargs(config)

    tok = AutoTokenizer.from_pretrained(config.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model_kwargs = {"torch_dtype": torch.bfloat16,
                    "attn_implementation": preferred_attn_impl()}
    if not use_fsdp:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    # Activation checkpointing (routed through fsdp_config) is INCOMPATIBLE with
    # the KV cache: the cache changes the tensor count between forward and
    # recompute -> torch.utils.checkpoint CheckpointError. HF's Trainer only
    # auto-disables use_cache when TrainingArguments.gradient_checkpointing is set,
    # which we turn OFF under FSDP (FSDP owns checkpointing), so disable it here.
    model.config.use_cache = False

    ds = load_sft_dataset(dataset_path, repair_weight=config.repair_loss_weight)
    log.info("sft: dataset loaded", dataset=str(dataset_path), model=config.model_id,
             use_lora=bool(config.use_lora), epochs=config.num_train_epochs,
             distributed=bool(config.distributed), fsdp=bool(fsdp_kwargs),
             repair_weight=config.repair_loss_weight, **_token_stats(ds, tok))

    # Honor the recipe: full-FT (use_lora=False) or LoRA adapter.
    peft_cfg = None
    if config.use_lora:
        peft_cfg = LoraConfig(
            r=config.lora.r, lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=list(config.lora.target_modules), task_type="CAUSAL_LM")

    # Activation checkpointing via HF's layer-internal, NON-REENTRANT path (the
    # FSDP-safe one). It is NOT routed through fsdp_config (that external wrapper
    # mismatches saved-tensor counts on an FSDP1/use_orig_params unit).
    grad_ckpt = bool(config.gradient_checkpointing)

    args = TRLSFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        max_length=config.max_seq_length,
        bf16=config.bf16,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to=[],
        **fsdp_kwargs,
    )
    # Lightweight per-log-step observability callback (guarded transformers import).
    from transformers import TrainerCallback

    class _ObsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs:  # skip eval/summary logs — train-step only
                return
            log.event("sft_step", step=int(state.global_step), loss=logs.get("loss"),
                      lr=logs.get("learning_rate"),
                      epoch=round(float(state.epoch), 4) if state.epoch is not None else None,
                      grad_norm=logs.get("grad_norm"), **gpu_mem_snapshot())

    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok,
                         callbacks=[_ObsCallback()])
    trainer.train()

    # Merge LoRA into the base before saving so downstream stages load a full
    # model; full-FT just saves the trained weights directly.
    if config.use_lora:
        log.info("sft: merging LoRA adapter into base", out=config.output_dir)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(config.output_dir)
    else:
        log.info("sft: saving full-FT model", out=config.output_dir)
        trainer.save_model(config.output_dir)
    tok.save_pretrained(config.output_dir)
    log.metric("sft_done", out=config.output_dir, merged_lora=bool(config.use_lora),
               **gpu_mem_snapshot())
    return config.output_dir


# --------------------------------------------------------------------------- #
# Distributed entry: `python -m kore.policy.sft <config.json>`
#
# Used by scripts/launch_distributed.sh under `accelerate launch`. Pure-stdlib
# JSON parsing (no torch at import time); the heavy trainer is only touched when
# train_sft() actually runs. The JSON is a flat map of SFTConfig fields, with an
# optional nested "lora" object and an optional "dataset_path".
# --------------------------------------------------------------------------- #
def sft_config_from_dict(d: dict) -> tuple[SFTConfig, str]:
    """Build an ``(SFTConfig, dataset_path)`` pair from a plain dict.

    ``dataset_path`` falls back to ``config.dataset_path`` when not given at the
    top level. A nested ``lora`` mapping is turned into a :class:`LoRAConfig`.
    """
    d = dict(d)
    lora = d.pop("lora", None)
    # dataset_path is a real SFTConfig field, so keep it on the config too.
    cfg = SFTConfig(**d)
    if lora is not None:
        cfg.lora = LoRAConfig(**lora)
    return cfg, cfg.dataset_path


def _main(argv: Optional[list[str]] = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m kore.policy.sft <config.json>", file=sys.stderr)
        return 2
    raw = json.loads(Path(argv[0]).read_text())
    # Launched via accelerate/FSDP -> default to the distributed full-FT path
    # unless the config explicitly opts out.
    raw.setdefault("distributed", True)
    cfg, dataset_path = sft_config_from_dict(raw)
    if not dataset_path:
        print("error: no dataset_path in config", file=sys.stderr)
        return 2
    out = train_sft(cfg, Path(dataset_path))
    print(f"[sft] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
