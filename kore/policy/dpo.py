"""Stage-2 DPO on ranked preference pairs.

Trains a LoRA policy with ``trl``'s ``DPOTrainer`` against a frozen reference
(the SFT checkpoint). Plan hyperparameters (see ``configs.DPOConfig``):
beta = 0.1, ref = SFT, bf16, max_len 16384.

Dataset is a preference JSONL of ``{"prompt", "chosen", "rejected"}`` rows
(``prompt`` may be a string or a chat-message list). Heavy imports are guarded.
"""

from __future__ import annotations

import json
from typing import Any

from kore.policy.configs import DPOConfig


def load_preference_jsonl(path: str) -> list[dict]:
    """Read preference rows: ``{"prompt", "chosen", "rejected"}`` per line.

    ``prompt`` may be a plain string or a list of chat messages; ``chosen`` /
    ``rejected`` are the two candidate completions. Lines missing any field are
    skipped.
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

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(rows)

    dtype = torch.bfloat16 if config.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=dtype)

    # With LoRA, TRL uses the frozen base weights as the implicit reference, so
    # an explicit ref model is optional. Load one only if a distinct id is set.
    ref_model = None
    if not config.use_lora and config.ref_model_id and config.ref_model_id != config.model_id:
        ref_model = AutoModelForCausalLM.from_pretrained(config.ref_model_id, torch_dtype=dtype)

    peft_config = _peft_config(config) if config.use_lora else None

    trl_args = TRLDPOConfig(
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
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        seed=config.seed,
        report_to=config.report_to,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=trl_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
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
