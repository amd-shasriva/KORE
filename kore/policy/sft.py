"""Repair-weighted supervised fine-tuning (LoRA) for the KORE cold start.

Trains on chat-formatted transcripts (repair transitions, verified wins, and
reasoning traces). Uses trl's SFTTrainer. Note: trl renamed ``max_seq_length`` ->
``max_length``; ``assistant_only_loss`` needs a ``{% generation %}`` marker in the
chat template, which Qwen3's template lacks, so we do not set it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from kore.policy.configs import SFTConfig


def load_sft_dataset(path: Path):
    from datasets import Dataset

    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rows.append({"messages": rec["messages"] if "messages" in rec else rec})
    return Dataset.from_list(rows)


def train_sft(config: SFTConfig, dataset_path: Path, tasks: Optional[list[str]] = None):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    tok = AutoTokenizer.from_pretrained(config.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_sft_dataset(dataset_path)
    peft_cfg = LoraConfig(
        r=config.lora.r, lora_alpha=config.lora.lora_alpha, lora_dropout=config.lora.lora_dropout,
        target_modules=list(config.lora.target_modules), task_type="CAUSAL_LM")

    args = TRLSFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        max_length=config.max_seq_length,
        bf16=True,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(config.output_dir)
    tok.save_pretrained(config.output_dir)
    return config.output_dir
