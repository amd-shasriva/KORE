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

Full-FT of a 14B needs a sharded multi-GPU launch (FSDP/DeepSpeed). When
``config.distributed`` is set and full-FT is requested, ``train_midtrain``
defers to the distributed launcher path (:func:`build_launch_command` /
``python -m kore.policy.midtrain``); otherwise it runs single-process (LoRA /
smoke). See docs/DISTRIBUTED.md for the multi-GPU launch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from kore.obs import get_logger, gpu_mem_snapshot
from kore.policy.configs import MidTrainConfig, build_fsdp_kwargs, fsdp_enabled

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


def build_launch_command(config: MidTrainConfig, corpus_path: str,
                         nproc_per_node: Optional[int] = None) -> list[str]:
    """Construct the sharded multi-GPU launch command for full-FT mid-train.

    Returns an ``accelerate launch`` argv that runs this module's ``__main__``
    entry (``python -m kore.policy.midtrain``) under a distributed launcher. The
    operator (or docs/DISTRIBUTED.md tooling) executes it; ``train_midtrain`` only
    builds + logs it so no heavy training is kicked off implicitly.
    """
    if nproc_per_node is None:
        try:
            import torch
            nproc_per_node = max(1, torch.cuda.device_count())
        except Exception:  # noqa: BLE001 - no torch/CUDA -> assume single process
            nproc_per_node = 1
    return [
        "accelerate", "launch",
        "--num_processes", str(nproc_per_node),
        "-m", "kore.policy.midtrain",
        "--model-id", config.model_id,
        "--corpus-path", str(corpus_path),
        "--output-dir", config.output_dir,
    ]


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
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if not use_fsdp:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)

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

    # Under FSDP full_shard, activation checkpointing is routed through
    # fsdp_config (see build_fsdp_kwargs); enabling TrainingArguments'
    # gradient_checkpointing on top adds a redundant AllGather (HF warns), so we
    # let FSDP own it there and only set it on the non-FSDP path.
    grad_ckpt = config.gradient_checkpointing and not use_fsdp

    # Plain-text completion mode: SFTTrainer trains the LM objective over the
    # ``text`` field (no chat template / no completion-only masking).
    args = TRLSFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        max_length=config.max_seq_length,
        bf16=bool(config.bf16),
        gradient_checkpointing=grad_ckpt,
        dataset_text_field="text",
        packing=True,
        logging_steps=10,
        save_steps=200,
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
    trainer.train()

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
# (incl. the DistributedMixin FSDP knobs). A legacy `--model-id ...` flag form is
# still accepted for the hand-rolled launch documented in docs/DISTRIBUTED.md.
# --------------------------------------------------------------------------- #
def midtrain_config_from_dict(d: dict) -> MidTrainConfig:
    """Build a :class:`MidTrainConfig` from a plain JSON dict.

    Presence of this builder is what the campaign's ``_stage_supports_launcher``
    detects to route ``--full-ft`` midtrain through the FSDP launcher.
    """
    return MidTrainConfig(**dict(d))


def _build_argparser():
    import argparse

    p = argparse.ArgumentParser(description="KORE Stage-0 continued pretraining")
    p.add_argument("--model-id", required=True, dest="model_id")
    p.add_argument("--corpus-path", required=True, dest="corpus_path")
    p.add_argument("--output-dir", default="runs/midtrain", dest="output_dir")
    p.add_argument("--learning-rate", type=float, default=1e-5, dest="learning_rate")
    p.add_argument("--num-train-epochs", type=float, default=1.0, dest="num_train_epochs")
    p.add_argument("--max-seq-length", type=int, default=8192, dest="max_seq_length")
    p.add_argument("--lora", action="store_true", dest="use_lora")
    return p


def main(argv: Optional[list[str]] = None) -> str:
    """Legacy flag entry: ``python -m kore.policy.midtrain --model-id ...``.

    Full-FT under this path defaults ``distributed=True`` so it takes the FSDP
    branch of :func:`_train_single_process` when run under ``accelerate launch``.
    """
    args = _build_argparser().parse_args(argv)
    cfg = MidTrainConfig(
        model_id=args.model_id, corpus_path=args.corpus_path,
        output_dir=args.output_dir, learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs, max_seq_length=args.max_seq_length,
        use_lora=args.use_lora, distributed=not args.use_lora)
    return _train_single_process(cfg, cfg.corpus_path)


def _main(argv: Optional[list[str]] = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    # JSON-config form (what scripts/launch_distributed.sh / the campaign invoke):
    # a single positional pointing at a `.json` file.
    if len(argv) == 1 and argv[0].endswith(".json"):
        raw = json.loads(Path(argv[0]).read_text())
        # Launched via accelerate/FSDP -> default to distributed full-FT unless
        # the config explicitly opts out.
        raw.setdefault("distributed", True)
        cfg = midtrain_config_from_dict(raw)
        out = _train_single_process(cfg, cfg.corpus_path)
        print(f"[midtrain] -> {out}")
        return 0
    # Legacy flag form.
    print(main(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
