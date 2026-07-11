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
        weight_decay=getattr(config, "weight_decay", 0.0),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        bf16=config.bf16,
        gradient_checkpointing=bool(config.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": True},
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=1,   # a 14B full-FT ckpt is ~220GB w/ optimizer; cap to avoid disk-fill
        seed=config.seed,
        report_to=config.report_to,
    )
    # loss_type may be a STRING ("sigmoid"/"ipo") OR a LIST of components for a
    # composite loss. ["sigmoid", "sft"] + loss_weights=[1,1] == RPO (DPO + an
    # NLL-on-chosen anchor). The SFT/NLL anchor is THE fix for the likelihood
    # displacement that degenerated DPO v1: widening the reward margin by tanking
    # BOTH chosen and rejected log-probs (entropy collapse -> garbage tokens /
    # incomplete kernels) is countered by a positive "keep the chosen likely"
    # gradient. TRL 0.29.1 has no rpo_alpha; the list form + loss_weights is the
    # supported equivalent (verified: loss_type is list[str], "sft" is a component).
    loss_type = getattr(config, "loss_type", None)
    if loss_type:
        kwargs["loss_type"] = (list(loss_type)
                               if isinstance(loss_type, (list, tuple)) else str(loss_type))
    loss_weights = getattr(config, "loss_weights", None)
    if loss_weights:
        kwargs["loss_weights"] = [float(w) for w in loss_weights]
    label_smoothing = getattr(config, "label_smoothing", None)
    if label_smoothing:
        kwargs["label_smoothing"] = float(label_smoothing)
    # LD-DPO length desensitization (down-weight the verbose tail) — optional guard
    # against length-driven degeneration on long code.
    ld_alpha = getattr(config, "ld_alpha", None)
    if ld_alpha is not None:
        kwargs["ld_alpha"] = float(ld_alpha)
    # Truncation guard: with max_prompt_length gone in trl>=0.29, if a pair ever
    # exceeds max_length the TRL default "keep_start" slices input_ids[:, :max_length]
    # — cutting the COMPLETION's tail (trains the model to stop mid-kernel; verified
    # against trl 0.29.1 DPOTrainer._truncate_inputs). We DEFAULT to "keep_end", which
    # keeps input_ids[:, -max_length:] (the kernel body + <|im_end|> stop) and drops
    # the prompt head instead. A config may still override it explicitly.
    truncation_mode = getattr(config, "truncation_mode", None) or "keep_end"
    kwargs["truncation_mode"] = str(truncation_mode)
    # Gradient clipping (DPO v1 saw pre-clip grad-norm ~192). Pass explicitly.
    max_grad_norm = getattr(config, "max_grad_norm", None)
    if max_grad_norm is not None:
        kwargs["max_grad_norm"] = float(max_grad_norm)
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

    # Drop any kwargs the INSTALLED trl DPOConfig doesn't accept (TRL renames/removes
    # fields across versions, e.g. `max_prompt_length` is gone in trl>=0.29 — the
    # total `max_length` still caps prompt+completion). Keeps us robust to trl drift
    # instead of hard-failing the whole DPO stage on one unknown kwarg.
    import inspect
    _dpo_kwargs = build_trl_dpo_kwargs(config)
    _valid = set(inspect.signature(TRLDPOConfig.__init__).parameters)
    _dropped = sorted(set(_dpo_kwargs) - _valid)
    if _dropped:
        log.event("dpo_config_kwargs_dropped", dropped=_dropped,
                  trl_version=getattr(__import__("trl"), "__version__", "?"))
        _dpo_kwargs = {k: v for k, v in _dpo_kwargs.items() if k in _valid}
    trl_args = TRLDPOConfig(**_dpo_kwargs)

    # Lightweight per-log-step observability callback (guarded transformers import).
    from transformers import TrainerCallback

    class _ObsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs:  # skip eval/summary logs — train-step only
                return
            # logps/chosen + rewards/chosen are the LIKELIHOOD-DISPLACEMENT alarms:
            # if logps_chosen trends DOWN or rewards_chosen goes/stays NEGATIVE while
            # rewards_margin grows, the policy is degenerating (v1's failure) — stop
            # and raise beta / lower LR / increase the sft weight.
            log.event("dpo_step", step=int(state.global_step), loss=logs.get("loss"),
                      lr=logs.get("learning_rate"),
                      epoch=round(float(state.epoch), 4) if state.epoch is not None else None,
                      grad_norm=logs.get("grad_norm"),
                      rewards_margin=logs.get("rewards/margins"),
                      rewards_acc=logs.get("rewards/accuracies"),
                      logps_chosen=logs.get("logps/chosen"),
                      logps_rejected=logs.get("logps/rejected"),
                      rewards_chosen=logs.get("rewards/chosen"),
                      rewards_rejected=logs.get("rewards/rejected"),
                      entropy=logs.get("entropy"), **gpu_mem_snapshot())

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
    # Fields that are NOT on the base DPOConfig dataclass (TRL loss knobs / guards):
    # pop them so DPOConfig(**d) doesn't choke, then attach as attributes that
    # build_trl_dpo_kwargs reads via getattr. Covers the iterative-DPO + KORE-DPO-v2
    # anti-degeneration recipe (RPO composite loss, LD-DPO, cDPO, clip, truncation).
    _extras = {}
    for k in ("loss_type", "loss_weights", "label_smoothing", "ld_alpha",
              "truncation_mode", "max_grad_norm"):
        if k in d:
            _extras[k] = d.pop(k)
    cfg = DPOConfig(**d)
    if lora is not None:
        cfg.lora = LoRAConfig(**lora)
    for k, v in _extras.items():
        if v is not None:
            setattr(cfg, k, v)
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
