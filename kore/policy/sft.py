"""Repair-weighted supervised fine-tuning for the KORE cold start.

Trains on chat-formatted transcripts (repair transitions, verified wins, and
reasoning traces). Uses trl's SFTTrainer. Note: trl renamed ``max_seq_length`` ->
``max_length``.

Completion-only loss (``config.assistant_only_loss``, default on): the base Qwen3
chat template has no ``{% generation %}`` marker, so :func:`build_assistant_masked_template`
injects one around the assistant body (content + tools + ``<|im_end|>``) while
keeping the rendered text byte-identical to the base template. TRL's
``assistant_only_loss`` then masks every prompt/user/system/tool token to ``-100``
and trains only on assistant responses (+ their stop token) — the standard SFT
recipe, and the fix for training capacity being spent predicting the user's prompt.
:func:`_verify_assistant_masking` asserts render-identity + non-empty masks before
training (and TRL itself raises if any example has no assistant tokens).

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


def build_assistant_masked_template(template: str) -> str:
    """Inject ``{% generation %}`` markers into a Qwen3 chat template for masking.

    TRL's ``assistant_only_loss`` needs the template to wrap the assistant's
    GENERATED tokens in ``{% generation %} ... {% endgeneration %}`` so
    ``apply_chat_template(..., return_assistant_tokens_mask=True)`` returns the
    per-token ``assistant_masks`` the collator uses to set non-assistant labels to
    ``-100``. The stock Qwen3 template lacks the marker.

    The edit is surgical and RENDER-PRESERVING: the ``<|im_start|>assistant\\n``
    header is pulled OUT of the three assistant body-emission sites (so it stays
    OUTSIDE the generation span — a masked prompt) and emitted once just before a
    single ``{% generation %}`` that spans the body (content [+ optional <think>],
    tool_calls, and the closing ``<|im_end|>`` stop token), closed by
    ``{% endgeneration %}``. Splitting the header off the body changes no emitted
    text, so the rendered string is byte-identical to the base template (asserted by
    :func:`_verify_assistant_masking`); the only effect is the token mask. Idempotent
    (returns unchanged if a generation marker is already present).

    Raises ``ValueError`` if the expected Qwen3 assistant-branch anchors are absent
    (e.g. a non-Qwen3 template) so we fail loudly rather than train unmasked.
    """
    if "{% generation %}" in template or "{%- generation %}" in template:
        return template  # already generation-tagged (newer template) — leave as-is
    t = template
    # 1) strip the header from the assistant BODY emissions (reasoning + 2 plain).
    before = t
    t = t.replace(
        "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}",
        "{{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}",
    )
    t = t.replace(
        "{{- '<|im_start|>' + message.role + '\\n' + content }}",
        "{{- content }}",
    )  # both remaining occurrences are assistant-only (user/system append <|im_end|>)
    # 2) emit the header once + OPEN generation just before the index test.
    t = t.replace(
        "        {%- if loop.index0 > ns.last_query_index %}",
        "        {{- '<|im_start|>' + message.role + '\\n' }}{% generation %}\n        {%- if loop.index0 > ns.last_query_index %}",
        1,
    )
    # 3) CLOSE generation right after the assistant <|im_end|> (anchored by the tool
    #    branch so the tool/user <|im_end|> sites are never matched).
    t = t.replace(
        "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}",
        "        {{- '<|im_end|>\\n' }}{% endgeneration %}\n    {%- elif message.role == \"tool\" %}",
        1,
    )
    if t == before or "{% generation %}" not in t or "{% endgeneration %}" not in t:
        raise ValueError(
            "build_assistant_masked_template: could not inject generation markers — "
            "the chat template does not match the expected Qwen3 assistant branch. "
            "Set assistant_only_loss=False or supply a generation-tagged template."
        )
    return t


def _verify_assistant_masking(tok, base_template: str, masked_template: str) -> None:
    """Fail-fast guard for the masked template (runs before any training).

    Asserts two invariants on representative single-turn, multi-turn, ``<think>``,
    and tool conversations:
      1. **Render-identity** — the masked template renders byte-identical text to the
         base template (the mask must not perturb what the model sees).
      2. **Correct masking** — assistant response tokens (and their ``<|im_end|>``)
         are unmasked while every system/user/tool/header token is masked, and the
         mask is non-empty.
    Raises ``AssertionError`` on any violation so a broken template aborts the run
    immediately instead of silently training on the wrong tokens.
    """
    cases = [
        [{"role": "user", "content": "Write a HIP kernel."},
         {"role": "assistant", "content": "KERNEL_BODY_A"}],
        [{"role": "system", "content": "SYS_TXT"},
         {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "RESP_ONE"},
         {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "RESP_TWO"}],
        [{"role": "user", "content": "think please"},
         {"role": "assistant", "content": "<think>\nreasoning_here\n</think>\nFINAL_ANS"}],
        [{"role": "user", "content": "call tool"},
         {"role": "assistant", "content": "TOOL_PREAMBLE"},
         {"role": "tool", "content": "TOOL_OUT"},
         {"role": "assistant", "content": "TOOL_FINAL"}],
    ]
    orig = tok.chat_template
    try:
        for msgs in cases:
            tok.chat_template = base_template
            a = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            tok.chat_template = masked_template
            b = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            assert a == b, f"masked template changed rendered text:\n  base={a!r}\n  mask={b!r}"
            out = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                return_assistant_tokens_mask=True, return_dict=True)
            ids, mask = out["input_ids"], out["assistant_masks"]
            assert 1 in mask, f"no assistant tokens unmasked for {msgs!r}"
            learned = tok.decode([i for i, m in zip(ids, mask) if m == 1])
            dropped = tok.decode([i for i, m in zip(ids, mask) if m == 0])
            for turn in msgs:
                if turn["role"] == "assistant":
                    tail = turn["content"].split("</think>")[-1].strip()[:12]
                    assert tail in learned, f"assistant text {tail!r} was masked out"
                else:
                    assert turn["content"][:10] not in learned, \
                        f"non-assistant text {turn['content'][:10]!r} was learned"
            assert "<|im_end|>" in learned, "stop token <|im_end|> not in the loss"
            assert "<|im_start|>assistant" in dropped, "assistant header should be masked"
    finally:
        tok.chat_template = orig


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


def _filter_overlong(ds, tok, max_length: int):
    """Drop rows whose chat-rendered token length exceeds ``max_length``.

    Returns ``(filtered_dataset, n_dropped)``. Deterministic so every rank computes
    the identical filtered set (consistent FSDP data shards). A row that fails to
    render is KEPT (conservative — let the trainer handle it). Returns the original
    dataset object unchanged when nothing is dropped.
    """
    from datasets import Dataset

    if not max_length or max_length <= 0:
        return ds, 0
    keep, dropped = [], 0
    for row in ds:
        try:
            n = len(tok.apply_chat_template(row["messages"], tokenize=True,
                                            add_generation_prompt=False))
            if n > max_length:
                dropped += 1
                continue
        except Exception:  # noqa: BLE001 - length check is advisory; keep on error
            pass
        keep.append({"messages": row["messages"]})
    if dropped == 0:
        return ds, 0
    return Dataset.from_list(keep), dropped


def train_sft(config: SFTConfig, dataset_path: Path) -> str:
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

    # Completion-only loss: inject {% generation %} markers so TRL masks every
    # prompt/user/system/tool token to -100 and trains only on assistant responses
    # (+ their <|im_end|> stop). The masked template is verified render-identical to
    # the base before use. On the real FSDP full-FT path a non-maskable template is
    # a hard error (we must not silently train unmasked); on a smoke/LoRA run with a
    # non-Qwen3 template we log and fall back to full-sequence loss.
    assistant_only = bool(getattr(config, "assistant_only_loss", False))
    base_chat_template = tok.chat_template  # restored before save (checkpoint keeps pristine template)
    if assistant_only:
        try:
            if not base_chat_template:
                raise ValueError("tokenizer has no chat_template")
            masked_tpl = build_assistant_masked_template(base_chat_template)
            _verify_assistant_masking(tok, base_chat_template, masked_tpl)
            tok.chat_template = masked_tpl
            log.info("sft: completion-only loss enabled", assistant_only_loss=True)
        except (ValueError, AssertionError) as e:
            if use_fsdp:
                raise
            assistant_only = False
            log.info("sft: assistant_only_loss disabled (template not maskable)",
                     reason=repr(e))

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
    # Drop rows whose rendered length exceeds max_seq_length — ONLY under completion-
    # only loss. There, an over-length row whose assistant span is truncated away would
    # make TRL raise ("no assistant tokens"), and even a partially-cut assistant tail
    # (missing <|im_end|>) teaches run-on. With full-sequence loss we keep TRL's stock
    # truncation (the prior status quo). Deterministic -> identical filtered set on every
    # rank (consistent FSDP shards). NB: on the current multicap mix this drops ~8.7% of
    # rows, almost entirely pathologically-long (>16k tok) math_reasoning CoTs (a data-
    # quality item for the data pass) and only ~0.6% of kernels.
    n_over = 0
    if assistant_only:
        ds, n_over = _filter_overlong(ds, tok, config.max_seq_length)
    log.info("sft: dataset loaded", dataset=str(dataset_path), model=config.model_id,
             use_lora=bool(config.use_lora), epochs=config.num_train_epochs,
             distributed=bool(config.distributed), fsdp=bool(fsdp_kwargs),
             assistant_only_loss=bool(assistant_only), dropped_overlong=n_over,
             repair_weight=config.repair_loss_weight, **_token_stats(ds, tok))

    # Honor the recipe: full-FT (use_lora=False) or LoRA adapter.
    peft_cfg = None
    if config.use_lora:
        peft_cfg = LoraConfig(
            r=config.lora.r, lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=list(config.lora.target_modules), task_type="CAUSAL_LM")

    # Activation checkpointing via HF's layer-internal path with REENTRANT
    # checkpointing (robust to the intermittent flash_attention_2 -> SDPA per-worker
    # downgrade: reentrant skips the saved-tensor-count check that NON-REENTRANT
    # does, which otherwise raises CheckpointError when SDPA swaps fused kernels
    # between forward and recompute). NOT routed through fsdp_config (the external
    # wrapper is the other source of the mismatch on FSDP1/use_orig_params).
    grad_ckpt = bool(config.gradient_checkpointing)

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
        packing=bool(config.packing),
        assistant_only_loss=bool(assistant_only),
        bf16=config.bf16,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=1,   # a 14B full-FT ckpt is ~220GB w/ optimizer; cap to avoid disk-fill
        report_to=config.report_to,
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
    from kore.policy.configs import latest_checkpoint
    _resume = latest_checkpoint(config.output_dir)
    if _resume:
        log.info("sft: resuming from checkpoint", ckpt=_resume)
    trainer.train(resume_from_checkpoint=_resume)

    # Merge LoRA into the base before saving so downstream stages load a full
    # model; full-FT just saves the trained weights directly.
    if config.use_lora:
        log.info("sft: merging LoRA adapter into base", out=config.output_dir)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(config.output_dir)
    else:
        log.info("sft: saving full-FT model", out=config.output_dir)
        trainer.save_model(config.output_dir)
    # Restore the pristine (un-tagged) chat template before saving so the checkpoint's
    # tokenizer is byte-identical to the base. The {% generation %} markers are only
    # needed for THIS run's mask generation (they render identically at inference).
    if base_chat_template is not None:
        tok.chat_template = base_chat_template
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
