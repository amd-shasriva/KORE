"""Repair-weighted supervised fine-tuning for the KORE cold start.

Trains on chat-formatted transcripts (repair transitions, verified wins, and
reasoning traces). Uses trl's SFTTrainer. Note: trl renamed ``max_seq_length`` ->
``max_length``.

Completion-only loss (``config.assistant_only_loss``, default on): the base Qwen3
chat template has no ``{% generation %}`` marker, so :func:`build_assistant_masked_template`
injects one around the assistant body (content + tools + ``<|im_end|>``) while
keeping the rendered text byte-identical to the base template. TRL's
``assistant_only_loss`` then masks every prompt/user/system/tool token to ``-100``
and trains only on assistant responses (+ their stop token) - the standard SFT
recipe, and the fix for training capacity being spent predicting the user's prompt.
:func:`_verify_assistant_masking` asserts render-identity + non-empty masks before
training (and TRL itself raises if any example has no assistant tokens).

Full-FT vs LoRA is governed by ``config.use_lora`` (the locked KORE recipe is
full-FT). When LoRA is used the adapter is merged into the base before saving so
every downstream stage (DPO, GRPO, soup) can load a plain full model.
"""

from __future__ import annotations

import itertools
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
from kore.policy.model_spec import (
    IDENTITY_CONFIG_KEYS,
    apply_runtime_settings,
    log_model_identity,
    model_identity_for_config,
    split_runtime_settings,
)
from kore.policy.resources import (
    PREFLIGHT_CONFIG_KEYS,
    log_stage_preflight,
    run_stage_preflight,
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
    OUTSIDE the generation span - a masked prompt) and emitted once just before a
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
        return template  # already generation-tagged (newer template) - leave as-is
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
        t = _inject_qwen3_coder_markers(template)
    if t is None or "{% generation %}" not in t or "{% endgeneration %}" not in t:
        raise ValueError(
            "build_assistant_masked_template: could not inject generation markers - "
            "the chat template does not match the expected Qwen3 or Qwen3-Coder "
            "assistant branch. Set assistant_only_loss=False or supply a "
            "generation-tagged template."
        )
    return t


def _inject_qwen3_coder_markers(template: str):
    """Generation markers for the Qwen3-Coder template, which is a different shape.

    Qwen3-Coder emits assistant turns from TWO sites, and the second is shared:

        {%- if message.role == "assistant" and message.tool_calls ... %}   <- tool calls
        {%- elif message.role == "user" or message.role == "system"
                 or message.role == "assistant" %}                        <- everything else

    Wrapping that shared branch would mark user and system tokens as generated
    and train the model on its own prompts, so the branch is split first: a
    dedicated assistant arm is inserted ahead of it, emitting byte-identical
    text, and only that arm is wrapped. As in the Qwen3 path the
    ``<|im_start|>assistant\\n`` header stays OUTSIDE the span -- it is a prompt
    the model is given, not something it produced.

    Render-preserving by construction, and ``_verify_assistant_masking`` proves
    it: every branch emits exactly the characters it emitted before.
    Returns ``None`` if the anchors are absent, so the caller can fail loudly.
    """
    tool_open = (
        '        {{- \'<|im_start|>\' + message.role }}\n'
        '        {%- if message.content is defined'
    )
    shared_branch = (
        '        {{- \'<|im_end|>\\n\' }}\n'
        '    {%- elif message.role == "user" or message.role == "system"'
        ' or message.role == "assistant" %}\n'
        '        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content'
        ' + \'<|im_end|>\' + \'\\n\' }}'
    )
    if tool_open not in template or shared_branch not in template:
        return None

    t = template.replace(
        tool_open,
        '        {{- \'<|im_start|>\' + message.role }}{% generation %}\n'
        '        {%- if message.content is defined',
        1,
    )
    t = t.replace(
        shared_branch,
        # close the tool-call span, then split assistant out of the shared arm
        '        {{- \'<|im_end|>\\n\' }}{% endgeneration %}\n'
        '    {%- elif message.role == "assistant" %}\n'
        '        {{- \'<|im_start|>\' + message.role + \'\\n\' }}'
        '{% generation %}{{- message.content + \'<|im_end|>\' + \'\\n\' }}'
        '{% endgeneration %}\n'
        '    {%- elif message.role == "user" or message.role == "system" %}\n'
        '        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content'
        ' + \'<|im_end|>\' + \'\\n\' }}',
        1,
    )
    return t


def _verify_assistant_masking(tok, base_template: str, masked_template: str) -> None:
    """Fail-fast guard for the masked template (runs before any training).

    Asserts two invariants on representative single-turn, multi-turn, ``<think>``,
    and tool conversations:
      1. **Render-identity** - the masked template renders byte-identical text to the
         base template (the mask must not perturb what the model sees).
      2. **Correct masking** - assistant response tokens (and their ``<|im_end|>``)
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
    tokens. Never raises - returns ``{}`` if the tokenizer/template can't render.
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
    # Largest `_tokens` seen. The v5 builder measures every row with this exact
    # tokenizer and revision and gates at build time, so when the field is present
    # the caller can skip re-tokenising 239k rows on all eight ranks -- a 13.5-minute
    # per-rank pass that drops nothing. Stays None if any row lacks the field, in
    # which case the caller does the work rather than trusting an absent measurement.
    max_tokens_seen: "int | None" = 0
    # Streamed, not read_text().splitlines(). The v5 mixture is 1.79GB, so the old
    # form held ~1.8GB of str plus ~1.8GB of list per rank, on all eight ranks at
    # once, on a box that has already lost a run to host-memory exhaustion.
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = (rec["messages"] if isinstance(rec, dict) and "messages" in rec
                        else rec)
            row = {"messages": messages}
            rows.append(row)
            if max_tokens_seen is not None:
                t = rec.get("_tokens") if isinstance(rec, dict) else None
                if isinstance(t, int) and t > 0:
                    max_tokens_seen = max(max_tokens_seen, t)
                else:
                    max_tokens_seen = None
            # Up-weight TRUE repair turns only (by provenance kind), not the whole
            # kernel bucket: after mixing, repairs AND optimization wins are both
            # tagged _source="kernel_repair_opt", so keying on _source doubled wins
            # too and reshaped the trained mix. Provenance kind is the honest signal
            # and also catches DAgger repairs (which lacked the bucket tag).
            if factor > 1 and isinstance(rec, dict):
                prov = rec.get("_provenance") or {}
                is_repair = ((prov.get("kind") == "repair")
                             or (rec.get("_source") == "repair"))
                if is_repair:
                    rows.extend([dict(row) for _ in range(factor - 1)])
    out = Dataset.from_list(rows)
    # Carried out-of-band: the Dataset itself keeps only `messages`, because that is
    # the one column the trainer reads.
    out._kore_max_tokens = max_tokens_seen  # type: ignore[attr-defined]
    return out


def load_eval_datasets(path: str | Path) -> dict:
    """Load the held-out slice as one ``Dataset`` per ``_eval_group``.

    Returned as a dict because ``Trainer`` accepts a mapping of eval datasets and
    logs ``eval_<key>_loss`` for each. That per-capability breakdown is the whole
    point: this run teaches kernels while risking instruction-following, and a
    single pooled eval loss averages the two together, so the number could sit flat
    while one half climbs and the other falls.

    No repair up-weighting here even when training uses it -- duplicating eval rows
    would just reweight the mean of a metric we want comparable across steps.
    """
    from datasets import Dataset

    groups: dict[str, list] = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict) or "messages" not in rec:
                continue
            group = str(rec.get("_eval_group") or "unlabelled")
            groups.setdefault(group, []).append({"messages": rec["messages"]})
    return {g: Dataset.from_list(r) for g, r in sorted(groups.items()) if r}


def _filter_overlong(ds, tok, max_length: int):
    """Drop rows whose chat-rendered token length exceeds ``max_length``.

    Returns ``(filtered_dataset, n_dropped)``. Deterministic so every rank computes
    the identical filtered set (consistent FSDP data shards). A row that fails to
    render is KEPT (conservative - let the trainer handle it). Returns the original
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
    # Resolve WHICH weights this stage trains before importing the heavy stack, so
    # an unpinned production run or a changed checkpoint fails in milliseconds
    # instead of after a multi-minute 14B load on every rank.
    identity = model_identity_for_config(config, stage="sft")
    log_model_identity(log, identity)
    log_stage_preflight(log, run_stage_preflight(
        stage="sft", config=config, model_spec=identity.spec,
        inspection=identity.inspection))

    # The reporting backend is imported by the Trainer AFTER the model is on the GPUs,
    # so a missing one is reported at the most expensive possible moment. Run 6620
    # died exactly there: report_to="tensorboard" with tensorboard absent from the
    # venv, discovered after eight ranks had loaded 61GB and tokenised 206k rows,
    # ~11 minutes in. Constructing TrainingArguments does NOT catch it -- the args
    # accept the string and the callback is only instantiated inside
    # Trainer.__init__ -- so the import has to be checked explicitly.
    for _backend in str(getattr(config, "report_to", "") or "").replace(",", " ").split():
        if _backend in ("none", "all"):
            continue
        _module = {"tensorboard": "tensorboard", "wandb": "wandb",
                   "mlflow": "mlflow", "comet_ml": "comet_ml"}.get(_backend)
        if not _module:
            continue
        try:
            __import__(_module)
        except ImportError as _exc:
            raise RuntimeError(
                f"sft: report_to={_backend!r} but {_module!r} is not importable "
                f"({_exc}). Install it (`pip install {_module}`) or set "
                'report_to="none". Checked here because the Trainer builds this '
                "callback only after the model is loaded on every rank, which turns "
                "a missing package into a wasted allocation."
            ) from _exc

    # The dataset is not read until after the model load (~minutes x world_size),
    # so a bad path would otherwise cost a full 14B load on every rank to report.
    if not Path(dataset_path).is_file():
        raise FileNotFoundError(
            f"sft: training dataset not found at {str(dataset_path)!r} "
            f"(cwd={Path.cwd()}). Run `cd data/release && ./reassemble.sh` to "
            "materialize the packaged corpus, or point dataset_path at the built shard."
        )
    # is_file() also passes on an empty or truncated shard, and an empty Dataset is
    # only noticed after every rank has loaded 61GB of weights. Cost of checking
    # here is one line read; cost of not checking is an allocation.
    _n_rows = 0
    with open(dataset_path, encoding="utf-8", errors="ignore") as _fh:
        _first = ""
        for _line in _fh:
            if _line.strip():
                _n_rows += 1
                if not _first:
                    _first = _line
    if _n_rows == 0:
        raise ValueError(f"sft: training dataset {str(dataset_path)!r} has no rows")
    try:
        if "messages" not in json.loads(_first):
            raise ValueError(
                f"sft: first row of {str(dataset_path)!r} has no 'messages' key; "
                "the loader reads nothing else, so this file would train on nothing")
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"sft: first row of {str(dataset_path)!r} is not valid JSON: {exc}") from exc
    log.info("sft: dataset preflight ok", path=str(dataset_path), rows=_n_rows)

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

    tok = AutoTokenizer.from_pretrained(config.model_id, **identity.load_kwargs)
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

    _attn_impl = preferred_attn_impl()
    model_kwargs = {"torch_dtype": torch.bfloat16,
                    "attn_implementation": _attn_impl}
    if not use_fsdp:
        model_kwargs["device_map"] = "auto"
    # Re-fingerprint immediately before the load, closing the window between
    # preflight and load (no-op when nothing was fingerprinted).
    identity.validate_before_load()
    model = AutoModelForCausalLM.from_pretrained(config.model_id,
                                                 **identity.load_kwargs,
                                                 **model_kwargs)

    # Packing safety guard (audit R2 / THEME B): TRL bfd packing needs a FLASH-ATTN
    # backend to build the block-diagonal (per-document) attention mask. On the SDPA
    # runtime it silently falls back to a plain causal mask, so tokens attend ACROSS
    # packed documents -- cross-contamination that corrupts every packed row. Enforce
    # the invariant at runtime (not just in the config comment): never pack on SDPA.
    _packing = bool(config.packing)
    if _packing and _attn_impl != "flash_attention_2":
        log.info("sft: packing DISABLED -- attn backend is SDPA (not flash_attention_2); "
                 "packing on SDPA cross-contaminates documents", attn=_attn_impl)
        _packing = False
    # Activation checkpointing (routed through fsdp_config) is INCOMPATIBLE with
    # the KV cache: the cache changes the tensor count between forward and
    # recompute -> torch.utils.checkpoint CheckpointError. HF's Trainer only
    # auto-disables use_cache when TrainingArguments.gradient_checkpointing is set,
    # which we turn OFF under FSDP (FSDP owns checkpointing), so disable it here.
    model.config.use_cache = False

    # MoE load balancing stays OFF, deliberately, and this is the note explaining
    # why the obvious fix is wrong.
    #
    # The checkpoint ships router_aux_loss_coef=0.001 and Qwen3MoeForCausalLM only
    # adds that term when the forward is asked for router logits, so the
    # coefficient is configured and inert. Setting output_router_logits=True looks
    # like the fix and is not one, for three independent reasons:
    #
    # 1. It adds no gradient here. Router logits are harvested by monkey-patching
    #    Qwen3MoeSparseMoeBlock.forward, and that block sits INSIDE the region
    #    wrapped by gradient checkpointing. With use_reentrant=True (set below,
    #    because non-reentrant raises CheckpointError on this SDPA runtime) the
    #    wrapped forward runs under torch.no_grad(), so every captured tensor is
    #    detached. Measured: router-gate gradient norms are byte-identical with
    #    the flag on and off. The aux term still lands in the REPORTED loss as a
    #    constant ~+0.008, which only breaks comparability with earlier runs.
    #
    # 2. If the gradient were live it would be 8x over-weighted. The CE term is
    #    normalised by a global token count and survives the num_processes/FSDP
    #    scaling exactly; the aux term is a per-micro-batch mean with no such
    #    denominator, so it comes out at coef * gradient_accumulation_steps.
    #
    # 3. If it were live AND correctly scaled it would still be the wrong loss.
    #    Qwen3 was pretrained with a GLOBAL-batch balancing loss; transformers
    #    computes the MICRO-batch variant, which at per_device_train_batch_size=2
    #    is effectively per-sequence. Qiu et al. (ACL 2025 Long 249), the paper
    #    behind Qwen3's choice, show that regime forces even domain-specific
    #    sequences to spread across all experts and measurably suppresses the
    #    expert specialisation we want -- and this corpus is 86% kernel code.
    #
    # It also costs up to ~40 GB of transient VRAM at the longest length band,
    # because load_balancing_loss_func materialises an int64 one-hot of shape
    # (num_layers * B * S, top_k, num_experts) plus float and mask copies.
    #
    # What routing actually needs here is measurement, not a loss: per-layer
    # expert-load entropy accumulated per OPTIMIZER step (not per micro-batch),
    # via forward hooks gated on `not torch.is_grad_enabled()` so the checkpoint
    # recompute is not double counted.
    if getattr(model.config, "num_experts", None):
        log.info("sft: MoE load-balancing loss intentionally OFF",
                 num_experts=getattr(model.config, "num_experts", None),
                 experts_per_tok=getattr(model.config, "num_experts_per_tok", None),
                 router_aux_loss_coef=getattr(model.config, "router_aux_loss_coef", None),
                 reason="detached under reentrant checkpointing; micro-batch LBL at "
                        "batch=2 suppresses expert specialisation")

    ds = load_sft_dataset(dataset_path, repair_weight=config.repair_loss_weight)
    _premeasured = getattr(ds, "_kore_max_tokens", None)
    # Drop rows whose rendered length exceeds max_seq_length - ONLY under completion-
    # only loss. There, an over-length row whose assistant span is truncated away would
    # make TRL raise ("no assistant tokens"), and even a partially-cut assistant tail
    # (missing <|im_end|>) teaches run-on. With full-sequence loss we keep TRL's stock
    # truncation (the prior status quo). Deterministic -> identical filtered set on every
    # rank (consistent FSDP shards). NB: on the current multicap mix this drops ~8.7% of
    # rows, almost entirely pathologically-long (>16k tok) math_reasoning CoTs (a data-
    # quality item for the data pass) and only ~0.6% of kernels.
    #
    # Skipped when the builder already measured every row. This pass costs ~3.4ms per
    # row -- 13.5 minutes for 239k rows, single-threaded, on all eight ranks
    # simultaneously -- and TRL re-tokenises the whole corpus properly afterwards
    # anyway with num_proc=32 under main_process_first. On the v5 mixture it dropped
    # exactly zero rows, because the build gates at 16,896 tokens using this same
    # tokenizer and revision. `_kore_max_tokens` is the builder's own measurement, so
    # this is an exact comparison rather than a character-count bound; it is None
    # whenever any row lacked the field, and then the full pass runs.
    n_over = 0
    if assistant_only:
        if isinstance(_premeasured, int) and 0 < _premeasured <= config.max_seq_length:
            log.info("sft: overlong filter SKIPPED (rows pre-measured at build time)",
                     max_tokens_in_corpus=_premeasured,
                     max_seq_length=config.max_seq_length)
        else:
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

    # Held-out eval, one dataset per capability group. Off unless a path is set, so
    # the no-eval behaviour is unchanged for callers that do not want the ~1% of
    # wall clock this costs.
    eval_sets: dict = {}
    eval_kwargs: dict = {}
    _eval_path = str(getattr(config, "eval_dataset_path", "") or "")
    if _eval_path:
        if not Path(_eval_path).is_file():
            raise FileNotFoundError(f"eval_dataset_path does not exist: {_eval_path}")
        eval_sets = load_eval_datasets(_eval_path)
        if not eval_sets:
            raise ValueError(f"eval_dataset_path parsed to zero rows: {_eval_path}")
        eval_kwargs = {
            "eval_strategy": "steps",
            "eval_steps": int(getattr(config, "eval_steps", 200)),
            "per_device_eval_batch_size": int(
                getattr(config, "per_device_eval_batch_size", 1)),
            # Baseline at step 0, i.e. the untrained model. Otherwise the first
            # measurement is step 200 and every delta is computed against a model
            # that has already been trained for 200 steps, hiding whatever those
            # steps cost.
            "eval_on_start": bool(getattr(config, "eval_on_start", True)),
            # Eval is a diagnostic, not a model-selection signal here (one epoch, a
            # fixed schedule, and nothing consumes best_model_at_end), so keep the
            # final checkpoint deterministic rather than metric-dependent.
            "load_best_model_at_end": False,
        }
        log.metric("sft.eval_groups",
                   **{g: len(d) for g, d in eval_sets.items()})

    args = TRLSFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=getattr(config, "lr_scheduler_kwargs", None) or {},
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        max_length=config.max_seq_length,
        packing=_packing,
        assistant_only_loss=bool(assistant_only),
        bf16=config.bf16,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        logging_steps=config.logging_steps,
        # OFF, against the HF default. With it on, a NaN or Inf micro-batch loss is
        # replaced in the logged average by the running average, so a diverging run
        # reports a clean curve while the gradient poisons the weights. _ObsCallback
        # alarms on the non-finite values this now lets through.
        logging_nan_inf_filter=False,
        save_steps=config.save_steps,
        save_total_limit=getattr(config, "save_total_limit", 2),
        # Weights-only checkpoints when asked: ~61GB instead of ~488GB at 30B, which
        # is what makes a write survivable on a volume other users can fill mid-save.
        save_only_model=bool(getattr(config, "save_only_model", False)),
        report_to=config.report_to,
        adam_beta2=getattr(config, "adam_beta2", 0.98),
        **({"logging_dir": config.logging_dir} if getattr(config, "logging_dir", "") else {}),
        dataloader_num_workers=getattr(config, "dataloader_num_workers", 8),
        dataloader_pin_memory=getattr(config, "dataloader_pin_memory", True),
        # THROUGHPUT (audit R2 perf): persist dataloader workers across the 3 SFT epochs
        # (they respawned every epoch) + deeper prefetch to keep the MI350X fed; and
        # group_by_length batches similar-length rows so dynamic pad-to-longest wastes
        # far less compute on the SDPA (no-packing) runtime with micro-batch>1.
        dataloader_persistent_workers=getattr(config, "dataloader_num_workers", 8) > 0,
        dataloader_prefetch_factor=getattr(config, "dataloader_prefetch_factor", 4),
        group_by_length=config.group_by_length,
        dataset_num_proc=getattr(config, "dataset_num_proc", 32),
        **eval_kwargs,
        **fsdp_kwargs,
    )
    # ------------------------------------------------------------------ #
    # MoE routing instrumentation.
    #
    # This replaces the load-balancing loss rather than complementing it. That loss
    # cannot be correctly scoped here (see the note at the model load: detached
    # under reentrant checkpointing, 8x over-weighted by the accumulation path, and
    # micro-batch rather than global-batch, which is the variant Qwen3 was
    # deliberately NOT pretrained with). What is genuinely missing is not pressure
    # on the router but VISIBILITY into it: nothing in this run logs routing at all,
    # so "did the router collapse onto a few of the 128 experts?" is unanswerable,
    # and the warmup ratio and gradient clip are justified by a hypothesis nobody
    # can check.
    #
    # Counting, not penalising. Hooks accumulate expert selections and reset per
    # OPTIMIZER step, so the entropy reported is the global-batch load -- the
    # quantity that actually predicts specialisation -- not the per-micro-batch
    # load. The `not torch.is_grad_enabled()` gate is what keeps the count honest:
    # under reentrant checkpointing the layer forward runs twice, once with grad
    # disabled and once during the backward recompute with it enabled, so counting
    # only the former de-duplicates exactly.
    _expert_counts: dict = {}

    def _install_router_hooks(m) -> int:
        try:
            from transformers.models.qwen3_moe.modeling_qwen3_moe import (
                Qwen3MoeSparseMoeBlock)
        except Exception:  # noqa: BLE001 - a dense model has no routers to watch
            return 0
        n_experts = int(getattr(m.config, "num_experts", 0) or 0)
        if not n_experts:
            return 0

        def _make(layer_idx: int):
            def _hook(_mod, _inp, out):
                if torch.is_grad_enabled():
                    return  # backward recompute; already counted
                logits = out[1] if isinstance(out, tuple) and len(out) > 1 else None
                if logits is None:
                    return
                with torch.no_grad():
                    k = int(getattr(m.config, "num_experts_per_tok", 8) or 8)
                    top = logits.detach().float().topk(k, dim=-1).indices.reshape(-1)
                    c = torch.bincount(top, minlength=n_experts).to("cpu")
                    prev = _expert_counts.get(layer_idx)
                    _expert_counts[layer_idx] = c if prev is None else prev + c
            return _hook

        n = 0
        for name, mod in m.named_modules():
            if isinstance(mod, Qwen3MoeSparseMoeBlock):
                idx = next((int(t) for t in name.split(".") if t.isdigit()), n)
                mod.register_forward_hook(_make(idx))
                n += 1
        return n

    def _routing_stats(watch=(0, 24, 47)) -> dict:
        """Normalised load entropy, peak share and dead-expert count per layer."""
        import math
        out: dict = {}
        for idx in watch:
            c = _expert_counts.get(idx)
            if c is None or float(c.sum()) <= 0:
                continue
            total = float(c.sum())
            n = int(c.numel())
            p = (c.float() / total).tolist()
            h = -sum(x * math.log(x) for x in p if x > 0) / math.log(n)
            uniform = 1.0 / n
            out[f"L{idx}_load_entropy"] = round(h, 4)
            out[f"L{idx}_max_share"] = round(max(p) / uniform, 2)
            out[f"L{idx}_dead"] = int(sum(1 for x in p if x < 0.1 * uniform))
        return out

    _n_hooks = _install_router_hooks(model)
    if _n_hooks:
        log.info("sft: MoE routing hooks installed", blocks=_n_hooks,
                 note="counts reset per optimizer step; global-batch load, not micro")

    # Lightweight per-log-step observability callback (guarded transformers import).
    from transformers import TrainerCallback

    class _ObsCallback(TrainerCallback):
        """Per-log-step observability, plus a divergence alarm.

        The alarm exists because the Trainer's default hides exactly the failure it
        should surface: with logging_nan_inf_filter on, a NaN or Inf micro-batch
        loss is REPLACED in the logged average by the running average, so a run
        training on poisoned weights reports a clean curve. That filter is turned
        off in the args below; this is the other half, which says so loudly.
        """

        def __init__(self):
            self._recent: list[float] = []
            self._entropy0: dict = {}
            self._eval0: dict = {}

        def _on_eval_log(self, state, logs: dict):
            """Report per-capability eval loss, and alarm on the divergence that
            defines catastrophic forgetting.

            The signature of forgetting is not "loss went up" -- taught capabilities
            are supposed to improve while retained ones drift a little. It is the two
            moving in OPPOSITE directions: kernel loss falling while chat or
            instruction-following climbs. Alarming on that shape rather than on an
            absolute threshold is what makes this readable at step 200 instead of in
            hindsight.
            """
            per_group = {k[len("eval_"):-len("_loss")]: v
                         for k, v in logs.items()
                         if k.startswith("eval_") and k.endswith("_loss")
                         and isinstance(v, (int, float))}
            if not per_group:
                return
            step = int(state.global_step)
            for group, val in per_group.items():
                self._eval0.setdefault(group, val)
            deltas = {g: (v - self._eval0[g]) / self._eval0[g]
                      for g, v in per_group.items() if self._eval0.get(g)}
            taught = [d for g, d in deltas.items() if g.startswith("kernel")]
            retained = {g: d for g, d in deltas.items() if not g.startswith("kernel")}
            if taught and retained:
                learning = min(taught) < -0.02
                regressed = {g: round(d, 4) for g, d in retained.items() if d > 0.05}
                if learning and regressed:
                    log.warn("sft: RETENTION REGRESSION -- kernel loss is falling "
                             "while retained capabilities climb", step=step,
                             kernel_delta=round(min(taught), 4), regressed=regressed,
                             note="raise the replay share or lower the LR")
            log.event("sft_eval", step=step,
                      **{f"{g}_loss": round(float(v), 5)
                         for g, v in sorted(per_group.items())},
                      **{f"{g}_delta": round(d, 4) for g, d in sorted(deltas.items())})

        def on_train_begin(self, args, state, control, **kwargs):
            # Fingerprint the data order. group_by_length uses LengthGroupedSampler,
            # which Trainer constructs with NO generator, so its permutation comes
            # from the global CPU RNG and is not bound to args.seed; accelerate's
            # seedable-sampler path only covers a plain RandomSampler. On resume the
            # Trainer computes how many batches to skip from the NEW dataloader and
            # calls skip_first_batches, which is only correct if the permutation is
            # reproduced. If it is not, some rows train twice and others never, with
            # nothing raising. This makes that falsifiable: compare the fingerprint
            # across a resume, and if it differs the skip was wrong.
            try:
                trainer = kwargs.get("train_dataloader")
                sampler = getattr(trainer, "sampler", None) or getattr(
                    trainer, "batch_sampler", None)
                head = list(itertools.islice(iter(sampler), 8)) if sampler else None
            except Exception as exc:  # noqa: BLE001 - diagnostic only
                head, sampler = None, None
                log.info("sft: sampler fingerprint unavailable", err=str(exc)[:120])
            if head is not None:
                log.metric("sft.sampler_order",
                           sampler=type(sampler).__name__,
                           first_8=str(head),
                           resumed=bool(state.global_step),
                           note="must match across a resume, or skip_first_batches "
                                "replayed a different ordering")

        def on_optimizer_step(self, args, state, control, **kwargs):
            # Reset AFTER the step so each logged value covers exactly one global
            # batch. Resetting per micro-batch would report the per-sequence load,
            # which is the misleading quantity.
            _expert_counts.clear()

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs:
                self._on_eval_log(state, logs)
                return
            loss, gnorm = logs.get("loss"), logs.get("grad_norm")
            step = int(state.global_step)

            def _bad(v):
                return v is not None and (v != v or v in (float("inf"), float("-inf")))

            if _bad(loss) or _bad(gnorm):
                log.warn("sft: NON-FINITE loss or grad_norm -- the run is diverging",
                         step=step, loss=loss, grad_norm=gnorm)
            elif isinstance(gnorm, (int, float)) and gnorm > 0:
                # A spike against the running median, not the mean: one large step
                # would drag a mean far enough to hide the next one.
                if len(self._recent) >= 20:
                    med = sorted(self._recent)[len(self._recent) // 2]
                    if med > 0 and gnorm > 10 * med:
                        log.warn("sft: grad_norm spike", step=step, grad_norm=gnorm,
                                 running_median=round(med, 4), ratio=round(gnorm / med, 1))
                self._recent.append(float(gnorm))
                if len(self._recent) > 200:
                    self._recent.pop(0)

            routing = _routing_stats() if _n_hooks else {}
            if routing:
                # Collapse is loudest in the deeper layers, so alarm on any watched
                # layer rather than an average that would hide one bad layer.
                first = self._entropy0
                for key, val in routing.items():
                    if not key.endswith("_load_entropy"):
                        continue
                    if first.get(key) is None:
                        first[key] = val
                    elif first[key] > 0 and val < 0.85 * first[key]:
                        log.warn("sft: router load entropy collapsing", step=step,
                                 layer=key, now=val, at_start=first[key])
            log.event("sft_step", step=step, loss=loss,
                      lr=logs.get("learning_rate"),
                      epoch=round(float(state.epoch), 4) if state.epoch is not None else None,
                      grad_norm=gnorm, **routing, **gpu_mem_snapshot())

    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         **({"eval_dataset": eval_sets} if eval_sets else {}),
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
        # Periodic checkpoints are SHARDED so the run can actually resume (see
        # build_fsdp_kwargs), but the artifact handed to the next stage has to be a
        # plain HF checkpoint that from_pretrained can open on any mesh. Flip the
        # plugin to FULL_STATE_DICT for this one final save so the handoff is
        # unchanged. Only the SAVE side of FULL is used here, which is the side that
        # was always working -- it is loading a consolidated 244 GB optimizer that
        # SIGBUSes, and this save writes no optimizer state at all.
        _consolidated = False
        try:
            plugin = trainer.accelerator.state.fsdp_plugin
            plugin.set_state_dict_type("FULL_STATE_DICT")
            _consolidated = True
        except Exception as exc:
            log.warn("sft: could not switch to FULL_STATE_DICT for the final save; "
                     "will merge the sharded weights instead", error=str(exc))
        log.info("sft: saving full-FT model", out=config.output_dir,
                 consolidated=_consolidated)
        trainer.save_model(config.output_dir)
        if not _consolidated:
            # Fallback: merge the sharded weights offline. Rank 0 only, and after a
            # barrier, so the shards are all on disk before they are read.
            trainer.accelerator.wait_for_everyone()
            if trainer.args.should_save:
                from accelerate.utils import merge_fsdp_weights
                sharded = Path(config.output_dir) / "pytorch_model_fsdp_0"
                if sharded.is_dir():
                    log.info("sft: merging sharded weights into a full checkpoint",
                             src=str(sharded), out=config.output_dir)
                    merge_fsdp_weights(str(sharded), config.output_dir,
                                       safe_serialization=True)
    # Restore the pristine (un-tagged) chat template before saving so the checkpoint's
    # tokenizer is byte-identical to the base. The {% generation %} markers are only
    # needed for THIS run's mask generation (they render identically at inference).
    #
    # ONE writer, and a barrier first. trainer.save_model has already written the
    # tokenizer from rank 0 -- with the tagged template -- so this rewrite is
    # necessary, but running it unguarded had all eight ranks writing the same NFS
    # files concurrently. On NFSv3 that can leave a half-written
    # tokenizer_config.json or chat_template.jinja, which every downstream stage
    # then loads.
    if base_chat_template is not None:
        tok.chat_template = base_chat_template
    trainer.accelerator.wait_for_everyone()
    if trainer.args.should_save:
        tok.save_pretrained(config.output_dir)
        log.metric("sft_done", out=config.output_dir,
                   merged_lora=bool(config.use_lora), **gpu_mem_snapshot())
    trainer.accelerator.wait_for_everyone()
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
    Identity/preflight keys (``model_revision``, ``resource_preflight``, ...) are
    not ``SFTConfig`` fields, so they are split off and attached as attributes -
    the strict dataclass parse stays strict and a pinned config still parses.
    """
    # Underscore-prefixed keys are the repo's in-config comment convention
    # (``_comment_<field>``); midtrain already drops them, so SFT must too or a
    # documented launch config crashes the strict dataclass parse.
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    lora = d.pop("lora", None)
    d, runtime = split_runtime_settings(
        d, IDENTITY_CONFIG_KEYS + PREFLIGHT_CONFIG_KEYS
    )
    # dataset_path is a real SFTConfig field, so keep it on the config too.
    cfg = SFTConfig(**d)
    if lora is not None:
        cfg.lora = LoRAConfig(**lora)
    apply_runtime_settings(cfg, runtime)
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
