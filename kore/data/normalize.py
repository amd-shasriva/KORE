"""Upgrade existing KORE training shards to the canonical prompt/response contract.

Pillar 0 (contract unification) applied OFFLINE to data already on disk, so the
current SFT/DPO shards match exactly what the policy is trained to emit at
inference - without a full (GPU + teacher) regeneration. The single source of
truth is :mod:`kore.policy.format`; this module only rewrites message *content*:

  * any KORE kernel SYSTEM prompt  -> the canonical ``SYSTEM_PROMPT``;
  * any kernel ASSISTANT turn (repair ``<think>/<answer>``, headerless gold-win,
    ``CHANGE:`` data-gen shape, raw teacher text) -> the canonical
    ``ANALYSIS / PROPOSED_CHANGE / FULL_KERNEL`` via ``normalize_assistant``;
  * DPO ``chosen``/``rejected`` kernel completions -> the canonical
    ``wrap_full_kernel`` (kernel-only completion), while PRESERVING the pair's
    ``_provenance`` and lifting its baseline-anchored ``margin`` / ``weight`` /
    ``anchor`` to the top level for the trainer (the margin is never stripped).

It is PURE (stdlib + kore.policy.format), deterministic, idempotent, and NEVER
touches non-kernel retention rows (general chat/code/math) or agentic tool
trajectories - those carry their own contracts and are left byte-for-byte intact.

CLI::

    python -m kore.data.normalize data/full14b/sft/multicap.jsonl --in-place
    python -m kore.data.normalize data/full14b/dpo/pairs.jsonl   --out pairs.v2.jsonl
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from kore.policy.format import (
    SYSTEM_PROMPT,
    normalize_assistant,
    wrap_full_kernel,
)

# Distinctive prefix shared by every historical KORE *kernel* system prompt
# (both the old data-gen and old policy prompts open with this). General-chat /
# math / code retention rows have no system message, and the agentic Hermes system
# prompt opens differently, so matching this prefix never rewrites a non-kernel row.
_KORE_SYS_PREFIX = "You are KORE, an expert AMD GPU kernel engineer"

# _source tags whose ASSISTANT turns are kernel responses to canonicalize.
_KERNEL_SOURCES = {"kernel_repair_opt"}
# _source tags that are kernel-domain but whose assistant is natural language
# (QA) - only their SYSTEM prompt is canonicalized, assistant left as-is.
_KERNEL_NL_SOURCES = {"kernel_qa"}


def _is_kore_system(content: str) -> bool:
    return isinstance(content, str) and content.lstrip().startswith(_KORE_SYS_PREFIX)


def _normalize_msg_list(messages: list) -> tuple[list, bool]:
    """Canonicalize a chat message list: KORE system -> canonical, assistant turns
    -> canonical ANALYSIS/PROPOSED_CHANGE/FULL_KERNEL. Returns (new_messages, changed)."""
    if not isinstance(messages, list):
        return messages, False
    changed = False
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role, content = m.get("role"), m.get("content", "")
        if role == "system" and _is_kore_system(content) and content != SYSTEM_PROMPT:
            m = {**m, "content": SYSTEM_PROMPT}
            changed = True
        elif role == "assistant":
            nc = normalize_assistant(content)
            if nc != content:
                m = {**m, "content": nc}
                changed = True
        out.append(m)
    return out, changed


def _extract_kernel_only(content: str) -> str:
    """Re-wrap a DPO completion as a canonical kernel-only FULL_KERNEL block."""
    from kore.policy.format import _extract_kernel  # local: keep module import surface small

    k = _extract_kernel(content or "").strip()
    return wrap_full_kernel(k) if k else content


def normalize_sft_row(row: dict) -> tuple[dict, bool]:
    """Return ``(row, changed)`` with the kernel contract canonicalized.

    Only rewrites rows whose ``_source`` is a kernel source; everything else is
    returned unchanged. System prompts matching the KORE kernel prefix are swapped
    to the canonical prompt; kernel assistant turns are re-rendered.
    """
    src = row.get("_source")
    msgs = row.get("messages")
    if not isinstance(msgs, list):
        return row, False
    do_assistant = src in _KERNEL_SOURCES
    do_system = src in _KERNEL_SOURCES or src in _KERNEL_NL_SOURCES
    if not (do_assistant or do_system):
        return row, False
    changed = False
    new_msgs = []
    for m in msgs:
        if not isinstance(m, dict):
            new_msgs.append(m)
            continue
        role, content = m.get("role"), m.get("content", "")
        if role == "system" and do_system and _is_kore_system(content):
            if content != SYSTEM_PROMPT:
                m = {**m, "content": SYSTEM_PROMPT}
                changed = True
        elif role == "assistant" and do_assistant:
            new_content = normalize_assistant(content)
            if new_content != content:
                m = {**m, "content": new_content}
                changed = True
        new_msgs.append(m)
    if changed:
        row = {**row, "messages": new_msgs}
    return row, changed


def normalize_raw_record_row(row: dict) -> tuple[dict, bool]:
    """Canonicalize a RAW datagen record (RepairRecord / WinRecord) in place.

    RepairRecord carries ``messages`` (legacy ``<think>/<answer>`` assistant) and
    WinRecord carries ``trajectory`` (raw teacher ``CHANGE:``/etc.); both are
    upgraded to the canonical contract so that re-running the (CPU) build stage
    emits canonical SFT rows WITHOUT regenerating the GPU-verified records. All
    verified scalar fields (snr/wall/speedup/preferences/...) are untouched.
    """
    if not isinstance(row, dict):
        return row, False
    key = "messages" if isinstance(row.get("messages"), list) else (
        "trajectory" if isinstance(row.get("trajectory"), list) else None)
    if key is None:
        return row, False
    new_msgs, changed = _normalize_msg_list(row[key])
    if changed:
        row = {**row, key: new_msgs}
    return row, changed


def _preserve_and_backfill_margin(row: dict, out: dict) -> bool:
    """Carry the speed-grounding curation signals through normalization.

    Contract fix: normalization must NEVER strip a DPO pair's ``_provenance`` (it
    carries the baseline-anchored margin/weight/anchor that the trainer margin-weights
    and filters on, and that curation audits). We keep it verbatim AND, for shards
    written before the trainer-facing top-level fields existed, we UPGRADE the row by
    lifting ``margin`` / ``weight`` / ``anchor`` out of ``_provenance`` to the top
    level (``_provenance.speedup`` is the legacy chosen-vs-rejected ratio == margin).
    Returns True iff a field was backfilled. Rows without any provenance/margin signal
    are left byte-for-byte unchanged (never fabricate a preference weight).
    """
    prov = row.get("_provenance")
    prov = prov if isinstance(prov, dict) else {}
    changed = False
    # margin: prefer an explicit top-level value, else provenance margin, else the
    # legacy provenance ``speedup`` (chosen-vs-rejected ratio, which IS the margin).
    if "margin" not in out:
        margin = prov.get("margin", prov.get("speedup"))
        if margin is not None:
            out["margin"] = margin
            changed = True
    if "weight" not in out:
        w = prov.get("weight")
        if w is not None:
            out["weight"] = w
            changed = True
    if "anchor" not in out:
        a = prov.get("anchor")
        if a is not None:
            out["anchor"] = a
            changed = True
    return changed


def normalize_dpo_row(row: dict) -> tuple[dict, bool]:
    """Canonicalize a DPO pair's system prompt + kernel completions.

    Preserves the row's ``_provenance`` (and lifts its baseline-anchored margin /
    weight / anchor to the top level for the trainer; see
    :func:`_preserve_and_backfill_margin`) - the margin is NOT discarded.
    """
    changed = False
    out = dict(row)  # shallow copy carries _provenance, margin, weight, anchor verbatim
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        new_prompt = []
        for m in prompt:
            if (isinstance(m, dict) and m.get("role") == "system"
                    and _is_kore_system(m.get("content", ""))
                    and m.get("content") != SYSTEM_PROMPT):
                m = {**m, "content": SYSTEM_PROMPT}
                changed = True
            new_prompt.append(m)
        out["prompt"] = new_prompt
    for side in ("chosen", "rejected"):
        val = row.get(side)
        if isinstance(val, list):
            new_side = []
            for m in val:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    nc = _extract_kernel_only(m.get("content", ""))
                    if nc != m.get("content"):
                        m = {**m, "content": nc}
                        changed = True
                new_side.append(m)
            out[side] = new_side
    changed = _preserve_and_backfill_margin(row, out) or changed
    return out, changed


def _detect_kind(first_row: dict) -> str:
    if not isinstance(first_row, dict):
        return "sft"
    if {"prompt", "chosen", "rejected"} <= set(first_row):
        return "dpo"
    # raw datagen records: WinRecord (trajectory) or RepairRecord (messages + record markers)
    if "trajectory" in first_row:
        return "raw"
    if "messages" in first_row and (
            first_row.get("type") in ("repair", "win")
            or "failure_class" in first_row or "child_snr_db" in first_row):
        return "raw"
    if "messages" in first_row:
        return "sft"
    return "sft"


def normalize_file(in_path: str | Path, out_path: Optional[str | Path] = None,
                   in_place: bool = False, backup: bool = True) -> dict:
    """Normalize a JSONL shard in place or to ``out_path``. Returns stats.

    Auto-detects SFT vs DPO from the first row. Writes atomically via a temp file.
    """
    in_path = Path(in_path)
    rows = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
    if not rows:
        return {"path": str(in_path), "rows": 0, "changed": 0, "kind": "empty"}
    kind = _detect_kind(rows[0])
    fn = {"sft": normalize_sft_row, "dpo": normalize_dpo_row,
          "raw": normalize_raw_record_row}[kind]

    changed = 0
    out_rows = []
    for r in rows:
        nr, ch = fn(r)
        changed += int(ch)
        out_rows.append(nr)

    if in_place:
        out_path = in_path
    out_path = Path(out_path) if out_path else in_path.with_suffix(".normalized.jsonl")
    if in_place and backup:
        bak = in_path.with_suffix(in_path.suffix + ".pre_normalize.bak")
        if not bak.exists():
            shutil.copy2(in_path, bak)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")
    tmp.replace(out_path)
    return {"path": str(in_path), "out": str(out_path), "kind": kind,
            "rows": len(rows), "changed": changed}


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Normalize KORE shards to the canonical contract")
    p.add_argument("paths", nargs="+", help="JSONL shard(s) to normalize")
    p.add_argument("--out", default=None, help="output path (single input only)")
    p.add_argument("--in-place", action="store_true", help="overwrite input (keeps a .pre_normalize.bak)")
    p.add_argument("--no-backup", action="store_true", help="skip the backup on --in-place")
    args = p.parse_args(argv)
    if args.out and len(args.paths) != 1:
        print("--out requires exactly one input path", flush=True)
        return 2
    for path in args.paths:
        stats = normalize_file(path, out_path=args.out, in_place=args.in_place,
                               backup=not args.no_backup)
        print(json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
