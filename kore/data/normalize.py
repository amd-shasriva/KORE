"""Upgrade existing KORE training shards to the canonical prompt/response contract.

Pillar 0 (contract unification) applied OFFLINE to data already on disk, so the
current SFT/DPO shards match exactly what the policy is trained to emit at
inference — without a full (GPU + teacher) regeneration. The single source of
truth is :mod:`kore.policy.format`; this module only rewrites message *content*:

  * any KORE kernel SYSTEM prompt  -> the canonical ``SYSTEM_PROMPT``;
  * any kernel ASSISTANT turn (repair ``<think>/<answer>``, headerless gold-win,
    ``CHANGE:`` data-gen shape, raw teacher text) -> the canonical
    ``ANALYSIS / PROPOSED_CHANGE / FULL_KERNEL`` via ``normalize_assistant``;
  * DPO ``chosen``/``rejected`` kernel completions -> the canonical
    ``wrap_full_kernel`` (kernel-only completion).

It is PURE (stdlib + kore.policy.format), deterministic, idempotent, and NEVER
touches non-kernel retention rows (general chat/code/math) or agentic tool
trajectories — those carry their own contracts and are left byte-for-byte intact.

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
# (QA) — only their SYSTEM prompt is canonicalized, assistant left as-is.
_KERNEL_NL_SOURCES = {"kernel_qa"}


def _is_kore_system(content: str) -> bool:
    return isinstance(content, str) and content.lstrip().startswith(_KORE_SYS_PREFIX)


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


def normalize_dpo_row(row: dict) -> tuple[dict, bool]:
    """Canonicalize a DPO pair's system prompt + kernel completions."""
    changed = False
    out = dict(row)
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
    return out, changed


def _detect_kind(first_row: dict) -> str:
    if isinstance(first_row, dict) and "messages" in first_row:
        return "sft"
    if isinstance(first_row, dict) and {"prompt", "chosen", "rejected"} <= set(first_row):
        return "dpo"
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
    fn = normalize_sft_row if kind == "sft" else normalize_dpo_row

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
