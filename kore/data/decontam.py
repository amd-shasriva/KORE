"""Eval decontamination (Pillar 5 hygiene).

A credible "best dataset in the world" must PROVE the training data never
contains the held-out generalization set. KORE reserves whole operator families
(``attention``) + any arch-specific task as held-out (see
``kore.tasks.registry``), but today two leaks exist:
  1. the midtrain corpus ingests ALL ``kore/tasks/*.py`` — including the held-out
     attention kernels — as raw text (``source == "kore_tasks"``);
  2. nothing checks general-replay / mined corpus chunks for a copied held-out
     kernel.

Two gates, both import-light (registry is imported lazily so this module stays
usable in CPU tests without the task tree loaded eagerly):

  * :func:`is_contaminated_record` — a labeled record whose op family is held out.
  * :func:`build_heldout_ngrams` + :func:`contaminated_by_text` — n-gram
    containment of arbitrary text against the held-out reference sources (catches
    a held-out kernel copied into a corpus/replay chunk).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable, Optional


@lru_cache(maxsize=1)
def heldout_task_ids() -> frozenset[str]:
    from kore.tasks.registry import heldout_tasks
    try:
        return frozenset(t.task_id for t in heldout_tasks())
    except Exception:  # noqa: BLE001 - registry unavailable (e.g. minimal test env)
        return frozenset()


@lru_cache(maxsize=1)
def heldout_families() -> frozenset[str]:
    from kore.tasks.registry import HELDOUT_FAMILIES
    return frozenset(HELDOUT_FAMILIES)


def _family_of(op_or_task: str) -> str:
    """Infer the operator family from an operation / task_id string (no Task obj)."""
    op = (op_or_task or "").lower()
    if "attn" in op or "attention" in op:
        return "attention"
    if "topk" in op:
        return "moe_router"
    if "moe" in op:
        return "moe"
    if "rmsnorm" in op:
        return "rmsnorm"
    if "layernorm" in op:
        return "layernorm"
    if "gemm" in op or "matmul" in op:
        return "gemm"
    if "quant" in op:
        return "quant"
    if "rope" in op:
        return "rope"
    if "softmax" in op:
        return "softmax"
    if "gelu" in op or "silu" in op or "relu" in op:
        return "activation"
    return op or "other"


def record_family(rec: Any) -> str:
    d = rec if isinstance(rec, dict) else getattr(rec, "__dict__", {})
    return _family_of(str(d.get("operation") or d.get("task_id") or ""))


def is_contaminated_record(rec: Any) -> bool:
    """True if a labeled record belongs to a held-out family or task id."""
    d = rec if isinstance(rec, dict) else getattr(rec, "__dict__", {})
    tid = str(d.get("task_id") or "")
    if tid and tid in heldout_task_ids():
        return True
    return record_family(rec) in heldout_families()


def decontaminate_records(records: Iterable[Any]) -> tuple[list, dict]:
    """Drop labeled records whose op family/task is held out. Returns (clean, stats)."""
    clean, dropped = [], 0
    for r in records:
        if is_contaminated_record(r):
            dropped += 1
            continue
        clean.append(r)
    clean_list = clean
    return clean_list, {"n_dropped_heldout": dropped, "n_kept": len(clean_list)}


# --------------------------------------------------------------------------- #
# Text-level (n-gram containment) decontamination for corpus / replay chunks
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]", text or "")


def ngram_set(text: str, n: int = 8) -> set[str]:
    toks = _tokens(text)
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


@lru_cache(maxsize=4)
def heldout_source_texts() -> tuple[str, ...]:
    """Raw source of every held-out task dir (seed/reference/driver .py).

    Reads ``kore/tasks/<task_id>/*.py`` directly by held-out task id + the known
    registry ``TASKS_DIR`` (no dependency on Task-object internals), so it is robust
    across Task representations.
    """
    from pathlib import Path
    out: list[str] = []
    try:
        from kore.tasks.registry import TASKS_DIR
        for tid in heldout_task_ids():
            d = Path(TASKS_DIR) / tid
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.py")):
                try:
                    out.append(p.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
    except Exception:  # noqa: BLE001
        return tuple()
    return tuple(out)


def build_heldout_ngrams(n: int = 8, extra_sources: Optional[Iterable[str]] = None) -> set[str]:
    """Union of n-grams over all held-out reference sources (+ any extras)."""
    grams: set[str] = set()
    for src in heldout_source_texts():
        grams |= ngram_set(src, n)
    for src in (extra_sources or []):
        grams |= ngram_set(src, n)
    return grams


def contaminated_by_text(text: str, heldout_ngrams: set[str], n: int = 8,
                         threshold: float = 0.10) -> bool:
    """True if >= ``threshold`` fraction of ``text``'s n-grams are held-out."""
    if not heldout_ngrams:
        return False
    grams = ngram_set(text, n)
    if not grams:
        return False
    overlap = sum(1 for g in grams if g in heldout_ngrams) / len(grams)
    return overlap >= threshold


def decontaminate_corpus(rows: Iterable[dict], text_key: str = "text",
                         n: int = 8, threshold: float = 0.10) -> tuple[list[dict], dict]:
    """Drop corpus rows whose text overlaps the held-out reference sources.

    Returns ``(clean_rows, stats)``. If the held-out sources can't be loaded (no
    registry), it is a safe no-op (keeps everything, reports 0 dropped).
    """
    heldout = build_heldout_ngrams(n)
    clean, dropped = [], 0
    for r in rows:
        if contaminated_by_text(str(r.get(text_key, "")), heldout, n, threshold):
            dropped += 1
            continue
        clean.append(r)
    return clean, {"n_dropped_contaminated": dropped, "n_kept": len(clean),
                   "heldout_ngrams": len(heldout)}


__all__ = [
    "heldout_task_ids", "heldout_families", "record_family",
    "is_contaminated_record", "decontaminate_records",
    "ngram_set", "build_heldout_ngrams", "contaminated_by_text",
    "decontaminate_corpus",
]
