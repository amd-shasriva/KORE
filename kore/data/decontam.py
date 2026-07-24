"""Eval decontamination (Pillar 5 hygiene).

A credible "best dataset in the world" must PROVE the training data never
contains the held-out generalization set. KORE reserves the structurally-distinct
MLA and paged-KV-decode product leaves, 43 exact near-probe provenance roots, and
foreign architecture/dtype slices (core attention otherwise trains); see the
versioned task taxonomy. Two leak paths must be closed:
  1. the midtrain corpus ingests ALL ``kore/tasks/*.py`` - including the held-out
     MLA / paged-attention kernels - as raw text (``source == "kore_tasks"``);
  2. nothing checks general-replay / mined corpus chunks for a copied held-out
     kernel.

Two gates, both import-light (registry is imported lazily so this module stays
usable in CPU tests without the task tree loaded eagerly):

  * :func:`is_contaminated_record` - a labeled record whose op family is held out.
  * :func:`build_heldout_ngrams` + :func:`contaminated_by_text` - n-gram
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
    return frozenset(t.task_id for t in heldout_tasks())


@lru_cache(maxsize=1)
def heldout_families() -> frozenset[str]:
    from kore.tasks.taxonomy import WHOLE_FAMILY_HOLDOUTS
    return frozenset(WHOLE_FAMILY_HOLDOUTS)


def _family_of(op_or_task: str) -> str:
    """Canonical product family adapter for unregistered text identities."""
    from kore.tasks.taxonomy import product_family_for_name
    return product_family_for_name(op_or_task) or "unclassified"


def record_family(rec: Any) -> str:
    from kore.tasks.registry import product_family_for_record
    return product_family_for_record(rec)


def is_contaminated_record(rec: Any) -> bool:
    """True if a labeled record belongs to a held-out family or task id."""
    from kore.tasks.registry import is_heldout_record
    return is_heldout_record(rec)


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


def eval_benchmark_texts() -> tuple[str, ...]:
    """Primary text of every RETENTION eval-benchmark item (MMLU questions, HumanEval /
    LiveCodeBench prompts, IFEval prompts, BFCL / MT-Bench questions) so the CPT /
    general-replay corpus can be decontaminated against the gate's OWN test set.

    Without this a general shard that happens to carry an eval question is trained on,
    which INFLATES the retention gate (train-on-test) and lets the gate rubber-stamp a
    model that memorized the benchmark. Uses the bundled SMOKE sets (offline, no
    network); safe no-op if retention is unavailable (audit R2 midtrain)."""
    out: list[str] = []
    try:
        from kore.eval.retention import DEFAULT_BENCHES, load_bench
    except Exception:  # noqa: BLE001 - retention optional at corpus-build time
        return tuple()
    _fields = ("question", "prompt", "text", "instruction")
    for name in DEFAULT_BENCHES:
        try:
            for it in load_bench(name):
                if not isinstance(it, dict):
                    continue
                for f in _fields:
                    v = it.get(f)
                    if isinstance(v, str) and v.strip():
                        out.append(v)
                        break
        except Exception:  # noqa: BLE001 - one bad bench must not abort decontam
            continue
    return tuple(out)


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


def decontaminate_chat_rows(rows: Iterable[dict], n: int = 8,
                            threshold: float = 0.10,
                            heldout_ngrams: Optional[set] = None) -> tuple[list[dict], dict]:
    """Drop chat rows ({"messages": [...]}) whose combined text overlaps held-out src.

    For the general-replay slices (code/math/chat/tool) - a KernelBook/OpenCode row
    could carry a held-out MLA / paged-attention kernel. Pass a prebuilt
    ``heldout_ngrams`` to avoid recomputing it per slice. Safe no-op when held-out
    sources can't be loaded.
    """
    heldout = heldout_ngrams if heldout_ngrams is not None else build_heldout_ngrams(n)
    if not heldout:
        rows = list(rows)
        return rows, {"n_dropped_contaminated": 0, "n_kept": len(rows)}
    clean, dropped = [], 0
    for r in rows:
        text = " ".join(m.get("content", "") for m in r.get("messages", [])
                        if isinstance(m, dict))
        if contaminated_by_text(text, heldout, n, threshold):
            dropped += 1
            continue
        clean.append(r)
    return clean, {"n_dropped_contaminated": dropped, "n_kept": len(clean)}


def decontaminate_corpus(rows: Iterable[dict], text_key: str = "text",
                         n: int = 8, threshold: float = 0.10,
                         extra_sources: Optional[Iterable[str]] = None) -> tuple[list[dict], dict]:
    """Drop corpus rows whose text overlaps the held-out reference sources.

    ``extra_sources`` adds more reference texts to decontaminate against -- e.g. the
    retention eval-benchmark texts (:func:`eval_benchmark_texts`) so the CPT corpus
    never trains on the gate's own test set. Returns ``(clean_rows, stats)``. If no
    reference sources can be loaded it is a safe no-op (keeps everything).
    """
    heldout = build_heldout_ngrams(n, extra_sources=extra_sources)
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
    "eval_benchmark_texts",
    "decontaminate_corpus", "decontaminate_chat_rows",
]
