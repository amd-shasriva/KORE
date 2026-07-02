"""Multi-capability SFT mixing (KORE Stage-1 anti-catastrophic-forgetting recipe).

Assembles the final Stage-1 SFT corpus by sampling several capability sources at
their configured fractions (see :class:`~kore.policy.configs.MultiCapSFTConfig`):

    kernel_repair_opt : 0.35   broken->fixed + optimization kernel turns
    kernel_qa         : 0.10   kernel/GPU/ROCm QA + explanation (gen_qa.py)
    agentic_tooluse   : 0.10   Hermes tool-use trajectories (the orchestration skill)
    general_code      : 0.20   \
    math_reasoning    : 0.15    >  ~45% general replay = the retention backbone
    general_chat      : 0.10   /

Design goals:
  * Sample each source at its target fraction of ``total``.
  * If a source is SHORT (fewer rows than its target), cap it and REDISTRIBUTE
    the deficit across the sources that still have capacity (water-filling), then
    REPORT realized vs target fractions.
  * Dedup by chat-content hash, tag every row with ``_source``, and shuffle
    deterministically so ``seed`` fully reproduces the mix.

Also provides ``build_midtrain_corpus`` for the Stage-0 continued-pretraining
corpus: mixing ~10-15% general shards into the ROCm/HIP/Triton corpus.

Everything here is PURE (no GPU / teacher / heavy imports) and unit-testable.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any, Iterable, Optional

from kore.env.replay import kernel_hash

# The canonical mixture source keys mapped to their ``MultiCapSFTConfig`` field.
SOURCE_FRACTION_FIELDS: dict[str, str] = {
    "kernel_repair_opt": "frac_kernel_repair_opt",
    "kernel_qa": "frac_kernel_qa",
    "agentic_tooluse": "frac_agentic_tooluse",
    "general_code": "frac_general_code",
    "math_reasoning": "frac_math_reasoning",
    "general_chat": "frac_general_chat",
}

# Sources that make up the ~45% general-retention half (for reporting/wiring).
GENERAL_REPLAY_SOURCES = ("general_code", "math_reasoning", "general_chat")

SOURCE_TAG_KEY = "_source"

# Midtrain corpus source tags.
KERNEL_CORPUS_TAG = "kernel_corpus"
GENERAL_SHARD_TAG = "general_shard"


# --------------------------------------------------------------------------- #
# Target fractions
# --------------------------------------------------------------------------- #
def target_fractions(config: Any) -> dict[str, float]:
    """Extract the target fraction per source key from a MultiCapSFTConfig."""
    out: dict[str, float] = {}
    for key, field in SOURCE_FRACTION_FIELDS.items():
        out[key] = float(getattr(config, field))
    return out


def normalized_fractions(fracs: dict[str, float]) -> dict[str, float]:
    """Renormalize a fraction dict to sum to 1 (uniform if all zero)."""
    s = sum(v for v in fracs.values() if v > 0)
    if s <= 0:
        n = len(fracs) or 1
        return {k: 1.0 / n for k in fracs}
    return {k: (v / s if v > 0 else 0.0) for k, v in fracs.items()}


# --------------------------------------------------------------------------- #
# Row tagging + content-hash dedup (local hash over chat messages)
# --------------------------------------------------------------------------- #
def _tag(row: Any, key: str) -> dict:
    """Return a shallow copy of ``row`` tagged with its source (never mutates)."""
    r = dict(row) if isinstance(row, dict) else {"value": row}
    r[SOURCE_TAG_KEY] = key
    return r


def _row_hash(row: Any) -> str:
    """Content hash of a chat row, ignoring the source tag / metadata keys."""
    if isinstance(row, dict) and "messages" in row:
        payload: Any = row["messages"]
    elif isinstance(row, dict):
        payload = {k: v for k, v in row.items() if not k.startswith("_")}
    else:
        payload = row
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return kernel_hash(blob)


def dedup_rows(rows: Iterable[Any]) -> list:
    """Keep the first row per distinct chat-content hash (source-tag agnostic)."""
    seen: set[str] = set()
    out: list = []
    for row in rows:
        h = _row_hash(row)
        if h in seen:
            continue
        seen.add(h)
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Capacity-aware allocation (water-filling with redistribution)
# --------------------------------------------------------------------------- #
def allocate_counts(
    available: dict[str, int],
    targets: dict[str, float],
    total: int,
) -> dict[str, int]:
    """Allocate per-source integer sample counts.

    Distributes ``min(total, sum(available))`` rows across sources in proportion
    to ``targets``, but never asks a source for more than it has: a short source
    is capped at ``available`` and its deficit is redistributed to sources that
    still have capacity. Deterministic; returns ``{key: count}`` for every key in
    ``available`` (0 if that source has no target weight).
    """
    keys = [k for k in available if k in targets]
    result = {k: 0 for k in available}
    if not keys or total <= 0:
        return result

    frac = normalized_fractions({k: targets[k] for k in keys})
    capacity = sum(available[k] for k in keys)
    goal = min(int(total), capacity)
    if goal <= 0:
        return result

    # Water-filling: real-valued shares, capping over-allocated sources.
    active = set(keys)
    fixed: dict[str, float] = {}
    while active:
        wsum = sum(frac[k] for k in active)
        if wsum <= 0:
            break
        remaining = goal - sum(fixed.values())
        capped = [k for k in active if remaining * frac[k] / wsum > available[k]]
        if not capped:
            for k in active:
                fixed[k] = remaining * frac[k] / wsum
            active = set()
            break
        for k in capped:
            fixed[k] = float(available[k])
            active.discard(k)

    # Largest-remainder rounding to hit ``goal`` exactly, respecting capacity.
    floor_counts = {k: int(fixed.get(k, 0.0)) for k in keys}
    deficit = goal - sum(floor_counts.values())
    order = sorted(keys, key=lambda k: fixed.get(k, 0.0) - floor_counts[k], reverse=True)
    while deficit > 0:
        progressed = False
        for k in order:
            if deficit == 0:
                break
            if floor_counts[k] < available[k]:
                floor_counts[k] += 1
                deficit -= 1
                progressed = True
        if not progressed:
            break

    result.update(floor_counts)
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def mixture_report(rows: Iterable[Any]) -> dict:
    """Report realized per-source counts + fractions over ``rows`` (by tag)."""
    rows = list(rows)
    counts = Counter(
        (r.get(SOURCE_TAG_KEY, "unknown") if isinstance(r, dict) else "unknown")
        for r in rows
    )
    total = len(rows)
    fractions = {k: (c / total if total else 0.0) for k, c in counts.items()}
    return {"total": total, "counts": dict(counts), "fractions": fractions}


def _format_realized_report(
    counts: dict[str, int], targets: dict[str, float], total_rows: int
) -> str:
    tnorm = normalized_fractions({k: targets[k] for k in counts if k in targets})
    lines = ["[mixing] realized vs target fractions:"]
    for k in sorted(counts):
        realized = counts[k] / total_rows if total_rows else 0.0
        tgt = tnorm.get(k, 0.0)
        lines.append(
            f"  {k:<20} n={counts[k]:<7} realized={realized:6.3f} "
            f"target={tgt:6.3f} (delta={realized - tgt:+.3f})"
        )
    lines.append(f"  {'TOTAL':<20} n={total_rows}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Stage-1 multi-capability SFT mix
# --------------------------------------------------------------------------- #
def build_multicap_sft(
    sources: dict[str, list],
    config: Any,
    total: int,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict]:
    """Assemble the Stage-1 multi-capability SFT mixture.

    ``sources`` maps source keys (subset of ``SOURCE_FRACTION_FIELDS``) to lists
    of chat rows. Each source is deduped, then sampled at its configured fraction
    of ``total`` (with short-source redistribution), tagged with ``_source``,
    concatenated, deduped again, and shuffled deterministically by ``seed``.

    Returns the mixed chat-row list. Prints a realized-vs-target report when
    ``verbose`` (use :func:`mixture_report` for the programmatic version).
    """
    targets = target_fractions(config)

    # Dedup + tag each source pool up front so counts reflect usable rows.
    pools: dict[str, list] = {}
    for key, rows in sources.items():
        if key not in targets:
            continue  # ignore keys with no configured fraction
        tagged = [_tag(r, key) for r in rows]
        pools[key] = dedup_rows(tagged)

    available = {k: len(v) for k, v in pools.items()}
    counts = allocate_counts(available, {k: targets[k] for k in available}, total)

    mixed: list[dict] = []
    for key in sorted(pools):
        n_k = counts.get(key, 0)
        if n_k <= 0:
            continue
        pool = pools[key]
        rng = random.Random(f"{seed}:{key}")
        idx = rng.sample(range(len(pool)), min(n_k, len(pool)))
        mixed.extend(pool[i] for i in idx)

    mixed = dedup_rows(mixed)
    random.Random(seed).shuffle(mixed)

    if verbose:
        realized = mixture_report(mixed)
        print(_format_realized_report(realized["counts"], targets, realized["total"]))

    return mixed


# --------------------------------------------------------------------------- #
# Stage-0 midtrain corpus (kernel corpus + a little general replay)
# --------------------------------------------------------------------------- #
def _norm_doc(doc: Any, tag: str) -> dict:
    """Normalize a corpus doc to a tagged dict (``{"text": ...}`` for strings)."""
    if isinstance(doc, dict):
        d = dict(doc)
    else:
        d = {"text": str(doc)}
    d[SOURCE_TAG_KEY] = tag
    return d


def build_midtrain_corpus(
    kernel_docs: list,
    general_shards: list,
    config: Any,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict]:
    """Mix ~10-15% general shards into the ROCm/Triton corpus for Stage-0.

    Keeps all ``kernel_docs`` and adds enough ``general_shards`` so that general
    data is ``config.general_replay_frac`` of the combined corpus (capped at the
    number of shards available). Tags each doc with ``_source`` and shuffles
    deterministically by ``seed``. Returns the mixed doc list.
    """
    frac = float(getattr(config, "general_replay_frac", 0.15))
    kernel = [_norm_doc(d, KERNEL_CORPUS_TAG) for d in kernel_docs]
    kernel = dedup_rows(kernel)
    shards = [_norm_doc(d, GENERAL_SHARD_TAG) for d in general_shards]
    shards = dedup_rows(shards)

    n_kernel = len(kernel)
    if frac <= 0.0 or n_kernel == 0:
        n_general = 0
    elif frac >= 1.0:
        n_general = len(shards)
    else:
        # g / (k + g) = frac  =>  g = frac * k / (1 - frac)
        n_general = int(round(frac * n_kernel / (1.0 - frac)))
    n_general = min(n_general, len(shards))

    rng = random.Random(f"{seed}:midtrain_general")
    idx = rng.sample(range(len(shards)), n_general) if n_general > 0 else []
    chosen_general = [shards[i] for i in idx]

    mixed = kernel + chosen_general
    random.Random(seed).shuffle(mixed)

    if verbose:
        total = len(mixed)
        realized = len(chosen_general) / total if total else 0.0
        print(
            f"[midtrain] kernel={n_kernel} general={len(chosen_general)} "
            f"total={total} realized_general_frac={realized:.3f} "
            f"target={frac:.3f}"
        )
    return mixed
