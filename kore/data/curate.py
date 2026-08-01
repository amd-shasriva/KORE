"""Curation & balancing of the assembled SFT mixture (Pillar 6).

Turns a raw, deduped, contract-unified pile of rows into a BALANCED, quality-
ranked, curriculum-ordered dataset - the difference between "a lot of verified
data" and "the best training mixture in the world". Operates on final chat rows
that carry the Pillar-5 ``_provenance`` block (kernel rows) and ``_source`` tag.

Levers (all deterministic, PURE stdlib):
  * :func:`quality_score` - a scalar from provenance (measured speedup, SNR,
    verified, kind). Retention rows get a neutral score (kept, not ranked out).
  * :func:`filter_trivial_wins` - drop win demos whose measured speedup is below a
    floor (the shipped wins were 50% in 1.0-1.1x - barely-better demos dilute the
    signal). Repairs are never dropped here (correctness lessons).
  * :func:`filter_implausible_wins` - the HIGH end: a win whose measured speedup
    exceeds the credible ceiling (``schemas.credible_speedup_max``) is excluded from
    the exemplar mixture. Such rows are real as measured but are usually timing a
    non-kernel baseline (the sequence/SSM tasks are benched against a Python
    ``for t in range(...)`` interpreter loop, so four-digit ratios are genuine and
    say nothing about kernel quality), and :func:`quality_score` used to clamp at
    log(10) so a 9,381x row scored exactly like a 10x one and survived curation.
  * :func:`win_speedup_stats` - the only honest way to aggregate win speedups: pool
    ONLY rows that are explicitly baseline-relative AND credible, and report what
    was excluded rather than averaging incomparable scales together.
  * :func:`balance_by_family` - cap how many rows any one operator family / dtype
    contributes so gemm (many tasks) can't drown rmsnorm/quant.
  * :func:`difficulty_score` + :func:`curriculum_order` - order easy->hard for a
    curriculum (Kevin/AlphaCode-style), by kernel length + inverse speedup margin.
  * :func:`curate` - the orchestrator used by the build stage.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Optional

# ``_family_of`` is a pure string classifier (no registry/GPU); reuse it.
from kore.data.decontam import _family_of
from kore.data.schemas import (
    baseline_relative_speedup,
    credible_speedup_max,
    is_credible_win,
)

_KERNEL_SOURCES = {"kernel_repair_opt", "kernel_qa"}


def _prov(row: dict) -> dict:
    p = row.get("_provenance")
    return p if isinstance(p, dict) else {}


def is_kernel_row(row: dict) -> bool:
    return bool(_prov(row)) or row.get("_source") in _KERNEL_SOURCES


def row_family(row: dict) -> str:
    p = _prov(row)
    return _family_of(str(p.get("operation") or p.get("task_id") or row.get("_source") or ""))


def _row_len(row: dict) -> int:
    return sum(len(m.get("content", "")) for m in row.get("messages", [])
              if isinstance(m, dict))


def is_implausible_win(row: dict, *, threshold: Optional[float] = None) -> bool:
    """True for a WIN row whose measured speedup exceeds the credible ceiling.

    Uses the flag the datagen path persisted when present, else re-derives it from
    the speedup so the already-shipped (unflagged) wins are graded too.
    """
    p = _prov(row)
    if p.get("kind") != "win":
        return False
    return not is_credible_win(p, threshold=threshold)


def _speedup_credit(p: dict, *, threshold: Optional[float] = None) -> float:
    """Log-speedup credit, awarded only for a speedup inside the credible ceiling.

    The previous ``log(min(sp, 10.0))`` clamp gave a 9,381x row exactly the same
    credit as a 10x row, so implausible rows tied the best real ones and survived
    every top-k selection. A ratio above the ceiling now earns NO speed credit: it
    is not evidence about the kernel (see :func:`filter_implausible_wins`).
    """
    sp = p.get("speedup")
    if isinstance(sp, bool) or not isinstance(sp, (int, float)) or sp <= 0:
        return 0.0
    ceiling = credible_speedup_max(threshold=threshold)
    if p.get("speedup_exceeds_credible") is True or float(sp) > ceiling:
        return 0.0
    # log-speedup: 1x -> 0, 2x -> ~0.69, 4x -> ~1.39 (diminishing, outlier-safe)
    import math
    return math.log(min(float(sp), ceiling))


def quality_score(row: dict) -> float:
    """Higher = keep. Kernel rows scored by measured speedup + SNR + verified.

    Retention (general_*) rows get a fixed neutral score so they are never ranked
    below kernel rows nor dropped by a quality floor.
    """
    p = _prov(row)
    if not p:
        return 1.0  # neutral retention row
    score = 0.0
    if p.get("verified"):
        score += 1.0
    score += _speedup_credit(p)
    snr = p.get("snr_db")
    if isinstance(snr, (int, float)):
        score += min(max(float(snr), 0.0), 100.0) / 200.0  # 0..0.5
    if p.get("kind") == "repair":
        score += 0.5  # correctness/repair lessons are valuable regardless of speed
    return score


def difficulty_score(row: dict) -> float:
    """0 (easy) .. 1 (hard). Longer kernels + smaller speedup margin = harder."""
    length = _row_len(row)
    len_term = min(length / 16000.0, 1.0)  # ~16k chars ~ hard/long
    p = _prov(row)
    sp = p.get("speedup")
    # small achievable margin -> harder; large speedup headroom -> easier
    margin_term = 0.5
    if isinstance(sp, (int, float)) and sp > 0:
        margin_term = 1.0 / (1.0 + max(float(sp) - 1.0, 0.0))  # sp=1 ->1(hard), sp=3 ->0.33
    return round(0.5 * len_term + 0.5 * margin_term, 4)


def filter_trivial_wins(rows: Iterable[dict], min_speedup: float = 1.1) -> tuple[list[dict], dict]:
    """Drop WIN rows whose measured speedup < ``min_speedup`` (keep everything else)."""
    kept, dropped = [], 0
    for r in rows:
        p = _prov(r)
        if p.get("kind") == "win":
            sp = p.get("speedup")
            if isinstance(sp, (int, float)) and sp < min_speedup:
                dropped += 1
                continue
        kept.append(r)
    return kept, {"n_dropped_trivial_wins": dropped, "n_kept": len(kept)}


def filter_implausible_wins(rows: Iterable[dict], *,
                            threshold: Optional[float] = None,
                            ) -> tuple[list[dict], dict]:
    """Exclude WIN rows whose measured speedup exceeds the credible ceiling.

    The counterpart to :func:`filter_trivial_wins` at the HIGH end. Nothing is
    deleted from the corpus - the record keeps its measured value and its explicit
    flag on disk - it is only excluded from the SFT exemplar mixture, because a
    ratio that large is (almost always) timing a baseline that is not a single GPU
    kernel and teaching it as a demonstration of kernel skill is a false claim.
    Only ``kind == "win"`` rows are eligible; repairs and retention rows are never
    touched.
    """
    kept, dropped = [], 0
    for r in rows:
        if is_implausible_win(r, threshold=threshold):
            dropped += 1
            continue
        kept.append(r)
    return kept, {"n_dropped_implausible_wins": dropped, "n_kept": len(kept),
                  "credible_speedup_max": credible_speedup_max(threshold=threshold)}


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile of a NON-EMPTY sorted list (q in [0, 1])."""
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    idx = min(len(sorted_values) - 1,
              max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def win_speedup_stats(rows: Iterable[dict], *,
                      threshold: Optional[float] = None) -> dict:
    """Aggregate win speedups over ONLY the pool where an aggregate means something.

    A row enters the pool exclusively when it declares a baseline-relative
    ``speedup_basis`` AND its ratio is within the credible ceiling. Everything else
    is counted as an explicit exclusion rather than silently averaged in: the
    gold-minted wins carry a sibling/parent-relative ratio and the gen_wins footers
    are relative to their own first measurement, so pooling all three scales in one
    mean is meaningless, and the four-digit outliers would dominate whatever pool
    they landed in.
    """
    pooled: list[float] = []
    n_win = 0
    n_not_baseline_relative = 0
    n_implausible = 0
    for r in rows:
        p = _prov(r)
        if p.get("kind") != "win":
            continue
        n_win += 1
        if not is_credible_win(p, threshold=threshold):
            n_implausible += 1
            continue
        speedup = baseline_relative_speedup(p)
        if speedup is None:
            n_not_baseline_relative += 1
            continue
        pooled.append(speedup)
    pooled.sort()
    stats = {
        "n_wins": n_win,
        "n_pooled": len(pooled),
        "n_excluded_not_baseline_relative": n_not_baseline_relative,
        "n_excluded_implausible": n_implausible,
        "credible_speedup_max": credible_speedup_max(threshold=threshold),
    }
    if pooled:
        stats.update({
            "median_speedup": round(_percentile(pooled, 0.5), 4),
            "p90_speedup": round(_percentile(pooled, 0.9), 4),
            "max_speedup": round(pooled[-1], 4),
        })
    return stats


def balance_by_family(rows: Iterable[dict], cap_per_family: Optional[int] = None,
                      cap_frac: Optional[float] = None,
                      key_fn: Callable[[dict], str] = row_family,
                      scorer: Callable[[dict], float] = quality_score,
                      ) -> tuple[list[dict], dict]:
    """Cap how many KERNEL rows any one family contributes (keep the best).

    Non-kernel (retention) rows are exempt (families are a kernel concept). Cap is
    ``cap_per_family`` if given, else ``round(cap_frac * total_kernel_rows)``.
    Deterministic: within a family, keeps the top-scoring rows, ties by input order.
    """
    rows = list(rows)
    kernel_rows = [r for r in rows if is_kernel_row(r)]
    if cap_per_family is None:
        if cap_frac is None:
            return rows, {"capped": 0}
        cap_per_family = max(1, round(cap_frac * len(kernel_rows)))
    by_fam: dict[str, list[dict]] = {}
    for i, r in enumerate(rows):
        if is_kernel_row(r):
            by_fam.setdefault(key_fn(r), []).append((i, r))  # type: ignore[arg-type]
    keep_idx: set[int] = {i for i, r in enumerate(rows) if not is_kernel_row(r)}
    capped = 0
    for fam, items in by_fam.items():
        ranked = sorted(items, key=lambda ir: (scorer(ir[1]), -ir[0]), reverse=True)
        for i, _r in ranked[:cap_per_family]:
            keep_idx.add(i)
        capped += max(0, len(items) - cap_per_family)
    out = [r for i, r in enumerate(rows) if i in keep_idx]
    return out, {"capped": capped, "n_kept": len(out), "families": len(by_fam)}


def curriculum_order(rows: Iterable[dict], reverse: bool = False) -> list[dict]:
    """Order rows easy->hard (kernel rows by difficulty; retention interleaved).

    Stable + deterministic. ``reverse=True`` gives hard->easy.
    """
    rows = list(rows)
    return sorted(rows, key=lambda r: (difficulty_score(r) if is_kernel_row(r) else 0.5),
                  reverse=reverse)


# --------------------------------------------------------------------------- #
# Headroom-aware rebalance (WS-C3): the audited kernel pool was ~82% low-headroom
# memory-bound / trivial-elementwise work (torch already at the roofline) and only
# ~18% compute-bound (gemm/attention/moe) where MI300X kernel skill actually
# matters. Training on that mix over-teaches trivial pointwise kernels. This caps
# the low-headroom share so the compute-bound demos drive the gradient.
# --------------------------------------------------------------------------- #
_COMPUTE_BOUND_FAMILIES = {"gemm", "attention", "moe"}
# Structured memory-bound ops with real fusion/reduction headroom (worth training on
# more than a bare elementwise op). Everything whose family is a *raw op name* (add,
# mul, abs, exp, row_sum, ...) - i.e. not one of the recognised structured families -
# is treated as trivial (near-roofline single-elementwise/reduction; lowest headroom).
# Names are the versioned-taxonomy PRODUCT families emitted by row_family/_family_of
# (kore.tasks.taxonomy.product_family_for_name): normalization=rmsnorm/layernorm,
# quantization=quant, reduction=softmax, positional=rope, activation=gelu/silu/relu,
# fusion=multi-op pointwise/projection fusion. Bare elementwise ops stay OUT -> trivial.
_MEMORY_BOUND_FAMILIES = {"normalization", "quantization", "reduction",
                          "positional", "activation", "fusion"}


def op_class(row: dict) -> str:
    """'compute_bound' | 'memory_bound' | 'trivial' | 'retention' for a chat row.

    compute_bound = gemm/attention/moe (high MFMA headroom); memory_bound = the
    structured norm/quant/softmax/rope/activation fusions; trivial = bare
    elementwise/reduction ops (``_family_of`` returns their raw op name), which are
    near-roofline in torch and teach the least.
    """
    if not is_kernel_row(row):
        return "retention"
    fam = row_family(row)
    if fam in _COMPUTE_BOUND_FAMILIES:
        return "compute_bound"
    if fam in _MEMORY_BOUND_FAMILIES:
        return "memory_bound"
    return "trivial"


def rebalance_by_headroom(rows: Iterable[dict], *, target_compute_frac: float = 0.5,
                          scorer: Callable[[dict], float] = quality_score,
                          ) -> tuple[list[dict], dict]:
    """Cap low-headroom kernel rows so compute-bound reaches ``target_compute_frac``
    of the KERNEL pool when the pool allows.

    ALL compute-bound + ALL retention rows are kept; the low-headroom (trivial +
    memory-bound) kernel rows are thinned to the top-scoring ``nc*(1-t)/t`` (so
    compute reaches the target). Deterministic (keeps the highest ``quality_score``
    low-headroom rows, ties by original order) and order-preserving. Degrades
    gracefully: no compute-bound rows, or a pool already above target -> unchanged.
    """
    import math
    rows = list(rows)

    def _is_repair(r: dict) -> bool:
        # A verified broken->fixed repair is a correctness lesson, not a speed demo;
        # it must NEVER be thinned to hit a compute-fraction target -- especially for
        # the memory-bound norm/quant/softmax families the model must still get right
        # (audit R2 sft I2: rebalance_by_headroom dropped repairs that filter_trivial_
        # wins had deliberately exempted).
        return (r.get("_provenance") or {}).get("kind") == "repair"

    low_idx = [i for i, r in enumerate(rows)
               if is_kernel_row(r) and op_class(r) != "compute_bound" and not _is_repair(r)]
    compute_idx = [i for i, r in enumerate(rows) if op_class(r) == "compute_bound"]
    nc = len(compute_idx)
    keep = {i for i in range(len(rows)) if i not in set(low_idx)}  # retention + compute
    capped = 0
    t = min(max(target_compute_frac, 1e-6), 1.0)
    if nc > 0 and low_idx:
        max_low = int(math.floor(nc * (1.0 - t) / t))
        if max_low >= len(low_idx):
            keep |= set(low_idx)
        else:
            ranked = sorted(low_idx, key=lambda i: (scorer(rows[i]), -i), reverse=True)
            keep |= set(ranked[:max_low])
            capped = len(low_idx) - max_low
    else:
        keep |= set(low_idx)  # degenerate (no compute-bound): keep everything
    out = [rows[i] for i in range(len(rows)) if i in keep]
    low_kept = len([i for i in keep if i in set(low_idx)])
    frac = nc / (nc + low_kept) if (nc + low_kept) else 0.0
    return out, {"compute_bound": nc, "low_in": len(low_idx), "low_kept": low_kept,
                 "capped": capped, "compute_frac": round(frac, 4)}


def curate(rows: Iterable[dict], *, min_win_speedup: float = 1.1,
           family_cap_frac: Optional[float] = 0.25, quality_floor: float = 0.0,
           curriculum: bool = False, drop_implausible_wins: bool = True,
           credible_speedup_threshold: Optional[float] = None,
           ) -> tuple[list[dict], dict]:
    """Full curation pass. Returns ``(curated_rows, stats)``.

    Order: drop trivial wins -> drop implausible wins -> quality floor ->
    family balance -> (curriculum). Both speedup gates are bounds on the SAME
    quantity: below ``min_win_speedup`` a demo teaches nothing, above the credible
    ceiling it teaches something false. Set ``drop_implausible_wins=False`` to keep
    them (they stay flagged either way).
    """
    rows = list(rows)
    n0 = len(rows)
    rows, s_triv = filter_trivial_wins(rows, min_win_speedup)
    s_impl = {"n_dropped_implausible_wins": 0}
    if drop_implausible_wins:
        rows, s_impl = filter_implausible_wins(
            rows, threshold=credible_speedup_threshold)
    if quality_floor > 0.0:
        rows = [r for r in rows if quality_score(r) >= quality_floor or not is_kernel_row(r)]
    rows, s_bal = balance_by_family(rows, cap_frac=family_cap_frac)
    if curriculum:
        rows = curriculum_order(rows)
    stats = {"n_in": n0, "n_out": len(rows),
             "dropped_trivial_wins": s_triv["n_dropped_trivial_wins"],
             "dropped_implausible_wins": s_impl["n_dropped_implausible_wins"],
             "family_capped": s_bal.get("capped", 0)}
    return rows, stats


__all__ = [
    "quality_score", "difficulty_score", "filter_trivial_wins",
    "filter_implausible_wins", "is_implausible_win", "win_speedup_stats",
    "balance_by_family", "curriculum_order", "curate", "is_kernel_row", "row_family",
    "op_class", "rebalance_by_headroom",
]
