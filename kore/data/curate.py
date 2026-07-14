"""Curation & balancing of the assembled SFT mixture (Pillar 6).

Turns a raw, deduped, contract-unified pile of rows into a BALANCED, quality-
ranked, curriculum-ordered dataset — the difference between "a lot of verified
data" and "the best training mixture in the world". Operates on final chat rows
that carry the Pillar-5 ``_provenance`` block (kernel rows) and ``_source`` tag.

Levers (all deterministic, PURE stdlib):
  * :func:`quality_score` — a scalar from provenance (measured speedup, SNR,
    verified, kind). Retention rows get a neutral score (kept, not ranked out).
  * :func:`filter_trivial_wins` — drop win demos whose measured speedup is below a
    floor (the shipped wins were 50% in 1.0-1.1x — barely-better demos dilute the
    signal). Repairs are never dropped here (correctness lessons).
  * :func:`balance_by_family` — cap how many rows any one operator family / dtype
    contributes so gemm (many tasks) can't drown rmsnorm/quant.
  * :func:`difficulty_score` + :func:`curriculum_order` — order easy->hard for a
    curriculum (Kevin/AlphaCode-style), by kernel length + inverse speedup margin.
  * :func:`curate` — the orchestrator used by the build stage.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Optional

# ``_family_of`` is a pure string classifier (no registry/GPU); reuse it.
from kore.data.decontam import _family_of

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
    sp = p.get("speedup")
    if isinstance(sp, (int, float)) and sp > 0:
        # log-speedup: 1x -> 0, 2x -> ~0.69, 4x -> ~1.39 (diminishing, outlier-safe)
        import math
        score += math.log(min(float(sp), 10.0))
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
# mul, abs, exp, row_sum, ...) — i.e. not one of the recognised structured families —
# is treated as trivial (near-roofline single-elementwise/reduction; lowest headroom).
_MEMORY_BOUND_FAMILIES = {"rmsnorm", "layernorm", "quant", "softmax", "rope",
                          "activation", "moe_router"}


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
    low_idx = [i for i, r in enumerate(rows) if is_kernel_row(r) and op_class(r) != "compute_bound"]
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
           curriculum: bool = False) -> tuple[list[dict], dict]:
    """Full curation pass. Returns ``(curated_rows, stats)``.

    Order: drop trivial wins -> quality floor -> family balance -> (curriculum).
    """
    rows = list(rows)
    n0 = len(rows)
    rows, s_triv = filter_trivial_wins(rows, min_win_speedup)
    if quality_floor > 0.0:
        rows = [r for r in rows if quality_score(r) >= quality_floor or not is_kernel_row(r)]
    rows, s_bal = balance_by_family(rows, cap_frac=family_cap_frac)
    if curriculum:
        rows = curriculum_order(rows)
    stats = {"n_in": n0, "n_out": len(rows),
             "dropped_trivial_wins": s_triv["n_dropped_trivial_wins"],
             "family_capped": s_bal.get("capped", 0)}
    return rows, stats


__all__ = [
    "quality_score", "difficulty_score", "filter_trivial_wins",
    "balance_by_family", "curriculum_order", "curate", "is_kernel_row", "row_family",
    "op_class", "rebalance_by_headroom",
]
