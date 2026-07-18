"""Competitor-anchored Opus baseline: derive a per-task ``regret_vs_opus`` map
from KORE's OWN existing verified data (no new Opus API calls).

KORE's data-generation TEACHER was ``claude-opus-4.8``, so the verified corpus in
``<data_root>/wins/*.jsonl`` and ``<data_root>/groups/*.jsonl`` already contains
Opus-authored kernels with MEASURED, hardware-verified speedups per task
(speedup = production-baseline wall / kernel wall, measured on MI350X). This module
mines that corpus to build the ``opus_scores`` map the task proposer consumes
(:func:`kore.openended.proposer.score_descriptor`), turning the open-ended
curriculum into a *regret-vs-Opus* curriculum: compute concentrates where catching
and overtaking Opus 4.8 is both valuable and learnable.

On-disk schema used (see :mod:`kore.data.schemas`)
--------------------------------------------------
* ``wins/*.jsonl`` -> :class:`~kore.data.schemas.WinRecord` dicts:
  ``{"task_id": str, "speedup": float, ...}``. A win is a full VERIFIED winning
  trajectory (correct + measured wall improvement), so its ``speedup`` is a
  genuine verified Opus result.
* ``groups/*.jsonl`` -> :class:`~kore.data.schemas.RankedGroupRecord` dicts:
  ``{"task_id": str, "candidates": [{"speedup": float, "correct": bool, ...}]}``.
  A group candidate counts as a verified Opus result only when ``correct is True``
  (``require_correct``), so slow/incorrect exploratory candidates never inflate the
  baseline.

For each ``task_id`` we take Opus's BEST (max) verified speedup.

Mapping: Opus speedup -> ``regret_vs_opus`` in ``[0, 1]``
--------------------------------------------------------
We interpret the signal as *competitor strength / headroom to catch up*: the
higher Opus's verified speedup on a task, the higher the bar Opus set, so the more
valuable it is to prioritize that task and close the gap (the task hint: "1.0 for
tasks where Opus is strong so we prioritize catching up"). KORE's own live speedup
is NOT in this static corpus, but the proposer already MULTIPLIES this term by
learnability ``4p(1-p)`` (live competence), so the static Opus-strength prior and
the live learnability signal together concentrate compute on tasks that are both
learnable AND where a strong Opus result is still to be matched.

Concretely, with ``s`` = Opus's best verified speedup for a task:

    regret_vs_opus = clamp( log(max(s, 1)) / log(S_ref),  0, 1 )

* **log**, because speedup is a multiplicative ratio (a 2x -> 4x step should count
  the same as 4x -> 8x); a linear map would let one 25x outlier crush everything.
* ``s`` floored at ``1.0``: an Opus kernel that only matched/'lost to' the baseline
  (``s <= 1``) implies no competitive headroom -> regret ``0``.
* ``S_ref`` = ``max(min_ref_speedup, percentile(best_speedups, ref_percentile))`` is
  the "maximally strong Opus" reference that maps to regret ``1.0``. Using a high
  percentile (default p95) makes the strongest tasks saturate at ``1.0`` and keeps
  the mapping self-calibrating across datasets; ``min_ref_speedup`` (default 2.0) is
  a floor so a degenerate corpus (all ~1x) can't blow up the normalization.

Fail-safe by construction
-------------------------
Every failure mode degrades to the feature being INERT (returns ``{}``): a missing
/ non-directory ``data_root``, unreadable or malformed files, malformed lines
(skipped), records missing ``task_id`` / ``speedup``, or a corpus with no verified
speedups. The proposer then keeps its plain learnability+regret+novelty score, so
enabling this feature can never *degrade* the curriculum, only refine it. Values
are further clamped/sanitized by the proposer, so a noisy map is also safe.

Caching
-------
``build_opus_scores(data_root, cache_path=...)`` writes the computed map to a JSON
file (atomically) and, on subsequent calls, loads it back instead of re-scanning
(the orchestrator's ``coevolve_opus_scores_path``). Caching is best-effort: a cache
read/write error simply falls back to a fresh scan.

Pure/stdlib-only (``json`` + ``math`` + ``pathlib``): no torch, CPU-unit-testable.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

_LOG = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Defaults for the speedup -> regret mapping (documented above).
DEFAULT_REF_PERCENTILE = 95.0
DEFAULT_MIN_REF_SPEEDUP = 2.0


# --------------------------------------------------------------------------- #
# numeric helpers (all fail-safe: never raise)
# --------------------------------------------------------------------------- #
def _finite_positive(x) -> Optional[float]:
    """Coerce ``x`` to a finite float ``> 0``, else ``None`` (drop it)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0.0:
        return None
    return v


def _finite01(x) -> Optional[float]:
    """Coerce ``x`` to a finite float clamped to ``[0, 1]``, else ``None``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return max(0.0, min(1.0, v))


def _percentile(sorted_vals: list, q: float) -> float:
    """Linear-interpolation percentile of an ASCENDING-sorted list (q in [0, 100]).

    Robust for tiny inputs: ``[]`` -> ``0.0``; a single value -> that value."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    q = max(0.0, min(100.0, float(q)))
    pos = (q / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


# --------------------------------------------------------------------------- #
# JSONL scanning (mirrors kore.data.schemas.read_jsonl robustness, inline so this
# module stays dependency-free within kore.openended)
# --------------------------------------------------------------------------- #
def _iter_jsonl_records(path: Path) -> Iterator[dict]:
    """Yield dict records from a JSONL file, skipping malformed lines. Never raises."""
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(d, dict):
                    yield d
    except OSError:
        return


def _passes_snr(snr, min_snr_db: Optional[float]) -> bool:
    """SNR gate: pass when no threshold is set OR the record has no snr, else
    require ``snr_db >= min_snr_db`` (a present-but-below-bar record is dropped)."""
    if min_snr_db is None:
        return True
    if snr is None:
        return True
    try:
        return float(snr) >= float(min_snr_db)
    except (TypeError, ValueError):
        return True


def _scan_best_speedups(root: Path, *, require_correct: bool, include_wins: bool,
                        include_groups: bool, min_snr_db: Optional[float]) -> Dict[str, float]:
    """Scan ``wins/`` + ``groups/`` under ``root`` for Opus's BEST verified speedup
    per ``task_id``. Returns ``{task_id: best_speedup}`` (may be empty). Never raises."""
    best: Dict[str, float] = {}

    def _bump(task_id, speedup) -> None:
        if not isinstance(task_id, str) or not task_id:
            return
        v = _finite_positive(speedup)
        if v is None:
            return
        if v > best.get(task_id, 0.0):
            best[task_id] = v

    if include_wins:
        # A win record is a verified winning trajectory: its speedup is verified.
        for p in sorted((root / "wins").glob("*.jsonl")):
            for d in _iter_jsonl_records(p):
                if _passes_snr(d.get("snr_db"), min_snr_db):
                    _bump(d.get("task_id"), d.get("speedup"))

    if include_groups:
        for p in sorted((root / "groups").glob("*.jsonl")):
            for d in _iter_jsonl_records(p):
                cands = d.get("candidates")
                if not isinstance(cands, list):
                    continue
                tid = d.get("task_id")
                for c in cands:
                    if not isinstance(c, dict):
                        continue
                    if require_correct and c.get("correct") is not True:
                        continue
                    if not _passes_snr(c.get("snr_db"), min_snr_db):
                        continue
                    _bump(tid, c.get("speedup"))

    return best


def _speedups_to_regret(best: Dict[str, float], *, ref_percentile: float,
                        min_ref_speedup: float) -> Dict[str, float]:
    """Map ``{task_id: best_speedup}`` -> ``{task_id: regret_vs_opus in [0, 1]}``.

    See the module docstring for the rationale (log-speedup normalized to a
    percentile reference). Returns ``{}`` for an empty input."""
    if not best:
        return {}
    # Floor each speedup at 1.0 (Opus at/below baseline => no headroom).
    floored = {t: max(1.0, float(s)) for t, s in best.items()}
    ref = _percentile(sorted(floored.values()), ref_percentile)
    # S_ref maps to regret 1.0; floor keeps log(S_ref) strictly positive so a
    # degenerate (all ~1x) corpus yields all-zeros instead of dividing by ~0.
    s_ref = max(float(min_ref_speedup), ref, 1.0 + 1e-9)
    denom = math.log(s_ref)
    if denom <= 0.0:  # unreachable given the floor, but stay safe
        return {}
    out: Dict[str, float] = {}
    for t, s in floored.items():
        out[t] = max(0.0, min(1.0, math.log(s) / denom))
    return out


# --------------------------------------------------------------------------- #
# JSON cache IO (best-effort, atomic write)
# --------------------------------------------------------------------------- #
def load_opus_scores(path: PathLike) -> Dict[str, float]:
    """Load + sanitize an opus-scores JSON map from ``path``. Fail-safe -> ``{}``.

    Only ``str`` keys with finite values clamped to ``[0, 1]`` survive, so a
    hand-edited or partially-corrupt cache can never inject a bad signal."""
    try:
        p = Path(path)
        if not p.is_file():
            return {}
        with p.open(encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        fv = _finite01(v)
        if fv is not None:
            out[k] = fv
    return out


def save_opus_scores(scores: Dict[str, float], path: PathLike) -> Optional[Path]:
    """Atomically write ``scores`` to ``path`` as JSON. Fail-safe -> ``None`` on error."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {str(k): float(v) for k, v in scores.items()}
        tmp = p.with_name(p.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        tmp.replace(p)
        return p
    except (OSError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_opus_scores(
    data_root: Optional[PathLike],
    *,
    cache_path: Optional[PathLike] = None,
    require_correct: bool = True,
    ref_percentile: float = DEFAULT_REF_PERCENTILE,
    min_ref_speedup: float = DEFAULT_MIN_REF_SPEEDUP,
    include_wins: bool = True,
    include_groups: bool = True,
    min_snr_db: Optional[float] = None,
) -> Dict[str, float]:
    """Build the per-task ``{task_id: regret_vs_opus}`` map from EXISTING data.

    Scans ``<data_root>/wins/*.jsonl`` + ``<data_root>/groups/*.jsonl`` for Opus's
    best verified speedup per task and maps it to a ``[0, 1]`` competitor-anchored
    regret signal (see the module docstring for the mapping rationale).

    Parameters
    ----------
    data_root:
        Directory holding ``wins/`` and ``groups/`` subdirs (e.g. ``data/full14b``).
        Missing / non-directory / ``None`` -> ``{}`` (feature inert).
    cache_path:
        Optional JSON cache (the orchestrator's ``coevolve_opus_scores_path``). If it
        already exists and holds a non-empty map, it is loaded and returned WITHOUT
        rescanning; otherwise the freshly computed map is written to it (best-effort).
    require_correct:
        Count a group candidate as verified only when ``correct is True`` (default).
        Win records are always treated as verified. Set ``False`` to also consider
        group candidates lacking a ``correct`` flag.
    ref_percentile, min_ref_speedup:
        Control ``S_ref`` (the speedup mapping to regret ``1.0``): the higher of
        ``min_ref_speedup`` and the ``ref_percentile``-th percentile of best speedups.
    include_wins, include_groups:
        Toggle each data source (both on by default).
    min_snr_db:
        Optional extra SNR gate; when set, records/candidates whose ``snr_db`` is
        present and below this threshold are dropped (default ``None`` = off).

    Returns
    -------
    dict[str, float]
        ``{task_id: regret_vs_opus}`` with values in ``[0, 1]`` (empty if inert).
        Feed directly as ``CoevolutionController(opus_scores=...)`` /
        ``proposer.propose(..., opus_scores=...)``.
    """
    # 1. Cache fast-path: a present, non-empty cache is authoritative.
    if cache_path is not None:
        cached = load_opus_scores(cache_path)
        if cached:
            return cached

    # 2. Resolve + validate the data root (fail-safe -> inert).
    if not data_root:
        return {}
    try:
        root = Path(data_root)
        if not root.is_dir():
            return {}
    except (OSError, TypeError, ValueError):
        return {}

    # 3. Scan + map (wrapped so no data pathology can raise).
    try:
        best = _scan_best_speedups(
            root, require_correct=require_correct, include_wins=include_wins,
            include_groups=include_groups, min_snr_db=min_snr_db)
        scores = _speedups_to_regret(
            best, ref_percentile=ref_percentile, min_ref_speedup=min_ref_speedup)
    except Exception:  # noqa: BLE001 - build must never break the training loop
        _LOG.warning("build_opus_scores: scan failed for %s; feature inert", data_root,
                     exc_info=True)
        return {}

    # 4. Populate the cache (best-effort; only when we actually derived scores).
    if cache_path is not None and scores:
        save_opus_scores(scores, cache_path)
    return scores


def summarize_opus_scores(scores: Dict[str, float]) -> dict:
    """Compact, JSON-friendly summary of an opus-scores map (for logging)."""
    vals = [v for v in (scores or {}).values() if isinstance(v, (int, float))]
    if not vals:
        return {"tasks": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "top": []}
    top = sorted((scores or {}).items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "tasks": len(vals),
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "top": [[t, round(float(s), 4)] for t, s in top],
    }
