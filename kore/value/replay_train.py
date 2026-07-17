"""Train the KORE value model from the campaign's REAL ranked-group data (P1a).

Audit finding: the value model is doubly dormant in the live run -- the agentic
path never calls the prefilter, and even when it would, ``value_model_path`` is
unset, so the ranker falls back to a hand-coded source heuristic; nothing ever
trains a model on real data. This module fixes the *data* half: it turns the
``groups/*.jsonl`` shards (``RankedGroupRecord``s already produced by datagen) into
schedule-conditioned value-training groups and fits the 3-head + pairwise
:class:`~kore.value.model.ValueModel`, so the measurement-efficiency prefilter AND
the AlphaKernel search heuristic run on a model grounded in this run's own verified
measurements.

Why the ranked groups (not the replay cache): the replay JSONL stores only
``(task_id -> Observation)`` with NO source, so it cannot learn to differentiate
sibling candidates. The ranked groups carry each candidate's SOURCE (hence the
schedule features: block sizes, warps, stages, tl.dot, fp32-accum) AND its measured
wall/speedup AND the group structure -- exactly the within-group ranking signal the
top-k bench selector and the search prior consume.

Pure/CPU (no GPU, no torch): reads JSONL, featurizes source + task meta, fits the
model. Safe to run while datagen is still writing (append-only shards; we snapshot
what exists).
"""

from __future__ import annotations

import glob
import math
import os
from typing import Optional

from kore.obs import get_logger

log = get_logger("value.replay_train")

_DTYPES = ("bf16", "fp16", "fp32", "fp8", "mxfp8", "mxfp4", "fp4", "int8", "int4")


def _dtype_from(task_id: str, rec_dtype: Optional[str]) -> str:
    if rec_dtype:
        return str(rec_dtype)
    s = (task_id or "").lower()
    for d in _DTYPES:
        if d in s:
            return d
    return "bf16"


def _dims_from_shape(shape: Optional[str]) -> dict:
    """Parse ``M=1024,N=512,K=...`` (or ``1024x512x...``) into an {M,N,K} dict."""
    dims: dict = {}
    if not shape:
        return dims
    s = str(shape)
    if "=" in s:
        for part in s.replace(";", ",").split(","):
            if "=" in part:
                k, _, v = part.partition("=")
                k = k.strip().upper()
                try:
                    dims[k] = int(float(v.strip()))
                except (TypeError, ValueError):
                    pass
    else:
        nums = [int(x) for x in _re_ints(s)]
        for name, val in zip(("M", "N", "K"), nums):
            dims[name] = val
    return dims


def _re_ints(s: str):
    import re
    return re.findall(r"\d+", s or "")


def _candidate_speedup(c: dict) -> Optional[float]:
    """Worst-case measured speedup for a candidate (baseline/cand), robust to shard age."""
    su = c.get("speedup")
    if isinstance(su, (int, float)) and su > 0:
        return float(su)
    base = c.get("baseline_wall_us")
    wall = c.get("wall_us")
    if isinstance(base, (int, float)) and isinstance(wall, (int, float)) and base > 0 and wall > 0:
        return float(base) / float(wall)
    return None


def group_rows_from_record(rec) -> list[dict]:
    """One ``RankedGroupRecord`` -> a list of value-table rows (one per candidate).

    Every candidate in a ranked group is a VERIFIED-correct kernel (that is how it
    entered the group), so ``compiled``/``snr_pass`` are True and the differentiating
    outcome is the measured speedup. The row carries the candidate ``source`` so the
    ranker is schedule-conditioned, plus task meta for the pointwise heads.
    """
    d = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
    task_id = d.get("task_id", "")
    op = d.get("operation") or (task_id.split("_")[1] if "_" in task_id else task_id)
    dtype = _dtype_from(task_id, d.get("dtype"))
    dims = _dims_from_shape(d.get("shape"))
    rows: list[dict] = []
    cands = d.get("candidates") or []
    # rank-based fallback speedup when no timing is stored: best rank -> highest.
    n = len(cands)
    for c in cands:
        if not isinstance(c, dict):
            continue
        src = c.get("source") or ""
        if not src:
            continue
        su = _candidate_speedup(c)
        if su is None:
            rank = c.get("rank")
            su = (1.0 + (n - 1 - int(rank)) / max(1, n)) if isinstance(rank, int) else 1.0
        snr = c.get("snr_db")
        rows.append({
            "operation": op, "dtype": dtype, "source": src,
            **{k: v for k, v in dims.items()},
            "pmc_bottleneck": "unknown",
            "compiled": True,
            "snr_pass": True if snr is None else bool(float(snr) >= 0),
            "speedup": float(su),
            "log_speedup": math.log(su) if su > 0 else 0.0,
        })
    return rows


def load_groups_from_dir(groups_dir: str, cap: Optional[int] = None) -> list[list[dict]]:
    """Load ``groups/*.jsonl`` -> list of value-training groups (>=2 candidates each)."""
    from kore.data.schemas import read_jsonl

    groups: list[list[dict]] = []
    paths = sorted(p for p in glob.glob(os.path.join(groups_dir, "*.jsonl"))
                   if not os.path.basename(p).startswith("_"))
    for p in paths:
        try:
            recs = read_jsonl(p, typed=True)
        except Exception:  # noqa: BLE001 - a malformed shard never aborts training
            continue
        for rec in recs:
            if getattr(rec, "type", None) != "ranked_group" and \
               (not isinstance(rec, dict) or rec.get("type") != "ranked_group"):
                continue
            rows = group_rows_from_record(rec)
            if len(rows) >= 2:  # need >=2 to define a within-group order
                groups.append(rows)
                if cap and len(groups) >= cap:
                    return groups
    return groups


def train_value_from_groups(groups_dir: str, out_path: str, *,
                            cap: Optional[int] = None,
                            use_sklearn: Optional[bool] = None) -> dict:
    """Train + save a schedule-conditioned ValueModel from real ranked groups.

    Returns metrics incl. the mean within-group Spearman rank-corr on a held-out
    split (the measurement-efficiency signal). Fails safe: raises only if there is
    literally no usable group data (caller can then skip the prefilter).
    """
    from kore.value.train_value import (
        groupwise_rank_corr,
        train_ranking,
    )
    from kore.value.rerank import score_candidates

    groups = load_groups_from_dir(groups_dir, cap=cap)
    if len(groups) < 2:
        raise ValueError(f"insufficient ranked-group data in {groups_dir} "
                         f"({len(groups)} usable groups)")
    # deterministic held-out split of whole groups (no candidate leaks across split)
    n_test = max(1, len(groups) // 5)
    test_groups = groups[:n_test]
    train_groups = groups[n_test:] or groups

    model = train_ranking(train_groups, use_sklearn=use_sklearn)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save(out_path)

    def _score(metas):
        return score_candidates(metas, model=model)

    rc = groupwise_rank_corr(_score, test_groups)
    metrics = {
        "backend": getattr(model, "backend", "?"),
        "n_groups": len(groups),
        "n_train_groups": len(train_groups),
        "n_test_groups": len(test_groups),
        "n_candidates": sum(len(g) for g in groups),
        "heldout_group_rank_corr": round(float(rc), 4),
        "out_path": out_path,
    }
    log.event("value_trained_from_groups", **metrics)
    return metrics


if __name__ == "__main__":  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-dir", default="data/full14b/groups")
    ap.add_argument("--out", default="runs/value/value_model.pkl")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args()
    print(train_value_from_groups(a.groups_dir, a.out, cap=a.cap))
