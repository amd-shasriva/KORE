"""Stage-2 Rejection-sampling Fine-Tuning (RFT).

RFT = SFT on the policy's OWN samples that the verifier confirms are both
correct AND faster than their parent. Mechanically this is identical to SFT, so
we reuse ``sft.train``; RFT's contribution is the *dataset* (built upstream by
the data module by filtering self-generated trajectories on
correct-and-faster). This module just adapts the filtering + delegates.

Import-guarded like the other trainers.
"""

from __future__ import annotations

import json
from pathlib import Path

from kore.policy.configs import SFTConfig
from kore.policy import sft


def filter_correct_and_faster(
    in_path: str,
    out_path: str,
    min_speedup: float = 1.0,
) -> int:
    """Keep only self-generated samples that are correct AND faster than parent.

    Reads a JSONL of candidate records (expects per-record ``correct`` bool and
    ``speedup`` float, plus a ``messages`` chat list), writes the surviving
    chat records to ``out_path``, and returns the kept count. This is a
    convenience for when the data module hands over raw self-play records; if
    the dataset is already filtered, point ``SFTConfig.dataset_path`` at it and
    call :func:`train` directly.
    """
    kept = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            correct = bool(d.get("correct", False))
            speedup = d.get("speedup")
            if correct and speedup is not None and speedup >= min_speedup and d.get("messages"):
                fout.write(json.dumps({"messages": d["messages"]}) + "\n")
                kept += 1
    return kept


def collect_onpolicy_wins(records, min_speedup: float = 1.0) -> list[dict]:
    """RFT chat-SFT rows from ON-POLICY self-play wins.

    RFT = SFT on the policy's own verified-correct(-and-faster) samples. This
    adapts the records produced by ``kore.data.onpolicy`` so RFT can learn from
    the states the policy visits, not just teacher data. Accepts a mix of:

      * ``RankedGroupRecord`` — on-policy relabel groups; the top-ranked candidate
        of each group is taken as the win (via ``build_datasets.build_rft``).
      * ``WinRecord`` — a full winning trajectory; kept when its ``speedup`` clears
        ``min_speedup`` (or is unknown).
      * raw ``dict`` self-play rows carrying ``{correct, speedup, messages}`` — the
        same shape :func:`filter_correct_and_faster` consumes from disk.

    Returns ``{"messages": [...]}`` rows (identical to the SFT corpus shape), so
    the caller can write them out and point ``SFTConfig.dataset_path`` at them for
    :func:`train`. Pure (no GPU / model)."""
    from kore.data.build_datasets import build_rft
    from kore.data.schemas import RankedGroupRecord, WinRecord

    rows: list[dict] = []
    groups: list = []
    for rec in records:
        if isinstance(rec, RankedGroupRecord):
            groups.append(rec)
        elif isinstance(rec, WinRecord):
            if rec.trajectory and (rec.speedup is None or rec.speedup >= min_speedup):
                rows.append({"messages": list(rec.trajectory)})
        elif isinstance(rec, dict):
            speedup = rec.get("speedup")
            if (rec.get("correct") and rec.get("messages")
                    and speedup is not None and speedup >= min_speedup):
                rows.append({"messages": list(rec["messages"])})
    if groups:
        rows.extend(build_rft(groups))
    return rows


def train(config: SFTConfig) -> dict:
    """RFT is SFT on verified-correct-and-faster self-generated samples.

    Delegates to :func:`sft.train_sft` (the real SFT entrypoint) using
    ``config.dataset_path`` as the already-filtered chat corpus, then tags the
    stage for the caller.
    """
    output_dir = sft.train_sft(config, Path(config.dataset_path))
    return {"stage": "rft", "output_dir": output_dir}
