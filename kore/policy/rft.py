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


def train(config: SFTConfig) -> dict:
    """RFT is SFT on verified-correct-and-faster self-generated samples."""
    result = sft.train(config)
    result["stage"] = "rft"
    return result
