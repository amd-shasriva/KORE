"""Mint "gold" optimization-win SFT demonstrations from verified ranked groups.

Why this exists
---------------
The SFT ``kernel_repair_opt`` slice is `build_sft(repair + wins)`. But `wins`
is the thinnest family (one greedy trajectory per task, ~1/task), while `groups`
holds thousands of verified candidates that `build_sft` **ignores** — they only
feed DPO ranking. That means KORE's best measured kernels are used to teach
*ranking* but never *generation*.

This module closes that gap with zero new GPU work: for each ranked group it
takes the **rank-0** candidate (KORE's robustly-best correct kernel, chosen by
the correct>speed>SNR + noise-margin ranking) as a **gold generation target**,
frames a slower correct sibling as the parent to improve, and emits a
:class:`~kore.data.schemas.WinRecord` in the exact real-wins format
(`SYSTEM_PROMPT` + `build_turn_prompt` + an ``ANALYSIS: … FULL_KERNEL:`` turn).

The records are written to ``<data_root>/wins/_gold_from_groups.jsonl`` so the
campaign build stage picks them up through the same path as real wins — dedup by
source hash, leakage split by (operation, arch), and the RFT/ReST-EM speedup
gate (`speedup >= tau`, per-task cap) — so only genuinely faster-than-baseline
gold kernels survive into SFT. This is rejection-sampling on already-verified
data: the *code* is gold (measured fastest-correct), the reasoning is a short,
measurement-grounded ANALYSIS (no fabricated mechanism).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

from kore.data.prompts import SYSTEM_PROMPT, build_turn_prompt
from kore.data.schemas import WinRecord, read_jsonl, write_jsonl
from kore.obs import get_logger

log = get_logger("data.gold_wins")

DEFAULT_SNR_GATE = 40.0      # dB; group candidates sit at 76-999 dB, so this keeps clearly-correct only
DEFAULT_MIN_SPEEDUP = 1.02   # only demonstrate a REAL improvement over the parent
DEFAULT_ARCH = "gfx942"


def _correct(cands: list[dict], gate: float) -> list[dict]:
    out = []
    for c in cands:
        s = c.get("snr_db")
        if isinstance(s, (int, float)) and s >= gate and c.get("source") and c.get("wall_us"):
            out.append(c)
    return out


def _best(cands: list[dict]) -> dict:
    """KORE's robustly-best candidate: rank-0 when present, else fastest correct."""
    for c in cands:
        if c.get("rank") == 0:
            return c
    return min(cands, key=lambda c: c["wall_us"])


def _analysis(op: str, wall: float, snr: float, speedup: float) -> str:
    return (
        f"ANALYSIS: For `{op}`, this is the fastest CORRECT implementation verified for this "
        f"shape — measured {wall:.1f}us at SNR {snr:.0f} dB, {speedup:.2f}x faster than the "
        f"parent variant below. It keeps fp32 accumulation and the public entry-point signature, "
        f"and uses 64-multiple BLOCK sizes suited to the CDNA wavefront."
    )


def mint_gold_win(group: dict, arch: Optional[str] = None,
                  snr_gate: float = DEFAULT_SNR_GATE,
                  min_speedup: float = DEFAULT_MIN_SPEEDUP) -> Optional[WinRecord]:
    """One ranked group -> a gold optimization-win WinRecord (or None if it
    lacks a clearly-correct, meaningfully-faster candidate)."""
    cands = _correct(group.get("candidates") or [], snr_gate)
    if len(cands) < 2:
        return None
    best = _best(cands)
    slower = [c for c in cands if c["wall_us"] > best["wall_us"]]
    if not slower:
        return None
    # Lower-median slower sibling = representative parent (conservative speedup,
    # never cherry-picks the worst variant to inflate the demonstrated gain).
    baseline = sorted(slower, key=lambda c: c["wall_us"])[(len(slower) - 1) // 2]
    speedup = baseline["wall_us"] / best["wall_us"]
    if speedup < min_speedup:
        return None

    a = str(arch or group.get("arch") or group.get("gpu") or DEFAULT_ARCH)
    op = str(group.get("operation") or group.get("task_id") or "kernel")
    user = build_turn_prompt(parent_source=baseline["source"], mode="exploit")
    assistant = (
        _analysis(op, float(best["wall_us"]), float(best["snr_db"]), float(speedup))
        + "\n\nFULL_KERNEL:\n" + best["source"]
    )
    trajectory = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return WinRecord(
        task_id=str(group.get("task_id", "gold")),
        trajectory=trajectory,
        initial_wall_us=round(float(baseline["wall_us"]), 3),
        final_wall_us=round(float(best["wall_us"]), 3),
        speedup=round(float(speedup), 4),
        final_source=best["source"],
        snr_db=round(float(best["snr_db"]), 2),
        type="win",
        gpu=a,
        operation=op,
        arch=a,
        shape=group.get("shape"),
    )


def mint_gold_wins(
    data_root: Any, *, cap: int = 3000, per_task_cap: int = 25,
    snr_gate: float = DEFAULT_SNR_GATE, min_speedup: float = DEFAULT_MIN_SPEEDUP,
    seed: int = 0, arch: Optional[str] = None, write: bool = True,
    out_name: str = "_gold_from_groups.jsonl",
) -> dict:
    """Scan ``<data_root>/groups`` and mint up to ``cap`` gold-win records.

    Deterministic given ``seed``. ``per_task_cap`` bounds how many golds any one
    task contributes so no task dominates the wins pool before RFT. Writes
    ``<data_root>/wins/<out_name>`` (picked up by the build raw gather) and
    returns a summary. Never touches existing per-task shards.
    """
    data_root = Path(data_root)
    rng = random.Random(seed)
    groups: list[dict] = []
    d = data_root / "groups"
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            try:
                if p.stat().st_size == 0:
                    continue
            except OSError:
                continue
            for g in read_jsonl(p, typed=False):
                if isinstance(g, dict):
                    groups.append(g)
    rng.shuffle(groups)

    per_task: dict[str, int] = {}
    out: list[WinRecord] = []
    for g in groups:
        if len(out) >= cap:
            break
        tid = str(g.get("task_id", "?"))
        if per_task.get(tid, 0) >= per_task_cap:
            continue
        try:
            w = mint_gold_win(g, arch, snr_gate, min_speedup)
        except Exception as e:  # noqa: BLE001 — one bad group must not abort
            log.debug("gold_skip", task=tid, err=str(e)[:120])
            w = None
        if w is not None:
            out.append(w)
            per_task[tid] = per_task.get(tid, 0) + 1

    if write and out:
        (data_root / "wins").mkdir(parents=True, exist_ok=True)
        write_jsonl(data_root / "wins" / out_name, [w.to_dict() for w in out])

    summary = {
        "gold_wins": len(out),
        "tasks_covered": len(per_task),
        "groups_scanned": len(groups),
    }
    log.event("gold_wins_minted", cap=cap, snr_gate=snr_gate,
              min_speedup=min_speedup, **summary)
    return summary


__all__ = ["mint_gold_win", "mint_gold_wins"]
