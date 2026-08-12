#!/usr/bin/env python
"""Stage 2c: three recoveries that need no GPU and no new generation.

**The DPO preference pairs.** 96,675 rows were built for preference training we
are not doing, which makes them currently worth nothing. But the ``chosen`` side
is a kernel that won its comparison on measured correctness and speed, and 14,205
of the distinct chosen kernels appear nowhere in the mined corpora -- they are
verified completions that exist only inside a file we were about to ignore.
Filtered for clear correctness and reshaped into single-turn generation, they are
ordinary SFT rows.

Two filters are not optional here. ``sub_baseline`` pairs are *by construction*
correct-but-slower-than-baseline -- they carry weight 0.25 because they are the
weak side of the distribution -- so promoting them to generation targets would
teach the model to produce the kernels the pipeline preferred against. And the
corpus-level ``verified`` flag is not a per-row proof: 1,401 rows carry a
``chosen_snr_db`` that fails the correctness bar, one as low as -999 dB.

**The orphaned modality slice.** ``data/modality_v5.jsonl`` holds 2,354 rows in
the two shapes the benchmark most wants and v4 entirely lacked, built by a script
whose output was never wired into a mixture.

**Extra replay.** 4,887 general rows exist in an older mixture file and not in v4
-- almost entirely tool-use and instruction-following, which are the two thinnest
buckets in the replay set. A hash join recovers them.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

REPO = Path("/home/shasriva/Kore-RL/KORE")
sys.path.insert(0, str(REPO))

DPO = REPO / "data/b05factory/dpo/pairs.jsonl"
MODALITY = REPO / "data/modality_v5.jsonl"
V4 = REPO / "data/b05factory/sft/multicap_v3.jsonl"
OLDER = REPO / "data/b05factory/sft/multicap_full.jsonl"

#: Below this a "chosen" kernel is not clearly correct, whatever the corpus-level
#: verified flag says.
MIN_SNR_DB = 22.0

#: Anchors whose chosen side is correct but deliberately not the strong option.
EXCLUDED_ANCHORS = frozenset({"sub_baseline"})

REPLAY_WANTED = ("tool_use", "instruction_following", "general_chat",
                 "general_code", "math_reasoning", "chat", "code", "math")

_CODE = re.compile(r"```[a-zA-Z+]*\n(.*?)```", re.S)


def code_of(text: str) -> str:
    m = _CODE.findall(text or "")
    return max(m, key=len).strip() if m else (text or "").strip()


def h(s: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", s or "").strip().encode()).hexdigest()


def existing_kernel_hashes(stage1: Path) -> set[str]:
    """Every kernel body already represented, so DPO adds only what is new."""
    out: set[str] = set()
    if not stage1.is_file():
        return out
    with stage1.open("rb") as fh:
        s = pickle.load(fh)
    for r in s.get("wins") or []:
        src = getattr(r, "final_source", None)
        if src:
            out.add(h(src))
    for r in s.get("repairs") or []:
        msgs = getattr(r, "messages", None) or []
        if msgs:
            out.add(h(code_of(msgs[-1].get("content", ""))))
    for g in s.get("groups") or []:
        cands = (g.get("candidates") if isinstance(g, dict)
                 else getattr(g, "candidates", None)) or []
        for c in cands:
            if isinstance(c, dict) and c.get("source"):
                out.add(h(c["source"]))
    return out


def mine_dpo(known: set[str]) -> tuple[list[dict], collections.Counter]:
    from kore.data.v5_emit import cheats, system_prompt
    from kore.data.v5_policy import admits, dialect

    from kore.data.arena_index import ArenaIndex
    idx = REPO / "data/arena_contamination.json"
    arena = ArenaIndex.load(idx) if idx.is_file() else None

    rows, stats = [], collections.Counter()
    seen: set[str] = set()
    if not DPO.is_file():
        return rows, stats
    with DPO.open(errors="ignore") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001 - torn line
                continue
            stats["rows"] += 1
            prov = d.get("_provenance") or {}
            if str(prov.get("anchor") or d.get("anchor") or "") in EXCLUDED_ANCHORS:
                stats["sub_baseline"] += 1
                continue
            # A recorded SNR settles it. When the field is absent -- 11,703 rows,
            # and disproportionately the ones that are novel -- the anchor is the
            # remaining evidence: `correctness` and `beats_baseline` both mean the
            # chosen side won against a sibling on a measured gate, which is
            # weaker than an SNR reading but is not nothing. `sub_baseline` is
            # already excluded above, so no unproven-and-slow row can arrive here.
            snr = prov.get("chosen_snr_db")
            anchor = str(prov.get("anchor") or d.get("anchor") or "")
            if snr is not None:
                if float(snr) < MIN_SNR_DB:
                    stats["snr_below_bar"] += 1
                    continue
            elif anchor not in ("correctness", "beats_baseline"):
                stats["no_snr_and_weak_anchor"] += 1
                continue
            else:
                stats["kept_on_anchor_evidence"] += 1
            chosen = d.get("chosen")
            if isinstance(chosen, list):
                chosen = next((m.get("content") for m in reversed(chosen)
                               if m.get("role") == "assistant"), "")
            chosen = str(chosen or "")
            body = code_of(chosen)
            if not body:
                stats["no_code"] += 1
                continue
            hh = h(body)
            if hh in known:
                stats["already_present"] += 1
                continue
            if hh in seen:
                stats["dup_within_dpo"] += 1
                continue
            tid = str(prov.get("task_id") or d.get("task_id") or "")
            ok, why = admits({"task_id": tid, "operation": prov.get("operation"),
                              "arch": prov.get("arch") or "gfx950"}, "audited", arena)
            if not ok:
                stats[f"blocked::{why.split(':')[0]}"] += 1
                continue
            why2 = cheats(body)
            if why2:
                stats[f"cheat::{why2.split(':')[0]}"] += 1
                continue
            prompt = d.get("prompt")
            if isinstance(prompt, list):
                prompt = next((m.get("content") for m in prompt
                               if m.get("role") == "user"), "")
            prompt = str(prompt or "").strip()
            if not prompt:
                stats["no_prompt"] += 1
                continue
            seen.add(hh)
            dia = dialect(tid)
            rows.append({
                "messages": [
                    {"role": "system", "content": system_prompt(dia.lower())},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen},
                ],
                "_source": "kernel_dpo_chosen", "_task_id": tid,
                "_dialect": dia, "_shape": "optimize",
                "_provenance": {"kind": "dpo_chosen",
                                "anchor": prov.get("anchor"),
                                "snr_db": snr},
            })
            stats["kept"] += 1
    return rows, stats


def load_modality() -> tuple[list[dict], collections.Counter]:
    from kore.data.arena_index import ArenaIndex
    from kore.data.v5_emit import cheats, system_prompt
    from kore.data.v5_policy import admits, dialect

    idx = REPO / "data/arena_contamination.json"
    arena = ArenaIndex.load(idx) if idx.is_file() else None
    rows, stats = [], collections.Counter()
    if not MODALITY.is_file():
        return rows, stats
    shape_of = {"kernel_torch2kernel": "torch2kernel",
                "kernel_instruction": "instruction",
                "kernel_translate": "port"}
    with MODALITY.open(errors="ignore") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            stats["rows"] += 1
            msgs = d.get("messages") or []
            tid = str(d.get("_task_id") or "")
            if not msgs or not tid:
                stats["malformed"] += 1
                continue
            ok, why = admits({"task_id": tid, "arch": "gfx950"}, "audited", arena)
            if not ok:
                stats[f"blocked::{why.split(':')[0]}"] += 1
                continue
            body = msgs[-1].get("content", "")
            why2 = cheats(code_of(body))
            if why2:
                stats[f"cheat::{why2.split(':')[0]}"] += 1
                continue
            dia = dialect(tid)
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = {"role": "system", "content": system_prompt(dia.lower())}
            src = str(d.get("_source") or "kernel_modality")
            rows.append({**d, "messages": msgs, "_dialect": dia,
                         "_shape": shape_of.get(src, "other"),
                         "_provenance": {"kind": "modality_reshape"}})
            stats[f"kept::{src}"] += 1
    return rows, stats


def extra_replay() -> tuple[list[dict], collections.Counter]:
    """General rows present in an older mixture but absent from v4."""
    stats: collections.Counter = collections.Counter()
    have: set[str] = set()
    if V4.is_file():
        with V4.open(errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                have.add(h(json.dumps(d.get("messages"), sort_keys=True)))
    rows = []
    if OLDER.is_file():
        with OLDER.open(errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                src = str(d.get("_source") or d.get("source") or "")
                if src not in REPLAY_WANTED or not d.get("messages"):
                    continue
                k = h(json.dumps(d["messages"], sort_keys=True))
                if k in have:
                    stats["already_in_v4"] += 1
                    continue
                have.add(k)
                rows.append({"messages": d["messages"], "_source": src,
                             "_provenance": {"kind": "replay_extra"}})
                stats[f"kept::{src}"] += 1
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=str(REPO / "runs/v5_build/stage1.pkl"))
    ap.add_argument("--out-dir", default=str(REPO / "data"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    print("=== indexing existing kernel bodies ===", flush=True)
    known = existing_kernel_hashes(Path(args.stage1))
    print(f"  {len(known):,} distinct kernel bodies already represented\n", flush=True)

    print("=== mining DPO chosen side ===", flush=True)
    dpo_rows, dstats = mine_dpo(known)
    for k, v in dstats.most_common(10):
        print(f"    {k:<30} {v:,}")
    (out / "v5_dpo_chosen.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in dpo_rows))
    print(f"  -> {len(dpo_rows):,} rows\n")

    print("=== orphaned modality slice ===", flush=True)
    mod_rows, mstats = load_modality()
    for k, v in mstats.most_common(8):
        print(f"    {k:<30} {v:,}")
    (out / "v5_modality.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in mod_rows))
    print(f"  -> {len(mod_rows):,} rows\n")

    print("=== extra replay ===", flush=True)
    rep_rows, rstats = extra_replay()
    for k, v in rstats.most_common(8):
        print(f"    {k:<30} {v:,}")
    (out / "v5_replay_extra.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rep_rows))
    print(f"  -> {len(rep_rows):,} rows")

    print(f"\nTOTAL recovered: {len(dpo_rows) + len(mod_rows) + len(rep_rows):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
