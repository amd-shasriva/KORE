#!/usr/bin/env python
"""Measure v4 and v5 from the files themselves, not from build logs.

A build log reports what the builder believed it did. This reads both datasets
end to end with the real tokenizer and reports what is actually in them, so every
number can be checked independently of the code that produced it.

It is also written to surface the places where a headline number flatters the
dataset: how much of the "kernel" side is generation versus question-answering,
how much of a slice is repetition rather than distinct content, and how the
token-weighted composition differs from the row-weighted one.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import re
import statistics
import sys
from pathlib import Path

REPO = Path("/home/shasriva/Kore-RL/KORE")
sys.path.insert(0, str(REPO))

V4 = REPO / "data/b05factory/sft/multicap_v3.jsonl"
V5 = REPO / "data/v5_sft.jsonl"

_CODE = re.compile(r"```[a-zA-Z+]*\n(.*?)```", re.S)

#: Slices that ask the model to EMIT a kernel. Everything else on the kernel side
#: is knowledge or discussion, which is useful but is not generation.
GENERATION_SOURCES = {
    "kernel_repair", "kernel_win", "kernel_gold_win", "kernel_step_centric",
    "kernel_dpo_chosen", "kernel_torch2hip", "kernel_triton2hip",
    "kernel_instruction_hip", "kernel_torch2flydsl", "kernel_triton2flydsl",
    "kernel_instruction_flydsl", "kernel_torch2kernel", "kernel_instruction",
    "kernel_translate", "kernel_flydsl_language",
}


def code_of(t: str) -> str:
    m = _CODE.findall(t or "")
    return max(m, key=len).strip() if m else (t or "").strip()


def h(s: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", s or "").strip().encode()).hexdigest()


def scan(path: Path, tok) -> dict:
    rows = 0
    per_src: dict[str, dict] = collections.defaultdict(
        lambda: {"rows": 0, "tokens": 0, "lens": [], "targets": collections.Counter(),
                 "tasks": set(), "dialects": collections.Counter(),
                 "shapes": collections.Counter()})
    all_targets: collections.Counter = collections.Counter()
    with path.open(errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            rows += 1
            src = str(d.get("_source") or d.get("source") or "(none)")
            msgs = d.get("messages") or []
            try:
                n = len(tok.apply_chat_template(msgs, tokenize=True,
                                                add_generation_prompt=False))
            except Exception:  # noqa: BLE001
                n = 0
            body = msgs[-1].get("content", "") if msgs else ""
            key = h(code_of(body))
            s = per_src[src]
            s["rows"] += 1
            s["tokens"] += n
            s["lens"].append(n)
            s["targets"][key] += 1
            all_targets[key] += 1
            tid = str(d.get("_task_id") or "")
            if tid:
                s["tasks"].add(tid)
            if d.get("_dialect"):
                s["dialects"][d["_dialect"]] += 1
            if d.get("_shape"):
                s["shapes"][d["_shape"]] += 1
    return {"rows": rows, "per_src": per_src, "all_targets": all_targets}


def table(res: dict, title: str) -> None:
    per = res["per_src"]
    total_rows = res["rows"]
    total_tok = sum(v["tokens"] for v in per.values())
    print(f"\n{'=' * 96}\n{title}: {total_rows:,} rows, {total_tok:,} tokens\n{'=' * 96}")
    print(f"{'source':<30}{'rows':>9}{'row%':>7}{'tokens':>14}{'tok%':>7}"
          f"{'med':>7}{'max':>7}{'uniq':>9}{'rep':>6}")
    for src, v in sorted(per.items(), key=lambda kv: -kv[1]["tokens"]):
        lens = sorted(v["lens"]) or [0]
        uniq = len(v["targets"])
        rep = v["rows"] / max(1, uniq)
        print(f"{src:<30}{v['rows']:>9,}{100*v['rows']/total_rows:>6.1f}%"
              f"{v['tokens']:>14,}{100*v['tokens']/max(1,total_tok):>6.1f}%"
              f"{lens[len(lens)//2]:>7,}{lens[-1]:>7,}{uniq:>9,}{rep:>6.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--revision", default="b2cff646eb4bb1d68355c01b18ae02e7cf42d120")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)

    print("reading v4 ...", flush=True)
    v4 = scan(V4, tok)
    print("reading v5 ...", flush=True)
    v5 = scan(V5, tok)

    table(v4, "V4  (data/b05factory/sft/multicap_v3.jsonl)")
    table(v5, "V5  (data/v5_sft.jsonl)")

    # ---- headline comparison, in both units --------------------------------
    def split(res):
        per = res["per_src"]
        k_rows = sum(v["rows"] for s, v in per.items() if s.startswith("kernel"))
        k_tok = sum(v["tokens"] for s, v in per.items() if s.startswith("kernel"))
        g_rows = sum(v["rows"] for s, v in per.items() if s in GENERATION_SOURCES)
        g_tok = sum(v["tokens"] for s, v in per.items() if s in GENERATION_SOURCES)
        return k_rows, k_tok, g_rows, g_tok

    print(f"\n{'=' * 96}\nHEADLINE\n{'=' * 96}")
    print(f"{'':<34}{'v4':>16}{'v5':>16}{'ratio':>10}")
    for label, f in (
        ("total rows", lambda r: r["rows"]),
        ("total tokens", lambda r: sum(v['tokens'] for v in r['per_src'].values())),
        ("kernel rows", lambda r: split(r)[0]),
        ("kernel tokens", lambda r: split(r)[1]),
        ("kernel GENERATION rows", lambda r: split(r)[2]),
        ("kernel GENERATION tokens", lambda r: split(r)[3]),
        ("distinct targets (whole file)", lambda r: len(r["all_targets"])),
    ):
        a, b = f(v4), f(v5)
        print(f"{label:<34}{a:>16,}{b:>16,}{(b/max(1,a)):>9.2f}x")

    for name, res in (("v4", v4), ("v5", v5)):
        k_rows, k_tok, g_rows, g_tok = split(res)
        tot_tok = sum(v["tokens"] for v in res["per_src"].values())
        print(f"\n{name}: kernel {100*k_rows/res['rows']:.1f}% of rows / "
              f"{100*k_tok/max(1,tot_tok):.1f}% of tokens   |   "
              f"generation-only {100*g_rows/res['rows']:.1f}% rows / "
              f"{100*g_tok/max(1,tot_tok):.1f}% tokens")

    # ---- v5 shape and dialect, weighted by tokens as well as rows ----------
    per = v5["per_src"]
    for key in ("shapes", "dialects"):
        agg_rows: collections.Counter = collections.Counter()
        agg_tok: collections.Counter = collections.Counter()
        for src, v in per.items():
            if not v[key]:
                continue
            tpr = v["tokens"] / max(1, v["rows"])
            for k, n in v[key].items():
                agg_rows[k] += n
                agg_tok[k] += int(n * tpr)
        tr, tt = sum(agg_rows.values()), sum(agg_tok.values())
        if not tr:
            continue
        print(f"\nV5 kernel body by {key[:-1]}:")
        print(f"  {'':<16}{'rows':>10}{'row%':>8}{'tokens':>14}{'tok%':>8}")
        for k in sorted(agg_rows, key=lambda x: -agg_tok[x]):
            print(f"  {k:<16}{agg_rows[k]:>10,}{100*agg_rows[k]/tr:>7.1f}%"
                  f"{agg_tok[k]:>14,}{100*agg_tok[k]/max(1,tt):>7.1f}%")

    # ---- distinct task and family coverage --------------------------------
    tasks: set[str] = set()
    for v in per.values():
        tasks |= v["tasks"]
    def base(t):
        for s in ("__hipf", "__hip", "__flydsl"):
            if t.endswith(s):
                return t[: -len(s)]
        return t
    bases = {base(t) for t in tasks}
    fams: collections.Counter = collections.Counter()
    for b in bases:
        m = re.match(r"(?:kbk|syn_synth|gen|genb|genv|hip)_([a-z0-9]+)", b)
        fams[m.group(1) if m else b.split("_")[0]] += 1
    print(f"\nV5 distinct task ids {len(tasks):,}  distinct base tasks {len(bases):,}"
          f"  distinct name-families {len(fams):,}")
    print("  top 18 families:", ", ".join(f"{k}({v})" for k, v in fams.most_common(18)))

    # ---- win speedups, measured from the records ---------------------------
    print(f"\n{'=' * 96}\nWIN / GOLD-WIN SPEEDUPS (from the records)\n{'=' * 96}")
    for label, pk, key in (("real wins", "stage1.pkl", "wins"),
                           ("gold wins", "stage3_audited.pkl", "gold_wins")):
        p = REPO / "runs/v5_build" / pk
        if not p.is_file():
            print(f"  {label}: {pk} missing")
            continue
        with p.open("rb") as fh:
            obj = pickle.load(fh)
        recs = obj.get(key) or []
        sp = sorted(float(getattr(r, "speedup", 0) or 0) for r in recs
                    if getattr(r, "speedup", None))
        if not sp:
            print(f"  {label}: no speedups recorded")
            continue
        gm = statistics.geometric_mean([x for x in sp if x > 0])
        print(f"  {label:<11} n={len(sp):>7,}  geomean={gm:.3f}x  "
              f"median={sp[len(sp)//2]:.3f}x  p90={sp[int(len(sp)*.9)]:.3f}x  "
              f"max={sp[-1]:.3f}x  >=1.5x={100*sum(1 for x in sp if x>=1.5)/len(sp):.1f}%"
              f"  >=2x={100*sum(1 for x in sp if x>=2)/len(sp):.1f}%")

    # ---- repetition, the honest version ----------------------------------
    at = v5["all_targets"]
    reps = sum(c - 1 for c in at.values() if c > 1)
    print(f"\nV5 repetition: {len(at):,} distinct targets, {reps:,} repeated rows "
          f"({100*reps/v5['rows']:.1f}%), max {max(at.values())}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
