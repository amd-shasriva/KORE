#!/usr/bin/env python
"""Build the v5 SFT mixture, balanced against what AgentKernelArena actually asks.

v4 was 244,732 rows and taught one skill well. The measured gap is not volume, it
is coverage, in two dimensions the corpus never varied:

TASK SHAPE. Every v4 kernel row is "here is a slow kernel, make it faster",
because all 82k of them came from the 1,546 registry tasks, whose seeds are
already kernels. The arena spends 133 of its 402 tasks on the opposite shape --
torch2hip, torch2flydsl, instruction2triton -- where the target ships empty and a
kernel has to be written from a PyTorch module or a paragraph of prose. All 57
torch2hip .hip files are zero bytes; there is nothing there to speed up. A model
can be fluent in Triton and still never have been asked that question.

LANGUAGE. 86% of kernel rows are Triton and 2.8% are HIP, while the arena is 22%
HIP and carries its two highest bars there (6.89x torch2hip, 6.69x hip2hip).

v5 therefore adds three sources on top of v4:

  1. POOL WINS. The external pool is 13,570 PyTorch modules whose seed contains no
     kernel, so mining it produces exactly the synthesize-from-reference shape.
     None of it had ever been reachable from datagen -- get_task could not resolve
     a pool id -- so 0 of 13,570 had a win against 76% of the registry.

  2. RESHAPED MODALITIES. For any kernel already verified on gfx950 we also hold
     the PyTorch reference that defined it, so (reference -> kernel) and
     (spec -> kernel) are verified examples we own at no GPU cost. Only the
     question is rewritten; the answer already passed.

  3. HIP WINS, to move 2.8% toward the arena's 22%.

Everything is deduplicated on message content across all sources, because these
overlap heavily by construction and a row that enters twice is a row trained on
twice.

    python scripts/build_sft_v5_mixture.py --out data/b05factory/sft/multicap_v5.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import sys

CHARS_PER_TOKEN = 3.6

#: What the arena spends its 402 tasks on, as a target to compare our mix
#: against. Not a quota -- training weight and task count are different things --
#: but a coverage floor: a category we have nothing for cannot be learned.
ARENA_WEIGHTS = {
    "optimize_triton": 165 + 51 + 5,   # triton2triton, triton2flydsl, flydsl2flydsl
    "optimize_hip": 32,                 # hip2hip
    "synthesize_hip": 57,               # torch2hip
    "synthesize_other": 45 + 31,        # torch2flydsl, instruction2triton
    "repository": 9 + 7,
}


def sig(rec: dict) -> str:
    msgs = rec.get("messages") or []
    txt = "".join(str(m.get("content") or "") for m in msgs if isinstance(m, dict))
    return hashlib.md5((txt or json.dumps(rec, sort_keys=True)).encode()).hexdigest()


def rows_of(path: pathlib.Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:  # noqa: BLE001 - a torn line after a kill
                continue


def approx_tokens(rec: dict) -> int:
    msgs = rec.get("messages") or []
    return int(sum(len(str(m.get("content") or "")) for m in msgs) / CHARS_PER_TOKEN)


#: HIP C++ markers. Needed because the v4 rows predate the _backend label, and
#: without sniffing them every HIP row in the base is counted as Triton -- which
#: reports 0% HIP coverage as 0% and 2.8% HIP coverage as 0% alike, hiding the
#: very number the report exists to show.
_HIP_MARKERS = ("__global__", "#include <hip/", "hipLaunchKernelGGL",
                "PYBIND11_MODULE", "torch::Tensor")


def _is_hip(rec: dict) -> bool:
    backend = str(rec.get("_backend") or "")
    if backend:
        return backend == "hip"
    msgs = rec.get("messages") or []
    txt = "".join(str(m.get("content") or "") for m in msgs if isinstance(m, dict))
    if "@triton.jit" in txt:
        return False
    return any(m in txt for m in _HIP_MARKERS)


def classify(rec: dict) -> str:
    """Which arena question does this row teach?

    Shape comes from _source, which is written by whichever builder produced the
    row and is the only authority on whether the prompt showed a kernel or a
    PyTorch module. Language falls back to sniffing the text, because rows built
    before the _backend label exists otherwise all read as Triton.
    """
    src = str(rec.get("_source") or "")
    hip = _is_hip(rec)
    if src in ("kernel_torch2kernel", "kernel_translate"):
        return "synthesize_hip" if hip else "synthesize_other"
    if src == "kernel_instruction":
        return "synthesize_hip" if hip else "synthesize_other"
    if src.startswith("kernel"):
        return "optimize_hip" if hip else "optimize_triton"
    return "replay"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/shared_nfs/shasriva/kore/datagen/multicap_v4.jsonl",
                    help="v4 mixture to build on")
    ap.add_argument("--add", nargs="*", default=[
        "data/modality_v5.jsonl",
    ], help="extra jsonl slices to fold in")
    ap.add_argument("--wins-roots", nargs="*", default=["data/v5pool", "data/hipwave1"],
                    help="datagen roots whose wins should be reshaped in")
    ap.add_argument("--out", default="data/b05factory/sft/multicap_v5.jsonl")
    ap.add_argument("--max-seq-tokens", type=int, default=17408,
                    help="drop rows that cannot fit the training window intact")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seen: set[str] = set()
    kept: list[dict] = []
    per_source = collections.Counter()
    dropped = collections.Counter()

    def take(rec: dict, origin: str) -> None:
        s = sig(rec)
        if s in seen:
            dropped["duplicate"] += 1
            return
        if approx_tokens(rec) > args.max_seq_tokens:
            # A row longer than the window is not "mostly used", it is truncated
            # mid-kernel, which teaches a kernel that stops halfway.
            dropped["too_long"] += 1
            return
        if not (rec.get("messages") or []):
            dropped["empty"] += 1
            return
        seen.add(s)
        rec.setdefault("_mix", origin)
        kept.append(rec)
        per_source[rec.get("_source", "?")] += 1

    for rec in rows_of(pathlib.Path(args.base)):
        take(rec, "v4_base")
    print(f"v4 base: {len(kept)} rows kept")

    for extra in args.add:
        before = len(kept)
        for rec in rows_of(pathlib.Path(extra)):
            take(rec, pathlib.Path(extra).stem)
        print(f"{extra}: +{len(kept) - before}")

    # Reshape any wins that have landed since the slices were last built, so a
    # mixture built mid-sweep still picks up everything mined so far.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    try:
        from build_modality_slices import build as build_slices

        tmp = pathlib.Path(args.out).with_suffix(".fresh_modalities.jsonl")
        stats = build_slices([pathlib.Path(r) for r in args.wins_roots], tmp, args.seed)
        print(f"fresh modality slices: {stats}")
        before = len(kept)
        for rec in rows_of(tmp):
            take(rec, "fresh_modalities")
        print(f"fresh modalities: +{len(kept) - before}")
    except Exception as exc:  # noqa: BLE001 - a mixture is still valid without them
        print(f"WARNING: could not build fresh modality slices: "
              f"{type(exc).__name__}: {exc}")

    rng = random.Random(args.seed)
    rng.shuffle(kept)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    shapes = collections.Counter(classify(r) for r in kept)
    toks = collections.Counter()
    for r in kept:
        toks[classify(r)] += approx_tokens(r)
    total_tok = sum(toks.values()) or 1
    kernel_tok = total_tok - toks.get("replay", 0)

    print(f"\nwrote {len(kept)} rows -> {out}")
    print(f"dropped: {dict(dropped)}")
    print(f"\n{'shape':<20}{'rows':>8}{'tokens(M)':>11}{'% kernel tok':>14}"
          f"{'arena %':>9}")
    for k in ("optimize_triton", "optimize_hip", "synthesize_hip",
              "synthesize_other", "repository"):
        arena = 100 * ARENA_WEIGHTS[k] / sum(ARENA_WEIGHTS.values())
        share = 100 * toks.get(k, 0) / (kernel_tok or 1)
        print(f"{k:<20}{shapes.get(k,0):>8}{toks.get(k,0)/1e6:>11.1f}"
              f"{share:>13.1f}%{arena:>8.1f}%")
    print(f"{'replay':<20}{shapes.get('replay',0):>8}{toks.get('replay',0)/1e6:>11.1f}"
          f"{'':>14}{'':>9}")
    print(f"\ntop sources: {dict(per_source.most_common(10))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
