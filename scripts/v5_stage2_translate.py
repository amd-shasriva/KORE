#!/usr/bin/env python
"""Stage 2 of the v5 build: turn verified twins into the task shapes the arena asks.

Every datagen episode we ever ran asked one question -- "here is a slow kernel,
make it faster". The arena asks five, and on the 390 tasks measured, v4 scores
55.1% against 55.9% for the base model it was fine-tuned from. It is not that
the model lacks kernels; it is that 184 of 413 tasks pose a question training
never posed. The categories where v4 is furthest behind Opus 5 are exactly the
ones whose *shape* is missing:

    torch2hip        57 tasks   v4  7.5%   70% of failures do not compile
    torch2flydsl     45 tasks   v4  0.0%   compiles, computes the wrong thing
    triton2flydsl    51 tasks   v4 72.0%   14pp WORSE than base
    instruction2*    31 tasks   v4 61.3%

The data to answer them is already on disk and cost nothing new. A verified twin
directory holds ``reference.py`` -- the PyTorch that defines the op's semantics,
executed by the oracle -- beside a kernel that passed the correctness gate on
real gfx950. That pair is a translation example we already own. The HIP
materializer synthesises its kernel *directly from the PyTorch module source*,
so ``(reference.py -> kernel.hip)`` is not an approximation of the torch2hip
shape, it is that shape. The twin's Triton original sits in the pool under the
same id with the backend suffix stripped, giving the dialect-port shape too.

Nothing is generated and nothing is re-run; only the question is rewritten.

Unlike ``build_modality_slices.py``, which reshapes *wins* and is therefore
capped at roughly one row per task that ever won (1,176 in v4), this reads the
twin registries directly and reaches 6,668 verified HIP kernels.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path("/home/shasriva/Kore-RL/KORE")
sys.path.insert(0, str(REPO))

#: Verified-twin registries. The ``_ok`` roots and ``frontier_hip_all`` hold only
#: twins that passed the numerical gate; the raw ``pool_hip``/``pool_flydsl``
#: roots hold every attempt including the failures and are deliberately absent.
TWIN_ROOTS = (
    ("frontier_hip_all", "hip"),
    ("pool_hip_ok", "hip"),
    ("frontier_twins_ok", "both"),
    ("pool_flydsl_ok", "flydsl"),
    ("registry_flydsl_frontier", "flydsl"),
    ("registry_hip_frontier", "hip"),
)

#: Where the Triton original of a twin lives, once its backend suffix is stripped.
ORIGIN_DIRS = (REPO / "data" / "task_pool" / "tasks", REPO / "kore" / "tasks")

#: A kernel longer than this is a whole file of boilerplate around the interesting
#: part; keeping it would spend most of the sequence budget on pybind glue.
MAX_KERNEL_CHARS = 24000
MAX_PROMPT_CHARS = 14000

#: Shapes emitted per verified kernel. The answer is identical across shapes, so
#: each extra shape is another copy of the same target tokens.
MAX_SHAPES_PER_KERNEL = 2


def yaml_field(text: str, key: str, default: str = "") -> str:
    m = re.search(rf'^\s*"?{key}"?\s*:\s*"?([^",\n]+)"?', text, re.M)
    return m.group(1).strip() if m else default


def strip_suffix(task_id: str) -> str:
    for s in ("__hipf", "__hip", "__flydsl"):
        if task_id.endswith(s):
            return task_id[: -len(s)]
    return task_id


def read(p: Path) -> str:
    try:
        return p.read_text(errors="ignore") if p.is_file() else ""
    except OSError:
        return ""


def describe(task_dir: Path, tyaml: str, backend: str) -> str:
    """A written spec with no implementation in it.

    instruction2triton hands the model a paragraph and a name and nothing else,
    so a spec-to-kernel example has to withhold the body -- otherwise it is a
    translation example wearing a different hat.
    """
    op = yaml_field(tyaml, "operation") or task_dir.name
    dtype = yaml_field(tyaml, "dtype", "fp32")
    fam = yaml_field(tyaml, "op_family") or yaml_field(tyaml, "taxonomy_family")
    snr = yaml_field(tyaml, "snr_threshold", "30")
    doc = ""
    for name in ("seed_triton.py", "seed_hip.hip", "reference.py"):
        head = read(task_dir / name)[:1500]
        if not head:
            continue
        m = re.search(r'"""(.*?)"""', head, re.S) or re.search(r"^((?://[^\n]*\n)+)", head, re.M)
        if m:
            doc = m.group(1).strip()
            break
    lines = [f"Implement the `{op}` operation as a GPU kernel for gfx950 (CDNA4).",
             f"dtype: {dtype}" + (f"   family: {fam}" if fam else ""),
             f"It must reach at least {snr} dB SNR against the PyTorch reference."]
    if doc:
        lines.append("\nDescription:\n" + doc[:900])
    return "\n".join(lines)


def row(shape: str, user: str, kernel: str, lang: str, meta: dict, tag: str) -> dict:
    from kore.data.v5_emit import system_prompt
    return {
        "messages": [
            {"role": "system", "content": system_prompt(tag)},
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"```{lang}\n{kernel}\n```"},
        ],
        "_source": shape,
        **meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/v5_translate.jsonl"))
    ap.add_argument("--heldout-policy", choices=("strict", "audited"), default="strict",
                    help="'audited' admits unclassified_operation tasks with cleared prefixes")
    ap.add_argument("--arena-index", default=str(REPO / "data/arena_contamination.json"))
    ap.add_argument("--allow-unscreened", action="store_true",
                    help="build without benchmark screening (not for training data)")
    args = ap.parse_args()

    from kore.data.arena_index import ArenaIndex
    from kore.data.v5_emit import cheats
    from kore.data.v5_policy import admits

    arena = None
    if Path(args.arena_index).is_file():
        arena = ArenaIndex.load(args.arena_index)
        print(f"arena screen: {arena}")
    elif not args.allow_unscreened:
        print(f"ERROR: no arena index at {args.arena_index}. Build it with "
              f"scripts/v5_build_arena_index.py, or pass --allow-unscreened to "
              f"build data that must not be trained on.", file=sys.stderr)
        return 2

    stats: collections.Counter = collections.Counter()
    seen: set[str] = set()          # (shape, target-kernel) content hashes
    seen_task_shape: set[tuple] = set()
    rows: list[dict] = []

    for root_name, kind in TWIN_ROOTS:
        tasks_dir = REPO / "data" / root_name / "tasks"
        if not tasks_dir.is_dir():
            continue
        n0 = len(rows)
        for td in sorted(tasks_dir.iterdir()):
            if not td.is_dir() or not (td / "task.yaml").is_file():
                continue
            tid = td.name
            is_hip = tid.endswith(("__hip", "__hipf"))
            is_fly = tid.endswith("__flydsl")
            if kind == "hip" and not is_hip:
                continue
            if kind == "flydsl" and not is_fly:
                continue
            if not (is_hip or is_fly):
                continue

            base = strip_suffix(tid)
            tyaml = read(td / "task.yaml")
            ok, reason = admits({
                "task_id": tid,
                "operation": yaml_field(tyaml, "operation") or base,
                "arch": yaml_field(tyaml, "gpu_target") or yaml_field(tyaml, "arch") or "gfx950",
                "dtype": yaml_field(tyaml, "dtype") or "fp32",
            }, args.heldout_policy, arena)
            if not ok:
                stats[f"blocked::{reason.split(':')[0]}"] += 1
                continue

            kernel = read(td / ("kernel.hip" if is_hip else "kernel.py"))
            if not kernel.strip():
                kernel = read(td / ("seed_hip.hip" if is_hip else "seed_flydsl.py"))
            if not kernel.strip() or len(kernel) > MAX_KERNEL_CHARS:
                stats["no_kernel_or_too_long"] += 1
                continue
            why = cheats(kernel)
            if why:
                stats[f"cheat::{why.split(':')[0]}"] += 1
                continue

            backend = "HIP C++" if is_hip else "FlyDSL"
            lang = "cpp" if is_hip else "python"
            tag = "hip" if is_hip else "flydsl"
            khash = hashlib.sha1(kernel.encode()).hexdigest()
            meta = {"_task_id": tid, "_backend": tag, "_root": root_name,
                    "_provenance": {"kind": "verified_twin"}}

            # At most TWO shapes per verified kernel, not three.
            #
            # The completion is byte-identical across shapes -- only the question
            # changes -- so a third shape does not add a third lesson, it adds a
            # third copy of the same answer tokens. Repeated targets are the
            # duplication regime that costs real capability: repeating 0.1% of a
            # corpus 100x degrades an 800M model to the quality of a 400M one, and
            # more capable models treat semantically identical targets more like
            # exact duplicates, not less.
            #
            # Shape diversity is still worth having -- format is learned early and
            # separately from content, and a model tuned on one question shape
            # measurably degrades on others -- so the answer is to spread shapes
            # ACROSS twins rather than stack them on each twin. Every twin emits
            # torch2kernel, because that shape is the binding constraint on the
            # whole mixture and the category where v4 is furthest behind; the
            # second slot alternates deterministically between the port and the
            # spec, keeping both populated at half the duplication cost.
            ref = read(td / "reference.py")
            origin = next((d / base for d in ORIGIN_DIRS
                           if (d / base / "seed_triton.py").is_file()), None)
            tri = read(origin / "seed_triton.py") if origin is not None else ""
            spec = describe(td, tyaml, tag)

            candidates = []
            if ref.strip():
                candidates.append((
                    f"kernel_torch2{tag}",
                    f"Implement the following PyTorch module as a single optimized "
                    f"{backend} kernel for gfx950 (CDNA4). Match its numerics.\n\n"
                    f"```python\n{ref[:MAX_PROMPT_CHARS]}\n```"))
            second = []
            if tri.strip():
                second.append((
                    f"kernel_triton2{tag}",
                    f"Port this Triton kernel to {backend} for gfx950 (CDNA4), "
                    f"preserving its numerics.\n\n"
                    f"```python\n{tri[:MAX_PROMPT_CHARS]}\n```"))
            if spec:
                second.append((
                    f"kernel_instruction_{tag}",
                    spec + f"\n\nWrite the complete {backend} implementation."))
            if second:
                # Hash-based so the choice is stable across rebuilds and does not
                # depend on directory iteration order.
                candidates.append(second[int(khash[:8], 16) % len(second)])
            if not candidates:
                candidates = second[:1]

            for shape, user in candidates[:MAX_SHAPES_PER_KERNEL]:
                key = f"{shape}:{khash}"
                if key in seen or (tid, shape) in seen_task_shape:
                    continue
                seen.add(key)
                seen_task_shape.add((tid, shape))
                rows.append(row(shape, user, kernel, lang, meta, tag))
                stats[shape] += 1

        print(f"  {root_name:<26} +{len(rows) - n0:,}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\n=== v5 translation slice: {len(rows):,} rows ===")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<28} {v:>7,}")
    out.with_suffix(".stats.json").write_text(json.dumps(dict(stats), indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
