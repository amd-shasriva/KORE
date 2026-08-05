#!/usr/bin/env python
"""Reshape already-verified kernels into the task shapes AgentKernelArena tests.

The arena asks five different questions and our training data only ever asked
one of them:

    optimize this kernel in place   triton2triton, hip2hip, flydsl2flydsl  202
    translate from PyTorch          torch2hip, torch2flydsl                102
    port between dialects           triton2flydsl                           51
    write one from a spec           instruction2triton                      31
    edit inside a repository        repository, image_kernel                16

Every datagen episode is "here is a slow kernel, make it faster", so a model
trained on it has never once been asked to produce a kernel from a PyTorch module
or from a written description. That is 133 arena tasks -- a third of the
benchmark -- whose *shape* is absent from training, independent of language.

The fix costs no GPU time. A win already contains a kernel that passed the
correctness gate on real gfx950, and the task directory beside it contains the
PyTorch reference that defined the semantics. So the pair (reference.py ->
final_source) is a verified translation example we already own, and
(description -> final_source) is a verified spec-to-kernel example. Nothing is
re-run; only the question is rewritten.

Emits, per win:
  * ``kernel_torch2kernel``  PyTorch module   -> verified kernel
  * ``kernel_instruction``   written spec     -> verified kernel  (no source shown)
  * ``kernel_translate``     kernel in dialect A -> verified kernel in dialect B,
                             for operations we have won in both backends

    python scripts/build_modality_slices.py --out data/modality_v5.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Where task directories live. The registry holds hand-authored and generated
#: tasks; the pool holds the 13,570 KernelBook/synthetic PyTorch modules.
TASK_DIRS = (REPO / "kore" / "tasks", REPO / "data" / "task_pool" / "tasks")

SYSTEM = ("You are KORE, an expert AMD GPU kernel engineer targeting MI355X "
          "(gfx950, CDNA4).")

#: A win is only training material if it was actually better than what it
#: started from. Speedup 1.0 means the model returned something equivalent to the
#: seed, which teaches "change nothing" -- and on in-place arena tasks that is
#: precisely the degenerate answer that already passes correctness for free.
MIN_SPEEDUP = 1.05


def _iter_wins(roots):
    """Every verified win under the given data roots.

    Quarantined directories are skipped: they were set aside by the pipeline as
    untrustworthy, and laundering them through a reshape would put exactly the
    rows someone rejected back into training under a new name.
    """
    for root in roots:
        if not root.is_dir():
            continue
        for f in root.rglob("wins/*.jsonl"):
            if "_quarantine" in f.parts:
                continue
            for line in f.read_text(errors="ignore").splitlines():
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001 - a torn line after a kill
                    continue
                if d.get("type") == "win" and d.get("final_source"):
                    yield d


def _task_dir(task_id: str):
    for base in TASK_DIRS:
        p = base / task_id
        if (p / "task.yaml").is_file():
            return p
    return None


def _yaml_field(text: str, key: str, default: str = "") -> str:
    m = re.search(rf'^\s*"?{key}"?\s*:\s*"?([^",\n]+)"?', text, re.M)
    return m.group(1).strip() if m else default


def _reference_source(task_dir: Path) -> str:
    """The PyTorch that defines this task's semantics.

    Pool tasks carry the module inline in the pool record and a thin reference.py
    shim; registry tasks carry it the same way. Either way reference.py is what
    the oracle actually executes, so it is the honest 'before' side of a
    translation pair.
    """
    p = task_dir / "reference.py"
    return p.read_text(errors="ignore") if p.is_file() else ""


def _describe(task_dir: Path, meta: dict) -> str:
    """A written spec for the op, with no source code in it.

    instruction2triton gives the model a paragraph and a function name and
    nothing else, so a spec-to-kernel example has to withhold the implementation
    -- otherwise it is just another translation example wearing a different hat.
    """
    y = (task_dir / "task.yaml").read_text(errors="ignore")
    op = _yaml_field(y, "operation") or meta.get("operation", "kernel")
    dtype = _yaml_field(y, "dtype", "fp32")
    fam = _yaml_field(y, "op_family") or _yaml_field(y, "taxonomy_family")
    snr = _yaml_field(y, "snr_threshold", "30")
    # Doc comments in the seed describe intent without giving the body away.
    doc = ""
    for name in ("seed_triton.py", "seed_hip.hip"):
        s = task_dir / name
        if s.is_file():
            head = s.read_text(errors="ignore")[:1200]
            m = re.search(r'"""(.*?)"""', head, re.S) or re.search(
                r"^((?://[^\n]*\n)+)", head, re.M)
            if m:
                doc = m.group(1).strip()
            break
    lines = [f"Implement the `{op}` operation as a GPU kernel.",
             f"dtype: {dtype}" + (f"   family: {fam}" if fam else ""),
             f"It must reach at least {snr} dB SNR against the reference."]
    if doc:
        lines.append("\nDescription:\n" + doc[:900])
    return "\n".join(lines)


def _backend_of(task_dir: Path) -> str:
    y = (task_dir / "task.yaml").read_text(errors="ignore")
    return _yaml_field(y, "backend", "triton")


def _row(source: str, messages: list, meta: dict) -> dict:
    return {"messages": messages, "_source": source, **meta}


def build(roots, out_path: Path, seed: int = 0) -> dict:
    rng = random.Random(seed)
    stats = collections.Counter()
    best_by_task = {}

    for w in _iter_wins(roots):
        if (w.get("speedup") or 0) < MIN_SPEEDUP:
            stats["skipped_no_gain"] += 1
            continue
        tid = w.get("task_id")
        prev = best_by_task.get(tid)
        # One row per task per shape, from its best win: many near-identical
        # attempts on one task would teach that task's answer by rote rather than
        # the skill, and the corpus is already long-tailed by task.
        if prev is None or (w.get("speedup") or 0) > (prev.get("speedup") or 0):
            best_by_task[tid] = w

    rows = []
    by_op_backend = collections.defaultdict(dict)

    for tid, w in best_by_task.items():
        td = _task_dir(tid)
        if td is None:
            stats["no_task_dir"] += 1
            continue
        backend = _backend_of(td)
        kernel = w["final_source"]
        lang = "cpp" if backend == "hip" else "python"
        meta = {"_task_id": tid, "_backend": backend,
                "_speedup": round(float(w.get("speedup") or 0), 4),
                "_snr_db": w.get("snr_db"), "_provenance": "reshaped_from_win"}
        op = w.get("operation") or _yaml_field(
            (td / "task.yaml").read_text(errors="ignore"), "operation")
        if op:
            by_op_backend[str(op).replace("hip_", "", 1)][backend] = (kernel, lang, meta)

        # --- shape 1: PyTorch -> kernel (torch2hip / torch2flydsl) -----------
        ref = _reference_source(td)
        if ref.strip():
            rows.append(_row("kernel_torch2kernel", [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                    f"Implement the following PyTorch module as a single "
                    f"optimized {backend} kernel for gfx950. Match its numerics.\n\n"
                    f"```python\n{ref[:12000]}\n```"},
                {"role": "assistant", "content": f"```{lang}\n{kernel}\n```"},
            ], meta))
            stats["torch2kernel"] += 1

        # --- shape 2: written spec -> kernel (instruction2triton) ------------
        spec = _describe(td, w)
        if spec:
            rows.append(_row("kernel_instruction", [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": spec +
                    f"\n\nWrite the complete {backend} implementation for gfx950."},
                {"role": "assistant", "content": f"```{lang}\n{kernel}\n```"},
            ], meta))
            stats["instruction"] += 1

    # --- shape 3: dialect A -> dialect B (triton2flydsl, and triton->hip) ----
    # Only for operations won in both backends, so both sides of the pair are
    # kernels that actually passed on hardware rather than a translation someone
    # hoped was right.
    for op, per_backend in by_op_backend.items():
        if len(per_backend) < 2:
            continue
        for src_b, dst_b in ((a, b) for a in per_backend for b in per_backend if a != b):
            (src_k, src_l, _), (dst_k, dst_l, dmeta) = per_backend[src_b], per_backend[dst_b]
            rows.append(_row("kernel_translate", [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                    f"Port this {src_b} kernel to {dst_b} for gfx950, preserving "
                    f"its numerics.\n\n```{src_l}\n{src_k[:12000]}\n```"},
                {"role": "assistant", "content": f"```{dst_l}\n{dst_k}\n```"},
            ], {**dmeta, "_translate": f"{src_b}2{dst_b}"}))
            stats["translate"] += 1

    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    stats["rows_written"] = len(rows)
    stats["tasks_used"] = len(best_by_task)
    return dict(stats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-roots", nargs="*", default=None,
                    help="datagen roots to scan (default: every data/* with wins)")
    ap.add_argument("--out", default="data/modality_v5.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    roots = ([Path(r) for r in args.data_roots] if args.data_roots
             else sorted(p for p in (REPO / "data").iterdir() if p.is_dir()))
    stats = build(roots, Path(args.out), args.seed)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
