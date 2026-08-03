#!/usr/bin/env python3
"""Partition the trainable task corpus by what each seed *actually is*.

Every task carries a ``seed_source``, but "there is a seed" says nothing about
which AgentKernelArena shape the task trains.  A seed that is already a Triton
kernel poses ``triton2triton`` (optimize an existing kernel).  A seed that is
plain eager torch poses ``torch2triton`` (lower a reference to a kernel) --
same harness, same "optimize this" framing, but a different capability, because
a seed with no kernel in it cannot be made faster by editing; it has to be
replaced by a kernel written from scratch.  This script decides which, per task,
by inspecting the seed with the AST rather than by trusting metadata.

The corpus is two stores, and the distinction matters: ``kore.tasks.registry``
holds the hand-authored tasks, while the external pool
(``kore.tasks.external``, ~13.5k tasks) is materialized separately and is
resolved by datagen through ``kore.data.saturated_agentic`` -- so a
registry-only census answers the wrong question.  Pass ``--pool-root`` (or set
``KORE_TASK_POOL``) to include it.

Classification is deliberately conservative: a seed counts as a device kernel
only when a kernel *definition* is present (``@triton.jit`` / ``__global__``),
not merely when the module imports triton.  Anything that cannot be parsed or
does not fit a known form is reported as ``unknown`` rather than bucketed by
guesswork.

Usage:
    python scripts/seed_provenance_partition.py [--json OUT] [--pool-root DIR]
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kore.tasks import registry  # noqa: E402
from kore.tasks.base import Task  # noqa: E402

# Seed kinds, in the vocabulary of the AKA shapes they pose.
TRITON_KERNEL = "triton_kernel"  # -> triton2triton (optimize existing)
HIP_KERNEL = "hip_kernel"  # -> hip2hip (optimize existing)
TORCH_EAGER = "torch_eager"  # -> torch2{triton,hip} (lower a reference)
TORCH_ALIAS_REFERENCE = "torch_alias_reference"  # torch_eager, via indirection
UNKNOWN = "unknown"

_LOWERING_SHAPES = {TORCH_EAGER, TORCH_ALIAS_REFERENCE}


@dataclass
class SeedVerdict:
    task_id: str
    backend: str
    kind: str
    # Evidence, so a reader can audit the call without rerunning the script.
    triton_jit_kernels: int
    hip_global_kernels: int
    imports_triton: bool
    aliases_reference_baseline: bool
    entry_is_torch_only: Optional[bool]
    note: str = ""
    store: str = "registry"

    @property
    def poses_lowering_task(self) -> bool:
        return self.kind in _LOWERING_SHAPES


def _decorator_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for dec in getattr(node, "decorator_list", []) or []:
        target = dec.func if isinstance(dec, ast.Call) else dec
        parts: list[str] = []
        while isinstance(target, ast.Attribute):
            parts.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            parts.append(target.id)
        if parts:
            names.append(".".join(reversed(parts)))
    return names


def _classify_python_seed(source: str, entry_name: str) -> SeedVerdict:
    """Classify a Python seed module. ``task_id``/``backend`` filled by caller."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return SeedVerdict(
            task_id="",
            backend="",
            kind=UNKNOWN,
            triton_jit_kernels=0,
            hip_global_kernels=0,
            imports_triton=False,
            aliases_reference_baseline=False,
            entry_is_torch_only=None,
            note=f"unparseable: {exc}",
        )

    jit_kernels = 0
    imports_triton = False
    aliases_baseline = False
    entry_fn: Optional[ast.FunctionDef] = None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = ""
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            else:
                mod = ",".join(a.name for a in node.names)
            if "triton" in mod:
                imports_triton = True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decs = _decorator_names(node)
            if any(d.endswith("jit") or d.endswith("autotune") or d.endswith("heuristics")
                   for d in decs) and any("triton" in d or d in {"jit", "autotune", "heuristics"}
                                          for d in decs):
                # triton.jit / tl.jit / bare `jit` imported from triton
                if any("triton" in d for d in decs) or imports_triton:
                    jit_kernels += 1
            if node.name == entry_name and isinstance(node, ast.FunctionDef):
                entry_fn = node
        # The external-pool seed form: `<entry> = _NS['baseline_fn']`
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == entry_name:
                    val = node.value
                    if isinstance(val, ast.Subscript):
                        idx = val.slice
                        key = idx.value if isinstance(idx, ast.Constant) else None
                        if isinstance(key, str) and "baseline" in key:
                            aliases_baseline = True

    entry_is_torch_only: Optional[bool] = None
    if entry_fn is not None:
        launches_kernel = False
        for node in ast.walk(entry_fn):
            # A triton launch is a Subscript call: kernel[grid](...)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                launches_kernel = True
        entry_is_torch_only = not launches_kernel

    if jit_kernels > 0:
        kind = TRITON_KERNEL
    elif aliases_baseline:
        kind = TORCH_ALIAS_REFERENCE
    elif not imports_triton:
        kind = TORCH_EAGER
    elif entry_is_torch_only:
        # Imports triton but defines no kernel and launches none.
        kind = TORCH_EAGER
    else:
        kind = UNKNOWN

    return SeedVerdict(
        task_id="",
        backend="",
        kind=kind,
        triton_jit_kernels=jit_kernels,
        hip_global_kernels=0,
        imports_triton=imports_triton,
        aliases_reference_baseline=aliases_baseline,
        entry_is_torch_only=entry_is_torch_only,
    )


def _classify_hip_seed(source: str) -> SeedVerdict:
    # HIP C++: count __global__ kernel definitions textually; there is no
    # stdlib C++ parser and a kernel definition is unambiguous enough.
    n = source.count("__global__")
    return SeedVerdict(
        task_id="",
        backend="",
        kind=HIP_KERNEL if n > 0 else UNKNOWN,
        triton_jit_kernels=0,
        hip_global_kernels=n,
        imports_triton=False,
        aliases_reference_baseline=False,
        entry_is_torch_only=None,
        note="" if n else "no __global__ kernel found in .hip seed",
    )


def _entry_name(task: Task) -> str:
    raw = task.raw
    for key in ("entry_name", "entry_point", "kernel_entry"):
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return task.operation


def classify(task: Task, store: str = "registry") -> SeedVerdict:
    try:
        source = task.seed_source
    except OSError as exc:
        v = SeedVerdict("", "", UNKNOWN, 0, 0, False, False, None, f"unreadable: {exc}")
    else:
        if task.seed_kernel_name.endswith((".hip", ".cpp", ".cu", ".cc")):
            v = _classify_hip_seed(source)
        else:
            v = _classify_python_seed(source, _entry_name(task))
    v.task_id = task.task_id
    v.backend = task.backend
    v.store = store
    return v


def _report(label: str, verdicts: list[SeedVerdict]) -> dict:
    total = len(verdicts)
    if not total:
        print(f"\n=== {label}: 0 tasks (not materialized here) ===")
        return {"total_tasks": 0}
    by_kind = collections.Counter(v.kind for v in verdicts)
    by_backend_kind = collections.Counter((v.backend, v.kind) for v in verdicts)
    lowering = sum(1 for v in verdicts if v.poses_lowering_task)
    optimize = sum(1 for v in verdicts if v.kind in {TRITON_KERNEL, HIP_KERNEL})

    print(f"\n=== {label}: {total} tasks ===")
    print("seed kind (what the seed ACTUALLY is):")
    for kind, n in by_kind.most_common():
        print(f"  {kind:24s} {n:6d}  ({100.0 * n / total:5.1f}%)")
    print("AKA shape posed by the seed:")
    print(f"  optimize-existing (X2X)      {optimize:6d}  ({100.0*optimize/total:5.1f}%)")
    print(f"  synthesize-from-reference    {lowering:6d}  ({100.0*lowering/total:5.1f}%)")
    print(f"  unknown                      {by_kind[UNKNOWN]:6d}")
    print("by declared backend x seed kind:")
    for (backend, kind), n in sorted(by_backend_kind.items()):
        print(f"  {backend:10s} {kind:24s} {n:6d}")

    unknowns = [v for v in verdicts if v.kind == UNKNOWN]
    if unknowns:
        print(f"unknown seeds ({len(unknowns)}), first 20:")
        for v in unknowns[:20]:
            print(f"  {v.task_id:50s} {v.backend:8s} {v.note}")
    return {
        "total_tasks": total,
        "counts_by_seed_kind": dict(by_kind),
        "shape_counts": {
            "optimize_existing": optimize,
            "synthesize_from_reference": lowering,
            "unknown": by_kind[UNKNOWN],
        },
        "by_backend_and_kind": {f"{b}|{k}": n for (b, k), n in by_backend_kind.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--pool-root", type=Path, default=None,
                    help="external pool root (default: KORE_TASK_POOL or data/task_pool)")
    ap.add_argument("--registry-only", action="store_true",
                    help="skip the external pool even when it is materialized")
    ap.add_argument("--per-task", action="store_true",
                    help="emit per-task verdicts into --json (large for the pool)")
    args = ap.parse_args()

    reg = [classify(t, "registry") for t in registry.all_tasks()]

    pool: list[SeedVerdict] = []
    if not args.registry_only:
        from kore.tasks.external import load_pool, pool_root

        root = args.pool_root or pool_root()
        try:
            pool = [classify(t, "pool") for t in load_pool(root)]
        except Exception as exc:  # noqa: BLE001 - absence is a normal local state
            print(f"pool at {root} not loadable: {type(exc).__name__}: {exc}")

    reg_report = _report("registry (kore.tasks.registry)", reg)
    pool_report = _report("external pool (kore.tasks.external)", pool)
    combined = reg + pool
    comb_report = _report("COMBINED datagen corpus (what saturated_agentic resolves)",
                          combined)

    if args.json:
        payload = {
            "registry": reg_report,
            "pool": pool_report,
            "combined": comb_report,
        }
        if args.per_task:
            payload["verdicts"] = [asdict(v) for v in combined]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
