#!/usr/bin/env python
"""Audit newly added HIP / low-precision tasks: dedup, decontamination, overlap.

Five independent checks, because each catches something the others miss:

  ids          a new task id colliding with an existing registry or pool task
  dedup        a new seed duplicating an existing seed (exact SHA-256, normalized
               AST, semantic graph, and directional containment)
  decontam     a new task's text overlapping any held-out reference document,
               which would put an evaluation kernel into a trainable task
  operation    operator-level overlap with EXISTING registry tasks -- expected
               along the backend axis (same math, different language), so this is
               reported as a disclosure rather than a failure
  benchmark    operator-level overlap with AgentKernelArena, the benchmark this
               project scores against to compare with Opus.  Training on a task
               whose operator AKA also tests makes that comparison weaker, and the
               only unacceptable version of that is the silent one.

Exit status is 0 only when ids, dedup and decontam are all clean.  The two overlap
sections never fail the audit; they print what a reviewer has to know.

    PYTHONPATH=. python scripts/audit_hip_tasks.py [--prefix hip_] [--json out.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
from typing import Any, Optional

REPO = pathlib.Path(__file__).resolve().parents[1]
AKA_ROOT = pathlib.Path(
    os.environ.get("KORE_AKA_ROOT", pathlib.Path.home() / "third_party/AgentKernelArena"))


#: Generated per-task boilerplate.  ``driver.py`` and ``reference.py`` are 3-10 line
#: shims that delegate to a shared module, so they are BYTE-IDENTICAL in structure
#: across ~1,300 tasks by construction.  Every dedup and AST check therefore
#: matches them against each other, which is a property of the generator and not
#: evidence about a task.  They are audited and reported separately rather than
#: excluded, because silently dropping files from a leakage audit is how a real hit
#: gets missed.
_SHIM_FILENAMES = frozenset({"driver.py", "reference.py"})
_SOURCE_SUFFIXES = (".py", ".hip", ".cpp", ".cu", ".h", ".hpp")


def _task_texts(task) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(pathlib.Path(task.dir).iterdir()):
        if path.is_file() and path.suffix in _SOURCE_SUFFIXES:
            try:
                out[path.name] = path.read_text(errors="replace")
            except OSError:
                continue
    return out


def _is_shim(filename: str) -> bool:
    return filename in _SHIM_FILENAMES


def _pool_ids() -> tuple[set[str], str]:
    """Task ids already in the external pool, if a pool is materialized."""
    from kore.tasks import external

    try:
        root = external.pool_root()
    except Exception:  # noqa: BLE001
        return set(), "pool root unavailable"
    index = pathlib.Path(root) / external.POOL_INDEX_NAME
    if not index.is_file():
        return set(), f"no pool index at {index} (pool is built on the cluster)"
    ids: set[str] = set()
    with index.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line).get("task_id") or ""))
            except json.JSONDecodeError:
                continue
    ids.discard("")
    return ids, f"{len(ids)} pool task ids from {index}"


def _aka_operators() -> dict[str, list[str]]:
    """Coarse operator keys per AgentKernelArena task type, from directory names."""
    if not (AKA_ROOT / "tasks").is_dir():
        return {}
    out: dict[str, list[str]] = collections.defaultdict(list)
    for config in sorted((AKA_ROOT / "tasks").rglob("config.yaml")):
        relative = config.parent.relative_to(AKA_ROOT / "tasks")
        parts = relative.parts
        if not parts:
            continue
        out[parts[0]].append(relative.parts[-1].lower())
    return dict(out)


_OPERATOR_KEYS = (
    "gelu", "silu", "swiglu", "relu", "sigmoid", "tanh", "softmax", "layernorm",
    "layer_norm", "rmsnorm", "rms_norm", "matmul", "gemm", "quant", "dequant",
    "attention", "attn", "transpose", "moe",
)


def _operator_key(name: str) -> Optional[str]:
    lowered = name.lower().replace("-", "_")
    for key in _OPERATOR_KEYS:
        if key in lowered:
            return key
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="hip_", help="task-id prefix to audit")
    parser.add_argument("--json", default="", help="write the full report here")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO))
    from kore.data.dedup import (
        content_hash,
        directional_containment,
        graph_fingerprint,
        normalized_ast_fingerprint,
    )
    from kore.data.decontam import analyze_text_contamination, build_heldout_ngrams
    from kore.tasks.registry import all_tasks

    tasks = all_tasks()
    new = [t for t in tasks if t.task_id.startswith(args.prefix)]
    existing = [t for t in tasks if not t.task_id.startswith(args.prefix)]
    if not new:
        print(f"no tasks with prefix {args.prefix!r}")
        return 1

    report: dict[str, Any] = {"prefix": args.prefix, "n_new": len(new),
                              "n_existing": len(existing)}
    failures: list[str] = []

    # ---- ids ------------------------------------------------------------- #
    pool, pool_note = _pool_ids()
    new_ids = {t.task_id for t in new}
    id_collisions = sorted(new_ids & ({t.task_id for t in existing} | pool))
    report["id_collisions"] = id_collisions
    report["pool_note"] = pool_note
    print(f"ids       : {len(new_ids)} new ids; {pool_note}")
    if id_collisions:
        failures.append(f"task id collisions: {id_collisions}")
        print(f"            COLLISION {id_collisions}")
    else:
        print("            clean (registry validation also enforces uniqueness)")

    # ---- dedup ----------------------------------------------------------- #
    existing_hashes: dict[str, str] = {}
    existing_ast: dict[str, str] = {}
    existing_graph: dict[str, str] = {}
    existing_sources: list[tuple[str, str]] = []
    for task in existing:
        for name, text in _task_texts(task).items():
            key = f"{task.task_id}/{name}"
            existing_hashes.setdefault(content_hash(text), key)
            fingerprint = normalized_ast_fingerprint(text)
            if fingerprint:
                existing_ast.setdefault(fingerprint, key)
            graph = graph_fingerprint(text)
            if graph:
                existing_graph.setdefault(graph, key)
            if name.startswith("seed_"):
                existing_sources.append((key, text))

    dupes: list[dict] = []
    for task in new:
        for name, text in _task_texts(task).items():
            key = f"{task.task_id}/{name}"
            digest = content_hash(text)
            if digest in existing_hashes:
                dupes.append({"new": key, "kind": "exact_content",
                              "existing": existing_hashes[digest]})
            fingerprint = normalized_ast_fingerprint(text)
            if fingerprint and fingerprint in existing_ast:
                dupes.append({"new": key, "kind": "normalized_ast",
                              "existing": existing_ast[fingerprint]})
            graph = graph_fingerprint(text)
            if graph and graph in existing_graph:
                dupes.append({"new": key, "kind": "semantic_graph",
                              "existing": existing_graph[graph]})
            if not name.startswith("seed_"):
                continue
            for other_key, other in existing_sources:
                score = directional_containment(text, other).get("containment", 0.0)
                if score >= 0.78:
                    dupes.append({"new": key, "kind": "directional_containment",
                                  "existing": other_key, "score": round(score, 4)})
    content_dupes = [d for d in dupes if not _is_shim(d["new"].split("/", 1)[1])]
    shim_dupes = [d for d in dupes if _is_shim(d["new"].split("/", 1)[1])]
    report["duplicates"] = content_dupes
    report["shim_duplicates"] = len(shim_dupes)
    print(f"dedup     : {len(content_dupes)} duplicate finding(s) in task CONTENT "
          f"against {len(existing_hashes)} existing task files")
    if content_dupes:
        failures.append(f"{len(content_dupes)} duplicate task sources")
        for item in content_dupes[:10]:
            print(f"            {item}")
    else:
        print("            clean (exact SHA-256, normalized AST, semantic graph, "
              "directional containment)")
    print(f"            {len(shim_dupes)} finding(s) in generated driver/reference "
          f"shims, which are identical across ~1,300 tasks by construction")

    # ---- decontamination ------------------------------------------------- #
    holdout = build_heldout_ngrams()
    hits: list[dict] = []
    for task in new:
        for name, text in _task_texts(task).items():
            match = analyze_text_contamination(
                text, holdout, task_id=task.task_id,
                family="",  # let the text/lineage checks decide
            )
            if match is not None:
                hits.append({"task_file": f"{task.task_id}/{name}", **match.to_dict()})
    content_hits = [h for h in hits if not _is_shim(h["task_file"].split("/", 1)[1])]
    shim_hits = [h for h in hits if _is_shim(h["task_file"].split("/", 1)[1])]
    report["contamination"] = content_hits
    report["shim_contamination"] = len(shim_hits)
    print(f"decontam  : {len(content_hits)} hit(s) in task CONTENT against "
          f"{len(holdout.references)} held-out reference documents")
    if content_hits:
        failures.append(f"{len(content_hits)} contamination hits")
        for item in content_hits[:10]:
            print(f"            {item}")
    else:
        print("            clean")
    if shim_hits:
        # Worth stating plainly: the held-out index contains the generated driver
        # shims of held-out tasks, and their normalized AST is shared by every
        # generated task. So the shared gate CAN fire on boilerplate. It is
        # pre-existing and low-impact (training rows carry kernels, not shims), but
        # it means an `normalized_ast` hit on a shim is not evidence of leakage.
        print(f"            {len(shim_hits)} hit(s) on generated shims "
              f"(reason={sorted({h['reason'] for h in shim_hits})}): the held-out "
              "index contains held-out tasks' own shims, whose normalized AST every "
              "generated task shares. Boilerplate, not leakage.")

    # ---- operator overlap with the existing registry --------------------- #
    existing_ops = collections.defaultdict(list)
    for task in existing:
        key = _operator_key(task.operation)
        if key:
            existing_ops[key].append(task.task_id)
    overlap = {}
    for task in new:
        key = _operator_key(task.operation)
        if key and key in existing_ops:
            overlap[task.task_id] = {
                "operator": key,
                "n_existing_registry_tasks": len(existing_ops[key]),
                "example": sorted(existing_ops[key])[:3],
            }
    report["registry_operator_overlap"] = overlap
    print(f"operation : {len(overlap)}/{len(new)} new tasks share an operator with an "
          f"existing registry task")
    print("            EXPECTED: these are the same math in a different LANGUAGE, "
          "which is the axis being added.")
    print("            They are separate task ids with separate provenance roots, so "
          "they are not duplicates.")

    # ---- operator overlap with AgentKernelArena -------------------------- #
    aka = _aka_operators()
    if not aka:
        print(f"benchmark : AgentKernelArena not found at {AKA_ROOT}; overlap unknown")
        report["aka_overlap"] = {"available": False, "root": str(AKA_ROOT)}
    else:
        aka_keys: dict[str, set[str]] = {}
        for task_type, names in aka.items():
            keys = {k for k in (_operator_key(n) for n in names) if k}
            aka_keys[task_type] = keys
        shared = {}
        for task in new:
            key = _operator_key(task.operation)
            if not key:
                continue
            types = sorted(t for t, keys in aka_keys.items() if key in keys)
            if types:
                shared[task.task_id] = {"operator": key, "aka_task_types": types}
        report["aka_overlap"] = {
            "available": True,
            "root": str(AKA_ROOT),
            "aka_task_counts": {k: len(v) for k, v in sorted(aka.items())},
            "shared": shared,
        }
        print(f"benchmark : {len(shared)}/{len(new)} new tasks share an operator with an "
              f"AgentKernelArena task")
        by_type: collections.Counter = collections.Counter()
        for value in shared.values():
            for task_type in value["aka_task_types"]:
                by_type[task_type] += 1
        for task_type, count in sorted(by_type.items()):
            print(f"            {task_type}: {count} shared-operator task(s)")
        print("            DISCLOSURE: AKA is the benchmark used to compare against the")
        print("            published Opus bars (hip2hip 6.69x, torch2hip 6.89x). Training")
        print("            on tasks whose OPERATORS AKA also tests weakens that")
        print("            comparison even though no AKA source is copied. Decide")
        print("            deliberately; do not let it pass silently.")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")

    print()
    if failures:
        print("AUDIT FAILED: " + "; ".join(failures))
        return 1
    print("AUDIT PASSED: ids, dedup and decontamination are clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
