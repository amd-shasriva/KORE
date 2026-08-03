#!/usr/bin/env python
"""Build the external task pool: ingest, decontaminate, dedup, report.

The registry is 1,334 tasks, 1,289 of them trainable, and that count is what
bounds how much non-redundant training data datagen can produce. This script
raises the trainable task count by mining PyTorch modules from outside the
registry and turning each one into a KORE task, WITHOUT touching the registry --
see ``kore/tasks/external.py`` for why growing the registry itself would
invalidate in-flight split manifests.

Two sources:

  kernelbook  ``GPUMODE/KernelBook`` at a pinned revision. ~18k standalone
              ``torch.nn`` modules mined from permissively-licensed GitHub. The
              dataset also ships Inductor-generated Triton, which is NVIDIA
              -targeted and is deliberately NOT ingested -- only the PyTorch
              reference side, which is what defines a task.
  synthetic   Deterministic operator-composition synthesis following
              ``meta-pytorch/popcorn-kernels``' architecture. Its released
              synthetic dataset is gated and its generator needs an LLM API
              credential, so the LLM step is replaced by template composition.

Decontamination is not optional and not local to this script: it is
``kore.data.decontam`` applied against the held-out KORE task sources AND the
KernelBench evaluation problems, because a module mined from GitHub can be a
KernelBench problem arriving through a different corpus. Missing benchmark
references abort the run rather than silently weakening the gate.

Exit status is 0 only when the pool was written and non-empty, so this can gate
a datagen launch.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

# Pinned upstream revisions. A mutable ref would make the corpus, its
# decontamination evidence, and its task IDs all unreproducible.
KERNELBOOK_DATASET = "GPUMODE/KernelBook"
KERNELBOOK_REVISION = "b76504d85f7f14ef4b1fad81f136f638f2ce625b"
KERNELBENCH_DATASET = "ScalingIntelligence/KernelBench"
KERNELBENCH_REVISION = "ca1464e5d4723fd14f87b8e68306aaa1fe66b81b"
KERNELBENCH_SPLITS = ("level_1", "level_2", "level_3", "level_4")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ingest external PyTorch modules into the KORE task pool.",
    )
    parser.add_argument("--source", action="append", default=None,
                        choices=["kernelbook", "synthetic"],
                        help="source to ingest (repeatable; default: both)")
    parser.add_argument("--out", default="",
                        help="pool root (default: KORE_TASK_POOL or data/task_pool)")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap candidates per source (0 = all)")
    parser.add_argument("--synthetic-count", type=int, default=8000,
                        help="modules to synthesize")
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument("--dtype", default="fp32",
                        help="task dtype; the oracle is fp32 regardless")
    parser.add_argument("--workers", type=int, default=0,
                        help="screening processes (0 = os.cpu_count())")
    parser.add_argument("--timeout", type=int, default=8,
                        help="per-forward wall-clock budget, seconds")
    parser.add_argument("--materialize", action="store_true",
                        help="also write the task dirs, not just the index")
    parser.add_argument("--allow-missing-benchmark", action="store_true",
                        help="proceed without the KernelBench decontam references")
    parser.add_argument("--dry-run", action="store_true",
                        help="screen and report, write nothing")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--rescreen-index", action="store_true",
                        help="re-apply the screening gates to an EXISTING pool.jsonl "
                             "and drop what no longer passes, instead of ingesting "
                             "upstream corpora (no network, no re-download)")
    parser.add_argument("--prune-tasks", action="store_true",
                        help="with --rescreen-index, also delete the materialized "
                             "task dirs of dropped specs")
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def load_kernelbook(limit: int):
    """Yield KernelBook rows as candidates, PyTorch side only."""
    from datasets import load_dataset

    from kore.data.task_mining import Candidate

    dataset = load_dataset(
        KERNELBOOK_DATASET, split="train", revision=KERNELBOOK_REVISION
    )
    out = []
    for index, row in enumerate(dataset):
        if limit and len(out) >= limit:
            break
        source = row.get("python_code") or ""
        entry = str(row.get("entry_point") or "").strip()
        if not source.strip() or not entry:
            continue
        out.append(Candidate(
            source_id="kernelbook",
            row_id=str(row.get("uuid") or index),
            module_name=str(row.get("module_name") or entry),
            entry_class=entry,
            module_source=source,
            metadata={
                "dataset": KERNELBOOK_DATASET,
                "revision": KERNELBOOK_REVISION,
                "repo_name": row.get("repo_name"),
                "repo_link": row.get("repo_link"),
                "sha": row.get("sha"),
                "licenses": row.get("licenses"),
                "stars": row.get("stars"),
            },
        ))
    return out


def load_synthetic(count: int, seed: int, limit: int):
    from kore.data.task_mining import Candidate
    from kore.tasks.synth_modules import synthesize

    out = []
    for regime, name, source in synthesize(count, seed=seed):
        if limit and len(out) >= limit:
            break
        out.append(Candidate(
            source_id="synthetic",
            row_id=name,
            module_name=name,
            entry_class=name,
            module_source=source,
            metadata={"dataset": "kore.tasks.synth_modules", "revision": str(seed),
                      "regime": regime,
                      "method": "deterministic_operator_composition"},
        ))
    return out


def load_benchmark_references(allow_missing: bool):
    """KernelBench problems, as decontamination reference documents."""
    from kore.data.task_mining import benchmark_references

    problems = []
    try:
        from datasets import load_dataset

        for split in KERNELBENCH_SPLITS:
            data = load_dataset(
                KERNELBENCH_DATASET, split=split, revision=KERNELBENCH_REVISION
            )
            problems.extend(dict(row) for row in data)
    except Exception as exc:  # noqa: BLE001
        if not allow_missing:
            raise SystemExit(
                f"KernelBench decontamination references unavailable ({exc}). "
                "A mined module can BE a benchmark problem, so this gate is not "
                "optional; pass --allow-missing-benchmark only for a labelled "
                "offline build."
            )
        log(f"WARNING: KernelBench references unavailable ({exc}); gate weakened")
        return [], False
    return benchmark_references(problems), True


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
_SCREEN_DTYPE = "fp32"
_SCREEN_TIMEOUT = 8

#: Address-space ceiling for a screening worker. Mined modules allocate whatever
#: their author wrote, and an unbounded allocation gets the worker killed by the
#: OOM reaper -- which Python cannot catch, so it takes the whole pool down. A
#: hard rlimit turns that same allocation into a MemoryError the worker reports
#: as an ordinary drop.
#
# 24 GiB, not the 8 GiB that looks sufficient for a 1M-element probe: RLIMIT_AS
# bounds VIRTUAL address space, and torch reserves arenas and thread stacks far
# beyond its resident set. Measured on 400 KernelBook rows, an 8 GiB cap could
# not even import scipy inside the worker and cost 7 otherwise-admissible tasks;
# 24 GiB reproduces the unlimited-baseline acceptance exactly.
WORKER_ADDRESS_SPACE_BYTES = 24 * 1024 ** 3


def _init_worker(dtype: str, timeout: int) -> None:
    global _SCREEN_DTYPE, _SCREEN_TIMEOUT
    _SCREEN_DTYPE, _SCREEN_TIMEOUT = dtype, timeout
    import warnings

    warnings.filterwarnings("ignore")
    import torch

    # One thread per worker: the pool is already saturating the node, and nested
    # threading turns a 64-way pool into heavy oversubscription.
    torch.set_num_threads(1)
    try:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS,
            (WORKER_ADDRESS_SPACE_BYTES, WORKER_ADDRESS_SPACE_BYTES),
        )
    except Exception:  # noqa: BLE001 - platform without rlimit; timeouts still apply
        pass


def _screen_one(item):
    from kore.data.task_mining import Outcome, screen_candidate

    index, candidate = item
    try:
        return index, screen_candidate(candidate, _SCREEN_DTYPE, _SCREEN_TIMEOUT)
    except BaseException as exc:  # noqa: BLE001 - a worker must never die silently
        return index, Outcome(candidate, False, "execution_failed",
                              f"{type(exc).__name__}: {exc}")


def screen_all(candidates, dtype: str, workers: int, timeout: int):
    """Run the pure gates across a process pool, preserving input order.

    Resilient to a worker dying outright. A mined module can still take a worker
    down below Python (a native crash in a torch kernel), which breaks the whole
    executor and fails every in-flight future. Rather than losing the run, each
    round keeps whatever completed, rebuilds the pool for the remainder, and --
    when a round completes nothing at all -- blames the head candidate and
    records it as a crash, which guarantees progress.
    """
    from kore.data.task_mining import Outcome, screen_candidate

    if workers <= 1:
        return [screen_candidate(c, dtype, timeout) for c in candidates]

    import concurrent.futures as futures

    results: dict[int, object] = {}
    pending = list(enumerate(candidates))
    total = len(pending)
    crashes = 0

    while pending:
        completed = 0
        try:
            with futures.ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker,
                initargs=(dtype, timeout),
            ) as pool:
                for index, outcome in pool.map(_screen_one, pending, chunksize=8):
                    results[index] = outcome
                    completed += 1
                    if len(results) % 2000 == 0:
                        log(f"  screened {len(results)}/{total}")
        except futures.process.BrokenProcessPool:
            log(f"  worker pool died after {completed} of {len(pending)}; rebuilding")
        except KeyboardInterrupt:
            raise
        pending = [(i, c) for i, c in pending if i not in results]
        if pending and completed == 0:
            index, candidate = pending[0]
            results[index] = Outcome(
                candidate, False, "execution_crashed",
                "took the screening worker down below Python",
            )
            crashes += 1
            pending = pending[1:]
    if crashes:
        log(f"  {crashes} candidate(s) crashed a worker and were dropped")
    return [results[i] for i in range(total)]


# --------------------------------------------------------------------------- #
# Re-screening an index that is already on disk
# --------------------------------------------------------------------------- #
def _rescreen_one(payload):
    """Verdict for one already-indexed spec. Runs in a screening worker."""
    index, raw = payload
    from kore.data.task_mining import hidden_state_reason
    from kore.tasks.external import (
        ExternalTaskSpec,
        exec_module_source,
    )

    try:
        spec = ExternalTaskSpec.from_dict(raw)
        namespace = exec_module_source(spec.module_source)
        reason = hidden_state_reason(
            namespace, spec.entry_class, spec.init_args, spec.init_kwargs,
            spec.input_specs, spec.dtype, _SCREEN_TIMEOUT,
        )
    except BaseException as exc:  # noqa: BLE001 - a worker must never die silently
        return index, "rescreen_failed", f"{type(exc).__name__}: {exc}"
    if reason is not None:
        return index, "hidden_state_oracle", reason
    return index, "", ""


def rescreen_index(root, workers: int, timeout: int, prune: bool, dry_run: bool):
    """Re-apply the pure gates to an existing ``pool.jsonl``.

    A full ingest needs the upstream corpora and hours of screening. When a gate
    is ADDED, the pool already on disk has to be brought under it without either
    of those, otherwise the practical choice is between re-downloading 18k modules
    and shipping a pool the new gate would have refused. This path reads the
    index, re-derives each verdict from the spec itself (the spec carries the
    module source and the constructor arguments, so nothing upstream is needed),
    and rewrites the index with only what still passes.
    """
    import concurrent.futures as futures

    from kore.tasks import external

    index_path = root / external.POOL_INDEX_NAME
    if not index_path.is_file():
        log(f"no index at {index_path}")
        return None

    raws = []
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                raws.append(json.loads(line))
    log(f"re-screening {len(raws)} indexed specs from {index_path}")

    verdicts: dict[int, tuple[str, str]] = {}
    pending = list(enumerate(raws))
    while pending:
        completed = 0
        try:
            with futures.ProcessPoolExecutor(
                max_workers=max(1, workers), initializer=_init_worker,
                initargs=("fp32", timeout),
            ) as pool:
                for index, reason, detail in pool.map(
                    _rescreen_one, pending, chunksize=8
                ):
                    verdicts[index] = (reason, detail)
                    completed += 1
                    if len(verdicts) % 2000 == 0:
                        log(f"  re-screened {len(verdicts)}/{len(raws)}")
        except futures.process.BrokenProcessPool:
            log(f"  worker pool died after {completed} of {len(pending)}; rebuilding")
        pending = [(i, r) for i, r in pending if i not in verdicts]
        if pending and completed == 0:
            index, _ = pending[0]
            verdicts[index] = ("rescreen_crashed", "took the worker down below Python")
            pending = pending[1:]

    kept = [raw for i, raw in enumerate(raws) if not verdicts[i][0]]
    drops = {}
    for reason, _ in verdicts.values():
        if reason:
            drops[reason] = drops.get(reason, 0) + 1

    print()
    print(f"indexed         : {len(raws):,}")
    print(f"still admissible: {len(kept):,}")
    print(f"dropped         : {len(raws) - len(kept):,}")
    for reason, count in sorted(drops.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>7,}  {reason}")
    families = {}
    for raw in kept:
        families[raw.get("family", "?")] = families.get(raw.get("family", "?"), 0) + 1
    print(f"families kept   : {dict(sorted(families.items()))}")

    result = {
        "mode": "rescreen_index",
        "indexed": len(raws),
        "kept": len(kept),
        "dropped": len(raws) - len(kept),
        "drops_by_reason": dict(sorted(drops.items())),
        "families_kept": dict(sorted(families.items())),
        "examples": [
            {"task_id": raws[i]["task_id"], "reason": v[0], "detail": v[1][:300]}
            for i, v in sorted(verdicts.items()) if v[0]
        ][:20],
    }
    if dry_run:
        log("dry run: index unchanged")
        return result

    # Keep the pre-filter index next to the new one: the drop is large and
    # irreversible from the filtered file alone, and the campaign's provenance
    # should be able to name exactly what was removed.
    backup = index_path.with_suffix(".jsonl.prefilter")
    if not backup.exists():
        index_path.replace(backup)
    else:
        log(f"prefilter backup already exists at {backup.name}; not overwriting")
    with index_path.open("w", encoding="utf-8") as handle:
        for raw in kept:
            handle.write(json.dumps(raw, sort_keys=True) + "\n")
    log(f"wrote {len(kept)} specs to {index_path} (was {len(raws)})")

    if prune:
        import shutil

        tasks_dir = external.pool_tasks_dir(root)
        keep_ids = {raw["task_id"] for raw in kept}
        removed = 0
        if tasks_dir.is_dir():
            for child in sorted(tasks_dir.iterdir()):
                if child.is_dir() and child.name not in keep_ids:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
        log(f"pruned {removed} materialized task dirs")
        result["pruned_task_dirs"] = removed

    manifest_path = root / external.POOL_MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("rescreens", []).append({
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gate": "kore.data.task_mining.hidden_state_reason",
            **{k: result[k] for k in
               ("indexed", "kept", "dropped", "drops_by_reason")},
        })
        manifest["pool_tasks"] = len(kept)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
        log(f"recorded the re-screen in {manifest_path.name}")
    return result


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

    from kore.data.task_mining import (
        Decontaminator,
        Deduplicator,
        MiningReport,
        admit,
        registry_task_sources,
    )
    from kore.tasks import external
    from kore.tasks.registry import all_tasks, taxonomy_digest, train_tasks

    sources = args.source or ["kernelbook", "synthetic"]
    root = external.pool_root(args.out or None)
    workers = args.workers or (os.cpu_count() or 8)
    started = time.time()

    if args.rescreen_index:
        result = rescreen_index(
            root, workers, args.timeout, args.prune_tasks, args.dry_run
        )
        if result is None:
            return 2
        if args.json_out:
            pathlib.Path(args.json_out).write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
        return 0 if result["kept"] else 1

    registry_total = len(all_tasks())
    registry_train = len(train_tasks())
    log(f"registry before: {registry_total} tasks, {registry_train} trainable")
    log(f"pool root      : {root}")
    log(f"sources        : {', '.join(sources)}  workers={workers}")

    candidates = []
    for source in sources:
        if source == "kernelbook":
            log(f"loading {KERNELBOOK_DATASET}@{KERNELBOOK_REVISION[:8]}")
            rows = load_kernelbook(args.limit)
        else:
            log(f"synthesizing {args.synthetic_count} modules (seed {args.synthetic_seed})")
            rows = load_synthetic(args.synthetic_count, args.synthetic_seed, args.limit)
        log(f"  {source}: {len(rows)} candidates")
        candidates.extend(rows)
    if not candidates:
        log("no candidates loaded")
        return 2

    references, benchmark_present = load_benchmark_references(
        args.allow_missing_benchmark
    )
    log(f"benchmark references: {len(references)} (present={benchmark_present})")

    log(f"screening {len(candidates)} candidates")
    screened = screen_all(candidates, args.dtype, workers, args.timeout)
    log(f"  screened in {time.time() - started:.0f}s")

    decontaminator = Decontaminator(extra_references=references)
    log(f"held-out gate: {len(decontaminator.heldout_task_ids)} task ids, "
        f"{len(decontaminator.heldout_families)} families, "
        f"{decontaminator.n_references} reference documents")
    deduplicator = Deduplicator(registry_task_sources())
    log(f"dedup seeded with {deduplicator.n_registry_fingerprints} registry sources")

    accepted, report = admit(screened, decontaminator, deduplicator, MiningReport())

    manifest = {
        "schema_version": external.POOL_SCHEMA_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dtype": args.dtype,
        "sources": {
            "kernelbook": {"dataset": KERNELBOOK_DATASET,
                           "revision": KERNELBOOK_REVISION},
            "synthetic": {"generator": "kore.tasks.synth_modules.synthesize",
                          "seed": args.synthetic_seed,
                          "requested": args.synthetic_count},
        },
        "decontamination": {
            "detector": "kore.data.decontam.analyze_text_contamination",
            "heldout_task_ids": len(decontaminator.heldout_task_ids),
            "heldout_families": sorted(decontaminator.heldout_families),
            "reference_documents": decontaminator.n_references,
            "benchmark_dataset": KERNELBENCH_DATASET,
            "benchmark_revision": KERNELBENCH_REVISION,
            "benchmark_references_present": benchmark_present,
        },
        "dedup": {
            "fingerprints": ["structural_fingerprint", "graph_fingerprint"],
            "registry_sources_seeded": deduplicator.n_registry_fingerprints,
        },
        "registry_at_build": {
            "tasks": registry_total,
            "train_tasks": registry_train,
            "taxonomy_digest": taxonomy_digest(),
        },
        "result": report.as_dict(),
        "pool_tasks": len(accepted),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    print()
    print(f"considered      : {report.considered:,}")
    print(f"accepted        : {report.accepted:,}")
    print(f"dropped         : {report.considered - report.accepted:,}")
    for reason, count in sorted(report.drops.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>7,}  {reason}")
    print(f"families        : {dict(sorted(report.families.items()))}")
    print(f"registry train  : {registry_train:,}")
    print(f"pool train      : {len(accepted):,}")
    print(f"combined train  : {registry_train + len(accepted):,}")

    if args.dry_run:
        log("dry run: nothing written")
    else:
        root.mkdir(parents=True, exist_ok=True)
        written = external.write_pool_index(accepted, root / external.POOL_INDEX_NAME)
        (root / external.POOL_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        log(f"wrote {written} specs to {root / external.POOL_INDEX_NAME}")
        if args.materialize:
            ids = external.materialize_pool(root, accepted)
            log(f"materialized {len(ids)} task dirs under {external.pool_tasks_dir(root)}")

    if args.json_out:
        pathlib.Path(args.json_out).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
