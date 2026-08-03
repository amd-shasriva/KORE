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
WORKER_ADDRESS_SPACE_BYTES = 8 * 1024 ** 3


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
