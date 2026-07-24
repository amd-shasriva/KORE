"""Parallel datagen with durable, exclusively owned task shards.

The campaign's datagen/agentic stages are embarrassingly parallel across tasks (each
task = independent teacher calls + GPU verification), but run sequentially by default.
On an N-GPU box that is an N-fold waste of both the GPU verifier AND the teacher
gateway latency (each teacher call is mostly network wait, during which the GPU is idle).

This module runs the SAME datagen work across ``n_workers`` processes, each pinned to
a distinct GPU (``HIP_VISIBLE_DEVICES``) with its OWN teacher stream. A shard is
resumable only when strict record validation and a completion receipt both pass.
Per-shard advisory locks give one process/campaign exclusive ownership while it
generates, so overlapping campaigns cannot overwrite one another's valid work.

No pipeline shortcut: identical generators (``generate_repairs`` / ``generate_groups``
/ ``generate_wins`` / ``generate_agentic_trajectories``) at identical counts - only the
scheduling is parallelized. Spawn start-method (never fork) so each worker gets a clean
interpreter and pins its GPU BEFORE importing torch.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from kore.data.schemas import (
    RECORD_SCHEMA_VERSION,
    _commit_prepared_jsonl,
    _durable_replace,
    _prepare_jsonl,
    atomic_write_json,
    validate_jsonl_shard,
)

DATAGEN_KINDS = ("repair", "groups", "wins")
AGENTIC_KINDS = ("agentic",)
ALL_KINDS = frozenset((*DATAGEN_KINDS, *AGENTIC_KINDS))

RECEIPT_VERSION = 1
_RECORD_TYPE_BY_KIND = {
    "repair": "repair",
    "groups": "ranked_group",
    "wins": "win",
    "agentic": "agentic",
}
_GENERATOR_BY_KIND = {
    "repair": "kore.data.gen_repair.generate_repairs",
    "groups": "kore.data.gen_groups.generate_groups",
    "wins": "kore.data.gen_wins.generate_wins",
    "agentic": "kore.data.gen_agentic.generate_agentic_trajectories",
}
_COUNT_KEYS_BY_KIND = {
    "repair": ("n_repair",),
    "groups": ("n_parents", "k"),
    "wins": ("wins_gens",),
    "agentic": ("n_agentic", "max_tool_turns"),
}
_FIXED_PARAMETERS_BY_KIND = {
    "repair": {"seed": 0, "natural_fraction": 0.3, "diagnostic": True},
    "groups": {"seed": 0},
    "wins": {"include_regression_lesson": True},
    "agentic": {"keep_only_useful": True, "thinking": True},
}

_RESULT_POLL_SECONDS = 0.2
_WORKER_EXIT_GRACE_SECONDS = 1.0
_WORKER_JOIN_SECONDS = 5.0
_TASK_QUEUE_IDLE_SECONDS = 1.0

_LOCAL_LOCKS_GUARD = threading.Lock()
_LOCAL_LOCKS: dict[str, threading.Lock] = {}


class ShardCompletionError(RuntimeError):
    """A shard or its completion receipt fails admission."""


class ShardContractConflict(ShardCompletionError):
    """A valid shard exists, but was generated under another contract."""


class DatagenRunError(RuntimeError):
    """One or more datagen workers failed or violated the result protocol."""

    def __init__(self, message: str, *, summary: dict | None = None):
        super().__init__(message)
        self.summary = dict(summary or {})


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable)
# --------------------------------------------------------------------------- #
def shard_tasks(task_ids: list[str], n_workers: int) -> list[list[str]]:
    """Round-robin shard task ids across ``n_workers`` (balanced, deterministic)."""
    n_workers = max(1, int(n_workers))
    shards: list[list[str]] = [[] for _ in range(n_workers)]
    for i, tid in enumerate(task_ids):
        shards[i % n_workers].append(tid)
    return [s for s in shards if s]


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not finite canonical JSON: {exc}") from exc


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _count_parameter(counts: dict, key: str, *, positive: bool = False) -> int:
    if key not in counts:
        raise ValueError(f"missing datagen count {key!r}")
    value = counts[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"datagen count {key!r} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        raise ValueError(f"datagen count {key!r} must be >= {minimum}")
    return value


def _validate_generator_contract(contract: Any, kind: str) -> dict:
    if not isinstance(contract, dict):
        raise ShardCompletionError("generator contract must be an object")
    _canonical_json(contract)
    for key, expected in (
        ("contract_version", 1),
        ("generator_revision", 1),
        ("record_schema_version", RECORD_SCHEMA_VERSION),
    ):
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ShardCompletionError(
                f"generator contract {key} expected {expected}, got {value!r}")
    if contract.get("kind") != kind:
        raise ShardCompletionError(
            f"generator contract kind expected {kind!r}, "
            f"got {contract.get('kind')!r}")
    if contract.get("generator") != _GENERATOR_BY_KIND[kind]:
        raise ShardCompletionError(
            f"generator contract has unknown generator "
            f"{contract.get('generator')!r}")
    parameters = contract.get("parameters")
    if not isinstance(parameters, dict):
        raise ShardCompletionError("generator contract parameters must be an object")
    expected_parameter_keys = (
        set(_COUNT_KEYS_BY_KIND[kind])
        | set(_FIXED_PARAMETERS_BY_KIND[kind])
    )
    if set(parameters) != expected_parameter_keys:
        raise ShardCompletionError(
            "generator contract parameter keys do not match the generator")
    for key in _COUNT_KEYS_BY_KIND[kind]:
        value = parameters[key]
        minimum = 1 if key == "k" else 0
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise ShardCompletionError(
                f"generator contract parameter {key!r} must be >= {minimum}")
    for key, expected in _FIXED_PARAMETERS_BY_KIND[kind].items():
        if parameters[key] != expected or type(parameters[key]) is not type(expected):
            raise ShardCompletionError(
                f"generator contract parameter {key!r} expected {expected!r}")
    teacher = contract.get("teacher")
    if not isinstance(teacher, dict):
        raise ShardCompletionError("generator contract teacher must be an object")
    teacher_kind = teacher.get("kind")
    if not isinstance(teacher_kind, str) or not teacher_kind.strip():
        raise ShardCompletionError("generator contract teacher kind must be non-empty")
    model = teacher.get("model")
    if model is not None and not isinstance(model, str):
        raise ShardCompletionError(
            "generator contract teacher model must be a string or null")
    if teacher.get("resilient") is not True:
        raise ShardCompletionError(
            "generator contract must use resilient teacher calls")
    flags = contract.get("behavior_flags")
    if not isinstance(flags, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in flags.items()
    ):
        raise ShardCompletionError(
            "generator contract behavior_flags must map strings to strings")
    return contract


def build_generator_contract(
    kind: str,
    counts: dict,
    *,
    teacher_kind: str,
    model_teacher: Any = None,
) -> dict:
    """Build the generation inputs that make a completed shard reusable."""
    if kind not in ALL_KINDS:
        raise ValueError(f"unknown datagen kind {kind!r}")
    if not isinstance(counts, dict):
        raise TypeError("counts must be a dict")
    if not isinstance(teacher_kind, str) or not teacher_kind.strip():
        raise ValueError("teacher_kind must be a non-empty string")
    parameters = {
        key: _count_parameter(counts, key, positive=(key == "k"))
        for key in _COUNT_KEYS_BY_KIND[kind]
    }
    parameters.update(_FIXED_PARAMETERS_BY_KIND[kind])
    behavior_flags: dict[str, str] = {}
    if kind == "groups":
        behavior_flags["KORE_GROUND_REASONING"] = os.environ.get(
            "KORE_GROUND_REASONING", "0")
    if kind == "wins":
        behavior_flags["KORE_WINS_PMC"] = os.environ.get("KORE_WINS_PMC", "1")
        behavior_flags["KORE_WINS_PMC_MAX"] = os.environ.get("KORE_WINS_PMC_MAX", "4")
    contract = {
        "contract_version": 1,
        "generator_revision": 1,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "kind": kind,
        "generator": _GENERATOR_BY_KIND[kind],
        "parameters": parameters,
        "teacher": {
            "kind": str(teacher_kind),
            "model": None if model_teacher is None else str(model_teacher),
            "resilient": True,
        },
        "behavior_flags": behavior_flags,
    }
    return _validate_generator_contract(contract, kind)


def shard_path(data_root: Any, task_id: str, kind: str) -> Path:
    return Path(data_root) / kind / f"{task_id}.jsonl"


def shard_receipt_path(path: Any) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.complete.json")


def _pending_receipt_path(path: Any) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.complete.pending.json")


def shard_lock_path(path: Any) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.lock")


def _load_json_object(path: Path) -> dict:
    def reject_constant(token: str):
        raise ValueError(f"non-finite JSON constant {token!r}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ShardCompletionError(f"{path}: receipt must be a JSON object")
    _canonical_json(value)
    return value


def _validate_receipt(
    path: Path,
    receipt_path: Path,
    *,
    task_id: str,
    kind: str,
    contract: dict | None,
) -> dict:
    if kind not in _RECORD_TYPE_BY_KIND:
        raise ShardCompletionError(f"unknown shard kind {kind!r}")
    receipt = _load_json_object(receipt_path)
    receipt_version = receipt.get("receipt_version")
    if (
        isinstance(receipt_version, bool)
        or not isinstance(receipt_version, int)
        or receipt_version != RECEIPT_VERSION
    ):
        raise ShardCompletionError(
            f"{receipt_path}: unsupported receipt version "
            f"{receipt_version!r}")
    schema_version = receipt.get("record_schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != RECORD_SCHEMA_VERSION
    ):
        raise ShardCompletionError(
            f"{receipt_path}: record schema version mismatch")
    bindings = {
        "data_file": path.name,
        "task_id": task_id,
        "kind": kind,
        "record_type": _RECORD_TYPE_BY_KIND[kind],
    }
    for key, expected in bindings.items():
        if receipt.get(key) != expected:
            raise ShardCompletionError(
                f"{receipt_path}: {key} expected {expected!r}, "
                f"got {receipt.get(key)!r}")
    count = receipt.get("record_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ShardCompletionError(f"{receipt_path}: invalid record_count")
    digest = receipt.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ShardCompletionError(f"{receipt_path}: invalid sha256")
    stored_contract = receipt.get("generator_contract")
    _validate_generator_contract(stored_contract, kind)
    stored_contract_digest = _canonical_digest(stored_contract)
    if receipt.get("generator_contract_sha256") != stored_contract_digest:
        raise ShardCompletionError(
            f"{receipt_path}: generator contract digest mismatch")
    if contract is not None and _canonical_json(stored_contract) != _canonical_json(contract):
        raise ShardContractConflict(
            f"{path}: completed under a different generator contract")

    validation = validate_jsonl_shard(
        path,
        expected_task_id=task_id,
        expected_type=_RECORD_TYPE_BY_KIND[kind],
    )
    if validation.record_count != count:
        raise ShardCompletionError(
            f"{path}: receipt count {count} != validated count "
            f"{validation.record_count}")
    if validation.sha256 != digest:
        raise ShardCompletionError(f"{path}: receipt data digest mismatch")
    return receipt


def validate_completed_shard(
    data_root: Any,
    task_id: str,
    kind: str,
    *,
    contract: dict | None = None,
) -> dict:
    """Fail closed unless data and its final completion receipt agree."""
    path = shard_path(data_root, task_id, kind)
    return _validate_receipt(
        path,
        shard_receipt_path(path),
        task_id=task_id,
        kind=kind,
        contract=contract,
    )


def _recover_pending_receipt(
    path: Path,
    *,
    task_id: str,
    kind: str,
    contract: dict | None,
) -> bool:
    """Promote a pre-commit receipt after a crash between the two replaces."""
    pending = _pending_receipt_path(path)
    if not pending.exists():
        return False
    try:
        _validate_receipt(
            path,
            pending,
            task_id=task_id,
            kind=kind,
            contract=contract,
        )
    except (OSError, ValueError, ShardCompletionError):
        return False
    final = shard_receipt_path(path)
    try:
        _durable_replace(pending, final)
    except FileNotFoundError:
        # A concurrent checker may have promoted it first.
        pass
    _validate_receipt(
        path,
        final,
        task_id=task_id,
        kind=kind,
        contract=contract,
    )
    return True


def shard_done(
    data_root: Any,
    task_id: str,
    kind: str,
    *,
    contract: dict | None = None,
) -> bool:
    """True only for a strict shard with a matching durable completion receipt."""
    path = shard_path(data_root, task_id, kind)
    try:
        _validate_receipt(
            path,
            shard_receipt_path(path),
            task_id=task_id,
            kind=kind,
            contract=contract,
        )
        return True
    except (OSError, ValueError, ShardCompletionError):
        try:
            return _recover_pending_receipt(
                path,
                task_id=task_id,
                kind=kind,
                contract=contract,
            )
        except (OSError, ValueError, ShardCompletionError):
            return False


def _local_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCAL_LOCKS_GUARD:
        return _LOCAL_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _exclusive_shard_lock(path: Path) -> Iterator[None]:
    """Serialize both threads and processes on a stable per-shard lock file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = shard_lock_path(path)
    local = _local_lock(lock_path)
    with local:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


@contextmanager
def claim_shard(
    data_root: Any,
    task_id: str,
    kind: str,
    *,
    contract: dict,
) -> Iterator[bool]:
    """Hold exclusive ownership; yield False for an already-complete shard.

    A valid shard under a different contract is a hard conflict, never silently
    replaced. Invalid/legacy/partial files may be regenerated atomically.
    """
    path = shard_path(data_root, task_id, kind)
    with _exclusive_shard_lock(path):
        if shard_done(data_root, task_id, kind, contract=contract):
            yield False
            return
        if shard_done(data_root, task_id, kind):
            raise ShardContractConflict(
                f"{path}: refusing to replace valid work from another contract")
        yield True


def detect_gpus(default: int = 8) -> int:
    """GPU count WITHOUT initializing CUDA in this process (clean for spawn).

    Runs ``torch.cuda.device_count()`` in a throwaway subprocess; falls back to
    ``default`` if torch/ROCm is unavailable."""
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import torch;print(torch.cuda.device_count())"],
            capture_output=True, text=True, timeout=120)
        n = int((out.stdout or "").strip().splitlines()[-1])
        return n if n > 0 else default
    except Exception:  # noqa: BLE001
        return default


def _completion_receipt(
    path: Path,
    *,
    task_id: str,
    kind: str,
    record_count: int,
    digest: str,
    contract: dict,
) -> dict:
    return {
        "receipt_version": RECEIPT_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "data_file": path.name,
        "task_id": task_id,
        "kind": kind,
        "record_type": _RECORD_TYPE_BY_KIND[kind],
        "record_count": record_count,
        "sha256": digest,
        "generator_contract": contract,
        "generator_contract_sha256": _canonical_digest(contract),
    }


def write_completed_shard(
    data_root: Any,
    task_id: str,
    kind: str,
    records: Iterable[Any],
    *,
    contract: dict,
) -> int:
    """Publish strict data first and its receipt last.

    The pending receipt is durable before the data replace. If the process dies
    after publishing data but before publishing the final receipt, the next
    resume check verifies and promotes that pending receipt without regenerating
    valid work. Callers must hold ``claim_shard`` for the entire operation.
    """
    if kind not in _RECORD_TYPE_BY_KIND:
        raise ValueError(f"unknown datagen kind {kind!r}")
    path = shard_path(data_root, task_id, kind)
    prepared = _prepare_jsonl(
        path,
        records,
        validate_records=True,
        expected_task_id=task_id,
        expected_type=_RECORD_TYPE_BY_KIND[kind],
    )
    pending = _pending_receipt_path(path)
    data_published = False
    pending_written = False
    try:
        receipt = _completion_receipt(
            path,
            task_id=task_id,
            kind=kind,
            record_count=prepared.record_count,
            digest=prepared.sha256,
            contract=contract,
        )
        atomic_write_json(pending, receipt)
        pending_written = True
        _commit_prepared_jsonl(prepared)
        data_published = True
        _validate_receipt(
            path,
            pending,
            task_id=task_id,
            kind=kind,
            contract=contract,
        )
        _durable_replace(pending, shard_receipt_path(path))
        validate_completed_shard(
            data_root,
            task_id,
            kind,
            contract=contract,
        )
        return prepared.record_count
    finally:
        try:
            prepared.temp_path.unlink()
        except FileNotFoundError:
            pass
        if pending_written and not data_published:
            try:
                pending.unlink()
            except FileNotFoundError:
                pass


# --------------------------------------------------------------------------- #
# Worker (runs in a spawned process, pinned to one GPU)
# --------------------------------------------------------------------------- #
def _generate(kind: str, task, teacher, env, counts: dict):
    if kind == "repair":
        from kore.data.gen_repair import generate_repairs
        return generate_repairs(task, teacher, env, n=counts["n_repair"])
    if kind == "groups":
        from kore.data.gen_groups import generate_groups
        return generate_groups(task, teacher, env, n_parents=counts["n_parents"], k=counts["k"])
    if kind == "wins":
        from kore.data.gen_wins import generate_wins
        return generate_wins(task, teacher, env, gens=counts["wins_gens"])
    if kind == "agentic":
        from kore.data.gen_agentic import generate_agentic_trajectories
        recs = generate_agentic_trajectories(
            task, teacher, env, n=counts["n_agentic"],
            max_turns=counts["max_tool_turns"], keep_only_useful=True)
        return [r.to_dict() for r in recs]
    raise ValueError(f"unknown datagen kind {kind!r}")


def _run_owned_shard(
    kind: str,
    task_id: str,
    task: Any,
    teacher: Any,
    env: Any,
    *,
    data_root: Any,
    counts: dict,
    teacher_kind: str,
    model_teacher: Any,
) -> tuple[str, int]:
    contract = build_generator_contract(
        kind,
        counts,
        teacher_kind=teacher_kind,
        model_teacher=model_teacher,
    )
    with claim_shard(
        data_root,
        task_id,
        kind,
        contract=contract,
    ) as claimed:
        if not claimed:
            return "skip", 0
        records = _generate(kind, task, teacher, env, counts)
        count = write_completed_shard(
            data_root,
            task_id,
            kind,
            records,
            contract=contract,
        )
        return "done", count


def _worker(payload: dict) -> list[tuple]:
    """Sequential payload worker retained for direct CPU/unit invocation."""
    gpu = str(payload["gpu_id"])
    os.environ["HIP_VISIBLE_DEVICES"] = gpu
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    from kore.data.teacher import load_env_local, make_teacher
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    load_env_local()
    tkw = {"model": payload["model_teacher"]} if payload.get("model_teacher") else {}
    teacher = make_teacher(payload["teacher_kind"], resilient=True, **tkw)
    results: list[tuple] = []
    for task_id in dict.fromkeys(payload["task_ids"]):
        task = get_task(task_id)
        env = KoreEnv(task)
        for kind in dict.fromkeys(payload["kinds"]):
            status, count = _run_owned_shard(
                kind,
                task_id,
                task,
                teacher,
                env,
                data_root=payload["data_root"],
                counts=payload["counts"],
                teacher_kind=payload["teacher_kind"],
                model_teacher=payload.get("model_teacher"),
            )
            results.append((task_id, kind, status, count))
    return results


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _queue_worker(worker_id, gpu, task_q, result_q, kinds, data_root, counts,
                  teacher_kind, model_teacher):
    """Persistent GPU-pinned datagen worker: pull tasks from a shared queue until
    drained. DYNAMIC load balancing so no worker idles at the tail while a few grind
    the heavy shards. The teacher client + GPU pin are set up ONCE per worker."""
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    from kore.data.teacher import load_env_local, make_teacher
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    teacher = None
    try:
        while True:
            try:
                task_id = task_q.get(timeout=_TASK_QUEUE_IDLE_SECONDS)
            except queue.Empty:
                # A missing drain sentinel must not strand a worker forever. All
                # tasks are enqueued before workers start, so an idle queue is done.
                break
            if task_id is None:
                break
            task = None
            env = None
            for kind in kinds:
                try:
                    contract = build_generator_contract(
                        kind,
                        counts,
                        teacher_kind=teacher_kind,
                        model_teacher=model_teacher,
                    )
                    with claim_shard(
                        data_root,
                        task_id,
                        kind,
                        contract=contract,
                    ) as claimed:
                        if not claimed:
                            status, count = "skip", 0
                        else:
                            if teacher is None:
                                load_env_local()
                                tkw = {"model": model_teacher} if model_teacher else {}
                                teacher = make_teacher(
                                    teacher_kind, resilient=True, **tkw)
                            if task is None:
                                task = get_task(task_id)
                                env = KoreEnv(task)
                            records = _generate(
                                kind, task, teacher, env, counts)
                            count = write_completed_shard(
                                data_root,
                                task_id,
                                kind,
                                records,
                                contract=contract,
                            )
                            status = "done"
                    if status == "skip":
                        print(
                            f"[datagen w{gpu}] {task_id}:{kind} skip (resume)",
                            flush=True,
                        )
                    else:
                        print(
                            f"[datagen w{gpu}] {task_id}:{kind} -> {count} records",
                            flush=True,
                        )
                    result_q.put({
                        "event": "result",
                        "worker_id": worker_id,
                        "task_id": task_id,
                        "kind": kind,
                        "status": status,
                        "records": count,
                    })
                except Exception as exc:  # noqa: BLE001 - report then stop this worker
                    print(
                        f"[datagen w{gpu}] {task_id}:{kind} ERROR "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    result_q.put({
                        "event": "result",
                        "worker_id": worker_id,
                        "task_id": task_id,
                        "kind": kind,
                        "status": "error",
                        "records": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    return
    except BaseException as exc:
        # KeyboardInterrupt/SystemExit and worker-initialization failures must be
        # observable even when the process exits before producing a shard result.
        result_q.put({
            "event": "fatal",
            "worker_id": worker_id,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    finally:
        result_q.put({"event": "finished", "worker_id": worker_id})


def _abort_workers(procs: list) -> None:
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    for proc in procs:
        proc.join(timeout=_WORKER_JOIN_SECONDS)
    for proc in procs:
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
    for proc in procs:
        if proc.is_alive():
            proc.join(timeout=_WORKER_JOIN_SECONDS)


def _join_workers(procs: list, summary: dict) -> None:
    for proc in procs:
        proc.join(timeout=_WORKER_JOIN_SECONDS)
    stuck = [index for index, proc in enumerate(procs) if proc.is_alive()]
    failed = [
        (index, proc.exitcode)
        for index, proc in enumerate(procs)
        if proc.exitcode not in (0, None)
    ]
    if stuck or failed:
        _abort_workers(procs)
        raise DatagenRunError(
            f"datagen workers did not exit cleanly; stuck={stuck}, failed={failed}",
            summary=summary,
        )


def _collect_worker_results(
    procs: list,
    result_q: Any,
    expected_pairs: set[tuple[str, str]],
    summary: dict,
    *,
    poll_seconds: float = _RESULT_POLL_SECONDS,
    exit_grace_seconds: float = _WORKER_EXIT_GRACE_SECONDS,
) -> None:
    """Collect a complete, exactly-once result protocol without blocking forever."""
    worker_ids = set(range(len(procs)))
    finished: set[int] = set()
    seen: set[tuple[str, str]] = set()
    dead_without_sentinel_since: dict[int, float] = {}

    while True:
        if finished == worker_ids:
            missing = expected_pairs - seen
            if missing:
                raise DatagenRunError(
                    f"workers finished with partial results; missing={sorted(missing)}",
                    summary=summary,
                )
            return

        item = None
        try:
            item = result_q.get(timeout=poll_seconds)
        except queue.Empty:
            pass
        except (EOFError, OSError) as exc:
            raise DatagenRunError(
                f"worker result queue failed: {exc}", summary=summary) from exc

        if item is not None:
            if not isinstance(item, dict):
                raise DatagenRunError(
                    f"invalid worker result message {item!r}", summary=summary)
            event = item.get("event")
            worker_id = item.get("worker_id")
            if (
                isinstance(worker_id, bool)
                or not isinstance(worker_id, int)
                or worker_id not in worker_ids
            ):
                raise DatagenRunError(
                    f"invalid worker id in result message {item!r}",
                    summary=summary,
                )
            if event == "finished":
                if worker_id in finished:
                    raise DatagenRunError(
                        f"duplicate finish sentinel from worker {worker_id}",
                        summary=summary,
                    )
                finished.add(worker_id)
            elif event == "fatal":
                summary["error"] += 1
                raise DatagenRunError(
                    f"worker {worker_id} failed: {item.get('error', 'unknown error')}",
                    summary=summary,
                )
            elif event == "result":
                status = item.get("status")
                task_id = item.get("task_id")
                kind = item.get("kind")
                pair = (task_id, kind)
                if status == "error":
                    summary["error"] += 1
                    raise DatagenRunError(
                        f"{task_id}:{kind} failed in worker {worker_id}: "
                        f"{item.get('error', 'unknown error')}",
                        summary=summary,
                    )
                if status not in ("done", "skip"):
                    raise DatagenRunError(
                        f"invalid shard status {status!r}", summary=summary)
                if pair not in expected_pairs:
                    raise DatagenRunError(
                        f"unexpected shard result {pair!r}", summary=summary)
                if pair in seen:
                    raise DatagenRunError(
                        f"duplicate shard result {pair!r}", summary=summary)
                count = item.get("records")
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                ):
                    raise DatagenRunError(
                        f"invalid record count for {pair!r}: {count!r}",
                        summary=summary,
                    )
                seen.add(pair)
                summary[status] += 1
                summary["records"] += count
            else:
                raise DatagenRunError(
                    f"unknown worker event {event!r}", summary=summary)

        now = time.monotonic()
        for worker_id, proc in enumerate(procs):
            exitcode = proc.exitcode
            if exitcode is None:
                dead_without_sentinel_since.pop(worker_id, None)
                continue
            if exitcode != 0:
                raise DatagenRunError(
                    f"worker {worker_id} died with exit code {exitcode}",
                    summary=summary,
                )
            if worker_id not in finished:
                since = dead_without_sentinel_since.setdefault(worker_id, now)
                if now - since >= exit_grace_seconds:
                    raise DatagenRunError(
                        f"worker {worker_id} exited without a finish sentinel",
                        summary=summary,
                    )


def run_parallel_datagen(task_ids, kinds, data_root, counts, *, n_workers: int,
                         n_gpus: int, teacher_kind: str = "claude",
                         model_teacher=None, gpu_ids=None, log=print) -> dict:
    """Run datagen for ``kinds`` over ``task_ids`` across GPU-pinned worker processes.
    Resumable (existing shards skipped). Uses a SHARED task queue so every worker
    stays busy pulling the next task until all are done (no tail draining).

    ``gpu_ids``: explicit PHYSICAL GPU ids to pin to (e.g. the free ones on a shared
    node). Datagen is TEACHER-bound, so we run MORE workers than GPUs (teacher-
    parallel), round-robining workers onto the pinned GPUs; verification
    oversubscribes those GPUs but the per-GPU timing lock keeps measurements clean.
    """
    import multiprocessing as mp

    if isinstance(task_ids, (str, bytes)):
        raise TypeError("task_ids must be an iterable of task-id strings")
    if isinstance(kinds, (str, bytes)):
        raise TypeError("kinds must be an iterable of kind strings")
    task_ids = list(dict.fromkeys(task_ids))
    kinds = list(dict.fromkeys(kinds))
    for task_id in task_ids:
        if (
            not isinstance(task_id, str)
            or not task_id.strip()
            or Path(task_id).name != task_id
            or task_id in (".", "..")
        ):
            raise ValueError(f"invalid task id {task_id!r}")
    unknown_kinds = [kind for kind in kinds if kind not in ALL_KINDS]
    if unknown_kinds:
        raise ValueError(f"unknown datagen kinds: {unknown_kinds}")
    for kind in kinds:
        build_generator_contract(
            kind,
            counts,
            teacher_kind=teacher_kind,
            model_teacher=model_teacher,
        )

    n_gpus = max(1, int(n_gpus))
    if gpu_ids:
        gpu_ids = [int(g) for g in gpu_ids]
        if any(gpu < 0 for gpu in gpu_ids):
            raise ValueError("gpu ids must be non-negative")
        nw = max(int(n_workers) if n_workers else len(gpu_ids), len(gpu_ids))
        worker_gpus = [gpu_ids[w % len(gpu_ids)] for w in range(nw)]
        dev_list = gpu_ids
    else:
        nw = max(1, int(n_workers))
        worker_gpus = [w % n_gpus for w in range(nw)]
        dev_list = list(range(n_gpus))

    log(f"parallel datagen: {len(task_ids)} tasks x {kinds} across "
        f"{nw} workers on GPUs {dev_list} (dynamic queue)")
    summary = {"done": 0, "skip": 0, "error": 0, "records": 0, "tasks": len(task_ids)}
    if not task_ids or not kinds:
        log(f"parallel datagen done: {summary}")
        return summary
    expected_pairs = {
        (task_id, kind) for task_id in task_ids for kind in kinds
    }

    ctxmp = mp.get_context("spawn")
    task_q: "mp.Queue" = ctxmp.Queue()
    result_q: "mp.Queue" = ctxmp.Queue()
    for tid in task_ids:
        task_q.put(tid)
    for _ in range(nw):
        task_q.put(None)  # one drain-sentinel per worker

    procs = [
        ctxmp.Process(
            target=_queue_worker,
            args=(
                worker_id,
                worker_gpus[worker_id],
                task_q,
                result_q,
                kinds,
                str(data_root),
                counts,
                teacher_kind,
                model_teacher,
            ),
        )
        for worker_id in range(nw)
    ]
    started: list = []
    try:
        for proc in procs:
            proc.start()
            started.append(proc)
        _collect_worker_results(
            procs,
            result_q,
            expected_pairs,
            summary,
        )
        _join_workers(procs, summary)
    except BaseException:
        _abort_workers(started)
        raise
    finally:
        for work_queue in (task_q, result_q):
            try:
                work_queue.cancel_join_thread()
            except (AttributeError, ValueError):
                pass
            try:
                work_queue.close()
            except (AttributeError, ValueError):
                pass

    log(f"parallel datagen done: {summary}")
    return summary
