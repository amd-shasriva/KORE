"""Monotonically merge independent datagen roots without losing records.

Destination records are preserved in their existing order and records present
only in the source are appended. Wins deduplicate by canonical ``final_source``;
repair and ranked-group records deduplicate by canonical JSON. Writes are
atomic, task-locked, and opt-in via ``--apply``.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile

KINDS = ("repair", "groups", "wins")


def _read_records(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    records = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"invalid JSONL record {path}:{line_no}: expected object"
                )
            records.append(record)
    return records


def _record_key(kind: str, record: dict) -> str:
    if kind == "wins":
        source = str(record.get("final_source", "") or "").strip()
        if source:
            return f"source:{source}"
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return f"record:{payload}"


def merge_records(kind: str, destination: list[dict], source: list[dict]) -> tuple[list[dict], int]:
    """Return destination-first union and the number of source records added."""
    merged = list(destination)
    seen = {_record_key(kind, record) for record in destination}
    added = 0
    for record in source:
        key = _record_key(kind, record)
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)
        added += 1
    return merged, added


@contextmanager
def _task_lock(root: Path, kind: str, task_id: str):
    # Use the same stage locks as the production drivers so a defensive merge
    # cannot race an accidentally active campaign.
    stage = "deepen" if kind == "wins" else "base"
    path = root / ".locks" / stage / f"{task_id}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def _atomic_write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        tmp.unlink(missing_ok=True)


def merge_roots(
    source_root: Path,
    destination_root: Path,
    *,
    kinds: tuple[str, ...] = KINDS,
    prefix: str = "genb_",
    apply: bool = False,
) -> dict:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    if source_root == destination_root:
        raise ValueError("source and destination roots must differ")

    summary = {
        "apply": apply,
        "source": str(source_root),
        "destination": str(destination_root),
        "files_scanned": 0,
        "files_changed": 0,
        "records_added": 0,
        "by_kind": {},
    }
    for kind in kinds:
        if kind not in KINDS:
            raise ValueError(f"unsupported kind: {kind}")
        source_dir = source_root / kind
        destination_dir = destination_root / kind
        task_ids = {
            path.stem
            for directory in (source_dir, destination_dir)
            if directory.exists()
            for path in directory.glob("*.jsonl")
            if path.stem.startswith(prefix)
        }
        kind_summary = {"files_scanned": 0, "files_changed": 0, "records_added": 0}
        for task_id in sorted(task_ids):
            source_path = source_dir / f"{task_id}.jsonl"
            destination_path = destination_dir / f"{task_id}.jsonl"
            source_records = _read_records(source_path)
            destination_records = _read_records(destination_path)
            merged, added = merge_records(kind, destination_records, source_records)
            kind_summary["files_scanned"] += 1
            if added:
                if apply:
                    with _task_lock(destination_root, kind, task_id):
                        # Reload under the lock so a writer between the dry read and
                        # lock acquisition cannot be overwritten.
                        current = _read_records(destination_path)
                        merged, locked_added = merge_records(kind, current, source_records)
                        if locked_added:
                            _atomic_write(destination_path, merged)
                        added = locked_added
                if added:
                    kind_summary["files_changed"] += 1
                    kind_summary["records_added"] += added
            summary["files_scanned"] += 1
        summary["files_changed"] += kind_summary["files_changed"]
        summary["records_added"] += kind_summary["records_added"]
        summary["by_kind"][kind] = kind_summary
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_root")
    ap.add_argument("destination_root")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--prefix", default="genb_")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    kinds = tuple(kind for kind in args.kinds.split(",") if kind)
    summary = merge_roots(
        Path(args.source_root),
        Path(args.destination_root),
        kinds=kinds,
        prefix=args.prefix,
        apply=args.apply,
    )
    print("MERGE_DATAGEN " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
