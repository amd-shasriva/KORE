"""Dry-run-first inventory and quarantine migration for legacy JSONL shards.

The default performs no writes. ``--apply`` requires a separate output root,
copies the exact source bytes to a digest-named backup, and writes a migrated
quarantine copy without modifying the source. Legacy semantic truth is never
inferred: migrated records are explicitly marked ``semantic_validity=unknown``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from kore.data.parallel_datagen import write_receipt_for_existing_shard
from kore.data.schemas import (
    JsonlReadMode,
    atomic_write_bytes,
    read_jsonl,
    stamp_legacy_record_unknown,
    validate_record_dict,
    write_jsonl,
)

_RECORD_DIR_TO_TYPE = {
    "repair": "repair",
    "groups": "ranked_group",
    "wins": "win",
    "agentic": "agentic",
}


@dataclass
class MigrationIssue:
    path: str
    line: int | None
    stage: str
    error: str


@dataclass
class ShardInventory:
    path: str
    lane: str
    byte_count: int
    sha256: str
    record_count: int = 0
    proposed_count: int = 0
    production_valid: bool = False
    receipt_eligible: bool = False
    receipt_created: bool = False
    backup_path: str | None = None
    output_path: str | None = None
    issues: list[MigrationIssue] = field(default_factory=list)


@dataclass
class MigrationInventory:
    dry_run: bool
    roots: list[str]
    paths: list[ShardInventory] = field(default_factory=list)

    def summary(self) -> dict:
        issues = [issue for path in self.paths for issue in path.issues]
        return {
            "dry_run": self.dry_run,
            "roots": self.roots,
            "paths": len(self.paths),
            "record_shards": sum(path.lane == "record" for path in self.paths),
            "generic_jsonl": sum(path.lane == "generic" for path in self.paths),
            "records": sum(path.record_count for path in self.paths),
            "proposed_records": sum(path.proposed_count for path in self.paths),
            "production_valid_paths": sum(path.production_valid for path in self.paths),
            "receipt_eligible_paths": sum(path.receipt_eligible for path in self.paths),
            "receipts_created": sum(path.receipt_created for path in self.paths),
            "issues": len(issues),
            "issue_stages": {
                stage: sum(issue.stage == stage for issue in issues)
                for stage in sorted({issue.stage for issue in issues})
            },
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "paths": [
                {
                    **{
                        key: value
                        for key, value in asdict(path).items()
                        if key != "issues"
                    },
                    "issues": [asdict(issue) for issue in path.issues],
                }
                for path in self.paths
            ],
        }


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token!r}")


def _record_lane(path: Path) -> tuple[str, str | None]:
    expected_type = _RECORD_DIR_TO_TYPE.get(path.parent.name)
    return ("record", expected_type) if expected_type else ("generic", None)


def _relative_destination(path: Path, root: Path, output_root: Path) -> Path:
    return output_root / root.name / path.relative_to(root)


def _issue(inventory: ShardInventory, line: int | None,
           stage: str, exc: BaseException | str) -> None:
    inventory.issues.append(MigrationIssue(
        path=inventory.path,
        line=line,
        stage=stage,
        error=str(exc),
    ))


def inventory_shard(
    path: Path,
    *,
    root: Path,
    output_root: Path | None = None,
    apply: bool = False,
    overwrite: bool = False,
    contracts_root: Path | None = None,
) -> ShardInventory:
    raw = path.read_bytes()
    lane, expected_type = _record_lane(path)
    inventory = ShardInventory(
        path=str(path),
        lane=lane,
        byte_count=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    proposed: list[dict] = []
    lines = raw.splitlines(keepends=True)
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            _issue(inventory, line_number, "parse", "blank line")
            continue
        if not line.endswith(b"\n"):
            _issue(inventory, line_number, "parse", "truncated line (missing newline)")
        try:
            value = json.loads(
                line.decode("utf-8"), parse_constant=_reject_constant)
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL row must be an object, got {type(value).__name__}")
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            _issue(inventory, line_number, "parse", exc)
            continue
        inventory.record_count += 1
        if lane == "generic":
            continue
        if value.get("type") != expected_type:
            _issue(
                inventory,
                line_number,
                "record_type",
                f"expected {expected_type!r}, got {value.get('type')!r}",
            )
            continue
        try:
            migrated = stamp_legacy_record_unknown(value)
            validate_record_dict(migrated, expected_type=expected_type)
            proposed.append(migrated)
        except (KeyError, TypeError, ValueError) as exc:
            _issue(inventory, line_number, "structural_validation", exc)

    inventory.proposed_count = len(proposed)
    if lane == "record" and not inventory.issues:
        try:
            read_jsonl(
                path,
                typed=False,
                mode=JsonlReadMode.PRODUCTION_STRICT,
                expected_type=expected_type,
            )
            inventory.production_valid = True
            inventory.receipt_eligible = True
        except (OSError, TypeError, ValueError):
            # Expected for legacy files; this is represented by production_valid,
            # not duplicated as a per-line migration error.
            pass

    if not apply:
        return inventory
    if output_root is None:
        raise ValueError("apply mode requires a separate output_root")
    destination = _relative_destination(path, root, output_root)
    if destination.exists() and not overwrite:
        _issue(inventory, None, "write", f"destination exists: {destination}")
        return inventory
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(
        f"{destination.name}.source.{inventory.sha256[:12]}.bak")
    if backup.exists() and not overwrite:
        _issue(inventory, None, "backup", f"backup exists: {backup}")
        return inventory
    atomic_write_bytes(backup, raw)
    inventory.backup_path = str(backup)
    if lane == "record":
        if inventory.issues:
            _issue(
                inventory,
                None,
                "write",
                "record shard has errors; quarantine output not written",
            )
            return inventory
        if inventory.production_valid:
            atomic_write_bytes(destination, raw)
        else:
            write_jsonl(destination, proposed)
    else:
        # Generic rows are inventoried and backed up but not semantically rewritten.
        atomic_write_bytes(destination, raw)
    inventory.output_path = str(destination)
    if inventory.production_valid and contracts_root is not None:
        kind = path.parent.name
        contract_path = contracts_root / kind / f"{path.stem}.json"
        if not contract_path.exists():
            _issue(
                inventory,
                None,
                "receipt",
                f"generation contract missing: {contract_path}",
            )
        else:
            try:
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                write_receipt_for_existing_shard(
                    destination.parent.parent,
                    path.stem,
                    kind,
                    contract=contract,
                )
                inventory.receipt_created = True
            except (OSError, TypeError, ValueError) as exc:
                _issue(inventory, None, "receipt", exc)
    # Legacy quarantine records never receive receipts: validity is unknown and
    # historical generator/evaluator identity cannot be reconstructed safely.
    return inventory


def inventory_roots(
    roots: Iterable[Path | str],
    *,
    output_root: Path | str | None = None,
    apply: bool = False,
    overwrite: bool = False,
    max_files: int | None = None,
    contracts_root: Path | str | None = None,
) -> MigrationInventory:
    root_paths = [Path(root).resolve() for root in roots]
    if apply and output_root is None:
        raise ValueError("apply mode requires --output-root")
    output_path = Path(output_root).resolve() if output_root is not None else None
    contracts_path = (
        Path(contracts_root).resolve() if contracts_root is not None else None)
    report = MigrationInventory(
        dry_run=not apply,
        roots=[str(root) for root in root_paths],
    )
    candidates: list[tuple[Path, Path]] = []
    for root in root_paths:
        if not root.exists():
            report.paths.append(ShardInventory(
                path=str(root),
                lane="missing",
                byte_count=0,
                sha256=hashlib.sha256(b"").hexdigest(),
                issues=[MigrationIssue(
                    path=str(root), line=None, stage="path", error="root does not exist")],
            ))
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
        for path in paths:
            if ".complete." in path.name:
                continue
            candidates.append((root if root.is_dir() else root.parent, path))
    if max_files is not None:
        candidates = candidates[:max(0, int(max_files))]
    for root, path in candidates:
        report.paths.append(inventory_shard(
            path,
            root=root,
            output_root=output_path,
            apply=apply,
            overwrite=overwrite,
            contracts_root=contracts_path,
        ))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inventory or quarantine-migrate legacy KORE JSONL shards")
    parser.add_argument("roots", nargs="+", help="JSONL files or directories")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write separate quarantine copies; default is dry-run only",
    )
    parser.add_argument("--output-root")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing files under output-root; never touches source roots",
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--contracts-root",
        help="optional kind/task.json contract tree for already-production-valid files",
    )
    parser.add_argument("--report", help="optional JSON report path")
    args = parser.parse_args(argv)
    report = inventory_roots(
        args.roots,
        output_root=args.output_root,
        apply=args.apply,
        overwrite=args.overwrite,
        max_files=args.max_files,
        contracts_root=args.contracts_root,
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.report:
        atomic_write_bytes(args.report, rendered.encode("utf-8") + b"\n")
    print(rendered)
    return 0 if not any(path.issues for path in report.paths) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MigrationInventory",
    "MigrationIssue",
    "ShardInventory",
    "inventory_roots",
    "inventory_shard",
    "main",
]
