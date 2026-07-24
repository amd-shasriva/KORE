"""Release contract for checked-in breadth-generated task artifacts."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest
import yaml


GENERATED_FILES = ("task.yaml", "reference.py", "driver.py", "seed_triton.py")


def _manifest(root: Path, task_ids: list[str]) -> dict[str, bytes]:
    return {
        f"{task_id}/{name}": (root / task_id / name).read_bytes()
        for task_id in task_ids
        for name in GENERATED_FILES
        if (root / task_id / name).is_file()
    }


@pytest.mark.release
def test_breadth_generation_is_deterministic_valid_and_hack_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regenerate twice and report every contract violation in one failure."""
    from kore.reward.reward import scan_for_hacks
    from kore.tasks import generate_breadth

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    monkeypatch.setattr(generate_breadth, "TASKS_DIR", first)
    first_ids = generate_breadth.generate()
    monkeypatch.setattr(generate_breadth, "TASKS_DIR", second)
    second_ids = generate_breadth.generate()

    source_root = Path(generate_breadth.__file__).resolve().parent
    committed_ids = sorted(
        path.parent.name for path in source_root.glob("genb_*/task.yaml")
    )
    errors: list[str] = []
    scanner_errors: list[str] = []

    if first_ids != second_ids:
        errors.append("generator returned a different task order across runs")
    if set(first_ids) != set(committed_ids):
        missing = sorted(set(first_ids) - set(committed_ids))
        extra = sorted(set(committed_ids) - set(first_ids))
        errors.append(
            "committed breadth task ids differ from the generator: "
            f"missing={missing} extra={extra}"
        )

    first_manifest = _manifest(first, first_ids)
    second_manifest = _manifest(second, second_ids)
    if first_manifest != second_manifest:
        differing = sorted(
            name
            for name in set(first_manifest) | set(second_manifest)
            if first_manifest.get(name) != second_manifest.get(name)
        )
        errors.append(f"generation is nondeterministic: {differing}")

    for task_id in first_ids:
        generated_dir = first / task_id
        committed_dir = source_root / task_id
        for name in GENERATED_FILES:
            generated = generated_dir / name
            committed = committed_dir / name
            if not generated.is_file():
                errors.append(f"{task_id}: generator omitted {name}")
                continue
            if not committed.is_file():
                errors.append(f"{task_id}: committed tree omitted {name}")
            elif generated.read_bytes() != committed.read_bytes():
                errors.append(f"{task_id}: committed {name} is stale")

        task_yaml = generated_dir / "task.yaml"
        if task_yaml.is_file():
            try:
                metadata = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    errors.append(f"{task_id}/task.yaml: expected a mapping")
                elif metadata.get("task_id") != task_id:
                    errors.append(
                        f"{task_id}/task.yaml: task_id={metadata.get('task_id')!r}"
                    )
            except Exception as exc:  # noqa: BLE001 - aggregate every parse error
                errors.append(f"{task_id}/task.yaml: YAML parse failed: {exc}")

        for name in GENERATED_FILES[1:]:
            path = generated_dir / name
            if not path.is_file():
                continue
            source = path.read_text(encoding="utf-8")
            try:
                ast.parse(source, filename=f"{task_id}/{name}")
            except SyntaxError as exc:
                errors.append(f"{task_id}/{name}: Python parse failed: {exc}")
            if name == "seed_triton.py":
                reason = scan_for_hacks(source)
                if reason:
                    scanner_errors.append(f"{task_id}/{name}: {reason}")

    errors.extend(scanner_errors)
    if errors:
        categories = Counter(
            "seed scanner" if error in scanner_errors else "artifact/parse"
            for error in errors
        )
        pytest.fail(
            "breadth release contract failed "
            f"({dict(categories)}):\n" + "\n".join(f"- {error}" for error in errors),
            pytrace=False,
        )
