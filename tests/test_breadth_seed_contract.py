"""CPU-only integration contract for generated breadth task seeds."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from kore.reward.reward import scan_for_hacks
from kore.tasks.generate_breadth import (
    TASKS_DIR,
    _validated_seed_source,
    generate,
)

_ARTIFACTS = ("task.yaml", "reference.py", "seed_triton.py", "driver.py")


def _snapshot(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_all_admitted_breadth_seeds_are_scanner_clean():
    """Collect every rejection so one run reports the complete failing task set."""
    failures: list[str] = []
    for task_id in generate(dry=True):
        task_dir = TASKS_DIR / task_id
        meta = yaml.safe_load((task_dir / "task.yaml").read_text())
        if meta.get("seed_state", "admitted") != "admitted":
            continue
        seed_name = meta.get("seed_kernel_name")
        if not seed_name:
            failures.append(f"{task_id}: admitted task has no seed_kernel_name")
            continue
        source = (task_dir / seed_name).read_text()
        reason = scan_for_hacks(source)
        if reason is not None:
            failures.append(f"{task_id}: {reason}")
    assert not failures, "admitted breadth seed failures:\n" + "\n".join(failures)


def test_generator_is_idempotent_valid_and_matches_tracked(tmp_path):
    out = tmp_path / "breadth"
    task_ids = generate(output_dir=out)
    assert len(task_ids) == 1052
    assert len(task_ids) == len(set(task_ids))

    first = _snapshot(out)
    assert len(first) == len(task_ids) * len(_ARTIFACTS)
    assert generate(output_dir=out) == task_ids
    assert _snapshot(out) == first

    failures: list[str] = []
    for task_id in task_ids:
        task_dir = out / task_id
        paths = {name: task_dir / name for name in _ARTIFACTS}
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            failures.append(f"{task_id}: missing {', '.join(missing)}")
            continue

        try:
            meta = yaml.safe_load(paths["task.yaml"].read_text())
        except Exception as exc:  # noqa: BLE001 - aggregate every malformed artifact
            failures.append(f"{task_id}: invalid YAML: {exc}")
            continue
        operation = meta.get("operation")
        expected = {
            "task_id": task_id,
            "backend": "triton",
            "gpu_target": "gfx950",
            "seed_kernel_name": "seed_triton.py",
            "generated": True,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                failures.append(
                    f"{task_id}: YAML {key}={meta.get(key)!r}, expected {value!r}")
        shapes = meta.get("shapes") or {}
        if set(shapes) != {"minimal", "primary", "validation"}:
            failures.append(f"{task_id}: invalid shapes keys {sorted(shapes)}")

        for name in ("reference.py", "seed_triton.py", "driver.py"):
            try:
                tree = ast.parse(paths[name].read_text(), filename=str(paths[name]))
            except SyntaxError as exc:
                failures.append(f"{task_id}/{name}: invalid Python: {exc}")
                continue
            if name == "seed_triton.py":
                entries = {
                    node.name for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if operation not in entries:
                    failures.append(
                        f"{task_id}: seed missing top-level entrypoint {operation!r}")

        for name, generated_path in paths.items():
            tracked_path = TASKS_DIR / task_id / name
            if not tracked_path.is_file():
                failures.append(f"{task_id}/{name}: missing tracked artifact")
            elif generated_path.read_bytes() != tracked_path.read_bytes():
                failures.append(f"{task_id}/{name}: tracked artifact is stale")

    assert not failures, "breadth generation contract failures:\n" + "\n".join(failures)


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("def wrong(x):\n    return x\n", "top-level entry"),
        ("import torch as t\ndef op(x):\n    return t.softmax(x, -1)\n", "rejected by scanner"),
    ],
)
def test_generator_rejects_non_admissible_engine_seeds(source, match):
    mod = SimpleNamespace(seed_source=lambda op, dtype: source)
    with pytest.raises(ValueError, match=match):
        _validated_seed_source(mod, "op", "bf16")
