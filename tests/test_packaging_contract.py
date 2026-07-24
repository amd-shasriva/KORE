"""Fast source-tree checks for data that must survive packaging."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "kore" / "tasks"
TASK_FILES = ("task.yaml", "reference.py", "driver.py", "seed_triton.py")


@pytest.mark.packaging
def test_source_task_assets_match_the_live_registry() -> None:
    from kore.tasks.registry import task_ids

    asset_errors: list[str] = []
    directory_ids: list[str] = []
    for task_yaml in sorted(TASKS.glob("*/task.yaml")):
        directory_ids.append(task_yaml.parent.name)
        for name in TASK_FILES:
            if not (task_yaml.parent / name).is_file():
                asset_errors.append(f"{task_yaml.parent.name}: missing {name}")

    registered_ids = task_ids()
    if directory_ids != registered_ids:
        asset_errors.append(
            "registry differs from task directories: "
            f"missing={sorted(set(directory_ids) - set(registered_ids))} "
            f"extra={sorted(set(registered_ids) - set(directory_ids))}"
        )
    assert not asset_errors, "\n".join(asset_errors)


@pytest.mark.packaging
def test_runtime_fixture_sets_are_complete() -> None:
    from kore.data import amd_knowledge, general_replay
    from kore.eval import retention

    eval_files = {path.stem for path in retention._DATA_DIR.glob("*.jsonl")}
    assert eval_files == set(retention.FULL_HF_SOURCES)

    replay_files = {path.stem for path in general_replay._SAMPLES_DIR.glob("*.jsonl")}
    assert replay_files == set(general_replay.REPLAY_KINDS)

    playbook_path = ROOT / "kore" / "data" / "knowledge" / "amd_triton_playbook.md"
    amd_knowledge.playbook.cache_clear()
    assert playbook_path.is_file()
    assert amd_knowledge.playbook() == playbook_path.read_text(encoding="utf-8").strip()

    golden = ROOT / "kore" / "openended" / "tests" / "_golden_mint_baseline.json"
    assert json.loads(golden.read_text(encoding="utf-8"))


@pytest.mark.packaging
def test_ci_dependencies_are_explicit_and_workflows_parse() -> None:
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows
    action_refs: list[str] = []
    for workflow in workflows:
        parsed = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), workflow
        action_refs.extend(
            match.group(1)
            for match in re.finditer(
                r"(?m)^\s*uses:\s*([^\s#]+)",
                workflow.read_text(encoding="utf-8"),
            )
        )
    assert action_refs
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", ref) for ref in action_refs)

    constraints = ROOT / ".github" / "constraints-ci.txt"
    requirements = [
        line.strip()
        for line in constraints.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^=\s]+", req) for req in requirements)


@pytest.mark.release
def test_release_has_approved_license_and_attribution() -> None:
    """Do not infer legal terms: an authorized owner must supply these files."""
    license_files = sorted(
        {
            *ROOT.glob("LICENSE*"),
            *ROOT.glob("COPYING*"),
        }
    )
    attribution_files = sorted(
        {
            *ROOT.glob("NOTICE*"),
            *ROOT.glob("THIRD_PARTY*"),
            *ROOT.glob("ATTRIBUTION*"),
        }
    )
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_block = pyproject.split("[project]", 1)[-1].split("\n[", 1)[0]
    has_license_metadata = bool(
        re.search(r"(?m)^license(?:-files)?\s*=", project_block)
    )

    errors: list[str] = []
    if not license_files:
        errors.append("missing an owner-approved LICENSE/COPYING file")
    if not attribution_files:
        errors.append("missing NOTICE/THIRD_PARTY/ATTRIBUTION documentation")
    if not has_license_metadata:
        errors.append("missing [project] license/license-files metadata")
    assert not errors, "release blocked:\n- " + "\n- ".join(errors)
