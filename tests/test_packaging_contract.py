"""Fast source-tree checks for data that must survive packaging."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "kore" / "tasks"
# Assets every task must ship regardless of backend.  The seed is NOT in this
# list: its filename is declared per task by ``seed_kernel_name`` (a Triton task
# ships ``seed_triton.py``, a HIP task ships ``seed_hip.hip``), so hard-coding one
# name here would either reject a valid HIP task or, worse, pass a task whose
# declared seed is absent while a stale ``seed_triton.py`` sits beside it.
TASK_FILES = ("task.yaml", "reference.py", "driver.py")


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
        # The seed the task actually declares must exist, whatever it is called.
        try:
            declared = (yaml.safe_load(task_yaml.read_text()) or {}).get("seed_kernel_name")
        except yaml.YAMLError as exc:
            asset_errors.append(f"{task_yaml.parent.name}: unreadable task.yaml ({exc})")
            continue
        if not isinstance(declared, str) or not declared.strip():
            asset_errors.append(f"{task_yaml.parent.name}: declares no seed_kernel_name")
        elif not (task_yaml.parent / declared.strip()).is_file():
            asset_errors.append(
                f"{task_yaml.parent.name}: missing declared seed {declared.strip()}")

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


# --------------------------------------------------------------------------- #
# THIRD_PARTY.md structure
#
# Attribution rots silently: a source gets added to the corpus catalog and never
# reaches the human-readable file, or an entry keeps a placeholder like SEE-REPO
# that reads as "recorded" while naming no actual terms. Both happened. These
# checks parse the document instead of grepping it, so a malformed or hollowed-out
# entry fails rather than passing on a substring match.
# --------------------------------------------------------------------------- #
THIRD_PARTY = ROOT / "THIRD_PARTY.md"
SOURCE_CATALOG = ROOT / "data" / "release" / "meta" / "source_metadata.json"

# An entry table is any markdown table keyed by one of these in column 1. Every
# such table must carry exactly these columns, in this order.
_ENTRY_KEYS = ("Source", "Dataset", "Artifact")
_ENTRY_COLUMNS = (
    "Licence (SPDX)",
    "Upstream URL",
    "Pinned revision",
    "Used for / flows into",
)
# Values that look like an answer but name no terms.
_PLACEHOLDER_LICENSES = frozenset(
    {
        "see-repo",
        "see repo",
        "unresolved",
        "unknown",
        "development-unknown",
        "development-internal",
        "none",
        "n/a",
        "na",
        "tbd",
        "todo",
        "?",
        "-",
        "",
    }
)


def _parse_third_party_entries(text: str) -> list[dict[str, str]]:
    """Parse THIRD_PARTY.md's attribution tables into structured entries.

    Returns one dict per data row of every table whose first column header is in
    ``_ENTRY_KEYS``. Raises on a table that claims to be an entry table but does
    not have the required columns, so a silently reshaped table cannot pass.
    """
    entries: list[dict[str, str]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not (line.startswith("|") and line.endswith("|")):
            index += 1
            continue
        header = [cell.strip() for cell in line.strip("|").split("|")]
        if not header or header[0] not in _ENTRY_KEYS:
            index += 1
            continue
        separator = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if not re.fullmatch(r"\|(?:\s*:?-{3,}:?\s*\|)+", separator):
            index += 1
            continue
        if tuple(header[1:]) != _ENTRY_COLUMNS:
            raise AssertionError(
                f"THIRD_PARTY.md: table keyed by {header[0]!r} has columns "
                f"{header[1:]}, expected {list(_ENTRY_COLUMNS)}"
            )
        index += 2
        while index < len(lines):
            row = lines[index].strip()
            if not (row.startswith("|") and row.endswith("|")):
                break
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if len(cells) != len(header):
                raise AssertionError(
                    f"THIRD_PARTY.md: row under {header[0]!r} has {len(cells)} "
                    f"cells, expected {len(header)}: {row}"
                )
            entries.append(
                {
                    "kind": header[0],
                    "name": cells[0],
                    "license": cells[1],
                    "url": cells[2],
                    "revision": cells[3],
                    "usage": cells[4],
                }
            )
            index += 1
    return entries


@pytest.mark.release
def test_third_party_attribution_is_complete_and_structured() -> None:
    """Every attributed source names real terms, a URL, and the revision used.

    A licence is only meaningful next to the revision it was read at: upstream
    projects relicense, and this repository pins commits. An entry that keeps the
    licence but drops the pin is not attribution, so all four fields are required
    together.
    """
    assert THIRD_PARTY.is_file(), "THIRD_PARTY.md is missing"
    text = THIRD_PARTY.read_text(encoding="utf-8")
    entries = _parse_third_party_entries(text)

    errors: list[str] = []
    if len(entries) < 30:
        errors.append(f"only {len(entries)} attribution entries parsed; expected >= 30")

    seen: set[tuple[str, str]] = set()
    for entry in entries:
        label = f"{entry['kind']} {entry['name']!r}"
        if not entry["name"]:
            errors.append("an attribution row has an empty name")
        if entry["license"].strip().lower() in _PLACEHOLDER_LICENSES:
            errors.append(f"{label}: placeholder licence {entry['license']!r}")
        if not re.fullmatch(r"https://\S+", entry["url"]):
            errors.append(f"{label}: not a URL: {entry['url']!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", entry["revision"]):
            errors.append(f"{label}: revision is not a 40-hex commit/revision")
        if not entry["usage"]:
            errors.append(f"{label}: missing what KORE uses it for")
        key = (entry["url"], entry["revision"])
        if key in seen:
            errors.append(f"{label}: duplicate entry for {key}")
        seen.add(key)

    assert not errors, "THIRD_PARTY.md attribution incomplete:\n- " + "\n- ".join(errors)


@pytest.mark.release
def test_third_party_covers_every_catalog_source() -> None:
    """The corpus catalog and the human-readable file must not drift apart.

    ``source_metadata.json`` is what the build actually read; THIRD_PARTY.md is
    what a human reviews. A source present in the first and absent from the second
    ships unattributed, which is exactly the failure this check exists to prevent.
    """
    if not SOURCE_CATALOG.is_file():
        pytest.skip(f"source catalog not present: {SOURCE_CATALOG}")

    catalog = json.loads(SOURCE_CATALOG.read_text(encoding="utf-8"))
    documented = {
        (entry["url"], entry["revision"])
        for entry in _parse_third_party_entries(
            THIRD_PARTY.read_text(encoding="utf-8")
        )
    }
    documented_urls = {url for url, _ in documented}

    errors: list[str] = []
    for source in catalog.get("sources", ()):
        url = str(source.get("repository_url", ""))
        commit = str(source.get("commit", ""))
        if url not in documented_urls:
            errors.append(f"source {source.get('source_id')!r} ({url}) is undocumented")
        elif (url, commit) not in documented:
            errors.append(
                f"source {source.get('source_id')!r} is documented at a different "
                f"revision than the catalog's {commit}"
            )
    for dataset in catalog.get("datasets", ()):
        url = str(dataset.get("repository_url", ""))
        revision = str(dataset.get("revision", ""))
        if (url, revision) not in documented:
            errors.append(
                f"dataset {dataset.get('dataset_id')!r} @ {revision} is undocumented"
            )

    assert not errors, (
        "THIRD_PARTY.md has drifted from the corpus source catalog:\n- "
        + "\n- ".join(errors)
    )
