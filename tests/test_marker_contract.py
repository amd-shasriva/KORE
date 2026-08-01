"""The resource-marker contract: a declared marker must select real tests.

``pyproject.toml`` registers marker names and then deselects most of them by
default via ``addopts = [..., "-m", "not gpu and not release"]``.  That gate is
only meaningful if the names in it correspond to tests that exist: an audit
found ``pytest -m gpu`` selecting **zero** tests, so the deselection was a no-op
and the documented opt-in group was decorative.  These tests make the
declaration load-bearing in both directions.

The census comes from the root ``conftest.py``, which counts markers in a
``tryfirst`` ``pytest_collection_modifyitems`` hook - i.e. before pytest applies
the ``-m`` expression - so the opt-in groups are visible even in a default run
that deselects them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

#: GPU suites are named so that ``pytest -m gpu`` and the file layout agree; a
#: new hardware suite belongs in a module matching this prefix.
GPU_MODULE_PREFIX = "test_gpu"

#: Groups the default suite must keep out of a plain ``pytest`` run.  Anything
#: needing an accelerator or a release-only artifact belongs here.
REQUIRED_DESELECTED_MARKERS = ("gpu", "release")

_SKIP_PARTIAL_RUN = (
    "the marker contract is only decidable when the whole tests/ tree is "
    "collected; run `python -m pytest` or `python -m pytest tests` to enforce it"
)


@pytest.fixture(scope="module")
def tests_tree_collected(marker_modules: dict[str, set[str]]) -> bool:
    """Whether this session collected every ``tests/test_*.py`` module.

    Every marker-bearing test lives under ``tests/``, and the root conftest
    censuses each collected item, so a collection covering that whole tree can
    decide the contract - which is true both for a bare ``pytest`` and for the
    ``pytest tests`` CI split. A narrower run (``pytest kore``, one file) cannot,
    and skips instead of failing on tests it never looked at.
    """
    on_disk = {path.name for path in (ROOT / "tests").glob("test_*.py")}
    collected = {
        module.rsplit("/", 1)[-1]
        for modules in marker_modules.values()
        for module in modules
        if module.startswith("tests/")
    }
    return bool(on_disk) and on_disk <= collected


def _ini_options_block() -> str:
    """The raw ``[tool.pytest.ini_options]`` table text from ``pyproject.toml``.

    Read from the file rather than via ``config.getini``: ``getini("markers")``
    also returns pytest's own built-ins (``parametrize``, ``skipif``, ...) and
    every plugin-registered marker, and the contract here is specifically about
    what THIS project declares.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    header = "[tool.pytest.ini_options]"
    assert header in text, f"{PYPROJECT} has no {header} table"
    block = text.split(header, 1)[1]
    next_table = re.search(r"^\[", block, re.MULTILINE)
    return block[: next_table.start()] if next_table else block


def _declared_array(key: str) -> list[str]:
    """The double-quoted entries of a TOML array under ``[tool.pytest.ini_options]``."""
    block = _ini_options_block()
    match = re.search(rf"^{re.escape(key)}\s*=\s*\[(.*?)\]", block,
                      re.MULTILINE | re.DOTALL)
    assert match, f"[tool.pytest.ini_options] declares no {key} array"
    return re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1))


def _declared_markers() -> list[str]:
    """Marker names declared in ``pyproject.toml``, in declaration order."""
    return [entry.split(":", 1)[0].strip()
            for entry in _declared_array("markers")
            if entry.split(":", 1)[0].strip()]


def _default_mark_expression() -> str:
    """The ``-m`` expression the project applies by default, via ``addopts``."""
    addopts = _declared_array("addopts")
    for index, opt in enumerate(addopts):
        if opt == "-m" and index + 1 < len(addopts):
            return addopts[index + 1]
        if opt.startswith("-m") and len(opt) > 2:
            return opt[2:]
    return ""


def test_declared_markers_are_registered_with_pytest(
    request: pytest.FixtureRequest,
) -> None:
    """Every declared marker must be the kind ``--strict-markers`` accepts."""
    declared = _declared_markers()
    assert declared, "pyproject.toml declares no pytest markers"
    duplicates = sorted({name for name in declared if declared.count(name) > 1})
    assert not duplicates, f"markers declared more than once: {duplicates}"
    registered = {str(entry).split(":", 1)[0].strip()
                  for entry in request.config.getini("markers")}
    assert set(declared) <= registered, (
        f"pytest did not register {sorted(set(declared) - registered)}")


def test_every_declared_marker_selects_at_least_one_test(
    marker_census: dict[str, int],
    tests_tree_collected: bool,
) -> None:
    """A declared marker must select something, or it must be deleted.

    This is the check that would have caught the audited defect: ``gpu``,
    ``model``, ``network`` and ``dependency`` were all registered and all
    selected zero tests, so the deselection in ``addopts`` protected nothing.
    """
    if not tests_tree_collected:
        pytest.skip(_SKIP_PARTIAL_RUN)
    declared = _declared_markers()
    assert declared, "pyproject.toml declares no pytest markers"
    empty = sorted(name for name in declared if marker_census.get(name, 0) < 1)
    assert not empty, (
        f"markers declared in pyproject.toml that select no tests: {empty}. "
        "Either add tests carrying the marker or delete it from "
        "[tool.pytest.ini_options] markers (and from the addopts -m expression) "
        "- a marker that selects nothing makes its deselection a silent no-op."
    )


def test_default_deselection_only_names_declared_markers() -> None:
    """Every group the default run excludes must be a declared marker."""
    expression = _default_mark_expression()
    assert expression, "addopts no longer carries a default -m expression"
    excluded = set(re.findall(r"\bnot\s+([A-Za-z_]\w*)", expression))
    declared = set(_declared_markers())
    undeclared = sorted(excluded - declared)
    assert not undeclared, (
        f"the default -m expression excludes unregistered markers {undeclared}; "
        "--strict-markers cannot catch this because no test carries them"
    )


def test_default_run_excludes_every_resource_group() -> None:
    """The default suite stays CPU-only and free of release-only artifacts."""
    expression = _default_mark_expression()
    excluded = set(re.findall(r"\bnot\s+([A-Za-z_]\w*)", expression))
    missing = [name for name in REQUIRED_DESELECTED_MARKERS if name not in excluded]
    assert not missing, (
        f"markers {missing} must stay out of the default suite; add "
        f"'not <marker>' to the addopts -m expression (currently {expression!r})"
    )


def test_gpu_marker_selects_only_the_gpu_suites(
    marker_modules: dict[str, set[str]],
    tests_tree_collected: bool,
) -> None:
    """``pytest -m gpu`` selects the hardware suites and nothing else."""
    if not tests_tree_collected:
        pytest.skip(_SKIP_PARTIAL_RUN)
    modules = marker_modules.get("gpu", set())
    assert modules, "no test carries @pytest.mark.gpu"
    stray = sorted(
        path for path in modules
        if not path.rsplit("/", 1)[-1].startswith(GPU_MODULE_PREFIX)
    )
    assert not stray, (
        f"gpu-marked tests outside a {GPU_MODULE_PREFIX}*.py module: {stray}. "
        "Keep hardware tests in dedicated modules so the marker and the file "
        f"layout cannot drift apart - either rename the module or widen "
        f"GPU_MODULE_PREFIX in {Path(__file__).name}."
    )


def test_resource_markers_never_share_an_item_with_cpu(
    marker_combinations: set[frozenset],
    tests_tree_collected: bool,
) -> None:
    """``cpu`` and an opt-in resource group are mutually exclusive per test.

    ``conftest.pytest_collection_modifyitems`` adds ``cpu`` only to items with no
    resource marker.  If that ever inverts, a GPU test would be selected by the
    default CPU run.
    """
    if not tests_tree_collected:
        pytest.skip(_SKIP_PARTIAL_RUN)
    resource = set(REQUIRED_DESELECTED_MARKERS)
    overlapping = sorted(
        sorted(names & (resource | {"cpu"}))
        for names in marker_combinations
        if "cpu" in names and names & resource
    )
    assert not overlapping, (
        f"tests carry both cpu and a resource marker: {overlapping}")
