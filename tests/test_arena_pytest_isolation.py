"""A task's tests must not be filtered by this repository's pytest config.

Workspaces live inside this repo, so pytest adopts our pyproject.toml as rootdir
config -- and our addopts carry `-m "not gpu and not release"`. On a GPU benchmark
most tests are marked gpu, so they were silently deselected and a task whose tests
were all GPU tests collected nothing and scored incorrect without being run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_agent_kernel_arena import _WORKSPACE_PYTEST_INI  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TWO_TESTS = (
    "import pytest\n\n"
    "@pytest.mark.gpu\n"
    "def test_on_gpu():\n    assert True\n\n"
    "def test_plain():\n    assert True\n"
)


def _run(ws: Path) -> str:
    p = subprocess.run([sys.executable, "-m", "pytest", "softmax.py"],
                       cwd=ws, capture_output=True, text=True)
    return (p.stdout or "") + (p.stderr or "")


def test_our_marker_filter_deselects_a_task_gpu_test(tmp_path_factory):
    """The bug, pinned: inside this repo and without isolation, the gpu test is
    dropped. If this ever stops being true our addopts changed and the fix below is
    no longer load-bearing."""
    ws = REPO / "runs" / "_pytest_isolation_probe_a"
    ws.mkdir(parents=True, exist_ok=True)
    try:
        (ws / "softmax.py").write_text(TWO_TESTS)
        assert "deselected" in _run(ws)
    finally:
        __import__("shutil").rmtree(ws, ignore_errors=True)


def test_the_workspace_ini_lets_every_task_test_run():
    ws = REPO / "runs" / "_pytest_isolation_probe_b"
    ws.mkdir(parents=True, exist_ok=True)
    try:
        (ws / "softmax.py").write_text(TWO_TESTS)
        (ws / "pytest.ini").write_text(_WORKSPACE_PYTEST_INI)
        out = _run(ws)
        assert "2 passed" in out
        assert "deselected" not in out
    finally:
        __import__("shutil").rmtree(ws, ignore_errors=True)


def test_the_ini_clears_inherited_addopts():
    assert "addopts =" in _WORKSPACE_PYTEST_INI
    body = _WORKSPACE_PYTEST_INI.split("addopts =", 1)[1].strip()
    assert body == "", "addopts must be emptied, not extended"
