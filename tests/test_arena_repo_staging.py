"""Making a task's repository checkout reachable on bare metal.

The image_kernel and repository tasks are edits inside a real library. Every one of
them scored zero *compiled* in the v4 run, and none was judged on its code: one
family looked for a directory whose name differs only in case from the checkout, and
the other pointed at an absolute path that exists only inside the arena's container.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.run_agent_kernel_arena as aka  # noqa: E402


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    root = tmp_path / "third_party"
    (root / "rocPRIM" / "rocprim").mkdir(parents=True)
    (root / "rocPRIM" / "rocprim" / "block.hpp").write_text("// rocprim\n")
    (root / "aiter" / "csrc").mkdir(parents=True)
    (root / "aiter" / "csrc" / "k.cu").write_text("// aiter\n")
    monkeypatch.setattr(aka, "_REPO_CACHE", root)
    return root


def test_checkout_resolves_across_capitalisation(cache):
    """`repository/rocprim` declares sources as rocprim/... and then asks for a
    directory called rocPRIM; the run failed on "Source directory not found"."""
    assert aka._resolve_checkout("rocPRIM") == cache / "rocPRIM"
    assert aka._resolve_checkout("rocprim") == cache / "rocPRIM"
    assert aka._resolve_checkout("ROCPRIM") == cache / "rocPRIM"
    assert aka._resolve_checkout("not_a_repo") is None


def test_container_repo_path_is_rewritten_to_a_local_checkout(cache, tmp_path):
    """image_kernel declares /sgl-workspace/aiter, a path inside the arena's Docker
    image. On bare metal it cannot even be created, so the task's runner resolved an
    empty repo root and died in os.chdir('') before reading any kernel."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config.yaml").write_text(
        "task_type: image_kernel\n"
        "image_repo_path: /sgl-workspace/aiter\n"
        "repository_language: triton\n")

    assert aka._provide_task_repo(ws) is None
    text = (ws / "config.yaml").read_text()
    assert "/sgl-workspace/aiter" not in text
    assert str(ws / "aiter") in text
    # The checkout is staged inside the workspace, so the task edits its own copy.
    assert (ws / "aiter" / "csrc" / "k.cu").is_file()
    # Other keys survive the rewrite.
    assert "repository_language: triton" in text
    assert "task_type: image_kernel" in text


def test_a_repo_path_that_already_exists_is_left_alone(cache, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    real = tmp_path / "already_here"
    real.mkdir()
    (ws / "config.yaml").write_text(f"image_repo_path: {real}\n")
    assert aka._provide_task_repo(ws) is None
    assert (ws / "config.yaml").read_text() == f"image_repo_path: {real}\n"


def test_a_repo_we_do_not_have_is_reported_not_silently_skipped(cache, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config.yaml").write_text("image_repo_path: /sgl-workspace/nope\n")
    err = aka._provide_task_repo(ws)
    assert err and "nope" in err


def test_a_leading_path_segment_that_is_not_a_repo_is_not_fatal(cache, tmp_path):
    """aiter's own sources are declared as csrc/... and sglang's as python/sglang/...,
    so the first path segment is often a directory inside a repo rather than a repo.
    Treating that as a missing checkout aborted tasks that were fine."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config.yaml").write_text("image_repo_path: /sgl-workspace/aiter\n")

    class T:
        source_files = ["csrc/py_itfs_ck/mha_batch_prefill_kernels.cu"]

    assert aka._link_required_repo(T(), ws) is None
    assert (ws / "aiter").is_dir()      # supplied via repo_path, not the segment


def test_staging_falls_back_to_copy_when_links_cannot_cross_filesystems(
        cache, tmp_path, monkeypatch):
    """A hard link cannot cross filesystems. With the workspace on node-local disk
    and the checkout on /home, every file raises EXDEV and staging fails entirely --
    so linking has to be an optimisation, not a requirement."""
    def _no_links(src, dst):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(aka.os, "link", _no_links)
    dst = tmp_path / "staged"
    aka._stage_repo(cache / "aiter", dst)
    assert (dst / "csrc" / "k.cu").read_text() == "// aiter\n"


def test_staging_is_idempotent(cache, tmp_path):
    """Workspaces are rebuilt per attempt; re-staging must not fail or duplicate."""
    dst = tmp_path / "staged"
    aka._stage_repo(cache / "aiter", dst)
    aka._stage_repo(cache / "aiter", dst)          # must be a no-op
    assert (dst / "csrc" / "k.cu").is_file()


def test_repo_url_stages_the_clone_the_task_expects(cache, tmp_path):
    """`repository` tasks declare repo_url and expect the clone already present under
    the repository's own name, failing with "Source directory not found: <ws>/rocPRIM".
    Nothing clones it during a run."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config.yaml").write_text(
        "repo_url: https://github.com/ROCm/rocPRIM.git\n"
        "task_type: repository\n")
    assert aka._provide_task_repo(ws) is None
    assert (ws / "rocPRIM" / "rocprim" / "block.hpp").is_file()


def test_the_key_is_image_repo_path_not_repo_path(cache, tmp_path):
    """Anchoring on `repo_path:` matched nothing, while grepping for it matched
    `image_repo_path` as a substring -- which made a broken rewrite look correct."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config.yaml").write_text("image_repo_path: /sgl-workspace/aiter\n")
    assert aka._provide_task_repo(ws) is None
    assert "/sgl-workspace" not in (ws / "config.yaml").read_text()
    assert str(ws / "aiter") in (ws / "config.yaml").read_text()
