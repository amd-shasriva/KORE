"""Both drivers must load the candidate the task actually staged.

A task directory holds exactly one candidate, and its language depends on the
backend: Triton and FlyDSL stage ``kernel.py``, HIP stages ``kernel.hip``. Two
driver families read that directory -- ``_genops`` for generated pool ops and
``_training_common`` for the registry's hand-authored tasks -- and only the
first knew to look for a ``.hip``.

So a HIP twin of a registry task died in the loader, on a missing kernel.py,
before anything was compiled, and the gate recorded compile_or_run_fail as
though the teacher's seed were at fault. It was 306 of the first 331 frontier
twins: flash attention, fused MoE and fp8 GEMM thrown away over a filename,
which is precisely the set the whole frontier effort exists to produce.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kore.env.hip_toolchain import (  # noqa: E402
    CANDIDATE_FILENAMES, HIP_BACKEND, HipCandidateModule, TRITON_BACKEND,
    load_candidate_module)


def test_python_candidate_is_imported(tmp_path):
    (tmp_path / "kernel.py").write_text("def entry(x):\n    return x + 1\n")
    mod = load_candidate_module(tmp_path)
    assert mod.entry(1) == 2


def test_hip_candidate_is_not_imported_as_python(tmp_path):
    """The case that was failing: only a .hip is staged."""
    (tmp_path / "kernel.hip").write_text("// a HIP kernel, not Python\n")
    mod = load_candidate_module(tmp_path)
    assert isinstance(mod, HipCandidateModule)


def test_hip_wins_when_both_are_present(tmp_path):
    (tmp_path / "kernel.hip").write_text("// hip\n")
    (tmp_path / "kernel.py").write_text("entry = None\n")
    assert isinstance(load_candidate_module(tmp_path), HipCandidateModule)


def test_hip_compile_is_deferred_to_first_use(tmp_path, monkeypatch):
    """A compile error must surface where the Triton import error would, not at
    load time -- the tier the failure is attributed to depends on it."""
    (tmp_path / "kernel.hip").write_text("// hip\n")
    mod = load_candidate_module(tmp_path)  # must not raise, no toolchain needed

    called = {}

    def fake(task_dir, entry, *, gpu_target=None):
        called["entry"] = entry
        return lambda: "compiled"

    monkeypatch.setattr("kore.env.hip_toolchain.load_hip_candidate", fake)
    assert mod.some_entry() == "compiled"
    assert called["entry"] == "some_entry"


def test_dunder_lookups_do_not_trigger_a_compile(tmp_path):
    """copy/pickle/inspect probe _-prefixed attributes; compiling for those
    would turn an unrelated introspection into a toolchain error."""
    (tmp_path / "kernel.hip").write_text("// hip\n")
    mod = load_candidate_module(tmp_path)
    with pytest.raises(AttributeError):
        mod._not_an_entry


def test_flydsl_stages_python(tmp_path):
    """FlyDSL needs no branch: its candidate is Python."""
    assert CANDIDATE_FILENAMES["flydsl"] == CANDIDATE_FILENAMES[TRITON_BACKEND]
    assert CANDIDATE_FILENAMES[HIP_BACKEND] == "kernel.hip"


# ---- both drivers must go through it --------------------------------------

@pytest.mark.parametrize("module", [
    "kore/tasks/_genops.py",
    "kore/tasks/_training_common.py",
    "kore/tasks/_attn_common.py",
    "kore/tasks/_moe_common.py",
    "kore/tasks/_quant_common.py",
    "kore/verify/runner.py",
])
def test_driver_families_share_the_dispatch(module):
    src = (Path(__file__).resolve().parents[1] / module).read_text()
    assert "load_candidate_module" in src, f"{module} does not share the dispatch"


def test_no_live_loader_hardcodes_kernel_py():
    """There were six driver families and only one knew about .hip. Fixing them
    one at a time is how flash-attention backward started passing while
    chunked-prefill went on failing in a different file for the same reason."""
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in list((root / "kore").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if "_drafts/" in rel:        # not imported at runtime
            continue
        src = path.read_text(errors="ignore")
        if 'os.path.join(task_dir, "kernel.py")' in src or \
           'os.path.abspath(__file__)), "kernel.py")' in src:
            offenders.append(rel)
    assert not offenders, f"still loading Python unconditionally: {offenders[:5]}"


def test_twin_drivers_are_kept_in_sync(tmp_path):
    """A twin copies its driver once, at seed time, so a later fix to that
    driver never reaches twins already on disk -- 34 of them went on failing in
    a loader that had already been fixed."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import refresh_twin_drivers as R

    repo = Path(__file__).resolve().parents[1]
    src = repo / "kore" / "tasks" / "flash_attn_decode_bf16"
    if not (src / "driver.py").is_file():
        pytest.skip("registry task not present")
    twin = tmp_path / "tasks" / "flash_attn_decode_bf16__hip"
    twin.mkdir(parents=True)
    (twin / "driver.py").write_text("# stale\n")

    import os
    rel = os.path.relpath(tmp_path, repo)
    assert R.refresh([rel], check=True) == (1, 0), "drift not detected"
    assert R.refresh([rel], check=False) == (1, 1), "drift not repaired"
    assert (twin / "driver.py").read_text() == (src / "driver.py").read_text()


def test_refresher_leaves_the_oracle_alone(tmp_path):
    """reference.py is generated for a functionalized twin; copying the
    source's over it would change what the twin is graded against."""
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "refresh_twin_drivers.py").read_text()
    assert "reference.py" not in src.split('"""', 2)[2], \
        "the refresher touches reference.py"


def test_pipeline_refreshes_twin_drivers():
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "frontier_pipeline.sh").read_text()
    assert "refresh_twin_drivers.py" in src


def test_per_task_drivers_were_all_converted():
    """Each hand-authored task carries its own loader, and a HIP twin copies it
    verbatim -- so one unconverted driver means that task's twin is ungradeable."""
    root = Path(__file__).resolve().parents[1] / "kore" / "tasks"
    stale = [p.parent.name for p in root.glob("*/driver.py")
             if 'spec_from_file_location("candidate_kernel"' in p.read_text(errors="ignore")]
    assert not stale, f"task drivers still importing Python directly: {stale[:5]}"
