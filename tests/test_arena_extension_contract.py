"""The prompt must state the extension contract when the target ships empty.

torch2hip scored 0 of 62 across both arena arms while hip2hip, on the same
toolchain and the same compiler, scored normally. The difference was not the
model: all 57 torch2hip targets ship as zero-byte .hip files, so there is no
existing interface to copy, and the loader the task generates reads
``ext.forward`` off the built extension. A candidate that contains only a
__global__ kernel has no pybind11 module, so the build fails before numerics are
ever considered. hip2hip targets all ship non-empty, so there the boilerplate is
visible in the file being rewritten and gets preserved.

These pin that the contract is stated exactly when it is needed and derived from
the task rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_agent_kernel_arena import (  # noqa: E402
    _extension_contract, _loader_entry_point)


class _Task:
    def __init__(self, root: Path):
        self.root = root


def _with_loader(tmp_path: Path, attr: str = "forward") -> _Task:
    tools = tmp_path / "eval_tools"
    tools.mkdir(parents=True, exist_ok=True)
    (tools / "kernel_loader_template.py").write_text(
        "kernel_loader_template = '''\n"
        "from torch.utils.cpp_extension import load\n"
        'hip_{kernel_name}_ext = load(name="{kernel_name}",\n'
        '                             sources=["{code_dir}/{code_file}"])\n'
        f"hip_fn = hip_{{kernel_name}}_ext.{attr}\n"
        "'''\n"
    )
    return _Task(tmp_path)


def test_entry_point_is_read_from_the_task_not_assumed(tmp_path):
    assert _loader_entry_point(_with_loader(tmp_path, "forward")) == "forward"


def test_a_task_with_a_different_entry_point_is_reported_as_such(tmp_path):
    """Hardcoding "forward" would mislead the first task that differs."""
    assert _loader_entry_point(_with_loader(tmp_path, "run_kernel")) == "run_kernel"


def test_no_loader_template_means_no_claim_about_the_contract(tmp_path):
    assert _loader_entry_point(_Task(tmp_path)) == ""


def test_empty_hip_target_gets_the_contract(tmp_path):
    """The torch2hip case: 57 of 57 targets are zero bytes."""
    ws = tmp_path / "ws"
    (ws / "hip").mkdir(parents=True)
    (ws / "hip" / "k.hip").write_text("")
    note = _extension_contract(ws, "hip/k.hip", _with_loader(tmp_path))
    assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME" in note
    assert 'm.def("forward"' in note
    assert "torch/extension.h" in note
    assert "hip/hip_runtime.h" in note


def test_missing_target_file_also_gets_the_contract(tmp_path):
    """Absent and empty are the same problem: nothing to copy an interface from."""
    ws = tmp_path / "ws"
    ws.mkdir()
    note = _extension_contract(ws, "hip/k.hip", _with_loader(tmp_path))
    assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME" in note


def test_non_empty_hip_target_gets_no_contract(tmp_path):
    """hip2hip: the file already shows the contract, so restating it wastes window."""
    ws = tmp_path / "ws"
    (ws / "hip").mkdir(parents=True)
    (ws / "hip" / "k.hip").write_text(
        "#include <torch/extension.h>\n"
        "torch::Tensor forward(torch::Tensor x) { return x; }\n"
        "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"forward\", &forward); }\n"
    )
    assert _extension_contract(ws, "hip/k.hip", _with_loader(tmp_path)) == ""


@pytest.mark.parametrize("rel", ["mod.py", "kernel.py", "harness.py"])
def test_python_targets_never_get_a_cpp_contract(tmp_path, rel):
    """triton2triton and instruction2triton write Python; this must not fire."""
    ws = tmp_path / "ws"
    ws.mkdir()
    assert _extension_contract(ws, rel, _with_loader(tmp_path)) == ""


def test_whitespace_only_target_counts_as_empty(tmp_path):
    ws = tmp_path / "ws"
    (ws / "hip").mkdir(parents=True)
    (ws / "hip" / "k.hip").write_text("\n\n   \n")
    assert "PYBIND11_MODULE" in _extension_contract(ws, "hip/k.hip",
                                                   _with_loader(tmp_path))


def test_contract_names_the_task_specific_entry_point(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    note = _extension_contract(ws, "k.hip", _with_loader(tmp_path, "apply"))
    assert 'm.def("apply"' in note
    assert "ext.apply" in note
