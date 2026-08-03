"""CPU-only tests for the HIP C++ backend and the HIP task family.

These cover the FAILURE MODES, because every one of them was a real bug or a real
near-miss during the work that added HIP support, and each fails silently or
misattributes blame if it regresses:

* a Triton task's candidate filename changing (1,334 drivers read ``kernel.py``);
* an unknown backend defaulting to ``kernel.py`` instead of failing closed;
* task staging dropping non-``.py`` assets, which is what made HIP impossible;
* a missing ninja/hipcc being charged to the model instead of to infra -- the
  failure that produces a ~100% error rate from a broken node;
* two concurrent workers sharing a build directory and importing each other's
  binary, which is a silent WRONG ANSWER rather than an error;
* ``//`` being stripped from Python (it is floor division) or ``#`` from C++ (it is
  the preprocessor, and stripping it hides ``#include <hipblaslt/...>``);
* a shape lane an op's own representation cannot express (MXFP4 needs N % 32 == 0);
* an op declared timing-inadmissible leaking into the registry anyway.

No GPU is required: nothing here compiles a kernel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from kore.env import hip_toolchain as hip
from kore.reward.reward import scan_for_hacks
from kore.tasks import generate_hip, taxonomy
from kore.tasks.hip_ops import HIP_OPS, MX_BLOCK, TORCH_DTYPES, seed_source
from kore.tasks.registry import all_tasks, get_task


# --------------------------------------------------------------------------- #
# Candidate filename: the Triton path must not move, HIP must be a .hip file
# --------------------------------------------------------------------------- #
def test_triton_candidate_filename_is_unchanged():
    """1,334 existing drivers read this exact name; moving it breaks all of them."""
    assert hip.candidate_filename("triton") == "kernel.py"
    assert hip.candidate_filename("TRITON") == "kernel.py"
    assert hip.candidate_filename(None) == "kernel.py"


def test_hip_candidate_is_a_hip_file():
    """torch.utils.cpp_extension dispatches the compiler off the EXTENSION, so a
    HIP candidate in a .py file cannot be compiled at all."""
    assert hip.candidate_filename("hip") == "kernel.hip"
    assert hip.candidate_filename("hip").endswith(".hip")


def test_unknown_backend_fails_closed():
    """Defaulting to kernel.py would stage a candidate the driver cannot compile
    and then report it as the model's compile failure."""
    with pytest.raises(hip.HipToolchainError) as excinfo:
        hip.candidate_filename("flydsl")
    assert "flydsl" in str(excinfo.value)
    assert "triton" in str(excinfo.value)  # the message names what IS supported


def test_source_language_selects_the_comment_syntax():
    assert hip.source_language("triton") == "python"
    assert hip.source_language("hip") == "cpp"
    assert hip.source_language("unknown-backend") == "python"  # strictest default


# --------------------------------------------------------------------------- #
# Architecture handling
# --------------------------------------------------------------------------- #
def test_gpu_arch_strips_rocm_feature_suffixes():
    """ROCm reports gfx950:sramecc+:xnack-; PYTORCH_ROCM_ARCH wants the bare arch,
    and hipcc fails on the decorated form."""
    assert hip.gpu_arch("gfx950:sramecc+:xnack-") == "gfx950"
    assert hip.gpu_arch("gfx950") == "gfx950"
    assert hip.gpu_arch("") == ""
    assert hip.gpu_arch(None) == ""


def test_compile_environment_pins_the_arch_and_finds_ninja():
    """Unpinned, hipcc fat-binaries for every supported arch: measured 114.6s
    versus 15.4s for one trivial kernel."""
    env = hip.compile_environment({"PATH": "/usr/bin"}, "gfx950:sramecc+:xnack-")
    assert env["PYTORCH_ROCM_ARCH"] == "gfx950"
    assert env["MAX_JOBS"] == hip.DEFAULT_MAX_JOBS
    assert "TORCH_EXTENSIONS_DIR" in env
    entries = env["PATH"].split(os.pathsep)
    for directory in hip.script_dirs():
        assert directory in entries, "the venv bin/ must be reachable for ninja"


def test_compile_environment_never_overrides_an_explicit_parent_setting():
    env = hip.compile_environment(
        {"PATH": "/usr/bin", "PYTORCH_ROCM_ARCH": "gfx942", "MAX_JOBS": "1"}, "gfx950")
    assert env["PYTORCH_ROCM_ARCH"] == "gfx942"
    assert env["MAX_JOBS"] == "1"


def test_script_dirs_stays_inside_the_virtualenv():
    """sys.executable resolves OUT of this venv (bin/python -> /usr/local/bin), so
    resolving it loses bin/ninja entirely."""
    dirs = hip.script_dirs()
    assert dirs, "at least one script directory must be discoverable"
    assert str(Path(sys.prefix) / "bin") in dirs or str(Path(sys.executable).parent) in dirs


# --------------------------------------------------------------------------- #
# Build isolation: a shared build dir is a silent wrong answer
# --------------------------------------------------------------------------- #
def test_extension_name_is_content_addressed():
    """load(name=N) builds in TORCH_EXTENSIONS_DIR/N. Two workers compiling
    DIFFERENT candidates under one name can import each other's .so, which is a
    wrong answer rather than an error."""
    a = hip.extension_name("__global__ void k() {}")
    b = hip.extension_name("__global__ void k() { }")
    assert a != b, "distinct sources must never share a build directory"
    assert a == hip.extension_name("__global__ void k() {}"), "identical sources cache"
    assert a.startswith("kore_hip_")
    assert a.replace("kore_hip_", "").isalnum()


# --------------------------------------------------------------------------- #
# Toolchain absence is INFRA, never the kernel's fault
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "RuntimeError: Ninja is required to load C++ extensions",
    f"{hip.TOOLCHAIN_MARKER}: missing ninja",
    "FileNotFoundError: [Errno 2] No such file or directory: 'hipcc'",
    "hipcc: command not found",
])
def test_toolchain_absence_is_recognized(text):
    assert hip.TOOLCHAIN_ABSENCE_PATTERN.search(text), text


@pytest.mark.parametrize("text", [
    "error: no matching function for call to 'bad_symbol'",
    "kernel.hip:12:5: error: use of undeclared identifier 'foo'",
    "SNR: -999.00 dB",
])
def test_a_real_compile_error_is_not_mistaken_for_a_toolchain_fault(text):
    """A kernel that does not compile must stay the kernel's fault; laundering it
    into infra would hide genuinely broken candidates from the error rate."""
    assert not hip.TOOLCHAIN_ABSENCE_PATTERN.search(text), text


def test_environment_classifier_routes_toolchain_absence_to_infra():
    from kore.env.kore_env import KoreEnv

    task = get_task("hip_silu_bf16")
    env = KoreEnv.__new__(KoreEnv)          # no GPU / no task staging needed
    env._sandbox_enabled = False
    kind, message = KoreEnv._classify(
        env, "Traceback (most recent call last):\n"
             "RuntimeError: Ninja is required to load C++ extensions\n", 1, False)
    assert kind == "infra", "a broken node must not be reported as a model failure"
    assert "toolchain" in message
    assert task.backend == "hip"


def test_environment_classifier_still_blames_a_broken_kernel():
    from kore.env.kore_env import KoreEnv

    env = KoreEnv.__new__(KoreEnv)
    env._sandbox_enabled = False
    kind, _ = KoreEnv._classify(
        env, "kernel.hip:9:3: error: use of undeclared identifier 'nope'\n", 1, False)
    assert kind == "compile"


# --------------------------------------------------------------------------- #
# Task staging must carry non-.py assets
# --------------------------------------------------------------------------- #
def test_task_source_files_include_hip_assets_and_exclude_the_candidate(tmp_path):
    from kore.env.kore_env import _task_source_files

    class _Task:
        dir = tmp_path

    (tmp_path / "reference.py").write_text("x = 1\n")
    (tmp_path / "seed_hip.hip").write_text("// seed\n")
    (tmp_path / "helper.h").write_text("// header\n")
    (tmp_path / "notes.md").write_text("ignored\n")
    # A stray candidate artifact must never shadow the one the environment writes.
    (tmp_path / "kernel.hip").write_text("// stale candidate\n")
    (tmp_path / "kernel.py").write_text("# stale candidate\n")

    names = {p.name for p in _task_source_files(_Task())}
    assert names == {"reference.py", "seed_hip.hip", "helper.h"}


def test_staged_suffixes_cover_the_hip_toolchain_inputs():
    for suffix in (".py", ".hip", ".h", ".hpp"):
        assert suffix in hip.STAGED_SUFFIXES


# --------------------------------------------------------------------------- #
# Language-aware hack scan
# --------------------------------------------------------------------------- #
def test_cpp_comments_are_stripped_so_a_comment_cannot_be_a_hack():
    """Measured over the 89 real HIP kernels in AgentKernelArena, 7 were rejected
    purely because a comment NAMED a vendor library the code never called."""
    source = """
// tuned to match the hipblaslt tile layout
/* layout follows aiter.gemm_a4w4 */
#include <hip/hip_runtime.h>
__global__ void k(float* x) { x[0] = 1.0f; }
"""
    assert scan_for_hacks(source, "cpp") is None
    assert scan_for_hacks(source, "python") is not None, (
        "python mode must stay strict: it cannot know // is a comment")


def test_cpp_mode_still_rejects_a_real_vendor_include():
    """`#` is the PREPROCESSOR in C++. Stripping it would delete the include and
    hide the delegation this scan exists to catch."""
    source = """
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
torch::Tensor forward(torch::Tensor a) { return a; }
"""
    assert scan_for_hacks(source, "cpp") is not None


def test_cpp_mode_still_rejects_verdict_forgery_and_oracle_access():
    forgery = 'printf("SNR: 999.00 dB\\n");'
    assert scan_for_hacks(forgery, "cpp") is not None
    assert scan_for_hacks('#include "reference.py"\nvoid f(){}', "cpp") is not None


def test_floor_division_is_not_treated_as_a_comment_in_python():
    """If `//` were stripped from Python, a delegation to the RIGHT of the operator
    would vanish from the scan."""
    source = "def f(a, b):\n    return a // torch.matmul(a, b)\n"
    assert scan_for_hacks(source, "python") is not None
    assert scan_for_hacks(source) is not None, "the default must be the strict mode"


def test_scan_default_language_is_python():
    """Every existing call site passes one argument; the default must not change."""
    assert scan_for_hacks("y = torch.matmul(a, b)") is not None


# --------------------------------------------------------------------------- #
# The seeds themselves must satisfy the gate a candidate must satisfy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op_id", sorted(HIP_OPS))
def test_every_hip_seed_passes_the_scanner_and_binds_forward(op_id):
    source = seed_source(op_id)
    assert scan_for_hacks(source, "cpp") is None, (
        f"{op_id}: a seed the scanner rejects is a task no kernel can score on")
    assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME" in source
    assert 'm.def("forward"' in source


@pytest.mark.parametrize("op_id", sorted(HIP_OPS))
def test_no_hip_seed_includes_a_vendor_math_library(op_id):
    assert generate_hip._FORBIDDEN_INCLUDES.search(seed_source(op_id)) is None


def test_generator_rejects_a_seed_that_reads_the_environment(monkeypatch):
    """A kernel that reads getenv can behave differently while timed. Two of the 89
    AgentKernelArena HIP kernels do exactly this for autotuning knobs, so a seed
    mined from there without cleaning would be rejected -- correctly."""
    poisoned = seed_source("silu").replace(
        "namespace {", 'namespace {\nstatic const char* k = getenv("KORE_MODE");', 1)
    monkeypatch.setattr(generate_hip, "seed_source", lambda op_id: poisoned)
    with pytest.raises(generate_hip.HipTaskError, match="scanner"):
        generate_hip._validated_seed("silu", "bf16")


def test_generator_rejects_a_seed_with_no_forward_binding(monkeypatch):
    monkeypatch.setattr(generate_hip, "seed_source", lambda op_id: "int main(){}")
    with pytest.raises(generate_hip.HipTaskError, match="PYBIND11_MODULE"):
        generate_hip._validated_seed("silu", "bf16")


# --------------------------------------------------------------------------- #
# Shape constraints an op's own representation imposes
# --------------------------------------------------------------------------- #
def test_mxfp4_declares_its_block_constraint():
    spec = HIP_OPS["dequant_mxfp4"]
    assert spec.dim_multiples == {"N": MX_BLOCK}


@pytest.mark.parametrize("op_id", sorted(HIP_OPS))
def test_every_declared_shape_satisfies_the_ops_own_constraints(op_id):
    """An MXFP4 lane with N=8193 compiles, passes a minimal/primary spot check, and
    throws at datagen on a validation shape. That happened; this is the gate."""
    generate_hip._check_shape_constraints(op_id, HIP_OPS[op_id])


def test_illegal_mx_lane_is_rejected():
    spec = HIP_OPS["dequant_mxfp4"]

    class _Bad:
        shapes = {"minimal": {"M": 8, "N": 8193}}
        dim_multiples = spec.dim_multiples

    with pytest.raises(generate_hip.HipTaskError, match="multiple of 32"):
        generate_hip._check_shape_constraints("dequant_mxfp4", _Bad())


# --------------------------------------------------------------------------- #
# Registry integration
# --------------------------------------------------------------------------- #
def test_hip_tasks_are_registered_and_trainable():
    hip_tasks = [t for t in all_tasks() if t.backend == "hip"]
    assert hip_tasks, "the HIP task family must be discoverable by the registry"
    from kore.tasks.registry import split_decision

    for task in hip_tasks:
        decision = split_decision(task)
        assert decision.split == "train", (
            f"{task.task_id}: {decision.reason} -- a new dtype or an unreviewed "
            "operation silently makes a task eval-only")
        assert task.dtype in taxonomy.TRAIN_DTYPES
        assert task.gpu_target in taxonomy.TRAIN_ARCHITECTURES
        assert task.seed_kernel_name.endswith(".hip")
        assert task.seed_path.is_file()


def test_hip_operations_are_reviewed_in_the_taxonomy():
    """A hand-authored task whose operation is absent from HAND_OPERATION_FAMILIES
    aborts the whole registry, so this is what keeps the failure legible."""
    for task in (t for t in all_tasks() if t.backend == "hip"):
        assert task.operation in taxonomy.HAND_OPERATION_FAMILIES
        assert taxonomy.product_family_for_name(task.operation) == \
            taxonomy.HAND_OPERATION_FAMILIES[task.operation], (
                f"{task.operation}: the exact map and the name adapter disagree")


def test_hip_tasks_use_the_publication_driver():
    """A hand-written driver that does not advertise KORE_DRIVER_CAPABILITIES is
    performance-INELIGIBLE, which silently costs the task its speedup reward."""
    for task in (t for t in all_tasks() if t.backend == "hip"):
        driver = (task.dir / "driver.py").read_text()
        assert "from kore.tasks._genops import driver_main" in driver


def test_timing_inadmissible_ops_are_not_generated_by_default():
    """An op that cannot be timed returns a performance-ineligible episode every
    run: real GPU cost, no speedup signal."""
    inadmissible = {op for op, spec in HIP_OPS.items() if not spec.timing_admissible}
    assert inadmissible, "gemm is expected here; see its measured timing_note"
    registered = {t.operation for t in all_tasks() if t.backend == "hip"}
    for op_id in inadmissible:
        assert generate_hip.operation_id(op_id) not in registered
        assert HIP_OPS[op_id].timing_note, (
            f"{op_id}: an op excluded on measurement must record the measurement")


def test_low_precision_coverage_exists_and_declares_real_dtypes():
    hip_dtypes = {t.dtype for t in all_tasks() if t.backend == "hip"}
    assert "fp8_e4m3fn" in hip_dtypes
    assert "mxfp4" in hip_dtypes
    for dtype_id in hip_dtypes:
        assert dtype_id in TORCH_DTYPES


def test_no_mxfp6_task_is_claimed():
    """torch.float6_e2m3fn / float6_e3m2fn do not exist in this stack. A task built
    on a hand-rolled 6-bit packing would have no library-corroborated oracle."""
    import torch

    assert not hasattr(torch, "float6_e2m3fn")
    assert not hasattr(torch, "float6_e3m2fn")
    assert not any(
        "fp6" in t.dtype or "mxfp6" in t.dtype for t in all_tasks())


# --------------------------------------------------------------------------- #
# Decontamination must see HIP sources
# --------------------------------------------------------------------------- #
def test_heldout_index_covers_hip_sources():
    """`*.py` alone was a hole: a held-out HIP task's kernel lives in a .hip file,
    so its text was invisible to every leakage check."""
    from kore.data.decontam import HELDOUT_SOURCE_SUFFIXES

    for suffix in (".py", ".hip", ".cpp", ".cu"):
        assert suffix in HELDOUT_SOURCE_SUFFIXES
