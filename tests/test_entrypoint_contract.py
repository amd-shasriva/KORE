"""Syntax and side-effect-free ``--help`` smoke tests for entrypoints."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# amd-burst-qos was deleted cluster-side (173dbbf); the General team's allocation is
# eight guaranteed non-preemptible nodes on amd-general-qos. A job that still asks
# for the removed QoS is rejected at submission.
REMOVED_QOS = "amd-burst-qos"

# Deployment root of the SPUR checkout. The hardcoded paths in the sbatch files are
# only meaningful on that host, so the check that uses this skips elsewhere.
DEPLOYMENT_PARENT = Path("/home/shasriva/Kore-RL")

# Each script below constructs argparse before doing operational work. Scripts
# that intentionally run immediately (gateway/GPU/supervisor probes) are syntax
# checked but are not safe to execute as a help probe.
SAFE_SCRIPT_HELP = (
    "verify_breadth.py",
    "sft_smoke.py",
    "grpo_smoke.py",
    "spur_partition.py",
    "run_campaign.py",
    "merge_datagen_roots.py",
    "run_sft_gate.py",
    "spur_supervise_datagen.py",
    "deepen_wins.py",
    "eval_bakeoff_multi.py",
    "complete_base.py",
    "_kf_verify.py",
)

CLI_SUBCOMMANDS = (
    "tasks",
    "datagen",
    "build-datasets",
    "sft",
    "dpo",
    "grpo",
    "value-train",
    "eval",
)


def _help(command: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{' '.join(command)} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "usage:" in (result.stdout + result.stderr).lower()


@pytest.mark.shell
def test_all_shell_entrypoints_parse() -> None:
    errors: list[str] = []
    shell_files = sorted((*SCRIPTS.glob("*.sh"), *SCRIPTS.glob("*.sbatch")))
    assert shell_files
    for path in shell_files:
        result = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=10,
        )
        if result.returncode:
            errors.append(f"{path.relative_to(ROOT)}: {result.stderr.strip()}")
    assert not errors, "\n".join(errors)


@pytest.mark.shell
def test_no_shell_entrypoint_is_empty() -> None:
    # ``bash -n`` accepts an empty file and ``sbatch`` rejects one, so a launcher
    # that was committed at 0 bytes stays green in CI and fails only at submission.
    empty = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (*SCRIPTS.glob("*.sh"), *SCRIPTS.glob("*.sbatch"))
        if path.stat().st_size == 0
    )
    assert not empty, f"empty launchers cannot be submitted: {empty}"


@pytest.mark.shell
def test_no_sbatch_requests_the_removed_burst_qos() -> None:
    offenders = sorted(
        path.relative_to(ROOT).as_posix()
        for path in SCRIPTS.glob("*.sbatch")
        if re.search(rf"^#SBATCH\s+--qos={re.escape(REMOVED_QOS)}\s*$",
                     path.read_text(encoding="utf-8"), re.M)
    )
    assert not offenders, f"{REMOVED_QOS} no longer exists on SPUR: {offenders}"


@pytest.mark.shell
def test_midtrain_launchers_default_to_their_node_count_config() -> None:
    # A launcher that allocates N nodes but defaults to a config tuned for a
    # different world size trains at the wrong effective batch size, and the launch
    # configs that no launcher names go stale unnoticed.
    launchers = sorted(SCRIPTS.glob("spur_midtrain_*.sbatch"))
    assert launchers
    for path in launchers:
        source = path.read_text(encoding="utf-8")
        nodes = re.search(r"^#SBATCH --nodes=(\d+)", source, re.M)
        assert nodes, f"{path.name}: no #SBATCH --nodes directive"
        default = re.search(r'^CFG="\$\{1:-\$REPO/([^}"]+)\}"', source, re.M)
        assert default, f"{path.name}: no ${{1}}-overridable CFG default"
        config = ROOT / default.group(1)
        assert config.is_file(), f"{path.name}: default config {default.group(1)} is absent"
        assert config.name == f"midtrain_{int(nodes.group(1)) * 8}gpu.json", (
            f"{path.name} allocates {nodes.group(1)} nodes x 8 GPUs "
            f"but defaults to {config.name}"
        )


@pytest.mark.shell
@pytest.mark.skipif(
    not DEPLOYMENT_PARENT.is_dir(),
    reason=f"{DEPLOYMENT_PARENT} is specific to the SPUR submit host",
)
def test_hardcoded_deployment_paths_exist() -> None:
    # spur_gpu_smoke.sbatch pointed every path at a misspelled worktree that had
    # since been deleted, so the job could only ever fail on the node.
    pattern = re.compile(rf"{re.escape(str(DEPLOYMENT_PARENT))}/[A-Za-z0-9_.-]+")
    missing: dict[str, list[str]] = {}
    for path in sorted((*SCRIPTS.glob("*.sh"), *SCRIPTS.glob("*.sbatch"))):
        absent = sorted({
            hit for hit in pattern.findall(path.read_text(encoding="utf-8"))
            if not Path(hit).exists()
        })
        if absent:
            missing[path.name] = absent
    assert not missing, f"launchers reference checkouts that do not exist: {missing}"


@pytest.mark.shell
def test_all_python_entrypoints_parse_as_ast() -> None:
    errors: list[str] = []
    python_files = sorted(SCRIPTS.glob("*.py"))
    assert python_files
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    assert not errors, "\n".join(errors)


@pytest.mark.shell
@pytest.mark.parametrize("script", SAFE_SCRIPT_HELP)
def test_safe_script_help(script: str) -> None:
    _help([sys.executable, str(SCRIPTS / script), "--help"])


@pytest.mark.shell
def test_package_cli_help() -> None:
    _help([sys.executable, "-m", "kore.cli", "--help"])


@pytest.mark.shell
@pytest.mark.parametrize("subcommand", CLI_SUBCOMMANDS)
def test_package_subcommand_help(subcommand: str) -> None:
    _help([sys.executable, "-m", "kore.cli", subcommand, "--help"])


@pytest.mark.shell
@pytest.mark.parametrize(
    "module",
    (
        "kore.tasks.generate_ops",
        "kore.tasks.generate_vendor_ops",
        "kore.tasks.generate_breadth",
    ),
)
def test_task_generator_help(module: str) -> None:
    _help([sys.executable, "-m", module, "--help"])
