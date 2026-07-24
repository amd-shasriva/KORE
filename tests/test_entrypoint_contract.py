"""Syntax and side-effect-free ``--help`` smoke tests for entrypoints."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

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
