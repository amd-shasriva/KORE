#!/usr/bin/env python3
"""Validate built artifacts and a clean wheel installation.

The source tree is the manifest of record: every task contract and every
non-Python package-data file must survive both the wheel and sdist unchanged.
The wheel is then installed into a fresh virtual environment and exercised
from outside the checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import zipfile


TASK_FILES = ("task.yaml", "reference.py", "driver.py", "seed_triton.py")
PACKAGE_DATA_SUFFIXES = {".json", ".jsonl", ".md", ".yaml", ".yml"}


def _source_manifest(repo: Path) -> tuple[dict[str, str], list[str]]:
    package = repo / "kore"
    tasks = package / "tasks"
    errors: list[str] = []
    task_ids: list[str] = []
    paths: set[Path] = set()

    for task_yaml in sorted(tasks.glob("*/task.yaml")):
        task_dir = task_yaml.parent
        task_ids.append(task_dir.name)
        for name in TASK_FILES:
            path = task_dir / name
            if path.is_file():
                paths.add(path)
            else:
                errors.append(f"{task_dir.name}: missing {name}")

    if not task_ids:
        errors.append("no task directories found")

    paths.update(
        path
        for path in package.rglob("*")
        if path.is_file() and path.suffix.lower() in PACKAGE_DATA_SUFFIXES
    )
    if errors:
        raise AssertionError("\n".join(errors))

    hashes = {
        path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    return hashes, task_ids


def _wheel_files(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }


def _sdist_files(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2:
                continue
            stream = archive.extractfile(member)
            if stream is not None:
                files[PurePosixPath(*parts[1:]).as_posix()] = stream.read()
    return files


def _check_artifact(
    label: str,
    files: dict[str, bytes],
    expected_hashes: dict[str, str],
) -> None:
    errors: list[str] = []
    for name, expected_hash in expected_hashes.items():
        payload = files.get(name)
        if payload is None:
            errors.append(f"{label}: missing {name}")
            continue
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != expected_hash:
            errors.append(f"{label}: content differs for {name}")

    expected_task_assets = {
        name
        for name in expected_hashes
        if name.startswith("kore/tasks/")
        and PurePosixPath(name).name in TASK_FILES
    }
    artifact_task_assets = {
        name
        for name in files
        if name.startswith("kore/tasks/")
        and PurePosixPath(name).name in TASK_FILES
    }
    for extra in sorted(artifact_task_assets - expected_task_assets):
        errors.append(f"{label}: unexpected task asset {extra}")

    if errors:
        raise AssertionError("\n".join(errors))


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode:
        rendered = " ".join(command)
        raise AssertionError(
            f"command failed ({result.returncode}): {rendered}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _installed_wheel_smoke(
    wheel: Path,
    expected_hashes: dict[str, str],
    expected_task_ids: list[str],
) -> None:
    with tempfile.TemporaryDirectory(prefix="kore-wheel-smoke-") as raw_tmp:
        tmp = Path(raw_tmp)
        venv_dir = tmp / "venv"
        clean_env = dict(os.environ)
        clean_env.pop("PYTHONPATH", None)
        clean_env["PYTHONNOUSERSITE"] = "1"
        clean_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

        _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=tmp, env=clean_env)
        bin_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
        python = bin_dir / ("python.exe" if os.name == "nt" else "python")
        kore = bin_dir / ("kore.exe" if os.name == "nt" else "kore")
        install_command = [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-input",
            "--no-compile",
        ]
        if constraint := os.environ.get("KORE_PIP_CONSTRAINT"):
            install_command.extend(["--constraint", str(Path(constraint).resolve())])
        install_command.append(str(wheel))
        _run(
            install_command,
            cwd=tmp,
            env=clean_env,
        )

        listed = _run([str(kore), "tasks"], cwd=tmp, env=clean_env, timeout=180)
        listed_ids = [line.split()[0] for line in listed.stdout.splitlines() if line.strip()]
        if listed_ids != expected_task_ids:
            missing = sorted(set(expected_task_ids) - set(listed_ids))
            extra = sorted(set(listed_ids) - set(expected_task_ids))
            raise AssertionError(
                "installed `kore tasks` registry differs from source: "
                f"expected={len(expected_task_ids)} actual={len(listed_ids)} "
                f"missing={missing} extra={extra}"
            )

        manifest = tmp / "expected-assets.json"
        manifest.write_text(json.dumps(expected_hashes, sort_keys=True), encoding="utf-8")
        validator = r"""
import hashlib
import json
from pathlib import Path

import kore
from kore.tasks.registry import all_tasks

expected = json.loads(Path(__import__("sys").argv[1]).read_text(encoding="utf-8"))
package_root = Path(kore.__file__).resolve().parent
errors = []
for relative, wanted in sorted(expected.items()):
    package_relative = Path(relative).relative_to("kore")
    installed = package_root / package_relative
    if not installed.is_file():
        errors.append(f"missing {relative}")
        continue
    actual = hashlib.sha256(installed.read_bytes()).hexdigest()
    if actual != wanted:
        errors.append(f"content differs for {relative}")
registered = {task.task_id for task in all_tasks()}
expected_ids = {
    Path(relative).parts[2]
    for relative in expected
    if relative.startswith("kore/tasks/") and relative.endswith("/task.yaml")
}
if registered != expected_ids:
    errors.append(
        f"registry mismatch expected={len(expected_ids)} actual={len(registered)}"
    )
if errors:
    raise SystemExit("\n".join(errors))
print(f"validated {len(registered)} tasks and {len(expected)} packaged assets")
"""
        _run(
            [str(python), "-c", validator, str(manifest)],
            cwd=tmp,
            env=clean_env,
            timeout=180,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--sdist", required=True, type=Path)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="source checkout used as the artifact manifest",
    )
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    wheel = args.wheel.resolve()
    sdist = args.sdist.resolve()
    expected_hashes, task_ids = _source_manifest(repo)
    _check_artifact("wheel", _wheel_files(wheel), expected_hashes)
    _check_artifact("sdist", _sdist_files(sdist), expected_hashes)
    _installed_wheel_smoke(wheel, expected_hashes, task_ids)
    print(
        f"validated wheel + sdist: {len(task_ids)} tasks, "
        f"{len(expected_hashes)} required assets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
