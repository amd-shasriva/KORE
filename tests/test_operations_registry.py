from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO / "scripts" / "operations_registry.json"


def _registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def _script_inventory() -> set[str]:
    files = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / "scripts").rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh", ".sbatch"}
    }
    files.add("logs/reli.sh")
    return files


def test_registry_classifies_every_operational_script_once():
    registry = _registry()
    records = registry["scripts"]
    paths = [record["path"] for record in records]

    assert len(paths) == len(set(paths))
    assert set(paths) == _script_inventory()
    assert {record["classification"] for record in records} <= {
        "active",
        "diagnostic",
        "deprecated",
        "destructive",
    }
    assert {"active", "diagnostic", "deprecated", "destructive"} <= {
        record["classification"] for record in records
    }


def test_spur_production_entrypoints_remain_active():
    records = {record["path"]: record for record in _registry()["scripts"]}
    for path in (
        "scripts/spur_supervise_datagen.py",
        "scripts/spur_submit_datagen.sh",
        "scripts/spur_datagen_array.sbatch",
    ):
        assert records[path]["classification"] == "active"
        assert records[path]["production"] is True


def test_retired_b05_ssh_and_14b_entrypoints_are_quarantined():
    records = {record["path"]: record for record in _registry()["scripts"]}
    retired = {
        "scripts/datagen_half.sh",
        "scripts/factory_supervise.sh",
        "scripts/_kf_split.py",
        "scripts/_kf_worker.sh",
        "scripts/kore_pause_after_datagen.py",
        "scripts/kore_resume_supervise.py",
        "scripts/kore_supervise.py",
        "scripts/run_conductor_14b.sh",
        "scripts/run_e2e_14b.sh",
        "scripts/run_full_14b.sh",
        "scripts/run_grpo_resilient.sh",
        "scripts/run_sft_gate.py",
        "scripts/run_sft_gate_dynamic.sh",
        "scripts/run_v2_reuse_14b.sh",
        "scripts/sft_finish_dynamic.sh",
        "scripts/tmux_campaign.sh",
        "scripts/two_node_maximize.sh",
        "scripts/wins_deepen_supervise.sh",
        "logs/reli.sh",
    }
    for path in retired:
        assert records[path]["lifecycle"] == "deprecated"
        source = (REPO / path).read_text()
        assert (
            "kore_deprecated_guard" in source
            or "deprecated_entrypoint" in source
        )


def test_no_pattern_based_process_kills_or_stale_tmux_replacement():
    sources = "\n".join(
        (REPO / path).read_text()
        for path in _script_inventory()
        if (REPO / path).suffix in {".py", ".sh", ".sbatch"}
    )

    assert not re.search(r"\bpkill\b|\bkillall\b", sources)
    assert not re.search(r"\bpgrep\b[^\n]*(?:-f|--full)", sources)
    assert not re.search(r"\bps\s+-eo\s+cmd\b", sources)
    assert not re.search(r"\btmux\s+kill-session\b", sources)


def test_all_deprecated_entrypoints_have_side_effect_free_help_and_dry_run(
    tmp_path: Path,
):
    records = [
        record
        for record in _registry()["scripts"]
        if record.get("lifecycle") == "deprecated"
    ]
    runtime = tmp_path / "runtime-must-not-exist"
    environment = os.environ.copy()
    environment.pop("KORE_ALLOW_DEPRECATED_DEV", None)
    environment.pop("KORE_ALLOW_DESTRUCTIVE_DEV", None)
    environment["KORE_RUNTIME_DIR"] = str(runtime)
    environment["PYTHONPATH"] = str(REPO)

    for record in records:
        path = REPO / record["path"]
        launcher = [sys.executable, str(path)] if path.suffix == ".py" else ["bash", str(path)]
        for flag in ("--help", "--dry-run"):
            result = subprocess.run(
                [*launcher, flag],
                cwd=REPO,
                env=environment,
                text=True,
                capture_output=True,
                timeout=10,
            )
            assert result.returncode == 0, (
                record["path"],
                flag,
                result.stdout,
                result.stderr,
            )
            assert not runtime.exists(), (record["path"], flag)


def test_deprecated_entrypoints_refuse_real_execution_without_override():
    environment = os.environ.copy()
    environment.pop("KORE_ALLOW_DEPRECATED_DEV", None)
    environment["PYTHONPATH"] = str(REPO)
    for record in _registry()["scripts"]:
        if record.get("lifecycle") != "deprecated":
            continue
        path = REPO / record["path"]
        launcher = [sys.executable, str(path)] if path.suffix == ".py" else ["bash", str(path)]
        result = subprocess.run(
            launcher,
            cwd=REPO,
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 64, (
            record["path"],
            result.stdout,
            result.stderr,
        )


def test_dynamic_wrappers_encode_nonzero_giveup():
    for relative in (
        "scripts/datagen_half.sh",
        "scripts/factory_supervise.sh",
        "scripts/run_grpo_resilient.sh",
        "scripts/run_sft_gate_dynamic.sh",
        "scripts/sft_finish_dynamic.sh",
        "scripts/two_node_maximize.sh",
        "scripts/wins_deepen_supervise.sh",
    ):
        source = (REPO / relative).read_text()
        assert re.search(r"\bexit\s+6\b", source), relative
