from __future__ import annotations

import json
from pathlib import Path

from kore.ops.verify import (
    verify_campaign,
    verify_grpo_config,
    verify_model_artifact,
    verify_sft_gate,
    verify_task_shards,
)


def _model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"model_type": "fake"}\n')
    (path / "model.safetensors").write_bytes(b"fake-weights")


def _jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def test_campaign_completion_requires_manifest_and_final_artifacts(tmp_path: Path):
    data = tmp_path / "data"
    model = tmp_path / "runs" / "sft"
    _model(model)
    _jsonl(data / "sft" / "multicap.jsonl", {"messages": []})
    _jsonl(data / "dpo" / "pairs.jsonl", {"prompt": "x"})
    (data / "eval").mkdir(parents=True)
    (data / "eval" / "bakeoff.json").write_text('{"ok": true}\n')
    (data / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "done_stages": ["build", "sft", "eval"],
                "sft_ckpt": str(model),
            }
        )
    )

    status = verify_campaign(tmp_path, data, ["build", "sft", "eval"])

    assert status.ok, status.errors


def test_manifest_done_flag_is_not_enough_without_model_weights(tmp_path: Path):
    data = tmp_path / "data"
    model = tmp_path / "runs" / "sft"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    data.mkdir()
    (data / "campaign_manifest.json").write_text(
        json.dumps({"done_stages": ["sft"], "sft_ckpt": str(model)})
    )

    status = verify_campaign(tmp_path, data, ["sft"])

    assert not status.ok
    assert any("final model weights" in error for error in status.errors)


def test_sft_gate_checks_recorded_candidate_and_artifact(tmp_path: Path):
    candidate = tmp_path / "runs" / "sft"
    _model(candidate)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"done_stages": ["sft"], "sft_ckpt": str(candidate)})
    )

    assert verify_sft_gate(manifest, candidate, repo=tmp_path).ok
    other = tmp_path / "runs" / "other"
    _model(other)
    mismatch = verify_sft_gate(manifest, other, repo=tmp_path)
    assert not mismatch.ok
    assert any("mismatch" in error for error in mismatch.errors)


def test_grpo_config_requires_final_model_artifact(tmp_path: Path):
    output = tmp_path / "runs" / "grpo"
    config = tmp_path / "grpo.json"
    config.write_text(json.dumps({"output_dir": str(output)}))

    assert not verify_grpo_config(config, repo=tmp_path).ok
    _model(output)
    assert verify_grpo_config(config, repo=tmp_path).ok


def test_task_shard_verifier_uses_exact_task_set_and_distinct_wins(tmp_path: Path):
    data = tmp_path / "data"
    for task_id in ("task-a", "task-b"):
        _jsonl(data / "repair" / f"{task_id}.jsonl", {"task_id": task_id})
        _jsonl(data / "groups" / f"{task_id}.jsonl", {"task_id": task_id})
        _jsonl(
            data / "wins" / f"{task_id}.jsonl",
            {"task_id": task_id, "final_source": f"source-{task_id}"},
        )

    status = verify_task_shards(data, ["task-b", "task-a"], target_wins=1)
    assert status.ok
    assert status.details["task_count"] == 2
    assert len(str(status.details["task_sha256"])) == 64

    (data / "wins" / "task-b.jsonl.inprogress").touch()
    status = verify_task_shards(data, ["task-a", "task-b"], target_wins=1)
    assert not status.ok
    assert any("incomplete shard marker" in error for error in status.errors)


def test_model_verifier_rejects_symlink_directory(tmp_path: Path):
    real = tmp_path / "real"
    _model(real)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    status = verify_model_artifact(linked, repo=tmp_path)

    assert not status.ok
    assert any("not a real directory" in error for error in status.errors)
