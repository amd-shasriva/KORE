"""Focused CPU tests for campaign lineage and fail-closed promotion paths."""

from __future__ import annotations

import copy
import json

import pytest

import scripts.run_campaign as rc
from kore.campaign_lineage import git_source_identity, resolve_model_snapshot


def _args(*extra):
    return rc.build_parser().parse_args(list(extra))


def _lineage(model: str = "Qwen/Qwen3-14B", suffix: str = "a") -> dict:
    return {
        "compatibility_digest": f"sha256:compat-{suffix}",
        "model": {
            "requested_id": model,
            "resolved_revision": f"rev-{suffix}",
            "content_digest": f"sha256:model-{suffix}",
            "architecture": {
                "model_type": "qwen3",
                "hidden_size": 5120 if "14B" in model else 5120,
                "num_hidden_layers": 40 if "14B" in model else 64,
            },
        },
        "tokenizer": {"content_digest": f"sha256:tokenizer-{suffix}"},
        "source": {
            "commit": "a" * 40,
            "dirty": False,
            "content_digest": f"sha256:source-{suffix}",
        },
        "stage_config": {"digest": f"sha256:config-{suffix}"},
        "tasks": {
            "registry_digest": f"sha256:registry-{suffix}",
            "split_digest": f"sha256:split-{suffix}",
            "train": ["rmsnorm_aiter"],
            "eval": ["mla_decode_bf16"],
        },
        "verifier_gate_contract": {"digest": f"sha256:gate-{suffix}"},
        "hardware_runtime": {"compatibility_digest": f"sha256:runtime-{suffix}"},
    }


def _ctx(tmp_path, *, mode: str = "production", lineage=None) -> dict:
    args = _args("--campaign-mode", mode, "--use-hf")
    return {
        "data_root": tmp_path,
        "dry": False,
        "args": args,
        "base": "base",
        "base_ref": "Qwen/Qwen3-14B",
        "midtrain_ckpt": None,
        "sft_ckpt": None,
        "dpo_ckpt": None,
        "grpo_ckpt": None,
        "final": None,
        "done_stages": set(),
        "artifacts": {},
        "lineage": lineage or _lineage(),
        "train_task_ids": ["rmsnorm_aiter"],
        "eval_task_ids": ["mla_decode_bf16"],
    }


def _write_manifest(ctx) -> None:
    rc._save_manifest(ctx)


def _suite(source: str = "full-hf", value: float = 0.7) -> dict:
    return {
        "scores": {key: value for key in rc._GENERAL_GATE_KEYS},
        "sources": {key: source for key in rc._GENERAL_GATE_KEYS},
        "full": True,
    }


def test_unreadable_and_legacy_manifests_fail_closed(tmp_path):
    ctx = _ctx(tmp_path)
    (tmp_path / "campaign_manifest.json").write_text("{broken")
    with pytest.raises(SystemExit, match="unreadable"):
        rc._read_manifest_strict(ctx)

    (tmp_path / "campaign_manifest.json").write_text(json.dumps({"model": "legacy"}))
    with pytest.raises(SystemExit, match="schema"):
        rc._read_manifest_strict(ctx)


def test_resume_rejects_14b_32b_model_lineage_mismatch(tmp_path):
    original = _ctx(tmp_path, lineage=_lineage("Qwen/Qwen3-14B", "14b"))
    _write_manifest(original)

    resumed = _ctx(tmp_path, lineage=_lineage("Qwen/Qwen3-32B", "32b"))
    with pytest.raises(SystemExit, match="model"):
        rc._load_manifest_into_ctx(resumed)


def test_build_lineage_rejects_14b_32b_before_download(monkeypatch, tmp_path):
    import kore.campaign_lineage as lineage_module

    monkeypatch.setattr(
        lineage_module,
        "resolve_model_snapshot",
        lambda *args, **kwargs: pytest.fail("incompatible model must not download"),
    )
    args = _args(
        "--campaign-mode", "development",
        "--model", "Qwen/Qwen3-32B",
    )
    prior = {"lineage": {"model": {"requested_id": "Qwen/Qwen3-14B"}}}
    with pytest.raises(SystemExit, match="before model download"):
        rc._build_lineage({
            "args": args,
            "data_root": tmp_path,
            "tasks": [],
            "train_task_ids": [],
            "eval_task_ids": [],
        }, prior=prior)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("stage_config", "digest"),
        ("tasks", "split_digest"),
        ("tasks", "registry_digest"),
        ("verifier_gate_contract", "digest"),
        ("hardware_runtime", "compatibility_digest"),
    ],
)
def test_resume_rejects_config_split_registry_gate_runtime_mismatch(
    tmp_path, section, key,
):
    stored = _lineage()
    _write_manifest(_ctx(tmp_path, lineage=stored))
    current = copy.deepcopy(stored)
    current[section][key] += "-changed"
    current["compatibility_digest"] += "-changed"
    with pytest.raises(SystemExit, match="incompatible"):
        rc._load_manifest_into_ctx(_ctx(tmp_path, lineage=current))


def test_local_model_and_tokenizer_lineage_is_content_bound(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4,
        "num_hidden_layers": 2,
    }))
    (model / "tokenizer_config.json").write_text('{"model_max_length": 128}')
    (model / "model.safetensors").write_bytes(b"weights-v1")

    first_model, first_tokenizer, load_path = resolve_model_snapshot(str(model))
    assert load_path == str(model.resolve())
    assert first_model["resolved_revision"].startswith("local-")
    assert first_tokenizer["resolved_revision"] == first_model["resolved_revision"]

    (model / "model.safetensors").write_bytes(b"weights-v2")
    second_model, _, _ = resolve_model_snapshot(str(model))
    assert second_model["content_digest"] != first_model["content_digest"]


def test_built_lineage_contains_every_phase0_contract(monkeypatch, tmp_path):
    import kore.campaign_lineage as lineage_module
    from kore.tasks.registry import get_task

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4,
        "num_hidden_layers": 2,
    }))
    (model / "tokenizer_config.json").write_text('{"model_max_length": 128}')
    (model / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(lineage_module, "git_source_identity", lambda root: {
        "commit": "a" * 40,
        "dirty": True,
        "dirty_status_digest": "sha256:dirty",
        "content_digest": "sha256:source",
        "scope": ["scripts", "kore", "configs", "pyproject.toml"],
    })
    monkeypatch.setattr(lineage_module, "runtime_identity", lambda: {
        "python": "3.test",
        "gpu_arches": ["gfx950"],
        "compatibility_digest": "sha256:runtime",
    })
    args = _args(
        "--campaign-mode", "development",
        "--model", str(model),
        "--tasks", "rmsnorm_aiter",
    )
    ctx = {
        "args": args,
        "data_root": tmp_path / "run",
        "tasks": [get_task("rmsnorm_aiter")],
        "base": str(model),
    }
    rc._apply_split(ctx)
    built = rc._build_lineage(ctx)

    assert built["model"]["resolved_revision"].startswith("local-")
    assert built["tokenizer"]["content_digest"].startswith("sha256:")
    assert built["source"]["dirty"] is True
    assert built["stage_config"]["digest"].startswith("sha256:")
    assert built["tasks"]["registry_digest"].startswith("sha256:")
    assert built["tasks"]["split_digest"].startswith("sha256:")
    assert built["verifier_gate_contract"]["digest"].startswith("sha256:")
    assert built["hardware_runtime"]["gpu_arches"] == ["gfx950"]
    assert built["compatibility_digest"].startswith("sha256:")
    ctx.update({
        "dry": False,
        "lineage": built,
        "done_stages": set(),
        "artifacts": {},
        "midtrain_ckpt": None,
        "sft_ckpt": None,
        "dpo_ckpt": None,
        "grpo_ckpt": None,
        "final": None,
    })
    rc._save_manifest(ctx)
    persisted = json.loads((ctx["data_root"] / "campaign_manifest.json").read_text())
    assert persisted["lineage"]["stage_config"]["resolved_stage_configs"]["grpo"]


def test_git_source_identity_records_commit_dirty_state_and_content():
    identity = git_source_identity(rc._repo_root())
    assert len(identity["commit"]) == 40
    assert isinstance(identity["dirty"], bool)
    assert identity["content_digest"].startswith("sha256:")
    assert identity["dirty_status_digest"].startswith("sha256:")


def test_production_rejects_smoke_and_source_mismatch(tmp_path):
    production = _ctx(tmp_path, mode="production")
    with pytest.raises(SystemExit, match="smoke/fallback"):
        rc._validate_retention_suite(
            production, _suite("smoke"), stage="eval", role="candidate",
        )

    development = _ctx(tmp_path, mode="development")
    expected = {key: "full-hf" for key in rc._GENERAL_GATE_KEYS}
    mismatched = _suite("full-hf")
    mismatched["sources"]["mmlu"] = "smoke"
    with pytest.raises(SystemExit, match="source mismatch"):
        rc._validate_retention_suite(
            development,
            mismatched,
            stage="eval",
            role="candidate",
            expected_sources=expected,
        )


def test_serving_unavailable_fails_gate(monkeypatch, tmp_path):
    import kore.policy.serve as serve

    def unavailable(*args, **kwargs):
        raise ImportError("torch backend absent")

    monkeypatch.setattr(serve, "load_generate", unavailable)
    monkeypatch.setattr(rc, "_gpu_ids", lambda ctx: [])
    with pytest.raises(SystemExit, match="serving is unavailable"):
        rc._load_generate_or_fail(_ctx(tmp_path), "candidate", stage="eval")


def test_production_mode_rejects_weakened_contract():
    with pytest.raises(SystemExit, match="missing --use-hf"):
        rc._validate_campaign_contract(_args())
    with pytest.raises(SystemExit, match="--no-retention-gate"):
        rc._validate_campaign_contract(_args("--use-hf", "--no-retention-gate"))
    with pytest.raises(SystemExit, match="--no-rigorous-verify"):
        rc._validate_campaign_contract(_args("--use-hf", "--no-rigorous-verify"))

    # Weaker behavior is available only under an explicit non-production mode.
    rc._validate_campaign_contract(
        _args("--campaign-mode", "smoke", "--no-retention-gate", "--no-rigorous-verify")
    )


def test_existence_only_artifact_is_rejected_and_digest_mutation_detected(tmp_path):
    ctx = _ctx(tmp_path, mode="development")
    ctx["artifacts"]["build"] = {"digest": "sha256:not-real"}
    (tmp_path / "sft").mkdir()
    (tmp_path / "dpo").mkdir()
    assert rc._artifact_ok(ctx, "build") is False

    sft = tmp_path / "sft" / "multicap.jsonl"
    dpo = tmp_path / "dpo" / "pairs.jsonl"
    sft.write_text('{"messages": [{"role": "user", "content": "x"}]}\n')
    dpo.write_text('{"prompt": "p", "chosen": "c", "rejected": "r"}\n')
    captured = rc._capture_stage_artifact(ctx, "build")
    ctx["artifacts"]["build"] = captured
    assert rc._artifact_ok(ctx, "build") is True

    sft.write_text('{"messages": [{"role": "user", "content": "changed"}]}\n')
    assert rc._artifact_ok(ctx, "build") is False


def test_failed_claim_profile_cannot_form_eval_artifact(tmp_path):
    ctx = _ctx(tmp_path, mode="development")
    ctx["args"].claim_profile = "kernel-frontier"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "bakeoff.json").write_text(json.dumps({
        "policies": {"seed": {"fast_p": {"1.0": 0.1}}, "kore": {"fast_p": {"1.0": 0.2}}},
    }))
    (eval_dir / "promotion_gate.json").write_text('{"passed": true}')
    (eval_dir / "claim_status.json").write_text(json.dumps({
        "profile": "kernel-frontier",
        "passed": False,
        "failed_required_tracks": ["kernelbench_amd"],
    }))
    with pytest.raises(RuntimeError, match="frontier"):
        rc._capture_stage_artifact(ctx, "eval")


def test_final_stagegate_requires_kernel_improvement_and_all_general_metrics():
    bakeoff = {
        "policies": {
            "seed": {"fast_p": {1.0: 0.2}},
            "kore": {"fast_p": {1.0: 0.3}},
        }
    }
    base = {key: 0.7 for key in rc._GENERAL_GATE_KEYS}
    candidate = dict(base)
    assert rc._evaluate_final_stage_gate(
        bakeoff, base, candidate, epsilon=0.02,
    ).passed

    flat = copy.deepcopy(bakeoff)
    flat["policies"]["kore"]["fast_p"][1.0] = 0.2
    assert not rc._evaluate_final_stage_gate(flat, base, candidate, epsilon=0.02).passed

    candidate.pop("mmlu")
    assert not rc._evaluate_final_stage_gate(
        bakeoff, base, candidate, epsilon=0.02,
    ).passed


def test_required_claim_track_failure_is_explicit(tmp_path):
    result = rc._run_claim_track(
        _ctx(tmp_path), "kernelbench_amd", {"kernelbench_amd"},
        lambda: {"passed": False, "reason": "no report"},
    )
    assert result["required"] is True
    assert result["passed"] is False
    assert result["status"] == "failed"


def test_production_refuses_orphan_artifacts(tmp_path):
    (tmp_path / "sft").mkdir()
    (tmp_path / "sft" / "multicap.jsonl").write_text('{"messages": []}\n')
    with pytest.raises(SystemExit, match="unbound adoption"):
        rc._reject_orphan_artifacts(_ctx(tmp_path, mode="production"))
    rc._reject_orphan_artifacts(_ctx(tmp_path, mode="development"))


def test_gate_receipt_skip_is_nonproduction_only(tmp_path):
    path = tmp_path / "gates" / "sft.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "status": "skipped",
        "result": {"passed": False},
    }))
    with pytest.raises(RuntimeError, match="no passing"):
        rc._gate_receipt_artifact(_ctx(tmp_path, mode="production"), "sft")
    assert rc._gate_receipt_artifact(_ctx(tmp_path, mode="development"), "sft")["kind"] == "json"
