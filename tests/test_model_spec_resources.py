"""CPU/offline tests for immutable model identity and resource preflight."""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from kore.policy.model_spec import (
    UNRESOLVED,
    ArchitectureMismatchError,
    ArchitectureSpec,
    FloatingRevisionError,
    ModelProfile,
    ModelSpec,
    ModelSpecError,
    QWEN3_32B_PROFILE,
    canonical_profile_hash,
    validate_pinned_revision,
)
from kore.policy.resources import (
    ABSENT,
    MEASURE,
    NOT_APPLICABLE,
    FilesystemCapacity,
    GPUDevice,
    InsufficientResourcesError,
    MeasurementProvenance,
    MeasuredPeakProfile,
    PhaseEvidence,
    RankPeakReport,
    ResourcePreflightError,
    ResourceSnapshot,
    UnresolvedProductionFieldError,
    WorkloadSpec,
    atomic_write_json,
    collect_amd_gpu_devices,
    compute_analytical_lower_bounds,
    required_measurement_phases,
    run_resource_preflight,
)


REVISION = "a" * 40
TINY_ARCH = ArchitectureSpec(
    model_type="qwen3",
    architecture="Qwen3ForCausalLM",
    decoder_class="Qwen3DecoderLayer",
    hidden_size=4,
    intermediate_size=8,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=2,
    vocab_size=16,
    max_position_embeddings=128,
)


def _tiny_tensor_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        "model.embed_tokens.weight": (16, 4),
        "model.norm.weight": (4,),
        "lm_head.weight": (16, 4),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.self_attn.q_proj.weight": (4, 4),
                f"{prefix}.self_attn.k_proj.weight": (2, 4),
                f"{prefix}.self_attn.v_proj.weight": (2, 4),
                f"{prefix}.self_attn.o_proj.weight": (4, 4),
                f"{prefix}.self_attn.q_norm.weight": (2,),
                f"{prefix}.self_attn.k_norm.weight": (2,),
                f"{prefix}.mlp.gate_proj.weight": (8, 4),
                f"{prefix}.mlp.up_proj.weight": (8, 4),
                f"{prefix}.mlp.down_proj.weight": (4, 8),
                f"{prefix}.input_layernorm.weight": (4,),
                f"{prefix}.post_attention_layernorm.weight": (4,),
            }
        )
    return shapes


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[int, ...]]
) -> int:
    header = {}
    offset = 0
    for name, shape in sorted(tensors.items()):
        size = 2
        for dimension in shape:
            size *= dimension
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw_header = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    raw_header += b" " * ((-len(raw_header)) % 8)
    path.write_bytes(
        len(raw_header).to_bytes(8, "little") + raw_header + bytes(offset)
    )
    return offset


def _make_checkpoint(root: Path) -> tuple[ModelProfile, int]:
    root.mkdir()
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 16,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "rope_theta": 1_000_000,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 128}), encoding="utf-8"
    )
    (root / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "model": {}}), encoding="utf-8"
    )
    (root / "generation_config.json").write_text(
        json.dumps({"do_sample": False}), encoding="utf-8"
    )

    shapes = _tiny_tensor_shapes()
    names = sorted(shapes)
    split = len(names) // 2
    shards = {
        "model-00001-of-00002.safetensors": {
            name: shapes[name] for name in names[:split]
        },
        "model-00002-of-00002.safetensors": {
            name: shapes[name] for name in names[split:]
        },
    }
    total_size = 0
    weight_map = {}
    for shard_name, shard_shapes in shards.items():
        total_size += _write_safetensors(root / shard_name, shard_shapes)
        weight_map.update({name: shard_name for name in shard_shapes})
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    parameter_count = sum(
        __import__("math").prod(shape) for shape in shapes.values()
    )
    profile = ModelProfile(
        name="tiny-qwen3",
        model_id="fixture/tiny-qwen3",
        revision=UNRESOLVED,
        architecture=TINY_ARCH,
        expected_parameter_count=parameter_count,
    )
    return profile, parameter_count


@pytest.fixture
def tiny_checkpoint(tmp_path):
    root = tmp_path / "tiny"
    profile, parameter_count = _make_checkpoint(root)
    return root, profile, parameter_count


def test_floating_and_unresolved_revisions_fail_closed(tiny_checkpoint):
    root, profile, _ = tiny_checkpoint
    for revision in (None, "", "main", "v1.0", "a" * 39):
        with pytest.raises(FloatingRevisionError):
            validate_pinned_revision(revision)
    with pytest.raises(FloatingRevisionError):
        ModelSpec.from_local_checkpoint(root, expected=profile)
    assert QWEN3_32B_PROFILE.revision == UNRESOLVED


def test_14b_32b_architecture_mismatch_rejected(tiny_checkpoint):
    root, profile, _ = tiny_checkpoint
    wrong_32b = replace(
        profile,
        name="declared-32b",
        architecture=replace(profile.architecture, num_hidden_layers=3),
    )
    with pytest.raises(ArchitectureMismatchError, match="num_hidden_layers"):
        ModelSpec.from_local_checkpoint(
            root, revision=REVISION, expected=wrong_32b
        )


def test_exact_parameter_count_and_deterministic_profile_hash(tiny_checkpoint):
    root, profile, expected_count = tiny_checkpoint
    first = ModelSpec.from_local_checkpoint(
        root, revision=REVISION, expected=profile
    )
    second = ModelSpec.from_local_checkpoint(
        root, revision=REVISION, expected=profile
    )
    assert first.parameter_count == expected_count
    assert first.profile_hash == second.profile_hash
    assert first.files.manifest_hash == second.files.manifest_hash
    assert canonical_profile_hash({"b": 2, "a": 1}) == canonical_profile_hash(
        {"a": 1, "b": 2}
    )
    with pytest.raises(ModelSpecError, match="cannot authorize a remote"):
        first.validate_for_load(first.model_id)


def test_config_tokenizer_generation_and_shard_changes_fingerprint(
    tiny_checkpoint,
):
    root, profile, _ = tiny_checkpoint

    def fingerprint():
        return ModelSpec.from_local_checkpoint(
            root, revision=REVISION, expected=profile
        ).profile_hash

    initial_spec = ModelSpec.from_local_checkpoint(
        root, revision=REVISION, expected=profile
    )
    original = initial_spec.profile_hash
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["rope_theta"] = 2_000_000
    config_path.write_text(json.dumps(config), encoding="utf-8")
    changed_config = fingerprint()
    assert changed_config != original

    tokenizer_path = root / "tokenizer_config.json"
    tokenizer_path.write_text(
        json.dumps({"model_max_length": 128, "padding_side": "left"}),
        encoding="utf-8",
    )
    changed_tokenizer = fingerprint()
    assert changed_tokenizer != changed_config

    generation_path = root / "generation_config.json"
    generation_path.write_text(
        json.dumps({"do_sample": False, "temperature": 0.0}),
        encoding="utf-8",
    )
    changed_generation = fingerprint()
    assert changed_generation != changed_tokenizer

    shard_path = root / "model-00001-of-00002.safetensors"
    shard = bytearray(shard_path.read_bytes())
    shard[-1] ^= 1
    shard_path.write_bytes(shard)
    assert fingerprint() != changed_generation
    with pytest.raises(ModelSpecError, match="changed"):
        initial_spec.validate_for_load(root)


def _resources(
    *,
    free_hbm: int | str = 10_000,
    topology=None,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        gpus=(
            GPUDevice(
                drm_card="card3",
                render_node="renderD129",
                pci_bdf="0000:01:00.0",
                hip_reported_pci_bdf="0000:01:00.0",
                uuid="gpu-0",
                hip_reported_uuid="gpu-0",
                physical_card=0,
                hip_ordinal=0,
                slurm_gres_id=NOT_APPLICABLE,
                slurm_allocated=NOT_APPLICABLE,
                name="fixture-gpu",
                numa_node=0,
                total_hbm_bytes=20_000,
                free_hbm_bytes=free_hbm,
                visible=True,
            ),
        ),
        gpu_topology=topology or {"gpu-0": {"gpu-0": "self"}},
        visible_device_policy={
            "authority": "unmasked",
            "raw": NOT_APPLICABLE,
            "hip_ordinals": [0],
        },
        slurm_allocation={
            "mode": "none",
            "job_id": NOT_APPLICABLE,
            "step_id": NOT_APPLICABLE,
            "gres": NOT_APPLICABLE,
            "physical_cards": [],
            "hip_ordinals": [],
        },
        host_ram_total_bytes=100_000,
        host_ram_available_bytes=80_000,
        filesystems=(
            FilesystemCapacity("model", "/models", 1, 1_000_000, 900_000),
            FilesystemCapacity("scratch", "/scratch", 2, 2_000_000, 1_800_000),
        ),
        software_versions={
            "python": "3.11",
            "kernel": "fixture",
            "rocm": "7.0",
            "amdgpu": "fixture",
            "torch": "2.8",
            "transformers": "4.53",
            "safetensors": "0.5",
            "accelerate": "1.8",
            "trl": "0.19",
            "peft": "0.15",
            "datasets": ABSENT,
        },
        code_fingerprint="c" * 40,
        source="fixture",
    )


def _workload(spec, resources, **overrides) -> WorkloadSpec:
    values = dict(
        stage="dpo",
        global_batch_size=8,
        microbatch_size=1,
        gradient_accumulation_steps=8,
        sequence_lengths={"max_length": 128, "prompt": 64, "completion": 64},
        precision="bf16",
        sharding="fsdp-full-shard",
        offload="none",
        backend="transformers",
        world_size=1,
        topology_hash=resources.topology_hash,
        optimizer="adamw",
        optimizer_initialized=True,
        model_copies=1,
        reference_copies=1,
        rollout_copies=0,
        resolved_config={
            "stage": "dpo",
            "batch": 8,
            "microbatch": 1,
            "max_length": 128,
            "precision": "bf16",
            "sharding": "fsdp-full-shard",
            "offload": "none",
            "backend": "transformers",
        },
        code_fingerprint=resources.code_fingerprint,
        dependency_profile_hash=resources.dependency_profile_hash,
        model_profile_hash=spec.profile_hash,
        required_dependencies=("torch", "transformers", "accelerate", "trl"),
    )
    values.update(overrides)
    return WorkloadSpec(**values)


def _phase(name: str, *, bdf: str = "0000:01:00.0") -> PhaseEvidence:
    artifact = hashlib.sha256(name.encode()).hexdigest()
    return PhaseEvidence(
        phase=name,
        rank_reports=(
            RankPeakReport(
                rank=0,
                hip_ordinal=0,
                pci_bdf=bdf,
                run_peak_hbm_bytes=(9_000, 9_010, 9_005),
            ),
        ),
        host_peak_runs_bytes=(2_000, 2_010, 2_005),
        filesystem_peak_runs_bytes={
            "model": (100, 101, 100),
            "scratch": (200, 201, 200),
        },
        safety_margin_fraction=0.10,
        max_peak_variance_fraction=0.05,
        optimizer_initialized=True,
        provenance=MeasurementProvenance(
            command=("python", "measure.py", "--phase", name),
            tool="rocprofv3",
            tool_version="7.0",
            hostname="fixture-host",
            started_at_utc="2026-07-23T00:00:00Z",
            artifact_sha256=artifact,
            exit_code=0,
        ),
    )


def _measured(
    workload: WorkloadSpec,
    resources: ResourceSnapshot,
    *,
    omit: tuple[str, ...] = (),
) -> MeasuredPeakProfile:
    return MeasuredPeakProfile(
        workload=workload,
        environment_hash=resources.environment_hash,
        phases=tuple(
            _phase(name)
            for name in required_measurement_phases(workload.stage)
            if name not in omit
        ),
    )


def test_analytical_bounds_use_exact_checkpoint_metadata(tiny_checkpoint):
    root, profile, parameter_count = tiny_checkpoint
    spec = ModelSpec.from_local_checkpoint(
        root, revision=REVISION, expected=profile
    )
    bounds = compute_analytical_lower_bounds(spec)
    assert bounds.exact_parameter_count == parameter_count
    assert bounds.bf16_weights_bytes == parameter_count * 2
    assert bounds.full_finetune_persistent_state_bytes == parameter_count * 16
    assert "LOWER BOUNDS" in bounds.label


def test_insufficient_resources_rejected(tiny_checkpoint):
    root, profile, _ = tiny_checkpoint
    spec = ModelSpec.from_local_checkpoint(
        root, revision=REVISION, expected=profile
    )
    with pytest.raises(InsufficientResourcesError, match="weights-only"):
        run_resource_preflight(spec, _resources(free_hbm=1))


def test_unresolved_measure_values_rejected(tiny_checkpoint):
    root, profile, _ = tiny_checkpoint
    spec = ModelSpec.from_local_checkpoint(
        root, revision=REVISION, expected=profile
    )
    with pytest.raises(UnresolvedProductionFieldError, match="MEASURE"):
        run_resource_preflight(spec, _resources(free_hbm=MEASURE))

    resources = _resources()
    workload = _workload(spec, resources)
    measured = _measured(workload, resources)
    broken_phase = replace(
        measured.phases[0],
        provenance=replace(measured.phases[0].provenance, tool=MEASURE),
    )
    measured = replace(
        measured, phases=(broken_phase,) + measured.phases[1:]
    )
    with pytest.raises(UnresolvedProductionFieldError, match="MEASURE"):
        run_resource_preflight(
            spec,
            resources,
            measured,
            expected_workload=workload,
        )


def test_analytical_only_never_asserts_fit(tiny_checkpoint):
    root, profile, _ = tiny_checkpoint
    spec = ModelSpec.from_local_checkpoint(
        root, revision=REVISION, expected=profile
    )
    report = run_resource_preflight(spec, _resources())
    assert report.status == "analytical_only"
    assert report.fit_asserted is False
    with pytest.raises(ResourcePreflightError):
        report.assert_production_ready()


def test_measured_profile_ingestion_and_hashes_are_deterministic(tiny_checkpoint):
    root, profile, _ = tiny_checkpoint
    spec = ModelSpec.from_local_checkpoint(
        root, revision=REVISION, expected=profile
    )
    resources = _resources()
    workload = _workload(spec, resources)
    first = _measured(workload, resources)
    reversed_phase = replace(
        first.phases[0],
        filesystem_peak_runs_bytes={
            "scratch": (200, 201, 200),
            "model": (100, 101, 100),
        },
    )
    second = replace(
        first, phases=(reversed_phase,) + first.phases[1:]
    )
    assert first.profile_hash == second.profile_hash
    assert (
        MeasuredPeakProfile.from_dict(first.to_dict()).profile_hash
        == first.profile_hash
    )
    assert (
        ResourceSnapshot.from_dict(resources.to_dict()).profile_hash
        == resources.profile_hash
    )
    report = run_resource_preflight(
        spec,
        resources,
        first,
        expected_workload=workload,
        require_measured=True,
        headroom_fraction=0.0,
    )
    assert report.production_ready is True
    assert report.status == "measured_pass"


def test_bdf_ordinal_and_slurm_mapping_mismatches_fail(tiny_checkpoint):
    resources = _resources()
    mismatched_gpu = replace(
        resources.gpus[0],
        hip_reported_pci_bdf="0000:02:00.0",
    )
    with pytest.raises(ResourcePreflightError, match="BDF"):
        replace(resources, gpus=(mismatched_gpu,)).validate_resolved()

    slurm_gpu = replace(
        resources.gpus[0],
        slurm_allocated=True,
        slurm_gres_id="gpu:1@physical:1",
    )
    wrong_slurm = replace(
        resources,
        gpus=(slurm_gpu,),
        slurm_allocation={
            "mode": "slurm",
            "job_id": "42",
            "step_id": "0",
            "gres": "gpu:1",
            "physical_cards": [1],
            "hip_ordinals": [0],
        },
    )
    with pytest.raises(ResourcePreflightError, match="physical-card"):
        wrong_slurm.validate_resolved()


def test_inventory_joins_drm_render_bdf_and_hip_without_discovery_index(tmp_path):
    drm = tmp_path / "drm"
    pci = tmp_path / "pci" / "0000:0a:00.0"
    drm.mkdir()
    pci.mkdir(parents=True)
    for name, value in {
        "vendor": "0x1002",
        "mem_info_vram_total": "20000",
        "mem_info_vram_used": "5000",
        "product_name": "fixture-gpu",
        "unique_id": "abc123",
        "numa_node": "1",
    }.items():
        (pci / name).write_text(value)
    (drm / "card7").mkdir()
    (drm / "renderD130").mkdir()
    (drm / "card7" / "device").symlink_to(pci, target_is_directory=True)
    (drm / "renderD130" / "device").symlink_to(pci, target_is_directory=True)
    inventory = (
        {
            "hip_ordinal": 3,
            "pci_bdf": "0000:0a:00.0",
            "uuid": "abc123",
            "physical_card": 11,
        },
    )
    devices = collect_amd_gpu_devices(
        drm,
        hip_inventory=inventory,
        environ={"HIP_VISIBLE_DEVICES": "3"},
    )
    assert len(devices) == 1
    gpu = devices[0]
    assert gpu.drm_card == "card7"
    assert gpu.render_node == "renderD130"
    assert gpu.hip_ordinal == 3
    assert gpu.physical_card == 11


def test_workload_mismatch_and_omitted_phases_fail_closed(tiny_checkpoint):
    root, profile, _ = tiny_checkpoint
    spec = ModelSpec.from_local_checkpoint(
        root, revision=REVISION, expected=profile
    )
    resources = _resources()
    measured_workload = _workload(spec, resources)
    measured = _measured(measured_workload, resources)
    expected = replace(
        measured_workload,
        microbatch_size=2,
        resolved_config={
            **measured_workload.resolved_config,
            "microbatch": 2,
        },
    )
    with pytest.raises(
        UnresolvedProductionFieldError, match="workload/config fingerprint"
    ):
        run_resource_preflight(
            spec,
            resources,
            measured,
            expected_workload=expected,
            require_measured=True,
        )

    incomplete = _measured(
        measured_workload,
        resources,
        omit=("synchronization", "checkpoint_save"),
    )
    with pytest.raises(
        UnresolvedProductionFieldError, match="required separate phases"
    ):
        run_resource_preflight(
            spec,
            resources,
            incomplete,
            expected_workload=measured_workload,
            require_measured=True,
        )


def test_stale_software_code_and_required_absent_dependency_fail(tiny_checkpoint):
    root, profile, _ = tiny_checkpoint
    spec = ModelSpec.from_local_checkpoint(
        root, revision=REVISION, expected=profile
    )
    resources = _resources()
    workload = _workload(spec, resources)
    measured = _measured(workload, resources)

    stale_code = replace(resources, code_fingerprint="d" * 40)
    with pytest.raises(UnresolvedProductionFieldError, match="code fingerprint"):
        run_resource_preflight(
            spec,
            stale_code,
            measured,
            expected_workload=workload,
            require_measured=True,
        )

    stale_software = replace(
        resources,
        software_versions={**resources.software_versions, "torch": "9.9"},
    )
    with pytest.raises(
        UnresolvedProductionFieldError, match="dependency/software fingerprint"
    ):
        run_resource_preflight(
            spec,
            stale_software,
            measured,
            expected_workload=workload,
            require_measured=True,
        )

    # datasets is known-absent but optional for this workload, so the base
    # profile still passes. Declaring it required turns that known fact into a
    # production failure without misclassifying it as an unknown MEASURE value.
    required_absent = replace(
        workload,
        required_dependencies=workload.required_dependencies + ("datasets",),
    )
    absent_measured = _measured(required_absent, resources)
    with pytest.raises(UnresolvedProductionFieldError, match="explicitly ABSENT"):
        run_resource_preflight(
            spec,
            resources,
            absent_measured,
            expected_workload=required_absent,
            require_measured=True,
        )


def test_measurement_provenance_repeats_and_variance_are_validated(
    tiny_checkpoint,
):
    with pytest.raises(ResourcePreflightError, match="free-form"):
        MeasuredPeakProfile.from_dict(
            {
                "workload": "dpo batch eight",
                "environment_hash": "a" * 64,
                "phases": [],
            }
        )
    with pytest.raises(ResourcePreflightError, match="argv list"):
        MeasurementProvenance.from_dict(
            {
                "command": "python measure.py",
                "tool": "rocprofv3",
            }
        )

    root, profile, _ = tiny_checkpoint
    spec = ModelSpec.from_local_checkpoint(
        root, revision=REVISION, expected=profile
    )
    resources = _resources()
    workload = _workload(spec, resources)
    measured = _measured(workload, resources)
    unstable_rank = replace(
        measured.phases[0].rank_reports[0],
        run_peak_hbm_bytes=(1_000, 1_500, 1_100),
    )
    unstable_phase = replace(
        measured.phases[0],
        rank_reports=(unstable_rank,),
        max_peak_variance_fraction=0.05,
    )
    unstable = replace(
        measured, phases=(unstable_phase,) + measured.phases[1:]
    )
    with pytest.raises(
        UnresolvedProductionFieldError, match="variance exceeds policy"
    ):
        run_resource_preflight(
            spec,
            resources,
            unstable,
            expected_workload=workload,
            require_measured=True,
        )


def test_atomic_report_interruption_preserves_previous_file(monkeypatch, tmp_path):
    import kore.policy.resources as resource_module

    destination = tmp_path / "preflight.json"
    destination.write_text('{"old": true}\n')

    def interrupted(_source, _destination):
        raise OSError("simulated interruption")

    monkeypatch.setattr(resource_module.os, "replace", interrupted)
    with pytest.raises(OSError, match="interruption"):
        atomic_write_json(destination, {"new": True})
    assert destination.read_text() == '{"old": true}\n'
    assert list(tmp_path.glob(".preflight.json.*.tmp")) == []
