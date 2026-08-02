"""The producer of the preflight runtime identity that authorizes replay caching.

These tests are CPU-only: the HIP probe and the DRM join are the two places that
touch hardware, and both are substituted so every rejection path can be proved
deterministically. One ``gpu``-marked test at the end runs the real thing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kore.env import evaluation_contract as contract_module
from kore.env import preflight_identity as producer
from kore.env.kore_env import KoreEnv
from kore.policy import resources
from kore.policy.resources import GPUDevice, HIPProbeError
from kore.tasks.base import Shape, Task


_ENV = (
    "KORE_PREFLIGHT_RUNTIME_IDENTITY",
    "KORE_PREFLIGHT_IDENTITY_FILE",
    "KORE_PREFLIGHT_IDENTITY_GPUS",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


@pytest.fixture(autouse=True)
def _isolated_process_state(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    producer.reset_preflight_identity()
    yield
    producer.reset_preflight_identity()


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(rocm_path=str(tmp_path / "missing-rocm"))


def _hip_entry(ordinal: int, **overrides):
    entry = {
        "hip_ordinal": ordinal,
        "physical_card": ordinal,
        "pci_bdf": f"0000:{0x80 + ordinal:02x}:00.0",
        "uuid": f"uuid{ordinal}",
        "name": "AMD Instinct MI350X",
        "gcn_arch_name": "gfx950:sramecc+:xnack-",
        "total_hbm_bytes": 270566162432,
        "multi_processor_count": 256,
        "compute_check": "pass",
        "torch_version": "2.10.0+rocm7.0",
        "hip_version": "7.0.51831",
    }
    entry.update(overrides)
    return entry


def _gpu_device(entry, **overrides):
    ordinal = int(entry["hip_ordinal"])
    fields = {
        "drm_card": f"card{ordinal * 8}",
        "render_node": f"renderD{128 + ordinal * 8}",
        "pci_bdf": entry["pci_bdf"],
        "hip_reported_pci_bdf": entry["pci_bdf"],
        "uuid": entry["uuid"],
        "hip_reported_uuid": entry["uuid"],
        "physical_card": ordinal,
        "hip_ordinal": ordinal,
        "slurm_gres_id": resources.NOT_APPLICABLE,
        "slurm_allocated": resources.NOT_APPLICABLE,
        "name": entry["name"],
        "numa_node": 0,
        "total_hbm_bytes": entry["total_hbm_bytes"],
        "free_hbm_bytes": entry["total_hbm_bytes"],
        "visible": True,
    }
    fields.update(overrides)
    return GPUDevice(**fields)


def _stub_hardware(monkeypatch, entries, *, repeat=None, devices=None):
    """Substitute the two hardware touch points: the HIP probe and the DRM join."""
    probes = [list(entries), list(entries if repeat is None else repeat)]

    def probe(**_kwargs):
        return tuple(dict(item) for item in probes.pop(0))

    monkeypatch.setattr(producer, "probe_hip_inventory", probe)
    joined = (
        devices
        if devices is not None
        else tuple(_gpu_device(entry) for entry in entries)
    )
    monkeypatch.setattr(
        producer, "collect_amd_gpu_devices", lambda **_kwargs: tuple(joined)
    )


def _reasons(bundle) -> str:
    return " | ".join(item["reason"] for item in bundle["rejected"])


# --------------------------------------------------------------------------- #
# The HIP inventory probe itself
# --------------------------------------------------------------------------- #
def _canned_probe(monkeypatch, devices, *, returncode=0, stderr=""):
    def run(argv, **kwargs):
        assert "-c" in argv, "the probe must run out of process"
        return SimpleNamespace(
            returncode=returncode,
            stdout=json.dumps({"torch_version": "t", "hip_version": "h",
                               "devices": devices}),
            stderr=stderr,
        )

    monkeypatch.setattr(subprocess, "run", run)


def test_probe_translates_child_ordinals_back_to_absolute_ids(monkeypatch):
    """A masked child numbers devices 0..n-1; the identity needs the ABSOLUTE id,
    because that is what KoreEnv writes into HIP_VISIBLE_DEVICES."""
    _canned_probe(
        monkeypatch,
        [
            {"hip_ordinal": 0, "pci_bdf": "0000:e8:00.0", "uuid": "a"},
            {"hip_ordinal": 1, "pci_bdf": "0000:88:00.0", "uuid": "b"},
        ],
    )

    inventory = resources.probe_hip_inventory(ordinals=[6, 5])

    assert [entry["hip_ordinal"] for entry in inventory] == [6, 5]
    assert [entry["physical_card"] for entry in inventory] == [6, 5]
    assert inventory[0]["pci_bdf"] == "0000:e8:00.0"


def test_probe_refuses_a_device_count_the_mask_does_not_explain(monkeypatch):
    _canned_probe(
        monkeypatch,
        [
            {"hip_ordinal": 0, "pci_bdf": "0000:e8:00.0", "uuid": "a"},
            {"hip_ordinal": 1, "pci_bdf": "0000:88:00.0", "uuid": "b"},
        ],
    )

    with pytest.raises(HIPProbeError):
        resources.probe_hip_inventory(ordinals=[6])


def test_probe_reports_a_failed_child_rather_than_an_empty_inventory(monkeypatch):
    _canned_probe(monkeypatch, [], returncode=1, stderr="hipErrorNoDevice")

    with pytest.raises(HIPProbeError, match="exited 1"):
        resources.probe_hip_inventory(ordinals=[5])


def test_probe_refuses_an_unparsable_visible_device_mask():
    with pytest.raises(HIPProbeError):
        resources.probe_hip_inventory(environ={"HIP_VISIBLE_DEVICES": "gpu-five"})


# --------------------------------------------------------------------------- #
# Per-GPU rejections: a card that fails validation gets no identity, and the
# host still produces a usable bundle for the cards that passed.
# --------------------------------------------------------------------------- #
def test_a_validated_gpu_gets_an_identity_bound_to_the_live_world(
    tmp_path, monkeypatch
):
    _stub_hardware(monkeypatch, [_hip_entry(5)])

    bundle = producer.build_preflight_identity_bundle(config=_config(tmp_path))

    assert bundle["rejected"] == []
    identity = bundle["identities"][0]
    assert identity["hardware"]["selected_gpu"] == "5"
    assert identity["hardware"]["gpu_target"] == "gfx950"
    assert identity["hardware"]["id"] == "uuid5"
    assert identity["hardware"]["drm_card"] == "card40"
    assert identity["runtime"]["boot_id"] == contract_module.boot_identity()
    assert identity["runtime"]["core_code_sha256"] == (
        contract_module.core_code_digest()["sha256"])
    assert identity["bindings"] == list(producer.DECLARED_BINDINGS)


def test_a_gpu_that_computes_the_wrong_answer_gets_no_identity(tmp_path, monkeypatch):
    """Enumeration proves the driver saw a card, not that the card works."""
    _stub_hardware(
        monkeypatch, [_hip_entry(5, compute_check="mismatch"), _hip_entry(6)]
    )

    bundle = producer.build_preflight_identity_bundle(config=_config(tmp_path))

    assert [i["hardware"]["selected_gpu"] for i in bundle["identities"]] == ["6"]
    assert "compute check did not pass" in _reasons(bundle)


def test_a_gpu_whose_two_sources_disagree_on_uuid_gets_no_identity(
    tmp_path, monkeypatch
):
    """DRM/sysfs and HIP report the UUID independently; a disagreement means the
    ordinal cannot be trusted to name that physical card."""
    entry = _hip_entry(5)
    _stub_hardware(
        monkeypatch, [entry], devices=(_gpu_device(entry, uuid="a-different-card"),)
    )

    bundle = producer.build_preflight_identity_bundle(config=_config(tmp_path))

    assert bundle["identities"] == []
    assert "disagree" in _reasons(bundle)


def test_a_gpu_with_no_drm_card_joined_by_bdf_gets_no_identity(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch, [_hip_entry(5)], devices=())

    bundle = producer.build_preflight_identity_bundle(config=_config(tmp_path))

    assert bundle["identities"] == []
    assert "no DRM/sysfs card joined" in _reasons(bundle)


def test_a_gpu_that_is_not_reproducible_across_probes_gets_no_identity(
    tmp_path, monkeypatch
):
    """'stable' is a claim the identity makes, so it has to be measured twice."""
    _stub_hardware(
        monkeypatch,
        [_hip_entry(5), _hip_entry(6)],
        repeat=[_hip_entry(5, uuid="uuid5", pci_bdf="0000:99:00.0"), _hip_entry(6)],
    )

    bundle = producer.build_preflight_identity_bundle(config=_config(tmp_path))

    assert [i["hardware"]["selected_gpu"] for i in bundle["identities"]] == ["6"]
    assert "different identity between probes" in _reasons(bundle)


def test_a_gpu_that_vanishes_between_probes_gets_no_identity(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch, [_hip_entry(5), _hip_entry(6)], repeat=[_hip_entry(6)])

    bundle = producer.build_preflight_identity_bundle(config=_config(tmp_path))

    assert [i["hardware"]["selected_gpu"] for i in bundle["identities"]] == ["6"]
    assert "disappeared" in _reasons(bundle)


def test_a_gpu_with_no_reported_architecture_gets_no_identity(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch, [_hip_entry(5, gcn_arch_name="")])

    bundle = producer.build_preflight_identity_bundle(config=_config(tmp_path))

    assert bundle["identities"] == []
    assert "no GPU architecture" in _reasons(bundle)


# --------------------------------------------------------------------------- #
# Host-level rejections: nothing can be attested to at all.
# --------------------------------------------------------------------------- #
def test_an_unreadable_boot_identity_refuses_to_establish(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch, [_hip_entry(5)])
    monkeypatch.setattr(contract_module, "_BOOT_IDENTITY_CACHE", [])
    monkeypatch.setattr(contract_module, "_BOOT_ID_PATH", tmp_path / "absent")

    with pytest.raises(producer.PreflightIdentityError, match="boot identity"):
        producer.build_preflight_identity_bundle(config=_config(tmp_path))


@pytest.mark.parametrize(
    "digest,message",
    [("toolchain_digest", "toolchain"), ("core_code_digest", "core evaluator code")],
)
def test_an_unstable_fingerprint_refuses_to_establish(
    tmp_path, monkeypatch, digest, message
):
    _stub_hardware(monkeypatch, [_hip_entry(5)])
    monkeypatch.setattr(
        producer, digest, lambda *_a, **_k: {"state": "unstable", "sha256": "x"}
    )

    with pytest.raises(producer.PreflightIdentityError, match=message):
        producer.build_preflight_identity_bundle(config=_config(tmp_path))


def test_an_unusable_probe_disables_caching_without_blocking_training(
    tmp_path, monkeypatch
):
    def explode(**_kwargs):
        raise HIPProbeError("no ROCm on this host")

    monkeypatch.setattr(producer, "probe_hip_inventory", explode)

    assert producer.establish_preflight_identity(config=_config(tmp_path)) is None
    assert producer.preflight_identity_for("5", "gfx950") is None
    with pytest.raises(producer.PreflightIdentityError):
        producer.establish_preflight_identity(config=_config(tmp_path), required=True)


# --------------------------------------------------------------------------- #
# Delivery to the workers
# --------------------------------------------------------------------------- #
def test_identities_are_content_addressed_so_a_restart_keeps_its_cache(
    tmp_path, monkeypatch
):
    """No timestamps, PIDs or run ids: re-running the preflight on an unchanged
    machine must not strand a persistent replay cache."""
    _stub_hardware(monkeypatch, [_hip_entry(5)])
    first = producer.build_preflight_identity_bundle(config=_config(tmp_path))
    _stub_hardware(monkeypatch, [_hip_entry(5)])
    second = producer.build_preflight_identity_bundle(config=_config(tmp_path))

    assert first["identities"] == second["identities"]
    assert json.dumps(first["identities"], sort_keys=True) == json.dumps(
        second["identities"], sort_keys=True)
    # Wall-clock provenance is recorded, but only at bundle level where the
    # consumer's per-GPU selection keeps it out of the replay key.
    assert "established_at_utc" in first["provenance"]
    assert "established_at_utc" not in json.dumps(first["identities"])


def test_lookup_selects_this_gpu_and_refuses_every_other_pairing(
    tmp_path, monkeypatch
):
    _stub_hardware(monkeypatch, [_hip_entry(5), _hip_entry(6)])
    producer.establish_preflight_identity(config=_config(tmp_path))

    assert producer.preflight_identity_for("5", "gfx950")["hardware"]["id"] == "uuid5"
    assert producer.preflight_identity_for("6", "gfx950")["hardware"]["id"] == "uuid6"
    assert producer.preflight_identity_for("7", "gfx950") is None
    assert producer.preflight_identity_for("5", "gfx942") is None
    assert producer.preflight_identity_for(None, "gfx950") is None
    assert producer.preflight_identity_for("5", None) is None


def test_lookup_is_pure_and_returns_nothing_without_a_preflight(monkeypatch):
    def explode(**_kwargs):
        raise AssertionError("a lookup must never probe hardware")

    monkeypatch.setattr(producer, "probe_hip_inventory", explode)

    assert producer.preflight_identity_for("5", "gfx950") is None


def test_an_externally_established_bundle_is_adopted_not_overridden(
    tmp_path, monkeypatch
):
    """A launcher that ran the preflight out of process holds the stronger
    evidence; an in-process call must not replace it with its own opinion."""
    external = {
        "identity_version": 1,
        "kind": contract_module.PREFLIGHT_IDENTITY_BUNDLE_KIND,
        "identities": [
            {"hardware": {"selected_gpu": "5", "gpu_target": "gfx950",
                          "id": "from-the-launcher"}},
        ],
    }
    monkeypatch.setenv(
        contract_module.PREFLIGHT_IDENTITY_ENV, json.dumps(external))

    def explode(**_kwargs):
        raise AssertionError("an adopted bundle must not trigger a second probe")

    monkeypatch.setattr(producer, "probe_hip_inventory", explode)
    producer.establish_preflight_identity(config=_config(tmp_path))

    assert producer.preflight_identity_for("5", "gfx950")["hardware"]["id"] == (
        "from-the-launcher")


def test_a_bundle_artifact_on_disk_is_adopted(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch, [_hip_entry(5)])
    artifact = tmp_path / "identity.json"
    producer.establish_preflight_identity(
        config=_config(tmp_path), artifact_path=artifact)
    producer.reset_preflight_identity()
    monkeypatch.setenv(producer.IDENTITY_ARTIFACT_ENV, str(artifact))

    def explode(**_kwargs):
        raise AssertionError("the artifact should have been adopted")

    monkeypatch.setattr(producer, "probe_hip_inventory", explode)
    producer.establish_preflight_identity(config=_config(tmp_path))

    assert producer.preflight_identity_for("5", "gfx950")["hardware"]["id"] == "uuid5"


# --------------------------------------------------------------------------- #
# The lookup must agree with the selection KoreEnv actually computes, or the
# identity silently never matches.
# --------------------------------------------------------------------------- #
def _task(tmp_path: Path) -> Task:
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text("task_id: t\n")
    return Task(
        task_id="t", operation="identity", dtype="bf16", backend="triton",
        gpu_target="gfx950", dir=task_dir, seed_kernel_name="seed.py",
        snr_threshold=25.0, comparison_baseline="aiter",
        shapes=[Shape("primary", {"M": 8})],
    )


@pytest.mark.parametrize(
    "gpu,visible", [("5", None), ("5", "1,2"), (None, None), (None, "4,6"), (6, None)]
)
def test_resolved_selection_matches_what_kore_env_computes(
    tmp_path, monkeypatch, gpu, visible
):
    if visible is None:
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", visible)
    task = _task(tmp_path)
    config = SimpleNamespace(
        runs_dir=tmp_path / "runs", gpu_target="gfx950",
        rocm_path=str(tmp_path / "missing-rocm"),
        snr_threshold_for=lambda _dtype: 25.0,
    )
    env = KoreEnv(task, config=config, use_replay=False, gpu=gpu)

    assert producer.resolve_selected_gpu(gpu) == (
        env._gpu_selection(task)["selected_gpu"])
