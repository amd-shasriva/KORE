"""On real hardware: the preflight identity is what makes replay reachable.

The CPU suite (``tests/test_preflight_identity.py``,
``tests/test_replay_contract.py``) substitutes the hardware to prove every
rejection path deterministically. These tests do the opposite: they run the real
HIP probe against real cards and drive a real ``KoreEnv.evaluate``, so the
claim "a duplicate candidate stops costing a GPU evaluation" is measured rather
than mocked.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from pathlib import Path

import pytest

from kore.env import preflight_identity as producer

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def real_bundle(gpu_id: str):
    """The real preflight, run once, for the card this suite is allowed to use."""
    bundle = producer.build_preflight_identity_bundle(gpus=[int(gpu_id)])
    if not bundle["identities"]:
        pytest.skip(f"GPU {gpu_id} did not pass the preflight: {bundle['rejected']}")
    return bundle


def _replay_env(task, config, gpu, identity, tmp_path):
    from kore.env.kore_env import KoreEnv
    from kore.policy.budget import BudgetLedgerV1

    ledger = BudgetLedgerV1()
    env = KoreEnv(
        task,
        config=dataclasses.replace(config, runs_dir=Path(tmp_path)),
        use_replay=True,
        gpu=gpu,
        runtime_identity=identity,
        budget_ledger=ledger,
    )
    return env, ledger


def test_the_real_preflight_agrees_with_sysfs_about_this_card(real_bundle):
    """HIP and DRM/sysfs are independent sources; the identity only exists
    because they agreed about the same PCI BDF."""
    for identity in real_bundle["identities"]:
        hardware = identity["hardware"]
        device = Path("/sys/class/drm") / hardware["drm_card"] / "device"

        assert (device / "unique_id").read_text().strip() == hardware["id"]
        assert device.resolve().name == hardware["pci_bdf"]
        assert hardware["gpu_target"].startswith("gfx")
        assert identity["runtime"]["checks"]["compute_verified"] is True
        assert identity["runtime"]["boot_id"]


def test_a_duplicate_candidate_replays_instead_of_re_measuring(
    gpu_harness, real_bundle, tmp_path
):
    from kore.config import CONFIG

    task = gpu_harness.task
    shapes = [gpu_harness.shape("minimal")]
    identity = real_bundle["identities"][0]
    env, ledger = _replay_env(task, CONFIG, gpu_harness.gpu, identity, tmp_path)

    started = time.perf_counter()
    first = env.evaluate(task, task.seed_source, shapes=shapes, do_bench=True)
    fresh_s = time.perf_counter() - started
    started = time.perf_counter()
    second = env.evaluate(task, task.seed_source, shapes=shapes, do_bench=True)
    replay_s = time.perf_counter() - started

    assert first.validation_passed and first.wall_ms is not None
    assert ledger.replay_hits == 1, "the duplicate must be served from the cache"
    assert ledger.correctness_calls == 1, "a hit must not re-measure anything"
    # Identical because it IS the first measurement, not a fresh one that agrees.
    assert second.wall_ms == first.wall_ms
    assert replay_s < 0.5 and replay_s < fresh_s / 10, (
        f"replay took {replay_s:.3f}s against a {fresh_s:.3f}s fresh evaluation")


def test_without_an_identity_the_same_duplicate_is_re_measured(
    gpu_harness, tmp_path
):
    """The status quo this change fixes: no proof, no caching, ever."""
    from kore.config import CONFIG

    task = gpu_harness.task
    shapes = [gpu_harness.shape("minimal")]
    env, ledger = _replay_env(task, CONFIG, gpu_harness.gpu, None, tmp_path)

    env.evaluate(task, task.seed_source, shapes=shapes, do_bench=True)
    env.evaluate(task, task.seed_source, shapes=shapes, do_bench=True)

    assert ledger.replay_hits == 0
    assert ledger.correctness_calls == 2
    assert len(env._cache_obj) == 0


@pytest.mark.parametrize(
    "kind",
    ["wrong_gpu", "wrong_architecture", "stale_boot", "toolchain_drift",
     "core_code_drift"],
)
def test_a_mismatched_identity_refuses_to_authorize_caching(
    gpu_harness, real_bundle, tmp_path, kind
):
    """Each refusal is proved by an EMPTY cache, not merely by a missing hit: a
    measurement taken under unvalidated conditions must never be stored either."""
    from kore.config import CONFIG

    identity = copy.deepcopy(real_bundle["identities"][0])
    if kind == "wrong_gpu":
        identity["hardware"]["selected_gpu"] = str(
            int(identity["hardware"]["selected_gpu"]) + 1)
    elif kind == "wrong_architecture":
        identity["hardware"]["gpu_target"] = "gfx942"
    elif kind == "stale_boot":
        identity["runtime"]["boot_id"] = "00000000-0000-4000-8000-000000000000"
    elif kind == "toolchain_drift":
        identity["runtime"]["toolchain_sha256"] = "0" * 64
    else:
        identity["runtime"]["core_code_sha256"] = "0" * 64

    task = gpu_harness.task
    shapes = [gpu_harness.shape("minimal")]
    env, ledger = _replay_env(task, CONFIG, gpu_harness.gpu, identity, tmp_path)
    obs = env.evaluate(task, task.seed_source, shapes=shapes, do_bench=True)

    assert obs.validation_passed, "a refused identity must not break evaluation"
    assert ledger.replay_hits == 0
    assert len(env._cache_obj) == 0, f"{kind} must not authorize a cache write"
