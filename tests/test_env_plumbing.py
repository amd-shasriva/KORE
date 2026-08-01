"""CPU-only regression tests for KoreEnv -> Observation -> reward plumbing.

Every test here pins a defect where a measurement was taken but never reached its
consumer, or where a failure was misclassified:

* a correct kernel with noisy timing must keep its correctness credit instead of
  being labelled an infrastructure failure and dropped from the training batch;
* ``cold_cache_verified`` and the ``profile_evidence_*`` pair must be assigned
  from the configuration/evidence that actually produced the observation;
* the per-GPU timing lock must not trust a predictable path in a shared tmpdir.

No GPU, driver, or profiler is required: the subprocess boundary
(``_exec`` / ``_bench_all`` / ``_collect_profile``) is stubbed.
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import stat as stat_module
import tempfile
import types
from pathlib import Path

import pytest

from kore.analysis.roofline import make_physical_model
from kore.env import kore_env as kore_env_module
from kore.env.kore_env import KoreEnv
from kore.policy.grpo import build_kevin_samples
from kore.reward.reward import CONFIG as REWARD_CONFIG
from kore.reward.reward import compute_reward
from kore.reward.shaping import _document_fingerprint
from kore.tasks._genops import DRIVER_CAPABILITY_PROTOCOL, DRIVER_PROTOCOL_ID
from kore.tasks._genops import PUBLICATION_GUARANTEES
from kore.tasks.base import Shape, Task


_SOURCE = "def kernel(x):\n    return x + 1\n"
_CORRECT_OUT = "SNR: 99.0\nallclose: True\nmedian_ms: 1.0\n"
_MODEL = make_physical_model("mi350x")
_ENV_KEYS = (
    "KORE_BENCH_COLD",
    "KORE_NO_BENCH_BOTH",
    "KORE_TIMING_LOCK",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


def _config(tmp_path: Path, **overrides) -> types.SimpleNamespace:
    cfg = types.SimpleNamespace(
        runs_dir=tmp_path / "runs",
        gpu_target="gfx950",
        rocm_path=str(tmp_path / "missing-rocm"),
        shape_augment=False,
        shape_augment_max=6,
        snr_threshold_for=lambda _dtype: 25.0,
        atol=1e-2,
        rtol=1e-2,
        # Off by default here: the determinism re-check costs a second stub exec
        # and is exercised by its own suite.
        verifier_determinism_check=False,
        determinism_snr_tol_db=10.0,
        warmup_iters=10,
        bench_iters=30,
        min_variance_runs=3,
        max_variance_runs=5,
        cv_threshold_pct=3.0,
        baseline_cv_threshold_pct=3.0,
        paired_ratio_cv_threshold_pct=3.0,
        paired_ci_threshold_pct=3.0,
        paired_confidence_z=1.96,
        noise_floor_pct=2.0,
        profile_reward_weight=0.0,
        physics_sku="mi350x",
        physics_calibration_path=None,
        physics_model_fingerprint=_MODEL.fingerprint,
        physics_shaping_evidence_path=None,
        physics_shaping_evidence_fingerprint=None,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def _task(tmp_path: Path, task_id: str = "plumbing_gemm_bf16") -> Task:
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text(f"task_id: {task_id}\ndtype: bf16\n")
    (task_dir / "reference.py").write_text("def reference(x):\n    return x\n")
    (task_dir / "driver.py").write_text("def driver_main():\n    return 0\n")
    return Task(
        task_id=task_id,
        operation="gemm",
        dtype="bf16",
        backend="triton",
        gpu_target="gfx950",
        dir=task_dir,
        seed_kernel_name="seed_triton.py",
        snr_threshold=25.0,
        comparison_baseline="aiter",
        shapes=[Shape("primary", {"M": 128, "N": 128, "K": 128})],
        raw={"baseline_tier": "vendor"},
    )


def _publication_caps() -> dict:
    return {
        "protocol": DRIVER_CAPABILITY_PROTOCOL,
        "protocol_id": DRIVER_PROTOCOL_ID,
        "performance_eligible": True,
        **PUBLICATION_GUARANTEES,
    }


def _pairs(candidate_ms: list[float], baseline_ms: list[float]) -> list[dict]:
    return [
        {"pair": i, "order": "AB" if i % 2 == 0 else "BA",
         "candidate_ms": c, "baseline_ms": b, "ratio": b / c}
        for i, (c, b) in enumerate(zip(candidate_ms, baseline_ms))
    ]


# Candidate medians that swing far beyond the 3% CV admission gate while every
# sample stays finite and positive, i.e. the timing is COMPLETE but unadmissible.
_NOISY = _pairs([1.0, 1.6, 0.6, 1.4, 0.8], [2.0] * 5)
_QUIET = _pairs([1.0] * 5, [2.0] * 5)


def _env(tmp_path: Path, *, use_replay: bool = False, **cfg_overrides):
    task = _task(tmp_path)
    config = _config(tmp_path, **cfg_overrides)
    env = KoreEnv(task, config=config, use_replay=use_replay, gpu="0")
    return env, task, config


def _stub_subprocess(env: KoreEnv, *, caps: dict | None = None,
                     pairs: list[dict] | None = None,
                     exec_result=(0, _CORRECT_OUT, False)):
    """Replace the GPU boundary: driver handshake, correctness exec, and bench."""
    env._driver_caps_cache = caps if caps is not None else _publication_caps()
    calls: list[list[str]] = []

    def fake_exec(cmd, workdir, environ, timeout):
        calls.append([str(part) for part in cmd])
        return exec_result

    def fake_bench_all(driver, shapes, workdir, environ, snr_threshold=None):
        if pairs is None:
            return {}, False
        return {shape.name: list(pairs) for shape in shapes}, False

    env._exec = fake_exec
    env._bench_all = fake_bench_all
    return calls


def _run(env: KoreEnv, task: Task, tmp_path: Path, do_bench: bool = True):
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    return env._run(task, _SOURCE, list(task.shapes), workdir, do_bench)


# --------------------------------------------------------------------------- #
# 1. correct kernel + noisy timing is NOT an infrastructure failure
# --------------------------------------------------------------------------- #
def test_noisy_timing_keeps_correctness_and_is_not_infra(tmp_path):
    env, task, _cfg = _env(tmp_path)
    _stub_subprocess(env, pairs=_NOISY)

    obs = _run(env, task, tmp_path)

    assert obs.validation_passed, "correctness verdict must survive noisy timing"
    assert not obs.infra_error, "measurement noise is not an infrastructure failure"
    assert obs.timing_grade == "screening"
    assert obs.performance_eligible is False
    # The raw evidence is retained so the demotion is auditable.
    assert obs.cv_pct is not None and obs.cv_pct > _cfg.cv_threshold_pct
    assert set(obs.wall_by_shape) == {"primary"}
    assert "measurement noise" in (obs.error_text or "")


def test_noisy_timing_earns_correctness_credit_but_no_speed_credit(tmp_path):
    env, task, _cfg = _env(tmp_path)
    _stub_subprocess(env, pairs=_NOISY)

    noisy = compute_reward(_run(env, task, tmp_path), _SOURCE, dtype="bf16")

    env2, task2, _cfg2 = _env(tmp_path / "quiet")
    _stub_subprocess(env2, pairs=_QUIET)
    quiet = compute_reward(_run(env2, task2, tmp_path / "quiet"), _SOURCE, dtype="bf16")

    assert noisy.tier == "correct_screening"
    assert noisy.correct is True
    assert noisy.speedup is None, "an unmeasurable candidate earns zero speed credit"
    assert noisy.reward >= REWARD_CONFIG.correctness_weight
    assert "infra" not in noisy.flags
    # A quiet measurement of the same kernel still reaches the timed tier, so the
    # demotion costs only the speed term.
    assert quiet.tier == "correct_timed" and quiet.reward > noisy.reward


def test_noisy_timing_turn_is_not_dropped_from_the_training_batch(tmp_path):
    env, task, _cfg = _env(tmp_path)
    _stub_subprocess(env, pairs=_NOISY)
    obs = _run(env, task, tmp_path)
    result = compute_reward(obs, _SOURCE, dtype="bf16")

    returns, index = build_kevin_samples(
        [[result.reward]], [[result.correct]], traj_infra=[[obs.infra_error]])

    assert index == [(0, 0)], "a verified-correct turn must stay in the batch"
    assert returns and returns[0] > 0.0


@pytest.mark.parametrize(
    "exec_result",
    [
        pytest.param((-9, "", True), id="timeout"),
        pytest.param((137, "Killed\n", False), id="oom-kill"),
        pytest.param((1, "RuntimeError: HIP error: invalid device function\n", False),
                     id="hip-error"),
        pytest.param((1, "ModuleNotFoundError: No module named 'torch'\n", False),
                     id="missing-torch"),
    ],
)
def test_genuine_infrastructure_failures_stay_infra(tmp_path, exec_result):
    env, task, _cfg = _env(tmp_path)
    _stub_subprocess(env, pairs=_QUIET, exec_result=exec_result)

    obs = _run(env, task, tmp_path)

    assert obs.infra_error, "timeout/OOM/HIP/missing-torch must remain infra"
    assert not obs.validation_passed


def test_incomplete_timing_data_remains_infra(tmp_path):
    """A bench subprocess that returned no samples is still an infra failure."""
    env, task, _cfg = _env(tmp_path)
    _stub_subprocess(env, pairs=None)

    obs = _run(env, task, tmp_path)

    assert obs.infra_error
    assert obs.timing_grade == "rejected"
    assert obs.performance_eligible is False
    assert compute_reward(obs, _SOURCE, dtype="bf16").tier == "infra"


def test_noise_demoted_observation_is_never_replayed(tmp_path):
    """Re-running may well admit the same kernel, so the demotion is not cached."""
    task = _task(tmp_path)
    config = _config(tmp_path)
    env = KoreEnv(
        task,
        config=config,
        use_replay=True,
        gpu="0",
        runtime_identity={
            "identity_version": 1,
            "validated": True,
            "stable": True,
            "hardware": {
                "id": "test-gpu-0",
                "gpu_target": "gfx950",
                "selected_gpu": "0",
            },
            "runtime": {"preflight_revision": "test"},
        },
    )
    _stub_subprocess(env, pairs=_NOISY)
    runs: list[int] = []
    inner = env._run

    def counting_run(task_, source, shapes, workdir, do_bench):
        runs.append(1)
        return inner(task_, source, shapes, workdir, do_bench)

    env._run = counting_run

    first = env.evaluate(task, _SOURCE, shapes=list(task.shapes), do_bench=True)
    env.evaluate(task, _SOURCE, shapes=list(task.shapes), do_bench=True)

    assert first.timing_grade == "screening" and not first.infra_error
    assert len(runs) == 2, "a noise-demoted verdict must not produce a cache hit"
    assert len(env._cache_obj) == 0


# --------------------------------------------------------------------------- #
# 4. cold_cache_verified is assigned from the environment that was timed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [(None, True), ("1", True), ("0", False)],
)
def test_cold_cache_verified_reflects_bench_cold_setting(
        tmp_path, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("KORE_BENCH_COLD", raising=False)
    else:
        monkeypatch.setenv("KORE_BENCH_COLD", value)
    env, task, _cfg = _env(tmp_path)
    _stub_subprocess(env, pairs=_QUIET)

    obs = _run(env, task, tmp_path)

    assert obs.cold_cache_verified is expected


def test_cold_cache_verified_is_false_without_a_protocol_driver(tmp_path, monkeypatch):
    """An unrecognized driver cannot be credited with flushing L2."""
    monkeypatch.setenv("KORE_NO_BENCH_BOTH", "1")
    env, task, _cfg = _env(tmp_path)
    _stub_subprocess(
        env,
        caps={"protocol": 0, "protocol_id": "unknown", "performance_eligible": False},
        pairs=_QUIET,
    )
    env._bench_multi = lambda *a, **k: (1.0, 0.5, False)

    obs = _run(env, task, tmp_path)

    assert obs.timing_grade == "screening"
    assert obs.cold_cache_verified is False


def test_cold_cache_verified_is_false_without_timing(tmp_path):
    env, task, _cfg = _env(tmp_path)
    _stub_subprocess(env, pairs=_QUIET)

    obs = _run(env, task, tmp_path, do_bench=False)

    assert obs.validation_passed and not obs.wall_by_shape
    assert obs.cold_cache_verified is False


def test_cold_cache_helper_reads_the_child_environment_not_os_environ(monkeypatch):
    """The sandbox path rebuilds the child env, so the child mapping is the truth."""
    monkeypatch.setenv("KORE_BENCH_COLD", "1")
    caps = _publication_caps()

    assert kore_env_module._cold_cache_timing({"KORE_BENCH_COLD": "0"}, caps) is False
    assert kore_env_module._cold_cache_timing({}, caps) is True


# --------------------------------------------------------------------------- #
# 5. profile_evidence_* is assigned only from validated held-out evidence
# --------------------------------------------------------------------------- #
def _evidence_file(tmp_path: Path, family: str) -> tuple[str, str]:
    document = {
        "shaping_evidence": {
            "families": {
                family: {
                    "family": family,
                    "report_fingerprint": "sha256:report",
                    "model_fingerprint": _MODEL.fingerprint,
                    "n_points": 100,
                    "n_task_clusters": 8,
                    "normalized_cv_r2": 0.8,
                    "baseline_cv_r2": 0.1,
                    "ci95": [0.5, 0.9],
                    "adjusted_p": 0.01,
                    "coefficients": [0.5, 0.25, 0.05],
                }
            }
        }
    }
    fingerprint = _document_fingerprint(document)
    document["evidence_fingerprint"] = fingerprint
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(document))
    return str(path), fingerprint


def _profiling_env(tmp_path: Path, efficiency, **cfg_overrides):
    env, task, config = _env(tmp_path, profile_reward_weight=0.15, **cfg_overrides)
    _stub_subprocess(env, pairs=_QUIET)
    env._collect_profile = lambda *a, **k: efficiency
    return env, task, config


def test_profile_evidence_is_withheld_without_a_configured_artifact(tmp_path):
    env, task, _cfg = _profiling_env(tmp_path, 0.75)

    obs = _run(env, task, tmp_path)

    assert obs.profile_efficiency == 0.75
    assert obs.profile_evidence_passed is False
    assert obs.profile_evidence_fingerprint is None


def test_profile_evidence_is_set_from_validated_held_out_evidence(tmp_path):
    from kore.eval.generalization import family_of

    family = family_of(_task(tmp_path).task_id)
    path, fingerprint = _evidence_file(tmp_path, family)
    env, task, cfg = _profiling_env(
        tmp_path,
        0.75,
        physics_shaping_evidence_path=path,
        physics_shaping_evidence_fingerprint=fingerprint,
    )

    obs = _run(env, task, tmp_path)

    assert obs.profile_evidence_passed is True
    assert obs.profile_evidence_fingerprint == "sha256:report"
    # The P5 bonus in the reward ladder is now reachable, and it is worth exactly
    # the configured weight times the measured efficiency.
    reward_cfg = dataclasses.replace(REWARD_CONFIG, profile_reward_weight=0.15)
    scored = compute_reward(obs, _SOURCE, dtype="bf16", cfg=reward_cfg)
    baseline = compute_reward(obs, _SOURCE, dtype="bf16", cfg=REWARD_CONFIG)
    assert any(flag.startswith("profile+") for flag in scored.flags)
    assert scored.reward - baseline.reward == pytest.approx(0.15 * 0.75)


@pytest.mark.parametrize("efficiency", [None, -0.1, 1.5, float("nan")])
def test_profile_evidence_is_withheld_for_an_unusable_measurement(tmp_path, efficiency):
    from kore.eval.generalization import family_of

    family = family_of(_task(tmp_path).task_id)
    path, fingerprint = _evidence_file(tmp_path, family)
    env, task, _cfg = _profiling_env(
        tmp_path,
        efficiency,
        physics_shaping_evidence_path=path,
        physics_shaping_evidence_fingerprint=fingerprint,
    )

    obs = _run(env, task, tmp_path)

    assert obs.profile_evidence_passed is False
    assert obs.profile_evidence_fingerprint is None


def test_profile_evidence_is_withheld_for_a_stale_fingerprint(tmp_path):
    from kore.eval.generalization import family_of

    family = family_of(_task(tmp_path).task_id)
    path, _fingerprint = _evidence_file(tmp_path, family)
    env, task, _cfg = _profiling_env(
        tmp_path,
        0.75,
        physics_shaping_evidence_path=path,
        physics_shaping_evidence_fingerprint="sha256:" + "0" * 64,
    )

    obs = _run(env, task, tmp_path)

    assert obs.profile_evidence_passed is False
    assert obs.profile_evidence_fingerprint is None


def test_profile_evidence_is_not_collected_when_the_bonus_is_off(tmp_path):
    env, task, _cfg = _env(tmp_path)
    _stub_subprocess(env, pairs=_QUIET)
    env._collect_profile = lambda *a, **k: pytest.fail("profiler must stay off")

    obs = _run(env, task, tmp_path)

    assert obs.profile_efficiency is None
    assert obs.profile_evidence_passed is False
    assert obs.profile_evidence_fingerprint is None


# --------------------------------------------------------------------------- #
# 8. the per-GPU timing lock is opened safely
# --------------------------------------------------------------------------- #
@pytest.fixture
def lock_dir(tmp_path, monkeypatch) -> Path:
    """Redirect the lockfile into a private dir.

    ``tempfile.gettempdir`` memoizes its answer on first use, so setting TMPDIR is
    not enough once anything in the process has already asked for a temp dir.
    """
    directory = tmp_path / "locks"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


def _lock_path(lock_dir: Path) -> Path:
    return lock_dir / f"kore_timing_gpu_0.uid{os.getuid()}.lock"


def test_timing_lock_file_is_private_and_owned(tmp_path, lock_dir):
    env, _task_obj, _cfg = _env(tmp_path)
    path = _lock_path(lock_dir)

    with env._timing_lock():
        assert path.exists()
        info = path.stat()
        assert stat_module.S_IMODE(info.st_mode) == 0o600
        assert info.st_uid == os.getuid()


def test_timing_lock_actually_excludes_a_second_holder(tmp_path, lock_dir):
    env, _task_obj, _cfg = _env(tmp_path)
    path = _lock_path(lock_dir)

    with env._timing_lock():
        fd = os.open(path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released on exit
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_timing_lock_refuses_a_symlinked_path_and_fails_open(tmp_path, lock_dir):
    env, _task_obj, _cfg = _env(tmp_path)
    path = _lock_path(lock_dir)
    target = tmp_path / "victim"
    path.symlink_to(target)

    entered = False
    with env._timing_lock():
        entered = True

    assert entered, "an unsafe lock path must degrade to unlocked timing, not raise"
    assert not target.exists(), "O_NOFOLLOW must refuse to create through the symlink"


def test_timing_lock_refuses_a_directory_at_the_path(tmp_path, lock_dir):
    env, _task_obj, _cfg = _env(tmp_path)
    _lock_path(lock_dir).mkdir()

    with env._timing_lock():
        pass  # fail-open


def test_private_lockfile_helper_rejects_a_foreign_owner(tmp_path, monkeypatch):
    path = tmp_path / "lock"
    real_fstat = os.fstat
    monkeypatch.setattr(
        os, "fstat",
        lambda fd: types.SimpleNamespace(
            st_mode=real_fstat(fd).st_mode, st_uid=os.getuid() + 1),
    )

    assert kore_env_module._open_private_lockfile(path) is None


def test_timing_lock_can_be_disabled(tmp_path, lock_dir, monkeypatch):
    monkeypatch.setenv("KORE_TIMING_LOCK", "0")
    env, _task_obj, _cfg = _env(tmp_path)

    with env._timing_lock():
        pass

    assert not _lock_path(lock_dir).exists()


# --------------------------------------------------------------------------- #
# 6 / 7. removed dead code, and the documented sandbox posture
# --------------------------------------------------------------------------- #
def test_superseded_bench_helpers_are_gone():
    assert not hasattr(KoreEnv, "_bench_pair")
    assert not hasattr(KoreEnv, "_batch_bench_ok")


def test_sandbox_package_is_present_but_the_execution_gate_is_off(tmp_path):
    """Pins what the module comment claims about the sandbox posture."""
    env, _task_obj, _cfg = _env(tmp_path)

    assert kore_env_module._SANDBOX_AVAILABLE is True
    assert type(env.isolation_controller).__name__ == "TrustedSubprocessController"
    assert env._sandbox_enabled is False
    assert env.last_execution_status is None
