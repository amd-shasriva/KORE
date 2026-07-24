"""CPU-only regression tests for replay/evaluation contract integrity."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kore.env.kore_env import KoreEnv
from kore.env.replay import (
    LEGACY_MIGRATION_POLICY,
    ReplayCache,
    kernel_hash,
    source_key,
)
from kore.env import evaluation_contract as contract_module
from kore.reward.reward import Observation
from kore.tasks.base import Shape, Task


_SOURCE = "def candidate(x):\n    return x\n"
_CONTRACT_ENV = (
    "KORE_VERIFIED_CORRECTNESS",
    "KORE_SHAPE_AUGMENT",
    "KORE_COMPILE_BASELINE",
    "KORE_BENCH_COLD",
    "KORE_CORRECTNESS_TRIALS",
    "KORE_FP8_ENCODING",
    "KORE_NO_BENCH_BOTH",
    "KORE_TIMING_LOCK",
    "KORE_PREFLIGHT_RUNTIME_IDENTITY",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


@pytest.fixture(autouse=True)
def _clear_contract_environment(monkeypatch):
    for name in _CONTRACT_ENV:
        monkeypatch.delenv(name, raising=False)


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        runs_dir=tmp_path / "runs",
        gpu_target="gfx950",
        rocm_path=str(tmp_path / "missing-rocm"),
        shape_augment=False,
        shape_augment_max=6,
        snr_threshold_for=lambda _dtype: 25.0,
        snr_threshold_fp32=30.0,
        snr_threshold_lowp=25.0,
        atol=1e-2,
        rtol=1e-2,
        verifier_determinism_check=True,
        determinism_snr_tol_db=10.0,
        warmup_iters=10,
        bench_iters=30,
        min_variance_runs=3,
        max_variance_runs=5,
        cv_threshold_pct=3.0,
        profile_reward_weight=0.0,
    )


def _runtime_identity(
    gpu: str = "0",
    *,
    hardware_id: str = "test-gpu-0",
    revision: str = "runtime-a",
    gpu_target: str = "gfx950",
) -> dict:
    return {
        "identity_version": 1,
        "validated": True,
        "stable": True,
        "hardware": {
            "id": hardware_id,
            "gpu_target": gpu_target,
            "selected_gpu": str(gpu),
        },
        "runtime": {"preflight_revision": revision},
    }


def _task(tmp_path: Path) -> Task:
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "task_id: replay_test\n"
        "dtype: bf16\n"
        "gpu_target: gfx950\n"
        "targets:\n"
        "  comparison_baseline: aiter\n"
    )
    (task_dir / "reference.py").write_text("def reference(x):\n    return x\n")
    (task_dir / "driver.py").write_text("def driver_main():\n    return 0\n")
    return Task(
        task_id="replay_test",
        operation="identity",
        dtype="bf16",
        backend="triton",
        gpu_target="gfx950",
        dir=task_dir,
        seed_kernel_name="seed_triton.py",
        snr_threshold=25.0,
        comparison_baseline="aiter",
        shapes=[
            Shape("primary", {"M": 128}),
            Shape("validation_0", {"M": 257}),
        ],
        raw={"baseline_tier": "vendor"},
    )


def _correct_observation(task: Task, shapes: list[Shape], timed: bool) -> Observation:
    names = [shape.name for shape in shapes]
    wall = {name: 1.0 for name in names} if timed else {}
    baseline = {name: 2.0 for name in names} if timed else {}
    return Observation(
        compiled=True,
        dtype=task.dtype,
        validation_passed=True,
        snr_db=40.0,
        snr_by_shape={name: 40.0 for name in names},
        wall_ms=1.0 if timed else None,
        baseline_ms=2.0 if timed else None,
        wall_by_shape=wall,
        baseline_by_shape=baseline,
        cv_pct=1.0 if timed else None,
    )


class _Runner:
    def __init__(self):
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def __call__(self, task, source, shapes, workdir, do_bench):
        del source, workdir
        self.calls.append((tuple(shape.name for shape in shapes), bool(do_bench)))
        return _correct_observation(task, shapes, bool(do_bench))


def _env(tmp_path: Path) -> tuple[KoreEnv, Task, SimpleNamespace, _Runner]:
    task = _task(tmp_path)
    config = _config(tmp_path)
    env = KoreEnv(
        task,
        config=config,
        use_replay=True,
        gpu="0",
        runtime_identity=_runtime_identity(),
    )
    runner = _Runner()
    env._run = runner
    return env, task, config, runner


def test_identical_context_hits_without_rerunning(tmp_path):
    env, task, _config_obj, runner = _env(tmp_path)
    shapes = [task.shapes[0]]

    first = env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    second = env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    assert first.validation_passed and second.validation_passed
    assert runner.calls == [(("primary",), False)]
    assert len(env._cache_obj) == 1


def test_correctness_only_cannot_satisfy_timed_request(tmp_path):
    env, task, _config_obj, runner = _env(tmp_path)
    shapes = [task.shapes[0]]

    untimed = env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    timed = env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)
    timed_hit = env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)

    assert untimed.wall_ms is None
    assert timed.wall_ms == timed_hit.wall_ms == 1.0
    assert runner.calls == [(("primary",), False), (("primary",), True)]
    assert len(env._cache_obj) == 2


def test_primary_shape_cannot_satisfy_all_shape_request(tmp_path):
    env, _task_obj, _config_obj, runner = _env(tmp_path)

    env.step(_SOURCE, full_validation=False, multi_shape=False)
    all_shapes = env.step(_SOURCE, full_validation=False, multi_shape=True)
    env.step(_SOURCE, full_validation=False, multi_shape=True)

    assert set(all_shapes.snr_by_shape) == {"primary", "validation_0"}
    assert runner.calls == [
        (("primary",), False),
        (("primary", "validation_0"), False),
    ]


def test_weak_rigor_cannot_satisfy_strong_rigor_request(tmp_path, monkeypatch):
    env, task, _config_obj, runner = _env(tmp_path)
    shapes = [task.shapes[0]]
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)

    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    monkeypatch.setenv("KORE_SHAPE_AUGMENT", "1")
    monkeypatch.setenv("KORE_COMPILE_BASELINE", "1")
    monkeypatch.setenv("KORE_BENCH_COLD", "1")
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)

    assert runner.calls == [(("primary",), True), (("primary",), True)]


def test_architecture_change_invalidates(tmp_path):
    env, task, _config_obj, runner = _env(tmp_path)
    shapes = [task.shapes[0]]
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    task.gpu_target = "gfx942"
    env._runtime_identity = _runtime_identity(gpu_target="gfx942")
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    assert len(runner.calls) == 2


@pytest.mark.parametrize("filename", ["task.yaml", "reference.py", "driver.py"])
def test_task_contract_content_change_invalidates(tmp_path, filename):
    env, task, _config_obj, runner = _env(tmp_path)
    shapes = [task.shapes[0]]
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    path = task.dir / filename
    path.write_text(path.read_text() + "\n# contract changed\n")
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    assert len(runner.calls) == 2


def test_baseline_config_determinism_and_augmentation_changes_invalidate(
    tmp_path, monkeypatch
):
    env, task, config, runner = _env(tmp_path)
    shapes = [task.shapes[0]]
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)

    task.comparison_baseline = "torch_compile"
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)
    config.bench_iters += 1
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)
    config.verifier_determinism_check = False
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)
    config.shape_augment = True
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)
    monkeypatch.setenv("KORE_BENCH_COLD", "0")
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)

    assert len(runner.calls) == 6


def test_successful_timed_request_without_timings_is_not_cached(tmp_path):
    env, task, _config_obj, runner = _env(tmp_path)

    def correctness_only(task, source, shapes, workdir, do_bench):
        del source, workdir, do_bench
        runner.calls.append((tuple(shape.name for shape in shapes), True))
        return _correct_observation(task, shapes, timed=False)

    env._run = correctness_only
    shapes = [task.shapes[0]]
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=True)

    assert len(runner.calls) == 2
    assert len(env._cache_obj) == 0


def test_infrastructure_failure_remains_uncached(tmp_path):
    env, task, _config_obj, runner = _env(tmp_path)
    calls = 0

    def flaky(task, source, shapes, workdir, do_bench):
        nonlocal calls
        del source, workdir
        calls += 1
        if calls == 1:
            return Observation(
                compiled=True,
                dtype=task.dtype,
                infra_error=True,
                error_text="infra: transient timeout",
            )
        return _correct_observation(task, shapes, bool(do_bench))

    env._run = flaky
    shapes = [task.shapes[0]]
    first = env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    second = env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    third = env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    assert first.infra_error
    assert second.validation_passed and third.validation_passed
    assert calls == 2


def test_legacy_and_malformed_tail_are_ignored_then_new_data_is_readable(tmp_path):
    path = tmp_path / "replay.jsonl"
    legacy = {
        "key": "legacy-key",
        "task_id": "task",
        "obs": {"compiled": True},
    }
    path.write_bytes(
        (json.dumps(legacy) + "\n" + '{"schema_version":2,"torn":').encode("utf-8")
    )
    context = {"contract_version": 9, "case": "valid-after-tail"}
    obs = Observation(compiled=True, validation_passed=False, error_text="wrong")

    cache = ReplayCache(path)
    assert cache.get("task", "source", context) is None
    cache.put("task", "source", obs, context)

    reopened = ReplayCache(path)
    loaded = reopened.get("task", "source", context)
    assert loaded is not None and loaded.error_text == "wrong"
    assert reopened.migration_policy == LEGACY_MIGRATION_POLICY == "ignore-and-recompute"
    assert reopened.ignored_records["legacy"] >= 1
    assert reopened.ignored_records["malformed"] >= 1
    assert path.read_bytes().endswith(b"\n")


def test_unscoped_public_api_remains_usable_but_isolated(tmp_path):
    cache = ReplayCache(tmp_path / "replay.jsonl")
    obs = Observation(compiled=False, error_text="compile failed")

    cache.put("task", "source", obs)

    assert cache.get("task", "source").error_text == "compile failed"
    assert cache.get("task", "source", {"contract_version": 1}) is None


def _concurrent_writer(path: str, start: int, count: int) -> None:
    cache = ReplayCache(Path(path))
    for slot in range(start, start + count):
        source = f"source-{slot}"
        context = {"contract_version": 1, "slot": slot}
        cache.put(
            "task",
            source,
            Observation(compiled=False, error_text=f"compile-{slot}"),
            context,
        )
        if cache.get("task", source, context) is None:
            raise RuntimeError(f"writer could not replay slot {slot}")


def test_process_concurrent_appends_are_live_readable_and_durable(tmp_path):
    path = tmp_path / "replay.jsonl"
    reader = ReplayCache(path)
    process_context = multiprocessing.get_context("fork")
    workers = 4
    per_worker = 15
    processes = [
        process_context.Process(
            target=_concurrent_writer,
            args=(str(path), worker * per_worker, per_worker),
        )
        for worker in range(workers)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("concurrent replay writer hung")
        assert process.exitcode == 0

    assert len(reader) == workers * per_worker
    for slot in range(workers * per_worker):
        loaded = reader.get(
            "task",
            f"source-{slot}",
            {"contract_version": 1, "slot": slot},
        )
        assert loaded is not None and loaded.error_text == f"compile-{slot}"

    reopened = ReplayCache(path)
    assert len(reopened) == workers * per_worker


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field",
    ["snr_db", "wall_ms", "baseline_ms", "cv_pct", "profile_efficiency"],
)
def test_all_nonfinite_observation_scalars_are_rejected(tmp_path, field, value):
    cache = ReplayCache(tmp_path / "replay.jsonl")
    obs = Observation(compiled=True, validation_passed=False)
    setattr(obs, field, value)

    cache.put("task", "source", obs)

    assert len(cache) == 0
    assert cache.get("task", "source") is None


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field",
    ["snr_by_shape", "wall_by_shape", "baseline_by_shape"],
)
def test_all_nonfinite_observation_map_values_are_rejected(tmp_path, field, value):
    cache = ReplayCache(tmp_path / "replay.jsonl")
    obs = Observation(compiled=True, validation_passed=False)
    setattr(obs, field, {"primary": value})

    cache.put("task", "source", obs)

    assert len(cache) == 0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_nested_context_is_rejected(tmp_path, value):
    cache = ReplayCache(tmp_path / "replay.jsonl")
    context = {"outer": {"inner": [value]}}

    with pytest.raises(TypeError):
        source_key("task", "source", context)
    with pytest.raises(TypeError):
        cache.put("task", "source", Observation(compiled=False), context)
    with pytest.raises(TypeError):
        cache.get("task", "source", context)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_config_cannot_form_an_evaluation_contract(tmp_path, value):
    env, task, config, _runner = _env(tmp_path)
    config.cv_threshold_pct = value

    with pytest.raises(ValueError):
        contract_module.build_evaluation_contract(
            task=task,
            shapes=[task.shapes[0]],
            do_bench=True,
            config=config,
            snr_threshold=task.snr_threshold,
            correctness_timeout=env.correctness_timeout,
            bench_timeout=env.bench_timeout,
            gpu_selection=env._gpu_selection(task),
            runtime_identity=env._runtime_identity,
        )


def test_nonfinite_current_schema_record_is_ignored(tmp_path):
    path = tmp_path / "replay.jsonl"
    context = {"contract_version": 1}
    source = "source"
    record = {
        "schema_version": 2,
        "key": source_key("task", source, context),
        "task_id": "task",
        "source_sha256": kernel_hash(source),
        "context": context,
        "obs": {"compiled": True, "snr_db": math.inf},
    }
    path.write_text(json.dumps(record) + "\n")

    cache = ReplayCache(path)

    assert cache.get("task", source, context) is None
    assert cache.ignored_records["malformed"] == 1


def test_physical_gpu_and_visibility_mapping_changes_invalidate(tmp_path):
    task = _task(tmp_path)
    config = _config(tmp_path)
    runner = _Runner()
    env0 = KoreEnv(
        task,
        config=config,
        gpu="0",
        runtime_identity=_runtime_identity("0", hardware_id="gpu-serial-0"),
    )
    env1 = KoreEnv(
        task,
        config=config,
        gpu="1",
        runtime_identity=_runtime_identity("1", hardware_id="gpu-serial-1"),
    )
    env0._run = runner
    env1._run = runner
    shapes = [task.shapes[0]]

    env0.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env1.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env1.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    assert len(runner.calls) == 2
    selection = env1._gpu_selection(task)
    assert selection["selected_gpu"] == "1"
    assert selection["child_visibility"] == {
        "ROCR_VISIBLE_DEVICES": None,
        "HIP_VISIBLE_DEVICES": "1",
        "CUDA_VISIBLE_DEVICES": "1",
    }


def test_preflight_runtime_identity_change_invalidates(tmp_path):
    task = _task(tmp_path)
    config = _config(tmp_path)
    runner = _Runner()
    env_a = KoreEnv(
        task,
        config=config,
        gpu="0",
        runtime_identity=_runtime_identity(revision="runtime-a"),
    )
    env_b = KoreEnv(
        task,
        config=config,
        gpu="0",
        runtime_identity=_runtime_identity(revision="runtime-b"),
    )
    env_a._run = runner
    env_b._run = runner
    shapes = [task.shapes[0]]

    env_a.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env_b.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env_b.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    assert len(runner.calls) == 2


def test_software_version_change_invalidates(tmp_path, monkeypatch):
    env, task, _config_obj, runner = _env(tmp_path)
    version = {"torch": "2.7.0"}

    def packages():
        return {
            "torch": {
                "state": "present",
                "distribution": "torch",
                "version": version["torch"],
            },
            "triton": {"state": "not-installed"},
            "aiter": {"state": "not-installed"},
        }, True

    monkeypatch.setattr(contract_module, "_package_versions", packages)
    shapes = [task.shapes[0]]
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    version["torch"] = "2.8.0"
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    assert len(runner.calls) == 2


def test_unknown_production_identity_disables_replay(tmp_path):
    task = _task(tmp_path)
    config = _config(tmp_path)
    env = KoreEnv(task, config=config, gpu="0", runtime_identity=None)
    runner = _Runner()
    env._run = runner
    shapes = [task.shapes[0]]

    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    assert len(runner.calls) == 2
    assert len(env._cache_obj) == 0


@pytest.mark.parametrize("failure", ["unstable", "nonfinite"])
def test_unstable_or_nonfinite_preflight_identity_disables_replay(tmp_path, failure):
    task = _task(tmp_path)
    config = _config(tmp_path)
    identity = _runtime_identity()
    if failure == "unstable":
        identity["stable"] = False
    else:
        identity["runtime"]["threshold"] = math.inf
    env = KoreEnv(task, config=config, gpu="0", runtime_identity=identity)
    runner = _Runner()
    env._run = runner
    shapes = [task.shapes[0]]

    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    assert len(runner.calls) == 2
    assert len(env._cache_obj) == 0


def test_core_evaluator_code_change_invalidates(tmp_path, monkeypatch):
    core = tmp_path / "core.py"
    core.write_text("SEMANTICS = 1\n")
    monkeypatch.setattr(
        contract_module,
        "_CORE_CODE_PATHS",
        (("test/core.py", core),),
    )
    contract_module._clear_fingerprint_caches()
    env, task, _config_obj, runner = _env(tmp_path / "env")
    shapes = [task.shapes[0]]

    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    core.write_text("SEMANTICS = 2\n")
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    replacement = tmp_path / "replacement.py"
    replacement.write_text("SEMANTICS = 3\n")
    os.replace(replacement, core)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)
    env.evaluate(task, _SOURCE, shapes=shapes, do_bench=False)

    assert len(runner.calls) == 3


def test_core_fingerprint_cache_revalidates_mtime_and_replacement(tmp_path):
    core = tmp_path / "core.py"
    core.write_text("SEMANTICS = 1\n")
    paths = (("core.py", core),)
    contract_module._clear_fingerprint_caches()

    first = contract_module._fingerprint_code_paths(paths)
    first_cache_entries = len(contract_module._CODE_SET_CACHE)
    stat = core.stat()
    os.utime(core, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    touched = contract_module._fingerprint_code_paths(paths)
    replacement = tmp_path / "new-core.py"
    replacement.write_text("SEMANTICS = 2\n")
    os.replace(replacement, core)
    replaced = contract_module._fingerprint_code_paths(paths)

    assert touched["sha256"] == first["sha256"]
    assert len(contract_module._CODE_SET_CACHE) > first_cache_entries
    assert replaced["sha256"] != first["sha256"]


def test_contract_records_toolchain_core_and_validated_hardware_without_imports(tmp_path):
    env, task, config, _runner = _env(tmp_path)
    initially_unloaded = {
        name for name in ("torch", "triton", "aiter") if name not in sys.modules
    }

    contract = contract_module.build_evaluation_contract(
        task=task,
        shapes=[task.shapes[0]],
        do_bench=True,
        config=config,
        snr_threshold=task.snr_threshold,
        correctness_timeout=env.correctness_timeout,
        bench_timeout=env.bench_timeout,
        gpu_selection=env._gpu_selection(task),
        runtime_identity=env._runtime_identity,
    )

    runtime = contract["runtime"]
    assert contract_module.contract_is_cacheable(contract)
    assert runtime["effective_gpu_target"] == "gfx950"
    assert runtime["preflight_identity"]["state"] == "validated"
    assert runtime["core_code"]["state"] == "stable"
    assert len(runtime["core_code"]["sha256"]) == 64
    assert set(runtime["toolchain"]["packages"]) == {"torch", "triton", "aiter"}
    assert set(runtime["toolchain"]["compilers"]) == {"cc", "cxx", "hipcc"}
    assert all(name not in sys.modules for name in initially_unloaded)


def test_warm_contract_build_cpu_overhead_is_bounded(tmp_path):
    env, task, config, _runner = _env(tmp_path)
    kwargs = {
        "task": task,
        "shapes": [task.shapes[0]],
        "do_bench": True,
        "config": config,
        "snr_threshold": task.snr_threshold,
        "correctness_timeout": env.correctness_timeout,
        "bench_timeout": env.bench_timeout,
        "gpu_selection": env._gpu_selection(task),
        "runtime_identity": env._runtime_identity,
    }
    contract_module.build_evaluation_contract(**kwargs)  # warm hashes/version probes

    iterations = 50
    started = time.perf_counter()
    contracts = [
        contract_module.build_evaluation_contract(**kwargs)
        for _ in range(iterations)
    ]
    elapsed = time.perf_counter() - started

    assert all(contract_module.contract_is_cacheable(c) for c in contracts)
    assert elapsed / iterations < 0.025, (
        f"warm contract build averaged {elapsed / iterations:.6f}s"
    )
