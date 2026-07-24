"""CPU-only regression tests for replay/evaluation contract integrity."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from kore.env.kore_env import KoreEnv
from kore.env.replay import LEGACY_MIGRATION_POLICY, ReplayCache
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
)


@pytest.fixture(autouse=True)
def _clear_contract_environment(monkeypatch):
    for name in _CONTRACT_ENV:
        monkeypatch.delenv(name, raising=False)


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        runs_dir=tmp_path / "runs",
        gpu_target="gfx950",
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


def _task(tmp_path: Path) -> Task:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
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
    env = KoreEnv(task, config=config, use_replay=True)
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
