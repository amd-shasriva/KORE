"""The shard runner must resolve external-pool task ids, not just registry ones.

This is the bug that broke a campaign. The partition plans over registry + pool
(~14.5k tasks) while the runner resolved ids through the registry alone (~1.3k),
so every pool task raised KeyError at episode start. 3,267 of 3,325 episodes
errored, and because failures are fast the throughput counter read 13,277
episodes/hour -- the run looked like a triumph while producing nothing.

Registry precedence is the other half of the contract: registry entries carry
the authoritative train/held-out split, so a pool id shadowing one could pull a
held-out task into training and invalidate every eval number downstream.
"""

from __future__ import annotations

import sys
import types

import pytest

from kore.data.saturated_agentic import _import_get_task


def _install(monkeypatch, registry_tasks, pool_tasks):
    """Stub both sources so the contract is tested without the real corpora."""
    reg = types.ModuleType("kore.tasks.registry")

    def get_task(task_id):
        if task_id in registry_tasks:
            return registry_tasks[task_id]
        raise KeyError(f"unknown task '{task_id}'; known: {sorted(registry_tasks)}")

    reg.get_task = get_task
    monkeypatch.setitem(sys.modules, "kore.tasks.registry", reg)

    ext = types.ModuleType("kore.tasks.external")
    ext.load_pool = lambda: list(pool_tasks.values())
    monkeypatch.setitem(sys.modules, "kore.tasks.external", ext)


class _T:
    def __init__(self, task_id, origin):
        self.task_id = task_id
        self.origin = origin


def test_registry_task_resolves(monkeypatch):
    _install(monkeypatch, {"reg_a": _T("reg_a", "registry")}, {})
    assert _import_get_task()("reg_a").origin == "registry"


def test_pool_task_resolves(monkeypatch):
    # The exact failure: a kbk_* id planned by the partition, absent from the
    # registry, must come back from the pool instead of raising.
    _install(monkeypatch, {}, {"kbk_x_fp32": _T("kbk_x_fp32", "pool")})
    assert _import_get_task()("kbk_x_fp32").origin == "pool"


def test_registry_wins_over_pool_on_collision(monkeypatch):
    _install(monkeypatch,
             {"dup": _T("dup", "registry")},
             {"dup": _T("dup", "pool")})
    assert _import_get_task()("dup").origin == "registry", (
        "a pool entry shadowing a registry id could move a held-out task into "
        "training"
    )


def test_genuinely_unknown_id_still_raises(monkeypatch):
    _install(monkeypatch, {"reg_a": _T("reg_a", "registry")},
             {"kbk_x": _T("kbk_x", "pool")})
    with pytest.raises(KeyError):
        _import_get_task()("nope")


def test_pool_is_loaded_once_and_only_when_needed(monkeypatch):
    calls = []
    reg = types.ModuleType("kore.tasks.registry")
    reg.get_task = lambda tid: (_T(tid, "registry") if tid.startswith("reg")
                                else (_ for _ in ()).throw(KeyError(tid)))
    monkeypatch.setitem(sys.modules, "kore.tasks.registry", reg)
    ext = types.ModuleType("kore.tasks.external")

    def load_pool():
        calls.append(1)
        return [_T("kbk_x", "pool")]

    ext.load_pool = load_pool
    monkeypatch.setitem(sys.modules, "kore.tasks.external", ext)

    resolve = _import_get_task()
    resolve("reg_a")
    assert calls == [], "a registry-only campaign must not pay to read the pool"
    resolve("kbk_x")
    resolve("kbk_x")
    assert calls == [1], "the pool must be cached, not re-read per episode"


def test_broken_pool_does_not_mask_the_keyerror(monkeypatch):
    reg = types.ModuleType("kore.tasks.registry")
    reg.get_task = lambda tid: (_ for _ in ()).throw(KeyError(tid))
    monkeypatch.setitem(sys.modules, "kore.tasks.registry", reg)
    ext = types.ModuleType("kore.tasks.external")

    def boom():
        raise RuntimeError("pool corrupt")

    ext.load_pool = boom
    monkeypatch.setitem(sys.modules, "kore.tasks.external", ext)
    with pytest.raises(KeyError):
        _import_get_task()("anything")


def test_concurrent_resolvers_all_wait_for_a_slow_pool_load(monkeypatch):
    """The race that cost ~870 episodes: a slow load must not be skipped.

    load_pool() takes ~60s for 13.5k tasks. An earlier version set the
    "loaded" flag before the load returned, so every other worker saw it,
    found an empty dict and re-raised the original KeyError. With 32 workers
    that failed the first ~870 episodes at 28,000/hour -- fast enough that the
    throughput counter looked like a triumph.
    """
    import threading
    import time as _time

    reg = types.ModuleType("kore.tasks.registry")
    reg.get_task = lambda tid: (_ for _ in ()).throw(KeyError(tid))
    monkeypatch.setitem(sys.modules, "kore.tasks.registry", reg)

    loads = []
    ext = types.ModuleType("kore.tasks.external")

    def slow_load():
        loads.append(1)
        _time.sleep(0.3)                      # stands in for the ~60s real load
        return [_T("kbk_x", "pool")]

    ext.load_pool = slow_load
    monkeypatch.setitem(sys.modules, "kore.tasks.external", ext)

    resolve = _import_get_task()
    results, errors = [], []

    def worker():
        try:
            results.append(resolve("kbk_x").origin)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"{len(errors)}/16 workers raced past the load: {errors[:2]}"
    assert results == ["pool"] * 16
    assert loads == [1], "the pool must load exactly once, not per worker"
