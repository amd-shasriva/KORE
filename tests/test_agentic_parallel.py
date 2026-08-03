"""Concurrency contract for agentic trajectory generation.

Episodes are independent samples, so running them concurrently is safe only if
each gets its own env -- the harness builds, tests and benches through it. These
tests pin both halves of that contract: real overlap when an env_factory is
supplied, and an honest fall back to serial when one is not, rather than a
thread pool that is secretly serialised behind a lock.
"""

from __future__ import annotations

import threading
import time

from kore.data import gen_agentic


class _Episode:
    def __init__(self):
        self.tool_trace = []
        self.turns_used = 1
        self.success = True
        self.best_reward = 1.0
        self.turns_to_best = 1
        self.best_kernel = "k"


class _Harness:
    """Stands in for AgentHarness; sleeps to represent teacher + env latency."""

    seen_envs: list = []
    concurrent = 0
    peak = 0
    _lock = threading.Lock()

    def __init__(self, task, teacher, env, max_turns=8):
        self.env = env

    def run(self):
        with _Harness._lock:
            _Harness.seen_envs.append(self.env)
            _Harness.concurrent += 1
            _Harness.peak = max(_Harness.peak, _Harness.concurrent)
        time.sleep(0.05)
        with _Harness._lock:
            _Harness.concurrent -= 1
        return _Episode()


def _reset():
    _Harness.seen_envs = []
    _Harness.concurrent = 0
    _Harness.peak = 0


def _patch(monkeypatch):
    monkeypatch.setattr(gen_agentic, "AgentHarness", _Harness)
    monkeypatch.setattr(
        gen_agentic, "episode_to_record",
        lambda ep, task, teacher=None, thinking=True: type(
            "R", (), {"provenance": {"category": "success"}, "to_dict": lambda s: {}}
        )(),
    )


def test_env_factory_gives_each_episode_its_own_env(monkeypatch):
    _reset()
    _patch(monkeypatch)
    made = []

    def factory():
        e = object()
        made.append(e)
        return e

    recs = gen_agentic.generate_agentic_trajectories(
        task=object(), teacher=object(), env=object(), n=8,
        workers=4, env_factory=factory,
    )
    assert len(recs) == 8
    # No env instance may be shared between episodes.
    assert len(set(id(e) for e in _Harness.seen_envs)) == 8
    assert len(made) == 8


def test_workers_actually_overlap(monkeypatch):
    _reset()
    _patch(monkeypatch)
    gen_agentic.generate_agentic_trajectories(
        task=object(), teacher=object(), env=object(), n=8,
        workers=4, env_factory=lambda: object(),
    )
    # A thread pool that never overlaps is serial execution in disguise, which
    # is the specific bug this guards: an earlier draft held a lock around the
    # whole episode and would have passed a "did it finish" check.
    assert _Harness.peak > 1, f"no overlap observed (peak={_Harness.peak})"


def test_shared_env_falls_back_to_serial(monkeypatch):
    _reset()
    _patch(monkeypatch)
    shared = object()
    gen_agentic.generate_agentic_trajectories(
        task=object(), teacher=object(), env=shared, n=6,
        workers=8, env_factory=None,
    )
    assert _Harness.peak == 1, "a shared env must never be used concurrently"
    assert all(e is shared for e in _Harness.seen_envs)


def test_serial_path_unchanged(monkeypatch):
    _reset()
    _patch(monkeypatch)
    recs = gen_agentic.generate_agentic_trajectories(
        task=object(), teacher=object(), env=object(), n=3, workers=1,
    )
    assert len(recs) == 3
    assert _Harness.peak == 1
