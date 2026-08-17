"""The timing lock must identify the GPU however the worker was pinned.

The lock serialises the TIMED region per physical GPU so concurrent kernels and
L2 flushes cannot inflate a measurement. Its whole value depends on the id being
per-GPU: if two workers on different GPUs derive the same id they serialise
against each other for no reason, and if a task hangs inside the timed region it
takes every other GPU's worker down with it.

A worker can be pinned three ways, and only one of them was consulted:

* ``KoreEnv(gpu=N)`` -- what distributed GRPO passes.
* ``HIP_VISIBLE_DEVICES`` -- the legacy path.
* ``ROCR_VISIBLE_DEVICES`` -- what the arena sweep uses, because it fans out one
  worker per GPU and ``kore.policy.serve`` rejects two masks at once, so it
  narrows the inherited ROCR mask and explicitly UNSETS ``HIP_VISIBLE_DEVICES``.

Reading only ``HIP_VISIBLE_DEVICES`` resolved all eight arena workers to ``"0"``,
so they shared one lock. Observed on job 13907: compiles and correctness ran in
parallel, 12 tasks finished quickly (compile failures never reach the timed
region), and then the whole node produced nothing further.

CPU only; nothing here initialises a GPU.
"""

from __future__ import annotations

import os

import pytest

PIN_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def _physid(explicit=None) -> str:
    """Mirror of the id derivation in ``KoreEnv._timing_lock``."""
    pin = explicit
    if pin is None:
        pin = (os.environ.get("HIP_VISIBLE_DEVICES")
               or os.environ.get("ROCR_VISIBLE_DEVICES")
               or "0")
    return str(pin).split(",")[0].strip() or "0"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in PIN_VARS:
        monkeypatch.delenv(name, raising=False)


def test_the_derivation_matches_the_implementation():
    """Guard the mirror above against drift in the real function."""
    import inspect

    from kore.env.kore_env import KoreEnv

    src = inspect.getsource(KoreEnv._timing_lock)
    assert 'os.environ.get("ROCR_VISIBLE_DEVICES")' in src, (
        "the timing lock no longer consults ROCR_VISIBLE_DEVICES; ROCR-pinned "
        "workers would collapse onto one lock again"
    )
    assert 'os.environ.get("HIP_VISIBLE_DEVICES")' in src


def test_rocr_pinned_workers_get_distinct_locks(monkeypatch):
    """The arena sweep's exact configuration: ROCR set, HIP unset."""
    ids = set()
    for gpu in range(8):
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", str(gpu))
        ids.add(_physid(None))
    assert ids == {str(i) for i in range(8)}, (
        f"8 ROCR-pinned workers produced {len(ids)} distinct lock id(s): {sorted(ids)}"
    )


def test_hip_still_wins_when_both_are_set(monkeypatch):
    """HIP_VISIBLE_DEVICES stays authoritative; this is an added fallback only."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "5")
    assert _physid(None) == "3"


def test_an_explicit_gpu_beats_every_mask(monkeypatch):
    """KoreEnv(gpu=N) is what distributed GRPO passes and must not be overridden."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "5")
    assert _physid(7) == "7"
    assert _physid(0) == "0"          # 0 is a real id, not a missing value


def test_a_device_list_uses_its_first_entry(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6")
    assert _physid(None) == "4"


def test_unpinned_workers_share_one_lock_by_design(monkeypatch):
    """With no pin at all, sharing is correct: they really are on the same GPU."""
    assert _physid(None) == "0"


def test_blank_and_malformed_masks_fall_back_safely(monkeypatch):
    for value in ("", "   ", ","):
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", value)
        assert _physid(None) == "0", value
