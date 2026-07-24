"""Repository-wide pytest classification.

Tests are CPU-safe unless they explicitly opt into a resource marker. Keeping
this hook at the repository root applies the same rule to top-level tests and
to tests colocated under ``kore/``.
"""

from __future__ import annotations

import pytest


_RESOURCE_MARKERS = ("gpu", "model", "network", "dependency", "release")



@pytest.fixture(autouse=True)
def _isolate_rigor_env():
    """Snapshot/restore RIGOR_ENV around every test.

    ``scripts.run_campaign`` calls ``set_rigorous_verification(True)`` during the
    datagen stage, which writes the KORE_* rigor vars into ``os.environ`` on
    purpose (production datagen subprocesses inherit them). Without isolation a
    campaign test leaks those vars into later tests and poisons the versioned
    generator-contract digest (``resolved_config_identity`` hashes RIGOR_ENV),
    e.g. breaking ``tests/test_parallel_datagen.py`` in a full-suite run.
    """
    import os
    try:
        from kore.data.verify_rigor import RIGOR_ENV
        keys = tuple(RIGOR_ENV)
    except Exception:  # noqa: BLE001 - never let isolation break collection
        keys = (
            "KORE_VERIFIED_CORRECTNESS",
            "KORE_COMPILE_BASELINE",
            "KORE_BENCH_COLD",
            "KORE_SHAPE_AUGMENT",
        )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Give every non-optional test an explicit ``cpu`` group."""
    for item in items:
        if not any(item.get_closest_marker(name) for name in _RESOURCE_MARKERS):
            item.add_marker(pytest.mark.cpu)
