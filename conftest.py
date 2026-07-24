"""Repository-wide pytest classification.

Tests are CPU-safe unless they explicitly opt into a resource marker. Keeping
this hook at the repository root applies the same rule to top-level tests and
to tests colocated under ``kore/``.
"""

from __future__ import annotations

import pytest


_RESOURCE_MARKERS = ("gpu", "model", "network", "dependency", "release")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Give every non-optional test an explicit ``cpu`` group."""
    for item in items:
        if not any(item.get_closest_marker(name) for name in _RESOURCE_MARKERS):
            item.add_marker(pytest.mark.cpu)
