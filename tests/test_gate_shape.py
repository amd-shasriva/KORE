"""The gate must run a task at the shape the task declares.

``driver.py`` defaults to ``--shape default``, which ``_parse_shape`` resolves to
``{"M": 4096, "N": 8192}``. That is correct for a generated pool op, which is
M/N shaped, and meaningless for a registry task, whose reference asks for the
dimensions it was authored with -- ``shape["B"]``, ``shape["H"]``,
``shape["HKV"]``.

So every registry breadth twin raised KeyError inside ``get_inputs``, before the
candidate was called at all, and the gate recorded it against the kernel:
``KeyError: 'B'`` and its siblings were 284 of roughly 451 registry FlyDSL
failures. It is also the cleanest explanation for the gap between the pool twins
at 96.5% and the registry twins far below it. The shape was in the task's own
task.yaml the whole time; the gate was not passing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from verify_pool_hip_seeds import shape_arg  # noqa: E402


def test_registry_shape_is_passed_by_name():
    cfg = {"shapes": {"primary": {"B": 2, "H": 32, "HKV": 32, "SQ": 1024,
                                  "SK": 768, "D": 128}}}
    flag, value = shape_arg(cfg)
    assert flag == "--shape"
    assert value == "B=2,H=32,HKV=32,SQ=1024,SK=768,D=128"


def test_parse_shape_round_trips_what_we_emit():
    """The driver has to be able to read back exactly what the gate writes."""
    from kore.tasks._genops import _parse_shape

    dims = {"B": 2, "H": 32, "HKV": 32, "SQ": 1024, "SK": 768, "D": 128}
    _, value = shape_arg({"shapes": {"primary": dims}})
    assert _parse_shape(value) == dims


def test_pool_tasks_keep_the_driver_default():
    """A generated pool op names no dimensions and its default is right; adding
    a flag there would change 96.5%-passing behaviour for no reason."""
    assert shape_arg({"task_id": "kbk_x__hip"}) == []
    assert shape_arg({"shapes": {}}) == []
    assert shape_arg({"shapes": {"primary": {}}}) == []


def test_non_integer_dimensions_are_dropped():
    """shape_policy strings and nested entries are not dimensions, and would
    make an unparseable --shape that fails every task in the root."""
    cfg = {"shapes": {"primary": {"B": 2, "note": "big", "sub": {"x": 1}}}}
    assert shape_arg(cfg) == ["--shape", "B=2"]


def test_the_gate_actually_passes_it():
    src = (REPO / "scripts" / "verify_pool_hip_seeds.py").read_text()
    assert "*shape_arg(cfg)" in src, "the gate still runs every task at the default shape"


def test_default_shape_is_the_gemm_one_this_guards_against():
    """Pinned so the failure mode stays legible if the default ever changes."""
    from kore.tasks._genops import _parse_shape

    assert _parse_shape("default") == {"M": 4096, "N": 8192}
    assert "B" not in _parse_shape("default")
