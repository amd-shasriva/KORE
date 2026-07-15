"""Pillar 1 - data-time verification rigor toggling."""

from __future__ import annotations

import os

import pytest

from kore.data.verify_rigor import RIGOR_ENV, rigor_status, set_rigorous_verification


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.get(k) for k in RIGOR_ENV}
    for k in RIGOR_ENV:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_enable_sets_all_rigor_vars():
    applied = set_rigorous_verification(True)
    assert applied == RIGOR_ENV
    assert all(os.environ[k] == "1" for k in RIGOR_ENV)


def test_disable_is_noop():
    assert set_rigorous_verification(False) == {}
    assert all(k not in os.environ for k in RIGOR_ENV)


def test_respects_operator_value_unless_override():
    os.environ["KORE_COMPILE_BASELINE"] = "0"
    applied = set_rigorous_verification(True)
    assert "KORE_COMPILE_BASELINE" not in applied
    assert os.environ["KORE_COMPILE_BASELINE"] == "0"
    set_rigorous_verification(True, override=True)
    assert os.environ["KORE_COMPILE_BASELINE"] == "1"


def test_rigor_status_reports():
    set_rigorous_verification(True)
    assert rigor_status() == {k: "1" for k in RIGOR_ENV}
