"""Mining must not be able to lock the arena out of the cluster.

The two are submitted by different owners against one job cap: the supervisor
submits the arena, staff_datagen submits miners, and neither asks the other
first. An arm's allocation ends every 8 hours, and its slot is then free for
exactly as long as it takes the next staffing pass to take it -- after which
the supervisor finds the cap full and, having had no branch for that, did
nothing and said nothing. The v4 arm sat dead for 50 minutes that way, with its
node still reserved for it and six miners and a gate holding all eight slots.

Two invariants close it: staffing holds a slot back for an arm that is absent,
and the supervisor says when it is blocked instead of looking idle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def staff() -> str:
    return (SCRIPTS / "staff_datagen.sh").read_text()


@pytest.fixture(scope="module")
def supervise() -> str:
    return (SCRIPTS / "supervise.sh").read_text()


def test_staffing_holds_back_slots_for_absent_arena_arms(staff):
    assert "arena_reserve" in staff, "staffing does not reserve for the arena"
    assert "free=$(( free - reserve ))" in staff, \
        "the reserve is computed but never subtracted from usable slots"


def test_reserve_counts_both_arms(staff):
    """One arm was the original assumption and is why this broke."""
    assert "kore-aka:" in staff and "kore-aka-base:" in staff


def test_running_arena_costs_no_reserve(staff):
    """A reserve charged for a *running* arena would idle slots permanently."""
    assert '-eq 0 ] && n=$(( n + 1 ))' in staff, \
        "reserve must count absent arms, not all arms"


def test_finished_arena_costs_no_reserve(staff):
    """An arm that has finished is not absent, it is done. Reserving for it
    would idle one of eight slots for the rest of the run."""
    assert "results_v4.json" in staff and "results_base.json" in staff, \
        "reserve cannot tell a finished arm from a missing one"
    body = staff.split("arena_reserve()")[1].split("\n}")[0]
    assert 'continue' in body, "no early-out for a completed arm"


def test_gate_gets_a_slot_held_for_it(staff):
    """Mining expands to the cap, and the gate is the only route by which a new
    seed becomes mineable -- 1,500 sat ungated behind exactly this."""
    assert "gate_reserve" in staff
    assert "$(arena_reserve) + $(gate_reserve)" in staff, \
        "the gate reserve is computed but not subtracted"


def test_gate_reserve_releases_when_nothing_needs_gating(staff):
    """A caught-up pipeline must give every slot back to mining."""
    body = staff.split("gate_reserve()")[1].split("\n}")[0]
    assert "echo 0" in body, "reserve never releases"


def test_supervisor_reports_being_blocked(supervise):
    assert "waiting for a slot" in supervise, \
        "a locked-out arena still reads as idle"


def test_slot_check_precedes_should_submit(supervise):
    """should_submit advances the backoff clock, so a capacity block must not
    consume a retry on a problem it cannot fix."""
    blocked = supervise.index('! have_slot')
    submit = supervise.index('should_submit aka;')
    assert blocked < submit, "capacity is still checked after the backoff clock"
