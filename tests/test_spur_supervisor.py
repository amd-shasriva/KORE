from __future__ import annotations

from scripts.spur_supervise_datagen import (
    _json_line,
    factory_jobs_active,
    progress_score,
)


def test_factory_job_detection_uses_job_name_column():
    empty = "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)\n"
    active = empty + "123 amd-spur kore-factory user R 00:10 1 node\n"
    unrelated = empty + "124 amd-spur bash user R 00:10 1 node\n"

    assert not factory_jobs_active(empty)
    assert factory_jobs_active(active)
    assert not factory_jobs_active(unrelated)


def test_progress_score_counts_partial_wins_and_base_stages():
    summary = {
        "tasks": 4,
        "wins_hist": {"0": 1, "1": 1, "2": 1, "3": 1},
        "missing_repair": 1,
        "missing_groups": 2,
    }

    assert progress_score(summary) == (0 + 1 + 2 + 3) + (8 - 1 - 2)


def test_json_line_uses_last_json_object():
    output = 'noise\n{"first": 1}\nmore noise\n{"last": 2}\n'

    assert _json_line(output) == {"last": 2}
