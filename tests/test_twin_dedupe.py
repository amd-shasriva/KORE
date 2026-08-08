"""A twin that exists must never be bought twice.

The materializers resume from a ``seed_attempts.jsonl`` ledger kept inside the
run's own ``--out``. That makes a single root idempotent and leaves a fleet of
roots blind to each other: aiming a fresh ``--out`` at a source that an earlier
root already swept restarts it at the first task, and every teacher call
rewrites a file already on disk. It is not a slow path, it is a free one --
measured on the frontier HIP root, 514 of the 514 tasks it seeded were already
materialized under data/pool_hip, so the whole run bought nothing.

These pin the cross-root check that closes it, and the two ways of getting it
wrong that would be worse than the bug: suppressing a twin in a language that
was never generated, and mis-splitting a suffix so an unrelated task is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kore.data.twins import (  # noqa: E402
    TWIN_SUFFIXES, existing_twins, mark_exhausted, read_task_cfg)


def _twin(data: Path, root: str, name: str) -> Path:
    d = data / root / "tasks" / name
    d.mkdir(parents=True)
    return d


# ---- cross-root visibility ------------------------------------------------

def test_twin_in_another_root_is_seen(tmp_path):
    """The case that cost 514 teacher calls: a twin under a different --out."""
    _twin(tmp_path, "pool_hip", "kbk_attention_1aac5955_fp32__hip")
    assert existing_twins(TWIN_SUFFIXES["hip"], tmp_path) == {
        "kbk_attention_1aac5955_fp32"}


def test_every_root_is_scanned(tmp_path):
    _twin(tmp_path, "pool_hip", "a__hip")
    _twin(tmp_path, "pool_hip_f", "b__hipf")
    _twin(tmp_path, "pool_hip_frontier", "c__hip")
    _twin(tmp_path, "registry_hip_frontier", "d__hip")
    assert existing_twins(TWIN_SUFFIXES["hip"], tmp_path) == {"a", "b", "c", "d"}


def test_untwinned_task_is_not_reported(tmp_path):
    _twin(tmp_path, "pool_hip", "a__hip")
    assert "kbk_alter_co_attn_d4032434_fp32" not in existing_twins(
        TWIN_SUFFIXES["hip"], tmp_path)


# ---- the suffix must identify the language --------------------------------

def test_flydsl_twin_does_not_suppress_hip(tmp_path):
    """Twinning is per-language: a FlyDSL port is not a HIP seed."""
    _twin(tmp_path, "pool_flydsl", "shared_task__flydsl")
    assert existing_twins(TWIN_SUFFIXES["hip"], tmp_path) == set()
    assert existing_twins(TWIN_SUFFIXES["flydsl"], tmp_path) == {"shared_task"}


def test_functional_suffix_is_split_whole(tmp_path):
    """``x__hipf`` is x twinned functionally, not ``x_`` twinned as ``__hip``.

    Splitting on the shorter suffix first would report ``x_`` and let the real
    task x be seeded again while skipping an unrelated one.
    """
    _twin(tmp_path, "pool_hip_f", "kbk_softmax__hipf")
    assert existing_twins(TWIN_SUFFIXES["hip"], tmp_path) == {"kbk_softmax"}


def test_source_task_is_not_mistaken_for_a_twin(tmp_path):
    """The registry's hand-authored hip_* tasks are sources, not twins."""
    _twin(tmp_path, "task_pool", "hip_flash_attn_bf16")
    assert existing_twins(TWIN_SUFFIXES["hip"], tmp_path) == set()


def test_files_are_not_counted_as_twins(tmp_path):
    (tmp_path / "pool_hip" / "tasks").mkdir(parents=True)
    (tmp_path / "pool_hip" / "tasks" / "stray__hip").write_text("not a task")
    assert existing_twins(TWIN_SUFFIXES["hip"], tmp_path) == set()


def test_missing_data_dir_is_empty_not_an_error(tmp_path):
    assert existing_twins(TWIN_SUFFIXES["hip"], tmp_path / "absent") == set()


# ---- task.yaml speaks two dialects ----------------------------------------

def test_reads_json_task_cfg(tmp_path):
    """Generated pool tasks write JSON."""
    (tmp_path / "task.yaml").write_text('{"task_id": "kbk_x", "backend": "triton"}')
    assert read_task_cfg(tmp_path)["task_id"] == "kbk_x"


def test_reads_yaml_task_cfg(tmp_path):
    """Registry tasks write real YAML; json.loads on one raises."""
    (tmp_path / "task.yaml").write_text(
        "task_id: flash_attn_decode_bf16\n"
        "backend: triton\n"
        "shapes:\n"
        "  primary:\n"
        "    batch: 4\n"
        "    heads: 32\n")
    cfg = read_task_cfg(tmp_path)
    assert cfg["task_id"] == "flash_attn_decode_bf16"
    assert cfg["shapes"]["primary"]["heads"] == 32


# ---- a finished root must say so, and must be able to reopen --------------

def test_empty_sweep_marks_the_root_exhausted(tmp_path):
    mark_exhausted(tmp_path, selected=0, examined=787)
    assert (tmp_path / ".exhausted").is_file()


def test_work_clears_the_marker(tmp_path):
    """The marker must never wedge a root shut once its source grows."""
    mark_exhausted(tmp_path, selected=0, examined=787)
    mark_exhausted(tmp_path, selected=4, examined=787)
    assert not (tmp_path / ".exhausted").exists()


def test_marker_records_what_was_examined(tmp_path):
    import json

    mark_exhausted(tmp_path, selected=0, examined=787)
    rec = json.loads((tmp_path / ".exhausted").read_text())
    assert rec["examined"] == 787 and rec["selected"] == 0


def test_both_dialects_read_a_registry_task():
    """Neither dialect may be the only one that can see the frontier.

    The registry is where the difficulty is -- flash attention, fused MoE, fp8
    GEMM -- and for a while only the HIP path could parse it. FlyDSL raised on
    every registry reference.py and was left porting the pool, which is the
    launch-bound half of the corpus and 25% of the arena scored against it.
    """
    for script in ("materialize_pool_hip.py", "materialize_pool_flydsl.py"):
        src = (Path(__file__).resolve().parents[1] / "scripts" / script).read_text()
        assert "from kore.data.twins import spec_of" in src, \
            f"{script} does not use the shared spec adapter"
        assert "no _SPEC in reference.py" not in src, \
            f"{script} still rejects registry tasks outright"


def test_registry_roots_do_not_mask_each_other():
    """Both registry streams pass --source-root kore/tasks, so liveness keyed
    on that string would make each answer for the other and whichever started
    second would never run. It must key on --out."""
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "frontier_pipeline.sh").read_text()
    assert 'pgrep -f "source-root kore/tasks"' not in src
    assert "start_registry_materializer" in src
    assert "REG_FLYDSL_ROOT" in src, "the registry FlyDSL root is not wired in"

    loops = (Path(__file__).resolve().parents[1]
             / "scripts" / "ensure_loops.sh").read_text()
    assert "REG_FLYDSL_ROOT=" in loops, "root not configured for the live loop"


def test_every_seed_root_is_gated():
    """A root that is seeded but never gated produces no training rows at all."""
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "frontier_pipeline.sh").read_text()
    gate_line = next(l for l in src.splitlines() if l.strip().startswith("for root in"))
    for root in ("REG_HIP_ROOT", "REG_FLYDSL_ROOT", "HIP_ROOT", "FLYDSL_ROOT"):
        assert root in gate_line, f"{root} is seeded but never gated"


def test_flydsl_gets_a_teacher_that_can_write_flydsl():
    """opus-5 writes HIP and cannot write FlyDSL: on the port prompt it runs to
    the token ceiling and returns zero characters, every call. The stream needs
    its own model, and it must reach the process, which only works because
    load_env_local defers to an already-set variable."""
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "frontier_pipeline.sh").read_text()
    assert "FLYDSL_TEACHER_MODEL" in src
    assert 'KORE_TEACHER_MODEL="$FLYDSL_TEACHER_MODEL"' in src, \
        "the per-stream model never reaches the materializer"

    teacher = (Path(__file__).resolve().parents[1]
               / "kore" / "data" / "teacher.py").read_text()
    assert "os.environ.setdefault" in teacher, \
        "load_env_local would overwrite the per-stream model with .env.local"


def test_pipeline_skips_exhausted_and_matches_roots_not_script_names():
    """Two roots run the same script, so liveness must key on --out.

    Matching the script name reports the registry materializer as the
    pool-sourced one, and the pool root then never restarts at all.
    """
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "frontier_pipeline.sh").read_text()
    assert 'pgrep -f -- "--out $1"' in src, "liveness still keys on script name"
    assert "root_exhausted" in src, "pipeline restarts settled roots every pass"


# ---- the materializers must actually consult it ---------------------------

@pytest.mark.parametrize("script,flag", [
    ("materialize_pool_hip.py", "--reseed-existing"),
    ("materialize_pool_flydsl.py", "--reseed-existing"),
])
def test_materializers_dedupe_across_roots(script, flag):
    """Both twin paths must call the cross-root check, and both must be able
    to opt out -- regenerating deliberately is a real operation, doing it by
    accident is what this closes."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / script).read_text()
    assert "existing_twins" in src, f"{script} does not dedupe across roots"
    assert flag in src, f"{script} has no opt-out for deliberate re-seeding"
