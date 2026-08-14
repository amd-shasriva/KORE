"""The run must refuse to start when it cannot measure what it exists to measure.

On 2026-08-10 a full sweep ran without a baseline pass. It printed one warning --
"reference latencies available for 0 task(s)" -- and then spent a day generating,
compiling, verifying and timing 413 tasks. 302 were correct. 49 got a speedup.
246 of the remaining 253 had a valid optimized time next to a null denominator,
and none of it is recoverable, because the workspaces are deleted after scoring.

A warning that costs a day is not a warning. These tests pin the refusal, its
override, and the ordering that makes it cheap: the gate has to fire before the
30B model is loaded, or "fail fast" costs a model load either way.

They also pin that every results file records which AgentKernelArena produced it.
AKA is a sibling clone rather than a submodule, so nothing else in this repo
captures it, and the task count alone moved 413 -> 416 on a single upstream pull
that also rewrote image_kernel timing code.
"""

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RUNNER = REPO / "scripts" / "run_agent_kernel_arena.py"
ARENA = pathlib.Path.home() / "third_party" / "AgentKernelArena"

pytestmark = pytest.mark.skipif(
    not (ARENA / "tasks").is_dir(),
    reason="AgentKernelArena checkout not present")


def _run(out_dir, *extra):
    """Invoke the real CLI. --model is deliberately nonexistent: the gate must be
    reached and decided before anything tries to load it."""
    cmd = [sys.executable, str(RUNNER), "run",
           "--arena-root", str(ARENA), "--gpu-arch", "gfx950",
           "--types", "triton2triton", "--limit", "2",
           "--out", str(out_dir), "--arm", "test",
           "--model", "/nonexistent/model", *extra]
    return subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                          timeout=600)


def test_a_run_without_a_baseline_refuses(tmp_path):
    p = _run(tmp_path / "empty")
    assert p.returncode == 2, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    assert "REFUSING TO RUN" in p.stdout
    # The message has to say what to do, not just that something is wrong.
    assert "baseline" in p.stdout and "--out" in p.stdout


def test_the_refusal_happens_before_the_model_is_loaded(tmp_path):
    """Ordering, not decoration.

    The gate originally sat after ``policy = model_policy(...)``, so a refusal
    still cost minutes and ~60GB of host RAM to reach. The proof that it now
    precedes the load is that a model path which cannot possibly resolve never
    produces a loader error.
    """
    p = _run(tmp_path / "empty")
    combined = p.stdout + p.stderr
    assert "REFUSING TO RUN" in combined
    for loader_noise in ("FloatingRevisionError", "Traceback",
                         "does not appear to have a file named",
                         "OSError"):
        assert loader_noise not in combined, (
            f"the gate let execution reach the model loader ({loader_noise})")


def test_the_override_is_explicit_and_works(tmp_path):
    """Scoring correctness only is a legitimate choice; it just has to be chosen.

    With the override the gate must not fire, which is shown by execution getting
    far enough to fail on the nonexistent model instead.
    """
    p = _run(tmp_path / "empty", "--allow-missing-baseline")
    assert "REFUSING TO RUN" not in p.stdout
    assert p.returncode != 2
    assert p.returncode != 0, "a nonexistent model should still fail the run"


def test_a_present_baseline_opens_the_gate(tmp_path):
    """A baseline ledger with a usable denominator must let the run proceed."""
    out = tmp_path / "withbase"
    out.mkdir()
    tasks = subprocess.run(
        [sys.executable, str(RUNNER), "discover", "--arena-root", str(ARENA),
         "--gpu-arch", "gfx950", "--types", "triton2triton"],
        cwd=str(REPO), capture_output=True, text=True, timeout=300)
    assert tasks.returncode == 0

    # Two rows, matching the --limit 2 the gate will measure against. Only the
    # fields the reader requires: a task id, correct=True, and a denominator.
    from kore.eval.agent_kernel_arena import discover_tasks
    ids = [t.task_id for t in discover_tasks(
        ARENA, task_types=["triton2triton"], gpu_arch="gfx950")][:2]
    with (out / "baseline.partial.jsonl").open("w") as fh:
        for tid in ids:
            fh.write(json.dumps({
                "task_id": tid, "task_type": "triton2triton",
                "compiled": True, "correct": True,
                "optimized_seconds": 1.25,
                "perf_cases": [{"shape": [8], "execution_time_ms": 1.25}],
            }) + "\n")

    p = _run(out)
    assert "REFUSING TO RUN" not in p.stdout, p.stdout[-1500:]
    assert "reference latencies available for 2 task(s)" in p.stdout


def test_only_a_correct_baseline_counts_toward_coverage(tmp_path):
    """A reference that failed its own correctness check is not a denominator.

    Counting it would open the gate on a baseline that cannot produce a valid
    ratio, which is the failure this gate exists to prevent, one level down.
    """
    out = tmp_path / "badbase"
    out.mkdir()
    from kore.eval.agent_kernel_arena import discover_tasks
    ids = [t.task_id for t in discover_tasks(
        ARENA, task_types=["triton2triton"], gpu_arch="gfx950")][:2]
    with (out / "baseline.partial.jsonl").open("w") as fh:
        for tid in ids:
            fh.write(json.dumps({
                "task_id": tid, "task_type": "triton2triton",
                "compiled": True, "correct": False,      # <- the point
                "optimized_seconds": 1.25,
            }) + "\n")

    p = _run(out)
    assert p.returncode == 2
    assert "REFUSING TO RUN" in p.stdout


# ------------------------------------------------------- provenance ---
def test_every_results_file_records_which_arena_produced_it(tmp_path):
    from scripts.run_agent_kernel_arena import _arena_provenance

    prov = _arena_provenance(ARENA)
    assert len(prov.get("arena_commit", "")) == 40, prov
    assert prov.get("arena_described")
    assert "arena_dirty_files" in prov


def test_provenance_never_breaks_a_run(tmp_path):
    """It is metadata. A missing git, or a path that is not a repo, must not cost
    a sweep its results file."""
    from scripts.run_agent_kernel_arena import _arena_provenance

    prov = _arena_provenance(tmp_path / "not-a-repo")
    assert isinstance(prov, dict)
    assert prov["arena_root"].endswith("not-a-repo")
    assert "arena_commit" not in prov or prov["arena_commit"] == ""
