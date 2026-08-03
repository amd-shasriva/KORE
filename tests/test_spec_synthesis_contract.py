"""What makes a spec-synthesis task a spec-synthesis task, pinned.

These tasks are defined by three properties, and each one is load-bearing in a
way that fails silently if it breaks:

* the prose contract exists and names the required entry point -- otherwise the
  task is unanswerable and every rollout reports as a model error;
* the seed is a stub with no implementation -- otherwise the task hands over its
  own answer and scores the model for returning it;
* the prompt does not present the stub as a kernel to optimize -- otherwise a
  synthesis task gets answered with a one-line edit.

None of the three is visible in a task count, which is why they are asserted
here rather than left to the verification sweep. The sweep proves the tasks run
on hardware; this proves they are the shape they claim to be.
"""

from __future__ import annotations

import ast

import pytest

from kore.tasks import generate_spec, registry


def _spec_tasks():
    return sorted(
        (t for t in registry.all_tasks() if t.is_spec_synthesis),
        key=lambda t: t.task_id,
    )


EXPECTED_SPEC_TASKS = 24


def test_the_registry_exposes_the_whole_spec_family():
    tasks = _spec_tasks()
    assert len(tasks) == EXPECTED_SPEC_TASKS
    # Every declared operation x dtype is present, so a half-generated family
    # cannot read as a complete one.
    expected = {
        generate_spec.task_id_for(spec.op, dtype)
        for spec in generate_spec.SPEC_OPS
        for dtype in generate_spec.DTYPES
    }
    assert {t.task_id for t in tasks} == expected


def test_task_kind_defaults_to_optimize_and_is_never_inferred():
    """An optimize task must not become a spec task by accident, or vice versa.

    The distinction cannot be recovered from the seed: an external-pool seed
    aliases eager torch and so also has to be replaced rather than edited, but it
    still ships a working implementation. Only the declaration separates them.
    """
    for task in registry.all_tasks():
        if task.task_id.startswith("spec_"):
            assert task.task_kind == "spec_synthesis"
            assert task.is_spec_synthesis
        else:
            assert task.task_kind == "optimize"
            assert not task.is_spec_synthesis


@pytest.mark.parametrize("task", _spec_tasks(), ids=lambda t: t.task_id)
def test_the_spec_states_the_contract_and_names_the_entry_point(task):
    spec = task.spec_source
    assert spec.strip(), "a spec task with no spec is unanswerable"
    # The entry point is the one thing the model cannot guess: the driver fetches
    # it by name with getattr, so a spec that never states it is unsolvable even
    # by a correct kernel.
    assert task.operation in spec
    # The gate has to be stated, because the model is being asked to hit it.
    assert f"{task.snr_threshold:.0f} dB" in spec
    for section in ("## Definition", "## Inputs", "## Output",
                    "## Required entry point"):
        assert section in spec, f"{task.task_id}: spec is missing {section}"


@pytest.mark.parametrize("task", _spec_tasks(), ids=lambda t: t.task_id)
def test_the_seed_is_a_stub_and_not_an_implementation(task):
    """The seed declares the signature and refuses to do the work.

    If this ever holds a real kernel the task silently stops being synthesis, and
    nothing else in the suite would notice: the counts, the families and the
    oracle would all still be correct.
    """
    source = task.seed_source
    tree = ast.parse(source)
    fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert task.operation in fns, "the stub must declare the required entry point"
    assert "@triton.jit" not in source, "the stub must not contain a kernel"
    assert "NotImplementedError" in source

    body = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == task.operation)
    statements = [s for s in body.body if not isinstance(s, ast.Expr)
                  or not isinstance(getattr(s, "value", None), ast.Constant)]
    assert len(statements) == 1 and isinstance(statements[0], ast.Raise), (
        "the stub's body must be exactly a raise, so there is nothing to edit"
    )


@pytest.mark.parametrize("task", _spec_tasks(), ids=lambda t: t.task_id)
def test_the_prompt_leads_with_the_spec_and_never_calls_the_stub_a_seed(task):
    """Every prompt path must agree, because they are used at different stages.

    ``format.build_task_prompt`` renders GRPO rollouts and DPO pairs,
    ``policies.task_prompt`` renders eval and the checkpoint A/B, and
    ``harness.build_agent_user_prompt`` renders agentic episodes. A spec task that
    is framed correctly in one and as "optimize this" in another would be trained
    and evaluated on different problems.
    """
    from kore.agent.harness import build_agent_user_prompt
    from kore.eval.policies import task_prompt
    from kore.policy.format import build_task_prompt

    for render in (task_prompt(task),
                   build_task_prompt(task),
                   build_agent_user_prompt(task, task.seed_source)):
        assert "## Definition" in render, "the prompt must carry the specification"
        assert "optimize this" not in render.lower()
        assert "NotImplementedError" not in render, (
            "the stub body must not be pasted into the prompt"
        )
        assert "no starting implementation" in render.lower()


def test_an_optimize_task_still_gets_its_seed_in_the_prompt():
    """The spec branch must not have changed the shape it was added beside."""
    from kore.agent.harness import build_agent_user_prompt
    from kore.eval.policies import task_prompt

    task = registry.get_task("gen_row_rms_bf16")
    assert not task.is_spec_synthesis
    assert "@triton.jit" in task_prompt(task)
    assert "optimize this" in build_agent_user_prompt(
        task, task.seed_source).lower()


def test_a_spec_task_missing_its_spec_file_is_refused_at_load(tmp_path):
    """Fail closed: the alternative is a task that cannot be solved by anyone.

    A missing spec file leaves the model a bare signature and no contract. That
    is not a degraded task, it is an impossible one, and it would spend GPU time
    per episode to produce a guaranteed error.
    """
    from kore.tasks.base import Task

    spec = generate_spec.SPEC_OP_BY_NAME["row_rms"]
    d = generate_spec.write_task(spec, "fp32", tmp_path)
    Task.from_dir(d)  # sanity: it loads while the spec is present

    (d / generate_spec.SPEC_FILENAME).unlink()
    with pytest.raises(ValueError, match="spec_file"):
        Task.from_dir(d)


def test_the_spec_gate_is_never_looser_than_the_optimize_task_over_the_same_op():
    """A spec task must not be an easier way to earn the same operator's reward.

    Same operator, same dtype, same oracle: if the spec variant declared a lower
    SNR gate it would be a discount on the harder task, which is backwards.
    """
    for task in _spec_tasks():
        sibling = registry.find_task(f"gen_{task.operation}_{task.dtype}")
        if sibling is None:
            continue
        assert task.snr_threshold >= sibling.snr_threshold, (
            f"{task.task_id} gates at {task.snr_threshold} dB but "
            f"{sibling.task_id} gates at {sibling.snr_threshold} dB"
        )
        assert task.comparison_baseline == sibling.comparison_baseline


def test_a_spec_task_shares_its_family_with_its_optimize_sibling():
    """Splitting one operator across families by task kind would let the same
    math be trained and held out at the same time."""
    for task in _spec_tasks():
        sibling = registry.find_task(f"gen_{task.operation}_{task.dtype}")
        if sibling is None:
            continue
        assert registry.operator_family(task) == registry.operator_family(sibling)


# --------------------------------------------------------------------------- #
# Hardware evidence. These tasks are NOT yet proven on gfx950, and the point of
# what follows is that the gap cannot be quoted away.
# --------------------------------------------------------------------------- #
VERIFICATION_ARTIFACT = "data/spec_task_verification.json"
_ALLOWED_STATUSES = {"PENDING", "PASS", "FAIL"}


def _evidence() -> dict:
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / VERIFICATION_ARTIFACT
    assert path.is_file(), (
        f"{VERIFICATION_ARTIFACT} is missing. A task family with no evidence "
        "artifact at all is worse than one recorded as unproven: there is "
        "nothing for a reader to check."
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_evidence_artifact_covers_exactly_the_live_spec_family():
    """Evidence that names a different task set than the registry is stale.

    This is the failure mode the AKA overlap disclosure already hit once: the
    prose said "20/20" while the registry held 188 HIP tasks, because the number
    lived somewhere nothing checked. Tying the artifact's task list to the live
    family makes adding a spec task without re-recording evidence a test failure.
    """
    evidence = _evidence()
    assert set(evidence["tasks"]) == {t.task_id for t in _spec_tasks()}


def test_a_pending_verdict_is_never_readable_as_a_pass():
    """PENDING must stay a third value, not a default of either outcome.

    The whole reason this artifact exists while the measurement is outstanding is
    so that "not measured" cannot be quietly rendered as "verified". If the status
    is PENDING then the artifact must say so in a way a reader cannot miss, and it
    must not carry per-task pass verdicts.
    """
    evidence = _evidence()
    status = evidence["status"]
    assert status in _ALLOWED_STATUSES

    if status == "PENDING":
        assert "NOT PROVEN" in evidence["status_meaning"]
        # No per-task result rows may exist yet: a row would be a measurement,
        # and there has been none.
        assert "rows" not in evidence
        # The reason has to name what blocked it, so the next reader can act.
        assert evidence["attempt"]["why_not_completed"].strip()
        # And the CPU pre-flight must not be dressed up as hardware evidence.
        assert "proves nothing about hardware" in \
            evidence["cpu_preflight"]["meaning"]


def test_no_document_claims_the_spec_family_is_hardware_proven():
    """Docs may describe the prover; they may not claim its verdict early.

    A doc that says these tasks are proven, while the artifact says PENDING, is
    exactly the drift the docs contract exists to prevent -- and it is the claim
    that would let an unproven task be counted as coverage.
    """
    from pathlib import Path

    if _evidence()["status"] == "PASS":
        pytest.skip("family is proven; the prohibition no longer applies")

    repo = Path(__file__).resolve().parents[1]
    forbidden = ("spec tasks are proven", "spec family is proven",
                 "24 proven spec", "proven spec-synthesis")
    offenders = []
    for doc in list(repo.glob("*.md")) + list(repo.glob("docs/*.md")) + \
            list(repo.glob("kore/tasks/*.md")):
        text = doc.read_text(encoding="utf-8").lower()
        for phrase in forbidden:
            if phrase in text:
                offenders.append(f"{doc.relative_to(repo).as_posix()}: {phrase!r}")
    assert not offenders, (
        "these documents claim hardware proof the evidence artifact does not "
        "record:\n- " + "\n- ".join(offenders)
    )
