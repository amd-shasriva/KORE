"""The checkpoint A/B harness, exercised end to end on CPU against stub arms.

``kore.eval.checkpoint_ab`` produced KORE's first evaluation of a trained
checkpoint against its own base (``docs/EVAL_RESULTS.md``). The number it
reported is only worth anything if the harness that produced it is itself under
test, so this module drives the WHOLE pipeline - scope selection, prompt
construction, generation, replay into the real bake-off, funnel scoring, and the
paired comparison - with a stub policy and a stub env, no GPU and no network.

Two properties are load-bearing and get dedicated tests:

  * the two arms must be prompted byte-identically, and a drift must be detected
    rather than assumed away (:func:`assert_prompts_matched`);
  * a kernel that PASSES the correctness gate but whose timing was demoted to
    screening grade must be counted as correct-and-unscoreable, not as a
    correctness failure. Conflating those two would report a verifier pass as a
    model defect, and would have been invisible in the headline number.
"""

from __future__ import annotations

import json

import pytest

from kore.eval import checkpoint_ab as ab
from kore.reward.reward import Observation

# --------------------------------------------------------------------------- #
# Fixtures: a small slice of the real scope + stub arms.
# --------------------------------------------------------------------------- #
N_TASKS = 4


@pytest.fixture(scope="module")
def tasks():
    scope = ab.generalization_scope()
    assert scope, "the generalization scope is empty"
    return scope[:N_TASKS]


def contract_gen(kernel_body: str = "import triton\n# fast kernel\n"):
    """A stub arm that honors the FULL_KERNEL response contract."""

    def generate(messages, max_tokens: int = 4096, temperature: float = 0.0) -> str:
        return ("ANALYSIS\nraise the block size\n\n"
                "PROPOSED_CHANGE\nBLOCK_M 64 -> 128\n\n"
                f"FULL_KERNEL\n```python\n{kernel_body}```\n")

    return generate


def prose_gen(messages, max_tokens: int = 4096, temperature: float = 0.0) -> str:
    """A stub arm that answers conversationally and never emits the contract."""
    return "Sure - here is how I would approach optimizing this kernel. First, ..."


class StubEnv:
    """A KoreEnv-shaped stub: one ``step(source) -> Observation``.

    ``grade`` selects the outcome under test: ``"good"`` is correct and
    publication-timed, ``"screening"`` is correct with a timing the reward
    refuses to score, ``"incorrect"`` fails the SNR gate, ``"nocompile"`` does
    not build.
    """

    def __init__(self, task, grade: str = "good", speedup: float = 1.4):
        self.task = task
        self.grade = grade
        self.speedup = speedup
        self.sources: list[str] = []

    def step(self, source: str):
        self.sources.append(source)
        shape = self.task.shapes[0].name
        if self.grade == "nocompile" or "triton" not in source:
            return Observation(compiled=False,
                               error_text="SyntaxError: invalid syntax")
        if self.grade == "incorrect":
            return Observation(compiled=True, validation_passed=False, snr_db=4.0,
                               snr_by_shape={shape: 4.0}, dtype=self.task.dtype,
                               error_text="SNR 4.0 dB below the gate")
        common = dict(
            compiled=True, validation_passed=True, snr_db=90.0,
            snr_by_shape={shape: 90.0}, requested_shapes=[shape],
            dtype=self.task.dtype, cv_pct=0.4,
        )
        if self.grade == "screening":
            # Correct, but the timing carries no admissible paired measurement:
            # compute_reward returns correct=True with speedup=None.
            return Observation(
                wall_ms=1.0, baseline_ms=self.speedup, wall_by_shape={shape: 1.0},
                baseline_by_shape={shape: self.speedup}, timing_requested=True,
                timing_grade="screening", **common)
        return Observation(
            wall_ms=1.0, baseline_ms=self.speedup, wall_by_shape={shape: 1.0},
            baseline_by_shape={shape: self.speedup}, timing_requested=False,
            **common)


def run_arm(tasks, generate, *, arm: str, grade: str = "good", speedup: float = 1.4):
    """Generate + measure + summarize one stub arm; returns ``(summary, rows)``."""
    records = ab.generate_arm(tasks, generate, arm=arm, max_tokens=64)
    rows = [r.to_dict() for r in records]
    result = ab.measure_arm(
        rows, tasks, arm=arm,
        env_factory=lambda t: StubEnv(t, grade=grade, speedup=speedup))
    return ab.arm_summary(result, rows, arm=arm), rows


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #
def test_scope_is_the_generalization_split_not_the_raw_holdout():
    """The scored scope excludes the tasks whose source leaked into pretraining."""
    from kore.tasks.registry import generalization_tasks, heldout_tasks

    scope = {t.task_id for t in ab.generalization_scope()}
    held = {t.task_id for t in heldout_tasks()}
    assert scope == {t.task_id for t in generalization_tasks()}
    assert scope < held, "the scope must be strictly smaller than the reservation"
    report = ab.scope_report(ab.generalization_scope())
    assert report["n_scored"] == len(scope)
    assert report["n_heldout_reservation"] == len(held)
    assert set(report["excluded_contaminated"]) == held - scope
    assert report["excluded_contaminated"], "no contamination exclusion recorded"
    assert report["taxonomy_digest"]


def test_scope_refuses_a_contaminated_task_id():
    """Naming a contaminated task cannot widen a zero-shot claim by hand."""
    from kore.tasks.registry import (
        ContaminatedGeneralizationError,
        generalization_tasks,
        heldout_tasks,
    )

    contaminated = sorted({t.task_id for t in heldout_tasks()}
                          - {t.task_id for t in generalization_tasks()})
    with pytest.raises(ContaminatedGeneralizationError):
        ab.generalization_scope([contaminated[0]])


def test_scope_refuses_a_training_task_id():
    from kore.tasks.registry import ContaminatedGeneralizationError, train_tasks

    with pytest.raises(ContaminatedGeneralizationError):
        ab.generalization_scope([train_tasks()[0].task_id])


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def test_first_turn_messages_match_the_live_model_policy(tasks):
    """The replayed prompt is the prompt the live bake-off would have sent."""
    from kore.eval.policies import model_policy

    seen: list = []

    def spy(messages, **kw):
        seen.append(messages)
        return "FULL_KERNEL\n```python\nimport triton\n```\n"

    policy = model_policy("spy", generate=spy)
    policy(tasks[0], None)
    assert seen[0] == ab.first_turn_messages(tasks[0])


def test_prompt_digest_is_stable_and_content_sensitive(tasks):
    a = ab.first_turn_messages(tasks[0])
    assert ab.prompt_digest(a) == ab.prompt_digest(ab.first_turn_messages(tasks[0]))
    assert ab.prompt_digest(a) != ab.prompt_digest(ab.first_turn_messages(tasks[1]))


def test_matched_prompts_pass_and_a_drift_is_caught(tasks):
    a = [r.to_dict() for r in ab.generate_arm(tasks, prose_gen, arm="a", max_tokens=8)]
    b = [r.to_dict() for r in ab.generate_arm(tasks, prose_gen, arm="b", max_tokens=8)]
    ab.assert_prompts_matched({"a": a, "b": b})  # identical prompts: no raise

    drifted = [dict(row) for row in b]
    drifted[0]["prompt_sha"] = "deadbeefcafe"
    with pytest.raises(AssertionError, match="prompted differently"):
        ab.assert_prompts_matched({"a": a, "b": drifted})


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def test_generate_arm_records_contract_compliance(tasks):
    good = ab.generate_arm(tasks, contract_gen(), arm="good", max_tokens=32)
    bad = ab.generate_arm(tasks, prose_gen, arm="bad", max_tokens=32)
    assert [r.contract_ok for r in good] == [True] * len(tasks)
    assert [r.contract_ok for r in bad] == [False] * len(tasks)
    assert all("triton" in r.kernel for r in good)
    assert all(r.kernel == "" for r in bad)
    # The fallback mirrors model_policy: no contract -> the raw response is what
    # gets benched, so an unparseable answer loses the task instead of erroring.
    assert bad[0].submitted_source == bad[0].response
    assert good[0].submitted_source == good[0].kernel


def test_generate_arm_uses_the_batched_backend_when_given_one(tasks):
    calls: list[int] = []

    def batch(messages_batch, max_tokens=4096, temperature=0.0):
        calls.append(len(messages_batch))
        return [contract_gen()(m) for m in messages_batch]

    records = ab.generate_arm(tasks, None, arm="b", generate_batch=batch,
                              batch_size=2, max_tokens=16)
    assert len(records) == len(tasks)
    assert calls == [2] * (len(tasks) // 2)
    assert all(r.contract_ok for r in records)


def test_generate_arm_needs_a_backend(tasks):
    with pytest.raises(ValueError, match="generate"):
        ab.generate_arm(tasks, None, arm="x")


def test_a_backend_failure_is_recorded_not_raised(tasks):
    def broken(messages, **kw):
        raise RuntimeError("gateway exploded")

    records = ab.generate_arm(tasks, broken, arm="broken", max_tokens=8)
    assert len(records) == len(tasks)
    assert all(r.error and "gateway exploded" in r.error for r in records)
    assert all(not r.contract_ok for r in records)


def test_samples_are_numbered_per_task(tasks):
    records = ab.generate_arm(tasks[:2], contract_gen(), arm="s", samples=3,
                              max_tokens=8)
    assert len(records) == 6
    assert sorted(r.sample for r in records if r.task_id == tasks[0].task_id) == [0, 1, 2]


def test_generations_roundtrip_through_jsonl(tmp_path, tasks):
    records = ab.generate_arm(tasks, contract_gen(), arm="rt", max_tokens=8)
    path = ab.write_generations(tmp_path / "g.jsonl", records, arm="rt", note="hi")
    meta, rows = ab.read_generations(path)
    assert meta["schema"] == ab.SCHEMA_GENERATIONS
    assert meta["n_records"] == len(records) and meta["note"] == "hi"
    assert [r["task_id"] for r in rows] == [r.task_id for r in records]
    assert rows[0]["response"] == records[0].response


# --------------------------------------------------------------------------- #
# Replay into the real bake-off
# --------------------------------------------------------------------------- #
def test_replay_policy_returns_the_recorded_kernel(tasks):
    rows = [r.to_dict() for r in ab.generate_arm(tasks, contract_gen(), arm="r",
                                                 max_tokens=8)]
    policy = ab.replay_policy(rows, arm="r")
    assert "triton" in policy(tasks[0], None)


def test_replay_policy_walks_samples_then_repeats_the_last(tasks):
    rows = []
    for s in range(2):
        rec = ab.generate_arm([tasks[0]], contract_gen(f"import triton  # s{s}\n"),
                              arm="r", max_tokens=8)[0]
        rec.sample = s
        rows.append(rec.to_dict())
    policy = ab.replay_policy(rows, arm="r")
    assert "s0" in policy(tasks[0], None)
    assert "s1" in policy(tasks[0], None)
    assert "s1" in policy(tasks[0], None), "budget beyond the samples reuses the last"


def test_replay_policy_refuses_an_ungenerated_task(tasks):
    rows = [r.to_dict() for r in ab.generate_arm([tasks[0]], contract_gen(),
                                                 arm="r", max_tokens=8)]
    policy = ab.replay_policy(rows, arm="r")
    with pytest.raises(KeyError, match="no recorded completion"):
        policy(tasks[1], None)


def test_measure_arm_benches_exactly_the_recorded_source(tasks):
    rows = [r.to_dict() for r in ab.generate_arm(tasks, contract_gen(), arm="m",
                                                 max_tokens=8)]
    envs: list[StubEnv] = []

    def factory(task):
        env = StubEnv(task, grade="good")
        envs.append(env)
        return env

    ab.measure_arm(rows, tasks, arm="m", env_factory=factory)
    assert len(envs) == len(tasks)
    for env in envs:
        assert env.sources and "triton" in env.sources[0]


def test_measure_arm_records_one_observation_per_task(tasks):
    rows = [r.to_dict() for r in ab.generate_arm(tasks, prose_gen, arm="m",
                                                 max_tokens=8)]
    res = ab.measure_arm(rows, tasks, arm="m",
                         env_factory=lambda t: StubEnv(t, grade="good"))
    obs = res["observations"]
    assert len(obs) == len(tasks)
    assert {o["task_id"] for o in obs} == {t.task_id for t in tasks}
    assert all(o["compiled"] is False for o in obs), "prose must not compile"
    assert all(o["error_tail"] for o in obs)


# --------------------------------------------------------------------------- #
# The funnel, including the correct-but-untimed stage
# --------------------------------------------------------------------------- #
def test_funnel_separates_contract_compile_correct_and_timed(tasks):
    summary, _ = run_arm(tasks, contract_gen(), arm="ok")
    counts = summary["counts"]
    assert counts == {"contract_ok": len(tasks), "compiled": len(tasks),
                      "correct": len(tasks), "timed": len(tasks),
                      "faster": len(tasks), "infra_error": 0}
    assert summary["fast_p"][1.0] == pytest.approx(1.0)
    assert summary["geometric_mean_speedup"] == pytest.approx(1.4, rel=1e-6)


def test_prose_arm_fails_at_the_contract_stage(tasks):
    summary, _ = run_arm(tasks, prose_gen, arm="prose")
    counts = summary["counts"]
    assert counts["contract_ok"] == 0
    assert counts["compiled"] == 0 and counts["correct"] == 0
    assert summary["fast_p"][0.0] == 0.0


def test_a_correct_but_screening_timed_kernel_is_correct_and_unscoreable(tasks):
    """The regression for the conflation this harness had to work around.

    ``bakeoff.evaluate_policy``'s per-task ``correct`` means "correct AND
    carrying an integrity-gated speedup", which is what fast_p needs. A kernel
    that passes the SNR gate but whose timing is only screening-grade has
    ``rr.speedup is None``, so that field is False for a kernel the verifier
    ACCEPTED. The funnel must report correct=1, timed=0.
    """
    summary, _ = run_arm(tasks, contract_gen(), arm="screen", grade="screening")
    counts = summary["counts"]
    assert counts["compiled"] == len(tasks)
    assert counts["correct"] == len(tasks), "the verifier accepted these kernels"
    assert counts["timed"] == 0, "no admissible paired timing, so nothing to score"
    assert counts["faster"] == 0
    assert summary["fast_p"][0.0] == 0.0
    assert all("timing:screening" in t["flags"] for t in summary["per_task"])


def test_bakeoff_per_task_record_exposes_both_correctness_fields(tasks):
    """``correct_gate`` / ``timed`` are additive; ``correct`` keeps its meaning."""
    rows = [r.to_dict() for r in ab.generate_arm(tasks, contract_gen(), arm="b",
                                                 max_tokens=8)]
    res = ab.measure_arm(rows, tasks, arm="b",
                         env_factory=lambda t: StubEnv(t, grade="screening"))
    for rec in res["per_task"]:
        assert rec["correct_gate"] is True
        assert rec["timed"] is False
        assert rec["correct"] is False  # unchanged fast_p semantics


def test_incorrect_arm_compiles_but_fails_the_gate(tasks):
    summary, _ = run_arm(tasks, contract_gen(), arm="wrong", grade="incorrect")
    counts = summary["counts"]
    assert counts["compiled"] == len(tasks)
    assert counts["correct"] == 0 and counts["timed"] == 0


def test_a_slower_correct_kernel_is_correct_but_not_faster(tasks):
    summary, _ = run_arm(tasks, contract_gen(), arm="slow", speedup=0.5)
    assert summary["counts"]["correct"] == len(tasks)
    assert summary["counts"]["timed"] == len(tasks)
    assert summary["counts"]["faster"] == 0
    assert summary["fast_p"][1.0] == 0.0


# --------------------------------------------------------------------------- #
# Wilson intervals
# --------------------------------------------------------------------------- #
def test_wilson_interval_stays_inside_zero_one_at_the_boundaries():
    zero = ab.wilson_interval(0, 34)
    assert zero["rate"] == 0.0 and zero["lo"] == 0.0
    assert 0.0 < zero["hi"] < 0.2, "a 0/34 rate still carries real uncertainty"
    full = ab.wilson_interval(34, 34)
    assert full["rate"] == 1.0 and full["hi"] == 1.0 and full["lo"] < 1.0
    mid = ab.wilson_interval(17, 34)
    assert mid["lo"] < 0.5 < mid["hi"]


def test_wilson_interval_narrows_with_n():
    small = ab.wilson_interval(1, 10)
    large = ab.wilson_interval(10, 100)
    assert (large["hi"] - large["lo"]) < (small["hi"] - small["lo"])


def test_wilson_interval_handles_an_empty_split():
    assert ab.wilson_interval(0, 0)["rate"] == 0.0


# --------------------------------------------------------------------------- #
# The paired comparison
# --------------------------------------------------------------------------- #
def test_comparison_detects_a_clean_win(tasks):
    good, _ = run_arm(tasks, contract_gen(), arm="cand")
    bad, _ = run_arm(tasks, prose_gen, arm="ref")
    cmp_ = ab.compare_arms(good, bad, n_boot=400)
    assert cmp_["verdict"]["direction"] == "candidate_better"
    correct = cmp_["binary"]["correct"]
    assert correct["a_count"] == len(tasks) and correct["b_count"] == 0
    assert correct["discordant_pairs"] == len(tasks)
    assert correct["a_only"] == len(tasks) and correct["b_only"] == 0
    assert cmp_["fast_p"]["delta"][1.0] > 0


def test_comparison_detects_a_regression(tasks):
    good, _ = run_arm(tasks, contract_gen(), arm="ref")
    bad, _ = run_arm(tasks, prose_gen, arm="cand")
    cmp_ = ab.compare_arms(bad, good, n_boot=400)
    assert cmp_["verdict"]["direction"] == "reference_better"
    assert cmp_["binary"]["correct"]["b_only"] == len(tasks)


def test_exact_mcnemar_p_matches_the_sign_test_on_the_deltas(tasks):
    from kore.eval.paired_stats import sign_test

    good, _ = run_arm(tasks, contract_gen(), arm="cand")
    bad, _ = run_arm(tasks, prose_gen, arm="ref")
    cmp_ = ab.compare_arms(good, bad, n_boot=200)
    expected = sign_test([1.0] * len(tasks))
    assert cmp_["binary"]["correct"]["p_sign"] == pytest.approx(expected.p_value)


def test_zero_correct_on_both_sides_is_reported_as_no_signal_not_parity(tasks):
    """An absence of measurement must never be written up as "no regression"."""
    a, _ = run_arm(tasks, prose_gen, arm="cand")
    b, _ = run_arm(tasks, prose_gen, arm="ref")
    verdict = ab.compare_arms(a, b, n_boot=200)["verdict"]
    assert verdict["direction"] == "no_signal"
    assert verdict["significant"] is False
    assert "NOT evidence of parity" in verdict["statement"]


def test_identical_arms_are_a_tie_with_no_discordant_pairs(tasks):
    a, _ = run_arm(tasks, contract_gen(), arm="cand")
    b, _ = run_arm(tasks, contract_gen(), arm="ref")
    cmp_ = ab.compare_arms(a, b, n_boot=200)
    assert cmp_["verdict"]["direction"] == "tie"
    assert cmp_["binary"]["correct"]["discordant_pairs"] == 0


def test_paired_speedup_uses_only_the_both_timed_tasks(tasks):
    fast, _ = run_arm(tasks, contract_gen(), arm="cand", speedup=2.0)
    slow, _ = run_arm(tasks, contract_gen(), arm="ref", speedup=1.0)
    cmp_ = ab.compare_arms(fast, slow, n_boot=1000)
    su = cmp_["speedup"]
    assert su is not None
    assert su["n"] == len(tasks)
    assert su["effect_kind"] == "geomean_speedup_ratio"
    assert su["effect_size"] == pytest.approx(2.0, rel=1e-6)
    assert set(su["task_ids"]) == {t.task_id for t in tasks}


def test_no_speedup_comparison_when_one_side_has_no_timed_kernel(tasks):
    timed, _ = run_arm(tasks, contract_gen(), arm="cand")
    untimed, _ = run_arm(tasks, contract_gen(), arm="ref", grade="screening")
    assert ab.compare_arms(timed, untimed, n_boot=200)["speedup"] is None


# --------------------------------------------------------------------------- #
# Report artifact
# --------------------------------------------------------------------------- #
def test_report_is_json_serializable_and_ascii(tasks):
    good, _ = run_arm(tasks, contract_gen(), arm="cand")
    bad, _ = run_arm(tasks, prose_gen, arm="ref")
    report = ab.build_report(good, bad, scope=ab.scope_report(tasks),
                             meta={"note": "unit test"}, n_boot=200)
    blob = json.dumps(report)
    assert ab.SCHEMA_REPORT in blob
    text = ab.format_report(report)
    text.encode("ascii")  # the repo's reports are ASCII only
    for expected in ("emitted FULL_KERNEL", "compiled", "correct (SNR gate)",
                     "Verdict", "exact McNemar"):
        assert expected in text
    assert str(len(tasks)) in text


def test_report_names_the_contamination_exclusion(tasks):
    good, _ = run_arm(tasks, contract_gen(), arm="cand")
    bad, _ = run_arm(tasks, prose_gen, arm="ref")
    report = ab.build_report(good, bad, scope=ab.scope_report(tasks), n_boot=200)
    assert "excluded as contaminated" in ab.format_report(report)
    assert report["scope"]["excluded_contaminated"]


def test_report_surfaces_infrastructure_faults_separately(tasks):
    """An OOM/timeout is a statement about the node; it must not read as a model failure.

    The reward gate scores an infra fault as incorrect, so without this line a
    funnel drop caused by a busy device is indistinguishable in the report from a
    model that writes worse kernels.
    """

    class InfraEnv(StubEnv):
        def step(self, source: str):
            obs = super().step(source)
            obs.infra_error = True
            return obs

    rows = [r.to_dict() for r in ab.generate_arm(tasks, contract_gen(), arm="cand",
                                                 max_tokens=8)]
    cand = ab.arm_summary(
        ab.measure_arm(rows, tasks, arm="cand", env_factory=lambda t: InfraEnv(t)),
        rows, arm="cand")
    assert cand["counts"]["infra_error"] == len(tasks)

    ref, _ = run_arm(tasks, contract_gen(), arm="ref")
    text = ab.format_report(ab.build_report(cand, ref, n_boot=200))
    assert "Infrastructure faults" in text
    assert f"cand {len(tasks)}" in text


def test_error_text_is_truncated_from_the_END_not_the_front(tasks):
    """A traceback's informative line is its last one.

    Truncating an ``error_text`` from the front returns harness frames and hides
    the exception, which is exactly what made the first run of this eval unable to
    say why a kernel had failed to build.
    """
    traceback_text = (
        "Traceback (most recent call last):\n"
        + "".join(f'  File "kore/tasks/_genops.py", line {n}, in _run_correctness\n'
                  f"    fn = _load_candidate(task_dir, ref.entry_name)\n"
                  for n in range(1400, 1460))
        + "SyntaxError: '(' was never closed\n"
    )

    class FailingEnv(StubEnv):
        def step(self, source: str):
            return Observation(compiled=False, error_text=traceback_text)

    rows = [r.to_dict() for r in ab.generate_arm(tasks[:1], prose_gen, arm="e",
                                                 max_tokens=8)]
    res = ab.measure_arm(rows, tasks[:1], arm="e",
                         env_factory=lambda t: FailingEnv(t))
    tail = res["observations"][0]["error_tail"]
    assert "SyntaxError: '(' was never closed" in tail
    assert tail.startswith("..."), "a clipped tail must announce that it is clipped"
    assert "Traceback (most recent call last)" not in tail
    assert len(tail) <= 404


def test_a_short_error_is_passed_through_whole():
    obs = Observation(compiled=False, error_text="  SyntaxError: unmatched ')'  ")
    assert ab.observation_summary("t", obs)["error_tail"] == "SyntaxError: unmatched ')'"


def test_no_error_is_none_not_empty_string():
    assert ab.observation_summary("t", Observation(compiled=True))["error_tail"] is None
