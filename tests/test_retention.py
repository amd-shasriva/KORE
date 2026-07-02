"""CPU-only tests for the KORE retention eval suite + stage gates.

No GPU, no torch/transformers, no network. Every model is a deterministic STUB
``model_generate(prompt, **kw) -> str`` built from the bundled smoke data, so the
scorers, the suite aggregation, and the gate PASS/FAIL logic are all exercised
end-to-end on CPU. A real HF/vLLM model satisfies the same callable interface and
plugs in unchanged.
"""

from __future__ import annotations

import json

import pytest

from kore.eval import retention as R
from kore.eval import gates as G


# ---------------------------------------------------------------------------
# Deterministic stub models
# ---------------------------------------------------------------------------

_IFEVAL_PERFECT = {
    "if-0": "the kernel runs fast",
    "if-1": "* one\n* two\n* three",
    "if-2": '{"answer": "yes"}',
    "if-3": "Triton is a python-like gpu kernel language. DONE",
    "if-4": "GPU KERNELS ARE FAST",
}


def make_perfect_generate():
    """A model that answers every bundled item correctly (dispatch by prompt)."""
    mmlu = R.load_bench("mmlu")
    humaneval = R.load_bench("humaneval")
    lcb = R.load_bench("livecodebench")
    ifeval = R.load_bench("ifeval")
    bfcl = R.load_bench("bfcl")
    mtbench = R.load_bench("mtbench")

    def generate(prompt: str, **kw) -> str:
        for it in humaneval:
            if it["prompt"] in prompt:
                return it["canonical_solution"]
        for it in lcb:
            if it["prompt"] in prompt:
                return it["reference_solution"]
        for it in bfcl:
            if it["question"] in prompt and "function-calling" in prompt:
                return json.dumps(it["answer"])
        for it in mmlu:
            if it["question"] in prompt:
                return f"The answer is {R._norm_letter(it['answer'])}."
        for it in ifeval:
            if it["prompt"] == prompt:
                return _IFEVAL_PERFECT[it["id"]]
        for it in mtbench:
            if it["question"] == prompt:
                return it.get("reference") or "A concise correct answer about the topic."
        return ""

    return generate


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

def test_mmlu_perfect_scores_full_accuracy():
    gen = make_perfect_generate()
    res = R.MMLUScorer().score(gen)
    assert res["accuracy"] == 1.0
    assert res["score"] == 1.0
    assert res["correct"] == res["n"] and res["n"] >= 10


def test_mmlu_wrong_scores_zero():
    def bad(prompt, **kw):
        # Always answer a letter that is never the gold one for these items.
        return "Z then H"

    res = R.MMLUScorer().score(bad)
    assert res["accuracy"] == 0.0


def test_mmlu_parse_answer_variants():
    assert R.parse_mmlu_answer("The answer is C.", 4) == "C"
    assert R.parse_mmlu_answer("B", 4) == "B"
    assert R.parse_mmlu_answer("I think it is (D)", 4) == "D"
    # last valid standalone letter wins when no explicit 'answer:'
    assert R.parse_mmlu_answer("Between A and D, pick D", 4) == "D"
    # out-of-range letters ignored
    assert R.parse_mmlu_answer("E F G", 4) == "?"


# ---------------------------------------------------------------------------
# HumanEval (sandboxed exec)
# ---------------------------------------------------------------------------

def test_humaneval_pass1_perfect():
    gen = make_perfect_generate()
    res = R.HumanEvalScorer().score(gen)
    assert res["pass@1"] == 1.0
    assert res["passed"] == res["n"] and res["n"] >= 2


def test_humaneval_wrong_body_fails():
    def bad(prompt, **kw):
        return "    return None\n"

    res = R.HumanEvalScorer().score(bad)
    assert res["pass@1"] == 0.0
    assert all(not pi["passed"] for pi in res["per_item"])


# ---------------------------------------------------------------------------
# LiveCodeBench-style (timed exec)
# ---------------------------------------------------------------------------

def test_livecodebench_timed_correctness():
    gen = make_perfect_generate()
    res = R.LiveCodeBenchScorer().score(gen)
    assert res["pass_rate"] == 1.0
    assert res["mean_elapsed_s"] >= 0.0
    assert all(not pi["timed_out"] for pi in res["per_item"])


def test_livecodebench_timeout_counts_as_failure():
    items = [
        {
            "task_id": "slow",
            "entry_point": "f",
            "prompt": "def f():",
            "test": "assert f() == 1\n",
            "time_limit_s": 0.4,
        }
    ]

    def slow(prompt, **kw):
        return "import time\ndef f():\n    time.sleep(3)\n    return 1\n"

    res = R.LiveCodeBenchScorer(items=items).score(slow)
    assert res["pass_rate"] == 0.0
    assert res["per_item"][0]["timed_out"] is True


# ---------------------------------------------------------------------------
# IFEval-style
# ---------------------------------------------------------------------------

def test_ifeval_perfect_prompt_strict():
    gen = make_perfect_generate()
    res = R.IFEvalScorer().score(gen)
    assert res["prompt_strict_accuracy"] == 1.0
    assert res["instruction_accuracy"] == 1.0


def test_ifeval_empty_fails_all():
    def empty(prompt, **kw):
        return ""

    res = R.IFEvalScorer().score(empty)
    assert res["prompt_strict_accuracy"] == 0.0


def test_ifeval_instruction_checkers():
    assert R._check_instruction({"type": "all_lowercase"}, "abc def") is True
    assert R._check_instruction({"type": "all_lowercase"}, "Abc") is False
    assert R._check_instruction({"type": "keyword_include", "keyword": "gpu"}, "the GPU") is True
    assert R._check_instruction({"type": "no_commas"}, "a, b") is False
    assert R._check_instruction({"type": "ends_with", "suffix": "DONE"}, "all DONE ") is True
    assert R._check_instruction({"type": "num_bullets", "n": 2}, "* a\n* b") is True
    assert R._check_instruction({"type": "json_format"}, '{"x": 1}') is True
    assert R._check_instruction({"type": "word_count_max", "n": 3}, "one two three") is True
    with pytest.raises(ValueError):
        R._check_instruction({"type": "nonsense"}, "x")


# ---------------------------------------------------------------------------
# BFCL-style
# ---------------------------------------------------------------------------

def test_bfcl_exact_call_correct():
    gen = make_perfect_generate()
    res = R.BFCLScorer().score(gen)
    assert res["accuracy"] == 1.0
    assert res["name_accuracy"] == 1.0


def test_bfcl_wrong_args_fail_full_but_name_ok():
    items = [
        {
            "id": "b0",
            "question": "Add the numbers.",
            "tools": [{"name": "add", "parameters": ["a", "b"]}],
            "answer": {"name": "add", "arguments": {"a": 2, "b": 3}},
        }
    ]

    def wrong_args(prompt, **kw):
        return '{"name": "add", "arguments": {"a": 9, "b": 9}}'

    res = R.BFCLScorer(items=items).score(wrong_args)
    assert res["name_accuracy"] == 1.0
    assert res["accuracy"] == 0.0


def test_bfcl_json_extraction_from_prose():
    items = [
        {
            "id": "b0",
            "question": "Add the numbers.",
            "tools": [{"name": "add", "parameters": ["a", "b"]}],
            "answer": {"name": "add", "arguments": {"a": 2, "b": 3}},
        }
    ]

    def prose(prompt, **kw):
        return 'Sure! Here is the call: {"name": "add", "arguments": {"a": 2, "b": 3}} done.'

    res = R.BFCLScorer(items=items).score(prose)
    assert res["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# MT-Bench (judge hook)
# ---------------------------------------------------------------------------

def test_mtbench_injected_judge_used():
    calls = {"n": 0}

    def judge(question, answer, reference=None):
        calls["n"] += 1
        return 8.0

    gen = make_perfect_generate()
    res = R.MTBenchScorer(judge=judge).score(gen)
    assert calls["n"] == res["n"]
    assert res["mean_judge_score"] == 8.0
    assert abs(res["score"] - 0.8) < 1e-9


def test_mtbench_default_stub_judge_deterministic():
    # Reference-guided overlap: identical answer -> perfect score.
    s1 = R.default_stub_judge("q", "the kernel is fast", reference="the kernel is fast")
    s2 = R.default_stub_judge("q", "the kernel is fast", reference="the kernel is fast")
    assert s1 == s2 == 10.0
    assert R.default_stub_judge("q", "", reference="anything") == 1.0


# ---------------------------------------------------------------------------
# Suite aggregation
# ---------------------------------------------------------------------------

def test_run_retention_suite_aggregates_all_benches():
    gen = make_perfect_generate()

    def judge(question, answer, reference=None):
        return 10.0

    out = R.run_retention_suite(gen, judge=judge)
    assert set(out["benches"]) == set(R.DEFAULT_BENCHES)
    assert set(out["scores"].keys()) == set(R.DEFAULT_BENCHES)
    # Perfect model + perfect judge => every bench maxed => aggregate == 1.0.
    for name, sc in out["scores"].items():
        assert sc == pytest.approx(1.0), name
    assert out["aggregate"] == pytest.approx(1.0)
    assert "per_bench" in out and "mmlu" in out["per_bench"]


def test_run_retention_suite_subset_and_aggregate_mean():
    gen = make_perfect_generate()
    out = R.run_retention_suite(gen, benches=["mmlu", "humaneval"])
    assert out["benches"] == ["mmlu", "humaneval"]
    expected = sum(out["scores"].values()) / 2.0
    assert out["aggregate"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# StageGate
# ---------------------------------------------------------------------------

_KERNEL = ["fast_p@1.0"]
_GENERAL = ["mmlu", "humaneval", "ifeval", "bfcl", "mtbench"]


def _base():
    return {"fast_p@1.0": 0.40, "mmlu": 0.70, "humaneval": 0.60, "ifeval": 0.55, "bfcl": 0.65, "mtbench": 0.80}


def test_stagegate_pass_improve_and_no_regression():
    before = _base()
    after = dict(before, **{"fast_p@1.0": 0.50, "mmlu": 0.71})  # kernel up, general steady/up
    res = G.StageGate().evaluate(before, after, kernel_keys=_KERNEL, general_keys=_GENERAL)
    assert res.passed is True
    assert "fast_p@1.0" in res.improvements
    assert res.regressions == []


def test_stagegate_fail_general_regression_over_epsilon():
    before = _base()
    # kernel improves but MMLU drops well beyond epsilon -> FAIL
    after = dict(before, **{"fast_p@1.0": 0.50, "mmlu": 0.60})
    res = G.StageGate(epsilon=0.005).evaluate(before, after, kernel_keys=_KERNEL, general_keys=_GENERAL)
    assert res.passed is False
    assert "mmlu" in res.regressions


def test_stagegate_small_general_dip_within_epsilon_passes():
    before = _base()
    # MMLU dips by 0.004 < epsilon(0.005); kernel improves -> PASS
    after = dict(before, **{"fast_p@1.0": 0.50, "mmlu": 0.696})
    res = G.StageGate(epsilon=0.005).evaluate(before, after, kernel_keys=_KERNEL, general_keys=_GENERAL)
    assert res.passed is True


def test_stagegate_fail_when_kernel_does_not_improve():
    before = _base()
    # general all fine, but kernel is flat (no strict improvement) -> FAIL
    after = dict(before, **{"fast_p@1.0": 0.40, "mmlu": 0.75})
    res = G.StageGate().evaluate(before, after, kernel_keys=_KERNEL, general_keys=_GENERAL)
    assert res.passed is False
    assert "fast_p@1.0" in res.regressions


def test_stagegate_missing_key_fails():
    before = _base()
    after = dict(before, **{"fast_p@1.0": 0.50})
    after.pop("mmlu")
    res = G.StageGate().evaluate(before, after, kernel_keys=_KERNEL, general_keys=_GENERAL)
    assert res.passed is False
    assert "mmlu" in res.regressions


def test_stagegate_require_all_kernel_flag():
    before = {"k1": 0.4, "k2": 0.5, "g": 0.7}
    after = {"k1": 0.5, "k2": 0.5, "g": 0.7}  # only k1 improves, k2 flat
    strict = G.StageGate(require_all_kernel=True).evaluate(
        before, after, kernel_keys=["k1", "k2"], general_keys=["g"]
    )
    lenient = G.StageGate(require_all_kernel=False).evaluate(
        before, after, kernel_keys=["k1", "k2"], general_keys=["g"]
    )
    assert strict.passed is False
    assert lenient.passed is True


# ---------------------------------------------------------------------------
# retention_gate + assert_gate_or_raise + report
# ---------------------------------------------------------------------------

def test_retention_gate_passes_when_no_regression():
    base = {"mmlu": 0.70, "humaneval": 0.60}
    cand = {"mmlu": 0.72, "humaneval": 0.60}
    res = G.retention_gate(base, cand, epsilon=0.005)
    assert res.passed is True
    assert "mmlu" in res.improvements


def test_retention_gate_fails_on_regression():
    base = {"mmlu": 0.70, "humaneval": 0.60}
    cand = {"mmlu": 0.70, "humaneval": 0.50}
    res = G.retention_gate(base, cand, epsilon=0.005)
    assert res.passed is False
    assert "humaneval" in res.regressions


def test_assert_gate_or_raise_returns_on_pass_and_raises_on_fail():
    before = _base()
    good = dict(before, **{"fast_p@1.0": 0.50})
    result = G.assert_gate_or_raise(before, good, kernel_keys=_KERNEL, general_keys=_GENERAL)
    assert result.passed is True

    bad = dict(before, **{"fast_p@1.0": 0.50, "bfcl": 0.10})
    with pytest.raises(G.GateError) as ei:
        G.assert_gate_or_raise(before, bad, kernel_keys=_KERNEL, general_keys=_GENERAL)
    assert "FAIL" in str(ei.value)
    assert ei.value.result.passed is False


def test_format_gate_report_is_readable():
    before = _base()
    after = dict(before, **{"fast_p@1.0": 0.50, "mmlu": 0.60})
    res = G.StageGate().evaluate(after=after, before=before, kernel_keys=_KERNEL, general_keys=_GENERAL)
    report = G.format_gate_report(res)
    assert isinstance(report, str) and "stage gate" in report.lower()
    assert "mmlu" in report and "fast_p@1.0" in report


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def test_extract_first_json_and_code_fence():
    assert R._extract_first_json('noise {"a": 1, "b": {"c": 2}} tail') == {"a": 1, "b": {"c": 2}}
    assert R._extract_first_json("no json here") is None
    fenced = "```python\nprint('hi')\n```"
    assert "print('hi')" in R._strip_code_fences(fenced)
