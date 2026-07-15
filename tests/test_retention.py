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


def test_strip_think_removes_reasoning_trace():
    from kore.policy.serve import _strip_think
    assert _strip_think("<think>weigh the options</think>B") == "B"
    assert _strip_think("<think>a</think>\nThe answer is C.") == "The answer is C."
    # budget-truncated (unclosed) trace -> nothing survives (no stray-letter pollution)
    assert _strip_think("<think>consider A vs B vs C before deciding") == ""
    assert _strip_think("D") == "D"  # no trace -> untouched


def test_mmlu_gate_un_vacuumed_when_thinking_stripped():
    """A hybrid-reasoning model (Qwen3) emits <think>...</think> before the answer.
    load_generate strips it (and disables thinking), so the MMLU letter is real and
    the retention gate can actually detect regression. Without stripping, a
    budget-truncated trace leaves no answer and the gate is vacuous (R2 soup-eval C1).
    """
    from kore.policy.serve import _strip_think

    mmlu = R.load_bench("mmlu")

    def real_answer_with_thinking(prompt, **kw):
        for it in mmlu:
            if it["question"] in prompt:
                return f"<think>weigh each option</think>{R._norm_letter(it['answer'])}"
        return "<think>hmm</think>?"

    stripped = lambda p, **kw: _strip_think(real_answer_with_thinking(p, **kw))  # noqa: E731
    assert R.MMLUScorer().score(stripped)["accuracy"] == 1.0  # gate now measures truth


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


# ---------------------------------------------------------------------------
# FULL bench loading (HF datasets) — monkeypatched, CPU/offline
# ---------------------------------------------------------------------------
#
# Every test here monkeypatches ``datasets.load_dataset`` so NO network is
# touched: either it returns fake rows in the upstream schema (exercising the
# real mappers -> source == "full-hf"), or it raises (exercising the smoke
# fallback -> source == "smoke").

def _fake_load_dataset(rows_by_path):
    """Build a ``datasets.load_dataset`` stand-in dispatching on the dataset path."""

    def fake(path, *args, **kwargs):
        if path in rows_by_path:
            return list(rows_by_path[path])
        raise FileNotFoundError(f"no fake rows for {path!r}")

    return fake


def _patch_datasets(monkeypatch, rows_by_path):
    import datasets  # installed; we only replace the loader entry point

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset(rows_by_path))


def test_full_falls_back_to_smoke_when_offline(monkeypatch):
    import datasets

    def boom(*a, **k):
        raise RuntimeError("offline: no HF access")

    monkeypatch.setattr(datasets, "load_dataset", boom)

    gen = make_perfect_generate()
    out = R.run_retention_suite(gen, full=True, judge=lambda q, a, r=None: 10.0)
    # Every bench must fall back to the bundled smoke set...
    for b in R.DEFAULT_BENCHES:
        assert out["sources"][b] == "smoke", b
    # ...and the smoke suite still scores perfectly with the perfect stub model.
    assert out["full"] is True
    assert out["aggregate"] == pytest.approx(1.0)


def test_full_flag_and_env_both_trigger_full_loading(monkeypatch):
    rows = [{"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1, "subject": "math"}]
    _patch_datasets(monkeypatch, {"cais/mmlu": rows})

    # (a) explicit full=True
    res_flag = R.MMLUScorer(full=True).score(lambda p, **k: "answer: B")
    assert res_flag["source"] == "full-hf"
    assert res_flag["n"] == 1 and res_flag["accuracy"] == 1.0

    # (b) env var KORE_EVAL_FULL, no flag
    monkeypatch.setenv("KORE_EVAL_FULL", "1")
    res_env = R.MMLUScorer().score(lambda p, **k: "answer: B")
    assert res_env["source"] == "full-hf" and res_env["n"] == 1


def test_mmlu_full_metric_on_fake_split(monkeypatch):
    rows = [
        {"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1, "subject": "math"},
        {"question": "Capital of France?", "choices": ["Paris", "Rome", "Berlin", "Madrid"], "answer": 0},
    ]
    _patch_datasets(monkeypatch, {"cais/mmlu": rows})

    def gen(prompt, **kw):
        for r in rows:
            if r["question"] in prompt:
                return f"answer: {R._norm_letter(r['answer'])}"
        return "?"

    res = R.MMLUScorer(full=True).score(gen)
    assert res["source"] == "full-hf"
    assert res["n"] == 2 and res["accuracy"] == 1.0


def test_humaneval_full_metric_on_fake_split(monkeypatch):
    rows = [
        {
            "task_id": "HE/f0",
            "entry_point": "add",
            "prompt": 'def add(a, b):\n    """add"""\n',
            "canonical_solution": "    return a + b\n",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
        }
    ]
    _patch_datasets(monkeypatch, {"openai_humaneval": rows})

    res_ok = R.HumanEvalScorer(full=True).score(lambda p, **k: "    return a + b\n")
    assert res_ok["source"] == "full-hf"
    assert res_ok["pass@1"] == 1.0

    res_bad = R.HumanEvalScorer(full=True).score(lambda p, **k: "    return a - b\n")
    assert res_bad["pass@1"] == 0.0


def test_livecodebench_full_metric_on_fake_native_split(monkeypatch):
    rows = [
        {
            "question_id": "lcb0",
            "question_content": "Return a + b.",
            "starter_code": "def add(a, b):\n",
            "metadata": '{"func_name": "add"}',
            "public_test_cases": '[{"input": "1\\n2", "output": "3", "testtype": "functional"}]',
        }
    ]
    _patch_datasets(monkeypatch, {"livecodebench/code_generation_lite": rows})

    res = R.LiveCodeBenchScorer(full=True).score(lambda p, **k: "def add(a, b):\n    return a + b\n")
    assert res["source"] == "full-hf"
    assert res["pass_rate"] == 1.0


def test_ifeval_full_metric_on_fake_split(monkeypatch):
    rows = [
        {
            "key": 1,
            "prompt": "write it lowercase and include the word kernel",
            "instruction_id_list": ["change_case:english_lowercase", "keywords:existence"],
            "kwargs": [{}, {"keywords": ["kernel"]}],
        }
    ]
    _patch_datasets(monkeypatch, {"google/IFEval": rows})

    res = R.IFEvalScorer(full=True).score(lambda p, **k: "the kernel is fast")
    assert res["source"] == "full-hf"
    assert res["prompt_strict_accuracy"] == 1.0
    assert res["instruction_accuracy"] == 1.0


def test_bfcl_full_metric_on_fake_nested_split(monkeypatch):
    rows = [
        {
            "id": "bf0",
            "question": [[{"role": "user", "content": "Add 2 and 3"}]],
            "function": [{"name": "add", "parameters": ["a", "b"]}],
            "answer": {"name": "add", "arguments": {"a": 2, "b": 3}},
        }
    ]
    _patch_datasets(monkeypatch, {"gorilla-llm/berkeley-function-calling-leaderboard": rows})

    res = R.BFCLScorer(full=True).score(
        lambda p, **k: '{"name": "add", "arguments": {"a": 2, "b": 3}}'
    )
    assert res["source"] == "full-hf"
    assert res["accuracy"] == 1.0 and res["name_accuracy"] == 1.0


def test_mtbench_full_metric_on_fake_split(monkeypatch):
    rows = [{"question_id": 1, "question": "Explain GPU kernels", "reference": "a kernel runs on the gpu"}]
    _patch_datasets(monkeypatch, {"lmsys/mt_bench": rows})

    res = R.MTBenchScorer(judge=lambda q, a, r=None: 9.0, full=True).score(lambda p, **k: "an answer")
    assert res["source"] == "full-hf"
    assert res["n"] == 1 and res["mean_judge_score"] == 9.0


def test_run_retention_suite_reports_full_sources(monkeypatch):
    rows = [{"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}]
    _patch_datasets(monkeypatch, {"cais/mmlu": rows})  # only MMLU has a fake full split

    gen = make_perfect_generate()
    out = R.run_retention_suite(gen, full=True, judge=lambda q, a, r=None: 10.0)
    # MMLU loaded the fake full split; the rest fell back to smoke.
    assert out["sources"]["mmlu"] == "full-hf"
    for b in [x for x in R.DEFAULT_BENCHES if x != "mmlu"]:
        assert out["sources"][b] == "smoke", b


def test_explicit_items_take_precedence_over_full(monkeypatch):
    # Even with full requested, explicit items win and are marked as such.
    rows = [{"question": "ignored", "choices": ["a", "b", "c", "d"], "answer": 0}]
    _patch_datasets(monkeypatch, {"cais/mmlu": rows})
    items = R.load_bench("mmlu")
    sc = R.MMLUScorer(items=items, full=True)
    sc.score(lambda p, **k: "A")
    assert sc.source == "explicit"


# ---------------------------------------------------------------------------
# E2E serving eval (stub backend, CPU/offline)
# ---------------------------------------------------------------------------

from kore.eval import e2e_sglang_vllm as E  # noqa: E402


def _stub_llm(prompt, max_tokens=128, temperature=0.0, **kw):
    p = prompt.lower()
    if "2 + 2" in p or "2+2" in p:
        return "4"
    if "capital of france" in p:
        return "Paris"
    if "opposite of hot" in p:
        return "cold"
    if "days are in a week" in p:
        return "7"
    return "some tokens here " * 4  # a few whitespace tokens for throughput


def test_e2e_throughput_runs_with_stub_model_generate():
    w = E.Workload(num_requests=4, max_new_tokens=16)
    res = E.e2e_throughput("m", "kernel", w, engine="vllm", model_generate=_stub_llm)
    assert res.kind == "throughput" and res.unit == "tokens/s"
    assert res.candidate_value > 0
    assert res.passed is True


def test_e2e_throughput_gate_threshold_vs_baseline():
    w = E.Workload(num_requests=4, max_new_tokens=16)
    # A giant baseline the stub cannot beat -> not passed.
    res = E.e2e_throughput("m", "kernel", w, model_generate=_stub_llm, baseline_tokens_per_s=1e12)
    assert res.passed is False


def test_e2e_accuracy_runs_with_stub_model_generate():
    res = E.e2e_accuracy("m", "kernel", engine="sglang", model_generate=_stub_llm)
    assert res.kind == "accuracy" and res.unit == "accuracy"
    assert res.candidate_value == pytest.approx(1.0)
    assert res.passed is True


def test_e2e_accuracy_regression_fails_against_baseline():
    def wrong(prompt, **kw):
        return "totally wrong"

    res = E.e2e_accuracy("m", "kernel", model_generate=wrong, baseline_accuracy=0.9)
    assert res.candidate_value == 0.0
    assert res.passed is False


def test_e2e_raises_only_when_no_backend():
    with pytest.raises(E.E2ENotProvisioned):
        E.e2e_throughput("m", "kernel", E.Workload())
    with pytest.raises(E.E2ENotProvisioned):
        E.e2e_accuracy("m", "kernel")


def test_e2e_unknown_engine_rejected():
    with pytest.raises(ValueError):
        E.e2e_throughput("m", "kernel", E.Workload(), engine="tensorrt", model_generate=_stub_llm)


def test_e2e_gate_is_pure_and_combines_measurements():
    tput = E.E2EResult("vllm", "throughput", 100.0, 120.0, "tokens/s", passed=True)
    acc = E.E2EResult("vllm", "accuracy", 0.80, 0.805, "accuracy", passed=True)
    g = E.e2e_gate(tput, acc)
    assert g["accept"] is True
    assert g["throughput_improved"] is True and g["accuracy_held"] is True

    slower = E.E2EResult("vllm", "throughput", 100.0, 90.0, "tokens/s", passed=False)
    g2 = E.e2e_gate(slower, acc)
    assert g2["accept"] is False and g2["throughput_improved"] is False

    acc_regressed = E.E2EResult("vllm", "accuracy", 0.80, 0.70, "accuracy", passed=False)
    g3 = E.e2e_gate(tput, acc_regressed)
    assert g3["accept"] is False and g3["accuracy_held"] is False
