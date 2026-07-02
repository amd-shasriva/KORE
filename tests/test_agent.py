"""CPU-only tests for the KORE agentic tool layer.

No GPU, no torch/vllm/transformers. A ``FakeEnv`` returns scripted
``Observation``s keyed by markers in the kernel source, and ``StubTeacher``
scripts the assistant turns. Covers: tool schema validity, ToolExecutor
dispatch, the harness multi-turn loop + keep/revert + best-tracking,
parse_tool_calls robustness, episode_to_chat shape, tool_use_reward
components, and gen_agentic record generation.
"""

from __future__ import annotations

import json

from kore.reward.reward import Observation
from kore.agent.tools import (
    TOOL_SCHEMAS,
    TOOL_NAMES,
    ToolExecutor,
    validate_tool_call,
    tool_use_reward,
)
from kore.agent.format import (
    parse_tool_calls,
    render_tool_result,
    episode_to_chat,
    build_agent_system_prompt,
    render_tool_call_message,
    strip_thinking,
)
from kore.agent.harness import AgentHarness, AgentEpisode
from kore.agent.schema import AgenticTrajectoryRecord
from kore.data.schemas import write_jsonl, read_jsonl
from kore.data.teacher import StubTeacher, TeacherClient
from kore.data.gen_agentic import generate_agentic_trajectories


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeTask:
    task_id = "fake_gemm_bf16"
    operation = "gemm"
    dtype = "bf16"
    gpu_target = "gfx942"


def _obs_bad_compile():
    return Observation(compiled=False, dtype="bf16", error_text="SyntaxError: bad")


def _obs_incorrect():
    return Observation(compiled=True, dtype="bf16", validation_passed=False,
                       snr_by_shape={"primary": 5.0}, snr_db=5.0,
                       error_text="worst SNR 5.0 < 25.0")


def _obs_correct_slow():
    return Observation(compiled=True, dtype="bf16", validation_passed=True,
                       snr_by_shape={"primary": 40.0}, snr_db=40.0,
                       wall_by_shape={"primary": 2.0}, baseline_by_shape={"primary": 2.0},
                       wall_ms=2.0, baseline_ms=2.0, registers=64, occupancy=0.5)


def _obs_correct_fast():
    return Observation(compiled=True, dtype="bf16", validation_passed=True,
                       snr_by_shape={"primary": 41.0}, snr_db=41.0,
                       wall_by_shape={"primary": 1.0}, baseline_by_shape={"primary": 2.0},
                       wall_ms=1.0, baseline_ms=2.0, registers=48, occupancy=0.75)


class FakeEnv:
    """Scripted env: the kernel source carries a marker choosing the Observation.

    Markers: ``__BAD__`` (compile fail), ``__WRONG__`` (incorrect),
    ``__FAST__`` (correct, 2x), anything else -> correct but no speedup.
    """

    def __init__(self):
        self.calls: list[tuple] = []

    def step(self, source, full_validation=True, multi_shape=True):
        self.calls.append((source, full_validation, multi_shape))
        if "__BAD__" in source:
            return _obs_bad_compile()
        if "__WRONG__" in source:
            return _obs_incorrect()
        if "__FAST__" in source:
            return _obs_correct_fast()
        return _obs_correct_slow()


def _mk_call(name, arguments):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


def scripted_teacher(script):
    """StubTeacher that returns ``script[i]`` on the i-th generate() call."""
    state = {"i": 0}

    def fn(_messages):
        i = state["i"]
        state["i"] = i + 1
        return script[i] if i < len(script) else "done, no more changes."

    return StubTeacher(fn=fn)


# --------------------------------------------------------------------------- #
# 1. Schemas
# --------------------------------------------------------------------------- #
def test_tool_schemas_valid_openai_shape():
    assert set(TOOL_NAMES) == {"build", "test", "bench", "pmc", "keep", "revert"}
    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        fn = s["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert isinstance(params["required"], list)
        # every required key is declared in properties
        for req in params["required"]:
            assert req in params["properties"]
    # schemas are JSON-serializable
    assert json.loads(json.dumps(TOOL_SCHEMAS)) == TOOL_SCHEMAS


def test_build_agent_system_prompt_lists_tools():
    p = build_agent_system_prompt()
    for name in TOOL_NAMES:
        assert name in p
    assert "<tool_call>" in p and "<tools>" in p


# --------------------------------------------------------------------------- #
# 2. Validation
# --------------------------------------------------------------------------- #
def test_validate_tool_call_good_and_bad():
    good = validate_tool_call({"name": "test", "arguments": {"kernel_src": "x"}})
    assert good["valid_name"] and good["valid_params"]

    unknown = validate_tool_call({"name": "frobnicate", "arguments": {}})
    assert not unknown["valid_name"]

    missing = validate_tool_call({"name": "build", "arguments": {}})
    assert missing["valid_name"] and not missing["valid_params"]

    badtype = validate_tool_call({"name": "build", "arguments": {"kernel_src": 123}})
    assert not badtype["valid_params"]

    noargs_ok = validate_tool_call({"name": "keep", "arguments": {}})
    assert noargs_ok["valid_name"] and noargs_ok["valid_params"]


# --------------------------------------------------------------------------- #
# 3. Executor dispatch
# --------------------------------------------------------------------------- #
def test_executor_build_test_bench_dispatch():
    ex = ToolExecutor(FakeEnv(), FakeTask())

    r = ex.dispatch({"name": "build", "arguments": {"kernel_src": "__BAD__"}})
    assert r["tool"] == "build" and r["compiled"] is False and r["ok"] is False

    r = ex.dispatch({"name": "test", "arguments": {"kernel_src": "__WRONG__"}})
    assert r["correct"] is False and r["snr_db"] == 5.0

    r = ex.dispatch({"name": "test", "arguments": {"kernel_src": "good"}})
    assert r["correct"] is True and r["tier"] in ("correct_no_bench", "correct_timed")

    r = ex.dispatch({"name": "bench", "arguments": {"kernel_src": "__FAST__"}})
    assert r["correct"] is True and r["speedup"] == 2.0 and r["reward"] > 0.3


def test_executor_pmc_surfaces_counters_or_stub():
    ex = ToolExecutor(FakeEnv(), FakeTask())
    r = ex.dispatch({"name": "pmc", "arguments": {"kernel_src": "good"}})
    assert r["tool"] == "pmc"
    assert r["registers"] == 64 and r["occupancy"] == 0.5
    assert r["available"] is True


def test_executor_unknown_and_malformed_calls():
    ex = ToolExecutor(FakeEnv(), FakeTask())
    r = ex.dispatch({"name": "nope", "arguments": {}})
    assert r["ok"] is False and "unknown" in r["error"]
    r = ex.dispatch({"name": "build", "arguments": {}})  # missing required
    assert r["ok"] is False and "kernel_src" in r["error"]


def test_executor_keep_revert_and_best_tracking():
    ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src="seed")

    # slow-correct candidate, then keep -> committed + best set
    ex.dispatch({"name": "bench", "arguments": {"kernel_src": "slow"}}, turn=0)
    slow_reward = ex.candidate_reward
    kr = ex.dispatch({"name": "keep", "arguments": {}}, turn=0)
    assert kr["kept"] is True and kr["improved"] is True
    assert ex.committed_src == "slow"

    # faster candidate -> best updates to the fast one
    ex.dispatch({"name": "bench", "arguments": {"kernel_src": "__FAST__"}}, turn=1)
    assert ex.best_reward > slow_reward
    assert ex.best_src == "__FAST__"
    assert ex.best_turn == 1

    # a wrong candidate then revert -> flagged as a (correct) regression revert
    ex.dispatch({"name": "test", "arguments": {"kernel_src": "__WRONG__"}}, turn=2)
    rv = ex.dispatch({"name": "revert", "arguments": {}}, turn=2)
    assert rv["reverted"] is True and rv["was_regression"] is True
    assert ex.candidate_src is None
    # best kernel is unaffected by the bad candidate
    assert ex.best_src == "__FAST__"


# --------------------------------------------------------------------------- #
# 4. parse_tool_calls
# --------------------------------------------------------------------------- #
def test_parse_hermes_single_and_multi():
    text = _mk_call("build", {"kernel_src": "abc"})
    calls = parse_tool_calls(text)
    assert len(calls) == 1 and calls[0]["name"] == "build"
    assert calls[0]["arguments"]["kernel_src"] == "abc"

    text2 = _mk_call("test", {"kernel_src": "x"}) + "\n" + _mk_call("keep", {})
    calls2 = parse_tool_calls(text2)
    assert [c["name"] for c in calls2] == ["test", "keep"]


def test_parse_fenced_json_and_openai_and_malformed():
    fenced = "sure:\n```json\n{\"name\": \"bench\", \"arguments\": {\"kernel_src\": \"k\"}}\n```"
    calls = parse_tool_calls(fenced)
    assert calls and calls[0]["name"] == "bench"

    # OpenAI-style tool_calls with stringified arguments
    openai = json.dumps({"tool_calls": [
        {"function": {"name": "test", "arguments": "{\"kernel_src\": \"z\"}"}}
    ]})
    calls = parse_tool_calls(openai)
    assert calls[0]["name"] == "test" and calls[0]["arguments"]["kernel_src"] == "z"

    # malformed JSON inside a hermes block -> flagged, not crashed
    bad = "<tool_call>\n{not json}\n</tool_call>"
    calls = parse_tool_calls(bad)
    assert len(calls) == 1 and calls[0]["malformed"] is True and calls[0]["name"] is None

    assert parse_tool_calls("") == []
    assert parse_tool_calls("no tools here") == []


def test_render_tool_result_role_tool():
    msg = render_tool_result("bench", {"ok": True, "speedup": 2.0})
    assert msg["role"] == "tool" and msg["name"] == "bench"
    assert json.loads(msg["content"])["speedup"] == 2.0


# --------------------------------------------------------------------------- #
# 5. Harness multi-turn loop
# --------------------------------------------------------------------------- #
def _win_script():
    return [
        _mk_call("build", {"kernel_src": "cand __WRONG__"}),   # turn 0: builds ok, wrong numerics
        _mk_call("test", {"kernel_src": "cand __WRONG__"}),    # turn 1: incorrect -> repair
        _mk_call("test", {"kernel_src": "cand fixed"}),         # turn 2: correct
        _mk_call("bench", {"kernel_src": "cand __FAST__"}) + "\n" + _mk_call("keep", {}),  # turn 3
        "All done — no further changes.",                       # turn 4: no tool call -> stop
    ]


def test_harness_runs_multiturn_and_tracks_best():
    env = FakeEnv()
    teacher = scripted_teacher(_win_script())
    ep = AgentHarness(FakeTask(), teacher, env, max_turns=8).run()

    assert isinstance(ep, AgentEpisode)
    assert ep.success is True
    assert ep.best_kernel == "cand __FAST__"
    assert ep.best_reward is not None and ep.best_reward > 0.3
    assert ep.turns_to_best == 3
    # loop stopped early on the no-tool-call turn
    assert ep.turns_used == 5
    # a keep decision was recorded
    assert any(d["action"] == "keep" for d in ep.keep_decisions)
    # message roles include system/user/assistant/tool
    roles = {m["role"] for m in ep.messages}
    assert {"system", "user", "assistant", "tool"} <= roles


def test_harness_no_tool_call_stops_immediately():
    env = FakeEnv()
    teacher = scripted_teacher(["I think we're done here."])
    ep = AgentHarness(FakeTask(), teacher, env, max_turns=5).run()
    assert ep.turns_used == 1
    assert ep.tool_trace == []
    assert ep.success is False and ep.best_kernel is None


def test_harness_respects_max_turns():
    env = FakeEnv()
    # always emits a tool call -> should stop at max_turns
    teacher = scripted_teacher([_mk_call("test", {"kernel_src": "good"})] * 20)
    ep = AgentHarness(FakeTask(), teacher, env, max_turns=3).run()
    assert ep.turns_used == 3
    assert len(ep.tool_trace) == 3


# --------------------------------------------------------------------------- #
# 6. episode_to_chat
# --------------------------------------------------------------------------- #
def test_episode_to_chat_shape_and_thinking_variants():
    env = FakeEnv()
    teacher = scripted_teacher(_win_script())
    ep = AgentHarness(FakeTask(), teacher, env, max_turns=8).run()

    chat = episode_to_chat(ep, thinking=True)
    assert chat[0]["role"] == "system"
    assert chat[1]["role"] == "user"
    assert {m["role"] for m in chat} <= {"system", "user", "assistant", "tool"}
    # tool messages keep their name
    assert any(m["role"] == "tool" and "name" in m for m in chat)

    # thinking stripping
    msg = render_tool_call_message([{"name": "keep", "arguments": {}}],
                                   thinking="secret reasoning")
    assert "<think>" in msg["content"]
    assert strip_thinking(msg["content"]).find("secret reasoning") == -1


# --------------------------------------------------------------------------- #
# 7. tool_use_reward
# --------------------------------------------------------------------------- #
def test_tool_use_reward_components_on_good_episode():
    env = FakeEnv()
    ep = AgentHarness(FakeTask(), scripted_teacher(_win_script()), env, max_turns=8).run()
    comp = tool_use_reward(ep)

    for key in ("tool_name", "param", "format", "outcome", "keep_revert",
                "penalty", "total", "n_calls"):
        assert key in comp
    # all tool names/params were valid and well-formed
    assert comp["tool_name"] == 1.0 and comp["param"] == 1.0 and comp["format"] == 1.0
    # best kernel was correct + fast -> positive outcome and total
    assert comp["outcome"] > 0.3
    assert comp["total"] > comp["outcome"]  # keep-revert + name/param bonuses
    assert comp["penalty"] <= 0.0


def test_tool_use_reward_penalizes_malformed_and_bad_calls():
    # an episode consisting only of a malformed call + an unknown tool
    text = "<tool_call>\n{broken}\n</tool_call>\n" + _mk_call("frobnicate", {})
    env = FakeEnv()
    ep = AgentHarness(FakeTask(), scripted_teacher([text, "done"]), env, max_turns=4).run()
    comp = tool_use_reward(ep)
    assert comp["n_malformed"] >= 1
    assert comp["format"] < 1.0
    assert comp["tool_name"] < 1.0
    assert comp["penalty"] < 0.0
    assert comp["outcome"] == 0.0  # never reached a correct kernel


def test_tool_use_reward_empty_episode():
    ep = AgentEpisode(task_id="t")
    comp = tool_use_reward(ep)
    assert comp["n_calls"] == 0
    assert comp["total"] == 0.0


# --------------------------------------------------------------------------- #
# 8. schema round-trip
# --------------------------------------------------------------------------- #
def test_agentic_record_roundtrip(tmp_path):
    env = FakeEnv()
    ep = AgentHarness(FakeTask(), scripted_teacher(_win_script()), env, max_turns=8).run()
    rec = AgenticTrajectoryRecord(
        task_id=ep.task_id,
        messages=episode_to_chat(ep),
        tool_trace=ep.tool_trace,
        best_kernel=ep.best_kernel,
        best_reward=ep.best_reward,
        turns_to_best=ep.turns_to_best,
        success=ep.success,
        provenance={"category": "success"},
    )
    d = rec.to_dict()
    assert AgenticTrajectoryRecord.from_dict(d) == rec
    assert d["type"] == "agentic"

    path = tmp_path / "agentic.jsonl"
    write_jsonl(path, [rec])
    raw = read_jsonl(path, typed=False)
    assert raw[0]["task_id"] == ep.task_id and raw[0]["type"] == "agentic"


# --------------------------------------------------------------------------- #
# 9. gen_agentic
# --------------------------------------------------------------------------- #
def test_generate_agentic_trajectories_wellformed():
    env = FakeEnv()
    teacher = scripted_teacher(_win_script() * 4)  # enough turns for repeated runs
    recs = generate_agentic_trajectories(FakeTask(), teacher, env, n=2, max_turns=8)
    assert len(recs) == 2
    for rec in recs:
        assert isinstance(rec, AgenticTrajectoryRecord)
        assert rec.task_id == "fake_gemm_bf16"
        assert rec.messages and rec.messages[0]["role"] == "system"
        assert all("role" in m and "content" in m for m in rec.messages)
        assert "category" in rec.provenance
        assert "tool_use_reward" in rec.provenance


def test_generate_agentic_keep_only_useful_filters_attempts():
    env = FakeEnv()
    # a teacher that never reaches correctness (only unknown tool then stop)
    script = [_mk_call("frobnicate", {}), "done"] * 5
    teacher = scripted_teacher(script)
    recs = generate_agentic_trajectories(FakeTask(), teacher, env, n=3,
                                         max_turns=4, keep_only_useful=True)
    assert recs == []  # all attempts filtered out


def test_stubteacher_is_teacherclient():
    assert isinstance(scripted_teacher(["x"]), TeacherClient)
