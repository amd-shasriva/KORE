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
    TRANSFORM_TOOL_SCHEMAS,
    ToolExecutor,
    agent_tool_schemas,
    validate_tool_call,
    tool_use_reward,
)
from kore.agent.tools import W_REFLECT, W_OUTCOME
from kore.agent.format import (
    parse_tool_calls,
    parse_reflection,
    render_reflection,
    render_tool_result,
    episode_to_chat,
    build_agent_system_prompt,
    render_tool_call_message,
    strip_thinking,
)
from kore.agent.harness import (
    AgentHarness,
    AgentEpisode,
    WinsKB,
    build_agent_user_prompt,
    PHASE_CORRECTNESS,
    PHASE_OPTIMIZE,
)
from kore.agent.schema import AgenticTrajectoryRecord
from kore.data.schemas import write_jsonl, read_jsonl, WinRecord
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
                       wall_ms=2.0, baseline_ms=2.0)


def _obs_correct_fast():
    return Observation(compiled=True, dtype="bf16", validation_passed=True,
                       snr_by_shape={"primary": 41.0}, snr_db=41.0,
                       wall_by_shape={"primary": 1.0}, baseline_by_shape={"primary": 2.0},
                       wall_ms=1.0, baseline_ms=2.0)


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


def test_bench_surfaces_measured_speedup_delta():
    """Per-turn measured-latency feedback: the ``bench`` tool reports the running
    best speedup, the signed delta vs that best, and whether the frontier moved -
    the signal the policy reads to know if THIS turn's change actually helped."""
    ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src="seed")

    # First bench: a correct 1x kernel establishes the frontier (no prior best).
    r0 = ex.dispatch({"name": "bench", "arguments": {"kernel_src": "slow"}}, turn=0)
    assert r0["speedup"] == 1.0
    assert r0["best_speedup_so_far"] == 1.0
    assert r0["delta_vs_best"] is None        # no prior frontier to diff against
    assert r0["improved_frontier"] is True    # first correct measurement sets it
    assert ex.candidate_speedup == 1.0 and ex.best_speedup == 1.0

    # Faster (2x) kernel -> positive delta, frontier advances.
    r1 = ex.dispatch({"name": "bench", "arguments": {"kernel_src": "__FAST__"}}, turn=1)
    assert r1["speedup"] == 2.0
    assert r1["best_speedup_so_far"] == 2.0
    assert r1["delta_vs_best"] == 1.0         # 2.0 - 1.0
    assert r1["improved_frontier"] is True
    assert ex.best_speedup == 2.0

    # Slower (1x) kernel -> negative delta, frontier UNCHANGED (never regresses).
    r2 = ex.dispatch({"name": "bench", "arguments": {"kernel_src": "slow2"}}, turn=2)
    assert r2["speedup"] == 1.0
    assert r2["best_speedup_so_far"] == 2.0
    assert r2["delta_vs_best"] == -1.0        # 1.0 - 2.0
    assert r2["improved_frontier"] is False
    assert ex.best_speedup == 2.0

    # An INCORRECT candidate carries no measured speedup (cannot fake a delta).
    r3 = ex.dispatch({"name": "bench", "arguments": {"kernel_src": "__WRONG__"}}, turn=3)
    assert r3["correct"] is False and r3["speedup"] is None
    assert r3["delta_vs_best"] is None and r3["improved_frontier"] is False
    assert ex.candidate_speedup is None and ex.best_speedup == 2.0  # frontier intact


def test_build_and_test_never_fabricate_a_measured_speedup():
    """Only a BENCHED, correct candidate has a trustworthy speedup; compile-only
    (build) and correctness-only (test) turns must leave it None so a turn that
    never timed the kernel can't inject a phantom latency signal into the trace."""
    ex = ToolExecutor(FakeEnv(), FakeTask())
    ex.dispatch({"name": "build", "arguments": {"kernel_src": "good"}}, turn=0)
    assert ex.candidate_speedup is None
    ex.dispatch({"name": "test", "arguments": {"kernel_src": "good"}}, turn=0)  # correct, unbenched
    assert ex.candidate_correct is True and ex.candidate_speedup is None
    assert ex.best_speedup is None  # the frontier only advances on a real bench


def test_executor_pmc_surfaces_efficiency_or_stub():
    ex = ToolExecutor(FakeEnv(), FakeTask())
    r = ex.dispatch({"name": "pmc", "arguments": {"kernel_src": "good"}})
    assert r["tool"] == "pmc"
    # FakeEnv doesn't populate profile_efficiency (profiling off) -> honest stub.
    assert r["available"] is False
    assert r["profile_efficiency"] is None
    assert "profiling disabled" in r["diagnosis"]


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
# 3b. Verified-transform action space (paradigm-v2)
# --------------------------------------------------------------------------- #
# A minimal Triton GEMM with tunable launch kwargs (num_warps/num_stages) + a
# tuple BLOCK defn, so exact knob transforms (e.g. set_num_warps) are admissible.
_XFORM_GEMM = '''\
import triton
import triton.language as tl


@triton.jit
def _k(a_ptr, b_ptr, c_ptr, M, N, K,
       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc += 1.0


def gemm(a, b, c):
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 128, 64, 8
    _k[(1,)](a, b, c, 1, 1, 1, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
             GROUP_M=GROUP_M, num_warps=4, num_stages=2)
'''


def test_agent_tool_schemas_gating_and_validation():
    base = agent_tool_schemas(transforms=False)
    full = agent_tool_schemas(transforms=True)
    base_names = {s["function"]["name"] for s in base}
    full_names = {s["function"]["name"] for s in full}
    assert base_names == {"build", "test", "bench", "pmc", "keep", "revert"}
    assert full_names == base_names | {"list_transforms", "apply_transform"}
    assert len(TRANSFORM_TOOL_SCHEMAS) == 2
    # validate/dispatch recognize the transform tools even though they are opt-in.
    ok = validate_tool_call({"name": "apply_transform",
                             "arguments": {"name": "set_num_warps", "params": {"value": 8}}})
    assert ok["valid_name"] and ok["valid_params"]
    bad = validate_tool_call({"name": "apply_transform", "arguments": {}})
    assert bad["valid_name"] and not bad["valid_params"]  # 'name' is required


def test_transform_tool_list_and_apply():
    ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src=_XFORM_GEMM)
    listed = ex.dispatch({"name": "list_transforms", "arguments": {}}, turn=0)
    assert listed["ok"] is True
    assert listed["n_admissible"] >= 1
    names = {a["name"] for a in listed["actions"]}
    assert "set_num_warps" in names  # an exact knob move on num_warps=4
    # apply an EXACT move -> rewritten source, budget unspent (exact costs 0 eps).
    applied = ex.dispatch({"name": "apply_transform",
                           "arguments": {"name": "set_num_warps", "params": {"value": 8}}}, turn=0)
    assert applied["ok"] is True
    assert "num_warps=8" in applied["kernel_src"]
    assert applied["kernel_src"] != _XFORM_GEMM
    assert not applied["rejected"]


def test_transform_tool_is_failsafe():
    ex = ToolExecutor(FakeEnv(), FakeTask(), seed_src=_XFORM_GEMM)
    # unknown transform -> rejected, source UNCHANGED, never raises.
    bad = ex.dispatch({"name": "apply_transform",
                       "arguments": {"name": "no_such_transform", "params": {}}}, turn=0)
    assert bad["ok"] is False
    assert bad["kernel_src"] == _XFORM_GEMM
    assert bad["rejected"]
    # no working source (empty seed) -> graceful error, no crash.
    ex2 = ToolExecutor(FakeEnv(), FakeTask(), seed_src=None)
    empty = ex2.dispatch({"name": "list_transforms", "arguments": {}}, turn=0)
    assert empty["ok"] is False and "no kernel source" in empty["error"]


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
        "All done - no further changes.",                       # turn 4: no tool call -> stop
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


# --------------------------------------------------------------------------- #
# 10. Per-turn reward trace (RL CONTRACT)
# --------------------------------------------------------------------------- #
def test_agent_episode_exposes_per_turn_reward_trace():
    env = FakeEnv()
    ep = AgentHarness(FakeTask(), scripted_teacher(_win_script()), env,
                      max_turns=8, use_kb=False).run()

    # Parallel arrays, one entry per turn (RL contract for GRPO Kevin credit).
    assert len(ep.turn_rewards) == ep.turns_used
    assert len(ep.turn_correct) == ep.turns_used
    assert all(isinstance(r, float) for r in ep.turn_rewards)
    assert all(isinstance(c, bool) for c in ep.turn_correct)

    # Correctness only after the fixed kernel (turns 0/1 are wrong builds/tests).
    assert ep.turn_correct[0] is False and ep.turn_correct[1] is False
    assert ep.turn_correct[2] is True and ep.turn_correct[-1] is True
    # The best reward is reflected in the trace and is the max over correct turns.
    assert ep.best_reward is not None
    assert max(ep.turn_rewards) == ep.best_reward
    # Trajectory-level summary still present.
    assert ep.turns_to_best == 3 and ep.success is True


def test_agent_episode_exposes_per_turn_speedup_and_code_trace():
    """The harness records per-turn MEASURED speedup + candidate source in lockstep
    with the reward/correct trace, so the GRPO agentic path can feed co-evolution
    distillation + the open-ended controller the same signal as the serial path."""
    env = FakeEnv()
    ep = AgentHarness(FakeTask(), scripted_teacher(_win_script()), env,
                      max_turns=8, use_kb=False).run()

    # Parallel arrays, one entry per turn, index-aligned with turn_rewards.
    assert len(ep.turn_speedups) == ep.turns_used
    assert len(ep.turn_codes) == ep.turns_used
    assert len(ep.turn_speedups) == len(ep.turn_rewards) == len(ep.turn_correct)

    # _win_script benches __FAST__ (2x) only on turn 3; turns 0-2 are build/test
    # (never timed) -> only turn 3 carries a measured speedup + its source.
    assert ep.turn_speedups[0] is None and ep.turn_speedups[1] is None
    assert ep.turn_speedups[2] is None            # correct on turn 2, but NOT benched
    assert ep.turn_speedups[3] == 2.0
    assert ep.turn_codes[3] == "cand __FAST__"
    # correctness-only turns still record their candidate source (for correct_kernels)
    assert ep.turn_codes[2] == "cand fixed"

    # the new fields survive serialization (record round-trip / datagen reuse)
    d = ep.to_dict()
    assert d["turn_speedups"] == ep.turn_speedups
    assert d["turn_codes"] == ep.turn_codes


# --------------------------------------------------------------------------- #
# 11. Structured reflection: parsing + bounded reward
# --------------------------------------------------------------------------- #
def test_parse_reflection_json_and_lines_and_render_roundtrip():
    block = ('<reflect>\n{"root_cause": "snr too low", "evidence": '
             '"worst SNR 5.0", "planned_fix": "rewrite reduction"}\n</reflect>')
    r = parse_reflection("thinking...\n" + block)
    assert r["root_cause"] == "snr too low"
    assert r["evidence"] == "worst SNR 5.0"
    assert r["planned_fix"] == "rewrite reduction"

    # key: value fallback when the block isn't valid JSON
    r2 = parse_reflection("<reflect>\nroot_cause: bad tile\nplanned_fix: retune\n</reflect>")
    assert r2["root_cause"] == "bad tile" and r2["planned_fix"] == "retune"

    # no block -> None
    assert parse_reflection("no reflection here") is None
    assert parse_reflection("") is None

    # render -> parse round-trip
    payload = {"root_cause": "a", "evidence": "b", "planned_fix": "c"}
    assert parse_reflection(render_reflection(payload)) == payload


def test_harness_captures_reflections_and_bounded_reward():
    env = FakeEnv()
    reflect = render_reflection({
        "root_cause": "kernel failed to compile",
        "evidence": "SyntaxError: bad",   # references the ACTUAL build error
        "planned_fix": "fix the kernel signature",
    })
    script = [
        _mk_call("build", {"kernel_src": "__BAD__"}),           # turn0: compile fail
        reflect + "\n" + _mk_call("test", {"kernel_src": "fixed"}),  # turn1: reflect+fix
        _mk_call("bench", {"kernel_src": "__FAST__"}) + "\n" + _mk_call("keep", {}),
        "done",
    ]
    ep = AgentHarness(FakeTask(), scripted_teacher(script), env,
                      max_turns=8, use_kb=False).run()

    assert len(ep.reflections) == 1
    assert ep.reflections[0]["turn"] == 1
    assert ep.reflections[0]["root_cause"] == "kernel failed to compile"

    comp = tool_use_reward(ep)
    assert comp["n_reflections"] == 1
    # grounded + complete reflection -> full reflection score, but BOUNDED.
    assert comp["reflection"] == 1.0
    assert 0.0 <= comp["reflection"] <= 1.0
    # the reflection term can never dominate the verified kernel outcome.
    assert W_REFLECT < W_OUTCOME
    assert W_REFLECT * comp["reflection"] < W_OUTCOME * comp["outcome"]


def test_reflection_reward_grounding_and_completeness():
    trace = [{"name": "build", "valid_name": True, "valid_params": True,
              "malformed": False, "result": {"ok": False, "error": "SyntaxError: bad"}}]
    base = {"tool_trace": trace, "best_reward": None, "keep_decisions": [],
            "success": False}

    grounded = tool_use_reward({**base, "reflections": [
        {"root_cause": "syntaxerror in signature", "evidence": "SyntaxError: bad",
         "planned_fix": "fix it"}]})
    empty = tool_use_reward({**base, "reflections": [
        {"root_cause": "", "evidence": "", "planned_fix": ""}]})
    ungrounded = tool_use_reward({**base, "reflections": [
        {"root_cause": "generic guess", "evidence": "nothing specific",
         "planned_fix": "try again"}]})

    assert grounded["reflection"] == 1.0
    assert empty["reflection"] == 0.0
    # complete-but-ungrounded gets completeness credit only (0.5), not grounding.
    assert ungrounded["reflection"] == 0.5
    # no reflections at all -> neutral zero (never negative).
    assert tool_use_reward(base)["reflection"] == 0.0


# --------------------------------------------------------------------------- #
# 12. Debugging-trap avoidance
# --------------------------------------------------------------------------- #
def test_trap_avoidance_reseeds_after_k_stalls():
    env = FakeEnv()
    # never reaches correctness -> every turn is non-improving
    script = [_mk_call("test", {"kernel_src": "cand __WRONG__"})] * 6
    ep = AgentHarness(FakeTask(), scripted_teacher(script), env, max_turns=6,
                      reseed_patience=3, seed_src="seed", use_kb=False).run()

    # After K=3 consecutive stalls the lineage is re-seeded exactly once (the
    # last turn can't reseed since there is nothing after it).
    assert len(ep.reseeds) == 1
    assert ep.reseeds[0]["turn"] == 2
    assert ep.reseeds[0]["stall"] == 3
    assert ep.reseeds[0]["reseeded"] is True
    assert ep.reseeds[0]["seeded_from"] == "task_seed"

    # a fresh 1-shot user prompt was injected to restart the design
    reseed_prompts = [m for m in ep.messages
                      if m["role"] == "user" and "abandon" in m["content"]]
    assert len(reseed_prompts) == 1


def test_trap_avoidance_does_not_fire_when_improving():
    env = FakeEnv()
    ep = AgentHarness(FakeTask(), scripted_teacher(_win_script()), env,
                      max_turns=8, reseed_patience=3, use_kb=False).run()
    assert ep.reseeds == []  # steady progress -> no reseed


# --------------------------------------------------------------------------- #
# 13. Correctness -> optimization phase split
# --------------------------------------------------------------------------- #
def test_phase_split_switches_prompt_on_correctness():
    # phase-specific system prompts are distinct and self-describing
    p_correct = build_agent_system_prompt(phase=PHASE_CORRECTNESS)
    p_opt = build_agent_system_prompt(phase=PHASE_OPTIMIZE)
    assert "PHASE 1" in p_correct and "PHASE 2" in p_opt
    assert p_correct != p_opt

    env = FakeEnv()
    ep = AgentHarness(FakeTask(), scripted_teacher(_win_script()), env,
                      max_turns=8, use_kb=False).run()

    phases = [p["phase"] for p in ep.phase_trace]
    assert len(ep.phase_trace) == ep.turns_used
    # correctness first (turns 0-2), optimize after the first correct kernel
    assert phases[0] == PHASE_CORRECTNESS
    assert phases[-1] == PHASE_OPTIMIZE
    assert PHASE_CORRECTNESS in phases and PHASE_OPTIMIZE in phases
    # the live system prompt was swapped to the phase-2 (optimize) prompt
    assert "PHASE 2" in ep.messages[0]["content"]


def test_phase_stays_correctness_when_never_correct():
    env = FakeEnv()
    script = [_mk_call("test", {"kernel_src": "cand __WRONG__"}), "done"]
    ep = AgentHarness(FakeTask(), scripted_teacher(script), env, max_turns=4,
                      use_kb=False).run()
    assert all(p["phase"] == PHASE_CORRECTNESS for p in ep.phase_trace)
    assert "PHASE 1" in ep.messages[0]["content"]


# --------------------------------------------------------------------------- #
# 14. Inference-time knowledge base (GEAK KB)
# --------------------------------------------------------------------------- #
def _write_win(tmp_path):
    wins_dir = tmp_path / "wins"
    rec = WinRecord(
        task_id="gemm_bf16",
        trajectory=[{"role": "system", "content": "s"}],
        initial_wall_us=2.0, final_wall_us=1.0, speedup=2.5,
        final_source="# WINNING_GEMM_KERNEL\nimport triton\n",
        snr_db=41.0, operation="gemm", arch="gfx942",
    )
    write_jsonl(wins_dir / "gemm.jsonl", [rec])
    return str(wins_dir)


def test_kb_retrieval_injects_prior_wins(tmp_path):
    wins_dir = _write_win(tmp_path)
    kb = WinsKB.from_dir(wins_dir)
    hits = kb.retrieve("gemm", "bf16", k=2)
    assert len(hits) == 1
    assert "WINNING_GEMM_KERNEL" in hits[0]["final_source"]

    # injected into the opening user prompt of the episode
    env = FakeEnv()
    ep = AgentHarness(FakeTask(), scripted_teacher(_win_script()), env,
                      max_turns=8, kb=kb).run()
    user0 = ep.messages[1]["content"]
    assert "Prior winning kernels" in user0
    assert "WINNING_GEMM_KERNEL" in user0


def test_kb_is_noop_when_no_wins(tmp_path):
    # empty KB (no wins dir) -> retrieval returns nothing, prompt unchanged
    empty = WinsKB.from_dir(str(tmp_path / "does_not_exist"))
    assert empty.retrieve("gemm", "bf16", k=3) == []
    assert WinsKB([]).retrieve("gemm", "bf16", k=3) == []

    env = FakeEnv()
    ep = AgentHarness(FakeTask(), scripted_teacher(_win_script()), env,
                      max_turns=8, use_kb=False).run()
    assert "Prior winning kernels" not in ep.messages[1]["content"]


def test_kb_does_not_leak_across_op_families(tmp_path):
    wins_dir = _write_win(tmp_path)  # a gemm win
    kb = WinsKB.from_dir(wins_dir)
    # an attention task should NOT retrieve the gemm win
    assert kb.retrieve("flash_attention", "bf16", k=2) == []


# --------------------------------------------------------------------------- #
# 15. gen_agentic carries reflections + phase structure
# --------------------------------------------------------------------------- #
def _reflect_win_script():
    reflect = render_reflection({
        "root_cause": "snr too low",
        "evidence": "worst SNR 5.0 < 25.0",
        "planned_fix": "rewrite the reduction",
    })
    return [
        _mk_call("test", {"kernel_src": "cand __WRONG__"}),          # fail
        reflect + "\n" + _mk_call("test", {"kernel_src": "cand fixed"}),  # reflect+fix
        _mk_call("bench", {"kernel_src": "cand __FAST__"}) + "\n" + _mk_call("keep", {}),
        "done",
    ]


def test_gen_agentic_records_carry_cognition():
    env = FakeEnv()
    teacher = scripted_teacher(_reflect_win_script())
    recs = generate_agentic_trajectories(FakeTask(), teacher, env, n=1, max_turns=8)
    assert len(recs) == 1
    rec = recs[0]

    # reflections + phase structure captured on the record
    assert rec.reflections and rec.reflections[0]["root_cause"] == "snr too low"
    assert {p["phase"] for p in rec.phase_trace} == {PHASE_CORRECTNESS, PHASE_OPTIMIZE}

    # provenance surfaces the cognition + the RL per-turn trace
    prov = rec.provenance
    assert prov["n_reflections"] == 1
    assert set(prov["phases"]) == {PHASE_CORRECTNESS, PHASE_OPTIMIZE}
    assert len(prov["turn_rewards"]) == len(prov["turn_correct"])

    # the reflection is woven into the trainable messages (SFT teaches cognition)
    assert any(m["role"] == "assistant" and "<reflect>" in m["content"]
               for m in rec.messages)

    # still trainer-valid + round-trips losslessly
    assert all("role" in m and "content" in m for m in rec.messages)
    assert AgenticTrajectoryRecord.from_dict(rec.to_dict()) == rec
