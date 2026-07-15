"""CPU-only tests for synthesizing native agentic trajectories from verified data.

No GPU / teacher / torch. We feed hand-built repair/wins/groups record dicts
(the exact on-disk schema) into the synthesizers and assert the reconstructed
Hermes trajectories are training-ready: correct role sequence, parseable tool
calls, a terminal ``keep``, faithful (measured) tool results, round-trip through
``episode_to_chat``, and pickup by ``assemble._agentic_rows``. Also pins the
backward-compatible arch parameterization of the system prompt.
"""

from __future__ import annotations

import json

from kore.agent.format import (
    build_agent_system_prompt,
    episode_to_chat,
    parse_tool_calls,
)
from kore.agent.schema import AgenticTrajectoryRecord
from kore.data.assemble import _agentic_rows
from kore.data.synth_agentic import (
    synth_from_group,
    synth_from_repair,
    synth_from_win,
    synthesize_agentic,
)

_KERNEL = (
    '"""seed."""\n'
    "import triton\nimport triton.language as tl\n\n"
    "@triton.jit\ndef _k(x_ptr, y_ptr, n):\n    pass\n\n"
    "def entry(x):\n    return x\n"
)
_FIXED = _KERNEL.replace("seed", "fixed")


def _repair_rec():
    return {
        "task_id": "genv_rmsnorm_bf16",
        "failure_class": "snr_fail",
        "error_text": "correctness failed (snr_db=-3.11)",
        "child_snr_db": 87.78,
        "operation": "rmsnorm",
        "arch": "gfx942",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user",
             "content": f"## Mode: REPAIR\nfix it.\n```python\n{_KERNEL}\n```"},
            {"role": "assistant",
             "content": f"<think>\nlow SNR: restore fp32 accumulation.\n</think>\n"
                        f"```python\n{_FIXED}\n```"},
        ],
    }


def _win_rec():
    return {
        "task_id": "genv_add_rmsnorm_bf16",
        "trajectory": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "optimize"},
            {"role": "assistant",
             "content": f"ANALYSIS: memory-bound.\n```python\n{_KERNEL}\n```"},
        ],
        "initial_wall_us": 48.0,
        "final_wall_us": 46.0,
        "speedup": 1.0434,
        "final_source": _FIXED,
        "snr_db": 95.4,
        "operation": "add_rmsnorm",
        "arch": "gfx942",
    }


def _group_rec():
    return {
        "task_id": "genv_softmax_fp16",
        "operation": "softmax",
        "arch": "gfx942",
        "candidates": [
            {"source": _FIXED, "wall_us": 40.0, "snr_db": 95.8, "rank": 0},
            {"source": _KERNEL, "wall_us": 52.0, "snr_db": 90.1, "rank": 1},
            {"source": _KERNEL.replace("seed", "v3"), "wall_us": 60.0, "snr_db": 88.0, "rank": 2},
        ],
        "preferences": [[0, 1], [0, 2], [1, 2]],
    }


def _roles(rec: AgenticTrajectoryRecord):
    return [m["role"] for m in rec.messages]


def _last_tool(rec: AgenticTrajectoryRecord) -> dict:
    tools = [m for m in rec.messages if m["role"] == "tool"]
    return json.loads(tools[-1]["content"])


# --------------------------------------------------------------------------- #
# Per-record synthesis
# --------------------------------------------------------------------------- #
def test_repair_trajectory_is_faithful_and_wellformed():
    rec = synth_from_repair(_repair_rec())
    assert rec is not None
    # system, user, then (assistant, tool) triples ending in keep.
    assert _roles(rec)[:2] == ["system", "user"]
    assert rec.messages[-1]["role"] == "tool"
    # First tool result reproduces the REAL failure (measured SNR from error_text).
    first_tool = json.loads([m for m in rec.messages if m["role"] == "tool"][0]["content"])
    assert first_tool["ok"] is False and first_tool["correct"] is False
    assert first_tool["snr_db"] == -3.11
    # The fix turn carries the real fixed-kernel SNR and a grounded reflection.
    assert any(t["result"].get("snr_db") == 87.78 for t in rec.tool_trace)
    assert rec.reflections and rec.reflections[0]["evidence"]
    # Ends on a committed keep of the fixed kernel.
    assert _last_tool(rec)["tool"] == "keep" and rec.best_kernel == _FIXED.strip()
    assert rec.provenance["category"] == "repair"


def test_repair_compile_fail_marks_not_compiled():
    r = _repair_rec()
    r["failure_class"] = "compile_fail"
    r["error_text"] = "NameError: tl.foo is not defined"
    rec = synth_from_repair(r)
    first_tool = json.loads([m for m in rec.messages if m["role"] == "tool"][0]["content"])
    assert first_tool["compiled"] is False and first_tool["ok"] is False


def test_win_trajectory_uses_measured_walltimes():
    rec = synth_from_win(_win_rec())
    assert rec is not None
    benches = [t["result"] for t in rec.tool_trace if t["name"] == "bench"]
    assert benches[0]["wall_ms"] == 0.048          # 48us seed
    assert benches[1]["wall_ms"] == 0.046          # 46us final
    assert benches[1]["speedup"] == 1.043
    assert _last_tool(rec)["tool"] == "keep"
    assert rec.provenance["category"] == "success"


def test_group_trajectory_keeps_fastest():
    rec = synth_from_group(_group_rec())
    assert rec is not None
    benches = [t["result"]["wall_ms"] for t in rec.tool_trace if t["name"] == "bench"]
    # Explored worst->best; the fastest (0.040ms) is benched last, then kept.
    assert benches[-1] == 0.040
    assert _last_tool(rec)["tool"] == "keep"
    assert rec.best_kernel == _FIXED       # groups keep the raw candidate source
    assert rec.provenance["category"] == "search"


# --------------------------------------------------------------------------- #
# Format compatibility (must match the live harness contract)
# --------------------------------------------------------------------------- #
def test_tool_calls_parse_and_chat_roundtrips():
    for rec in (synth_from_repair(_repair_rec()),
                synth_from_win(_win_rec()),
                synth_from_group(_group_rec())):
        assistants = [m for m in rec.messages if m["role"] == "assistant"]
        # Every assistant turn emits exactly one parseable, non-malformed call.
        for a in assistants:
            calls = parse_tool_calls(a["content"])
            assert len(calls) == 1 and not calls[0]["malformed"]
            assert calls[0]["name"] in {"test", "bench", "keep"}
        # Both SFT variants render without loss of the tool/role structure.
        for thinking in (True, False):
            chat = episode_to_chat(rec, thinking=thinking)
            assert [m["role"] for m in chat] == _roles(rec)
            if not thinking:
                assert all("<think>" not in m["content"]
                           for m in chat if m["role"] == "assistant")


def test_arch_parameterization():
    # Default (no arch) is now the KORE target hardware, gfx950/CDNA4 (MI350X).
    assert "MI350X (gfx950 / CDNA4)" in build_agent_system_prompt()
    # Explicit gfx950 -> MI350X (the reported board, not MI355X).
    assert "MI350X (gfx950 / CDNA4)" in build_agent_system_prompt(arch="gfx950")
    # Explicit gfx942 -> previous-gen descriptor still available for cross-arch.
    assert "gfx942" in build_agent_system_prompt(arch="gfx942")
    # Synthesized prompts follow the source record's arch (corpus-consistent).
    rec = synth_from_repair(_repair_rec())
    assert "gfx9" in rec.messages[0]["content"]


# --------------------------------------------------------------------------- #
# Driver + pickup by the SFT assembler
# --------------------------------------------------------------------------- #
def test_synthesize_writes_and_assemble_picks_up(tmp_path):
    from kore.data.schemas import write_jsonl
    dr = tmp_path / "data"
    for kind, rec in (("repair", _repair_rec()), ("wins", _win_rec()), ("groups", _group_rec())):
        (dr / kind).mkdir(parents=True)
        write_jsonl(dr / kind / "shard.jsonl", [rec] * 5)

    summary = synthesize_agentic(dr, cap=30, seed=0)
    assert summary["total"] >= 3 and summary["repair"] >= 1

    rows = _agentic_rows(dr)
    assert rows and all("messages" in r for r in rows)
    # Determinism: a second identical run yields the same counts.
    assert synthesize_agentic(dr, cap=30, seed=0) == summary
