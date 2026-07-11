"""Pillar 0 — prompt/response contract unification + offline normalization.

Verifies the single source of truth (kore.policy.format) is used everywhere and
that legacy shards can be upgraded to the canonical contract without touching
non-kernel retention rows.
"""

from __future__ import annotations

import json

from kore.policy.format import (
    OUTPUT_CONTRACT,
    SYSTEM_PROMPT,
    build_task_prompt,
    build_transcript,
    format_assistant_turn,
    normalize_assistant,
    parse_response,
    wrap_full_kernel,
)
from kore.data import normalize as N
import kore.data.prompts as P


def test_single_source_of_truth():
    assert P.SYSTEM_PROMPT is SYSTEM_PROMPT
    # canonical output contract asks for PROPOSED_CHANGE (not the old CHANGE:)
    assert "PROPOSED_CHANGE" in OUTPUT_CONTRACT
    btp = P.build_turn_prompt(parent_source="def k(): pass", mode="repair", feedback="x")
    assert "PROPOSED_CHANGE" in btp


class _Task:
    task_id = "gen_relu_fp16"
    dtype = "fp16"
    operation = "relu"
    gpu_target = "gfx942"
    backend = "triton"
    comparison_baseline = "torch_relu"
    seed_source = "def k():\n    pass"


def test_build_task_prompt_is_inference_context():
    # Pillar 3: DPO pairs must use the SAME turn-1 prompt as GRPO/eval (seed + contract).
    tp = build_task_prompt(_Task())
    assert "Seed kernel:" in tp and "def k():" in tp
    assert "ANALYSIS" in tp and "PROPOSED_CHANGE" in tp and "FULL_KERNEL" in tp
    # build_transcript wraps it into [system, user]
    msgs = build_transcript(tp, [])
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["role"] == "user" and "Seed kernel:" in msgs[1]["content"]


def test_format_assistant_turn_roundtrip():
    t = format_assistant_turn("mem bound", "Vectorize loads", "def k():\n    return 1")
    p = parse_response(t)
    assert p["analysis"] == "mem bound"
    assert p["proposed_change"] == "Vectorize loads"
    assert p["kernel"].strip() == "def k():\n    return 1"
    # PROPOSED_CHANGE omitted when empty
    t2 = format_assistant_turn("just analysis", "", "def k(): pass")
    assert "PROPOSED_CHANGE" not in t2 and t2.startswith("ANALYSIS:")


def test_wrap_full_kernel():
    w = wrap_full_kernel("def k():\n    return 1")
    assert w.startswith("FULL_KERNEL:") and "```python" in w
    assert parse_response(w)["kernel"].strip() == "def k():\n    return 1"


def test_normalize_assistant_all_legacy_shapes_and_idempotent():
    repair = ("<think>\nverifier rejected: snr low. fix fp32.\n</think>\n"
              "<answer>\nFULL_KERNEL:\n```python\ndef fixed():\n    return 2\n```\n</answer>")
    n = normalize_assistant(repair)
    assert "<think>" not in n and n.startswith("ANALYSIS:")
    assert "verifier rejected" in parse_response(n)["analysis"]
    assert parse_response(n)["kernel"].strip() == "def fixed():\n    return 2"

    gold = "ANALYSIS: fastest correct\n\nFULL_KERNEL:\ndef g():\n    return 3"
    ng = normalize_assistant(gold)
    assert "```python" in ng
    assert parse_response(ng)["kernel"].strip() == "def g():\n    return 3"

    change = ("ANALYSIS: mem\nCHANGE: bump_block\nFULL_KERNEL:\n"
              "```python\ndef c():\n    return 4\n```")
    ncx = normalize_assistant(change)
    assert parse_response(ncx)["proposed_change"] == "bump_block"

    # idempotent
    assert normalize_assistant(n) == n
    # no kernel -> unchanged (safe no-op for QA / NL)
    assert normalize_assistant("just some prose, no kernel") == "just some prose, no kernel"


def test_normalize_sft_row_source_aware():
    old_sys = "You are KORE, an expert AMD GPU kernel engineer. [OLD PROMPT]"
    # kernel row: system + assistant rewritten
    krow = {"_source": "kernel_repair_opt", "messages": [
        {"role": "system", "content": old_sys},
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "<think>diag</think>\n<answer>\nFULL_KERNEL:\n```python\ndef k():\n    return 1\n```\n</answer>"},
    ]}
    nrow, changed = N.normalize_sft_row(krow)
    assert changed
    assert nrow["messages"][0]["content"] == SYSTEM_PROMPT
    assert nrow["messages"][2]["content"].startswith("ANALYSIS:")

    # general_chat row: untouched entirely
    grow = {"_source": "general_chat", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>whatever</think> hello"},
    ]}
    nrow2, changed2 = N.normalize_sft_row(grow)
    assert not changed2 and nrow2 == grow

    # kernel_qa: system canonicalized, NL assistant untouched
    qrow = {"_source": "kernel_qa", "messages": [
        {"role": "system", "content": old_sys},
        {"role": "user", "content": "what raises occupancy?"},
        {"role": "assistant", "content": "Increase num_warps and reduce VGPR pressure."},
    ]}
    nrow3, changed3 = N.normalize_sft_row(qrow)
    assert changed3 and nrow3["messages"][0]["content"] == SYSTEM_PROMPT
    assert nrow3["messages"][2]["content"] == "Increase num_warps and reduce VGPR pressure."


def test_normalize_dpo_row():
    old_sys = "You are KORE, an expert AMD GPU kernel engineer. [OLD]"
    row = {
        "prompt": [{"role": "system", "content": old_sys},
                   {"role": "user", "content": "optimize"}],
        "chosen": [{"role": "assistant", "content": "FULL_KERNEL:\n```python\ndef a():\n    return 1\n```"}],
        "rejected": [{"role": "assistant", "content": "FULL_KERNEL:\n```python\ndef b():\n    return 2\n```"}],
    }
    nrow, changed = N.normalize_dpo_row(row)
    assert changed
    assert nrow["prompt"][0]["content"] == SYSTEM_PROMPT
    assert parse_response(nrow["chosen"][0]["content"])["kernel"].strip() == "def a():\n    return 1"


def test_normalize_raw_repair_and_win_records_preserve_verified_fields():
    # RAW RepairRecord (legacy <think>/<answer>) -> canonical, scalars untouched
    repair = {"task_id": "t", "failure_class": "snr_fail", "type": "repair",
              "child_snr_db": 87.5, "parent_hash": "abc",
              "messages": [
                  {"role": "system", "content": "You are KORE, an expert AMD GPU kernel engineer. OLD"},
                  {"role": "user", "content": "fix"},
                  {"role": "assistant", "content": "<think>diag</think>\n<answer>\nFULL_KERNEL:\n```python\ndef k():\n    return 1\n```\n</answer>"}]}
    nr, ch = N.normalize_raw_record_row(repair)
    assert ch
    a = nr["messages"][-1]["content"]
    assert a.startswith("ANALYSIS:") and "<think>" not in a
    assert nr["messages"][0]["content"] == SYSTEM_PROMPT
    assert nr["child_snr_db"] == 87.5 and nr["failure_class"] == "snr_fail"  # verified fields intact

    # RAW WinRecord (trajectory, CHANGE:) -> canonical, metrics untouched
    win = {"task_id": "t", "type": "win", "speedup": 2.0, "snr_db": 99.0,
           "final_source": "def k(): pass",
           "trajectory": [
               {"role": "user", "content": "opt"},
               {"role": "assistant", "content": "ANALYSIS: mem\nCHANGE: bump\nFULL_KERNEL:\n```python\ndef k():\n    return 2\n```"}]}
    nw, chw = N.normalize_raw_record_row(win)
    assert chw
    aw = nw["trajectory"][-1]["content"]
    assert "PROPOSED_CHANGE:" in aw and "CHANGE: bump" not in aw
    assert nw["speedup"] == 2.0 and nw["final_source"] == "def k(): pass"


def test_detect_kind_distinguishes_raw_records():
    assert N._detect_kind({"messages": [], "failure_class": "snr_fail", "type": "repair"}) == "raw"
    assert N._detect_kind({"trajectory": [], "type": "win"}) == "raw"
    assert N._detect_kind({"messages": [], "_source": "general_chat"}) == "sft"
    assert N._detect_kind({"prompt": [], "chosen": [], "rejected": []}) == "dpo"


def test_normalize_file_roundtrip(tmp_path):
    f = tmp_path / "sft.jsonl"
    rows = [
        {"_source": "kernel_repair_opt", "messages": [
            {"role": "system", "content": "You are KORE, an expert AMD GPU kernel engineer. OLD"},
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "<think>d</think>\n<answer>\nFULL_KERNEL:\n```python\ndef k():\n    return 1\n```\n</answer>"}]},
        {"_source": "general_chat", "messages": [
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    stats = N.normalize_file(f, in_place=True, backup=True)
    assert stats["kind"] == "sft" and stats["rows"] == 2 and stats["changed"] == 1
    out = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    assert out[0]["messages"][2]["content"].startswith("ANALYSIS:")
    assert out[1] == rows[1]  # general row untouched
    assert (tmp_path / "sft.jsonl.pre_normalize.bak").exists()
    # idempotent: second pass changes nothing
    stats2 = N.normalize_file(f, in_place=True, backup=False)
    assert stats2["changed"] == 0
