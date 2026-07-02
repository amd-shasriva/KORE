"""CPU-only tests for kore.data.assemble (multicap SFT mix + DPO hard negatives)."""

from __future__ import annotations

from kore.data import assemble
from kore.data.teacher import StubTeacher
from kore.policy.configs import MultiCapSFTConfig
from kore.tasks.registry import all_tasks


def test_build_multicap_dataset_offline_stub(tmp_path):
    tasks = all_tasks()[:3]
    cfg = MultiCapSFTConfig()
    rows = assemble.build_multicap_dataset(tmp_path, tasks, StubTeacher(), cfg,
                                           total=200, use_hf=False, verbose=False)
    assert rows, "mixture should be non-empty from general replay + QA alone"
    # every row is a chat row tagged with a source bucket
    for r in rows:
        assert "messages" in r and isinstance(r["messages"], list)
        assert r.get("_source") in {
            "kernel_repair_opt", "kernel_qa", "agentic_tooluse",
            "general_code", "math_reasoning", "general_chat",
        }
    rep = assemble.summarize_multicap(rows)
    assert abs(sum(rep["fractions"].values()) - 1.0) < 1e-6


def test_dpo_hard_negatives_meets_floor(tmp_path):
    tasks = all_tasks()[:4]
    out = assemble.build_dpo_with_hard_negatives(tmp_path, tasks)
    # no ranked groups on disk -> all pairs are hard negatives -> target trivially met
    assert out["n_total"] == out["n_hard"] > 0
    assert out["meets_target"] is True
    # each row is a DPO triple
    for row in out["rows"][:5]:
        assert set(row) >= {"prompt", "chosen", "rejected"}


def test_dpo_hard_negatives_are_rejected_by_scanner():
    from kore.reward.reward import scan_for_hacks
    from kore.data.hard_negatives import build_hard_negative_group

    task = all_tasks()[0]
    grp = build_hard_negative_group(task.seed_source, task)
    # candidate 0 is the trusted correct source; the rest are hacks.
    assert scan_for_hacks(grp.candidates[0]["source"]) is None
    hacks_flagged = sum(scan_for_hacks(c["source"]) is not None for c in grp.candidates[1:])
    assert hacks_flagged >= 3  # at least the Layer-A vendor/torch/try-except/copy-ref
