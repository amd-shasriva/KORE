"""CPU-only tests for repair->DPO pair minting (fixed > broken).

Asserts: the minter extracts the broken kernel (user turn) and the teacher fix
(assistant turn), packages them as a 2-candidate ranked group with preference
[0,1], skips degenerate/short records, honors the per-task cap, and round-trips
through the typed reader so the build's raw gather + build_dpo turn it into a
real preference row.
"""

from __future__ import annotations

from kore.data.build_datasets import build_dpo
from kore.data.repair_dpo import mint_repair_dpo, mint_repair_pair
from kore.data.schemas import RankedGroupRecord, read_jsonl

_BROKEN = '"""k."""\ndef entry(x):\n    return x  # BROKEN wrong accumulation\n'
_FIXED = '"""k."""\ndef entry(x):\n    return x  # FIXED fp32 accumulation\n'


def _repair(**over):
    r = {
        "task_id": "genv_rmsnorm_bf16",
        "failure_class": "snr_fail",
        "parent_hash": "abc123",
        "error_text": "correctness failed (snr_db=-3.11)",
        "child_snr_db": 87.78,
        "operation": "rmsnorm",
        "arch": "gfx942",
        "shape": {"n": 4096},
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"fix this:\n```python\n{_BROKEN}\n```"},
            {"role": "assistant", "content": f"<think>fp32</think>\n```python\n{_FIXED}\n```"},
        ],
    }
    r.update(over)
    return r


def test_mint_pair_orders_fixed_over_broken():
    g = mint_repair_pair(_repair())
    assert isinstance(g, RankedGroupRecord) and g.type == "ranked_group"
    assert g.preferences == [[0, 1]]
    assert g.candidates[0]["source"] == _FIXED.strip()   # chosen = fixed (fence-stripped)
    assert g.candidates[1]["source"] == _BROKEN.strip()  # rejected = broken
    assert g.candidates[0]["snr_db"] == 87.78
    assert g.operation == "rmsnorm" and g.arch == "gfx942"


def test_mint_pair_skips_degenerate_and_short():
    # identical broken/fixed -> no learnable signal.
    same = _repair(messages=[
        {"role": "system", "content": "s"},
        {"role": "user", "content": f"```python\n{_FIXED}\n```"},
        {"role": "assistant", "content": f"```python\n{_FIXED}\n```"},
    ])
    assert mint_repair_pair(same) is None
    # too few messages.
    assert mint_repair_pair({"messages": [{"role": "user", "content": "x"}]}) is None


def test_pair_builds_a_real_dpo_row():
    g = mint_repair_pair(_repair())
    rows = build_dpo([g])
    assert len(rows) == 1
    r = rows[0]
    assert "prompt" in r and r["chosen"] and r["rejected"]
    assert "FIXED" in r["chosen"][0]["content"]
    assert "BROKEN" in r["rejected"][0]["content"]


def test_driver_writes_typed_roundtrip_and_caps(tmp_path):
    from kore.data.schemas import write_jsonl
    dr = tmp_path / "data"
    (dr / "repair").mkdir(parents=True)
    write_jsonl(dr / "repair" / "shard.jsonl", [_repair() for _ in range(10)])

    summary = mint_repair_dpo(dr, cap=100, per_task_cap=4, seed=0)
    assert summary["repair_pairs"] == 4          # per_task_cap bounds one task
    assert summary["tasks_covered"] == 1

    back = read_jsonl(
        dr / "groups" / "_repair_pairs.jsonl",
        typed=True,
        mode="production_strict",
    )
    assert back and all(isinstance(r, RankedGroupRecord) for r in back)
    assert len(build_dpo(back)) == 4
    # Determinism.
    assert mint_repair_dpo(dr, cap=100, per_task_cap=4, seed=0) == summary
