"""CPU-only tests for gold-win mining from verified ranked groups.

Asserts: the minter picks the robustly-best (rank-0) correct candidate as the
gold target, frames a slower correct sibling as the parent, computes a real
measured speedup, emits a real-wins-format trajectory (SYSTEM + writer user +
`ANALYSIS: … FULL_KERNEL:` assistant), skips groups without a faster-correct
candidate, honors the per-task cap, and round-trips through the typed reader so
the build's raw gather treats it exactly like a real win.
"""

from __future__ import annotations

from kore.data.build_datasets import build_sft
from kore.data.gold_wins import mint_gold_win, mint_gold_wins
from kore.data.prompts import extract_kernel
from kore.data.schemas import WinRecord, read_jsonl

_SRC_FAST = '"""fast."""\ndef entry(x):\n    return x  # fast\n'
_SRC_SLOW = '"""slow."""\ndef entry(x):\n    return x  # slow\n'
_SRC_MID = '"""mid."""\ndef entry(x):\n    return x  # mid\n'


def _group(**over):
    g = {
        "task_id": "genv_softmax_fp16",
        "operation": "softmax",
        "arch": "gfx942",
        "shape": {"n": 4096},
        "candidates": [
            {"source": _SRC_FAST, "wall_us": 40.0, "snr_db": 999.0, "rank": 0},
            {"source": _SRC_MID, "wall_us": 55.0, "snr_db": 120.0, "rank": 1},
            {"source": _SRC_SLOW, "wall_us": 80.0, "snr_db": 95.0, "rank": 2},
        ],
        "preferences": [[0, 1], [0, 2]],
    }
    g.update(over)
    return g


def test_mint_picks_rank0_and_measures_real_speedup():
    w = mint_gold_win(_group())
    assert isinstance(w, WinRecord) and w.type == "win"
    assert w.final_source == _SRC_FAST           # rank-0 gold target
    assert w.final_wall_us == 40.0
    # median-slower sibling is the parent (55us) -> speedup 55/40 = 1.375
    assert w.initial_wall_us == 55.0
    assert abs(w.speedup - 1.375) < 1e-3
    assert w.operation == "softmax" and w.arch == "gfx942"


def test_trajectory_is_real_wins_format():
    w = mint_gold_win(_group())
    roles = [m["role"] for m in w.trajectory]
    assert roles == ["system", "user", "assistant"]
    asst = w.trajectory[-1]["content"]
    assert "ANALYSIS:" in asst and "FULL_KERNEL:" in asst
    # The gold kernel is recoverable by the corpus's own extractor.
    assert extract_kernel(asst).strip() == _SRC_FAST.strip()
    # And build_sft turns it straight into an SFT chat row.
    rows = build_sft([w])
    assert rows and rows[0]["messages"][0]["role"] == "system"


def test_gate_rejects_no_improvement_and_low_snr():
    # All candidates same speed -> no faster-correct sibling -> skip.
    flat = _group(candidates=[
        {"source": _SRC_FAST, "wall_us": 40.0, "snr_db": 999.0, "rank": 0},
        {"source": _SRC_MID, "wall_us": 40.0, "snr_db": 999.0, "rank": 1},
    ])
    assert mint_gold_win(flat) is None
    # Only one clearly-correct candidate (other below the SNR gate) -> skip.
    low = _group(candidates=[
        {"source": _SRC_FAST, "wall_us": 40.0, "snr_db": 999.0, "rank": 0},
        {"source": _SRC_SLOW, "wall_us": 80.0, "snr_db": 5.0, "rank": 1},
    ])
    assert mint_gold_win(low) is None


def test_driver_writes_typed_roundtrip_and_respects_caps(tmp_path):
    from kore.data.schemas import write_jsonl
    dr = tmp_path / "data"
    (dr / "groups").mkdir(parents=True)
    write_jsonl(dr / "groups" / "shard.jsonl", [_group() for _ in range(10)])

    summary = mint_gold_wins(dr, cap=100, per_task_cap=3, seed=0)
    assert summary["gold_wins"] == 3            # per_task_cap bounds a single task
    assert summary["tasks_covered"] == 1

    back = read_jsonl(dr / "wins" / "_gold_from_groups.jsonl", typed=True)
    assert back and all(isinstance(r, WinRecord) for r in back)
    # Determinism.
    assert mint_gold_wins(dr, cap=100, per_task_cap=3, seed=0) == summary
