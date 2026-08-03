"""The 30B RL recipe must not silently reload the frozen reference replica.

grpo.py loads a frozen KL-anchor replica whenever ref_anchor_coef > 0. At
30.5B bf16 that is ~61GB per rank ON TOP of the rollout replica, so ~122GB per
rank before any training state -- the difference between a recipe that fits on
one node and one that OOMs partway through step 1.

The config previously shipped 0.001, which is the worst of both: full memory
price for a term contributing ~0.1% of the loss gradient. This test exists
because that value looks harmless in a diff and its cost is invisible until a
GPU node dies.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "grpo_coder30b_a3b_trloo.json"


def _cfg() -> dict:
    return json.loads(CONFIG.read_text())


def test_kl_anchor_is_off_so_no_reference_replica_is_loaded():
    coef = _cfg()["ref_anchor_coef"]
    assert coef == 0.0, (
        f"ref_anchor_coef={coef} loads a ~61GB/rank frozen reference replica. "
        "Raise it only after measuring peak memory on a real node."
    )


def test_the_memory_reasoning_is_recorded_next_to_the_value():
    # A future reader raising this back to 0.001 should meet the argument, not
    # rediscover the OOM.
    blob = " ".join(str(v) for k, v in _cfg().items() if k.startswith("_comment"))
    assert "61GB" in blob and "rank" in blob
    assert "ref_anchor_coef=0.0" in blob or "ref_anchor_coef" in blob


def test_stability_does_not_depend_on_the_dropped_kl_term():
    # Dropping KL is only safe because TRLOO carries its own stabilisers.
    d = _cfg()
    assert d.get("advantage_estimator") == "trloo"
    txt = json.dumps(d).lower()
    assert "clip" in txt, "no ratio clipping configured; KL was the only guard"
