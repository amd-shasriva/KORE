"""Host-memory and resumability contract for the shipped training configs.

A 14B full-FT stage saves a FULL_STATE_DICT checkpoint, which gathers the whole
~220GB model + optimizer state onto the host in one go. Two config-level choices
decide whether that survives:

* the dataloader pipeline, because every rank spawns its own workers and pinned
  pages cannot be reclaimed to make room for the gather (12 workers x pinned x
  prefetch 4 on 8 ranks took a 14B midtrain down at step ~492 with
  ``HSA_STATUS_ERROR_OUT_OF_RESOURCES`` / "Available Free mem: 0 MB"); and
* ``save_total_limit``, because keeping exactly one checkpoint means a crash
  during the save that rotates the previous one out leaves nothing resumable and
  the run silently restarts from step 0.

These tests pin both across every shipped config, plus the pass-through in
``kore.policy.midtrain`` that makes the dataloader fields real rather than
decorative. CPU-only: no torch / transformers / trl import.

GRPO configs are deliberately out of scope -- its native loop builds no torch
DataLoader at all, so the fields would be inert there.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kore.policy.configs import DistributedMixin, MidTrainConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

# Loader shape that leaves the host pool free for the checkpoint gather.
MAX_SAFE_WORKERS = 4
MAX_SAFE_PREFETCH = 2
# One checkpoint is not crash-safe; the previous one is rotated out around the
# new save. Two costs ~220GB of extra disk and always leaves a resumable
# generation behind.
MIN_SAVE_TOTAL_LIMIT = 2

MIDTRAIN_CONFIGS = (
    "configs/midtrain_14b_full.json",
    "data/b05factory/launch/midtrain_8gpu.json",
    "data/b05factory/launch/midtrain_24gpu.json",
    "data/b05factory/launch/midtrain_32gpu.json",
    "data/b05factory/launch/midtrain_64gpu.json",
    "data/b05factory/launch/midtrain_frontier.json",
)
OTHER_TRAINER_CONFIGS = (
    "configs/sft_14b_full.json",
    "configs/dpo_14b_full.json",
)
TRAINER_CONFIGS = MIDTRAIN_CONFIGS + OTHER_TRAINER_CONFIGS


def _load(rel_path: str) -> dict:
    return json.loads((REPO_ROOT / rel_path).read_text())


def _settings(rel_path: str) -> dict:
    return {k: v for k, v in _load(rel_path).items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# every shipped Trainer-stage config carries safe dataloader settings
# --------------------------------------------------------------------------- #
def test_every_shipped_midtrain_config_is_enumerated():
    """A new launch config must opt into this contract, not slip past it."""
    on_disk = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "data" / "b05factory" / "launch").glob("*.json")
    }

    assert on_disk == {p for p in MIDTRAIN_CONFIGS if p.startswith("data/")}


@pytest.mark.parametrize("rel_path", TRAINER_CONFIGS)
def test_config_carries_safe_dataloader_settings(rel_path):
    config = _settings(rel_path)

    assert config["dataloader_num_workers"] <= MAX_SAFE_WORKERS, rel_path
    assert config["dataloader_num_workers"] >= 1, rel_path
    # Pinned pages are page-locked: they are NOT reclaimable when the ~200GB
    # CPU-side checkpoint gather needs the same host pool.
    assert config["dataloader_pin_memory"] is False, rel_path
    assert config["dataloader_prefetch_factor"] <= MAX_SAFE_PREFETCH, rel_path
    assert config["dataloader_prefetch_factor"] >= 1, rel_path


@pytest.mark.parametrize("rel_path", MIDTRAIN_CONFIGS)
def test_midtrain_config_keeps_a_resumable_spare_checkpoint(rel_path):
    assert _settings(rel_path)["save_total_limit"] >= MIN_SAVE_TOTAL_LIMIT, rel_path


def test_sft_and_dpo_configs_cannot_carry_save_total_limit():
    """Documents why ``save_total_limit`` is checked on midtrain configs only.

    ``SFTConfig`` / ``DPOConfig`` have no such field and their entrypoints
    hardcode the Trainer value, so putting the key in these JSONs would not
    change anything -- it would crash the strict ``Config(**d)`` parse. The
    contract is therefore enforced where the knob is real.
    """
    from kore.policy.configs import DPOConfig, SFTConfig

    for cls in (SFTConfig, DPOConfig):
        assert not hasattr(cls(), "save_total_limit")
    for rel_path in OTHER_TRAINER_CONFIGS:
        assert "save_total_limit" not in _load(rel_path), rel_path


# --------------------------------------------------------------------------- #
# the configs still parse into their dataclasses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel_path", MIDTRAIN_CONFIGS)
def test_midtrain_config_round_trips_through_its_builder(rel_path):
    from kore.policy.midtrain import midtrain_config_from_dict

    raw = _load(rel_path)
    config = midtrain_config_from_dict(raw)

    assert config.dataloader_num_workers == raw["dataloader_num_workers"]
    assert config.dataloader_pin_memory is False
    assert config.dataloader_prefetch_factor == raw["dataloader_prefetch_factor"]
    assert config.save_total_limit >= MIN_SAVE_TOTAL_LIMIT


def test_midtrain_builder_accepts_comment_keys_but_not_typos():
    from kore.policy.midtrain import midtrain_config_from_dict

    config = midtrain_config_from_dict(
        {"_comment_save_total_limit": "why", "save_total_limit": 2}
    )
    assert config.save_total_limit == 2
    with pytest.raises(TypeError):
        midtrain_config_from_dict({"dataloader_workers": 4})


def test_sft_and_dpo_configs_round_trip_with_safe_loader_settings():
    from kore.policy.dpo import dpo_config_from_dict
    from kore.policy.sft import sft_config_from_dict

    sft_cfg, _ = sft_config_from_dict(_load("configs/sft_14b_full.json"))
    dpo_cfg = dpo_config_from_dict(_load("configs/dpo_14b_full.json"))

    for config in (sft_cfg, dpo_cfg):
        assert config.dataloader_num_workers <= MAX_SAFE_WORKERS
        assert config.dataloader_pin_memory is False
        assert config.dataloader_prefetch_factor <= MAX_SAFE_PREFETCH


# --------------------------------------------------------------------------- #
# the defaults are safe too, so an un-tuned config cannot inherit a crash
# --------------------------------------------------------------------------- #
def test_distributed_mixin_defaults_are_host_memory_safe():
    defaults = DistributedMixin()

    assert defaults.dataloader_num_workers <= MAX_SAFE_WORKERS
    assert defaults.dataloader_num_workers >= 1
    assert defaults.dataloader_pin_memory is False
    assert defaults.dataloader_prefetch_factor <= MAX_SAFE_PREFETCH
    # dataset_num_proc is a one-shot map phase, not resident during training.
    assert defaults.dataset_num_proc >= 1


def test_midtrain_inherits_the_safe_loader_defaults():
    config = MidTrainConfig()

    assert config.dataloader_num_workers <= MAX_SAFE_WORKERS
    assert config.dataloader_pin_memory is False
    assert config.dataloader_prefetch_factor <= MAX_SAFE_PREFETCH


# --------------------------------------------------------------------------- #
# midtrain actually hands all four dataloader fields to the trainer
# --------------------------------------------------------------------------- #
def _trainer_kwargs(module_rel: str, function: str, call: str) -> set[str]:
    tree = ast.parse((REPO_ROOT / module_rel).read_text())
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    trl_call = next(
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == call
    )
    return {kw.arg for kw in trl_call.keywords if kw.arg}


def test_midtrain_passes_every_dataloader_field_to_the_trainer():
    kwargs = _trainer_kwargs("kore/policy/midtrain.py", "_train_single_process",
                             "TRLSFTConfig")

    # Dropping prefetch_factor / persistent_workers made `dataloader_prefetch_factor`
    # in the launch JSONs a dead field that looked like control.
    assert {
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "dataloader_prefetch_factor",
        "dataloader_persistent_workers",
        "dataset_num_proc",
    } <= kwargs


def test_midtrain_dataloader_pass_through_matches_sft():
    def loader_fields(names: set[str]) -> set[str]:
        return {name for name in names if name.startswith("dataloader_")}

    midtrain = _trainer_kwargs("kore/policy/midtrain.py", "_train_single_process",
                               "TRLSFTConfig")
    sft = _trainer_kwargs("kore/policy/sft.py", "train_sft", "TRLSFTConfig")

    assert loader_fields(sft) <= loader_fields(midtrain)


def test_midtrain_reads_loader_fields_directly_not_via_getattr_defaults():
    """No second, disagreeing default.

    The old ``getattr(config, "dataloader_num_workers", 8)`` fallbacks named 8
    while the dataclass said 12, so the code advertised a default it never used.
    These are ``DistributedMixin`` fields and are always present.
    """
    source = (REPO_ROOT / "kore" / "policy" / "midtrain.py").read_text()

    assert 'getattr(config, "dataloader' not in source


# --------------------------------------------------------------------------- #
# the `-m` entry preflights the corpus before loading a 14B model
# --------------------------------------------------------------------------- #
def test_midtrain_entry_rejects_a_missing_corpus(tmp_path):
    from kore.policy import midtrain

    config = tmp_path / "midtrain.json"
    config.write_text(json.dumps({
        "model_id": "Qwen/Qwen3-14B",
        "corpus_path": str(tmp_path / "absent.jsonl"),
        "use_lora": False,
    }))

    assert midtrain._main([str(config)]) == 2
    assert midtrain._main([]) == 2


def test_midtrain_entry_preflight_runs_before_any_torch_import(tmp_path):
    config = tmp_path / "midtrain.json"
    config.write_text(json.dumps({"corpus_path": str(tmp_path / "absent.jsonl")}))
    code = (
        "import sys\n"
        "from kore.policy import midtrain\n"
        f"rc = midtrain._main([{str(config)!r}])\n"
        "assert rc == 2, rc\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
