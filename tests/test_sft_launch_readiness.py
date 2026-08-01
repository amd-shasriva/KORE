"""CPU-runnable launch-readiness regressions for Stage-1 SFT.

These lock down the parts of the ``midtrain -> sft`` handoff that were verified
end-to-end on 2x MI350X (gfx950) for ``docs/SFT_READINESS.md``: the packaged
corpus, the ``{% generation %}`` template surgery, model-identity resolution,
checkpoint discovery, and the step-count arithmetic that sets the run length.

Nothing here needs a GPU. Tests that need the real Qwen3-14B tokenizer skip when
that exact commit is not in the local Hugging Face cache, and tests that read the
packaged corpus skip when the committed ``data/release/sft/`` parts are absent,
so a bare checkout still collects and passes.

Four tests are marked ``xfail``: they assert the behaviour the readiness review
concluded is *correct*, against code that does not implement it yet. Each one
turns green the moment the corresponding proposed patch in
``docs/SFT_READINESS.md`` lands, which is the point - they are the executable
form of the blocker list.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SFT_CONFIG_PATH = REPO / "configs" / "sft_14b_full.json"
RELEASE_SFT = REPO / "data" / "release" / "sft"
REASSEMBLED_SFT = REPO / "data" / "b05factory" / "sft" / "multicap.jsonl"

#: Immutable Hub commit the frontier dataset was built against (DATASET_STATUS.md).
QWEN3_14B = "Qwen/Qwen3-14B"
PINNED_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"

#: Documented corpus shape, measured from the committed package on 2026-08-01.
DOCUMENTED_ROWS = 56_493
REPAIR_ROWS = 15_083            # _provenance.kind == "repair"
ROWS_AFTER_UPSAMPLE = 71_576    # repair_loss_weight 2.0 duplicates each repair row
ROWS_OVER_16384 = 3_299         # dropped by _filter_overlong at max_seq_length 16384
ROWS_TRAINED = 68_277           # ROWS_AFTER_UPSAMPLE - ROWS_OVER_16384


def _sft_config() -> dict:
    return json.loads(SFT_CONFIG_PATH.read_text())


def _local_snapshot() -> Path | None:
    from kore.policy.model_spec import resolve_local_snapshot

    return resolve_local_snapshot(QWEN3_14B, PINNED_REVISION)


_NEEDS_TOKENIZER = pytest.mark.skipif(
    _local_snapshot() is None,
    reason=(
        f"{QWEN3_14B}@{PINNED_REVISION[:12]} is not in the local Hugging Face "
        "cache; these tests read the real chat template rather than a fixture."
    ),
)
_NEEDS_PACKAGE = pytest.mark.skipif(
    not sorted(RELEASE_SFT.glob("multicap.jsonl.gz.part*")),
    reason="packaged SFT corpus (data/release/sft/) is not present in this checkout",
)


def _packaged_stream():
    """Decompress the split gzip package without materializing 630MB on disk."""
    parts = sorted(RELEASE_SFT.glob("multicap.jsonl.gz.part*"))

    class _Cat:
        def __init__(self, paths):
            self._paths, self._i, self._fh = list(paths), 0, None

        def read(self, n=-1):
            while True:
                if self._fh is None:
                    if self._i >= len(self._paths):
                        return b""
                    self._fh = self._paths[self._i].open("rb")
                    self._i += 1
                block = self._fh.read(n)
                if block:
                    return block
                self._fh.close()
                self._fh = None

    return gzip.GzipFile(fileobj=_Cat(parts))


def _packaged_rows(limit: int | None = None):
    with _packaged_stream() as raw:
        for i, line in enumerate(raw):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# 1. Packaged SFT corpus
# --------------------------------------------------------------------------- #
@_NEEDS_PACKAGE
def test_packaged_sft_parts_are_ordered_and_gzip_valid():
    """``cat part* | gunzip`` (reassemble.sh) must produce a readable stream.

    Glob order is the reassembly order, so a part named out of sequence would
    silently corrupt the corpus; asserting the exact names pins that down.
    """
    parts = sorted(RELEASE_SFT.glob("multicap.jsonl.gz.part*"))
    assert [p.name for p in parts] == [
        "multicap.jsonl.gz.part00",
        "multicap.jsonl.gz.part01",
    ]
    first = next(_packaged_rows(limit=1))
    assert isinstance(first, dict) and isinstance(first.get("messages"), list)


@_NEEDS_PACKAGE
def test_packaged_sft_head_rows_match_the_chat_schema():
    """Every row is ``{"messages": [{role, content}, ...]}`` with a real assistant turn."""
    valid_roles = {"system", "user", "assistant", "tool"}
    seen_sources = set()
    for rec in _packaged_rows(limit=2000):
        messages = rec["messages"]
        assert messages, "empty conversation"
        roles = []
        for message in messages:
            assert message["role"] in valid_roles, message["role"]
            assert isinstance(message["content"], str) and message["content"].strip()
            roles.append(message["role"])
        assert roles[0] in {"system", "user"}, roles[0]
        # TRL's assistant_only_loss raises on a row with no assistant tokens.
        assert "assistant" in roles
        seen_sources.add(rec.get("_source"))
    assert seen_sources, "rows carry no _source provenance tag"


@pytest.mark.release
@_NEEDS_PACKAGE
def test_packaged_sft_row_count_matches_the_documented_corpus():
    """Full-corpus count: the number DATASET_STATUS.md and the run plan depend on."""
    rows = repairs = 0
    for rec in _packaged_rows():
        rows += 1
        provenance = rec.get("_provenance") or {}
        if provenance.get("kind") == "repair" or rec.get("_source") == "repair":
            repairs += 1
    assert rows == DOCUMENTED_ROWS
    assert repairs == REPAIR_ROWS
    assert rows + repairs == ROWS_AFTER_UPSAMPLE


def test_documented_row_count_is_consistent_across_the_repo():
    status = (REPO / "DATASET_STATUS.md").read_text()
    assert f"{DOCUMENTED_ROWS:,}" in status or str(DOCUMENTED_ROWS) in status


@pytest.mark.xfail(
    reason=(
        "BLOCKER 1: configs/sft_14b_full.json points at data/sft/multicap.jsonl, "
        "but data/release/reassemble.sh writes data/b05factory/sft/multicap.jsonl. "
        "A direct `launch_distributed.sh sft configs/sft_14b_full.json` therefore "
        "loads 14B on every rank and only then raises FileNotFoundError."
    ),
)
def test_shipped_config_dataset_path_is_what_reassemble_produces():
    reassemble = (REPO / "data" / "release" / "reassemble.sh").read_text()
    assert "../b05factory/sft/multicap.jsonl" in reassemble
    configured = Path(_sft_config()["dataset_path"])
    assert configured == Path("data/b05factory/sft/multicap.jsonl"), (
        f"config dataset_path {configured} is not the path reassemble.sh writes"
    )


# --------------------------------------------------------------------------- #
# 2. Loss-masking template surgery
# --------------------------------------------------------------------------- #
@_NEEDS_TOKENIZER
def test_generation_markers_are_injected_and_idempotent():
    from kore.policy.sft import build_assistant_masked_template

    snapshot = _local_snapshot()
    template = json.loads(
        (snapshot / "tokenizer_config.json").read_text()
    )["chat_template"]
    assert "{% generation %}" not in template, "base Qwen3 template already tagged"

    masked = build_assistant_masked_template(template)
    assert "{% generation %}" in masked and "{% endgeneration %}" in masked
    assert build_assistant_masked_template(masked) == masked


def test_non_qwen3_template_fails_loudly_rather_than_training_unmasked():
    from kore.policy.sft import build_assistant_masked_template

    with pytest.raises(ValueError, match="could not inject generation markers"):
        build_assistant_masked_template("{{ messages[0]['content'] }}")


@_NEEDS_TOKENIZER
def test_masked_template_renders_byte_identically_and_masks_correctly():
    """The repo's own fail-fast guard, run against the real tokenizer."""
    from transformers import AutoTokenizer

    from kore.policy.sft import (
        _verify_assistant_masking,
        build_assistant_masked_template,
    )

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_14B, revision=PINNED_REVISION)
    base = tokenizer.chat_template
    _verify_assistant_masking(tokenizer, base, build_assistant_masked_template(base))
    assert tokenizer.chat_template == base, "guard leaked the masked template"


@_NEEDS_TOKENIZER
@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [{"role": "user", "content": "Write a HIP kernel."},
             {"role": "assistant", "content": "KERNEL_BODY_A"}],
            id="single-turn",
        ),
        pytest.param(
            [{"role": "system", "content": "SYS_TXT"},
             {"role": "user", "content": "Q1"},
             {"role": "assistant", "content": "RESP_ONE"},
             {"role": "user", "content": "Q2"},
             {"role": "assistant", "content": "RESP_TWO"}],
            id="multi-turn",
        ),
        pytest.param(
            [{"role": "user", "content": "think please"},
             {"role": "assistant",
              "content": "<think>\nreasoning_here\n</think>\nFINAL_ANS"}],
            id="think",
        ),
        pytest.param(
            [{"role": "user", "content": "call tool"},
             {"role": "assistant", "content": "TOOL_PREAMBLE"},
             {"role": "tool", "content": "TOOL_OUT"},
             {"role": "assistant", "content": "TOOL_FINAL"}],
            id="tool",
        ),
    ],
)
def test_only_assistant_token_ids_survive_into_the_loss(messages):
    """On real token ids: assistant spans train, everything else becomes -100."""
    from transformers import AutoTokenizer

    from kore.policy.sft import build_assistant_masked_template

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_14B, revision=PINNED_REVISION)
    base = tokenizer.chat_template
    tokenizer.chat_template = build_assistant_masked_template(base)

    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_assistant_tokens_mask=True, return_dict=True)
    ids, mask = encoded["input_ids"], encoded["assistant_masks"]
    labels = [i if m == 1 else -100 for i, m in zip(ids, mask)]

    assert any(label != -100 for label in labels), "nothing to train on"
    assert any(label == -100 for label in labels), "nothing was masked"

    learned = tokenizer.decode([i for i, m in zip(ids, mask) if m == 1])
    dropped = tokenizer.decode([i for i, m in zip(ids, mask) if m == 0])
    for turn in messages:
        if turn["role"] == "assistant":
            tail = turn["content"].split("</think>")[-1].strip()
            assert tail in learned
        else:
            assert turn["content"] not in learned, (
                f"{turn['role']} content leaked into the loss"
            )
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") in [
        i for i, m in zip(ids, mask) if m == 1
    ], "the assistant stop token must be in the loss"
    assert "<|im_start|>assistant" in dropped, "assistant header must be masked"


@_NEEDS_TOKENIZER
@_NEEDS_PACKAGE
def test_real_corpus_rows_mask_correctly():
    from transformers import AutoTokenizer

    from kore.policy.sft import build_assistant_masked_template

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_14B, revision=PINNED_REVISION)
    base = tokenizer.chat_template
    masked = build_assistant_masked_template(base)

    checked = 0
    for rec in _packaged_rows(limit=64):
        messages = rec["messages"]
        tokenizer.chat_template = base
        plain = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False)
        tokenizer.chat_template = masked
        tagged = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=False)
        assert plain == tagged, "masked template perturbed a real row's rendering"

        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_assistant_tokens_mask=True, return_dict=True)
        mask = encoded["assistant_masks"]
        assert 1 in mask, "real row has no assistant tokens (TRL would raise)"
        assert 0 in mask, "real row trains on its own prompt"
        checked += 1
    assert checked, "no rows were checked"


# --------------------------------------------------------------------------- #
# 3. Model identity
# --------------------------------------------------------------------------- #
def test_shipped_config_parses_and_carries_an_immutable_revision():
    from kore.policy.model_spec import validate_pinned_revision
    from kore.policy.sft import sft_config_from_dict

    payload = _sft_config()
    assert validate_pinned_revision(payload["model_revision"]) == PINNED_REVISION

    config, dataset_path = sft_config_from_dict(dict(payload))
    assert config.model_id == QWEN3_14B
    assert getattr(config, "model_revision") == PINNED_REVISION
    assert dataset_path == payload["dataset_path"]
    # The locked recipe: full-FT, FSDP, completion-only loss, packing off (SDPA).
    assert config.use_lora is False and config.distributed is True
    assert config.assistant_only_loss is True and config.packing is False


@pytest.mark.skipif(_local_snapshot() is None,
                    reason="pinned Qwen3-14B snapshot is not cached locally")
def test_pinned_revision_resolves_offline_and_is_passed_to_from_pretrained():
    from kore.policy.model_spec import model_identity_for_config
    from kore.policy.sft import sft_config_from_dict

    config, _ = sft_config_from_dict(_sft_config())
    identity = model_identity_for_config(
        config, stage="sft", environ={"HF_HUB_OFFLINE": "1"})

    assert identity.revision == PINNED_REVISION
    assert identity.pin_load is True
    assert identity.load_kwargs == {"revision": PINNED_REVISION}
    assert identity.local_path and identity.local_path.endswith(PINNED_REVISION)
    assert identity.inspection is not None
    assert identity.inspection.parameter_count == 14_768_307_200
    assert identity.inspection.architecture.decoder_class == "Qwen3DecoderLayer"


@pytest.mark.parametrize("revision", ["main", "v1.0", "40c0698", "refs/pr/1",
                                      PINNED_REVISION + "x"])
def test_mutable_or_malformed_revisions_are_rejected_in_every_mode(revision):
    from kore.policy.model_spec import FloatingRevisionError, model_identity_for_config
    from kore.policy.sft import sft_config_from_dict

    config, _ = sft_config_from_dict(dict(_sft_config(), model_revision=revision))
    for mode in ({}, {"KORE_MODEL_IDENTITY_MODE": "production"}):
        with pytest.raises(FloatingRevisionError):
            model_identity_for_config(config, stage="sft",
                                      environ={"HF_HUB_OFFLINE": "1", **mode})


def test_unpinned_config_is_fatal_in_production_and_a_warning_in_development():
    from kore.policy.model_spec import UnpinnedModelError, model_identity_for_config
    from kore.policy.sft import sft_config_from_dict

    payload = {k: v for k, v in _sft_config().items() if k != "model_revision"}
    config, _ = sft_config_from_dict(payload)

    development = model_identity_for_config(
        config, stage="sft", environ={"HF_HUB_OFFLINE": "1"})
    assert development.revision is None and development.load_kwargs == {}
    assert development.notes, "an unpinned production model must be reported"

    with pytest.raises(UnpinnedModelError):
        model_identity_for_config(
            config, stage="sft",
            environ={"HF_HUB_OFFLINE": "1",
                     "KORE_MODEL_IDENTITY_MODE": "production"})


def test_uncached_commit_degrades_offline_in_dev_and_fails_in_production():
    from kore.policy.model_spec import ModelSpecError, model_identity_for_config
    from kore.policy.sft import sft_config_from_dict

    config, _ = sft_config_from_dict(dict(_sft_config(), model_revision="0" * 40))

    development = model_identity_for_config(
        config, stage="sft", environ={"HF_HUB_OFFLINE": "1"})
    assert development.pin_load is False, (
        "pinning an uncached commit under HF_HUB_OFFLINE=1 must degrade, not "
        "turn into an unexplained load failure"
    )
    assert any("not present in any local" in note for note in development.notes)

    with pytest.raises(ModelSpecError):
        model_identity_for_config(
            config, stage="sft",
            environ={"HF_HUB_OFFLINE": "1",
                     "KORE_MODEL_IDENTITY_MODE": "production"})


# --------------------------------------------------------------------------- #
# 4/5. midtrain -> SFT handoff
# --------------------------------------------------------------------------- #
def _write_fake_checkpoint(directory: Path, *, layers: int = 2, hidden: int = 8):
    """Minimal but structurally valid HF checkpoint (header-only inspection passes)."""
    import struct

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": hidden,
        "intermediate_size": hidden * 2,
        "num_hidden_layers": layers,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": hidden // 2,
        "vocab_size": 16,
        "max_position_embeddings": 128,
        "tie_word_embeddings": True,
    }))
    (directory / "generation_config.json").write_text("{}")
    (directory / "tokenizer_config.json").write_text("{}")

    shapes = {"model.embed_tokens.weight": (16, hidden), "model.norm.weight": (hidden,)}
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        shapes.update({
            f"{prefix}.self_attn.q_proj.weight": (hidden, hidden),
            f"{prefix}.self_attn.k_proj.weight": (hidden // 2, hidden),
            f"{prefix}.self_attn.v_proj.weight": (hidden // 2, hidden),
            f"{prefix}.self_attn.o_proj.weight": (hidden, hidden),
            f"{prefix}.mlp.gate_proj.weight": (hidden * 2, hidden),
            f"{prefix}.mlp.up_proj.weight": (hidden * 2, hidden),
            f"{prefix}.mlp.down_proj.weight": (hidden, hidden * 2),
            f"{prefix}.input_layernorm.weight": (hidden,),
            f"{prefix}.post_attention_layernorm.weight": (hidden,),
        })
    header, offset = {}, 0
    for name, shape in shapes.items():
        size = 1
        for dim in shape:
            size *= dim
        header[name] = {"dtype": "BF16", "shape": list(shape),
                        "data_offsets": [offset, offset + size * 2]}
        offset += size * 2
    blob = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\0" * offset)
    return directory


def test_latest_checkpoint_finds_a_complete_checkpoint(tmp_path):
    from kore.policy.configs import latest_checkpoint

    assert latest_checkpoint(tmp_path) is None
    for step in (100, 300, 200):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}))
    assert latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-300")


@pytest.mark.xfail(
    reason=(
        "BLOCKER 3: latest_checkpoint() only inspects the highest-numbered "
        "checkpoint dir. A crash while writing checkpoint-N leaves that dir "
        "without trainer_state.json, so the function returns None and the run "
        "silently restarts from step 0 even though a complete older checkpoint "
        "is sitting right next to it. It should fall back to the newest "
        "checkpoint that actually has trainer_state.json."
    ),
)
def test_latest_checkpoint_falls_back_past_a_half_written_checkpoint(tmp_path):
    from kore.policy.configs import latest_checkpoint

    complete = tmp_path / "checkpoint-100"
    complete.mkdir()
    (complete / "trainer_state.json").write_text(json.dumps({"global_step": 100}))
    (tmp_path / "checkpoint-200").mkdir()      # crashed mid-save: no trainer_state
    assert latest_checkpoint(tmp_path) == str(complete)


def test_directory_handoff_ignores_a_stale_hub_revision(tmp_path):
    """The campaign overrides ``model_id`` with the midtrain output dir but leaves
    ``model_revision`` in place; the base model's commit must not be attached to
    the midtrain checkpoint, nor forwarded to ``from_pretrained``."""
    from kore.policy.model_spec import model_identity_for_config
    from kore.policy.sft import sft_config_from_dict

    checkpoint = _write_fake_checkpoint(tmp_path / "runs" / "midtrain_14b_full")
    config, _ = sft_config_from_dict(
        dict(_sft_config(), model_id=str(checkpoint)))
    identity = model_identity_for_config(
        config, stage="sft", environ={"HF_HUB_OFFLINE": "1"})

    assert identity.revision is None
    assert identity.pin_load is False
    assert identity.load_kwargs == {}, (
        "a local directory has no Hub commit; passing revision= would be a lie"
    )
    assert identity.local_path == str(checkpoint.resolve())
    assert identity.inspection is not None, "handoff architecture was not verified"
    assert identity.inspection.architecture.decoder_class == "Qwen3DecoderLayer"
    assert any("IGNORED" in note for note in identity.notes)


def test_directory_handoff_rejects_a_non_checkpoint_in_production(tmp_path):
    from kore.policy.model_spec import ModelSpecError, model_identity_for_config
    from kore.policy.sft import sft_config_from_dict

    empty = tmp_path / "not_a_checkpoint"
    empty.mkdir()
    config, _ = sft_config_from_dict(dict(_sft_config(), model_id=str(empty)))

    development = model_identity_for_config(
        config, stage="sft", environ={"HF_HUB_OFFLINE": "1"})
    assert development.inspection is None and development.verify == "none"

    with pytest.raises(ModelSpecError):
        model_identity_for_config(
            config, stage="sft",
            environ={"HF_HUB_OFFLINE": "1",
                     "KORE_MODEL_IDENTITY_MODE": "production"})


def test_fsdp_kwargs_wrap_the_qwen3_decoder_and_consolidate_the_handoff():
    from kore.policy.configs import build_fsdp_kwargs, fsdp_enabled
    from kore.policy.sft import sft_config_from_dict

    config, _ = sft_config_from_dict(_sft_config())
    assert fsdp_enabled(config)
    kwargs = build_fsdp_kwargs(config)
    assert kwargs["fsdp"] == "full_shard auto_wrap"
    fsdp_config = kwargs["fsdp_config"]
    assert fsdp_config["transformer_layer_cls_to_wrap"] == ["Qwen3DecoderLayer"]
    # A sharded state dict is only reloadable under an identical mesh, which the
    # cross-stage handoff is not.
    assert fsdp_config["state_dict_type"] == "FULL_STATE_DICT"


@pytest.mark.shell
def test_launcher_dry_run_emits_the_expected_accelerate_command():
    launcher = REPO / "scripts" / "launch_distributed.sh"
    result = subprocess.run(
        ["bash", str(launcher), "sft", "configs/sft_14b_full.json", "--dry-run"],
        cwd=REPO, capture_output=True, text=True, check=True,
        env={**os.environ, "GPU_IDS": "6,7"})
    printed = result.stdout.strip()
    for fragment in ("accelerate launch", "--config_file",
                     "configs/accelerate_fsdp.yaml", "--gpu_ids 6,7",
                     "--num_processes 2", "-m kore.policy.sft",
                     "configs/sft_14b_full.json"):
        assert fragment in printed, f"{fragment!r} missing from: {printed}"


@pytest.mark.shell
@pytest.mark.xfail(
    reason=(
        "GAP: --dry-run returns before the launcher validates that the config "
        "and accelerate YAML exist, so the CI syntax check cannot catch a "
        "mistyped config path."
    ),
)
def test_launcher_dry_run_rejects_a_missing_config():
    launcher = REPO / "scripts" / "launch_distributed.sh"
    result = subprocess.run(
        ["bash", str(launcher), "sft", "configs/does_not_exist.json", "--dry-run"],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode != 0


# --------------------------------------------------------------------------- #
# 6. Checkpoint retention and run-length arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    reason=(
        "BLOCKER 2: sft.py hardcodes save_total_limit=1 and SFTConfig has no "
        "such field, so a launch config cannot even express the >= 2 retention "
        "that MidTrainConfig has and configs/midtrain_14b_full.json sets. "
        "Measured on gfx950: one 14B SFT checkpoint is 221 GB."
    ),
)
def test_sft_launch_config_can_request_more_than_one_checkpoint():
    from kore.policy.configs import MidTrainConfig
    from kore.policy.sft import sft_config_from_dict

    # The comparison stage already has the knob, and ships it set to 2.
    assert MidTrainConfig().save_total_limit == 1
    midtrain = json.loads((REPO / "configs" / "midtrain_14b_full.json").read_text())
    assert midtrain["save_total_limit"] == 2

    config, _ = sft_config_from_dict(dict(_sft_config(), save_total_limit=2))
    assert getattr(config, "save_total_limit") == 2


def test_sft_save_total_limit_is_currently_hardcoded():
    """Pins today's behaviour so the blocker above cannot be quietly re-broken."""
    source = (REPO / "kore" / "policy" / "sft.py").read_text()
    assert "save_total_limit=1," in source, (
        "sft.py no longer hardcodes save_total_limit=1 - update "
        "docs/SFT_READINESS.md and the xfail above"
    )


def _sft_step_plan(rows_trained: int, config: dict, world_size: int) -> dict:
    """Reproduce the HF Trainer's step arithmetic for a distributed SFT run."""
    microbatch = config["per_device_train_batch_size"]
    accumulation = config["gradient_accumulation_steps"]
    per_rank = -(-rows_trained // world_size)          # DistributedSampler pads up
    micro_batches = -(-per_rank // microbatch)
    steps_per_epoch = max(micro_batches // accumulation, 1)
    return {
        "effective_batch": microbatch * accumulation * world_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": int(steps_per_epoch * config["num_train_epochs"]),
    }


def test_full_run_step_count_is_what_the_readiness_plan_assumes():
    """A config edit that changes the run length must change this number too."""
    plan = _sft_step_plan(ROWS_TRAINED, _sft_config(), world_size=8)
    assert plan["effective_batch"] == 128
    assert plan["steps_per_epoch"] == 533
    assert plan["total_steps"] == 1599

    save_steps = 200  # SFTConfig default; the shipped launch JSON does not override
    from kore.policy.configs import SFTConfig

    assert SFTConfig().save_steps == save_steps
    # 7 periodic saves plus the Trainer's end-of-training checkpoint, each 221 GB.
    assert plan["total_steps"] // save_steps == 7


def test_overlong_filter_only_engages_under_completion_only_loss():
    """``_filter_overlong`` drops 5.8% of the corpus; it must be a no-op when the
    mask is off, and must be deterministic so every FSDP rank shards identically."""
    from kore.policy.sft import _filter_overlong

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(len(messages[0]["content"])))

    dataset = [{"messages": [{"role": "user", "content": "x" * n}]}
               for n in (4, 40, 4)]
    kept, dropped = _filter_overlong(dataset, _Tokenizer(), 10)
    assert dropped == 1 and len(kept) == 2
    again, dropped_again = _filter_overlong(dataset, _Tokenizer(), 10)
    assert [r["messages"] for r in again] == [r["messages"] for r in kept]
    assert dropped_again == dropped

    unchanged, none_dropped = _filter_overlong(dataset, _Tokenizer(), 0)
    assert unchanged is dataset and none_dropped == 0


def test_repair_upweighting_duplicates_by_provenance_not_by_bucket(tmp_path):
    """``kernel_repair_opt`` tags repairs AND optimization wins; only true repair
    turns may be up-weighted or the trained mixture silently shifts."""
    from kore.policy.sft import load_sft_dataset

    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"messages": [{"role": "user", "content": "a"},
                      {"role": "assistant", "content": "A"}],
         "_source": "kernel_repair_opt", "_provenance": {"kind": "repair"}},
        {"messages": [{"role": "user", "content": "b"},
                      {"role": "assistant", "content": "B"}],
         "_source": "kernel_repair_opt", "_provenance": {"kind": "win"}},
        {"messages": [{"role": "user", "content": "c"},
                      {"role": "assistant", "content": "C"}],
         "_source": "general_chat"},
    ]) + "\n")

    assert len(load_sft_dataset(path, repair_weight=1.0)) == 3
    assert len(load_sft_dataset(path, repair_weight=2.0)) == 4
    assert len(load_sft_dataset(path, repair_weight=3.0)) == 5


def test_packing_stays_off_because_the_rocm_stack_has_no_flash_attention():
    """TRL's bfd packing needs a flash-attn backend for the block-diagonal mask;
    on SDPA it silently cross-contaminates packed documents."""
    from kore.policy.configs import SFTConfig, preferred_attn_impl

    assert SFTConfig().packing is False
    assert _sft_config().get("packing", False) is False
    assert preferred_attn_impl() in {"sdpa", "flash_attention_2"}
    source = (REPO / "kore" / "policy" / "sft.py").read_text()
    assert 'if _packing and _attn_impl != "flash_attention_2"' in source


if __name__ == "__main__":  # pragma: no cover - convenience for ad-hoc runs
    sys.exit(pytest.main([__file__, "-v"]))
