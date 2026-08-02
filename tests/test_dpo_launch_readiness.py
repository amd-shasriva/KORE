"""CPU-runnable launch-readiness regressions for Stage-2 DPO.

Companion to ``tests/test_sft_launch_readiness.py``, locking down what was
verified end to end on 2x MI350X (gfx950) for ``docs/DPO_READINESS.md``: the
96,675-pair preference corpus, the preference-weighting multiplier that sets the
real run length, the prompt/completion split TRL derives by subtraction, the
truncation guard, checkpoint retention and discovery, and model identity for
BOTH the policy and the frozen reference.

Nothing here needs a GPU. Tests that need the real Qwen3-14B tokenizer skip when
that exact commit is not in the local Hugging Face cache, and tests that read the
preference corpus skip when ``data/b05factory/dpo/pairs.jsonl`` is absent, so a
bare checkout still collects and passes.

``xfail`` markers ARE the blocker list, per this repo's convention: a marked test
asserts the behaviour the readiness review concluded is correct, against code
that does not implement it yet, and turns green when the patch lands.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DPO_CONFIG_PATH = REPO / "configs" / "dpo_14b_full.json"
PAIRS = REPO / "data" / "b05factory" / "dpo" / "pairs.jsonl"
DPO_SBATCH = REPO / "scripts" / "spur_dpo_1node.sbatch"

QWEN3_14B = "Qwen/Qwen3-14B"
PINNED_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"

#: Measured from the corpus on the cluster and locally on 2026-08-01.
DPO_ROWS = 96_675
#: What ``apply_pref_weights`` turns those rows into at its default seed. The
#: 1.728x multiplier is the single biggest input to the run length, and the
#: launcher's step/wall-time budget was written as though it were 1.0x.
DPO_EFFECTIVE_ROWS = 167_054
#: Optimizer steps for one epoch at 2 x 8 x 8 ranks over the WEIGHTED rows.
DPO_STEPS_WORLD8 = 1_305
#: Longest prompt+completion actually present, tokenized (2,500-pair sample).
DPO_LONGEST_SEQUENCE = 4_802


def _dpo_config() -> dict:
    return json.loads(DPO_CONFIG_PATH.read_text())


def _local_snapshot():
    from kore.policy.model_spec import resolve_local_snapshot

    return resolve_local_snapshot(QWEN3_14B, PINNED_REVISION)


_NEEDS_TOKENIZER = pytest.mark.skipif(
    _local_snapshot() is None,
    reason=(
        f"{QWEN3_14B}@{PINNED_REVISION[:12]} is not in the local Hugging Face "
        "cache; these tests read the real chat template rather than a fixture."
    ),
)
_NEEDS_CORPUS = pytest.mark.skipif(
    not PAIRS.is_file(),
    reason="DPO preference corpus (data/b05factory/dpo/pairs.jsonl) is absent",
)


def _corpus_rows(limit: int | None = None):
    with PAIRS.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# 1. The preference corpus
# --------------------------------------------------------------------------- #
@_NEEDS_CORPUS
def test_corpus_rows_are_the_conversational_shape_trl_consumes():
    """``prompt`` is a chat list, ``chosen``/``rejected`` are ONE assistant turn.

    TRL's conversational DPO path applies the chat template per column and then
    derives the completion by subtracting the tokenized prompt. A row whose
    ``chosen`` is a bare string, or holds a non-assistant role, silently changes
    what the loss covers.
    """
    checked = 0
    for row in _corpus_rows(limit=2000):
        assert {"prompt", "chosen", "rejected"} <= set(row)
        assert isinstance(row["prompt"], list) and row["prompt"]
        assert row["prompt"][-1]["role"] == "user", "prompt must end on the user turn"
        for side in ("chosen", "rejected"):
            turns = row[side]
            assert isinstance(turns, list) and len(turns) == 1, f"{side} is not one turn"
            assert turns[0]["role"] == "assistant"
            assert isinstance(turns[0]["content"], str) and turns[0]["content"].strip()
        checked += 1
    assert checked, "no rows were checked"


@pytest.mark.release
@_NEEDS_CORPUS
def test_corpus_row_count_and_integrity():
    """Full-corpus count: the number the run-length budget depends on."""
    rows = degenerate = 0
    for row in _corpus_rows():
        rows += 1
        if row["chosen"] == row["rejected"]:
            degenerate += 1
    assert rows == DPO_ROWS
    # Two identical-pair rows contribute log(2) and a mathematically ZERO
    # gradient (the chosen and rejected log-probs are the same tensor). Harmless
    # but wasted compute; assert the count so a regression that multiplies them
    # is visible.
    assert degenerate == 2


@_NEEDS_CORPUS
def test_launcher_pair_count_matches_the_corpus():
    """The sbatch header quotes the pair count; it must not drift from the file."""
    assert f"{DPO_ROWS:,}" in DPO_SBATCH.read_text()


# --------------------------------------------------------------------------- #
# 2. Preference weighting -- what actually sets the run length
# --------------------------------------------------------------------------- #
def test_preference_weighting_multiplicity_is_deterministic_and_bounded():
    """``apply_pref_weights`` must be identical on every FSDP rank.

    Ranks that disagree about the row list shard differently and train on
    mismatched data under one optimizer. The seeded sub-1.0 sampling is the only
    stochastic part, so pin its determinism.
    """
    from kore.policy.dpo import apply_pref_weights

    rows = [
        {"prompt": "p", "chosen": "c", "rejected": "r", "weight": 1.0},
        {"prompt": "p", "chosen": "c", "rejected": "r", "weight": 8.0},
        {"prompt": "p", "chosen": "c", "rejected": "r", "weight": 0.25},
        {"prompt": "p", "chosen": "c", "rejected": "r", "weight": 0.0},
    ]
    first = apply_pref_weights(rows, enabled=True, seed=0)
    second = apply_pref_weights(rows, enabled=True, seed=0)
    assert len(first) == len(second), "weighting is not reproducible across ranks"
    # weight<=0 is dropped, weight>=1 is duplicated round(w) times.
    assert len(apply_pref_weights(rows[:2], enabled=True, seed=0)) == 9
    # Disabled -> exactly one copy of every row, extras stripped to TRL's schema.
    plain = apply_pref_weights(rows, enabled=False)
    assert len(plain) == len(rows)
    assert set(plain[0]) == {"prompt", "chosen", "rejected"}, (
        "extra columns would reach trl's Dataset and change its schema"
    )


@pytest.mark.release
@_NEEDS_CORPUS
def test_preference_weighting_multiplies_the_real_corpus_by_1_73x():
    """The multiplier is the run length. 96,675 pairs is NOT what trains."""
    from kore.policy.dpo import apply_pref_weights, load_preference_jsonl

    rows = load_preference_jsonl(str(PAIRS))
    assert len(rows) == DPO_ROWS
    weighted = apply_pref_weights(rows, enabled=True, seed=0)
    assert len(weighted) == DPO_EFFECTIVE_ROWS
    assert len(weighted) / len(rows) > 1.7


def _dpo_step_plan(rows: int, config: dict, world_size: int) -> dict:
    """Reproduce the HF Trainer's step arithmetic for a distributed DPO run."""
    microbatch = config["per_device_train_batch_size"]
    accumulation = config["gradient_accumulation_steps"]
    per_rank = -(-rows // world_size)
    micro_batches = -(-per_rank // microbatch)
    steps_per_epoch = max(micro_batches // accumulation, 1)
    return {
        "effective_batch": microbatch * accumulation * world_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": int(steps_per_epoch * config["num_train_epochs"]),
    }


def test_full_run_step_count_uses_the_WEIGHTED_row_count():
    """A config edit that changes the run length must change this number too."""
    config = _dpo_config()
    plan = _dpo_step_plan(DPO_EFFECTIVE_ROWS, config, world_size=8)
    assert plan["effective_batch"] == 128
    assert plan["total_steps"] == DPO_STEPS_WORLD8

    unweighted = _dpo_step_plan(DPO_ROWS, config, world_size=8)
    assert unweighted["total_steps"] == 755, (
        "755 is the count the launcher header quoted; it ignores the weighting"
    )


def test_launcher_step_budget_accounts_for_preference_weighting():
    """The sbatch header's step/wall-time budget must be the WEIGHTED one.

    The header sized the allocation off 96,675 pairs -> ~755 steps, but
    ``apply_pref_weights`` is on by default and makes it 1,305. A 73% under-
    estimate of a multi-hour stage is how a run silently fails to finish inside
    its allocation.
    """
    header = DPO_SBATCH.read_text().split("#SBATCH")[0]
    assert f"{DPO_STEPS_WORLD8:,}" in header or str(DPO_STEPS_WORLD8) in header, (
        "spur_dpo_1node.sbatch does not quote the weighted step count"
    )
    assert "apply_pref_weights" in header, (
        "the header must name the multiplier that makes the step count what it is"
    )
    assert "~755 optimizer steps" not in header, (
        "the unweighted 755-step budget is still presented as the run length"
    )


# --------------------------------------------------------------------------- #
# 3. What the DPO loss actually covers
# --------------------------------------------------------------------------- #
@_NEEDS_TOKENIZER
@_NEEDS_CORPUS
def test_prompt_prefix_is_a_strict_prefix_of_prompt_plus_completion():
    """TRL derives the completion by SUBTRACTION, so the prefix must match exactly.

    ``chosen_ids = prompt_chosen_ids[len(prompt_ids):]``. If Qwen3's
    ``add_generation_prompt=True`` rendering is not a byte-prefix of the full
    rendering, TRL only logs a warning and then slices at the wrong offset --
    training on a completion that starts mid-token-sequence.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_14B, revision=PINNED_REVISION)
    checked = 0
    for row in _corpus_rows(limit=32):
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=True,
            return_dict=False)
        for side in ("chosen", "rejected"):
            full = tokenizer.apply_chat_template(
                row["prompt"] + row[side], tokenize=True, return_dict=True)["input_ids"]
            assert full[: len(prompt_ids)] == prompt_ids, (
                f"{side}: tokenized prompt is not a prefix of prompt+completion; "
                "TRL would mis-slice the completion"
            )
            assert len(full) > len(prompt_ids), f"{side} completion is empty"
        checked += 1
    assert checked, "no rows were checked"


@_NEEDS_TOKENIZER
@_NEEDS_CORPUS
def test_only_completion_tokens_survive_into_the_dpo_loss():
    """On real token ids: the completion trains, the prompt is masked out.

    Reproduces TRL 0.29.1's pipeline exactly -- ``DataCollatorForPreference``
    builds ``completion_mask``, then ``_compute_loss`` shifts it by one and
    zeroes every non-completion per-token log-prob.
    """
    from transformers import AutoTokenizer
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_14B, revision=PINNED_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    row = next(_corpus_rows(limit=1))

    prompt_ids = tokenizer.apply_chat_template(
        row["prompt"], add_generation_prompt=True, tokenize=True, return_dict=False)
    encoded = {"prompt_ids": prompt_ids}
    for side in ("chosen", "rejected"):
        full = tokenizer.apply_chat_template(
            row["prompt"] + row[side], tokenize=True, return_dict=True)["input_ids"]
        encoded[f"{side}_ids"] = full[len(prompt_ids):]

    batch = DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)([encoded])
    shift_mask = batch["completion_mask"][..., 1:]
    shift_ids = batch["input_ids"][..., 1:]

    system_text = row["prompt"][0]["content"][:60]
    user_text = row["prompt"][-1]["content"][:60]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for index, name in enumerate(("chosen", "rejected")):
        in_loss = shift_ids[index][shift_mask[index] == 1]
        masked = shift_ids[index][shift_mask[index] == 0]
        assert len(in_loss) and len(masked), f"{name}: degenerate mask"
        learned = tokenizer.decode(in_loss)
        dropped = tokenizer.decode(masked)
        assert system_text not in learned, f"{name}: system prompt leaked into the loss"
        assert user_text not in learned, f"{name}: user prompt leaked into the loss"
        assert im_end in in_loss.tolist(), f"{name}: the stop token must be in the loss"
        assert "<|im_start|>assistant" in dropped, f"{name}: assistant header not masked"


@_NEEDS_TOKENIZER
@_NEEDS_CORPUS
def test_keep_end_truncation_is_what_preserves_the_kernel_tail():
    """``truncation_mode`` is load-bearing, not cosmetic.

    The prompt is ~922 tokens of fixed system+task preamble. Under TRL's
    ``keep_start`` default a tight ``max_length`` spends the whole budget on the
    prompt and keeps ZERO completion tokens -- a batch with an empty loss mask.
    ``keep_end`` keeps the kernel body and its stop token.
    """
    from transformers import AutoTokenizer
    from trl.trainer.dpo_trainer import DataCollatorForPreference
    from trl.trainer.utils import flush_left, flush_right

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_14B, revision=PINNED_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    row = next(_corpus_rows(limit=1))
    prompt_ids = tokenizer.apply_chat_template(
        row["prompt"], add_generation_prompt=True, tokenize=True, return_dict=False)
    encoded = {"prompt_ids": prompt_ids}
    for side in ("chosen", "rejected"):
        full = tokenizer.apply_chat_template(
            row["prompt"] + row[side], tokenize=True, return_dict=True)["input_ids"]
        encoded[f"{side}_ids"] = full[len(prompt_ids):]
    batch = DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)([encoded])

    max_length = 512  # deliberately below the ~922-token prompt
    def truncate(mode):
        ids, attn, comp = (batch["input_ids"], batch["attention_mask"],
                           batch["completion_mask"])
        if mode == "keep_start":
            ids, attn, comp = ids[:, :max_length], attn[:, :max_length], comp[:, :max_length]
            attn, ids, comp = flush_right(attn, ids, comp)
        else:
            ids, attn, comp = ids[:, -max_length:], attn[:, -max_length:], comp[:, -max_length:]
            attn, ids, comp = flush_left(attn, ids, comp)
        return int(comp[0].sum())

    assert truncate("keep_start") == 0, (
        "expected TRL's default to destroy the completion on this corpus"
    )
    assert truncate("keep_end") > 0, "keep_end must preserve the completion tail"


def test_truncation_mode_defaults_to_keep_end_even_when_unset():
    """A config that omits ``truncation_mode`` must not silently get keep_start."""
    from kore.policy.dpo import build_trl_dpo_kwargs, dpo_config_from_dict

    payload = {k: v for k, v in _dpo_config().items() if k != "truncation_mode"}
    kwargs = build_trl_dpo_kwargs(dpo_config_from_dict(payload))
    assert kwargs["truncation_mode"] == "keep_end"
    # and an explicit override is still honoured
    explicit = dpo_config_from_dict(dict(_dpo_config(), truncation_mode="keep_start"))
    assert build_trl_dpo_kwargs(explicit)["truncation_mode"] == "keep_start"


@pytest.mark.release
@_NEEDS_TOKENIZER
@_NEEDS_CORPUS
def test_max_length_is_never_reached_by_this_corpus():
    """``max_length: 16384`` is 3.4x the longest real pair.

    Recorded so that shrinking it becomes a deliberate, measured decision rather
    than an accident: nothing in this corpus is truncated today, so the
    ``keep_end`` guard above is insurance, not an active code path.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_14B, revision=PINNED_REVISION)
    longest = 0
    for row in _corpus_rows(limit=500):
        for side in ("chosen", "rejected"):
            full = tokenizer.apply_chat_template(
                row["prompt"] + row[side], tokenize=True, return_dict=True)["input_ids"]
            longest = max(longest, len(full))
    assert longest <= DPO_LONGEST_SEQUENCE
    assert _dpo_config()["max_length"] >= 2 * longest


# --------------------------------------------------------------------------- #
# 4. TRL kwargs: the arity guard and the retention knob
# --------------------------------------------------------------------------- #
def test_composite_loss_and_weights_always_have_matching_arity():
    """TRL raises if ``loss_weights`` and a composite ``loss_type`` disagree.

    Iterative DPO rewrites ``loss_type`` per round (sigmoid+sft -> ipo+sft), so a
    round that changed only one of the pair would hard-stop the stage.
    """
    from kore.policy.dpo import build_trl_dpo_kwargs, dpo_config_from_dict

    config = dpo_config_from_dict(_dpo_config())
    kwargs = build_trl_dpo_kwargs(config)
    assert kwargs["loss_type"] == ["sigmoid", "sft"]
    assert len(kwargs["loss_weights"]) == len(kwargs["loss_type"])

    # composite loss, mismatched weights -> reconciled, never raised
    config.loss_type, config.loss_weights = ["ipo", "sft"], [1.0]
    reconciled = build_trl_dpo_kwargs(config)
    assert reconciled["loss_weights"] == [1.0, 1.0]

    # scalar loss must carry NO weights at all
    config.loss_type, config.loss_weights = "sigmoid", [1.0, 1.0]
    scalar = build_trl_dpo_kwargs(config)
    assert scalar["loss_type"] == "sigmoid"
    assert "loss_weights" not in scalar


def test_dpo_save_total_limit_is_read_from_config_and_defaults_to_two():
    """One retained checkpoint is not crash-safe; the Trainer rotates around a save."""
    from kore.policy.configs import DPOConfig
    from kore.policy.dpo import build_trl_dpo_kwargs, dpo_config_from_dict

    assert DPOConfig().save_total_limit >= 2
    kwargs = build_trl_dpo_kwargs(dpo_config_from_dict(_dpo_config()))
    assert kwargs["save_total_limit"] >= 2
    explicit = dpo_config_from_dict(dict(_dpo_config(), save_total_limit=3))
    assert build_trl_dpo_kwargs(explicit)["save_total_limit"] == 3


def test_shipped_dpo_config_states_its_retention_explicitly():
    """Repo convention (MidTrainConfig's note, followed by midtrain and SFT):
    every shipped launch config sets ``save_total_limit`` >= 2 EXPLICITLY rather
    than inheriting a default that a later dataclass edit could lower."""
    shipped = _dpo_config()
    assert "save_total_limit" in shipped, (
        "configs/dpo_14b_full.json relies on the dataclass default"
    )
    assert shipped["save_total_limit"] >= 2


def test_group_by_length_stays_off_for_preference_rows():
    """HF's LengthGroupedSampler needs an ``input_ids`` column; DPO rows have none."""
    from kore.policy.dpo import build_trl_dpo_kwargs, dpo_config_from_dict

    kwargs = build_trl_dpo_kwargs(dpo_config_from_dict(_dpo_config()))
    assert kwargs["group_by_length"] is False


# --------------------------------------------------------------------------- #
# 5. Model identity: policy AND frozen reference
# --------------------------------------------------------------------------- #
def test_shipped_config_parses_and_carries_an_immutable_revision():
    from kore.policy.dpo import dpo_config_from_dict
    from kore.policy.model_spec import validate_pinned_revision

    payload = _dpo_config()
    assert validate_pinned_revision(payload["model_revision"]) == PINNED_REVISION
    config = dpo_config_from_dict(dict(payload))
    assert config.model_id == QWEN3_14B
    assert config.use_lora is False and config.distributed is True
    assert config.beta == 0.1


def test_reference_identity_reuses_the_policy_identity_when_they_are_the_same():
    """One checkpoint must not be able to disagree with itself about its own pin."""
    from kore.policy.dpo import _reference_identity, dpo_config_from_dict
    from kore.policy.model_spec import model_identity_for_config

    config = dpo_config_from_dict(_dpo_config())
    assert config.ref_model_id is None, "shipped config pins no separate reference"
    identity = model_identity_for_config(
        config, stage="dpo", environ={"HF_HUB_OFFLINE": "1"})
    ref_id, ref_identity = _reference_identity(config, identity)
    assert ref_id == config.model_id
    assert ref_identity is identity


def test_a_distinct_reference_checkpoint_gets_its_own_identity(tmp_path):
    """Iterative DPO points the reference at an earlier round; it must be verified
    independently of the policy rather than inheriting the policy's pin."""
    from tests.test_sft_launch_readiness import _write_fake_checkpoint

    from kore.policy.dpo import _reference_identity, dpo_config_from_dict
    from kore.policy.model_spec import model_identity_for_config

    reference = _write_fake_checkpoint(tmp_path / "runs" / "dpo_round0")
    config = dpo_config_from_dict(dict(_dpo_config(), ref_model_id=str(reference)))
    identity = model_identity_for_config(
        config, stage="dpo", environ={"HF_HUB_OFFLINE": "1"})
    ref_id, ref_identity = _reference_identity(config, identity)
    assert ref_id == str(reference)
    assert ref_identity is not identity
    # A directory has no Hub commit, so the base model's revision must not be
    # laundered onto it.
    assert ref_identity.load_kwargs == {}
    assert ref_identity.inspection is not None


@pytest.mark.parametrize("revision", ["main", "v1.0", "40c0698", "refs/pr/1"])
def test_mutable_or_malformed_revisions_are_rejected(revision):
    from kore.policy.dpo import dpo_config_from_dict
    from kore.policy.model_spec import FloatingRevisionError, model_identity_for_config

    config = dpo_config_from_dict(dict(_dpo_config(), model_revision=revision))
    with pytest.raises(FloatingRevisionError):
        model_identity_for_config(config, stage="dpo",
                                  environ={"HF_HUB_OFFLINE": "1"})


# --------------------------------------------------------------------------- #
# 6. Handoff, resume and the launcher contract
# --------------------------------------------------------------------------- #
def test_resolver_refuses_a_from_stage_that_is_not_a_real_checkpoint(tmp_path):
    """``spur_dpo_1node.sbatch`` defaults FROM_STAGE to ``runs/sft_14b_frontier``.

    Until Stage-1 lands that directory does not exist, and the failure must be a
    clear resolver message in milliseconds, not an 8-rank 14B load.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from spur_resolve_launch_config import resolve

    with pytest.raises(ValueError, match="not a loadable checkpoint"):
        resolve("dpo", _dpo_config(), from_stage="runs/sft_14b_frontier",
                output_dir="runs/dpo_14b_frontier", repo_root=tmp_path)


def test_resolver_rewrites_model_id_and_output_dir_for_a_real_checkpoint(tmp_path):
    from tests.test_sft_launch_readiness import _write_fake_checkpoint

    sys.path.insert(0, str(REPO / "scripts"))
    from spur_resolve_launch_config import resolve

    _write_fake_checkpoint(tmp_path / "runs" / "sft_14b_frontier")
    corpus = tmp_path / "data" / "b05factory" / "dpo"
    corpus.mkdir(parents=True)
    (corpus / "pairs.jsonl").write_text(json.dumps({
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a"}],
        "rejected": [{"role": "assistant", "content": "b"}]}) + "\n")

    resolved, changes = resolve(
        "dpo", _dpo_config(), from_stage="runs/sft_14b_frontier",
        output_dir="runs/dpo_14b_frontier", repo_root=tmp_path)
    assert resolved["model_id"] == "runs/sft_14b_frontier"
    assert resolved["output_dir"] == "runs/dpo_14b_frontier"
    # The reference defaults to model_id, so overriding model_id also anchors the
    # frozen reference to the SFT policy -- the intended recipe.
    assert resolved.get("ref_model_id") is None
    assert any("model_id" in c for c in changes)


def test_dpo_resumes_from_the_newest_complete_checkpoint(tmp_path):
    from kore.policy.configs import latest_checkpoint

    complete = tmp_path / "checkpoint-6"
    complete.mkdir()
    (complete / "trainer_state.json").write_text(json.dumps({"global_step": 6}))
    (tmp_path / "checkpoint-8").mkdir()  # crashed mid-save: no trainer_state yet
    assert latest_checkpoint(tmp_path) == str(complete)
    (tmp_path / "checkpoint-8" / "trainer_state.json").write_text(
        json.dumps({"global_step": 8}))
    assert latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-8")


def test_dpo_wires_latest_checkpoint_into_trainer_train():
    """Resumability needs the discovery call AND the hand-off to trl."""
    source = (REPO / "kore" / "policy" / "dpo.py").read_text()
    assert "latest_checkpoint(config.output_dir)" in source
    assert "trainer.train(resume_from_checkpoint=_resume)" in source


def test_fsdp_kwargs_wrap_the_qwen3_decoder_and_consolidate_the_handoff():
    from kore.policy.configs import build_fsdp_kwargs, fsdp_enabled
    from kore.policy.dpo import dpo_config_from_dict

    config = dpo_config_from_dict(_dpo_config())
    assert fsdp_enabled(config)
    kwargs = build_fsdp_kwargs(config)
    assert kwargs["fsdp"] == "full_shard auto_wrap"
    assert kwargs["fsdp_config"]["transformer_layer_cls_to_wrap"] == ["Qwen3DecoderLayer"]
    assert kwargs["fsdp_config"]["state_dict_type"] == "FULL_STATE_DICT"


# --------------------------------------------------------------------------- #
# 7. Iterative DPO
# --------------------------------------------------------------------------- #
def test_iterative_dpo_refreshes_the_frozen_reference_every_round():
    """Round N must train FROM, and anchor TO, round N-1's checkpoint.

    A reference that stayed pinned to the SFT checkpoint across rounds would make
    every round's KL term measure drift from the same stale policy, which is not
    the iterative-DPO recipe.
    """
    from kore.data.onpolicy import iterative_dpo

    seen = []

    def policy_factory(round_idx, prev_ckpt):
        seen.append(("factory", round_idx, prev_ckpt))

        class _Policy:
            def generate(self, messages):
                return ""

            def close(self):
                return None

        return _Policy()

    def train_fn(rd):
        seen.append(("train", rd.round, rd.ref_model_id))
        return f"ckpt-round-{rd.round}"

    rounds = iterative_dpo(
        tasks=[], rounds=3, policy_factory=policy_factory,
        env_factory=lambda task: None, train_fn=train_fn)

    assert [r.ref_model_id for r in rounds] == [None, "ckpt-round-0", "ckpt-round-1"]
    assert [r.policy_ckpt for r in rounds] == [
        "ckpt-round-0", "ckpt-round-1", "ckpt-round-2"]
    assert ("factory", 2, "ckpt-round-1") in seen


def test_campaign_default_dpo_rounds_matches_what_the_launcher_runs():
    """``run_campaign.py`` defaults to 2 iterative rounds; the sbatch launcher
    runs exactly one non-iterative pass. Whichever is intended, the two must not
    silently disagree about what Stage 2 is."""
    campaign = (REPO / "scripts" / "run_campaign.py").read_text()
    assert '"--dpo-rounds", type=int, default=2' in campaign, (
        "campaign default changed; re-check the launcher agreement"
    )
    header = DPO_SBATCH.read_text().split("#SBATCH")[0]
    assert "round" in header.lower(), (
        "spur_dpo_1node.sbatch runs ONE DPO pass but never says so, while the "
        "campaign's default is 2 iterative on-policy rounds"
    )


def test_every_stage_loader_drops_the_in_config_comment_convention():
    from kore.policy.dpo import dpo_config_from_dict

    noise = {"_comment_anything": "explanatory prose", "_note": 123}
    dpo_config_from_dict({**_dpo_config(), **noise})


if __name__ == "__main__":  # pragma: no cover - convenience for ad-hoc runs
    sys.exit(pytest.main([__file__, "-v"]))
