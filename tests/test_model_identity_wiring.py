"""Model identity + resource preflight wiring for the training entrypoints.

``model_spec`` and ``resources`` were complete instruments that nothing in
training called, so no run pinned a model revision and no capacity claim was
backed by anything. These tests pin the wiring itself:

* a configured revision reaches ``from_pretrained`` on every stage that loads a
  model (midtrain, sft, dpo policy AND dpo frozen reference);
* a mutable ref (branch/tag/short hash) is rejected in BOTH modes, because an
  explicitly configured floating revision is a config defect, not a missing pin;
* production fails closed, with an actionable message, on a missing revision and
  on a pin that cannot be resolved from a local snapshot (the jobs run with
  ``HF_HUB_OFFLINE=1``, so an uncached commit cannot be loaded at all);
* development reports the same facts and proceeds unpinned, so a job that is
  already in flight cannot be broken by resuming into this code;
* the preflight-to-load TOCTOU re-fingerprint fires when a checkpoint file
  changes between verification and load;
* resource preflight is reachable and never blocks a run on an unprofiled
  machine unless the caller asked for production assurance.

CPU-only and offline: transformers/torch/trl are replaced by fakes, and the
Hugging Face cache is a fixture directory, so nothing here loads a model.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import pytest

from kore.policy.configs import DPOConfig, MidTrainConfig, SFTConfig
from kore.policy.model_spec import (
    DEVELOPMENT,
    PRODUCTION,
    VERIFY_FINGERPRINT,
    VERIFY_METADATA,
    VERIFY_NONE,
    FloatingRevisionError,
    ModelSpec,
    ModelSpecError,
    UnpinnedModelError,
    hf_repo_cache_dirname,
    inspect_local_checkpoint,
    model_identity_for_config,
    resolve_identity_mode,
    resolve_local_snapshot,
    resolve_model_identity,
)
from kore.policy.resources import (
    PREFLIGHT_OFF,
    PREFLIGHT_REPORT,
    PREFLIGHT_STRICT,
    ResourcePreflightError,
    build_stage_workload_spec,
    resolve_preflight_mode,
    run_stage_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
OTHER_REVISION = "b" * 40
MODEL_ID = "fixture/tiny-qwen3"


# --------------------------------------------------------------------------- #
# fixtures: a tiny Qwen3-shaped checkpoint inside a Hugging Face cache layout
# --------------------------------------------------------------------------- #
def _tiny_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        "model.embed_tokens.weight": (16, 4),
        "model.norm.weight": (4,),
        "lm_head.weight": (16, 4),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        shapes.update({
            f"{prefix}.self_attn.q_proj.weight": (4, 4),
            f"{prefix}.self_attn.k_proj.weight": (2, 4),
            f"{prefix}.self_attn.v_proj.weight": (2, 4),
            f"{prefix}.self_attn.o_proj.weight": (4, 4),
            f"{prefix}.mlp.gate_proj.weight": (8, 4),
            f"{prefix}.mlp.up_proj.weight": (8, 4),
            f"{prefix}.mlp.down_proj.weight": (4, 8),
            f"{prefix}.input_layernorm.weight": (4,),
            f"{prefix}.post_attention_layernorm.weight": (4,),
        })
    return shapes


def _write_safetensors(path: Path, shapes: dict[str, tuple[int, ...]]) -> int:
    header: dict[str, dict] = {}
    offset = 0
    for name, shape in sorted(shapes.items()):
        size = 2 * math.prod(shape)
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(offset))
    return offset


def _write_checkpoint(root: Path) -> None:
    """Write a minimal but structurally valid Qwen3 checkpoint."""
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 16,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
    }), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 128}), encoding="utf-8")
    (root / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "model": {}}), encoding="utf-8")
    (root / "generation_config.json").write_text(
        json.dumps({"do_sample": False}), encoding="utf-8")
    shapes = _tiny_shapes()
    names = sorted(shapes)
    split = len(names) // 2
    shards = {
        "model-00001-of-00002.safetensors": {n: shapes[n] for n in names[:split]},
        "model-00002-of-00002.safetensors": {n: shapes[n] for n in names[split:]},
    }
    total = 0
    weight_map: dict[str, str] = {}
    for shard, shard_shapes in shards.items():
        total += _write_safetensors(root / shard, shard_shapes)
        weight_map.update({name: shard for name in shard_shapes})
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map},
        sort_keys=True), encoding="utf-8")


@pytest.fixture
def hf_cache(tmp_path):
    """A Hugging Face hub cache holding exactly one snapshot of ``MODEL_ID``.

    Mirrors the real layout the offline training jobs read, including the
    ``snapshots/<commit>`` directory keyed by the immutable commit.
    """
    root = tmp_path / "hub"
    snapshot = root / hf_repo_cache_dirname(MODEL_ID) / "snapshots" / REVISION
    _write_checkpoint(snapshot)
    env = {
        "HF_HUB_CACHE": str(root),
        "HF_HUB_OFFLINE": "1",
        "KORE_RESOURCE_PREFLIGHT": PREFLIGHT_OFF,
    }
    return types.SimpleNamespace(root=root, snapshot=snapshot, env=env)


@pytest.fixture
def symlinked_cache(tmp_path):
    """A snapshot whose files are symlinks into a sibling ``blobs/`` directory.

    This is what ``huggingface_hub`` actually writes, and resolving those links
    puts every shard outside the snapshot - so a containment check that resolves
    symlinks rejects every real offline checkpoint.
    """
    real = tmp_path / "real"
    _write_checkpoint(real)
    blobs = tmp_path / "hub" / hf_repo_cache_dirname(MODEL_ID) / "blobs"
    snapshot = (
        tmp_path / "hub" / hf_repo_cache_dirname(MODEL_ID) / "snapshots" / REVISION
    )
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for source in sorted(real.iterdir()):
        blob = blobs / f"blob-{source.name}"
        blob.write_bytes(source.read_bytes())
        (snapshot / source.name).symlink_to(Path("../../blobs") / blob.name)
    return snapshot


@pytest.fixture
def fake_training_stack(monkeypatch):
    """Replace transformers/trl/torch/datasets/peft with recording fakes.

    Records every ``from_pretrained`` call so a test can assert exactly which
    revision each stage passed for each model it loads.
    """
    calls: list[tuple[str, str, dict]] = []

    class _Tok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        chat_template = None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("tokenizer", str(model_id), kwargs))
            return cls()

        def save_pretrained(self, path):
            calls.append(("tokenizer-save", str(path), {}))

        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3]

    class _Config:
        use_cache = True

    class _Model:
        def __init__(self):
            self.config = _Config()

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", str(model_id), kwargs))
            return cls()

    class _Trainer:
        def __init__(self, **kwargs):
            self.model = _Model()
            self.kwargs = kwargs

        def train(self, resume_from_checkpoint=None):
            calls.append(("train", str(resume_from_checkpoint), {}))

        def save_model(self, output_dir):
            calls.append(("save", str(output_dir), {}))

    class _TrainerCallback:
        pass

    class _Dataset:
        def __init__(self, rows):
            self.rows = list(rows)

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

        def __iter__(self):
            return iter(self.rows)

    def _args(**kwargs):
        return types.SimpleNamespace(**kwargs)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _Tok
    transformers.AutoModelForCausalLM = _Model
    transformers.TrainerCallback = _TrainerCallback
    trl = types.ModuleType("trl")
    trl.SFTConfig = _args
    trl.SFTTrainer = _Trainer
    trl.DPOConfig = _args
    trl.DPOTrainer = _Trainer
    trl.__version__ = "0.0.0-fake"
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.float32 = "fp32"
    datasets = types.ModuleType("datasets")
    datasets.Dataset = _Dataset
    peft = types.ModuleType("peft")
    peft.LoraConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    for name, module in {
        "transformers": transformers,
        "trl": trl,
        "torch": torch,
        "datasets": datasets,
        "peft": peft,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def _loads(calls, kind: str) -> list[tuple[str, dict]]:
    return [(target, kwargs) for name, target, kwargs in calls if name == kind]


# --------------------------------------------------------------------------- #
# the HF cache symlink farm must be inspectable at all
# --------------------------------------------------------------------------- #
def test_symlinked_snapshot_is_inspectable_and_fingerprintable(symlinked_cache):
    """A real cache snapshot links out to ``blobs/``; containment must allow it."""
    inspection = inspect_local_checkpoint(
        symlinked_cache, revision=REVISION, model_id=MODEL_ID)

    assert inspection.parameter_count == sum(
        math.prod(shape) for shape in _tiny_shapes().values())
    assert inspection.architecture.model_type == "qwen3"
    spec = ModelSpec.from_local_checkpoint(
        symlinked_cache, revision=REVISION, model_id=MODEL_ID)
    assert spec.revision == REVISION


def test_index_path_traversal_is_still_rejected(tmp_path):
    """Allowing symlinks must not allow an index to escape the checkpoint."""
    root = tmp_path / "evil"
    _write_checkpoint(root)
    index = root / "model.safetensors.index.json"
    payload = json.loads(index.read_text())
    payload["weight_map"] = {
        name: "../../../etc/passwd" for name in payload["weight_map"]
    }
    index.write_text(json.dumps(payload))

    with pytest.raises(ModelSpecError, match="unsafe shard path"):
        inspect_local_checkpoint(root, revision=REVISION)


# --------------------------------------------------------------------------- #
# revision resolution: pinned, mutable, missing
# --------------------------------------------------------------------------- #
def test_pinned_revision_resolves_to_the_local_snapshot(hf_cache):
    assert resolve_local_snapshot(
        MODEL_ID, REVISION, environ=hf_cache.env) == hf_cache.snapshot
    assert resolve_local_snapshot(
        MODEL_ID, OTHER_REVISION, environ=hf_cache.env) is None

    identity = resolve_model_identity(
        MODEL_ID, revision=REVISION, stage="midtrain", environ=hf_cache.env)

    assert identity.revision == REVISION
    assert identity.load_kwargs == {"revision": REVISION}
    assert identity.local_path == str(hf_cache.snapshot)
    assert identity.verify == VERIFY_METADATA
    assert identity.notes == ()


@pytest.mark.parametrize("mutable", ["main", "v1.0", "refs/pr/3", "40c0698", "a" * 39])
def test_mutable_or_malformed_revisions_are_rejected_in_both_modes(
    hf_cache, mutable
):
    """A configured floating ref is a config defect, so neither mode accepts it."""
    for mode in (DEVELOPMENT, PRODUCTION):
        with pytest.raises(FloatingRevisionError, match="40- or 64-hex"):
            resolve_model_identity(
                MODEL_ID, revision=mutable, mode=mode, environ=hf_cache.env)


def test_missing_revision_fails_closed_in_production(hf_cache):
    with pytest.raises(UnpinnedModelError) as excinfo:
        resolve_model_identity(
            MODEL_ID, stage="sft", mode=PRODUCTION, environ=hf_cache.env)

    message = str(excinfo.value)
    # The message must say which stage, which key to set, and where to find it.
    assert "sft" in message
    assert "model_revision" in message
    assert "KORE_MODEL_REVISION" in message
    assert "DATASET_STATUS.md" in message


def test_missing_revision_warns_and_proceeds_in_development(hf_cache):
    identity = resolve_model_identity(
        MODEL_ID, stage="sft", mode=DEVELOPMENT, environ=hf_cache.env)

    assert identity.revision is None
    assert identity.pinned is False
    # No revision kwarg at all -> byte-identical to the pre-wiring load.
    assert identity.load_kwargs == {}
    assert identity.verify == VERIFY_NONE
    assert any("no immutable revision is configured" in note
               for note in identity.notes)
    identity.validate_before_load()  # must be a safe no-op


def test_uncached_pin_fails_closed_in_production_and_degrades_offline(hf_cache):
    with pytest.raises(ModelSpecError, match="not present in any local"):
        resolve_model_identity(
            MODEL_ID, revision=OTHER_REVISION, stage="dpo",
            mode=PRODUCTION, environ=hf_cache.env)

    # Offline development: pinning a commit the cache lacks would hard-fail the
    # load, which must never happen to a run that works today.
    offline = resolve_model_identity(
        MODEL_ID, revision=OTHER_REVISION, stage="dpo", environ=hf_cache.env)
    assert offline.load_kwargs == {}
    assert any("proceeding UNPINNED" in note for note in offline.notes)

    # With the Hub reachable the pin is kept: the Hub can resolve that commit.
    online_env = dict(hf_cache.env)
    online_env.pop("HF_HUB_OFFLINE")
    online = resolve_model_identity(
        MODEL_ID, revision=OTHER_REVISION, stage="dpo", environ=online_env)
    assert online.load_kwargs == {"revision": OTHER_REVISION}


def test_production_is_a_one_way_opt_in_from_config_or_environment(hf_cache):
    assert resolve_identity_mode(None, environ={}) == DEVELOPMENT
    assert resolve_identity_mode(PRODUCTION, environ={}) == PRODUCTION
    assert resolve_identity_mode(
        None, environ={"KORE_MODEL_IDENTITY_MODE": PRODUCTION}) == PRODUCTION
    # Neither source can silently downgrade the other.
    assert resolve_identity_mode(
        DEVELOPMENT, environ={"KORE_MODEL_IDENTITY_MODE": PRODUCTION}) == PRODUCTION
    assert resolve_identity_mode(
        PRODUCTION, environ={"KORE_MODEL_IDENTITY_MODE": DEVELOPMENT}) == PRODUCTION
    with pytest.raises(ModelSpecError, match="identity mode"):
        resolve_identity_mode("strict-ish", environ={})


def test_local_checkpoint_directory_does_not_claim_a_hub_commit(hf_cache, tmp_path):
    """The stage handoff loads a directory, which has no Hub commit to claim.

    The campaign overlays ``model_id`` with the previous stage's output dir, so a
    base-model SHA left in the config must not be reported as this checkpoint's
    identity - and must not be handed to ``from_pretrained``, which ignores it.
    """
    stage_output = tmp_path / "runs" / "sft_out"
    _write_checkpoint(stage_output)

    identity = resolve_model_identity(
        str(stage_output), revision=REVISION, stage="dpo",
        mode=PRODUCTION, environ=hf_cache.env)

    assert identity.revision is None
    assert identity.load_kwargs == {}
    assert identity.local_path == str(stage_output.resolve())
    # Production still verifies the handoff's architecture and tensor shapes.
    assert identity.verify == VERIFY_METADATA
    assert identity.inspection is not None
    assert any("IGNORED" in note for note in identity.notes)


def test_empty_model_id_is_not_read_as_the_working_directory(hf_cache):
    """``Path("")`` is the CWD, which must never be mistaken for a checkpoint."""
    identity = resolve_model_identity("", stage="sft", environ=hf_cache.env)

    assert identity.local_path is None
    assert identity.load_kwargs == {}


def test_production_rejects_a_corrupt_local_handoff(hf_cache, tmp_path):
    stage_output = tmp_path / "runs" / "broken_out"
    _write_checkpoint(stage_output)
    (stage_output / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
        "hidden_size": 4, "intermediate_size": 8,
        "num_hidden_layers": 5,  # the checkpoint only has two decoder layers
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
        "vocab_size": 16, "max_position_embeddings": 128,
    }))

    with pytest.raises(ModelSpecError, match="layer coverage"):
        resolve_model_identity(
            str(stage_output), stage="dpo", mode=PRODUCTION, environ=hf_cache.env)

    # Development reports the same failure but does not block the run.
    identity = resolve_model_identity(
        str(stage_output), stage="dpo", environ=hf_cache.env)
    assert identity.verify == VERIFY_NONE
    assert any("verification (metadata) failed" in note for note in identity.notes)


# --------------------------------------------------------------------------- #
# TOCTOU: the pre-load re-fingerprint
# --------------------------------------------------------------------------- #
def test_toctou_refingerprint_fires_when_a_shard_changes(hf_cache):
    identity = resolve_model_identity(
        MODEL_ID, revision=REVISION, stage="midtrain",
        mode=PRODUCTION, environ=hf_cache.env)

    assert identity.verify == VERIFY_FINGERPRINT
    assert identity.spec is not None
    identity.validate_before_load()  # unchanged tree: passes

    shard = hf_cache.snapshot / "model-00001-of-00002.safetensors"
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(ModelSpecError, match="changed after ModelSpec validation"):
        identity.validate_before_load()


def test_toctou_refingerprint_also_catches_a_swapped_config(hf_cache):
    identity = resolve_model_identity(
        MODEL_ID, revision=REVISION, stage="sft",
        mode=PRODUCTION, environ=hf_cache.env)
    config_path = hf_cache.snapshot / "config.json"
    config = json.loads(config_path.read_text())
    config["rope_theta"] = 12345
    config_path.write_text(json.dumps(config))

    with pytest.raises(ModelSpecError, match="changed"):
        identity.validate_before_load()


# --------------------------------------------------------------------------- #
# per-stage wiring: the revision actually reaches from_pretrained
# --------------------------------------------------------------------------- #
def test_midtrain_threads_the_pinned_revision_to_from_pretrained(
    hf_cache, fake_training_stack, tmp_path, monkeypatch
):
    from kore.policy import midtrain

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "a hip kernel"}) + "\n")
    for key, value in hf_cache.env.items():
        monkeypatch.setenv(key, value)
    config = midtrain.midtrain_config_from_dict({
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "corpus_path": str(corpus),
        "output_dir": str(tmp_path / "out"),
        "use_lora": False,
    })
    assert config.model_revision == REVISION

    midtrain.train_midtrain(config)

    tokenizer_loads = _loads(fake_training_stack, "tokenizer")
    model_loads = _loads(fake_training_stack, "model")
    assert tokenizer_loads and model_loads
    assert all(kwargs["revision"] == REVISION for _, kwargs in tokenizer_loads)
    assert all(kwargs["revision"] == REVISION for _, kwargs in model_loads)


def test_sft_threads_the_pinned_revision_to_from_pretrained(
    hf_cache, fake_training_stack, tmp_path, monkeypatch
):
    from kore.policy import sft

    dataset = tmp_path / "sft.jsonl"
    dataset.write_text(json.dumps({"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]}) + "\n")
    for key, value in hf_cache.env.items():
        monkeypatch.setenv(key, value)
    config, dataset_path = sft.sft_config_from_dict({
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "dataset_path": str(dataset),
        "output_dir": str(tmp_path / "out"),
        "use_lora": False,
        # The fake tokenizer has no chat template, and masked-loss verification is
        # orthogonal to identity wiring.
        "assistant_only_loss": False,
    })

    sft.train_sft(config, Path(dataset_path))

    loads = _loads(fake_training_stack, "tokenizer") + _loads(
        fake_training_stack, "model")
    assert len(loads) >= 2
    assert all(kwargs["revision"] == REVISION for _, kwargs in loads)


def test_dpo_pins_both_the_policy_and_the_frozen_reference(
    hf_cache, fake_training_stack, tmp_path, monkeypatch
):
    """DPO loads two 14B models; an unpinned reference is an unpinned baseline."""
    from kore.policy import dpo

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "good"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
    }) + "\n")
    for key, value in hf_cache.env.items():
        monkeypatch.setenv(key, value)
    config = dpo.dpo_config_from_dict({
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "dataset_path": str(pairs),
        "output_dir": str(tmp_path / "out"),
        "use_lora": False,
    })

    dpo.train(config)

    model_loads = _loads(fake_training_stack, "model")
    # policy + explicit frozen reference (never left implicit under full-FT)
    assert len(model_loads) == 2
    assert all(kwargs["revision"] == REVISION for _, kwargs in model_loads)
    assert all(target == MODEL_ID for target, _ in model_loads)


def test_dpo_reference_takes_its_own_pin_when_it_is_a_different_model(
    hf_cache, fake_training_stack, tmp_path, monkeypatch
):
    from kore.policy import dpo

    reference = tmp_path / "runs" / "round1"
    _write_checkpoint(reference)
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "good"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
    }) + "\n")
    for key, value in hf_cache.env.items():
        monkeypatch.setenv(key, value)
    config = dpo.dpo_config_from_dict({
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "ref_model_id": str(reference),
        "dataset_path": str(pairs),
        "output_dir": str(tmp_path / "out"),
        "use_lora": False,
    })

    dpo.train(config)

    model_loads = dict(_loads(fake_training_stack, "model"))
    assert model_loads[MODEL_ID]["revision"] == REVISION
    # A local reference directory takes no revision kwarg: transformers ignores it.
    assert "revision" not in model_loads[str(reference)]


def test_unpinned_config_loads_exactly_as_before(
    hf_cache, fake_training_stack, tmp_path, monkeypatch
):
    """An in-flight job restarting into this code must not change behaviour."""
    from kore.policy import midtrain

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "a hip kernel"}) + "\n")
    for key, value in hf_cache.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("KORE_MODEL_REVISION", raising=False)
    monkeypatch.delenv("KORE_MODEL_IDENTITY_MODE", raising=False)
    config = MidTrainConfig(
        model_id=MODEL_ID, corpus_path=str(corpus),
        output_dir=str(tmp_path / "out"), use_lora=False)
    assert not hasattr(config, "model_revision")

    midtrain.train_midtrain(config)

    loads = _loads(fake_training_stack, "tokenizer") + _loads(
        fake_training_stack, "model")
    assert loads
    assert all("revision" not in kwargs for _, kwargs in loads)


def test_environment_supplies_the_pin_when_the_config_cannot(hf_cache, monkeypatch):
    """GRPO and any un-migrated config can be pinned without a schema change."""
    monkeypatch.setenv("KORE_MODEL_REVISION", REVISION)
    for key, value in hf_cache.env.items():
        monkeypatch.setenv(key, value)

    identity = model_identity_for_config(
        SFTConfig(model_id=MODEL_ID), stage="sft")

    assert identity.load_kwargs == {"revision": REVISION}


# --------------------------------------------------------------------------- #
# the shipped configs carry the pin the dataset was built against
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel_path", [
    "configs/midtrain_14b_full.json",
    "configs/sft_14b_full.json",
    "configs/dpo_14b_full.json",
    "data/b05factory/launch/midtrain_8gpu.json",
    "data/b05factory/launch/midtrain_24gpu.json",
    "data/b05factory/launch/midtrain_32gpu.json",
    "data/b05factory/launch/midtrain_64gpu.json",
    "data/b05factory/launch/midtrain_frontier.json",
])
def test_shipped_configs_pin_the_documented_base_revision(rel_path):
    payload = json.loads((REPO_ROOT / rel_path).read_text())
    status = (REPO_ROOT / "DATASET_STATUS.md").read_text(encoding="utf-8")

    assert payload["model_id"] == "Qwen/Qwen3-14B", rel_path
    # The trainer's pin and the dataset's provenance must be the same commit.
    assert payload["model_revision"] in status, rel_path


@pytest.mark.parametrize("rel_path,builder", [
    ("configs/midtrain_14b_full.json", "midtrain"),
    ("data/b05factory/launch/midtrain_frontier.json", "midtrain"),
    ("configs/sft_14b_full.json", "sft"),
    ("configs/dpo_14b_full.json", "dpo"),
])
def test_pinned_configs_still_parse_through_their_builders(rel_path, builder):
    """The pin must not crash the strict ``Config(**payload)`` parse."""
    payload = json.loads((REPO_ROOT / rel_path).read_text())
    if builder == "midtrain":
        from kore.policy.midtrain import midtrain_config_from_dict
        config = midtrain_config_from_dict(payload)
    elif builder == "sft":
        from kore.policy.sft import sft_config_from_dict
        config, _ = sft_config_from_dict(payload)
    else:
        from kore.policy.dpo import dpo_config_from_dict
        config = dpo_config_from_dict(payload)

    assert config.model_revision == payload["model_revision"]
    assert not hasattr(type(config), "model_revision")  # attribute, not a field


def test_grpo_config_is_pinned_and_parses_through_the_identity_split():
    """GRPO's launch config pins the base model, and its parser still fails closed.

    ``grpo_config_from_dict`` does a strict ``GRPOConfig(**payload)``, so identity
    and preflight keys must be split off BEFORE construction. This asserts both
    halves: the pin is present and reaches the config, and every remaining key is
    still a real field, so a typo cannot ride in behind the split.
    """
    from kore.policy.configs import GRPOConfig
    from kore.policy.grpo import grpo_config_from_dict
    from kore.policy.model_spec import IDENTITY_CONFIG_KEYS
    from kore.policy.resources import PREFLIGHT_CONFIG_KEYS

    payload = json.loads((REPO_ROOT / "configs/grpo_14b_full.json").read_text())
    assert payload["model_revision"] == REVISION

    runtime_keys = set(IDENTITY_CONFIG_KEYS) | set(PREFLIGHT_CONFIG_KEYS)
    fields = set(GRPOConfig.__dataclass_fields__)
    # ``_comment_<field>`` keys are this repo's in-config documentation and are
    # dropped by every stage loader before the strict parse, exactly as here.
    leftover = {key for key in payload if not key.startswith("_")}
    leftover -= runtime_keys | {"tasks", "lora"}
    assert leftover <= fields, sorted(leftover - fields)

    config = grpo_config_from_dict(dict(payload))
    assert getattr(config, "model_revision", None) == REVISION


# --------------------------------------------------------------------------- #
# resource preflight: reachable, honest, and opt-in-strict
# --------------------------------------------------------------------------- #
def test_preflight_mode_defaults_to_report_and_takes_the_stricter_source():
    assert resolve_preflight_mode(None, environ={}) == PREFLIGHT_REPORT
    assert resolve_preflight_mode(PREFLIGHT_OFF, environ={}) == PREFLIGHT_OFF
    assert resolve_preflight_mode(
        PREFLIGHT_OFF, environ={"KORE_RESOURCE_PREFLIGHT": PREFLIGHT_STRICT}
    ) == PREFLIGHT_STRICT
    assert resolve_preflight_mode(
        PREFLIGHT_STRICT, environ={"KORE_RESOURCE_PREFLIGHT": PREFLIGHT_OFF}
    ) == PREFLIGHT_STRICT
    with pytest.raises(ResourcePreflightError, match="preflight mode"):
        resolve_preflight_mode("maybe", environ={})


def test_report_mode_reports_exact_bounds_and_never_blocks(hf_cache, tmp_path):
    """An unprofiled machine must still start the run."""
    identity = resolve_model_identity(
        MODEL_ID, revision=REVISION, stage="midtrain", environ=hf_cache.env)
    config = MidTrainConfig(model_id=MODEL_ID, output_dir=str(tmp_path / "out"))
    report_path = tmp_path / "preflight.json"

    result = run_stage_preflight(
        stage="midtrain", config=config, inspection=identity.inspection,
        mode=PREFLIGHT_REPORT, report_path=report_path,
        environ={"HF_HUB_CACHE": str(hf_cache.root)},
        scratch_path=str(tmp_path))

    parameters = identity.inspection.parameter_count
    assert result.analytical_lower_bounds.exact_parameter_count == parameters
    assert result.analytical_lower_bounds.bf16_weights_bytes == parameters * 2
    # No measured evidence and no fingerprinted spec -> no fit claim, ever.
    assert result.fit_asserted is False
    assert result.production_ready is False
    assert result.report is None
    persisted = json.loads(report_path.read_text())
    assert persisted["stage"] == "midtrain"
    assert persisted["production_ready"] is False
    assert persisted["analytical_lower_bounds"]["exact_parameter_count"] == parameters


def test_strict_mode_refuses_to_start_without_measured_evidence(hf_cache, tmp_path):
    identity = resolve_model_identity(
        MODEL_ID, revision=REVISION, stage="midtrain",
        mode=PRODUCTION, environ=hf_cache.env)
    config = MidTrainConfig(model_id=MODEL_ID, output_dir=str(tmp_path / "out"))

    with pytest.raises(ResourcePreflightError, match="did not establish a measured"):
        run_stage_preflight(
            stage="midtrain", config=config, model_spec=identity.spec,
            mode=PREFLIGHT_STRICT, environ={"HF_HUB_CACHE": str(hf_cache.root)},
            scratch_path=str(tmp_path))


def test_unpinned_stage_does_not_probe_the_host_at_all(tmp_path, monkeypatch):
    """With nothing to compare against, preflight must not add startup work.

    An in-flight job restarting into this code has no pin, so it must not pay for
    a ``rocm-smi`` / ``git status`` / sysfs inventory it cannot use.
    """
    import kore.policy.resources as resources_module

    def _explode(*args, **kwargs):
        raise AssertionError("the host must not be probed without a checkpoint")

    monkeypatch.setattr(resources_module, "collect_resource_snapshot", _explode)

    result = run_stage_preflight(
        stage="midtrain", config=MidTrainConfig(output_dir=str(tmp_path)),
        mode=PREFLIGHT_REPORT, environ={})

    assert result.status == "skipped"
    assert result.resources is None
    assert any("no local checkpoint metadata" in reason for reason in result.reasons)


def test_off_mode_touches_nothing(tmp_path):
    result = run_stage_preflight(
        stage="sft", config=SFTConfig(output_dir=str(tmp_path)),
        mode=PREFLIGHT_OFF, environ={})

    assert result.status == "off"
    assert result.resources is None
    assert result.analytical_lower_bounds is None
    assert result.production_ready is False


def test_stage_workload_spec_binds_config_topology_and_model(hf_cache, tmp_path):
    from kore.policy.resources import collect_resource_snapshot

    spec = ModelSpec.from_local_checkpoint(
        hf_cache.snapshot, revision=REVISION, model_id=MODEL_ID)
    resources = collect_resource_snapshot(
        hf_cache.snapshot, tmp_path, sysfs_root=tmp_path / "absent-drm",
        environ={}, code_root=REPO_ROOT)
    config = DPOConfig(
        model_id=MODEL_ID, use_lora=False, distributed=True,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        max_length=16384)

    workload = build_stage_workload_spec(
        config, stage="dpo", resources=resources, model_spec=spec,
        world_size=8, environ={})

    assert workload.stage == "dpo"
    assert workload.global_batch_size == 2 * 8 * 8
    assert workload.sequence_lengths == {"max_length": 16384}
    assert workload.sharding == "fsdp:full_shard auto_wrap"
    assert workload.offload == "none"
    # DPO holds a second, frozen full model; that copy has to be in the bound.
    assert (workload.model_copies, workload.reference_copies) == (1, 1)
    assert workload.model_profile_hash == spec.profile_hash
    assert workload.topology_hash == resources.topology_hash
    assert workload.dependency_profile_hash == resources.dependency_profile_hash
    assert "trl" in workload.required_dependencies


def test_stages_run_preflight_by_default_without_blocking(
    hf_cache, fake_training_stack, tmp_path, monkeypatch
):
    """The default path reaches preflight; a CPU box with no GPUs still trains."""
    from kore.policy import midtrain

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "a hip kernel"}) + "\n")
    for key, value in hf_cache.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("KORE_RESOURCE_PREFLIGHT", raising=False)
    seen: list[dict] = []
    real_run = midtrain.run_stage_preflight

    def _record(**kwargs):
        result = real_run(**kwargs)
        seen.append({"stage": kwargs["stage"], "status": result.status})
        return result

    monkeypatch.setattr(midtrain, "run_stage_preflight", _record)
    config = midtrain.midtrain_config_from_dict({
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "corpus_path": str(corpus),
        "output_dir": str(tmp_path / "out"),
        "use_lora": False,
        "resource_preflight": PREFLIGHT_REPORT,
    })

    midtrain.train_midtrain(config)

    assert seen and seen[0]["stage"] == "midtrain"
    assert _loads(fake_training_stack, "model")  # training still proceeded
