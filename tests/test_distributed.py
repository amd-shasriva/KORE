"""Tests for KORE distributed full fine-tuning (FSDP) wiring.

Almost all of these are CPU-only and heavy-dep-free: they exercise the pure
config -> TrainingArguments FSDP kwargs translation, the JSON entry parsing, and
the launcher script. One test optionally constructs a real ``trl``/transformers
``TrainingArguments`` to prove the kwargs are accepted (skipped if the stack is
missing); it restores FSDP-related env vars afterwards so it can't leak into
other tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kore.policy.configs import (
    DPOConfig,
    MultiCapSFTConfig,
    SFTConfig,
    build_fsdp_kwargs,
    detect_transformer_layer_cls,
    fsdp_enabled,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "launch_distributed.sh"
ACCEL_YAML = REPO_ROOT / "configs" / "accelerate_fsdp.yaml"


# --------------------------------------------------------------------------- #
# config mixin fields
# --------------------------------------------------------------------------- #
def test_distributed_fields_present_with_safe_defaults():
    for cfg in (SFTConfig(), DPOConfig(), MultiCapSFTConfig()):
        # Defaults keep every existing (single-process / LoRA / CPU) path intact.
        assert cfg.distributed is False
        assert cfg.fsdp == "full_shard auto_wrap"
        assert cfg.fsdp_transformer_layer_cls is None
        assert cfg.fsdp_cpu_offload is False


# --------------------------------------------------------------------------- #
# audit-fix config defaults (completion-only SFT, disk-fill guard, live knobs)
# --------------------------------------------------------------------------- #
def test_sft_completion_only_and_knob_defaults():
    s = SFTConfig()
    assert s.assistant_only_loss is True          # completion-only loss on by default
    assert s.max_grad_norm == 1.0                 # explicit clip (was HF-default only)
    assert s.weight_decay == 0.0


def test_dpo_weight_decay_field():
    assert DPOConfig().weight_decay == 0.0


def test_midtrain_has_save_total_limit_and_live_knobs():
    from kore.policy.configs import MidTrainConfig
    mt = MidTrainConfig()
    # HIGH: bounds the 14B full-FT checkpoint count so CPT can't fill disk.
    assert mt.save_total_limit == 1
    # Knobs previously hardcoded in midtrain.py are now config-driven.
    assert mt.per_device_train_batch_size == 1
    assert mt.gradient_accumulation_steps == 16
    # packing is False on the SDPA runtime: TRL bfd packing silently cross-contaminates
    # documents without a flash-attn backend (audit THEME B/C2).
    assert mt.packing is False
    # input-pipeline parallelism (THEME E): loader workers + multiproc tokenization.
    assert mt.dataloader_num_workers >= 1 and mt.dataset_num_proc >= 1
    assert mt.save_steps == 200 and mt.logging_steps == 10
    assert mt.max_grad_norm == 1.0 and mt.seed == 0
    # recipe replay fraction matches the dataclass intent (Ibrahim/DeepSeek-V2)
    assert mt.general_replay_frac == 0.30


def test_dpo_truncation_defaults_keep_end():
    from kore.policy.dpo import build_trl_dpo_kwargs
    k = build_trl_dpo_kwargs(DPOConfig())
    # keep_end preserves the completion (kernel) tail + stop; keep_start would cut it.
    assert k["truncation_mode"] == "keep_end"
    assert k["weight_decay"] == 0.0
    # an explicit override is still honored
    d = DPOConfig()
    setattr(d, "truncation_mode", "keep_start")
    assert build_trl_dpo_kwargs(d)["truncation_mode"] == "keep_start"


def test_dpo_loss_arity_guard_reconciles_type_and_weights():
    """audit R2 dpo C1: loss_type and loss_weights can never reach TRL with a
    len mismatch. A composite loss gets matching weights (synthesized if missing);
    a scalar loss carries NO weights (a lingering multi-weight list is the crash)."""
    from kore.policy.dpo import build_trl_dpo_kwargs

    def kw(**over):
        c = DPOConfig()
        for a, v in over.items():
            setattr(c, a, v)
        return build_trl_dpo_kwargs(c)

    # composite RPO loss + matching weights -> preserved
    r = kw(loss_type=["sigmoid", "sft"], loss_weights=[1.0, 1.0])
    assert r["loss_type"] == ["sigmoid", "sft"] and r["loss_weights"] == [1.0, 1.0]
    # composite loss with MISSING weights -> synthesized equal weights (no crash)
    r = kw(loss_type=["ipo", "sft"])
    assert r["loss_type"] == ["ipo", "sft"] and r["loss_weights"] == [1.0, 1.0]
    # composite loss with MISMATCHED weights -> reconciled to equal weights
    assert kw(loss_type=["sigmoid", "sft"], loss_weights=[1.0])["loss_weights"] == [1.0, 1.0]
    # scalar loss + lingering multi-weight list -> weights DROPPED (the arity crash)
    r = kw(loss_type="ipo", loss_weights=[1.0, 1.0])
    assert r["loss_type"] == "ipo" and "loss_weights" not in r
    # scalar loss alone, and no-loss config -> no stray loss_weights
    assert "loss_weights" not in kw(loss_type="sigmoid")
    assert "loss_type" not in kw() and "loss_weights" not in kw()


def test_build_assistant_masked_template_injects_and_is_safe():
    from kore.policy.sft import build_assistant_masked_template

    stub = (
        '{%- for message in messages %}\n'
        '    {%- if message.role == "assistant" %}\n'
        "        {%- if loop.index0 > ns.last_query_index %}\n"
        "            {%- if loop.last %}\n"
        "                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n"
        "            {%- else %}\n"
        "                {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
        "            {%- endif %}\n"
        "        {%- else %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
        "        {%- endif %}\n"
        "        {{- '<|im_end|>\\n' }}\n"
        '    {%- elif message.role == "tool" %}\n'
        "        {{- content }}\n"
        "    {%- endif %}\n"
        "{%- endfor %}"
    )
    masked = build_assistant_masked_template(stub)
    assert "{% generation %}" in masked and "{% endgeneration %}" in masked
    # header pulled OUT of the body (so it stays masked); body header removed
    assert "{{- content }}" in masked
    # idempotent (already tagged) + fails loudly on a non-Qwen3 template
    assert build_assistant_masked_template(masked) == masked
    with pytest.raises(ValueError):
        build_assistant_masked_template("{{ messages[0].content }}")


# --------------------------------------------------------------------------- #
# transformer layer auto-detect (14B/32B/70B bases)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("Qwen/Qwen3-14B", "Qwen3DecoderLayer"),
        ("Qwen/Qwen3-32B", "Qwen3DecoderLayer"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "Qwen2DecoderLayer"),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "LlamaDecoderLayer"),
        ("meta-llama/Llama-3.1-70B", "LlamaDecoderLayer"),
        ("mistralai/Mistral-7B-v0.3", "MistralDecoderLayer"),
        ("", "Qwen3DecoderLayer"),
    ],
)
def test_detect_transformer_layer_cls(model_id, expected):
    assert detect_transformer_layer_cls(model_id) == expected


# --------------------------------------------------------------------------- #
# config -> TrainingArguments FSDP kwargs (the core wiring)
# --------------------------------------------------------------------------- #
def test_fsdp_enabled_only_for_distributed_full_ft():
    assert fsdp_enabled(SFTConfig(use_lora=False, distributed=True)) is True
    assert fsdp_enabled(SFTConfig(use_lora=True, distributed=True)) is False
    assert fsdp_enabled(SFTConfig(use_lora=False, distributed=False)) is False
    assert fsdp_enabled(DPOConfig(use_lora=False, distributed=True)) is True


def test_build_fsdp_kwargs_full_ft_distributed():
    cfg = SFTConfig(model_id="Qwen/Qwen3-14B", use_lora=False, distributed=True)
    kw = build_fsdp_kwargs(cfg)
    assert kw["fsdp"] == "full_shard auto_wrap"
    fc = kw["fsdp_config"]
    assert fc["transformer_layer_cls_to_wrap"] == ["Qwen3DecoderLayer"]
    # Activation checkpointing is NOT driven from fsdp_config (the external
    # checkpoint_wrapper mismatches saved-tensor counts on FSDP1/use_orig_params);
    # the Trainer stages enable HF's layer-internal use_reentrant=False path.
    assert "activation_checkpointing" not in fc
    assert fc["use_orig_params"] is True
    assert fc["sync_module_states"] is True
    assert fc["cpu_ram_efficient_loading"] is True
    assert "offload_params" not in fc


def test_build_fsdp_kwargs_lora_path_stays_empty():
    # LoRA full/distributed still returns {} -> keeps the legacy device_map path.
    assert build_fsdp_kwargs(SFTConfig(use_lora=True, distributed=True)) == {}
    assert build_fsdp_kwargs(DPOConfig(use_lora=True, distributed=True)) == {}


def test_build_fsdp_kwargs_single_process_stays_empty():
    assert build_fsdp_kwargs(SFTConfig(use_lora=False, distributed=False)) == {}


def test_build_fsdp_kwargs_cpu_offload():
    cfg = DPOConfig(model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                    use_lora=False, distributed=True, fsdp_cpu_offload=True)
    kw = build_fsdp_kwargs(cfg)
    assert kw["fsdp"] == "full_shard auto_wrap offload"
    assert kw["fsdp_config"]["offload_params"] is True
    assert kw["fsdp_config"]["transformer_layer_cls_to_wrap"] == ["LlamaDecoderLayer"]


def test_build_fsdp_kwargs_explicit_layer_cls_overrides_autodetect():
    cfg = SFTConfig(model_id="Qwen/Qwen3-14B", use_lora=False, distributed=True,
                    fsdp_transformer_layer_cls="CustomLayer")
    kw = build_fsdp_kwargs(cfg)
    assert kw["fsdp_config"]["transformer_layer_cls_to_wrap"] == ["CustomLayer"]


def test_fsdp_config_never_sets_activation_checkpointing():
    # Activation checkpointing is owned by the Trainer stage (HF gradient_checkpointing
    # + use_reentrant=False), never by fsdp_config — regardless of the flag — because
    # the FSDP-plugin external wrapper mismatches saved-tensor counts on FSDP1.
    for gc in (True, False):
        cfg = SFTConfig(use_lora=False, distributed=True, gradient_checkpointing=gc)
        assert "activation_checkpointing" not in build_fsdp_kwargs(cfg)["fsdp_config"]


# --------------------------------------------------------------------------- #
# JSON entry parsing
# --------------------------------------------------------------------------- #
def test_sft_config_from_dict_with_nested_lora_and_dataset():
    from kore.policy.sft import sft_config_from_dict

    cfg, ds = sft_config_from_dict({
        "model_id": "Qwen/Qwen3-14B",
        "use_lora": False,
        "distributed": True,
        "dataset_path": "data/sft/train.jsonl",
        "fsdp_transformer_layer_cls": "Qwen3DecoderLayer",
        "lora": {"r": 8, "lora_alpha": 16},
    })
    assert cfg.model_id == "Qwen/Qwen3-14B"
    assert cfg.use_lora is False and cfg.distributed is True
    assert cfg.lora.r == 8 and cfg.lora.lora_alpha == 16
    assert ds == "data/sft/train.jsonl"


def test_sft_config_from_dict_dataset_falls_back_to_config_field():
    from kore.policy.sft import sft_config_from_dict

    cfg, ds = sft_config_from_dict({"dataset_path": "x.jsonl"})
    assert ds == "x.jsonl" and cfg.dataset_path == "x.jsonl"


def test_dpo_config_from_dict():
    from kore.policy.dpo import dpo_config_from_dict

    cfg = dpo_config_from_dict({
        "model_id": "m", "dataset_path": "pairs.jsonl",
        "use_lora": False, "distributed": True, "beta": 0.2,
        "lora": {"r": 4},
    })
    assert cfg.dataset_path == "pairs.jsonl"
    assert cfg.use_lora is False and abs(cfg.beta - 0.2) < 1e-9
    assert cfg.lora.r == 4


def test_sft_entry_main_json_roundtrip(tmp_path, monkeypatch):
    # Entry reads JSON, defaults distributed=True, then (here) fails cleanly on a
    # missing dataset_path -> rc 2, WITHOUT importing torch/trl.
    from kore.policy import sft

    p = tmp_path / "sft.json"
    p.write_text(json.dumps({"model_id": "Qwen/Qwen3-14B", "use_lora": False}))
    assert sft._main([str(p)]) == 2  # no dataset_path
    assert sft._main([]) == 2        # no args -> usage


def test_dpo_entry_main_missing_dataset(tmp_path):
    from kore.policy import dpo

    p = tmp_path / "dpo.json"
    p.write_text(json.dumps({"model_id": "m", "use_lora": False}))
    assert dpo._main([str(p)]) == 2
    assert dpo._main([]) == 2


def test_entry_defaults_distributed_true(tmp_path):
    # A JSON that omits `distributed` should be treated as distributed by the
    # entry (it's launched via accelerate). We verify the parse function honors an
    # explicit value and that the entry default is applied to the raw dict.
    from kore.policy.sft import sft_config_from_dict

    raw = {"model_id": "Qwen/Qwen3-14B", "use_lora": False, "dataset_path": "d.jsonl"}
    raw.setdefault("distributed", True)  # mirrors _main
    cfg, _ = sft_config_from_dict(raw)
    assert cfg.distributed is True and fsdp_enabled(cfg) is True


# --------------------------------------------------------------------------- #
# import safety: no heavy deps at module import
# --------------------------------------------------------------------------- #
def test_entry_modules_import_without_torch():
    code = (
        "import sys; import kore.policy.sft, kore.policy.dpo, kore.policy.configs; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m=='torch'); "
        "print('ok')"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# --------------------------------------------------------------------------- #
# real TrainingArguments accepts the FSDP kwargs (optional / heavy)
# --------------------------------------------------------------------------- #
@pytest.fixture
def _restore_fsdp_env():
    keys = [k for k in os.environ if k.startswith("FSDP_") or k in ("ACCELERATE_USE_FSDP",)]
    snapshot = {k: os.environ[k] for k in keys}
    try:
        yield
    finally:
        for k in list(os.environ):
            if k.startswith("FSDP_") or k == "ACCELERATE_USE_FSDP":
                os.environ.pop(k, None)
        os.environ.update(snapshot)


def test_training_arguments_accept_fsdp_kwargs(tmp_path, _restore_fsdp_env):
    trl = pytest.importorskip("trl")
    TRLSFTConfig = trl.SFTConfig
    TRLDPOConfig = trl.DPOConfig

    kw = build_fsdp_kwargs(SFTConfig(model_id="Qwen/Qwen3-14B", use_lora=False, distributed=True))
    a = TRLSFTConfig(output_dir=str(tmp_path / "sft"), bf16=True,
                     gradient_checkpointing=False, **kw)
    # transformers normalizes fsdp into a list of FSDPOption and keeps the wrap cls.
    fsdp_vals = [str(getattr(o, "value", o)) for o in a.fsdp]
    assert "full_shard" in fsdp_vals
    assert a.fsdp_config["transformer_layer_cls_to_wrap"] == ["Qwen3DecoderLayer"]

    kw_d = build_fsdp_kwargs(DPOConfig(model_id="Qwen/Qwen3-14B", use_lora=False, distributed=True))
    d = TRLDPOConfig(output_dir=str(tmp_path / "dpo"), bf16=True,
                     gradient_checkpointing=False, **kw_d)
    assert "full_shard" in [str(getattr(o, "value", o)) for o in d.fsdp]


# --------------------------------------------------------------------------- #
# accelerate FSDP yaml
# --------------------------------------------------------------------------- #
def test_accelerate_fsdp_yaml_is_valid_full_shard():
    import yaml

    cfg = yaml.safe_load(ACCEL_YAML.read_text())
    assert cfg["distributed_type"] == "FSDP"
    assert cfg["mixed_precision"] == "bf16"
    assert cfg["num_processes"] == 8
    fc = cfg["fsdp_config"]
    assert fc["fsdp_auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"
    assert fc["fsdp_transformer_layer_cls_to_wrap"] == "Qwen3DecoderLayer"
    assert fc["fsdp_reshard_after_forward"] == "FULL_SHARD"  # ZeRO-3 equivalent
    # Activation checkpointing is done via HF Trainer gradient_checkpointing
    # (use_reentrant=False), NOT the FSDP plugin (which mismatches tensor counts).
    assert fc["fsdp_activation_checkpointing"] is False
    assert fc["fsdp_offload_params"] is False  # 14B default


# --------------------------------------------------------------------------- #
# launcher script
# --------------------------------------------------------------------------- #
def test_launcher_syntax_ok():
    r = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_launcher_dry_run_prints_accelerate_command():
    r = subprocess.run(
        ["bash", str(LAUNCHER), "sft", "configs/sft_14b_full.json", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "accelerate launch" in out
    assert "configs/accelerate_fsdp.yaml" in out
    assert "-m kore.policy.sft" in out
    assert "configs/sft_14b_full.json" in out


def test_launcher_dry_run_dpo_with_nproc():
    r = subprocess.run(
        ["bash", str(LAUNCHER), "dpo", "cfg.json", "--nproc", "8", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "-m kore.policy.dpo" in r.stdout
    assert "--num_processes 8" in r.stdout


def test_launcher_no_args_exits_nonzero():
    r = subprocess.run(["bash", str(LAUNCHER)], capture_output=True, text=True)
    assert r.returncode != 0
    assert "usage" in r.stderr.lower()


def test_launcher_bad_stage_exits_nonzero():
    r = subprocess.run(["bash", str(LAUNCHER), "foo", "cfg.json"], capture_output=True, text=True)
    assert r.returncode != 0
    assert "stage must be" in r.stderr
