"""CPU-only tests for the end-to-end campaign wiring (scripts/run_campaign.py).

No GPU, no teacher, no torch/trl. Every heavy stage entrypoint is monkeypatched;
we only assert that the campaign WIRES the newly-implemented research capabilities
together correctly:

  * the AUTHORITATIVE registry train/held-out split threads through ctx + manifest
    (item 1) - training stages get TRAIN tasks, eval gets the held-out family;
  * ``--dpo-rounds > 1`` drives ``iterative_dpo`` (on-policy DPO + DAgger, item 2)
    while ``== 1`` keeps the single-pass DPO;
  * the evolutionary datagen stage is callable and writes wins/groups shards (item 3);
  * ``--grpo-curriculum`` runs TWO GRPO phases (correctness -> latency) with the
    phase-1 checkpoint threaded into phase-2 init (item 4);
  * ``assemble`` folds on-policy / evolve / DAgger records into the SFT + DPO
    products (item 5); and
  * the dry-run import preflight includes every new symbol.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import scripts.run_campaign as rc
from kore.tasks.registry import get_task, heldout_tasks


def _args(argv):
    return rc.build_parser().parse_args(argv)


# --------------------------------------------------------------------------- #
# 1. Authoritative held-out split threads through ctx + manifest
# --------------------------------------------------------------------------- #
def test_heldout_split_threads_through():
    args = _args(["--tasks", "rmsnorm_aiter,gemm_bf16"])
    ctx = {"tasks": [get_task("rmsnorm_aiter"), get_task("gemm_bf16")], "args": args}
    rc._apply_split(ctx)

    # both selected tasks are TRAIN (not a held-out family); eval falls back to the
    # registry's held-out generalization set (the reserved MLA + paged-KV decode families).
    assert set(ctx["train_task_ids"]) == {"rmsnorm_aiter", "gemm_bf16"}
    held_ids = {t.task_id for t in heldout_tasks()}
    assert held_ids  # the registry reserves at least one family
    assert set(ctx["eval_task_ids"]) == held_ids
    # no leakage: nothing trained is also evaluated
    assert not (set(ctx["train_task_ids"]) & set(ctx["eval_task_ids"]))


def test_selected_heldout_task_routes_to_eval_not_train():
    held = heldout_tasks()[0].task_id
    args = _args(["--tasks", f"rmsnorm_aiter,{held}"])
    ctx = {"tasks": [get_task("rmsnorm_aiter"), get_task(held)], "args": args}
    rc._apply_split(ctx)
    assert ctx["train_task_ids"] == ["rmsnorm_aiter"]
    assert held in ctx["eval_task_ids"]
    assert held not in ctx["train_task_ids"]


def test_manifest_threads_train_and_eval_ids(tmp_path):
    lineage = {
        "compatibility_digest": "sha256:test",
        "model": {"requested_id": "Qwen/Qwen3-14B", "content_digest": "sha256:model"},
        "tokenizer": {"content_digest": "sha256:tokenizer"},
        "source": {"content_digest": "sha256:source"},
        "stage_config": {"digest": "sha256:config"},
        "tasks": {
            "registry_digest": "sha256:registry",
            "split_digest": "sha256:split",
            "train": ["rmsnorm_aiter", "gemm_bf16"],
            "eval": ["flash_attn_decode_bf16"],
        },
        "verifier_gate_contract": {"digest": "sha256:gates"},
        "hardware_runtime": {"compatibility_digest": "sha256:runtime"},
    }
    ctx = {
        "data_root": tmp_path, "dry": False, "base": "Qwen/Qwen3-14B",
        "midtrain_ckpt": None, "sft_ckpt": "sft", "dpo_ckpt": None,
        "grpo_ckpt": None, "final": None, "done_stages": {"build"},
        "train_task_ids": ["rmsnorm_aiter", "gemm_bf16"],
        "eval_task_ids": ["flash_attn_decode_bf16"],
        "lineage": lineage, "artifacts": {},
    }
    rc._save_manifest(ctx)

    persisted = json.loads((tmp_path / "campaign_manifest.json").read_text())
    assert persisted["schema"] == {"name": "kore.campaign", "version": 1}
    assert persisted["lineage"]["tasks"]["train"] == ["rmsnorm_aiter", "gemm_bf16"]
    assert persisted["lineage"]["tasks"]["eval"] == ["flash_attn_decode_bf16"]

    ctx2 = {
        "data_root": tmp_path, "midtrain_ckpt": None, "sft_ckpt": None,
        "dpo_ckpt": None, "grpo_ckpt": None, "final": None,
        "done_stages": set(), "eval_task_ids": ["flash_attn_decode_bf16"],
        "train_task_ids": ["rmsnorm_aiter", "gemm_bf16"],
        "lineage": lineage, "artifacts": {},
    }
    rc._load_manifest_into_ctx(ctx2)
    assert ctx2["train_task_ids"] == ["rmsnorm_aiter", "gemm_bf16"]
    assert ctx2["eval_task_ids"] == ["flash_attn_decode_bf16"]


def test_apply_split_overrides_a_stale_manifest_split():
    """audit R2: on a --force clean re-run the authoritative split must be recomputed
    from the LIVE registry, not reused from a stale manifest. _apply_split replaces any
    pre-seeded (stale) eval/train ids with the correct held-out probes (MLA/paged)."""
    ctx = {
        "tasks": [get_task("rmsnorm_aiter"), get_task("gemm_bf16"),
                  get_task("mla_decode_bf16"), get_task("paged_attn_decode_bf16")],
        "args": _args(["--tasks", "x"]),
        # STALE split as if loaded from a prior run's manifest (pre-MLA/paged fix)
        "eval_task_ids": ["flash_attn_decode_bf16", "flash_attn_prefill_bf16"],
        "train_task_ids": ["mla_decode_bf16"],  # stale: MLA wrongly in train
    }
    rc._apply_split(ctx)
    ev = set(ctx["eval_task_ids"])
    tr = set(ctx["train_task_ids"])
    assert {"mla_decode_bf16", "paged_attn_decode_bf16"} <= ev   # correct probes held out
    assert "flash_attn_decode_bf16" not in ev                    # stale entry gone
    assert "mla_decode_bf16" not in tr and "paged_attn_decode_bf16" not in tr  # not trained
    assert {"rmsnorm_aiter", "gemm_bf16"} <= tr


def test_rec_is_heldout_uses_registry_authority():
    # Core attention (flash decode) now TRAINS (product capability); the structurally
    # distinct paged-KV decode is the held-out generalization probe (registry HELDOUT_TASKS).
    attn_train = {"type": "repair", "task_id": "flash_attn_decode_bf16",
                  "operation": "flash_attn", "arch": "gfx950"}
    attn_held = {"type": "repair", "task_id": "paged_attn_decode_bf16",
                 "operation": "paged_attn", "arch": "gfx950"}
    rms = {"type": "repair", "task_id": "rmsnorm_aiter",
           "operation": "rmsnorm", "arch": "gfx950"}
    assert rc._rec_is_heldout(attn_train, set()) is False   # trains now
    assert rc._rec_is_heldout(attn_held, set()) is True     # held-out probe (HELDOUT_TASKS)
    assert rc._rec_is_heldout(rms, set()) is False
    # gfx942 (CDNA3, previous gen) is ACCEPTED into the train set (TRAIN_ARCHS
    # lineage) so a mid-flight campaign's legacy-tagged records keep training.
    assert rc._rec_is_heldout({"type": "repair", "operation": "rmsnorm",
                               "arch": "gfx942", "task_id": "x"}, set()) is False
    # a truly FOREIGN arch (e.g. RDNA / NVIDIA) is still held out.
    assert rc._rec_is_heldout({"type": "repair", "operation": "rmsnorm",
                               "arch": "gfx1100", "task_id": "x"}, set()) is True
    # explicit reserved id -> held out
    assert rc._rec_is_heldout(rms, {"rmsnorm_aiter"}) is True
    # MLA / paged-KV are held out by FAMILY now (audit R2), so a VARIANT record whose
    # task_id is not one of the two seed ids is still kept out of TRAIN, while core
    # attention (flash prefill/decode) keeps training.
    assert rc._rec_is_heldout({"type": "win", "operation": "mla_decode",
                               "task_id": "mla_variant_x", "arch": "gfx950"}, set()) is True
    assert rc._rec_is_heldout({"type": "win", "operation": "paged_attn_decode",
                               "task_id": "paged_variant_y", "arch": "gfx950"}, set()) is True
    assert rc._rec_is_heldout({"type": "win", "operation": "flash_attn_prefill",
                               "task_id": "flash_x", "arch": "gfx950"}, set()) is False


def test_build_trains_on_ALL_non_heldout_families_no_random_drop():
    """Regression for audit R2 crosscut C1: the build stage must retain EVERY
    non-held-out op family in TRAIN. The old random 80/10/10 leakage_split exiled
    ~20% of trainable families into val/test partitions nothing consumed -- silent
    data loss. The authoritative _rec_is_heldout filter is the ONLY holdout, so a
    record survives iff it is not reserved."""
    heldout_ids: set = set()
    # a wide spread of distinct NON-held-out op families on the train arch
    fams = ["relu", "add", "mul", "gelu", "silu", "softmax", "layernorm",
            "rmsnorm", "abs", "tanh", "sigmoid", "exp", "add_relu", "add_mul"]
    raw = [{"type": "win", "task_id": f"gen_{op}_fp16", "operation": op,
            "arch": "gfx950"} for op in fams]
    # same filter the build stage applies
    train = [r for r in raw if not rc._rec_is_heldout(r, heldout_ids)]
    kept = {r["operation"] for r in train}
    assert kept == set(fams)              # NOT ONE family randomly dropped
    assert len(train) == len(raw)         # every trainable record survives


# --------------------------------------------------------------------------- #
# 2. Iterative on-policy DPO + DAgger
# --------------------------------------------------------------------------- #
def _dpo_ctx(tmp_path, args):
    return {
        "data_root": tmp_path, "args": args, "dry": False, "base": "base_model",
        "tasks": [get_task("rmsnorm_aiter")], "train_tasks": [get_task("rmsnorm_aiter")],
        "sft_ckpt": "sft_ckpt", "eval_task_ids": [], "train_task_ids": ["rmsnorm_aiter"],
    }


def test_dpo_rounds_gt1_triggers_iterative_dpo(monkeypatch, tmp_path):
    import kore.data.onpolicy as onp

    calls = {}

    def fake_iter(rounds, policy_factory, tasks, env_factory, **kw):
        calls["rounds"] = rounds
        calls["kw"] = kw
        calls["tasks"] = list(tasks)
        return [SimpleNamespace(round=rounds - 1, policy_ckpt="final_dpo_ckpt")]

    monkeypatch.setattr(onp, "iterative_dpo", fake_iter)
    monkeypatch.setattr(rc, "_teacher", lambda args: object())
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)

    ctx = _dpo_ctx(tmp_path, _args(["--tasks", "rmsnorm_aiter", "--dpo-rounds", "3"]))
    rc._stage_dpo(ctx)

    assert calls["rounds"] == 3
    assert calls["kw"]["aggregate"] is True
    assert callable(calls["kw"]["train_fn"])
    assert ctx["dpo_ckpt"] == "final_dpo_ckpt"


def test_dpo_rounds_eq1_uses_single_pass(monkeypatch, tmp_path):
    import kore.data.onpolicy as onp
    import kore.policy.dpo as dpo_mod

    seen = {"iter": False}
    monkeypatch.setattr(onp, "iterative_dpo",
                        lambda *a, **k: seen.__setitem__("iter", True) or [])
    monkeypatch.setattr(dpo_mod, "train", lambda cfg: {"output_dir": "single_ckpt"})
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)

    ctx = _dpo_ctx(tmp_path, _args(["--tasks", "rmsnorm_aiter", "--dpo-rounds", "1"]))
    rc._stage_dpo(ctx)

    assert seen["iter"] is False
    assert ctx["dpo_ckpt"] == "single_ckpt"


def test_dagger_fold_appends_to_sft_corpus(monkeypatch, tmp_path):
    import kore.data.build_datasets as bd
    import kore.data.onpolicy as onp
    import kore.env.kore_env as ke

    monkeypatch.setattr(ke, "KoreEnv", lambda task: object())
    # one mined+repaired failure per task
    monkeypatch.setattr(onp, "dagger_repairs", lambda *a, **k: [{"type": "repair", "x": 1}])
    monkeypatch.setattr(bd, "build_sft",
                        lambda recs: [{"messages": [{"role": "user", "content": "q"}]}
                                      for _ in recs])

    args = _args(["--tasks", "rmsnorm_aiter,gemm_bf16"])
    ctx = {"data_root": tmp_path, "args": args,
           "train_tasks": [get_task("rmsnorm_aiter"), get_task("gemm_bf16")]}
    n = rc._dagger_fold_into_sft(ctx, policy=object(), teacher=object(),
                                 round_idx=1, rounds=2)
    assert n == 2  # one repair per train task
    sft_file = tmp_path / "sft" / "multicap.jsonl"
    assert sft_file.exists()
    assert len(sft_file.read_text().strip().splitlines()) == 2
    assert (tmp_path / "dagger" / "round1.jsonl").exists()


# --------------------------------------------------------------------------- #
# 3. Evolutionary datagen stage
# --------------------------------------------------------------------------- #
def test_evolve_stage_callable_writes_shards(monkeypatch, tmp_path):
    import kore.data.evolve as ev
    import kore.env.kore_env as ke

    monkeypatch.setattr(ke, "KoreEnv", lambda task: object())
    monkeypatch.setattr(rc, "_teacher", lambda args: object())

    captured = {}

    def fake_evolve(task, generator, env, generations, cfg):
        captured["generations"] = generations
        return SimpleNamespace(
            wins=[{"type": "win", "task_id": task.task_id}],
            groups=[{"type": "ranked_group", "task_id": task.task_id}],
            stats={"best_speedup": 1.5},
        )

    monkeypatch.setattr(ev, "evolve_task", fake_evolve)

    args = _args(["--tasks", "rmsnorm_aiter,gemm_bf16", "--evolve-generations", "2"])
    ctx = {"data_root": tmp_path, "args": args, "dry": False,
           "tasks": [get_task("rmsnorm_aiter"), get_task("gemm_bf16")],
           "train_tasks": [get_task("rmsnorm_aiter"), get_task("gemm_bf16")]}
    rc._stage_evolve(ctx)

    assert captured["generations"] == 2
    assert (tmp_path / "wins" / "rmsnorm_aiter.evolve.jsonl").exists()
    assert (tmp_path / "groups" / "rmsnorm_aiter.evolve.jsonl").exists()
    assert (tmp_path / "wins" / "gemm_bf16.evolve.jsonl").exists()


def test_evolve_stage_spliced_after_datagen():
    # --evolve splices the stage in right after datagen in the default plan.
    args = _args(["--tasks", "rmsnorm_aiter", "--evolve"])
    stages = list(rc.DEFAULT_STAGES)
    if args.evolve and "evolve" not in stages:
        stages.insert(stages.index("datagen") + 1, "evolve")
    assert stages[stages.index("datagen") + 1] == "evolve"
    assert "evolve" not in rc.DEFAULT_STAGES  # not on by default


# --------------------------------------------------------------------------- #
# 4. Correctness -> latency GRPO curriculum
# --------------------------------------------------------------------------- #
def _grpo_ctx(tmp_path, args):
    return {
        "data_root": tmp_path, "args": args, "dry": False, "base": "base_model",
        "tasks": [get_task("rmsnorm_aiter")], "train_tasks": [get_task("rmsnorm_aiter")],
        "sft_ckpt": "sft_ckpt", "dpo_ckpt": "dpo_ckpt", "eval_task_ids": [],
    }


def test_grpo_curriculum_runs_two_phases(monkeypatch, tmp_path):
    import kore.policy.grpo as grpo_mod

    seen = []

    def fake_train(cfg, tasks=None, backend="inprocess"):
        seen.append((cfg.reward_phase, cfg.model_id, cfg.output_dir))
        return cfg.output_dir + "/ckpt"

    monkeypatch.setattr(grpo_mod, "train_grpo", fake_train)
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)

    # curriculum defaults ON
    ctx = _grpo_ctx(tmp_path, _args(["--tasks", "rmsnorm_aiter", "--grpo-out", "runs/grpo"]))
    rc._stage_grpo(ctx)

    assert len(seen) == 2
    assert seen[0][0] == "correctness"
    assert seen[1][0] == "latency"
    # phase-2 initializes FROM the phase-1 checkpoint
    phase1_ckpt = seen[0][2] + "/ckpt"
    assert seen[1][1] == phase1_ckpt
    assert ctx["grpo_ckpt"] == seen[1][2] + "/ckpt"


def test_grpo_single_phase_when_curriculum_off(monkeypatch, tmp_path):
    import kore.policy.grpo as grpo_mod

    seen = []
    monkeypatch.setattr(grpo_mod, "train_grpo",
                        lambda cfg, tasks=None, backend="inprocess":
                        seen.append(cfg.reward_phase) or "grpo_ckpt")
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)

    ctx = _grpo_ctx(tmp_path, _args(["--tasks", "rmsnorm_aiter", "--no-grpo-curriculum"]))
    rc._stage_grpo(ctx)
    assert seen == ["all"]
    assert ctx["grpo_ckpt"] == "grpo_ckpt"


def test_grpo_trains_only_on_train_split(monkeypatch, tmp_path):
    import kore.policy.grpo as grpo_mod

    seen_tasks = []
    monkeypatch.setattr(grpo_mod, "train_grpo",
                        lambda cfg, tasks=None, backend="inprocess":
                        seen_tasks.append(list(tasks or [])) or (cfg.output_dir + "/ckpt"))
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)

    ctx = _grpo_ctx(tmp_path, _args(["--tasks", "rmsnorm_aiter", "--no-grpo-curriculum"]))
    ctx["eval_task_ids"] = ["flash_attn_decode_bf16"]  # held out; must not train on it
    rc._stage_grpo(ctx)
    assert seen_tasks[0] == ["rmsnorm_aiter"]
    assert "flash_attn_decode_bf16" not in seen_tasks[0]


# --------------------------------------------------------------------------- #
# 5. assemble folds on-policy / evolve / DAgger records
# --------------------------------------------------------------------------- #
def test_assemble_multicap_folds_extra_records(tmp_path, monkeypatch):
    monkeypatch.setenv("KORE_DECONTAM_DEVELOPMENT", "1")
    from kore.data import assemble
    from kore.data.schemas import WinRecord
    from kore.data.teacher import StubTeacher
    from kore.policy.configs import MultiCapSFTConfig

    win = WinRecord(
        task_id="gemm_bf16",
        trajectory=[{"role": "assistant",
                     "content": "FULL_KERNEL:\n```python\ndef c():\n    return 3\n```"}],
        initial_wall_us=200.0, final_wall_us=100.0, speedup=2.0,
        final_source="def c():\n    return 3",
    )
    cfg = MultiCapSFTConfig()
    base = assemble.assemble_multicap_sources(tmp_path, [], StubTeacher(), cfg,
                                              total=100, kernel_records=[])
    withx = assemble.assemble_multicap_sources(tmp_path, [], StubTeacher(), cfg,
                                               total=100, kernel_records=[],
                                               extra_records=[win])
    assert len(withx["kernel_repair_opt"]) == len(base["kernel_repair_opt"]) + 1


def test_assemble_dpo_folds_extra_group_records(tmp_path):
    from kore.data import assemble
    from kore.data.schemas import RankedGroupRecord
    from kore.tasks.registry import all_tasks

    grp = RankedGroupRecord(
        task_id="gemm_bf16", parent_id="p",
        candidates=[
            {"source": "def a():\n    return 1", "wall_us": 100.0, "snr_db": 40.0, "rank": 0},
            {"source": "def b():\n    return 2", "wall_us": 200.0, "snr_db": 39.0, "rank": 1},
        ],
        preferences=[[0, 1]],
    )
    tasks = all_tasks()[:2]
    base = assemble.build_dpo_with_hard_negatives(tmp_path, tasks)
    withx = assemble.build_dpo_with_hard_negatives(tmp_path, tasks,
                                                   extra_group_records=[grp])
    assert withx["n_total"] == base["n_total"] + 1


# --------------------------------------------------------------------------- #
# 6. Dry-run import preflight includes the new symbols
# --------------------------------------------------------------------------- #
def test_preflight_includes_new_symbols():
    names = {(mod, attr) for (mod, attr, _req, _params) in rc._IMPORT_CHECKS}
    for sym in [
        ("kore.tasks.registry", "split_tasks"),
        ("kore.tasks.registry", "train_tasks"),
        ("kore.tasks.registry", "heldout_tasks"),
        ("kore.tasks.registry", "operator_family"),
        ("kore.data.onpolicy", "iterative_dpo"),
        ("kore.data.onpolicy", "dagger_repairs"),
        ("kore.data.onpolicy", "dagger_teacher_frac"),
        ("kore.data.evolve", "evolve_task"),
        ("kore.policy.grpo", "apply_reward_phase"),
        ("kore.data.assemble", "build_multicap_dataset"),
        ("kore.data.assemble", "build_dpo_with_hard_negatives"),
        # Fix 4: the real-run-only symbols the audit found were previously
        # imported lazily inside stage bodies and never preflight-checked.
        ("kore.data.gen_repair", "generate_repairs"),
        ("kore.data.gen_groups", "generate_groups"),
        ("kore.data.gen_wins", "generate_wins"),
        ("kore.data.gen_agentic", "generate_agentic_trajectories"),
        ("kore.data.schemas", "write_jsonl"),
        ("kore.data.teacher", "make_teacher"),
        ("kore.data.teacher", "load_env_local"),
        ("kore.data.build_datasets", "build_sft"),
        ("kore.agent.harness", "AgentHarness"),
        ("kore.agent.tools", "tool_use_reward"),
        ("kore.policy.anticollapse", "avspo_advantages"),
        ("kore.policy.anticollapse", "scgrpo_weight_from_kl"),
        ("kore.policy.anticollapse", "gtpo_codesim_shaping"),
        ("kore.policy.anticollapse", "variance_floor"),
        ("kore.value.rerank", "rank_candidates"),
    ]:
        assert sym in names, f"preflight missing {sym}"


def test_preflight_passes_clean():
    # every required symbol imports + has the required params (no drift) -> no raise.
    rc._dry_import_check()


# --------------------------------------------------------------------------- #
# 7. --full-ft engages FSDP UNDER THE HOOD (Fix 1): distributed=True + the
#    campaign shells out to scripts/launch_distributed.sh (subprocess) for the
#    stages whose `-m` JSON entry supports it (sft/dpo/grpo), and falls back
#    in-process with a LOUD warning for the sibling-owned stage (midtrain).
# --------------------------------------------------------------------------- #
def _capture_subprocess(monkeypatch):
    calls = []

    def fake_run(cmd, check=False, **kw):
        cmdl = list(cmd)
        # Capture only the DISTRIBUTED LAUNCHER invocations. Ignore incidental
        # subprocess calls (e.g. the `rocm-smi` free-GPU auto-detection), which are
        # correct behavior but not what these launcher-wiring tests assert.
        if cmdl and cmdl[0] == "bash" and any("launch_distributed" in str(c) for c in cmdl):
            calls.append({"cmd": cmdl, "check": check})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    return calls


def test_full_ft_sft_invokes_launcher_and_sets_distributed(monkeypatch, tmp_path):
    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)

    args = _args(["--tasks", "rmsnorm_aiter", "--full-ft", "--sft-out", "runs/sft"])
    ctx = {"data_root": tmp_path, "args": args, "dry": False, "base": "base_model",
           "tasks": [get_task("rmsnorm_aiter")], "train_tasks": [get_task("rmsnorm_aiter")],
           "midtrain_ckpt": "midtrain_ckpt"}
    rc._stage_sft(ctx)

    # the launcher was invoked via subprocess, NOT the in-process trainer.
    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert cmd[0] == "bash" and cmd[1].endswith("scripts/launch_distributed.sh")
    assert cmd[2] == "sft"
    assert calls[0]["check"] is True
    # the rendered config forces distributed=True + use_lora=False and threads the
    # run's dynamic paths (model = the midtrain ckpt, dataset, output_dir).
    written = json.loads((tmp_path / "launch" / "sft.json").read_text())
    assert written["distributed"] is True
    assert written["use_lora"] is False
    assert written["model_id"] == "midtrain_ckpt"
    assert written["dataset_path"].endswith("sft/multicap.jsonl")
    assert ctx["sft_ckpt"] == "runs/sft"


def test_lora_sft_stays_in_process_no_launcher(monkeypatch, tmp_path):
    # the DEFAULT (LoRA) path never shells out - pure single-process one command.
    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)
    import kore.policy.sft as sft_mod
    seen = {}
    monkeypatch.setattr(sft_mod, "train_sft",
                        lambda cfg, ds: seen.update(use_lora=cfg.use_lora) or "runs/sft")

    args = _args(["--tasks", "rmsnorm_aiter", "--sft-out", "runs/sft"])  # LoRA default
    ctx = {"data_root": tmp_path, "args": args, "dry": False, "base": "base_model",
           "tasks": [get_task("rmsnorm_aiter")], "train_tasks": [get_task("rmsnorm_aiter")],
           "midtrain_ckpt": None}
    rc._stage_sft(ctx)
    assert calls == []            # no subprocess / launcher
    assert seen["use_lora"] is True
    assert ctx["sft_ckpt"] == "runs/sft"


def test_full_ft_dpo_single_invokes_launcher(monkeypatch, tmp_path):
    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)

    ctx = _dpo_ctx(tmp_path, _args(["--tasks", "rmsnorm_aiter", "--dpo-rounds", "1",
                                    "--full-ft"]))
    rc._stage_dpo(ctx)

    assert len(calls) == 1 and calls[0]["cmd"][2] == "dpo"
    written = json.loads((tmp_path / "launch" / "dpo.json").read_text())
    assert written["distributed"] is True and written["use_lora"] is False
    assert written["model_id"] == "sft_ckpt"
    assert ctx["dpo_ckpt"] == ctx["args"].dpo_out


def test_full_ft_dpo_iterative_shells_out_per_round(monkeypatch, tmp_path):
    import kore.data.onpolicy as onp

    # drive the iterative loop's train_fn once with a fake round, capturing the
    # per-round launcher shell-out.
    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(rc, "_teacher", lambda args: object())
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)

    def fake_iter(rounds, policy_factory, tasks, env_factory, **kw):
        rd = SimpleNamespace(round=1, ref_model_id="round0_ckpt", dpo_pairs=[{"p": 1}],
                             n_pairs=1)
        out = kw["train_fn"](rd)
        return [SimpleNamespace(round=1, policy_ckpt=out)]

    monkeypatch.setattr(onp, "iterative_dpo", fake_iter)

    ctx = _dpo_ctx(tmp_path, _args(["--tasks", "rmsnorm_aiter", "--dpo-rounds", "3",
                                    "--full-ft"]))
    rc._stage_dpo(ctx)

    assert len(calls) == 1 and calls[0]["cmd"][2] == "dpo"
    written = json.loads((tmp_path / "launch" / "dpo_round1.json").read_text())
    assert written["distributed"] is True and written["use_lora"] is False
    # iterative DPO keeps the SFT anchor (IPO+SFT composite), never a bare "ipo" that
    # can still collapse likelihoods; loss_weights arity matches (R2 dpo C1).
    assert written["loss_type"] == ["ipo", "sft"]
    assert written["loss_weights"] == [1.0, 1.0]
    assert written["ref_model_id"] == "round0_ckpt"


def _grpo_launcher_supported(monkeypatch):
    """Simulate grpo shipping the JSON `-m` entry (grpo_config_from_dict), so the
    campaign routes --full-ft grpo through the FSDP launcher exactly like sft/dpo.
    Forward-compatible: once the sibling entry actually lands this is a no-op."""
    monkeypatch.setattr(rc, "_stage_supports_launcher",
                        lambda stage: True if stage == "grpo" else rc._stage_supports_launcher(stage))


def test_grpo_supported_by_launcher_detection(monkeypatch):
    # _stage_supports_launcher flips True for grpo the moment the sibling ships
    # grpo_config_from_dict (the JSON `-m` builder) - no campaign change needed.
    import kore.policy.grpo as grpo_mod

    monkeypatch.setattr(grpo_mod, "grpo_config_from_dict", lambda d: object(),
                        raising=False)
    assert rc._stage_supports_launcher("grpo") is True


def test_full_ft_grpo_invokes_launcher_full_param(monkeypatch, tmp_path):
    # Under --full-ft the GRPO RL stage runs FULL-PARAMETER + SHARDED via the
    # launcher (no LoRA shortcut): subprocess is invoked and the rendered config
    # forces distributed=True + use_lora=False, threading model/tasks through.
    monkeypatch.setattr(rc, "_stage_supports_launcher", lambda stage: True)
    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)

    ctx = _grpo_ctx(tmp_path, _args(["--tasks", "rmsnorm_aiter", "--no-grpo-curriculum",
                                     "--full-ft", "--grpo-out", "runs/grpo"]))
    ctx["train_task_ids"] = ["rmsnorm_aiter"]
    rc._stage_grpo(ctx)

    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert cmd[0] == "bash" and cmd[1].endswith("scripts/launch_distributed.sh")
    assert cmd[2] == "grpo"
    written = json.loads((tmp_path / "launch" / "grpo.json").read_text())
    assert written["distributed"] is True          # sharded full-param
    assert written["use_lora"] is False            # NO LoRA shortcut under --full-ft
    assert written["model_id"] == "dpo_ckpt"       # init = dpo ckpt (or sft)
    assert written["reward_phase"] == "all"
    assert written["tasks"] == ["rmsnorm_aiter"]   # TRAIN-split tasks travel in the JSON
    assert ctx["grpo_ckpt"] == "runs/grpo"


def test_full_ft_grpo_curriculum_two_phases_under_launcher(monkeypatch, tmp_path):
    # The correctness->latency curriculum under --full-ft = TWO launched
    # full-parameter GRPO runs, phase-1 checkpoint threaded into phase-2 init.
    monkeypatch.setattr(rc, "_stage_supports_launcher", lambda stage: True)
    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)

    # curriculum defaults ON
    ctx = _grpo_ctx(tmp_path, _args(["--tasks", "rmsnorm_aiter", "--full-ft",
                                     "--grpo-out", "runs/grpo"]))
    ctx["train_task_ids"] = ["rmsnorm_aiter"]
    rc._stage_grpo(ctx)

    # two launcher shell-outs, both to the grpo stage
    assert len(calls) == 2
    assert all(c["cmd"][2] == "grpo" for c in calls)

    p1 = json.loads((tmp_path / "launch" / "grpo_phase1_correctness.json").read_text())
    p2 = json.loads((tmp_path / "launch" / "grpo_phase2_latency.json").read_text())
    assert p1["reward_phase"] == "correctness" and p1["model_id"] == "dpo_ckpt"
    assert p1["distributed"] is True and p1["use_lora"] is False
    # phase-2 initializes FROM the phase-1 checkpoint (its output_dir)
    assert p2["reward_phase"] == "latency"
    assert p2["model_id"] == "runs/grpo/phase1_correctness"
    assert p2["distributed"] is True and p2["use_lora"] is False
    assert ctx["grpo_ckpt"] == "runs/grpo"


def test_lora_grpo_stays_in_process_no_launcher(monkeypatch, tmp_path):
    # --lora (default) keeps GRPO single-process in-process (LoRA bring-up),
    # never shelling out to the launcher.
    import kore.policy.grpo as grpo_mod

    _grpo_launcher_supported(monkeypatch)  # even if grpo COULD shard, LoRA stays local
    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(grpo_mod, "train_grpo",
                        lambda cfg, tasks=None: seen.append(cfg) or (cfg.output_dir + "/ckpt"))

    ctx = _grpo_ctx(tmp_path, _args(["--tasks", "rmsnorm_aiter", "--no-grpo-curriculum"]))
    rc._stage_grpo(ctx)

    assert calls == []                          # no subprocess / launcher
    assert seen[0].use_lora is True             # LoRA bring-up path
    assert getattr(seen[0], "distributed", False) is False


# --------------------------------------------------------------------------- #
# 8. Anti-collapse + efficiency levers ON by default (Fix 2)
# --------------------------------------------------------------------------- #
def _run_grpo_capture_cfg(monkeypatch, tmp_path, argv):
    import kore.policy.grpo as grpo_mod

    seen = []
    # NOTE: no `backend=` kwarg - Fix 3 removed the verl-era backend switch, so the
    # campaign must call train_grpo(cfg, tasks=...) only. A stray backend arg would
    # blow up this signature.
    monkeypatch.setattr(grpo_mod, "train_grpo",
                        lambda cfg, tasks=None: seen.append(cfg) or (cfg.output_dir + "/ckpt"))
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)
    ctx = _grpo_ctx(tmp_path, _args(argv))
    rc._stage_grpo(ctx)
    return seen


def test_grpo_levers_on_by_default(monkeypatch, tmp_path):
    seen = _run_grpo_capture_cfg(
        monkeypatch, tmp_path, ["--tasks", "rmsnorm_aiter", "--no-grpo-curriculum"])
    cfg = seen[0]
    # anti-collapse ladder
    assert cfg.rc_grpo is True
    assert cfg.sc_grpo is True
    assert cfg.gtpo_codesim is True
    assert cfg.variance_floor > 0.0
    # measurement efficiency + agentic + StarPO-S
    assert cfg.value_prefilter is True
    assert cfg.agentic is True
    assert cfg.starpo_s is True


def test_grpo_levers_can_be_disabled(monkeypatch, tmp_path):
    seen = _run_grpo_capture_cfg(
        monkeypatch, tmp_path,
        ["--tasks", "rmsnorm_aiter", "--no-grpo-curriculum",
         "--no-anticollapse", "--no-value-prefilter"])
    cfg = seen[0]
    assert cfg.rc_grpo is False
    assert cfg.sc_grpo is False
    assert cfg.gtpo_codesim is False
    assert cfg.variance_floor == 0.0
    assert cfg.value_prefilter is False


def test_grpo_value_model_path_threads_through(monkeypatch, tmp_path):
    seen = _run_grpo_capture_cfg(
        monkeypatch, tmp_path,
        ["--tasks", "rmsnorm_aiter", "--no-grpo-curriculum",
         "--value-model-path", "runs/value/model.json"])
    assert seen[0].value_prefilter is True
    assert seen[0].value_model_path == "runs/value/model.json"


# --------------------------------------------------------------------------- #
# 9. Fix 3: the verl-era --grpo-backend flag is gone (no dangling backend switch)
# --------------------------------------------------------------------------- #
def test_grpo_backend_flag_removed():
    args = _args([])
    assert not hasattr(args, "grpo_backend")
    with pytest.raises(SystemExit):
        _args(["--grpo-backend", "fallback"])


# --------------------------------------------------------------------------- #
# 10. Full-FT midtrain: shells out to the FSDP launcher (JSON `-m` entry now
#     ships via midtrain_config_from_dict) - real full-parameter sharded.
# --------------------------------------------------------------------------- #
def test_full_ft_midtrain_invokes_launcher_full_param(monkeypatch, tmp_path):
    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)

    # pre-create the corpus so the (heavy) corpus build is skipped.
    corpus = tmp_path / "midtrain" / "corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text('{"text": "x"}\n')

    args = _args(["--tasks", "rmsnorm_aiter", "--full-ft", "--midtrain-out", "runs/midtrain"])
    ctx = {"data_root": tmp_path, "args": args, "dry": False, "base": "base_model",
           "tasks": [get_task("rmsnorm_aiter")], "train_tasks": [get_task("rmsnorm_aiter")]}
    rc._stage_midtrain(ctx)

    # the launcher was invoked via subprocess, NOT the in-process trainer.
    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert cmd[0] == "bash" and cmd[1].endswith("scripts/launch_distributed.sh")
    assert cmd[2] == "midtrain"
    assert calls[0]["check"] is True
    # the rendered config forces distributed=True + use_lora=False and threads the
    # run's dynamic paths (base model, corpus, output_dir).
    written = json.loads((tmp_path / "launch" / "midtrain.json").read_text())
    assert written["distributed"] is True
    assert written["use_lora"] is False
    assert written["model_id"] == "base_model"
    assert written["corpus_path"].endswith("midtrain/corpus.jsonl")
    assert ctx["midtrain_ckpt"] == "runs/midtrain"


def test_lora_midtrain_stays_in_process_no_launcher(monkeypatch, tmp_path):
    # the DEFAULT (LoRA) path never shells out - pure single-process one command.
    import kore.policy.midtrain as mt

    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)
    seen = {}

    def fake_train(cfg, corpus_path=None):
        seen["use_lora"] = cfg.use_lora
        return "midtrain_ckpt"

    monkeypatch.setattr(mt, "train_midtrain", fake_train)
    corpus = tmp_path / "midtrain" / "corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text('{"text": "x"}\n')

    args = _args(["--tasks", "rmsnorm_aiter", "--midtrain-out", "runs/midtrain"])  # LoRA default
    ctx = {"data_root": tmp_path, "args": args, "dry": False, "base": "base_model",
           "tasks": [get_task("rmsnorm_aiter")], "train_tasks": [get_task("rmsnorm_aiter")]}
    rc._stage_midtrain(ctx)

    assert seen["use_lora"] is True
    assert calls == []                     # LoRA -> in-process, no launcher
    assert ctx["midtrain_ckpt"] == "midtrain_ckpt"
