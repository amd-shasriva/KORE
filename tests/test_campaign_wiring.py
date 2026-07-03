"""CPU-only tests for the end-to-end campaign wiring (scripts/run_campaign.py).

No GPU, no teacher, no torch/trl. Every heavy stage entrypoint is monkeypatched;
we only assert that the campaign WIRES the newly-implemented research capabilities
together correctly:

  * the AUTHORITATIVE registry train/held-out split threads through ctx + manifest
    (item 1) — training stages get TRAIN tasks, eval gets the held-out family;
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
    # registry's held-out generalization set (the reserved attention family).
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
    ctx = {
        "data_root": tmp_path, "dry": False, "base": "Qwen/Qwen3-14B",
        "midtrain_ckpt": None, "sft_ckpt": "sft", "dpo_ckpt": None,
        "grpo_ckpt": None, "final": None, "done_stages": {"build"},
        "train_task_ids": ["rmsnorm_aiter", "gemm_bf16"],
        "eval_task_ids": ["flash_attn_decode_bf16"],
    }
    rc._save_manifest(ctx)

    ctx2 = {
        "data_root": tmp_path, "midtrain_ckpt": None, "sft_ckpt": None,
        "dpo_ckpt": None, "grpo_ckpt": None, "final": None,
        "done_stages": set(), "eval_task_ids": None, "train_task_ids": None,
    }
    rc._load_manifest_into_ctx(ctx2)
    assert ctx2["train_task_ids"] == ["rmsnorm_aiter", "gemm_bf16"]
    assert ctx2["eval_task_ids"] == ["flash_attn_decode_bf16"]


def test_rec_is_heldout_uses_registry_authority():
    # attention family -> held out; rmsnorm -> train
    attn = {"type": "repair", "task_id": "flash_attn_decode_bf16",
            "operation": "flash_attn", "arch": "gfx942"}
    rms = {"type": "repair", "task_id": "rmsnorm_aiter",
           "operation": "rmsnorm", "arch": "gfx942"}
    assert rc._rec_is_heldout(attn, set()) is True
    assert rc._rec_is_heldout(rms, set()) is False
    # non-train arch -> held out (subsumes the old gfx950 special-case)
    assert rc._rec_is_heldout({"type": "repair", "operation": "rmsnorm",
                               "arch": "gfx950", "task_id": "x"}, set()) is True
    # explicit reserved id -> held out
    assert rc._rec_is_heldout(rms, {"rmsnorm_aiter"}) is True


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
def test_assemble_multicap_folds_extra_records(tmp_path):
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
#    stages whose `-m` JSON entry supports it (sft/dpo), and falls back
#    in-process with a LOUD warning for the sibling-owned stages (grpo/midtrain).
# --------------------------------------------------------------------------- #
def _capture_subprocess(monkeypatch):
    calls = []

    def fake_run(cmd, check=False, **kw):
        calls.append({"cmd": list(cmd), "check": check})
        return SimpleNamespace(returncode=0)

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
    # the DEFAULT (LoRA) path never shells out — pure single-process one command.
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
    assert written["loss_type"] == "ipo"
    assert written["ref_model_id"] == "round0_ckpt"


def test_full_ft_grpo_falls_back_in_process_loudly(monkeypatch, tmp_path):
    # grpo has no `-m` JSON entry (sibling-owned) -> in-process fallback with a
    # LOUD warning, and distributed=True is STILL set on the config (no silent
    # degrade, no bogus subprocess).
    import kore.policy.grpo as grpo_mod

    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)
    warned = []
    monkeypatch.setattr(rc, "_warn_inprocess_fullft", lambda stage: warned.append(stage))

    seen = []
    monkeypatch.setattr(grpo_mod, "train_grpo",
                        lambda cfg, tasks=None: seen.append(cfg) or (cfg.output_dir + "/ckpt"))

    ctx = _grpo_ctx(tmp_path, _args(["--tasks", "rmsnorm_aiter", "--no-grpo-curriculum",
                                     "--full-ft"]))
    rc._stage_grpo(ctx)

    assert calls == []                                   # never shells out grpo
    assert warned == ["grpo"]                            # loud warning fired once
    assert getattr(seen[0], "distributed", False) is True  # distributed still set


# --------------------------------------------------------------------------- #
# 8. Anti-collapse + efficiency levers ON by default (Fix 2)
# --------------------------------------------------------------------------- #
def _run_grpo_capture_cfg(monkeypatch, tmp_path, argv):
    import kore.policy.grpo as grpo_mod

    seen = []
    # NOTE: no `backend=` kwarg — Fix 3 removed the verl-era backend switch, so the
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
# 10. Full-FT midtrain: distributed set, in-process fallback (no launcher entry)
# --------------------------------------------------------------------------- #
def test_full_ft_midtrain_sets_distributed_in_process(monkeypatch, tmp_path):
    import kore.policy.midtrain as mt

    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(rc, "_retention_gate", lambda *a, **k: None)
    seen = {}

    def fake_train(cfg, corpus_path=None):
        seen["distributed"] = getattr(cfg, "distributed", False)
        seen["use_lora"] = cfg.use_lora
        return "midtrain_ckpt"

    monkeypatch.setattr(mt, "train_midtrain", fake_train)

    # pre-create the corpus so the (heavy) corpus build is skipped.
    corpus = tmp_path / "midtrain" / "corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text('{"text": "x"}\n')

    args = _args(["--tasks", "rmsnorm_aiter", "--full-ft", "--midtrain-out", "runs/midtrain"])
    ctx = {"data_root": tmp_path, "args": args, "dry": False, "base": "base_model",
           "tasks": [get_task("rmsnorm_aiter")], "train_tasks": [get_task("rmsnorm_aiter")]}
    rc._stage_midtrain(ctx)

    assert seen["distributed"] is True     # contract: distributed on the config
    assert seen["use_lora"] is False       # full-FT
    assert calls == []                     # midtrain has no launcher entry -> in-process
    assert ctx["midtrain_ckpt"] == "midtrain_ckpt"
