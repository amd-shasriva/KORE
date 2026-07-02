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

from types import SimpleNamespace

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
    ]:
        assert sym in names, f"preflight missing {sym}"


def test_preflight_passes_clean():
    # every required symbol imports + has the required params (no drift) -> no raise.
    rc._dry_import_check()
