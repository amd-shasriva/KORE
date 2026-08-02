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

Section 11 covers the four subsystems that shipped a tested production writer or
API with NO caller, which is worse than shipping nothing: a reader reasonably
assumes a tested module is doing something. Each of those tests pins the CALL -
the frozen held-out shape lane being written before any stage runs, the five
evaluation budget counters being charged through ``KoreEnv.evaluate``, hardware
eligibility narrowing datagen selection only when asked, and the KernelBench
claim gate being conjuncted into the track verdict.
"""

from __future__ import annotations

import json
import os
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


def test_all_heldout_selection_refuses_fallback_training():
    # taxonomy (fail-closed split): a selection that is ENTIRELY held-out must NOT
    # silently fall back to training on the eval probes -- the split is authored
    # fail-closed and raises SplitManifestError with an empty train split.
    from kore.tasks.registry import SplitManifestError
    held = heldout_tasks()[0]
    ctx = {"tasks": [held], "args": _args(["--tasks", held.task_id])}
    with pytest.raises(SplitManifestError, match="train split is empty"):
        rc._apply_split(ctx)


def test_manifest_threads_train_and_eval_ids(tmp_path):
    # frontier-integration: the campaign manifest carries a complete versioned
    # lineage contract (schema {"name": "kore.campaign", "version": 1}); the
    # train/eval task ids live under lineage.tasks and round-trip through
    # _save_manifest / _load_manifest_into_ctx.
    # taxonomy (fail-closed split): the manifest ALSO carries a versioned
    # split_manifest whose taxonomy {version, digest} must be present so a stale
    # split can be detected on resume.
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
    # taxonomy: the versioned split_manifest is persisted alongside the lineage.
    assert persisted["split_manifest"]["taxonomy"]["version"]
    assert persisted["split_manifest"]["taxonomy"]["digest"]

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
    # taxonomy: the loaded ctx exposes the validated split_manifest taxonomy stamp.
    assert ctx2["split_manifest"]["taxonomy"]["version"]
    assert ctx2["split_manifest"]["taxonomy"]["digest"]


def test_stale_campaign_manifest_is_invalidated(tmp_path):
    # taxonomy (fail-closed split): a manifest whose persisted taxonomy digest no
    # longer matches the LIVE taxonomy must be rejected on load -- never silently
    # reused -- so a stale train/eval split cannot corrupt a resumed campaign.
    from kore.tasks.registry import StaleSplitManifestError
    args = _args(["--tasks", "rmsnorm_aiter"])
    ctx = {
        "data_root": tmp_path,
        "dry": False,
        "base": "base",
        "midtrain_ckpt": None,
        "sft_ckpt": None,
        "dpo_ckpt": None,
        "grpo_ckpt": None,
        "final": None,
        "done_stages": set(),
        "tasks": [get_task("rmsnorm_aiter")],
        "args": args,
        # The merged campaign requires complete lineage before a manifest is written
        # (frontier lineage hardening); a consistent stub suffices to exercise the
        # taxonomy stale-split rejection on load.
        "lineage": {"compatibility_digest": "test-lineage",
                    "tasks": {"train": ["rmsnorm_aiter"], "eval": []}},
    }
    rc._apply_split(ctx)
    rc._save_manifest(ctx)
    path = tmp_path / "campaign_manifest.json"
    payload = json.loads(path.read_text())
    payload["split_manifest"]["taxonomy"]["digest"] = "stale"
    path.write_text(json.dumps(payload))
    with pytest.raises(StaleSplitManifestError, match="taxonomy digest changed"):
        rc._load_manifest_into_ctx(ctx)


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
    """The stage publishes one ``.evolve.jsonl`` pair per TRAIN task.

    The records are real ``WinRecord``/``RankedGroupRecord``s and the teacher is the
    stub, because the stage now publishes into the PRODUCTION record lane behind a
    generation contract (it used to write bare JSONL that the production build
    reader rejected and that no resume receipt covered). Skeletal fakes and a
    teacher with no immutable revision are both refused, exactly as in datagen.
    """
    import kore.data.evolve as ev
    import kore.env.kore_env as ke
    from kore.data.schemas import RankedGroupRecord, WinRecord

    monkeypatch.setattr(ke, "KoreEnv", lambda task: object())
    monkeypatch.setattr(rc, "_teacher", lambda args: object())

    captured = {}

    def fake_evolve(task, generator, env, generations, cfg):
        captured["generations"] = generations
        source = f"import triton\n\n\ndef k_{task.task_id}(x):\n    return x + 1\n"
        return SimpleNamespace(
            wins=[WinRecord(
                task_id=task.task_id,
                trajectory=[{"role": "user", "content": "optimize"},
                            {"role": "assistant", "content": f"FULL_KERNEL:\n{source}"}],
                initial_wall_us=200.0, final_wall_us=100.0, speedup=2.0,
                final_source=source, operation=task.operation, arch="gfx950")],
            groups=[RankedGroupRecord(
                task_id=task.task_id, parent_id=f"parent-{task.task_id}",
                candidates=[{"source": source + f"# {i}\n", "wall_us": 100.0 * (i + 1),
                             "snr_db": 40.0, "rank": i} for i in range(2)],
                preferences=[[0, 1]], operation=task.operation, arch="gfx950")],
            stats={"best_speedup": 1.5},
        )

    monkeypatch.setattr(ev, "evolve_task", fake_evolve)

    args = _args(["--tasks", "rmsnorm_aiter,gemm_bf16", "--evolve-generations", "2",
                  "--teacher", "stub"])
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
    # frontier-integration: decontamination runs in development mode so the
    # StubTeacher fixtures are not rejected by the contamination gate.
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

    # LoRA default, and deliberately no midtrain checkpoint: this is the
    # documented single-process bring-up path, which starts from the raw base.
    # It must therefore declare development mode -- production refuses to train
    # a stage from the untrained base (see _resolve_stage_input).
    args = _args(["--tasks", "rmsnorm_aiter", "--sft-out", "runs/sft",
                  "--campaign-mode", "development"])
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


# =========================================================================== #
# 11. The four subsystems whose PRODUCTION WRITERS/APIs existed with no caller.
#
# Each block below pins the CALL, not the callee: the callees are unit-tested
# elsewhere and were all green while the campaign silently never invoked them.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 11a. The held-out shape lane is frozen at TRAINING time.
#
# kore.eval.champion.held_out_shapes REFUSES to derive a lane (it hard-raises
# without a manifest), so an unwritten lane does not degrade certification - it
# removes it. The campaign therefore has to write it before anything runs, and
# every rank of a sharded GRPO launch has to agree on exactly one lane.
# --------------------------------------------------------------------------- #
def _local_model(tmp_path):
    """A content-addressable local checkpoint so lineage needs no download."""
    model = tmp_path / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4,
        "num_hidden_layers": 2,
    }))
    (model / "tokenizer_config.json").write_text('{"model_max_length": 128}')
    (model / "model.safetensors").write_bytes(b"weights")
    return model


def _split_ctx(tmp_path, *extra_argv):
    args = _args(["--tasks", "rmsnorm_aiter", "--data-root", str(tmp_path), *extra_argv])
    ctx = {"data_root": tmp_path, "args": args, "dry": False,
           "tasks": [get_task("rmsnorm_aiter")]}
    rc._apply_split(ctx)
    return ctx


def test_campaign_freezes_the_shape_lane_before_the_first_stage(monkeypatch, tmp_path):
    """End-to-end through ``run()``: the lane exists on disk when a stage starts."""
    from kore.tasks.shape_policy import SPLIT_INDEX_FILENAME
    import kore.campaign_lineage as lineage_module

    monkeypatch.setattr(lineage_module, "git_source_identity", lambda root: {
        "commit": "a" * 40, "dirty": False, "dirty_status_digest": "sha256:clean",
        "content_digest": "sha256:source", "scope": ["kore"],
    })
    seen: dict = {}

    def recording_stage(ctx):
        directory = ctx["shape_split_dir"]
        seen["receipt_exists"] = (directory / SPLIT_INDEX_FILENAME).exists()
        seen["manifest_exists"] = (directory / "rmsnorm_aiter.json").exists()
        seen["env"] = os.environ.get("KORE_SHAPE_SPLIT_DIR")
        seen["directory"] = directory

    monkeypatch.setattr(rc, "_stage_grpo", recording_stage)
    monkeypatch.setattr(rc, "_capture_stage_artifact", lambda ctx, stage: {})
    monkeypatch.delenv("KORE_SHAPE_SPLIT_DIR", raising=False)

    args = _args([
        "--tasks", "rmsnorm_aiter", "--stages", "grpo",
        "--campaign-mode", "development",
        "--model", str(_local_model(tmp_path)),
        "--data-root", str(tmp_path / "run"),
    ])
    assert rc.run(args) == 0

    # The lane is OLDER than the first candidate the run could produce.
    assert seen["receipt_exists"] is True, "no stage may start before the lane exists"
    assert seen["manifest_exists"] is True
    # ... and it is published to every stage + every rank that shells out.
    assert seen["env"] == str(seen["directory"])
    assert seen["directory"] == tmp_path / "run" / "shape_splits"


def test_frozen_lane_is_consumable_by_champion_certification(tmp_path, monkeypatch):
    """The written artifact is exactly what the certification reader demands."""
    from kore.eval.champion import held_out_shapes, load_shape_manifests

    monkeypatch.delenv("KORE_SHAPE_SPLIT_DIR", raising=False)
    ctx = _split_ctx(tmp_path)
    directory = rc._freeze_shape_splits(ctx)

    task = get_task("rmsnorm_aiter")
    # require_index=True refuses any directory not produced by the writer at all.
    manifests = load_shape_manifests(
        str(directory), tasks=[task], require_index=True)
    shapes = held_out_shapes(task, frozen_split=manifests[task.task_id])

    assert shapes, "certification would have nothing to re-evaluate on"
    declared = {tuple(sorted(s.dims.items())) for s in task.shapes}
    assert not declared & {tuple(sorted(s.dims.items())) for s in shapes}


def test_second_campaign_run_reuses_the_lane_instead_of_re_deriving_it(
        tmp_path, monkeypatch):
    """A re-run must inherit the lane it started with, not choose a new one."""
    from kore.tasks import shape_policy
    from kore.tasks.shape_policy import SPLIT_INDEX_FILENAME

    monkeypatch.delenv("KORE_SHAPE_SPLIT_DIR", raising=False)
    ctx = _split_ctx(tmp_path)
    directory = rc._freeze_shape_splits(ctx)
    receipt = directory / SPLIT_INDEX_FILENAME
    first = receipt.read_bytes()
    manifest = (directory / "rmsnorm_aiter.json").read_bytes()

    # Deriving a split is what must NOT happen the second time; re-validating and
    # reusing the stored one is.
    monkeypatch.setattr(
        shape_policy, "freeze_shape_split",
        lambda *a, **k: pytest.fail("a second run re-derived the hidden lane"))
    assert rc._freeze_shape_splits(_split_ctx(tmp_path)) == directory

    assert receipt.read_bytes() == first, "the receipt must stay byte-identical"
    assert (directory / "rmsnorm_aiter.json").read_bytes() == manifest


def test_the_sharded_grpo_entry_freezes_the_lane_before_training_starts(
        monkeypatch, tmp_path):
    """``python -m kore.policy.grpo <config.json>`` is what the FSDP launcher runs
    on every rank, so it is the per-rank entry that must publish the lane."""
    import kore.policy.grpo as grpo
    from kore.tasks.shape_policy import SPLIT_INDEX_FILENAME

    lane = tmp_path / "lane"
    monkeypatch.setenv("KORE_SHAPE_SPLIT_DIR", str(lane))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    seen: dict = {}

    def fake_train(config, tasks=None):
        seen["lane_ready"] = (lane / SPLIT_INDEX_FILENAME).exists()
        seen["manifest_ready"] = (lane / "rmsnorm_aiter.json").exists()
        seen["tasks"] = list(tasks or [])
        return "runs/grpo_out"

    monkeypatch.setattr(grpo, "train_grpo", fake_train)
    config = tmp_path / "grpo.json"
    config.write_text(json.dumps({
        "model_id": "fake/model", "use_lora": False,
        "output_dir": str(tmp_path / "out"), "tasks": ["rmsnorm_aiter"],
    }))

    assert grpo._main([str(config)]) == 0

    assert seen["lane_ready"] is True, "training started before the lane existed"
    assert seen["manifest_ready"] is True
    # The frozen lane and the trained task list are provably the same list.
    assert seen["tasks"] == ["rmsnorm_aiter"]


def test_the_sharded_entry_freezes_exactly_the_tasks_it_will_train_on(
        monkeypatch, tmp_path):
    """A config with no explicit task list trains the whole train split, so the
    lane has to cover the whole train split too."""
    import kore.policy.grpo as grpo

    lane = tmp_path / "lane"
    monkeypatch.setenv("KORE_SHAPE_SPLIT_DIR", str(lane))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    frozen: dict = {}
    trained: dict = {}
    real_freeze = grpo.freeze_training_shape_splits

    def recording_freeze(config, task_ids):
        frozen["ids"] = list(task_ids)
        return real_freeze(config, task_ids)

    def recording_train(config, tasks=None):
        trained["ids"] = list(tasks or [])
        return "runs/grpo_out"

    monkeypatch.setattr(grpo, "freeze_training_shape_splits", recording_freeze)
    monkeypatch.setattr(grpo, "train_grpo", recording_train)
    config = tmp_path / "grpo.json"
    config.write_text(json.dumps({
        "model_id": "fake/model", "use_lora": False,
        "output_dir": str(tmp_path / "out"),
    }))

    assert grpo._main([str(config)]) == 0

    assert frozen["ids"] == trained["ids"] == grpo.default_grpo_task_ids()


def _rank(monkeypatch, rank, world, lane):
    """Put this process on ``rank`` of a ``world``-rank launch sharing ``lane``."""
    import kore.policy.grpo as grpo
    from kore.policy.configs import GRPOConfig

    monkeypatch.setenv("KORE_SHAPE_SPLIT_DIR", str(lane))
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setattr(grpo, "_SHAPE_SPLIT_BARRIER_TIMEOUT_S", 0.05)
    return GRPOConfig(model_id="fake/model", output_dir=str(lane.parent / "out"),
                      use_lora=False, total_steps=1)


def _forbid_writing(monkeypatch):
    from kore.tasks import shape_policy

    monkeypatch.setattr(
        shape_policy, "freeze_shape_splits",
        lambda *a, **k: pytest.fail("a follower rank wrote the frozen lane"))


def test_a_follower_rank_never_writes_the_lane(monkeypatch, tmp_path):
    """Concurrent writers on one directory can publish an index that OMITS the
    manifests another rank added, and certification rejects exactly that. A
    follower that finds no receipt must fail loudly instead of rolling out
    against a lane nobody will be able to certify against."""
    import kore.policy.grpo as grpo

    config = _rank(monkeypatch, 1, 4, tmp_path / "lane")
    _forbid_writing(monkeypatch)

    with pytest.raises(RuntimeError, match="frozen shape split receipt"):
        grpo.freeze_training_shape_splits(config, ["rmsnorm_aiter"])


def test_a_follower_rank_proceeds_on_the_lane_rank_zero_published(
        monkeypatch, tmp_path):
    import kore.policy.grpo as grpo
    from kore.tasks.shape_policy import SPLIT_INDEX_FILENAME

    lane = tmp_path / "lane"
    config = _rank(monkeypatch, 0, 4, lane)
    assert grpo.freeze_training_shape_splits(config, ["rmsnorm_aiter"]) == lane
    assert (lane / SPLIT_INDEX_FILENAME).exists()

    config = _rank(monkeypatch, 2, 4, lane)
    _forbid_writing(monkeypatch)

    assert grpo.freeze_training_shape_splits(config, ["rmsnorm_aiter"]) == lane


# --------------------------------------------------------------------------- #
# 11b. The five evaluation budget counters are charged through KoreEnv.evaluate.
#
# evaluate() is the only funnel that can observe a replay hit, so it owns all
# five. Until it charged them, a limit on any of them - including a hard 0 - was
# silently unenforceable. No GPU: the subprocess boundary is stubbed exactly as
# tests/test_env_plumbing.py stubs it.
# --------------------------------------------------------------------------- #
_BUDGET_SOURCE = "def kernel(x):\n    return x + 1\n"
_BUDGET_OUT = "SNR: 99.0\nallclose: True\nmedian_ms: 1.0\n"
_BUDGET_COUNTERS = (
    "correctness_calls", "fresh_timed_calls", "replay_hits",
    "verifier_gpu_seconds", "profiler_gpu_seconds",
)
_RUNTIME_IDENTITY = {
    "identity_version": 1, "validated": True, "stable": True,
    "hardware": {"id": "test-gpu-0", "gpu_target": "gfx950", "selected_gpu": "0"},
    "runtime": {"preflight_revision": "test"},
}


def _budget_config(tmp_path, **overrides):
    from kore.analysis.roofline import make_physical_model

    model = make_physical_model("mi350x")
    cfg = SimpleNamespace(
        runs_dir=tmp_path / "runs", gpu_target="gfx950",
        rocm_path=str(tmp_path / "missing-rocm"),
        shape_augment=False, shape_augment_max=6,
        snr_threshold_for=lambda _dtype: 25.0, atol=1e-2, rtol=1e-2,
        verifier_determinism_check=False, determinism_snr_tol_db=10.0,
        warmup_iters=10, bench_iters=30, min_variance_runs=3, max_variance_runs=5,
        cv_threshold_pct=3.0, baseline_cv_threshold_pct=3.0,
        paired_ratio_cv_threshold_pct=3.0, paired_ci_threshold_pct=3.0,
        paired_confidence_z=1.96, noise_floor_pct=2.0, profile_reward_weight=0.0,
        physics_sku="mi350x", physics_calibration_path=None,
        physics_model_fingerprint=model.fingerprint,
        physics_shaping_evidence_path=None,
        physics_shaping_evidence_fingerprint=None,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def _budget_task(tmp_path):
    from kore.tasks.base import Shape, Task

    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text("task_id: budget_gemm_bf16\ndtype: bf16\n")
    (task_dir / "reference.py").write_text("def reference(x):\n    return x\n")
    (task_dir / "driver.py").write_text("def driver_main():\n    return 0\n")
    return Task(
        task_id="budget_gemm_bf16", operation="gemm", dtype="bf16", backend="triton",
        gpu_target="gfx950", dir=task_dir, seed_kernel_name="seed_triton.py",
        snr_threshold=25.0, comparison_baseline="aiter",
        shapes=[Shape("primary", {"M": 128, "N": 128, "K": 128})],
        raw={"baseline_tier": "vendor"},
    )


def _budget_env(tmp_path, ledger, *, use_replay=False, **cfg_overrides):
    from kore.env.kore_env import KoreEnv
    from kore.tasks._genops import (
        DRIVER_CAPABILITY_PROTOCOL, DRIVER_PROTOCOL_ID, PUBLICATION_GUARANTEES,
    )

    task = _budget_task(tmp_path)
    env = KoreEnv(
        task, config=_budget_config(tmp_path, **cfg_overrides),
        use_replay=use_replay, gpu="0", budget_ledger=ledger,
        runtime_identity=_RUNTIME_IDENTITY if use_replay else None,
    )
    env._driver_caps_cache = {
        "protocol": DRIVER_CAPABILITY_PROTOCOL, "protocol_id": DRIVER_PROTOCOL_ID,
        "performance_eligible": True, **PUBLICATION_GUARANTEES,
    }
    env._exec = lambda cmd, workdir, environ, timeout: (0, _BUDGET_OUT, False)
    quiet = [
        {"pair": i, "order": "AB" if i % 2 == 0 else "BA",
         "candidate_ms": 1.0, "baseline_ms": 2.0, "ratio": 2.0}
        for i in range(5)
    ]
    env._bench_all = lambda driver, shapes, workdir, environ, snr_threshold=None: (
        {shape.name: list(quiet) for shape in shapes}, False)
    return env, task


def _evaluate(env, task, *, do_bench=True):
    return env.evaluate(task, _BUDGET_SOURCE, shapes=list(task.shapes),
                        do_bench=do_bench)


def _counters(ledger):
    return {name: getattr(ledger, name) for name in _BUDGET_COUNTERS}


def test_a_fresh_timed_evaluation_charges_correctness_timing_and_verifier_seconds(
        tmp_path):
    from kore.policy.budget import BudgetLedgerV1

    ledger = BudgetLedgerV1()
    env, task = _budget_env(tmp_path, ledger)
    assert _counters(ledger) == dict.fromkeys(_BUDGET_COUNTERS, 0) | {
        "verifier_gpu_seconds": 0.0, "profiler_gpu_seconds": 0.0}

    obs = _evaluate(env, task)

    assert obs.validation_passed and obs.wall_by_shape
    assert ledger.correctness_calls == 1
    assert ledger.fresh_timed_calls == 1
    assert ledger.replay_hits == 0
    assert ledger.verifier_gpu_seconds > 0.0
    assert ledger.profiler_gpu_seconds == 0.0


def test_a_correctness_only_evaluation_charges_no_timed_call(tmp_path):
    from kore.policy.budget import BudgetLedgerV1

    ledger = BudgetLedgerV1()
    env, task = _budget_env(tmp_path, ledger)

    obs = _evaluate(env, task, do_bench=False)

    assert obs.validation_passed and not obs.wall_by_shape
    assert ledger.correctness_calls == 1
    assert ledger.fresh_timed_calls == 0, "an unbenched candidate spent no timing"
    assert ledger.verifier_gpu_seconds > 0.0


def test_a_replay_hit_charges_a_replay_hit_and_nothing_else(tmp_path):
    from kore.policy.budget import BudgetLedgerV1

    ledger = BudgetLedgerV1()
    env, task = _budget_env(tmp_path, ledger, use_replay=True)
    _evaluate(env, task)
    after_fresh = _counters(ledger)
    env._run = lambda *a, **k: pytest.fail("a cache hit must not re-run anything")

    _evaluate(env, task)

    assert ledger.replay_hits == 1
    # A hit consumed no GPU, so nothing else may move.
    assert _counters(ledger) == {**after_fresh, "replay_hits": 1}


def test_profiler_seconds_are_carved_out_of_the_same_measured_interval(tmp_path):
    """Two second-counters over one interval must not double-count it."""
    import time

    from kore.policy.budget import BudgetLedgerV1

    ledger = BudgetLedgerV1()
    env, task = _budget_env(tmp_path, ledger, profile_reward_weight=0.15)

    def slow_profile(*_a, **_k):
        time.sleep(0.05)
        return 0.75

    env._collect_profile = slow_profile
    started = time.perf_counter()
    obs = _evaluate(env, task)
    elapsed = time.perf_counter() - started

    assert obs.profile_efficiency == 0.75
    assert ledger.profiler_gpu_seconds >= 0.05
    assert ledger.verifier_gpu_seconds > 0.0
    total = ledger.profiler_gpu_seconds + ledger.verifier_gpu_seconds
    assert total <= elapsed, "the profiler pass was charged twice"
    # kore.eval.champion reads this exact attribute to attribute profiler time.
    assert env.last_profiler_seconds == pytest.approx(
        ledger.profiler_gpu_seconds)


def test_profiler_attribution_does_not_leak_into_a_later_evaluation(tmp_path):
    from kore.policy.budget import BudgetLedgerV1

    ledger = BudgetLedgerV1()
    env, task = _budget_env(tmp_path, ledger, profile_reward_weight=0.15)
    env._collect_profile = lambda *a, **k: 0.5
    _evaluate(env, task)
    assert env.last_profiler_seconds > 0.0

    env.cfg.profile_reward_weight = 0.0
    _evaluate(env, task)

    assert env.last_profiler_seconds == 0.0


@pytest.mark.parametrize(
    ("counter", "limit", "cfg"),
    [
        pytest.param("correctness_calls", 0, {}, id="correctness_calls"),
        pytest.param("fresh_timed_calls", 0, {}, id="fresh_timed_calls"),
        pytest.param("verifier_gpu_seconds", 0.0, {}, id="verifier_gpu_seconds"),
        pytest.param("profiler_gpu_seconds", 0.0, {"profile_reward_weight": 0.15},
                     id="profiler_gpu_seconds"),
    ],
)
def test_a_limit_of_zero_refuses_to_launch_the_evaluation(
        tmp_path, counter, limit, cfg):
    """Fail CLOSED: the refusal must land before the subprocess, not after.

    A zero seconds budget is refused for the same reason a zero call budget is:
    a launched evaluation is certain to spend a positive amount of both.
    """
    from kore.policy.budget import BudgetExceededError, BudgetLedgerV1

    ledger = BudgetLedgerV1(limits={counter: limit})
    env, task = _budget_env(tmp_path, ledger, **cfg)
    env._run = lambda *a, **k: pytest.fail("a zero budget still launched the GPU work")

    with pytest.raises(BudgetExceededError, match=counter):
        _evaluate(env, task)

    assert _counters(ledger)[counter] == limit


def test_a_pre_flight_never_invents_a_duration_it_cannot_predict(tmp_path):
    """The seconds claim exists only to catch an exhausted budget; a real budget
    must be spent by the MEASURED duration, not by the pre-flight."""
    from kore.policy.budget import BudgetLedgerV1

    ledger = BudgetLedgerV1(limits={"verifier_gpu_seconds": 60.0})
    env, task = _budget_env(tmp_path, ledger)

    _evaluate(env, task)

    assert 0.0 < ledger.verifier_gpu_seconds < 60.0
    assert env.last_profiler_seconds == 0.0


def test_a_replay_hit_limit_of_zero_is_enforced(tmp_path):
    from kore.policy.budget import BudgetExceededError, BudgetLedgerV1

    ledger = BudgetLedgerV1(limits={"replay_hits": 0})
    env, task = _budget_env(tmp_path, ledger, use_replay=True)
    _evaluate(env, task)

    with pytest.raises(BudgetExceededError, match="replay_hits"):
        _evaluate(env, task)


def test_an_unbudgeted_environment_stays_a_no_op(tmp_path):
    env, task = _budget_env(tmp_path, None)

    assert _evaluate(env, task).validation_passed


def test_grpo_hands_its_ledger_to_the_rollout_environment(monkeypatch):
    """The ledger lives on the GRPOConfig; the env is handed kore.config.CONFIG,
    so it can only arrive through an explicit argument."""
    import kore.env.kore_env as env_module
    import kore.policy.grpo as grpo
    from kore.policy.budget import BudgetLedgerV1

    ledger = BudgetLedgerV1()
    seen: dict = {}
    monkeypatch.setattr(
        env_module, "KoreEnv",
        lambda task, **kw: seen.update(kw) or SimpleNamespace(task=task))
    config = SimpleNamespace(
        _grpo_feature_runtime=SimpleNamespace(ledger=ledger),
        profile_reward_weight=0.0, physics_shaping_weight=0.0)

    grpo._make_rollout_env(get_task("rmsnorm_aiter"), config, gpu="4")

    assert seen["budget_ledger"] is ledger
    assert seen["gpu"] == "4"


def test_an_unbudgeted_rollout_builds_the_environment_exactly_as_before(monkeypatch):
    """No ledger means no argument, so an unbudgeted rollout is unchanged."""
    import kore.env.kore_env as env_module
    import kore.policy.grpo as grpo

    seen: list = []
    monkeypatch.setattr(
        env_module, "KoreEnv",
        lambda task, **kw: seen.append(kw) or SimpleNamespace(task=task))

    grpo._make_rollout_env(get_task("rmsnorm_aiter"), SimpleNamespace())

    assert seen == [{}]


# --------------------------------------------------------------------------- #
# 11c. Hardware eligibility is an OPT-IN narrowing of datagen selection.
#
# The registry's train split is 1289 tasks. The default eligibility policy drops
# the structurally-broken and SNR-shortfall seeds; both bands are empty on the
# current gfx950 sweep, so it drops nothing today, and the STRICT policy is what
# still narrows (it admits only a recorded PASS, so every unmeasured task goes).
# Applying either implicitly would be its own bug - an operator who did not ask
# for a smaller scope would read the shrunken totals as progress - so the default
# must not move.
# --------------------------------------------------------------------------- #
def _partition(tmp_path, name, *extra_argv):
    import scripts.spur_partition as sp

    out = tmp_path / name
    argv = ["spur_partition.py", "--data-root", str(tmp_path / "data"),
            "--out-dir", str(out), "--shards", "2", *extra_argv]
    import sys

    original = sys.argv
    sys.argv = argv
    try:
        assert sp.main() == 0
    finally:
        sys.argv = original
    return json.loads((out / "manifest.json").read_text())


def test_partitioner_selection_is_unchanged_unless_a_policy_is_named(tmp_path):
    from kore.tasks.registry import train_tasks

    default = _partition(tmp_path, "default")

    assert default["eligibility_policy"] is None
    assert default["n_train_tasks"] == len(train_tasks())
    assert default["n_ineligible_excluded"] == 0
    assert default["ineligible_excluded"] == {}


def test_naming_a_policy_narrows_selection_and_records_what_it_dropped(tmp_path):
    from kore.tasks.registry import eligible_train_tasks, train_tasks

    # The default policy is honoured, and on a clean sweep that means it removes
    # nobody -- naming it must still be RECORDED, or a later run could not tell
    # which scope a manifest was built under.
    named = _partition(tmp_path, "named",
                       "--eligibility-policy", "exclude_broken_and_shortfall")
    eligible = {task.task_id for task in eligible_train_tasks()}
    assert named["eligibility_policy"] == "exclude_broken_and_shortfall"
    assert named["n_train_tasks"] == len(eligible)
    assert {item["task_id"] for item in named["items"]} <= eligible

    # The narrowing path itself is exercised by the strict policy, which admits
    # only a recorded PASS and so drops every unmeasured task.
    strict = _partition(tmp_path, "strict",
                        "--eligibility-policy", "strict_hardware_verified")
    strict_eligible = {task.task_id
                       for task in eligible_train_tasks("strict_hardware_verified")}
    assert strict["eligibility_policy"] == "strict_hardware_verified"
    assert strict["n_train_tasks"] == len(strict_eligible)
    assert strict["n_train_tasks"] < len(train_tasks())
    dropped = strict["ineligible_excluded"]
    assert len(dropped) == strict["n_ineligible_excluded"] > 0
    assert not set(dropped) & strict_eligible
    # Every drop names the hardware evidence behind it, never a bare count.
    assert all(reason for reason in dropped.values())
    assert {item["task_id"] for item in strict["items"]} <= strict_eligible


def test_admit_all_is_a_nameable_no_op_policy(tmp_path):
    from kore.tasks.registry import train_tasks

    admit_all = _partition(tmp_path, "admit_all",
                           "--eligibility-policy", "admit_all")

    assert admit_all["n_train_tasks"] == len(train_tasks())
    assert admit_all["n_ineligible_excluded"] == 0


def test_partitioner_prefix_default_still_covers_every_train_task(tmp_path):
    """The invariant tests/test_spur_supervisor.py asserts stays TRUE: the
    default scope is the whole train split, so the eligibility option is the
    only thing that can narrow it."""
    from kore.tasks.registry import train_tasks

    default = _partition(tmp_path, "coverage")
    task_ids = {task.task_id for task in train_tasks()}

    assert {item["task_id"] for item in default["items"]} == task_ids


# --------------------------------------------------------------------------- #
# 11d. The KernelBench-AMD claim gate is conjuncted into the track verdict.
#
# A finite, non-empty report is necessary but not sufficient: without a metric
# bar this track passed at fast_1 == 0.0 on a real KernelBench checkout.
# --------------------------------------------------------------------------- #
def _kb_report(fast_1, *, n=25, correct_rate=0.8):
    return {"n": n, "correct_rate": correct_rate, "fast_1": fast_1,
            "fast_p": {1.0: fast_1, 1.5: fast_1 / 2}}


def _kb_ctx(tmp_path, **overrides):
    args = _args(["--tasks", "rmsnorm_aiter", "--eval-budget", "1"])
    for name, value in overrides.items():
        setattr(args, name, value)
    return {"data_root": tmp_path, "args": args, "dry": False}


def _stub_kernelbench(monkeypatch, report, *, real_specs):
    import kore.eval.kernelbench_amd as kb

    monkeypatch.setattr(kb, "bundled_specs", lambda: [{"spec": "smoke"}])
    monkeypatch.setattr(kb, "load_real_kernelbench", lambda root: [{"spec": "real"}])
    monkeypatch.setattr(kb, "format_kernelbench_report", lambda report: "report")
    monkeypatch.setattr(
        kb, "run_kernelbench_amd",
        lambda policy, specs, **kw: {"report": report})
    return real_specs


@pytest.mark.parametrize(
    ("fast_1", "expected"),
    [pytest.param(0.0, False, id="fast_1-zero-is-not-a-claim"),
     pytest.param(0.19, False, id="below-bar"),
     pytest.param(0.35, True, id="clears-bar")],
)
def test_kernelbench_track_passes_only_when_the_claim_gate_passes(
        monkeypatch, tmp_path, fast_1, expected):
    from kore.eval.kernelbench_amd import kernelbench_claim_gate

    _stub_kernelbench(monkeypatch, _kb_report(fast_1), real_specs=True)
    ctx = _kb_ctx(tmp_path, kernelbench_root="/kb")

    result = rc._eval_kernelbench_amd(ctx, object(), object())

    assert result["source"] == "full" and result["source_ok"] is True
    assert result["passed"] is expected
    assert result["gate"] == kernelbench_claim_gate(
        _kb_report(fast_1), source="full")
    if not expected:
        assert result["gate"]["reasons"], "a failure must say what missed the bar"
    # The bar the verdict used travels with the persisted verdict.
    assert result["gate"]["thresholds"]["min_fast_1"] > 0.0


def test_kernelbench_bundled_smoke_specs_can_never_pass_the_track(
        monkeypatch, tmp_path):
    _stub_kernelbench(monkeypatch, _kb_report(1.0), real_specs=False)
    ctx = _kb_ctx(tmp_path)

    result = rc._eval_kernelbench_amd(ctx, object(), object())

    assert result["source"] == "bundled-smoke"
    assert result["passed"] is False
    assert any("not claimable" in reason for reason in result["gate"]["reasons"])


# --------------------------------------------------------------------------- #
# 12. Stage chaining must never silently substitute the untrained base
# --------------------------------------------------------------------------- #
def _ctx(campaign_mode="production", allow_fallback=False, **ckpts):
    args = SimpleNamespace(campaign_mode=campaign_mode,
                           allow_base_fallback=allow_fallback)
    ctx = {"base": "Qwen/Qwen3-14B", "args": args, "current_stage": "-",
           "midtrain_ckpt": None, "sft_ckpt": None, "dpo_ckpt": None,
           "grpo_ckpt": None, "final": None}
    ctx.update(ckpts)
    return ctx


def test_production_refuses_to_train_a_stage_from_the_untrained_base():
    """`--stages sft` used to silently train from raw Qwen3-14B.

    ``ctx["midtrain_ckpt"]`` is only set by the midtrain stage running in the
    same invocation or by resuming a schema-current manifest, so a mid-train
    submitted directly with sbatch was invisible and every later stage fell
    back to ``ctx["base"]`` with no log line at all.
    """
    campaign = rc
    for stage in ("sft", "dpo", "grpo"):
        with pytest.raises(SystemExit) as excinfo:
            campaign._resolve_stage_input(_ctx(), stage)
        message = str(excinfo.value)
        assert "untrained base" in message
        assert "--allow-base-fallback" in message
        # It must name a flag that actually exists.
        named = [tok for tok in message.split() if tok.startswith("--")]
        assert named, message
        for flag in named:
            flag = flag.rstrip(",.")
            if flag == "--allow-base-fallback":
                continue
            assert flag in {"--midtrain-ckpt", "--sft-ckpt", "--dpo-ckpt"}, flag


def test_the_fallback_is_reachable_but_never_silent():
    campaign = rc
    logged: list[str] = []
    original = campaign._log
    campaign._log = lambda stage, msg, **kw: logged.append(str(msg))
    try:
        resolved = campaign._resolve_stage_input(
            _ctx(campaign_mode="development"), "sft")
    finally:
        campaign._log = original
    assert resolved == "Qwen/Qwen3-14B"
    assert any("UNTRAINED base" in line for line in logged), logged


def test_an_injected_checkpoint_is_used_and_reported():
    campaign = rc
    logged: list[str] = []
    original = campaign._log
    campaign._log = lambda stage, msg, **kw: logged.append(str(msg))
    try:
        resolved = campaign._resolve_stage_input(
            _ctx(midtrain_ckpt="runs/midtrain_14b_frontier"), "sft")
    finally:
        campaign._log = original
    assert resolved == "runs/midtrain_14b_frontier"
    assert any("midtrain_ckpt" in line for line in logged), logged


def test_grpo_falls_back_through_dpo_to_sft_before_the_base():
    """Skipping DPO is legitimate; skipping straight to the base is not."""
    campaign = rc
    logged: list[str] = []
    original = campaign._log
    campaign._log = lambda stage, msg, **kw: logged.append(str(msg))
    try:
        resolved = campaign._resolve_stage_input(_ctx(sft_ckpt="runs/sft"), "grpo")
    finally:
        campaign._log = original
    assert resolved == "runs/sft"
    assert any("sft_ckpt" in line for line in logged), logged
