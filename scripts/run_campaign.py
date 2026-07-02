"""KORE end-to-end campaign orchestrator (the full agentic recipe).

Stages (each gated on the previous via retention + kernel metrics):
    datagen  : teacher -> repair + ranked-groups + wins per task
    agentic  : teacher-driven build/test/bench/pmc tool-use trajectories
    build    : leakage-split the raw records, hold out an op family (+gfx950) as
               eval-only, then assemble a multi-capability SFT mix (kernel + QA +
               agentic + ~45% general) and a DPO set with >=8% hard negatives from
               the TRAIN split only
    midtrain : Stage-0 full-FT continued pretrain on the ROCm/Triton corpus
    sft      : Stage-1 multi-capability SFT (retains chat/code/orchestration)
    dpo      : Stage-2 preference tuning
    grpo     : Stage-3 multi-turn AGENTIC GRPO (Kevin credit + StarPO-S + KL anchor)
    soup     : Stage-4 base-ward model soup (retention-gated alpha SWEEP)
    eval     : matched-budget fast_p bake-off (seed vs the TRAINED model) + retention

Every training stage is retention-gated (hard-stop on general regression). The run
is resumable: a JSON manifest at ``<data_root>/campaign_manifest.json`` records the
real checkpoints + which stages finished, and per-stage JSONL events are appended to
``<data_root>/campaign_events.jsonl`` for observability.

--dry-run validates the WHOLE wiring with no GPU/teacher (it import-checks every
symbol the campaign will call). --stages runs a subset (and reuses prior checkpoints
from the manifest, so a crash mid-run is recoverable).

    PYTHONPATH=. python scripts/run_campaign.py --model Qwen/Qwen3-14B \
        --tasks rmsnorm_aiter,gemm_bf16,flash_attn_decode_bf16 \
        --teacher claude --stages datagen,agentic,build,sft,dpo,grpo,soup,eval
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import time
from pathlib import Path

ALL_STAGES = ["datagen", "agentic", "build", "midtrain", "sft", "dpo", "grpo", "soup", "eval"]
DEFAULT_STAGES = ["datagen", "agentic", "build", "sft", "dpo", "grpo", "soup", "eval"]

# Kernel metric key used to drive the soup alpha sweep (fast_p at p=1.0).
_SOUP_KERNEL_KEY = "kernel_fast1"


def _log(stage: str, msg: str) -> None:
    print(f"[campaign:{stage}] {msg}", flush=True)


def _write_rows(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# --------------------------------------------------------------------------- #
# Fix 6: dry-run import-check — fail fast on a missing symbol / signature drift
# --------------------------------------------------------------------------- #
# (module, attribute, required, [param names that MUST exist on the callable]).
# ``required=False`` symbols are provided by a parallel track (the serving
# backend); their absence is a LOUD warning, not a hard failure, so the offline
# dry-run stays green until that track lands.
_IMPORT_CHECKS = [
    ("kore.tasks.registry", "get_task", True, []),
    ("kore.tasks.registry", "all_tasks", True, []),
    ("kore.env.kore_env", "KoreEnv", True, []),
    ("kore.data.assemble", "build_multicap_dataset", True, ["kernel_records"]),
    ("kore.data.assemble", "build_dpo_with_hard_negatives", True, ["group_records"]),
    ("kore.data.assemble", "summarize_multicap", True, []),
    ("kore.data.build_datasets", "leakage_split", True, ["records", "by"]),
    ("kore.data.build_datasets", "dedup_by_source_hash", True, []),
    ("kore.data.schemas", "read_jsonl", True, []),
    ("kore.policy.configs", "MultiCapSFTConfig", True, []),
    ("kore.policy.configs", "DPOConfig", True, []),
    ("kore.policy.configs", "GRPOConfig", True, []),
    ("kore.policy.configs", "SoupConfig", True, []),
    ("kore.policy.sft", "train_sft", True, []),
    ("kore.policy.dpo", "train", True, []),
    ("kore.policy.grpo", "train_grpo", True, ["tasks", "backend"]),
    ("kore.policy.soup", "build_soup", True, []),
    ("kore.policy.soup", "soup_sweep", True, ["kernel_key", "general_keys", "base_scores", "epsilon"]),
    ("kore.policy.format", "parse_response", True, []),
    ("kore.eval.gates", "retention_gate", True, []),
    ("kore.eval.gates", "format_gate_report", True, []),
    ("kore.eval.retention", "run_retention_suite", True, []),
    ("kore.eval.bakeoff", "matched_budget_bakeoff", True, ["env_factory", "budget", "dry_run"]),
    ("kore.eval.bakeoff", "evaluate_policy", True, ["env_factory", "budget"]),
    ("kore.eval.report", "format_bakeoff_table", True, []),
    ("kore.eval.report", "save_report", True, []),
    ("kore.eval.fastp", "fastp", True, []),
    ("kore.eval.policies", "seed_policy", True, []),
    ("kore.eval.policies", "model_policy", True, ["checkpoint"]),
    # Serving backend (parallel track): needed by the retention gate, model_policy
    # and the soup sweep at RUN time. Absence -> loud warning, not a dry-run failure.
    ("kore.policy.serve", "load_generate", False, []),
]


def _dry_import_check() -> None:
    """Import-check every symbol the campaign will call so drift fails fast."""
    problems: list[str] = []
    warnings: list[str] = []
    for mod, attr, required, params in _IMPORT_CHECKS:
        sink = problems if required else warnings
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            sink.append(f"{mod}: import failed: {e!r}")
            continue
        obj = getattr(m, attr, None)
        if obj is None:
            sink.append(f"{mod}.{attr}: MISSING")
            continue
        if params:
            try:
                sig = inspect.signature(obj)
                missing = [p for p in params if p not in sig.parameters]
                if missing:
                    sink.append(f"{mod}.{attr}: signature drift — missing params {missing}")
            except (TypeError, ValueError):
                pass  # some builtins/objects have no introspectable signature
    for w in warnings:
        _log("preflight", f"WARNING: {w} (serving backend not yet provisioned)")
    _log("preflight", f"import-check: {len(_IMPORT_CHECKS)} symbols, "
                      f"{len(problems)} problems, {len(warnings)} warnings")
    if problems:
        raise SystemExit("preflight import-check FAILED:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------- #
# Fix 3: run manifest (resume) + Fix 7: structured JSONL events
# --------------------------------------------------------------------------- #
def _manifest_path(ctx) -> Path:
    return ctx["data_root"] / "campaign_manifest.json"


def _load_manifest_into_ctx(ctx) -> None:
    """Populate ctx from a prior manifest so a resumed run reuses real ckpts."""
    p = _manifest_path(ctx)
    if not p.exists():
        return
    try:
        m = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        _log("resume", f"WARNING: could not read manifest ({e}); starting fresh")
        return
    for k in ("sft_ckpt", "dpo_ckpt", "grpo_ckpt", "final"):
        if m.get(k):
            ctx[k] = m[k]
    ctx["done_stages"] = set(m.get("done_stages") or [])
    if m.get("eval_tasks"):
        ctx["eval_task_ids"] = list(m["eval_tasks"])
    _log("resume", f"manifest loaded: done={sorted(ctx['done_stages'])} "
                   f"sft={ctx['sft_ckpt']} dpo={ctx['dpo_ckpt']} "
                   f"grpo={ctx['grpo_ckpt']} final={ctx['final']}")


def _save_manifest(ctx) -> None:
    if ctx["dry"]:
        return
    p = _manifest_path(ctx)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": ctx["base"],
        "sft_ckpt": ctx.get("sft_ckpt"),
        "dpo_ckpt": ctx.get("dpo_ckpt"),
        "grpo_ckpt": ctx.get("grpo_ckpt"),
        "final": ctx.get("final"),
        "done_stages": sorted(ctx["done_stages"]),
        "eval_tasks": ctx.get("eval_task_ids"),
        "updated": time.time(),
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)  # atomic: a crash mid-write never corrupts the manifest


def _emit_event(ctx, stage: str, status: str, elapsed: float, artifact=None) -> None:
    """Append a structured per-stage event to campaign_events.jsonl."""
    if ctx["dry"]:
        return
    ev = {"ts": time.time(), "stage": stage, "status": status,
          "elapsed_s": round(elapsed, 4), "artifact": artifact}
    p = ctx["data_root"] / "campaign_events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(ev) + "\n")


def _path_exists(p) -> bool:
    return bool(p) and Path(p).exists()


def _artifact_of(ctx, stage: str):
    dr = ctx["data_root"]
    return {
        "datagen": str(dr / "repair"),
        "agentic": str(dr / "agentic"),
        "build": str(dr / "sft" / "multicap.jsonl"),
        "midtrain": str(dr / "midtrain"),
        "sft": ctx.get("sft_ckpt"),
        "dpo": ctx.get("dpo_ckpt"),
        "grpo": ctx.get("grpo_ckpt"),
        "soup": ctx.get("final"),
        "eval": str(dr / "eval" / "bakeoff.json"),
    }.get(stage)


def _artifact_ok(ctx, stage: str) -> bool:
    """True iff a completed stage's on-disk artifact is present (resume skip)."""
    dr = ctx["data_root"]
    checks = {
        "datagen": lambda: any((dr / k).exists() for k in ("repair", "groups", "wins")),
        "agentic": lambda: (dr / "agentic").exists(),
        "build": lambda: (dr / "sft" / "multicap.jsonl").exists()
                          and (dr / "dpo" / "pairs.jsonl").exists(),
        "midtrain": lambda: True,  # optional stage; nothing to verify
        "sft": lambda: _path_exists(ctx.get("sft_ckpt")),
        "dpo": lambda: _path_exists(ctx.get("dpo_ckpt")),
        "grpo": lambda: _path_exists(ctx.get("grpo_ckpt")),
        "soup": lambda: _path_exists(ctx.get("final")),
        "eval": lambda: (dr / "eval" / "bakeoff.json").exists(),
    }
    fn = checks.get(stage)
    return bool(fn and fn())


# --------------------------------------------------------------------------- #
def run(args) -> int:
    from kore.tasks.registry import all_tasks, get_task

    tasks = [get_task(t) for t in args.tasks.split(",")] if args.tasks else all_tasks()
    stages = args.stages.split(",") if args.stages else DEFAULT_STAGES
    data_root = Path(args.data_root)
    dry = args.dry_run

    ctx = {
        "data_root": data_root, "tasks": tasks, "dry": dry, "args": args,
        "base": args.model, "sft_ckpt": None, "dpo_ckpt": None,
        "grpo_ckpt": None, "final": None, "metrics": {},
        "done_stages": set(), "eval_task_ids": None,
    }

    if dry:
        _dry_import_check()
    else:
        _load_manifest_into_ctx(ctx)
        if args.force:
            for st in stages:
                ctx["done_stages"].discard(st)
            _log("plan", f"--force: will re-run {stages} regardless of manifest")

    _log("plan", f"model={args.model} tasks={[t.task_id for t in tasks]} "
                 f"stages={stages} dry_run={dry}")

    dispatch = {
        "datagen": _stage_datagen, "agentic": _stage_agentic, "build": _stage_build,
        "midtrain": _stage_midtrain, "sft": _stage_sft, "dpo": _stage_dpo,
        "grpo": _stage_grpo, "soup": _stage_soup, "eval": _stage_eval,
    }
    for st in stages:
        if st not in dispatch:
            _log("plan", f"unknown stage '{st}', skipping")
            continue
        if (not dry) and st in ctx["done_stages"] and _artifact_ok(ctx, st):
            _log(st, "already complete (artifact present) — skipping [resume]")
            _emit_event(ctx, st, "skipped", 0.0, _artifact_of(ctx, st))
            continue

        start = time.perf_counter()
        _emit_event(ctx, st, "start", 0.0, None)
        try:
            dispatch[st](ctx)
        except BaseException as e:  # noqa: BLE001 - record the failure then re-raise
            _emit_event(ctx, st, "error", time.perf_counter() - start, repr(e))
            raise
        elapsed = time.perf_counter() - start
        ctx["done_stages"].add(st)
        _save_manifest(ctx)
        _emit_event(ctx, st, "done", elapsed, _artifact_of(ctx, st))
        if not dry:
            _log(st, f"completed in {elapsed:.2f}s")

    _log("done", "campaign complete")
    return 0


# --------------------------------------------------------------------------- #
def _teacher(args):
    from kore.data.teacher import load_env_local, make_teacher
    load_env_local()
    kw = {"model": args.model_teacher} if args.model_teacher else {}
    return make_teacher(args.teacher, **kw)


def _eval_tasks(ctx) -> list:
    """The held-out (eval-only) tasks from the leakage split, else all tasks."""
    ids = ctx.get("eval_task_ids")
    if not ids:
        return ctx["tasks"]
    from kore.tasks.registry import get_task
    out = []
    for t in ids:
        try:
            out.append(get_task(t))
        except Exception:  # noqa: BLE001 - a held-out id with no registered task
            continue
    return out or ctx["tasks"]


def _stage_datagen(ctx):
    if ctx["dry"]:
        _log("datagen", "would generate repair/groups/wins per task (teacher + GPU env)")
        return
    from kore.data.gen_groups import generate_groups
    from kore.data.gen_repair import generate_repairs
    from kore.data.gen_wins import generate_wins
    from kore.data.schemas import write_jsonl
    from kore.env.kore_env import KoreEnv

    t = _teacher(ctx["args"])
    for task in ctx["tasks"]:
        env = KoreEnv(task)
        for kind, recs in (("repair", generate_repairs(task, t, env, n=ctx["args"].n_repair)),
                           ("groups", generate_groups(task, t, env, n_parents=ctx["args"].n_parents, k=ctx["args"].k)),
                           ("wins", generate_wins(task, t, env, gens=ctx["args"].wins_gens))):
            out = ctx["data_root"] / kind / f"{task.task_id}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(out, recs)
            _log("datagen", f"{task.task_id}:{kind} -> {len(recs)} records")


def _stage_agentic(ctx):
    if ctx["dry"]:
        _log("agentic", "would generate build/test/bench/pmc tool-use trajectories per task")
        return
    from kore.data.gen_agentic import generate_agentic_trajectories
    from kore.data.schemas import write_jsonl
    from kore.env.kore_env import KoreEnv

    t = _teacher(ctx["args"])
    for task in ctx["tasks"]:
        env = KoreEnv(task)
        recs = generate_agentic_trajectories(task, t, env, n=ctx["args"].n_agentic,
                                             max_turns=ctx["args"].max_tool_turns, keep_only_useful=True)
        out = ctx["data_root"] / "agentic" / f"{task.task_id}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(out, [r.to_dict() for r in recs])
        _log("agentic", f"{task.task_id} -> {len(recs)} trajectories")


# --- leakage-split helpers (Fix 5) ---
def _rec_dict(rec) -> dict:
    return rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)


def _rec_type(rec) -> str:
    return _rec_dict(rec).get("type", "")


def _rec_op(rec) -> str:
    d = _rec_dict(rec)
    op = d.get("operation")
    if op:
        return op
    tid = d.get("task_id", "") or ""
    return tid.split("_")[0] if tid else ""


def _rec_arch(rec):
    return _rec_dict(rec).get("arch")


def _force_holdout(train: list, test: list) -> tuple[list, list]:
    """Force at least one operator family (and any arch=='gfx950') eval-only.

    Moves every arch=='gfx950' record and one whole operator family out of TRAIN
    into TEST, guarding against emptying TRAIN when only one family exists.
    """
    # 1. arch gfx950 -> eval-only (if present at all).
    gfx = [r for r in train if _rec_arch(r) == "gfx950"]
    if gfx:
        train = [r for r in train if _rec_arch(r) != "gfx950"]
        test = test + gfx

    # 2. hold out one whole operator family (only if >=2 families remain in TRAIN).
    train_ops = sorted({_rec_op(r) for r in train if _rec_op(r)})
    if len(train_ops) >= 2:
        chosen = train_ops[0]
        moved = [r for r in train if _rec_op(r) == chosen]
        train = [r for r in train if _rec_op(r) != chosen]
        test = test + moved
    return train, test


def _stage_build(ctx):
    from kore.policy.configs import MultiCapSFTConfig

    if ctx["dry"]:
        _log("build", "would leakage-split records (by operation+arch), hold out an op "
                      "family (+gfx950), then assemble train-only SFT mix + DPO(+>=8% hard negs)")
        return
    from kore.data.assemble import (build_dpo_with_hard_negatives, build_multicap_dataset,
                                    summarize_multicap)
    from kore.data.build_datasets import dedup_by_source_hash, leakage_split
    from kore.data.schemas import read_jsonl
    from kore.data.teacher import make_teacher

    # 1. gather + dedup all raw generated records that carry leakage provenance.
    raw: list = []
    for sub in ("repair", "wins", "groups"):
        d = ctx["data_root"] / sub
        if d.exists():
            for p in sorted(d.glob("*.jsonl")):
                raw += read_jsonl(p, typed=True)
    raw = dedup_by_source_hash(raw)
    _log("build", f"gathered {len(raw)} deduped raw records")

    # 2. leakage-aware split by (operation, arch); force an op family + gfx950 eval-only.
    train, val, test = leakage_split(raw, by=("operation", "arch"))
    train, test = _force_holdout(train, test)
    ctx["eval_task_ids"] = sorted({_rec_dict(r).get("task_id") for r in test
                                   if _rec_dict(r).get("task_id")})
    _log("build", f"leakage split: train={len(train)} val={len(val)} test={len(test)}; "
                  f"eval-only tasks={ctx['eval_task_ids']}")

    # 3. build SFT/DPO from the TRAIN partition only.
    try:
        teacher = _teacher(ctx["args"])
    except Exception as e:  # noqa: BLE001 - QA gen is optional if the teacher is down
        _log("build", f"teacher unavailable for QA ({e}); using stub")
        teacher = make_teacher("stub")

    kernel_records = [r for r in train if _rec_type(r) in ("repair", "win")]
    group_records = [r for r in train if _rec_type(r) == "ranked_group"]

    cfg = MultiCapSFTConfig()
    rows = build_multicap_dataset(ctx["data_root"], ctx["tasks"], teacher, cfg,
                                  total=ctx["args"].sft_total, use_hf=ctx["args"].use_hf,
                                  kernel_records=kernel_records)
    _write_rows(ctx["data_root"] / "sft" / "multicap.jsonl", rows)
    _log("build", f"multicap SFT (train-only): {len(rows)} rows; "
                  f"mix={summarize_multicap(rows)['fractions']}")

    dpo = build_dpo_with_hard_negatives(ctx["data_root"], ctx["tasks"],
                                        group_records=group_records)
    _write_rows(ctx["data_root"] / "dpo" / "pairs.jsonl", dpo["rows"])
    _log("build", f"DPO (train-only): {dpo['n_total']} pairs ({dpo['n_hard']} hard, "
                  f">=8% target met={dpo['meets_target']})")


def _stage_midtrain(ctx):
    if ctx["dry"]:
        _log("midtrain", "would full-FT continued-pretrain on ROCm/Triton corpus (~15% general replay)")
        return
    corpus = ctx["data_root"] / "midtrain" / "corpus.jsonl"
    if not corpus.exists():
        _log("midtrain", f"no corpus at {corpus}; skipping Stage-0 (optional)")
        return
    _log("midtrain", "Stage-0 continued pretraining (full-FT) — see docs/rl_server.md for the multi-GPU launch")


def _stage_sft(ctx):
    if ctx["dry"]:
        _log("sft", "would multi-capability SFT (full-FT, ~45% general replay retention)")
        return
    from kore.policy.configs import MultiCapSFTConfig
    from kore.policy.sft import train_sft

    # Fix 8: --lora keeps the 14B validation run single-GPU-feasible (full-FT of a
    # 14B needs an FSDP/DeepSpeed multi-GPU launch — see docs/rl_server.md).
    cfg = MultiCapSFTConfig(model_id=ctx["base"], output_dir=ctx["args"].sft_out,
                            use_lora=ctx["args"].lora)
    ctx["sft_ckpt"] = train_sft(cfg, ctx["data_root"] / "sft" / "multicap.jsonl")
    _log("sft", f"-> {ctx['sft_ckpt']}")
    _retention_gate(ctx, stage="sft", candidate=ctx["sft_ckpt"], base=ctx["base"])


def _stage_dpo(ctx):
    if ctx["dry"]:
        _log("dpo", "would DPO on ranked-groups + hard-negative pairs")
        return
    from kore.policy.configs import DPOConfig
    from kore.policy.dpo import train

    sft = ctx.get("sft_ckpt") or ctx["base"]
    cfg = DPOConfig(model_id=sft, dataset_path=str(ctx["data_root"] / "dpo" / "pairs.jsonl"),
                    output_dir=ctx["args"].dpo_out, use_lora=ctx["args"].lora)
    result = train(cfg)
    ctx["dpo_ckpt"] = (result.get("output_dir") if isinstance(result, dict) else None) or ctx["args"].dpo_out
    _log("dpo", f"-> {ctx['dpo_ckpt']}")
    _retention_gate(ctx, stage="dpo", candidate=ctx["dpo_ckpt"], base=sft)


def _stage_grpo(ctx):
    if ctx["dry"]:
        _log("grpo", "would run multi-turn AGENTIC GRPO (Kevin credit + StarPO-S + KL-anchor to SFT ckpt)")
        return
    from kore.policy.configs import GRPOConfig
    from kore.policy.grpo import train_grpo

    sft = ctx.get("sft_ckpt") or ctx["base"]
    init = ctx.get("dpo_ckpt") or sft

    # Fix 4: GRPO must train ONLY on the TRAIN-split tasks. The eval-only ids
    # (the forced-holdout op family + any gfx950, from ``_stage_build``) are the
    # held-out generalization set; training on them would invalidate the eval.
    eval_ids = set(ctx.get("eval_task_ids") or [])
    train_task_ids = [t.task_id for t in ctx["tasks"] if t.task_id not in eval_ids]
    if not train_task_ids:
        _log("grpo", f"WARNING: every task is held out for eval ({sorted(eval_ids)}); "
                     "falling back to training on all tasks (no leakage split available)")
        train_task_ids = [t.task_id for t in ctx["tasks"]]
    else:
        _log("grpo", f"training on TRAIN-split tasks={train_task_ids} "
                     f"(held-out eval-only={sorted(eval_ids)})")

    # Fix 8: --lora fits the 14B validation run without an FSDP/DeepSpeed launch.
    # For LoRA bring-up use feasibly small rollout shapes; with the O(1-sample)
    # micro-batched backward these bound activation memory regardless, but smaller
    # groups also cut rollout wall-clock for the validation run.
    use_lora = ctx["args"].lora
    grpo_kw = dict(model_id=init, output_dir=ctx["args"].grpo_out,
                   agentic=True, starpo_s=True, ref_checkpoint=sft, use_lora=use_lora)
    if use_lora:
        grpo_kw.update(num_trajectories=8, tasks_per_step=2, num_turns=3)
    cfg = GRPOConfig(**grpo_kw)
    ctx["grpo_ckpt"] = train_grpo(cfg, tasks=train_task_ids, backend=ctx["args"].grpo_backend)
    _log("grpo", f"-> {ctx['grpo_ckpt']}")
    _retention_gate(ctx, stage="grpo", candidate=ctx["grpo_ckpt"], base=sft)


def _stage_soup(ctx):
    if ctx["dry"]:
        _log("soup", "would SWEEP alpha via soup_sweep (score kernel fast_p + retention per "
                     "alpha; pick best kernel s.t. no general regression) then build_soup")
        return
    import tempfile

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kore.env.kore_env import KoreEnv
    from kore.eval.bakeoff import evaluate_policy
    from kore.eval.policies import model_policy
    from kore.eval.retention import run_retention_suite
    from kore.policy.configs import SoupConfig
    from kore.policy.serve import load_generate
    from kore.policy.soup import build_soup, soup_sweep

    base = ctx["base"]
    kore_ckpt = ctx.get("grpo_ckpt") or ctx.get("dpo_ckpt") or ctx.get("sft_ckpt") or base
    cfg = SoupConfig(base_model_id=base, kore_checkpoint=kore_ckpt, output_dir=ctx["args"].soup_out)
    tasks = _eval_tasks(ctx)
    budget = ctx["args"].eval_budget

    # Base-model general scores define the no-regression floor for the sweep.
    base_ret = run_retention_suite(load_generate(base))
    base_scores = dict(base_ret["scores"])
    general_keys = list(base_scores.keys())

    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    kore_model = AutoModelForCausalLM.from_pretrained(kore_ckpt, torch_dtype=torch.bfloat16)
    # Fix 3: snapshot IMMUTABLE copies of the endpoint weights. ``state_dict()``
    # returns tensors that ALIAS the live params; the eval_fn below materializes
    # each interpolation via ``scratch.load_state_dict(...)`` (an in-place write),
    # which would otherwise mutate ``kore_sd``/``base_sd`` and make every alpha
    # after the first interpolate from ALREADY-interpolated weights -> wrong
    # best_alpha. Cloning detaches the sweep endpoints from any live model.
    base_sd = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    kore_sd = {k: v.detach().clone() for k, v in kore_model.state_dict().items()}
    tok = AutoTokenizer.from_pretrained(kore_ckpt)
    # Dedicated scratch model for materialization so load_state_dict never touches
    # the immutable sweep endpoints; base_model is no longer needed.
    scratch = kore_model
    del base_model

    def eval_fn(state_dict) -> dict:
        """Score an interpolated state dict: kernel fast_p + general retention.

        Writes into the scratch model only; the immutable ``base_sd``/``kore_sd``
        endpoints are never mutated, so the sweep is order-independent.
        """
        with tempfile.TemporaryDirectory() as td:
            scratch.load_state_dict(state_dict)
            scratch.save_pretrained(td)
            tok.save_pretrained(td)
            gen = load_generate(td)
            scores = dict(run_retention_suite(gen)["scores"])
            pol = model_policy(td, generate=gen)
            kres = evaluate_policy(pol, tasks, env_factory=lambda t: KoreEnv(t), budget=budget)
            scores[_SOUP_KERNEL_KEY] = float(kres["fast_p"].get(1.0, 0.0))
            return scores

    sweep = soup_sweep(base_sd, kore_sd, cfg.alphas, eval_fn,
                       kernel_key=_SOUP_KERNEL_KEY, general_keys=general_keys,
                       base_scores=base_scores, epsilon=cfg.epsilon)
    best_alpha = sweep["best_alpha"]
    _log("soup", f"alpha sweep over {list(cfg.alphas)} -> best_alpha={best_alpha} "
                 f"(gate_satisfied={sweep['gate_satisfied']}, "
                 f"kernel={sweep['best']['kernel']:.4f})")
    ctx["final"] = build_soup(cfg.base_model_id, cfg.kore_checkpoint, best_alpha, cfg.output_dir)
    _log("soup", f"materialized best-alpha soup -> {ctx['final']}")


def _stage_eval(ctx):
    from kore.eval.bakeoff import matched_budget_bakeoff
    from kore.eval.report import format_bakeoff_table, save_report

    if ctx["dry"]:
        _log("eval", "would run matched-budget fast_p bake-off (seed vs the TRAINED model) "
                     "+ full retention suite on the held-out split")
        return
    from kore.env.kore_env import KoreEnv
    from kore.eval.policies import model_policy, seed_policy

    tasks = _eval_tasks(ctx)
    kore_ckpt = ctx.get("final") or ctx.get("grpo_ckpt") or ctx.get("dpo_ckpt") \
        or ctx.get("sft_ckpt") or ctx["base"]
    _log("eval", f"scoring seed vs KORE checkpoint={kore_ckpt} on tasks="
                 f"{[t.task_id for t in tasks]}")

    policies = {"seed": seed_policy, "kore": model_policy(kore_ckpt)}
    res = matched_budget_bakeoff(policies, tasks, budget=ctx["args"].eval_budget,
                                 env_factory=lambda t: KoreEnv(t), dry_run=None)
    _log("eval", "\n" + format_bakeoff_table(res))
    paths = save_report(res, ctx["data_root"] / "eval" / "bakeoff")
    _log("eval", f"report -> {paths['json']}")


def _retention_gate(ctx, *, stage, candidate, base):
    """Hard-stop the campaign if a stage regresses general ability past epsilon.

    Uses ``retention_gate`` on the retention-suite ``scores`` of base vs candidate;
    a FAIL raises ``SystemExit`` with the formatted report (a real, enforced gate).
    The ONLY swallowed case is an unprovisioned serving backend (no GPU / no
    ``load_generate``), which is logged LOUDLY as "gate NOT enforced" — never a
    blanket except.
    """
    if ctx["dry"]:
        _log(stage, "would run retention gate (no general-bench regression vs base)")
        return

    from kore.eval.gates import format_gate_report, retention_gate
    from kore.eval.retention import run_retention_suite

    # Serving backend provisioning is the ONLY thing we tolerate missing.
    try:
        from kore.policy.serve import load_generate
    except ImportError as e:
        _log(stage, f"WARNING: retention gate NOT enforced — serving backend not "
                    f"provisioned (kore.policy.serve.load_generate unavailable: {e})")
        _emit_event(ctx, stage, "gate_not_enforced", 0.0, None)
        return
    # Fix 5: the ONLY tolerated failure is the serving backend not being
    # provisioned — i.e. an ImportError raised when load_generate tries to import
    # vLLM/torch on a box without them. A CUDA OOM (RuntimeError /
    # torch.cuda.OutOfMemoryError) or a corrupt-checkpoint load error (OSError)
    # is a REAL failure and MUST propagate to fail the run — never swallow it, or
    # the hard-stop retention gate silently disables itself.
    try:
        base_gen = load_generate(base)
        cand_gen = load_generate(candidate)
    except ImportError as e:
        _log(stage, f"WARNING: retention gate NOT enforced — serving backend not "
                    f"provisioned (torch/vLLM unavailable: {e})")
        _emit_event(ctx, stage, "gate_not_enforced", 0.0, None)
        return

    # From here on, failures are REAL and must propagate (no swallowing).
    base_scores = run_retention_suite(base_gen)
    cand_scores = run_retention_suite(cand_gen)
    res = retention_gate(base_scores["scores"], cand_scores["scores"], epsilon=0.005)
    if not res.passed:
        _emit_event(ctx, stage, "gate_failed", 0.0, None)
        raise SystemExit(format_gate_report(res, title=f"KORE retention gate [{stage}]"))
    _log(stage, "retention gate PASSED (no general-bench regression)")
    _emit_event(ctx, stage, "gate_passed", 0.0, None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KORE end-to-end campaign")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--tasks", default=None)
    p.add_argument("--stages", default=None)
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--force", action="store_true",
                   help="re-run requested stages even if the manifest marks them done")
    # Fix 8: LoRA is the default for the 14B validation run so SFT/DPO/GRPO fit on
    # a single node without FSDP/DeepSpeed. Pass --full-ft for the locked full-FT
    # recipe, which REQUIRES a sharded multi-GPU launch (see docs/rl_server.md).
    p.add_argument("--lora", dest="lora", action="store_true", default=True,
                   help="use LoRA on SFT/DPO/GRPO (default; fits the 14B validation run)")
    p.add_argument("--full-ft", dest="lora", action="store_false",
                   help="full fine-tune instead of LoRA (needs an FSDP/DeepSpeed launch)")
    p.add_argument("--data-root", default="data", dest="data_root")
    p.add_argument("--teacher", default="claude")
    p.add_argument("--model-teacher", default=None, dest="model_teacher")
    p.add_argument("--use-hf", action="store_true", dest="use_hf")
    p.add_argument("--n-repair", type=int, default=50, dest="n_repair")
    p.add_argument("--n-parents", type=int, default=20, dest="n_parents")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--wins-gens", type=int, default=8, dest="wins_gens")
    p.add_argument("--n-agentic", type=int, default=16, dest="n_agentic")
    p.add_argument("--max-tool-turns", type=int, default=8, dest="max_tool_turns")
    p.add_argument("--sft-total", type=int, default=20000, dest="sft_total")
    p.add_argument("--sft-out", default="runs/sft", dest="sft_out")
    p.add_argument("--dpo-out", default="runs/dpo", dest="dpo_out")
    p.add_argument("--grpo-out", default="runs/grpo", dest="grpo_out")
    p.add_argument("--grpo-backend", default="fallback", dest="grpo_backend")
    p.add_argument("--soup-out", default="runs/soup", dest="soup_out")
    p.add_argument("--eval-budget", type=int, default=5, dest="eval_budget")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
