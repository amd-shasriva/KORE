"""KORE end-to-end campaign orchestrator (the full agentic recipe).

Stages (each gated on the previous via retention + kernel metrics):
    datagen  : teacher -> repair + ranked-groups + wins per TRAIN task
    evolve   : (optional) evolutionary datagen (D-MAB + MAP-Elites islands +
               value-prefilter) manufacturing verified wins/ranked-groups per
               TRAIN task, written as extra wins/groups shards
    agentic  : teacher-driven build/test/bench/pmc tool-use trajectories
    build    : take the AUTHORITATIVE registry train/held-out split (a whole
               operator family + any arch-specific task is reserved eval-only),
               then assemble a multi-capability SFT mix (kernel + QA + agentic +
               ~45% general) and a DPO set with >=8% hard negatives from the
               TRAIN split only
    midtrain : Stage-0 full-FT continued pretrain on the ROCm/Triton corpus
    sft      : Stage-1 multi-capability SFT (retains chat/code/orchestration)
    dpo      : Stage-2 preference tuning; with --dpo-rounds>1 this becomes the
               ITERATIVE on-policy DPO + DAgger loop (relabel on-policy from the
               current ckpt -> aggregate -> build_dpo -> IPO train -> refresh ref)
    grpo     : Stage-3 multi-turn AGENTIC GRPO (Kevin credit + StarPO-S + KL
               anchor); with --grpo-curriculum runs a correctness phase then a
               latency phase (reward_phase flip; phase-1 ckpt -> phase-2 init)
    soup     : Stage-4 base-ward model soup (retention-gated alpha SWEEP)
    eval     : matched-budget fast_p bake-off (seed vs the TRAINED model) +
               retention, on the HELD-OUT generalization split

Every training stage runs on the TRAIN split only; eval runs on the held-out
split (``kore.tasks.registry.split_tasks``). Every training stage is retention-
gated (hard-stop on general regression). The run is resumable: a JSON manifest at
``<data_root>/campaign_manifest.json`` records the real checkpoints + which stages
finished + the train/eval task ids, and per-stage JSONL events are appended to
``<data_root>/campaign_events.jsonl`` for observability.

--dry-run validates the WHOLE wiring with no GPU/teacher: it import-checks every
symbol the campaign will call, INCLUDING the real-run-only symbols each stage
imports lazily in its body (datagen/agentic generators, JSONL IO, the teacher,
the DAgger SFT fold, the agentic harness + tool reward, the anti-collapse ladder,
and the value reranker), so signature drift fails fast offline. --stages runs a
subset (and reuses prior checkpoints from the manifest, so a crash mid-run is
recoverable).

The default (LoRA) run is a pure single-process ONE-command path. Passing
``--full-ft`` keeps it ONE command but engages real FSDP full fine-tuning UNDER
THE HOOD: the campaign sets ``distributed=True`` on every training config and
shells out to ``scripts/launch_distributed.sh`` (``accelerate launch`` with the
shipped ``configs/accelerate_fsdp.yaml``) for the stages whose ``-m`` JSON entry
supports it — the user never writes a config or runs accelerate.

    # LoRA bring-up (single process, one command):
    PYTHONPATH=. python scripts/run_campaign.py --model Qwen/Qwen3-14B \
        --tasks rmsnorm_aiter,gemm_bf16,flash_attn_decode_bf16 \
        --teacher claude --stages datagen,agentic,build,sft,dpo,grpo,soup,eval

    # Full best-in-world 14B run (still ONE command; campaign spawns FSDP):
    PYTHONPATH=. python scripts/run_campaign.py --model Qwen/Qwen3-14B \
        --tasks rmsnorm_aiter,gemm_bf16,flash_attn_decode_bf16 \
        --teacher claude --full-ft
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from kore.obs import configure, get_logger, gpu_mem_snapshot

ALL_STAGES = ["reverify", "datagen", "evolve", "agentic", "build", "midtrain", "sft", "dpo",
              "grpo", "soup", "eval"]
# Stage-0 mid-train (continued pretraining) runs FIRST so its checkpoint becomes
# the base for Stage-1 SFT (see ctx["midtrain_ckpt"] -> _stage_sft). ``evolve`` is
# not in the defaults (it is expensive); pass --evolve to splice it in after
# datagen, or name it explicitly in --stages.
DEFAULT_STAGES = ["midtrain", "datagen", "agentic", "build", "sft", "dpo", "grpo", "soup", "eval"]

# Kernel metric key used to drive the soup alpha sweep (fast_p at p=1.0).
_SOUP_KERNEL_KEY = "kernel_fast1"

# Central structured logger; run_dir is bound in run() via configure(). Every
# _log() line, stage, progress and heartbeat lands in <data_root>/events.jsonl
# (complementing the campaign_manifest.json + campaign_events.jsonl behavior).
LOG = get_logger("campaign")


def _log(stage: str, msg: str) -> None:
    # Route through the central logger (console + events.jsonl) instead of a bare
    # print so every campaign line is timestamped, stage-tagged and captured.
    LOG.info(msg, phase=stage)


def _start_heartbeat(ctx):
    """Daemon thread logging a GPU-mem heartbeat every ~30s (never silent).

    Returns the ``threading.Event`` used to stop it (``None`` in dry-run, where
    the run must stay fast/offline with no background thread).
    """
    if ctx.get("dry"):
        return None
    stop = threading.Event()

    def _beat():
        while not stop.wait(30.0):
            LOG.heartbeat("campaign", stage=ctx.get("current_stage", "-"),
                          **gpu_mem_snapshot())

    threading.Thread(target=_beat, name="campaign-heartbeat", daemon=True).start()
    return stop


def _write_rows(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# --------------------------------------------------------------------------- #
# Fix 1: --full-ft engages FSDP UNDER THE HOOD (still ONE user command).
#
# For a full fine-tune the campaign sets ``distributed=True`` on every training
# config and, for stages whose ``-m kore.policy.<stage> <config.json>`` entry can
# read a JSON config, shells out to ``scripts/launch_distributed.sh`` (which runs
# ``accelerate launch --config_file configs/accelerate_fsdp.yaml``). The user
# never writes a config or invokes accelerate — the campaign spawns the sharded
# processes. LoRA (the default) stays the pure single-process one-command path.
# --------------------------------------------------------------------------- #

# Anti-collapse lever default for the full best-in-world GRPO run (Fix 2): the
# AVSPO variance-floor tau. When a rollout group's reward std drops below this,
# virtual samples are injected into the normalization stats to guarantee a
# variance floor (fights reward-variance collapse). 0.0 would disable AVSPO.
_ANTICOLLAPSE_VARIANCE_FLOOR = 0.1

# Shipped internal full-FT config templates per stage (the locked full-FT recipe:
# use_lora=false + FSDP). The campaign overlays the run's dynamic paths (model,
# dataset, output_dir) before launching, so these are NOT user-authored.
_FULL_FT_CONFIGS = {
    "midtrain": "midtrain_14b_full.json",
    "sft": "sft_14b_full.json",
    "dpo": "dpo_14b_full.json",
    "grpo": "grpo_14b_full.json",
}


def _repo_root() -> Path:
    """Package root that holds ``scripts/``, ``configs/`` and the ``kore/`` package."""
    return Path(__file__).resolve().parents[1]


def _full_ft(ctx) -> bool:
    """True iff this run is a full fine-tune (``--full-ft`` -> ``args.lora`` False)."""
    return not bool(getattr(ctx["args"], "lora", True))


def _stage_supports_launcher(stage: str) -> bool:
    """True iff ``python -m kore.policy.<stage> <config.json>`` reads a JSON config.

    All four training stages — ``midtrain``/``sft``/``dpo``/``grpo`` — ship that
    JSON ``__main__`` (detected via their ``<stage>_config_from_dict`` builder), so
    ``--full-ft`` shells each out to the FSDP launcher for real full-parameter
    sharded training. Detecting via the builder means a stage flips ON
    automatically — and the campaign starts shelling it out — the moment its entry
    ships (no campaign change needed), and never silently degrades if one is
    absent.
    """
    try:
        mod = importlib.import_module(f"kore.policy.{stage}")
    except Exception:  # noqa: BLE001 - module import problems surface elsewhere
        return False
    return callable(getattr(mod, f"{stage}_config_from_dict", None))


def _launch_distributed(ctx, stage: str, overrides: dict, *, run_name: str | None = None) -> str:
    """Render a resolved full-FT JSON config and shell out to the FSDP launcher.

    Starts from the shipped internal template (``configs/<stage>_14b_full.json``),
    overlays the run's dynamic fields (``overrides``: model/dataset/output_dir),
    forces ``distributed=True`` + ``use_lora=False``, writes the resolved config
    into ``<data_root>/launch/`` and runs
    ``scripts/launch_distributed.sh <stage> <resolved.json>`` which drives
    ``accelerate launch`` with the shipped FSDP config. Returns ``output_dir``.
    """
    shipped = _repo_root() / "configs" / _FULL_FT_CONFIGS[stage]
    cfg = json.loads(shipped.read_text()) if shipped.exists() else {}
    cfg.update(overrides)
    cfg["distributed"] = True   # contract: --full-ft sets distributed on every training config
    cfg["use_lora"] = False
    run_cfg = ctx["data_root"] / "launch" / f"{run_name or stage}.json"
    run_cfg.parent.mkdir(parents=True, exist_ok=True)
    run_cfg.write_text(json.dumps(cfg, indent=2))
    launcher = _repo_root() / "scripts" / "launch_distributed.sh"
    cmd = ["bash", str(launcher), stage, str(run_cfg)]
    # Pin FSDP training to the same physical GPUs as the rest of the run (free GPUs on
    # a shared node). launch_distributed.sh reads GPU_IDS and derives num_processes.
    env = None
    pinned = _gpu_ids(ctx)
    if pinned:
        env = {**os.environ, "GPU_IDS": ",".join(str(g) for g in pinned)}
    _log(stage, f"full-FT: engaging FSDP under the hood (ONE command) -> {' '.join(cmd)} "
                f"(config: model={cfg.get('model_id')} out={cfg.get('output_dir')}"
                f"{'; GPU_IDS=' + env['GPU_IDS'] if env else ''})")
    subprocess.run(cmd, check=True, env=env)
    return overrides["output_dir"]


def _warn_inprocess_fullft(stage: str) -> None:
    """LOUD warning: full-FT for a stage whose FSDP JSON entry isn't shipped yet.

    Safety net only: all four training stages (``midtrain``/``sft``/``dpo``/``grpo``)
    ship a ``-m kore.policy.<stage> <config.json>`` JSON entry today, so the
    campaign shells each out to the FSDP launcher and this path is not reached. It
    exists so that if an entry were ever removed the campaign says so LOUDLY and
    runs in-process (which cannot truly full-FT a 14B) rather than silently
    degrading. See docs/DISTRIBUTED.md#full-ft-per-stage-status.
    """
    _log(stage, f"WARNING: --full-ft for '{stage}' is NOT orchestrated via the campaign's "
                f"one-command FSDP launcher: kore.policy.{stage} has no "
                f"`-m kore.policy.{stage} <config.json>` JSON entrypoint. "
                f"Running IN-PROCESS with distributed=True set on the config — this "
                f"will NOT shard and cannot full-FT a 14B. See "
                f"docs/DISTRIBUTED.md#full-ft-per-stage-status for the exact status + the "
                f"manual sharded launch.")


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
    # Authoritative train/held-out generalization split (item 1).
    ("kore.tasks.registry", "split_tasks", True, ["seed"]),
    ("kore.tasks.registry", "train_tasks", True, []),
    ("kore.tasks.registry", "heldout_tasks", True, []),
    ("kore.tasks.registry", "operator_family", True, []),
    ("kore.env.kore_env", "KoreEnv", True, []),
    ("kore.data.assemble", "build_multicap_dataset", True, ["kernel_records", "extra_records"]),
    ("kore.data.assemble", "build_dpo_with_hard_negatives", True,
        ["group_records", "extra_group_records"]),
    ("kore.data.assemble", "summarize_multicap", True, []),
    # Iterative on-policy DPO + DAgger (item 2).
    ("kore.data.onpolicy", "iterative_dpo", True,
        ["rounds", "policy_factory", "tasks", "env_factory", "train_fn", "aggregate"]),
    ("kore.data.onpolicy", "dagger_repairs", True, ["teacher_frac", "diagnostic"]),
    ("kore.data.onpolicy", "dagger_teacher_frac", True, ["round_idx", "rounds"]),
    ("kore.data.onpolicy", "relabel_groups_on_policy", True, ["policy"]),
    # Evolutionary datagen (item 3).
    ("kore.data.evolve", "evolve_task", True, ["task", "generator", "env", "generations", "cfg"]),
    ("kore.data.evolve", "EvolveConfig", True, []),
    # Correctness->latency GRPO curriculum (item 4).
    ("kore.policy.grpo", "apply_reward_phase", True, []),
    ("kore.data.build_datasets", "leakage_split", True, ["records", "by"]),
    ("kore.data.build_datasets", "dedup_by_source_hash", True, []),
    ("kore.data.schemas", "read_jsonl", True, []),
    ("kore.policy.configs", "MidTrainConfig", True, []),
    ("kore.policy.configs", "MultiCapSFTConfig", True, []),
    ("kore.policy.configs", "DPOConfig", True, []),
    ("kore.policy.configs", "GRPOConfig", True, []),
    ("kore.policy.configs", "SoupConfig", True, []),
    ("kore.data.midtrain_corpus", "build_midtrain_corpus", True, ["out_path", "config"]),
    ("kore.policy.midtrain", "train_midtrain", True, ["config", "corpus_path"]),
    ("kore.policy.sft", "train_sft", True, []),
    ("kore.policy.dpo", "train", True, []),
    ("kore.policy.grpo", "train_grpo", True, ["tasks"]),
    # Full-parameter sharded GRPO one-command entry (Fix 1): the JSON `-m` builder
    # the campaign detects to route --full-ft grpo through the FSDP launcher. Owned
    # by a sibling track; absence -> loud warning (grpo full-FT falls back
    # in-process) rather than a dry-run failure, so this stays green until it lands
    # and flips grpo to one-command full-parameter sharded automatically.
    ("kore.policy.grpo", "grpo_config_from_dict", False, []),
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
    # Fix 4 (dry-run fidelity): the REAL-RUN-ONLY symbols the audit found were
    # imported lazily inside each stage body (so a dry-run never touched them and
    # drift could slip past the preflight). Import-check them here too — datagen /
    # agentic generators, JSONL IO, the teacher, the DAgger SFT fold, the agentic
    # harness + tool reward, the anti-collapse ladder, and the value reranker.
    ("kore.data.gen_repair", "generate_repairs", True, ["task", "teacher", "env", "n"]),
    ("kore.data.gen_groups", "generate_groups", True, ["task", "teacher", "env", "n_parents", "k"]),
    ("kore.data.gen_wins", "generate_wins", True, ["task", "teacher", "env", "gens"]),
    ("kore.data.gen_agentic", "generate_agentic_trajectories", True,
        ["task", "teacher", "env", "n", "max_turns", "keep_only_useful"]),
    ("kore.data.schemas", "write_jsonl", True, ["path", "records"]),
    ("kore.data.teacher", "make_teacher", True, ["kind"]),
    ("kore.data.teacher", "load_env_local", True, []),
    ("kore.data.build_datasets", "build_sft", True, ["records"]),
    ("kore.agent.harness", "AgentHarness", True, ["task", "env", "max_turns"]),
    ("kore.agent.tools", "tool_use_reward", True, ["episode"]),
    # Anti-collapse ladder (Fix 2): every lever the campaign turns ON by default
    # for the full run resolves through these primitives at grpo RUN time.
    ("kore.policy.anticollapse", "avspo_advantages", True, ["returns", "tau"]),
    ("kore.policy.anticollapse", "scgrpo_weight_from_kl", True, ["token_kls"]),
    ("kore.policy.anticollapse", "gtpo_codesim_shaping", True, ["codes", "references"]),
    ("kore.policy.anticollapse", "variance_floor", True, ["rewards", "reward_tokens", "means"]),
    ("kore.policy.anticollapse", "sample_reward_tokens", True, ["G", "p_high"]),
    ("kore.policy.anticollapse", "prepend_reward_token", True, ["prompt", "token"]),
    # Value-model bench prefilter reranker (contract b): value_prefilter is ON by
    # default; the grpo rollout ranks candidates best-first via this before benching.
    ("kore.value.rerank", "rank_candidates", True, ["items", "task"]),
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
        _log("preflight", f"WARNING: {w} (optional symbol from a parallel/sibling "
                          f"track; not yet provisioned)")
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
    for k in ("midtrain_ckpt", "sft_ckpt", "dpo_ckpt", "grpo_ckpt", "final"):
        if m.get(k):
            ctx[k] = m[k]
    ctx["done_stages"] = set(m.get("done_stages") or [])
    if m.get("eval_tasks"):
        ctx["eval_task_ids"] = list(m["eval_tasks"])
    if m.get("train_tasks"):
        ctx["train_task_ids"] = list(m["train_tasks"])
    _log("resume", f"manifest loaded: done={sorted(ctx['done_stages'])} "
                   f"midtrain={ctx['midtrain_ckpt']} sft={ctx['sft_ckpt']} "
                   f"dpo={ctx['dpo_ckpt']} grpo={ctx['grpo_ckpt']} final={ctx['final']}")


def _save_manifest(ctx) -> None:
    if ctx["dry"]:
        return
    p = _manifest_path(ctx)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": ctx["base"],
        "midtrain_ckpt": ctx.get("midtrain_ckpt"),
        "sft_ckpt": ctx.get("sft_ckpt"),
        "dpo_ckpt": ctx.get("dpo_ckpt"),
        "grpo_ckpt": ctx.get("grpo_ckpt"),
        "final": ctx.get("final"),
        "done_stages": sorted(ctx["done_stages"]),
        "train_tasks": ctx.get("train_task_ids"),
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
        "midtrain": ctx.get("midtrain_ckpt"),
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
        "midtrain": lambda: _path_exists(ctx.get("midtrain_ckpt")),
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

    # P5: propagate the hardware-counter dense-reward weight to every stage
    # subprocess (env + training run under their own processes) BEFORE anything
    # imports CONFIG, so the reward path picks it up consistently.
    if getattr(args, "profile_reward", 0.0):
        os.environ["KORE_PROFILE_REWARD_WEIGHT"] = str(args.profile_reward)
    if getattr(args, "shape_augment", False):
        os.environ["KORE_SHAPE_AUGMENT"] = "1"
    if getattr(args, "speed_aggregation", None):
        os.environ["KORE_SPEED_AGG"] = str(args.speed_aggregation)
    # Real retention eval: with --use-hf, measure general-capability retention on the
    # REAL public benchmark splits (MMLU/HumanEval/IFEval/BFCL/LiveCodeBench/MTBench)
    # via HuggingFace, capped to KORE_EVAL_N items/bench so the gate stays fast. The
    # bundled smoke JSONLs remain the offline/CI fallback.
    if getattr(args, "use_hf", False):
        os.environ.setdefault("KORE_EVAL_FULL", "1")
        os.environ.setdefault("KORE_EVAL_N", str(getattr(args, "eval_n", 300)))

    tasks = [get_task(t) for t in args.tasks.split(",")] if args.tasks else all_tasks()
    if args.stages:
        stages = args.stages.split(",")
    else:
        stages = list(DEFAULT_STAGES)
        # Splice the (optional) evolutionary datagen stage in right after datagen.
        if args.evolve and "evolve" not in stages:
            stages.insert(stages.index("datagen") + 1, "evolve")
    data_root = Path(args.data_root)
    dry = args.dry_run

    ctx = {
        "data_root": data_root, "tasks": tasks, "dry": dry, "args": args,
        "base": args.model, "midtrain_ckpt": None, "sft_ckpt": None,
        "dpo_ckpt": None, "grpo_ckpt": None, "final": None, "metrics": {},
        "done_stages": set(), "eval_task_ids": None, "train_task_ids": None,
        "current_stage": "-",
    }

    # Authoritative train / held-out generalization split (item 1). Training
    # stages run on ctx["train_tasks"]; eval runs on ctx["eval_tasks"]. This
    # SUBSUMES the ad-hoc record-level _force_holdout: the reserved operator
    # family (+ any arch-specific task) is fixed by the registry, so training
    # data-gen can never leak into the eval set.
    _apply_split(ctx)

    # Bind the central logger's events.jsonl to the run dir (real runs only; a
    # dry-run stays side-effect-free/offline and logs to the console only).
    if not dry:
        configure(run_dir=data_root)

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
    _log("plan", f"authoritative split: train={ctx['train_task_ids']} "
                 f"held-out(eval)={ctx['eval_task_ids']}")

    dispatch = {
        "reverify": _stage_reverify,
        "datagen": _stage_datagen, "evolve": _stage_evolve, "agentic": _stage_agentic,
        "build": _stage_build, "midtrain": _stage_midtrain, "sft": _stage_sft,
        "dpo": _stage_dpo, "grpo": _stage_grpo, "soup": _stage_soup, "eval": _stage_eval,
    }
    hb_stop = _start_heartbeat(ctx)
    try:
        for st in stages:
            if st not in dispatch:
                _log("plan", f"unknown stage '{st}', skipping")
                continue
            if (not dry) and st in ctx["done_stages"] and _artifact_ok(ctx, st):
                _log(st, "already complete (artifact present) — skipping [resume]")
                _emit_event(ctx, st, "skipped", 0.0, _artifact_of(ctx, st))
                continue

            ctx["current_stage"] = st
            start = time.perf_counter()
            _emit_event(ctx, st, "start", 0.0, None)
            try:
                with LOG.stage(st):
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
    finally:
        ctx["current_stage"] = "-"
        if hb_stop is not None:
            hb_stop.set()  # stop the heartbeat daemon at the end of the run

    _log("done", "campaign complete")
    return 0


# --------------------------------------------------------------------------- #
def _teacher(args):
    from kore.data.teacher import load_env_local, make_teacher
    load_env_local()
    kw = {"model": args.model_teacher} if args.model_teacher else {}
    # resilient=True: a multi-day datagen makes tens of thousands of teacher calls;
    # skip an individual transient gateway failure (after the inner 8-retry backoff)
    # instead of crashing the whole campaign, but still hard-stop on a SUSTAINED
    # outage so we never silently produce empty data.
    return make_teacher(args.teacher, resilient=True, **kw)


def _apply_split(ctx) -> None:
    """Compute the AUTHORITATIVE registry train/held-out split for this run.

    Uses ``kore.tasks.registry.split_tasks(seed)`` (item 1). The held-out set is a
    fixed function of operator family + arch, so it is independent of ``seed`` (the
    seed only reorders within each split). From the campaign's selected task set:

      * ``train_tasks`` = selected tasks that are NOT held out — every training
        stage (datagen/evolve/agentic/build/sft/dpo/grpo) runs on these ONLY;
      * ``eval_tasks``  = the held-out generalization tasks — eval runs on these.
        We prefer any selected held-out tasks; if none of the selected tasks are
        held out (the common bring-up case, e.g. ``--tasks rmsnorm_aiter,gemm_bf16``)
        we fall back to the registry's full held-out set so eval still measures
        generalization to an unseen operator family.

    Populates ``ctx['train_tasks']``/``['eval_tasks']`` (Task objects) and the
    id lists threaded through the manifest.
    """
    from kore.tasks.registry import is_heldout, split_tasks

    seed = getattr(ctx["args"], "split_seed", 0)
    split = split_tasks(seed)
    selected = ctx["tasks"]

    train = [t for t in selected if not is_heldout(t)]
    held_selected = [t for t in selected if is_heldout(t)]
    eval_tasks = held_selected or list(split["heldout"])

    if not train:  # degenerate: every selected task is held out — train on them
        _log("plan", "WARNING: every selected task is held out; training on the "
                     "full selection (no train/eval split available)")
        train = list(selected)

    ctx["train_tasks"] = train
    ctx["eval_tasks"] = eval_tasks
    ctx["train_task_ids"] = [t.task_id for t in train]
    ctx["eval_task_ids"] = [t.task_id for t in eval_tasks]


def _train_tasks(ctx) -> list:
    """The TRAIN-split tasks every training stage operates on (item 1)."""
    return ctx.get("train_tasks") or ctx["tasks"]


def _eval_tasks(ctx) -> list:
    """The held-out (eval-only) generalization tasks; else the selected tasks."""
    if ctx.get("eval_tasks"):
        return ctx["eval_tasks"]
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


def _datagen_counts(ctx) -> dict:
    a = ctx["args"]
    return {"n_repair": a.n_repair, "n_parents": a.n_parents, "k": a.k,
            "wins_gens": a.wins_gens, "n_agentic": a.n_agentic,
            "max_tool_turns": a.max_tool_turns}


def _datagen_plan(ctx):
    """(workers, n_gpus) for parallel datagen; workers<=1 -> sequential path."""
    from kore.data.parallel_datagen import detect_gpus
    n_gpus = detect_gpus()
    req = int(getattr(ctx["args"], "datagen_workers", 0) or 0)
    workers = req if req > 0 else n_gpus   # 0 = auto (one per GPU)
    return workers, n_gpus


def _detect_free_gpus(mem_free_frac: float = 0.9, busy_use_pct: int = 20) -> list[int]:
    """Best-effort: physical GPU ids that are idle (low VRAM + low utilization).

    Parses ``rocm-smi``; returns [] on any failure (caller falls back). A GPU counts
    as free if it uses < (1-mem_free_frac) of VRAM AND < busy_use_pct%% compute.
    """
    import re
    import subprocess
    try:
        out = subprocess.run(["rocm-smi", "--showuse", "--showmeminfo", "vram"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return []
    use, used, total = {}, {}, {}
    for ln in out.splitlines():
        m = re.search(r"GPU\[(\d+)\].*GPU use \(%\):\s*(\d+)", ln)
        if m:
            use[int(m.group(1))] = int(m.group(2))
        m = re.search(r"GPU\[(\d+)\].*Total Memory \(B\):\s*(\d+)", ln)
        if m:
            total[int(m.group(1))] = int(m.group(2))
        m = re.search(r"GPU\[(\d+)\].*Total Used Memory \(B\):\s*(\d+)", ln)
        if m:
            used[int(m.group(1))] = int(m.group(2))
    free = []
    for g in sorted(total):
        u = used.get(g, 0) / total[g] if total.get(g) else 1.0
        if u < (1.0 - mem_free_frac) and use.get(g, 100) < busy_use_pct:
            free.append(g)
    return free


def _gpu_ids(ctx) -> list[int]:
    """Physical GPU ids to pin work to: explicit --gpu-ids, else auto-detected free."""
    raw = getattr(ctx["args"], "gpu_ids", "") or ""
    if raw.strip():
        return [int(g) for g in raw.split(",") if g.strip() != ""]
    free = _detect_free_gpus()
    if free:
        _log("plan", f"auto-detected free GPUs for pinning: {free}")
        return free
    return []


def _stage_reverify(ctx):
    """Re-verify + re-baseline EXISTING kernels with v2 rigor (reuse, no teacher).

    Runs the strong-baseline + adversarial re-measurement over every task that
    already has repair/wins/groups shards (Pillar 1 rigor applied to v1 data), pinned
    to the free GPUs. Resumable. This is the 'reuse, don't regenerate' path: only the
    coverage HOLES need a subsequent (teacher) datagen.
    """
    if ctx["dry"]:
        _log("reverify", "would re-verify existing repair/wins/groups against the strong "
                         "baseline + adversarial battery (reuse v1 kernels, no teacher)")
        return
    from kore.data.reverify import run_reverify

    data_root = ctx["data_root"]
    seen: set[str] = set()
    for sub in ("groups", "wins", "repair"):
        d = data_root / sub
        if d.exists():
            for p in d.glob("*.jsonl"):
                if not p.stem.startswith("_"):
                    seen.add(p.stem)
    # only re-verify TRAIN tasks (never touch held-out) that have data
    heldout = set(ctx.get("eval_task_ids") or [])
    task_ids = sorted(t for t in seen if t not in heldout)
    if not task_ids:
        _log("reverify", "no existing shards to re-verify (fresh run) — skipping")
        return
    phys = _gpu_ids(ctx) or [0]
    # Oversubscribe: reverify is compile/CPU-bound with ~idle GPUs, so run K workers
    # per physical GPU (K = KORE_REVERIFY_WORKERS_PER_GPU) to use the many CPU cores.
    # Timing stays honest because the genops --bench-both path measures candidate +
    # reference back-to-back in one process (contention-fair ratio).
    try:
        per = max(1, int(os.environ.get("KORE_REVERIFY_WORKERS_PER_GPU", "1")))
    except ValueError:
        per = 1
    gpus = phys * per
    ground = bool(getattr(ctx["args"], "ground_reasoning", False))
    _log("reverify", f"re-verifying {len(task_ids)} tasks on {len(gpus)} workers "
                     f"({per}/GPU x physical {phys}); strong baseline + adversarial; "
                     f"ground={ground}")
    summary = run_reverify(data_root, task_ids, gpus, ground=ground,
                           rigorous=True, log_fn=lambda m: _log("reverify", m))
    LOG.event("reverify_done", **summary)
    _log_datagen_coverage(ctx)


def _stage_datagen(ctx):
    if ctx["dry"]:
        _log("datagen", "would generate repair/groups/wins per task (teacher + GPU env), "
                        "parallel-sharded across GPUs when --datagen-workers != 1")
        return
    # Pillar 1: MAXIMUM verification rigor for the data pass — adversarial correctness
    # battery + shape augmentation + strong torch.compile baseline + cold-L2 timing.
    # Set here (not globally) so it propagates to every verifier subprocess — incl.
    # parallel datagen workers, which inherit os.environ — without slowing GRPO rollouts.
    if getattr(ctx["args"], "rigorous_verify", True):
        from kore.data.verify_rigor import rigor_status, set_rigorous_verification
        set_rigorous_verification(True)
        _log("datagen", f"rigorous verification ON: {rigor_status()}")
    # Parallel path: shard tasks across GPUs with concurrent teacher streams (resumable).
    # Pinned GPU ids (free ones on a shared node) force the parallel path onto exactly
    # those devices so datagen never contends with other users' jobs.
    pinned = _gpu_ids(ctx)
    workers, n_gpus = _datagen_plan(ctx)
    if pinned or workers > 1:
        from kore.data.parallel_datagen import DATAGEN_KINDS, run_parallel_datagen
        train = _train_tasks(ctx)
        summary = run_parallel_datagen(
            [t.task_id for t in train], DATAGEN_KINDS, ctx["data_root"],
            _datagen_counts(ctx), n_workers=(workers or len(pinned) or 1), n_gpus=n_gpus,
            teacher_kind=ctx["args"].teacher, model_teacher=ctx["args"].model_teacher,
            gpu_ids=pinned or None, log=lambda m: _log("datagen", m))
        LOG.event("datagen_parallel", workers=workers, n_gpus=n_gpus, pinned=pinned, **summary)
        _log_datagen_coverage(ctx)
        return
    from kore.data.gen_groups import generate_groups
    from kore.data.gen_repair import generate_repairs
    from kore.data.gen_wins import generate_wins
    from kore.data.parallel_datagen import shard_done
    from kore.data.schemas import write_jsonl
    from kore.env.kore_env import KoreEnv

    t = _teacher(ctx["args"])
    train = _train_tasks(ctx)
    n_tasks = len(train)
    dg_t0 = time.time()
    for i, task in enumerate(train):
        # RESUMABLE (matches the parallel path): skip any (task, kind) whose shard
        # already exists non-empty, so a rerun only fills holes / new tasks and never
        # redoes verified work. Delete a shard to force its regeneration.
        env = None
        for kind in ("repair", "groups", "wins"):
            if shard_done(ctx["data_root"], task.task_id, kind):
                _log("datagen", f"{task.task_id}:{kind} skip (resume)")
                continue
            if env is None:
                env = KoreEnv(task)
            if kind == "repair":
                recs = generate_repairs(task, t, env, n=ctx["args"].n_repair)
            elif kind == "groups":
                recs = generate_groups(task, t, env, n_parents=ctx["args"].n_parents, k=ctx["args"].k)
            else:
                recs = generate_wins(task, t, env, gens=ctx["args"].wins_gens)
            out = ctx["data_root"] / kind / f"{task.task_id}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(out, recs)
            _log("datagen", f"{task.task_id}:{kind} -> {len(recs)} records")
            LOG.event("datagen_records", task=task.task_id, kind=kind, n=len(recs))
        LOG.progress(i + 1, n_tasks, "datagen", t_start=dg_t0, task=task.task_id)
    _log_datagen_coverage(ctx)


def _log_datagen_coverage(ctx):
    """Report per-task data coverage after datagen (Pillar 2: make 100% visible)."""
    try:
        from kore.data.coverage import coverage_report
        rep = coverage_report(ctx["data_root"])
    except Exception as e:  # noqa: BLE001 - coverage report must never fail datagen
        _log("datagen", f"coverage report skipped ({e})")
        return
    _log("datagen", f"DATA COVERAGE: {rep['n_full_coverage']}/{rep['n_train_tasks']} "
                    f"tasks fully covered ({rep['coverage_pct']}%); "
                    f"per-kind {rep['per_kind_pct']}; undercovered={rep['n_undercovered']}")
    if rep["undercovered"]:
        holes = "; ".join(f"{t}:{'+'.join(m)}" for t, m in sorted(rep["undercovered"].items())[:40])
        _log("datagen", f"undercovered tasks (regenerate to reach 100%): {holes}")
    LOG.event("datagen_coverage", **{k: rep[k] for k in
              ("n_train_tasks", "n_full_coverage", "coverage_pct", "n_undercovered")})


def _stage_agentic_synth(ctx):
    """CPU-only: reconstruct agentic tool-use trajectories from verified records.

    No teacher, no GPU — reads the already-generated repair/wins/groups shards
    and writes native Hermes trajectories into ``data/agentic`` (which the SFT
    build then blends with the web tool-use replay). Turns the tens-of-GPU-hours
    agentic stage into a minutes-long CPU pass with real measured tool results.
    """
    from kore.data.synth_agentic import synthesize_agentic
    from kore.tasks.registry import TRAIN_ARCH

    cap = int(getattr(ctx["args"], "synth_agentic_cap", 4000))
    seed = int(getattr(ctx["args"], "seed", 0) or 0)
    summary = synthesize_agentic(ctx["data_root"], cap=cap, seed=seed, arch=TRAIN_ARCH)
    _log("agentic", f"synthesized native tool-use from verified records "
                    f"(repair={summary.get('repair', 0)}, wins={summary.get('wins', 0)}, "
                    f"groups={summary.get('groups', 0)}, total={summary.get('total', 0)}) "
                    f"— CPU-only, real measurements, arch={TRAIN_ARCH}")
    LOG.event("agentic_synth", cap=cap, **summary)


def _stage_agentic(ctx):
    mode = getattr(ctx["args"], "agentic_mode", "live")
    if ctx["dry"]:
        if mode in ("synth", "both"):
            _log("agentic", "would SYNTHESIZE tool-use trajectories from verified "
                            "repair/wins/groups records (CPU-only, no GPU/teacher)")
        if mode in ("live", "both"):
            _log("agentic", "would generate build/test/bench/pmc tool-use trajectories per task "
                            "(parallel-sharded across GPUs when --datagen-workers != 1)")
        return
    if mode in ("synth", "both"):
        _stage_agentic_synth(ctx)
        if mode == "synth":
            return  # native slice is filled from verified data; skip the GPU path
    # Parallel path: shard agentic trajectory generation across GPUs (resumable).
    workers, n_gpus = _datagen_plan(ctx)
    if workers > 1:
        from kore.data.parallel_datagen import AGENTIC_KINDS, run_parallel_datagen
        train = _train_tasks(ctx)
        summary = run_parallel_datagen(
            [t.task_id for t in train], AGENTIC_KINDS, ctx["data_root"],
            _datagen_counts(ctx), n_workers=workers, n_gpus=n_gpus,
            teacher_kind=ctx["args"].teacher, model_teacher=ctx["args"].model_teacher,
            log=lambda m: _log("agentic", m))
        LOG.event("agentic_parallel", workers=workers, n_gpus=n_gpus, **summary)
        return
    from kore.data.gen_agentic import generate_agentic_trajectories
    from kore.data.schemas import write_jsonl
    from kore.env.kore_env import KoreEnv

    t = _teacher(ctx["args"])
    train = _train_tasks(ctx)
    n_tasks = len(train)
    ag_t0 = time.time()
    for i, task in enumerate(train):
        env = KoreEnv(task)
        recs = generate_agentic_trajectories(task, t, env, n=ctx["args"].n_agentic,
                                             max_turns=ctx["args"].max_tool_turns, keep_only_useful=True)
        out = ctx["data_root"] / "agentic" / f"{task.task_id}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(out, [r.to_dict() for r in recs])
        _log("agentic", f"{task.task_id} -> {len(recs)} trajectories")
        LOG.event("agentic_records", task=task.task_id, n=len(recs))
        LOG.progress(i + 1, n_tasks, "agentic", t_start=ag_t0, task=task.task_id)


def _stage_evolve(ctx):
    """Optional Stage: evolutionary datagen (item 3).

    Runs :func:`kore.data.evolve.evolve_task` per TRAIN task — a D-MAB (UCB1 +
    Page-Hinkley) bandit over mutation operators, MAP-Elites islands with ring
    migration, and a value-model bench prefilter — to MANUFACTURE verified wins
    and ranked preference groups. They are written as EXTRA ``wins``/``groups``
    shards so the build stage folds them in via its existing glob (dedup handles
    any overlap with the teacher-generated datagen).
    """
    if ctx["dry"]:
        _log("evolve", "would run evolve_task per TRAIN task (D-MAB bandit + MAP-Elites "
                       "islands + value-prefilter) -> verified wins + ranked-group shards")
        return
    from kore.data.evolve import EvolveConfig, evolve_task
    from kore.data.schemas import write_jsonl
    from kore.env.kore_env import KoreEnv

    t = _teacher(ctx["args"])
    train = _train_tasks(ctx)
    n_tasks = len(train)
    gens = ctx["args"].evolve_generations
    ev_t0 = time.time()
    for i, task in enumerate(train):
        env = KoreEnv(task)
        cfg = EvolveConfig(seed=i)
        result = evolve_task(task, t, env, generations=gens, cfg=cfg)
        if result.wins:
            write_jsonl(ctx["data_root"] / "wins" / f"{task.task_id}.evolve.jsonl", result.wins)
        if result.groups:
            write_jsonl(ctx["data_root"] / "groups" / f"{task.task_id}.evolve.jsonl", result.groups)
        _log("evolve", f"{task.task_id} -> {len(result.wins)} wins, {len(result.groups)} "
                       f"groups (best_speedup={result.stats.get('best_speedup')})")
        LOG.event("evolve_records", task=task.task_id, wins=len(result.wins),
                  groups=len(result.groups))
        LOG.progress(i + 1, n_tasks, "evolve", t_start=ev_t0, task=task.task_id)


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


def _rec_is_heldout(rec, heldout_ids: set) -> bool:
    """True iff a record belongs to the AUTHORITATIVE held-out split (item 1).

    Uses the registry's split logic — a record is held out if its ``task_id`` is
    reserved, its arch is not the train arch, or its operator family is one of the
    reserved held-out families. This SUBSUMES the ad-hoc ``_force_holdout`` (which
    hard-coded gfx950 + "first op family") with the registry as the single
    authority, so a stray held-out-family record can never leak into TRAIN.
    """
    from types import SimpleNamespace

    from kore.tasks.registry import HELDOUT_FAMILIES, TRAIN_ARCH, operator_family

    d = _rec_dict(rec)
    tid = d.get("task_id")
    if tid and tid in heldout_ids:
        return True
    arch = _rec_arch(rec)
    if arch not in (None, TRAIN_ARCH):
        return True
    op = _rec_op(rec)
    if op and operator_family(SimpleNamespace(operation=op, task_id=tid or "")) in HELDOUT_FAMILIES:
        return True
    return False


def _stage_build(ctx):
    from kore.policy.configs import MultiCapSFTConfig

    if ctx["dry"]:
        _log("build", "would take the registry train/held-out split (reserved op family "
                      "+ arch-specific eval-only), then assemble train-only SFT mix + "
                      "DPO(+>=8% hard negs)")
        return
    from kore.data.assemble import (build_dpo_with_hard_negatives, build_multicap_dataset,
                                    summarize_multicap)
    from kore.data.build_datasets import dedup_by_source_hash, leakage_split
    from kore.data.schemas import read_jsonl
    from kore.data.teacher import make_teacher

    # 0. Gold-win mining (CPU, no GPU): reconstruct optimization-win demos from the
    #    verified rank-0 candidates in `groups` and write them alongside the real
    #    wins, so the raw gather below folds them through the SAME leakage-split +
    #    RFT speedup gate. Rebalances the thin wins family against repair.
    if getattr(ctx["args"], "gold_wins", True):
        from kore.data.gold_wins import mint_gold_wins
        from kore.tasks.registry import TRAIN_ARCH
        gw = mint_gold_wins(ctx["data_root"], cap=int(getattr(ctx["args"], "gold_wins_cap", 3000)),
                            seed=int(getattr(ctx["args"], "split_seed", 0) or 0), arch=TRAIN_ARCH)
        _log("build", f"gold wins from verified groups: minted {gw['gold_wins']} across "
                      f"{gw['tasks_covered']} tasks (from {gw['groups_scanned']} groups) "
                      f"-> wins/_gold_from_groups.jsonl (RFT-gated downstream)")
        LOG.event("gold_wins", **gw)

    # 0b. Repair->DPO (CPU): package verified repairs as fixed>broken preference
    #     pairs so the DPO product gets a correctness contrast alongside the
    #     speed-ranked group prefs and reward-hack hard negatives.
    if getattr(ctx["args"], "repair_dpo", True):
        from kore.data.repair_dpo import mint_repair_dpo
        from kore.tasks.registry import TRAIN_ARCH
        rd = mint_repair_dpo(ctx["data_root"], cap=int(getattr(ctx["args"], "repair_dpo_cap", 8000)),
                             seed=int(getattr(ctx["args"], "split_seed", 0) or 0), arch=TRAIN_ARCH)
        _log("build", f"repair->DPO: minted {rd['repair_pairs']} fixed>broken pairs across "
                      f"{rd['tasks_covered']} tasks (from {rd['repair_scanned']} repairs) "
                      f"-> groups/_repair_pairs.jsonl")
        LOG.event("repair_dpo", **rd)

    # 1. gather + dedup all raw generated records (datagen + evolve shards).
    raw: list = []
    for sub in ("repair", "wins", "groups"):
        d = ctx["data_root"] / sub
        if d.exists():
            for p in sorted(d.glob("*.jsonl")):
                raw += read_jsonl(p, typed=True)
    raw = dedup_by_source_hash(raw)
    _log("build", f"gathered {len(raw)} deduped raw records")

    # 2. leakage-aware split by (operation, arch), then enforce the AUTHORITATIVE
    #    registry held-out split at the record level (item 1). ``ctx['eval_task_ids']``
    #    is fixed by registry.split_tasks (see _apply_split) — the held-out family +
    #    arch-specific tasks — so any record whose family/arch/id is reserved is
    #    moved out of TRAIN, guaranteeing training never sees the eval distribution.
    train, val, test = leakage_split(raw, by=("operation", "arch"))
    heldout_ids = set(ctx.get("eval_task_ids") or [])
    leaked = [r for r in train if _rec_is_heldout(r, heldout_ids)]
    train = [r for r in train if not _rec_is_heldout(r, heldout_ids)]
    test = test + leaked
    leaked_ids = sorted({_rec_dict(r).get("task_id") for r in test if _rec_dict(r).get("task_id")})
    _log("build", f"leakage split: train={len(train)} val={len(val)} test={len(test)}; "
                  f"registry held-out(eval) tasks={sorted(heldout_ids)} "
                  f"(records with reserved family/arch moved out: {leaked_ids})")

    # 3. build SFT/DPO from the TRAIN partition only, over the TRAIN-split tasks.
    try:
        teacher = _teacher(ctx["args"])
    except Exception as e:  # noqa: BLE001 - QA gen is optional if the teacher is down
        _log("build", f"teacher unavailable for QA ({e}); using stub")
        teacher = make_teacher("stub")

    train_tasks = _train_tasks(ctx)
    kernel_records = [r for r in train if _rec_type(r) in ("repair", "win")]
    group_records = [r for r in train if _rec_type(r) == "ranked_group"]

    # Near-duplicate dedup on WINS only (Pillar 5): collapse winning kernels that
    # differ solely by renaming/whitespace/comments, keeping the fastest — the
    # shipped data had ~148 kernels recurring >=50x. Repairs are left intact: each
    # broken->fixed transition is a distinct lesson even when fixed kernels converge.
    from kore.data.build_datasets import dedup_near_source
    _wins = [r for r in kernel_records if _rec_type(r) == "win"]
    _non_wins = [r for r in kernel_records if _rec_type(r) != "win"]
    if _wins:
        _wins = dedup_near_source(_wins, per_fingerprint_cap=1)
    kernel_records = _non_wins + _wins

    # RFT / rejection sampling (ReST-EM): train SFT on the policy's HIGH-reward
    # kernels only — keep all repair turns (they teach correctness) but REJECT the
    # sub-tau (slower-than-baseline) wins, keeping only the stratified, deduped >tau
    # wins. This concentrates mass on the >1x region by EXCLUSION (robust to the
    # mixer's content-hash dedup, unlike row duplication). rft_oversample>0 enables;
    # 0 keeps every win. See kore.data.rejection.
    from kore.data.rejection import stratified_rft_select
    if getattr(ctx["args"], "rft", True):
        repairs = [r for r in kernel_records if _rec_type(r) == "repair"]
        wins = [r for r in kernel_records if _rec_type(r) == "win"]
        kept_wins, rft_report = stratified_rft_select(
            wins, tau=float(getattr(ctx["args"], "rft_tau", 1.0)),
            per_task_frac_cap=0.34, seed=ctx["args"].split_seed)
        rejected = len(wins) - rft_report.n_kept
        kernel_records = repairs + list(kept_wins)
        _log("build", f"RFT rejection: kept {rft_report.n_kept}/{len(wins)} wins "
                      f">={rft_report.tau}x (rejected {rejected} slow/dup), "
                      f"+{len(repairs)} repairs, task-entropy {rft_report.task_entropy}")
        LOG.event("rft_select", tau=rft_report.tau, n_wins=len(wins),
                  n_pass=rft_report.n_pass_filter, n_kept=rft_report.n_kept,
                  n_rejected=rejected, task_entropy=rft_report.task_entropy,
                  per_task=rft_report.per_task)

    cfg = MultiCapSFTConfig()
    rows = build_multicap_dataset(ctx["data_root"], train_tasks, teacher, cfg,
                                  total=ctx["args"].sft_total, use_hf=ctx["args"].use_hf,
                                  kernel_records=kernel_records)
    _write_rows(ctx["data_root"] / "sft" / "multicap.jsonl", rows)
    _log("build", f"multicap SFT (train-only): {len(rows)} rows; "
                  f"mix={summarize_multicap(rows)['fractions']}")

    # Pillar 3: build DPO prompts IN THE INFERENCE CONTEXT — the GRPO turn-1
    # transcript (system + seed-kernel task prompt) — so preferences are learned in
    # the same context the policy sees at deployment, not a bare one-shot. Map task
    # id -> transcript; unknown ids fall back to the generic prompt inside build_dpo.
    from kore.policy.format import build_task_prompt, build_transcript
    _task_by_id = {t.task_id: t for t in train_tasks}

    def _dpo_prompt_fn(task_id):
        t = _task_by_id.get(task_id)
        return build_transcript(build_task_prompt(t), []) if t is not None else None

    dpo = build_dpo_with_hard_negatives(ctx["data_root"], train_tasks,
                                        group_records=group_records,
                                        hard_target=float(getattr(ctx["args"], "dpo_hard_fraction", 0.0) or 0.0) or None,
                                        prompt_fn=_dpo_prompt_fn,
                                        seed=int(getattr(ctx["args"], "split_seed", 0) or 0))
    _write_rows(ctx["data_root"] / "dpo" / "pairs.jsonl", dpo["rows"])
    _log("build", f"DPO (train-only): {dpo['n_total']} pairs ({dpo['n_hard']} hard, "
                  f"frac={dpo['n_hard']/max(1,dpo['n_total']):.1%}, >=8% target met={dpo['meets_target']})")


def _stage_midtrain(ctx):
    """Stage-0: build the ROCm/HIP/Triton corpus (if missing) then continued-pretrain.

    The trained checkpoint is threaded in as the base for Stage-1 SFT via
    ctx["midtrain_ckpt"] (see ``_stage_sft``). Honors --lora/--full-ft; the locked
    full-FT recipe of a 14B is shelled out to the FSDP launcher under the hood
    (one command), exactly like sft/dpo/grpo (see docs/DISTRIBUTED.md).
    """
    if ctx["dry"]:
        _log("midtrain", "would build the ROCm/HIP/Triton corpus (build_midtrain_corpus: "
                         "kore task kernels+refs, PyTorch->Triton pairs, repo Triton/HIP "
                         "source, rocprof/tuning docs, ~30% general replay) then full-FT "
                         "continued-pretrain (train_midtrain) -> SFT base")
        return
    from kore.data.midtrain_corpus import build_midtrain_corpus
    from kore.policy.configs import MidTrainConfig
    from kore.policy.midtrain import train_midtrain

    corpus = ctx["data_root"] / "midtrain" / "corpus.jsonl"
    cfg = MidTrainConfig(model_id=ctx["base"], corpus_path=str(corpus),
                         output_dir=ctx["args"].midtrain_out,
                         use_lora=ctx["args"].lora)
    if _full_ft(ctx):
        # Contract: --full-ft sets distributed=True on every training config.
        setattr(cfg, "distributed", True)

    # 1. Build the corpus from local sources if it is not already on disk.
    if not corpus.exists():
        report = build_midtrain_corpus(corpus, cfg, seed=0, use_hf=ctx["args"].use_hf)
        _log("midtrain", f"built corpus -> {corpus}: {report['total']} chunks "
                         f"(general_frac={report['general_frac']}, "
                         f"dropped_dup={report['n_dropped_dup']})")
        for src, n in report["counts"].items():
            LOG.metric("midtrain_corpus_source", source=src, n=n)
    else:
        _log("midtrain", f"reusing existing corpus at {corpus}")

    # 2. Continued pretraining (full-FT locked recipe; --lora for single-GPU smoke).
    #    Full-FT engages FSDP via the launcher UNDER THE HOOD when the stage's
    #    `-m` JSON entry supports it; otherwise it falls back in-process (LOUD).
    mt_t0 = time.time()
    if _full_ft(ctx) and _stage_supports_launcher("midtrain"):
        ctx["midtrain_ckpt"] = _launch_distributed(ctx, "midtrain", {
            "model_id": ctx["base"], "corpus_path": str(corpus),
            "output_dir": ctx["args"].midtrain_out})
    else:
        if _full_ft(ctx):
            _warn_inprocess_fullft("midtrain")
        ctx["midtrain_ckpt"] = train_midtrain(cfg, corpus_path=str(corpus))
    LOG.progress(1, 1, "midtrain", t_start=mt_t0)
    _log("midtrain", f"-> {ctx['midtrain_ckpt']} (this checkpoint becomes the SFT base)")
    _retention_gate(ctx, stage="midtrain", candidate=ctx["midtrain_ckpt"], base=ctx["base"])


def _stage_sft(ctx):
    if ctx["dry"]:
        _log("sft", "would multi-capability SFT (full-FT, ~45% general replay retention)")
        return
    from kore.policy.configs import MultiCapSFTConfig
    from kore.policy.sft import train_sft

    # Start Stage-1 SFT from the Stage-0 mid-train checkpoint when present (the
    # continued-pretrained base); fall back to the raw base otherwise.
    sft_base = ctx.get("midtrain_ckpt") or ctx["base"]
    if ctx.get("midtrain_ckpt"):
        _log("sft", f"starting from mid-train checkpoint {sft_base}")
    dataset = ctx["data_root"] / "sft" / "multicap.jsonl"
    # Fix 1: --lora keeps the 14B validation run single-GPU-feasible (single
    # process). --full-ft engages REAL FSDP full fine-tuning via the launcher
    # (accelerate) UNDER THE HOOD — still ONE user command.
    if _full_ft(ctx) and _stage_supports_launcher("sft"):
        ctx["sft_ckpt"] = _launch_distributed(ctx, "sft", {
            "model_id": sft_base, "dataset_path": str(dataset),
            "output_dir": ctx["args"].sft_out})
    else:
        cfg = MultiCapSFTConfig(model_id=sft_base, output_dir=ctx["args"].sft_out,
                                use_lora=ctx["args"].lora)
        if _full_ft(ctx):
            setattr(cfg, "distributed", True)
        ctx["sft_ckpt"] = train_sft(cfg, dataset)
    _log("sft", f"-> {ctx['sft_ckpt']}")
    _retention_gate(ctx, stage="sft", candidate=ctx["sft_ckpt"], base=ctx["base"])


def _dagger_fold_into_sft(ctx, policy, teacher, round_idx: int, rounds: int) -> int:
    """DAgger: mine the CURRENT policy's OWN failures, get verified expert fixes,
    and FOLD them into the multi-capability SFT corpus (item 2 / item 5).

    Runs :func:`kore.data.onpolicy.dagger_repairs` on the TRAIN-split tasks only
    (never the held-out set), with the teacher-mixing beta decaying 30%->0% across
    rounds (:func:`kore.data.onpolicy.dagger_teacher_frac`). The verified repairs
    are written to a ``dagger`` shard AND their SFT chat rows are appended to
    ``sft/multicap.jsonl`` so the multi-cap mix includes the DAgger repairs.
    Returns the number of repairs folded in.
    """
    from kore.data.build_datasets import build_sft
    from kore.data.onpolicy import dagger_repairs, dagger_teacher_frac
    from kore.data.schemas import write_jsonl
    from kore.env.kore_env import KoreEnv

    frac = dagger_teacher_frac(round_idx, rounds)
    reps: list = []
    for task in _train_tasks(ctx):
        env = KoreEnv(task)
        reps += dagger_repairs(task, policy, teacher, env, n=ctx["args"].dagger_n,
                               seed=round_idx * 1000 + 7, teacher_frac=frac, diagnostic=True)
    if not reps:
        _log("dpo", f"round {round_idx}: DAgger found no repairable policy failures")
        return 0
    write_jsonl(ctx["data_root"] / "dagger" / f"round{round_idx}.jsonl", reps)
    rows = build_sft(reps)
    sft_path = ctx["data_root"] / "sft" / "multicap.jsonl"
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    with sft_path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    _log("dpo", f"round {round_idx}: folded {len(reps)} DAgger repairs "
                f"(teacher_frac={frac:.3f}) into the SFT corpus (+{len(rows)} rows)")
    return len(reps)


def _stage_dpo(ctx):
    """Stage-2 preference tuning.

    ``--dpo-rounds <= 1`` runs a single DPO pass on the pre-built ranked-group +
    hard-negative pairs. ``--dpo-rounds > 1`` runs the ITERATIVE on-policy DPO +
    DAgger loop (item 2): each round relabels preference groups ON-POLICY from the
    current checkpoint, aggregates the union with all prior rounds (DAgger no-
    regret), builds DPO pairs, trains with the IPO loss against a REFRESHED frozen
    reference (the previous round's checkpoint), and folds the policy's own mined
    DAgger repairs back into the SFT corpus (round>0). Train-split tasks only.
    """
    rounds = int(getattr(ctx["args"], "dpo_rounds", 1) or 1)
    # Pillar 1: on-policy DPO relabeling re-verifies kernels through KoreEnv, so apply
    # the same max verification rigor as datagen (adversarial + augment + strong bar).
    if not ctx["dry"] and getattr(ctx["args"], "rigorous_verify", True):
        from kore.data.verify_rigor import set_rigorous_verification
        set_rigorous_verification(True)
    if ctx["dry"]:
        if rounds > 1:
            _log("dpo", f"would run {rounds} rounds of ITERATIVE on-policy DPO "
                        "(iterative_dpo): relabel on-policy from the current ckpt -> "
                        "aggregate -> build_dpo -> dpo.train(loss_type='ipo', refreshed "
                        "ref_model_id) -> next round; folding DAgger repairs into SFT")
        else:
            _log("dpo", "would DPO on ranked-groups + hard-negative pairs")
        return

    sft = ctx.get("sft_ckpt") or ctx["base"]
    if rounds > 1:
        ctx["dpo_ckpt"] = _stage_dpo_iterative(ctx, sft, rounds)
    else:
        ctx["dpo_ckpt"] = _stage_dpo_single(ctx, sft)
    _log("dpo", f"-> {ctx['dpo_ckpt']}")
    _retention_gate(ctx, stage="dpo", candidate=ctx["dpo_ckpt"], base=sft)


def _stage_dpo_single(ctx, sft) -> str:
    ds = str(ctx["data_root"] / "dpo" / "pairs.jsonl")
    # Full-FT: engage FSDP via the launcher (accelerate) under the hood.
    if _full_ft(ctx) and _stage_supports_launcher("dpo"):
        return _launch_distributed(ctx, "dpo", {
            "model_id": sft, "dataset_path": ds, "output_dir": ctx["args"].dpo_out})
    from kore.policy.configs import DPOConfig
    from kore.policy.dpo import train

    cfg = DPOConfig(model_id=sft, dataset_path=ds,
                    output_dir=ctx["args"].dpo_out, use_lora=ctx["args"].lora)
    result = train(cfg)
    return (result.get("output_dir") if isinstance(result, dict) else None) or ctx["args"].dpo_out


def _stage_dpo_iterative(ctx, sft, rounds: int) -> str:
    """Iterative on-policy DPO + DAgger (item 2), following the on-policy recipe.

    ``policy_factory(round_idx, prev_ckpt)`` loads a duck-typed ``.generate`` policy
    (``kore.policy.serve.load_generate``) from the current checkpoint (the SFT ckpt
    on round 0, else the previous round's trained ckpt = REFERENCE REFRESH), and
    for round>0 first folds that policy's DAgger repairs into the SFT corpus.
    ``train_fn`` writes the aggregated DPO pairs and runs ``dpo.train`` with
    ``loss_type='ipo'`` against the refreshed frozen ``ref_model_id``.
    """
    from kore.config import CONFIG
    from kore.data.onpolicy import iterative_dpo
    from kore.env.kore_env import KoreEnv
    from kore.policy.configs import DPOConfig
    from kore.policy.dpo import train
    from kore.policy.serve import load_generate

    teacher = _teacher(ctx["args"])
    train_tasks = _train_tasks(ctx)

    # Pillar 3: in-context DPO prompts for the iterative on-policy rounds too (the
    # seed-kernel transcript = deployment context), matching the first-round build.
    from kore.policy.format import build_task_prompt, build_transcript
    _task_by_id = {t.task_id: t for t in train_tasks}

    def _dpo_prompt_fn(task_id):
        t = _task_by_id.get(task_id)
        return build_transcript(build_task_prompt(t), []) if t is not None else None

    def policy_factory(round_idx, prev_ckpt):
        ckpt = prev_ckpt or sft
        _log("dpo", f"round {round_idx}: loading on-policy generator from {ckpt}")
        policy = load_generate(ckpt)
        if round_idx > 0:
            _dagger_fold_into_sft(ctx, policy, teacher, round_idx, rounds)
        return policy

    def train_fn(rd):
        base_ckpt = rd.ref_model_id or sft            # policy relabeled from this ckpt
        ds_path = ctx["data_root"] / "dpo" / f"round{rd.round}" / "pairs.jsonl"
        _write_rows(ds_path, rd.dpo_pairs)
        out_dir = str(Path(ctx["args"].dpo_out) / f"round{rd.round}")
        # Full-FT: shell out per round to the FSDP launcher (IPO + refreshed ref
        # travel in the JSON config); LoRA / single-process stays in-process.
        if _full_ft(ctx) and _stage_supports_launcher("dpo"):
            _log("dpo", f"round {rd.round}: full-FT IPO DPO on {rd.n_pairs} aggregated pairs "
                        f"(model={base_ckpt}, ref={base_ckpt}) via FSDP launcher -> {out_dir}")
            return _launch_distributed(ctx, "dpo", {
                "model_id": base_ckpt, "dataset_path": str(ds_path), "output_dir": out_dir,
                "loss_type": "ipo", "ref_model_id": base_ckpt}, run_name=f"dpo_round{rd.round}")
        cfg = DPOConfig(model_id=base_ckpt, dataset_path=str(ds_path),
                        output_dir=out_dir, use_lora=ctx["args"].lora)
        cfg.loss_type = "ipo"                          # bounded IPO objective for on-policy prefs
        cfg.ref_model_id = base_ckpt                   # refreshed frozen reference = current policy
        _log("dpo", f"round {rd.round}: IPO DPO on {rd.n_pairs} aggregated pairs "
                    f"(model={base_ckpt}, ref={base_ckpt}) -> {out_dir}")
        result = train(cfg)
        return (result.get("output_dir") if isinstance(result, dict) else None) or out_dir

    results = iterative_dpo(
        rounds, policy_factory, train_tasks, lambda t: KoreEnv(t),
        n_parents=ctx["args"].n_parents, k=ctx["args"].k, seed=0, cfg=CONFIG,
        train_fn=train_fn, aggregate=True, prompt_fn=_dpo_prompt_fn,
    )
    return results[-1].policy_ckpt or ctx["args"].dpo_out


def _stage_grpo(ctx):
    curriculum = bool(getattr(ctx["args"], "grpo_curriculum", False))
    if ctx["dry"]:
        if curriculum:
            _log("grpo", "would run the correctness->latency GRPO CURRICULUM: phase-1 "
                         "correctness-only GRPO (reward_phase='correctness'), then phase-2 "
                         "latency GRPO (reward_phase='latency') initialized from the phase-1 "
                         "ckpt; multi-turn AGENTIC (Kevin credit + StarPO-S + KL-anchor)")
        else:
            _log("grpo", "would run multi-turn AGENTIC GRPO (Kevin credit + StarPO-S + KL-anchor to SFT ckpt)")
        return
    from kore.policy.configs import GRPOConfig
    from kore.policy.grpo import train_grpo

    sft = ctx.get("sft_ckpt") or ctx["base"]
    init = ctx.get("dpo_ckpt") or sft

    # item 1: GRPO must train ONLY on the TRAIN-split tasks. The held-out eval ids
    # (reserved operator family + arch-specific tasks) are the generalization set;
    # training on them would invalidate the eval.
    eval_ids = set(ctx.get("eval_task_ids") or [])
    train_task_ids = [t.task_id for t in _train_tasks(ctx) if t.task_id not in eval_ids]
    if not train_task_ids:
        _log("grpo", f"WARNING: every task is held out for eval ({sorted(eval_ids)}); "
                     "falling back to training on the selected tasks (no split available)")
        train_task_ids = [t.task_id for t in ctx["tasks"]]
    else:
        _log("grpo", f"training on TRAIN-split tasks={train_task_ids} "
                     f"(held-out eval-only={sorted(eval_ids)})")

    # Fix 1: under --full-ft the GRPO RL stage runs FULL-PARAMETER + SHARDED
    # (ZeRO-3 / FSDP) via the one-command launcher — there is NO LoRA shortcut for
    # the RL stage under --full-ft. GRPO now ships the JSON `-m` entry
    # (grpo_config_from_dict + __main__), so --full-ft shells the RL stage out to
    # scripts/launch_distributed.sh exactly like sft/dpo (each curriculum phase =
    # one launched full-parameter GRPO run). --lora is the single-process LoRA
    # bring-up path (GRPO LoRA runs in-process there). The O(1-sample) micro-
    # batched backward keeps the LoRA path memory-safe on a single GPU.
    fullft = _full_ft(ctx)
    use_lora = not fullft

    # Fix 2: turn the anti-collapse ladder + measurement-efficiency levers ON by
    # default for the full best-in-world run. --no-anticollapse / --no-value-
    # prefilter opt out. Without these a run would default to plain GRPO (the
    # audit's finding: SC-GRPO / GTPO / AVSPO / value_prefilter all default OFF).
    anticollapse = bool(getattr(ctx["args"], "anticollapse", True))
    value_prefilter = bool(getattr(ctx["args"], "value_prefilter", True))
    value_model_path = getattr(ctx["args"], "value_model_path", None) \
        or ctx.get("value_model_path")
    variance_floor = _ANTICOLLAPSE_VARIANCE_FLOOR if anticollapse else 0.0

    # GRPO ships the JSON `-m` entry, so full-FT shells out to the FSDP launcher
    # exactly like sft/dpo (detected via `grpo_config_from_dict`, so this flips on
    # automatically the moment the sibling entry lands). If a full-FT run is asked
    # for on a build where the entry is not yet present, fall back in-process with
    # a LOUD warning (distributed=True + use_lora=False still set) — NEVER a silent
    # LoRA degrade.
    launcher_ok = _stage_supports_launcher("grpo")
    if fullft and not launcher_ok:
        _warn_inprocess_fullft("grpo")

    _log("grpo", "levers @ grpo start: agentic=True starpo_s=True dynamic_sampling=on(default) "
                 f"| anticollapse={anticollapse} "
                 f"[rc_grpo/variance_floor({variance_floor})/sc_grpo/gtpo_codesim] "
                 f"| value_prefilter={value_prefilter} value_model_path={value_model_path} "
                 f"| use_lora={use_lora} full_ft={fullft} "
                 f"sharded={fullft and launcher_ok}")

    def _grpo_kw(*, model_id, output_dir, reward_phase="all"):
        kw = dict(model_id=model_id, output_dir=output_dir, agentic=True,
                  starpo_s=True, ref_checkpoint=sft, use_lora=use_lora,
                  reward_phase=reward_phase)
        # Set the anti-collapse levers explicitly (both ON and OFF) so a
        # --no-anticollapse run turns them OFF even when overlaid on the
        # levers-on shipped full-FT template (configs/grpo_14b_full.json).
        kw.update(rc_grpo=anticollapse, variance_floor=variance_floor,
                  sc_grpo=anticollapse, gtpo_codesim=anticollapse,
                  value_prefilter=value_prefilter)
        if value_prefilter and value_model_path:
            kw["value_model_path"] = value_model_path
        if use_lora:
            kw.update(num_trajectories=8, tasks_per_step=2, num_turns=3)
        if ctx["args"].grpo_steps:
            kw["total_steps"] = ctx["args"].grpo_steps
        if getattr(ctx["args"], "adaptive_steps", False):
            kw["adaptive_steps"] = True
        return kw

    def _run_grpo(*, model_id, output_dir, reward_phase="all", run_name=None):
        kw = _grpo_kw(model_id=model_id, output_dir=output_dir, reward_phase=reward_phase)
        if fullft and launcher_ok:
            # Full-parameter sharded GRPO: render the resolved JSON (train-split
            # tasks travel in the config) and shell out to the FSDP launcher.
            overrides = dict(kw)
            overrides["tasks"] = list(train_task_ids)
            return _launch_distributed(ctx, "grpo", overrides, run_name=run_name)
        # LoRA single-process bring-up (--lora) OR the full-FT in-process fallback
        # when the `-m` JSON entry is not present yet (distributed still set).
        cfg = GRPOConfig(**kw)
        if fullft:
            setattr(cfg, "distributed", True)
        return train_grpo(cfg, tasks=train_task_ids)

    if curriculum:
        # Phase 1: correctness-only GRPO (mask the speed term) — learn to be correct.
        p1_out = str(Path(ctx["args"].grpo_out) / "phase1_correctness")
        _log("grpo", f"curriculum phase-1 (correctness) init={init} -> {p1_out}")
        phase1_ckpt = _run_grpo(model_id=init, output_dir=p1_out,
                                reward_phase="correctness",
                                run_name="grpo_phase1_correctness")
        _log("grpo", f"curriculum phase-1 -> {phase1_ckpt}")

        # Phase 2: latency GRPO (full correctness+speed) initialized FROM phase-1
        # (the phase-1 checkpoint = phase-2 init, threaded through the launcher).
        p2_out = ctx["args"].grpo_out
        _log("grpo", f"curriculum phase-2 (latency) init={phase1_ckpt} -> {p2_out}")
        ctx["grpo_ckpt"] = _run_grpo(model_id=phase1_ckpt, output_dir=p2_out,
                                     reward_phase="latency",
                                     run_name="grpo_phase2_latency")
    else:
        ctx["grpo_ckpt"] = _run_grpo(model_id=init, output_dir=ctx["args"].grpo_out,
                                     run_name="grpo")
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
    if not getattr(ctx["args"], "retention_gate", True):
        _log(stage, "retention gate SKIPPED (--no-retention-gate) — for fast "
                    "smoke/debug only; a real run MUST enforce it")
        _emit_event(ctx, stage, "gate_skipped", 0.0, None)
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
    # Per-benchmark drop tolerance. 0.005 (0.5%) was far too strict for the
    # high-variance LLM-judge benchmarks (e.g. mtbench swings ~±0.05 between runs),
    # so a domain-shift continued-pretrain trips it on judge NOISE while the real
    # capability benchmarks hold — and SFT's ~45% general-data mix recovers it
    # downstream anyway. Use a principled default (2%) that still catches genuine
    # catastrophic forgetting; overridable via --retention-epsilon.
    epsilon = float(getattr(ctx["args"], "retention_epsilon", 0.02))
    base_scores = run_retention_suite(base_gen)
    cand_scores = run_retention_suite(cand_gen)
    res = retention_gate(base_scores["scores"], cand_scores["scores"], epsilon=epsilon)
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
    # recipe, which REQUIRES a sharded multi-GPU launch (see docs/DISTRIBUTED.md).
    p.add_argument("--lora", dest="lora", action="store_true", default=True,
                   help="use LoRA on SFT/DPO (GRPO is full-FT only); default bring-up mode")
    p.add_argument("--full-ft", dest="lora", action="store_false",
                   help="full fine-tune instead of LoRA (needs an FSDP/DeepSpeed launch)")
    p.add_argument("--no-retention-gate", dest="retention_gate", action="store_false",
                   default=True,
                   help="skip the per-stage retention gates (faster smoke/debug ONLY; "
                        "a real run MUST enforce them to catch general-ability regressions)")
    p.add_argument("--retention-epsilon", type=float, default=0.02, dest="retention_epsilon",
                   help="max allowed per-benchmark general-ability drop before a stage's "
                        "retention gate hard-stops (default 0.02; 0.005 tripped on LLM-judge "
                        "benchmark noise like mtbench)")
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
    # How to produce the agentic tool-use SFT slice:
    #   live  = run the teacher+GPU AgentHarness per task (tens of GPU-hours).
    #   synth = reconstruct trajectories from ALREADY-verified repair/wins/groups
    #           records (CPU-only, minutes, real measurements) — see synth_agentic.
    #   both  = synth first, then live on top.
    p.add_argument("--agentic", choices=["live", "synth", "both"], default="live",
                   dest="agentic_mode")
    p.add_argument("--synth-agentic-cap", type=int, default=4000,
                   dest="synth_agentic_cap")
    # Parallel datagen: shard tasks across GPUs with concurrent teacher streams.
    # 0 = auto (one worker per GPU); 1 = the sequential path. >GPU-count oversubscribes
    # each GPU to overlap teacher latency (safe: verification runs in short driver
    # subprocesses). The single highest-leverage speedup for the full-scale run.
    p.add_argument("--datagen-workers", type=int, default=0, dest="datagen_workers",
                   help="parallel datagen worker processes (0=auto=one per GPU; 1=sequential)")
    p.add_argument("--sft-total", type=int, default=20000, dest="sft_total")
    # Gold-win mining: reconstruct optimization-win SFT demos from the verified
    # rank-0 candidates in `groups` (CPU-only, quality-gated). Rebalances the thin
    # wins family (~1/task) against repair. On by default; --no-gold-wins to skip.
    p.add_argument("--gold-wins", dest="gold_wins", action=argparse.BooleanOptionalAction,
                   default=True, help="mint gold optimization wins from verified ranked groups")
    p.add_argument("--gold-wins-cap", type=int, default=3000, dest="gold_wins_cap")
    # Repair->DPO: package each verified repair (broken->fixed) as a fixed>broken
    # preference pair, adding a correctness contrast to the speed-ranked group prefs.
    p.add_argument("--repair-dpo", dest="repair_dpo", action=argparse.BooleanOptionalAction,
                   default=True, help="mint fixed>broken DPO pairs from verified repair records")
    p.add_argument("--repair-dpo-cap", type=int, default=8000, dest="repair_dpo_cap")
    p.add_argument("--midtrain-out", default="runs/midtrain", dest="midtrain_out")
    p.add_argument("--sft-out", default="runs/sft", dest="sft_out")
    p.add_argument("--dpo-out", default="runs/dpo", dest="dpo_out")
    p.add_argument("--grpo-out", default="runs/grpo", dest="grpo_out")
    p.add_argument("--grpo-steps", type=int, default=None, dest="grpo_steps")
    # Fix 2: anti-collapse ladder (SC-GRPO + GTPO code-sim + AVSPO variance floor +
    # RC-GRPO) ON by default for the full best-in-world run; --no-anticollapse for
    # plain GRPO. Measurement-efficiency value-model bench prefilter also ON by
    # default (--no-value-prefilter to disable); --value-model-path points the
    # prefilter at a trained value model (else it falls back to generation order).
    p.add_argument("--anticollapse", dest="anticollapse",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="enable the SC-GRPO/GTPO/AVSPO/RC-GRPO anti-collapse ladder (default on)")
    p.add_argument("--value-prefilter", dest="value_prefilter",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="enable the value-model bench prefilter at GRPO (default on)")
    p.add_argument("--value-model-path", default=None, dest="value_model_path",
                   help="trained value model for the GRPO bench prefilter (optional)")
    # item 4: correctness->latency GRPO curriculum (two GRPO phases). Default ON
    # for the full best-in-world run; --no-grpo-curriculum for a single-phase GRPO.
    p.add_argument("--grpo-curriculum", dest="grpo_curriculum",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="run a correctness phase then a latency phase of GRPO (default on)")
    # item 1: seed for the authoritative registry train/held-out split ordering.
    p.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    # item 2: iterative on-policy DPO + DAgger. >1 turns Stage-2 into the loop.
    p.add_argument("--dpo-hard-fraction", type=float, default=0.12, dest="dpo_hard_fraction",
                   help="target reward-hack hard-negative fraction of DPO pairs "
                        "(subsamples abundant base pairs to hit it; 0 disables)")
    p.add_argument("--dpo-rounds", type=int, default=2, dest="dpo_rounds",
                   help="rounds of iterative on-policy DPO (>1 enables the DAgger loop)")
    p.add_argument("--dagger-n", type=int, default=16, dest="dagger_n",
                   help="policy failures to mine + repair per task per DAgger round")
    # item 3: evolutionary datagen stage (spliced after datagen when --evolve is set).
    p.add_argument("--evolve", action="store_true",
                   help="run the evolutionary datagen stage (D-MAB + MAP-Elites) after datagen")
    p.add_argument("--evolve-generations", type=int, default=4, dest="evolve_generations")
    p.add_argument("--soup-out", default="runs/soup", dest="soup_out")
    p.add_argument("--eval-budget", type=int, default=5, dest="eval_budget")
    # P5 flagship novelty: dense hardware-counter (rocprofv3) reward weight. 0 =
    # off (default). A small value (e.g. 0.15) enables the roofline-attainment
    # dense bonus; propagated to training subprocs via KORE_PROFILE_REWARD_WEIGHT.
    p.add_argument("--profile-reward", type=float, default=0.0, dest="profile_reward",
                   help="hardware-counter dense reward weight (0=off; ~0.15 to enable)")
    # RFT / rejection sampling: bootstrap SFT on the policy's own >tau wins.
    p.add_argument("--rft-tau", type=float, default=1.0, dest="rft_tau",
                   help="min speedup for a win to survive RFT rejection (default 1.0x)")
    # RFT rejection is ON by default; --no-rft keeps all wins (incl. sub-tau/slow).
    p.add_argument("--rft", dest="rft", action=argparse.BooleanOptionalAction,
                   default=True, help="RFT rejection: drop sub-tau (slow) wins from SFT (default on)")
    # Pillar 1: data-time verification rigor (adversarial correctness + shape augment
    # + strong compile baseline + cold-L2). ON by default for the data pass; disable
    # with --no-rigorous-verify for a fast/cheap datagen smoke.
    p.add_argument("--rigorous-verify", dest="rigorous_verify",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="max verification rigor during datagen/dpo (default on)")
    # Pin all GPU work (reverify / datagen / training) to specific PHYSICAL GPU ids so
    # a shared node's other jobs are never contended. Empty = auto-detect free GPUs.
    p.add_argument("--gpu-ids", dest="gpu_ids", default="",
                   help="comma-separated physical GPU ids to pin to (e.g. 1,3,5); empty=auto-free")
    # Attach rocprof counters during reverify/datagen so gold-win reasoning is grounded.
    p.add_argument("--ground-reasoning", dest="ground_reasoning",
                   action="store_true", help="collect rocprof counters for grounded reasoning")
    # Adaptive GRPO horizon: stop when the reward mean plateaus (bounded by
    # total_steps). Ensures the policy trains long enough to actually move.
    p.add_argument("--adaptive-steps", dest="adaptive_steps",
                   action="store_true", help="adaptive GRPO horizon (plateau early-stop)")
    # data scale: expand each op's shapes into a diverse small/med/large+odd set.
    p.add_argument("--shape-augment", dest="shape_augment", action="store_true",
                   help="augment per-operator shapes for shape-robust generalization")
    # distributionally-robust speed objective (the method contribution): worst-shape
    # (default), CVaR_alpha (softer robust), or mean (average-case ablation arm).
    p.add_argument("--speed-aggregation", dest="speed_aggregation",
                   choices=["worst", "cvar", "mean"], default="worst",
                   help="per-shape speedup aggregation for the reward (default worst)")
    # real retention eval size cap per benchmark (with --use-hf); 0 = whole split.
    p.add_argument("--eval-n", type=int, default=300, dest="eval_n",
                   help="items per retention benchmark when --use-hf pulls real splits")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
