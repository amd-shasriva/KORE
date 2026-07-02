"""KORE end-to-end campaign orchestrator.

Staged curriculum (each stage gated on the previous):
    1. datagen   : teacher (Opus) -> repair + ranked-groups + wins per task
    2. build     : assemble SFT / DPO / RFT chat datasets (dedup + leakage split)
    3. sft       : repair-weighted LoRA cold start
    4. dpo       : preference optimization on ranked groups
    5. grpo      : multi-turn verifiable-reward RL
    6. eval      : matched-budget bake-off (fast_p) vs the AITER baseline

Use --dry-run to validate wiring with no GPU/teacher. Use --stages to run a
subset (e.g. --stages datagen,build,sft).

    PYTHONPATH=. python scripts/run_campaign.py --tasks rmsnorm_aiter,silu_mul_bf16 \
        --teacher claude --model claude-opus-4.8 --stages datagen,build,sft,dpo,grpo,eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

ALL_STAGES = ["datagen", "build", "sft", "dpo", "grpo", "eval"]


def _log(stage: str, msg: str) -> None:
    print(f"[campaign:{stage}] {msg}", flush=True)


def run(args) -> int:
    from kore.tasks.registry import all_tasks, get_task

    tasks = [get_task(t) for t in args.tasks.split(",")] if args.tasks else all_tasks()
    stages = args.stages.split(",") if args.stages else ALL_STAGES
    data_root = Path(args.data_root)
    dry = args.dry_run
    _log("plan", f"tasks={[t.task_id for t in tasks]} stages={stages} dry_run={dry}")

    if "datagen" in stages:
        _run_datagen(args, tasks, data_root, dry)
    if "build" in stages:
        _run_build(data_root, dry)
    if "sft" in stages:
        _run_stage_sft(args, data_root, dry)
    if "dpo" in stages:
        _run_stage_dpo(args, data_root, dry)
    if "grpo" in stages:
        _run_stage_grpo(args, tasks, dry)
    if "eval" in stages:
        _run_eval(tasks, dry)
    _log("done", "campaign complete")
    return 0


def _run_datagen(args, tasks, data_root, dry):
    if dry:
        _log("datagen", "dry-run: would generate repair/groups/wins per task")
        return
    from kore.data.gen_groups import generate_groups
    from kore.data.gen_repair import generate_repairs
    from kore.data.gen_wins import generate_wins
    from kore.data.schemas import write_jsonl
    from kore.data.teacher import load_env_local, make_teacher
    from kore.env.kore_env import KoreEnv

    load_env_local()
    teacher = make_teacher(args.teacher, **({"model": args.model} if args.model else {}))
    for task in tasks:
        env = KoreEnv(task)
        r = generate_repairs(task, teacher, env, n=args.n_repair)
        g = generate_groups(task, teacher, env, n_parents=args.n_parents, k=args.k)
        w = generate_wins(task, teacher, env, gens=args.wins_gens)
        for kind, recs in (("repair", r), ("groups", g), ("wins", w)):
            out = data_root / kind / f"{task.task_id}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(out, recs)
            _log("datagen", f"{task.task_id}:{kind} -> {len(recs)} records")


def _run_build(data_root, dry):
    if dry:
        _log("build", "dry-run: would build sft/dpo/rft datasets")
        return
    import glob
    import json

    from kore.data.build_datasets import build_dpo, build_rft, build_sft
    from kore.data.schemas import read_jsonl

    def _collect(*globs):
        recs = []
        for g in globs:
            for p in glob.glob(g):
                recs += read_jsonl(p, typed=True)
        return recs

    repair = _collect(str(data_root / "repair/*.jsonl"), str(data_root / "wins/*.jsonl"))
    groups = _collect(str(data_root / "groups/*.jsonl"))
    for kind, recs, fn, out in (
        ("sft", repair, build_sft, data_root / "sft/train.jsonl"),
        ("dpo", groups, build_dpo, data_root / "dpo/pairs.jsonl"),
        ("rft", _collect(str(data_root / "wins/*.jsonl")), build_rft, data_root / "rft/train.jsonl"),
    ):
        rows = fn(recs)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        _log("build", f"{kind}: {len(rows)} rows -> {out}")


def _run_stage_sft(args, data_root, dry):
    if dry:
        _log("sft", "dry-run: would LoRA-SFT on data/sft/train.jsonl")
        return
    from kore.policy.configs import SFTConfig
    from kore.policy.sft import train_sft

    cfg = SFTConfig(model_id=args.sft_model, output_dir=args.sft_out)
    _log("sft", f"training {cfg.model_id} -> {train_sft(cfg, data_root / 'sft/train.jsonl')}")


def _run_stage_dpo(args, data_root, dry):
    if dry:
        _log("dpo", "dry-run: would DPO on data/dpo/pairs.jsonl")
        return
    from kore.policy.configs import DPOConfig
    from kore.policy.dpo import train

    cfg = DPOConfig(model_id=args.sft_out, dataset_path=str(data_root / "dpo/pairs.jsonl"),
                    output_dir=args.dpo_out)
    _log("dpo", str(train(cfg)))


def _run_stage_grpo(args, tasks, dry):
    if dry:
        _log("grpo", "dry-run: would run multi-turn GRPO")
        return
    from kore.policy.configs import GRPOConfig
    from kore.policy.grpo import train_grpo

    cfg = GRPOConfig(model_id=args.grpo_model, output_dir=args.grpo_out)
    _log("grpo", str(train_grpo(cfg, tasks=[t.task_id for t in tasks], backend=args.grpo_backend)))


def _run_eval(tasks, dry):
    if dry:
        _log("eval", "dry-run: would run matched-budget fast_p bake-off vs AITER")
        return
    from kore.eval.bakeoff import matched_budget_bakeoff
    from kore.eval.report import format_bakeoff_table
    from kore.env.kore_env import KoreEnv

    env_factory = lambda t: KoreEnv(t)  # noqa: E731

    def seed_policy(task, feedback=None):
        return task.seed_source

    res = matched_budget_bakeoff({"seed": seed_policy}, tasks, budget=5,
                                 env_factory=env_factory, dry_run=None)
    _log("eval", "\n" + format_bakeoff_table(res))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KORE campaign orchestrator")
    p.add_argument("--tasks", default=None)
    p.add_argument("--stages", default=None)
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--data-root", default="data", dest="data_root")
    p.add_argument("--teacher", default="claude")
    p.add_argument("--model", default=None)
    p.add_argument("--n-repair", type=int, default=50, dest="n_repair")
    p.add_argument("--n-parents", type=int, default=20, dest="n_parents")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--wins-gens", type=int, default=8, dest="wins_gens")
    p.add_argument("--sft-model", default="Qwen/Qwen3-14B", dest="sft_model")
    p.add_argument("--sft-out", default="runs/sft", dest="sft_out")
    p.add_argument("--dpo-out", default="runs/dpo", dest="dpo_out")
    p.add_argument("--grpo-model", default="Qwen/Qwen3-14B", dest="grpo_model")
    p.add_argument("--grpo-out", default="runs/grpo", dest="grpo_out")
    p.add_argument("--grpo-backend", default="fallback", dest="grpo_backend")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
