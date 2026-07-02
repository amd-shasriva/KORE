"""KORE command-line interface.

    kore tasks                       # list registered tasks
    kore datagen repair  --task T --n 50 --teacher claude --out data/repair/T.jsonl
    kore datagen groups  --task T --n-parents 20 --k 6 --teacher claude
    kore datagen wins    --task T --gens 8 --teacher claude
    kore build-datasets  --kind sft --in data/repair/*.jsonl --out data/sft/train.jsonl
    kore sft   --data data/sft/train.jsonl --model Qwen/Qwen3-14B --out runs/sft
    kore dpo   --data data/dpo/pairs.jsonl --out runs/dpo
    kore grpo  --tasks T1,T2 --backend fallback --out runs/grpo
    kore value-train --table data/value/table.jsonl --out runs/value/model.json
    kore eval  --tasks T1,T2 --budget 5           # matched-budget bake-off

Heavy deps (torch/trl/vllm) are imported lazily inside the subcommand handlers so
listing tasks / building datasets stays dependency-light.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path


def _expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in patterns:
        out += [Path(x) for x in glob.glob(p)]
    return out


def cmd_tasks(_args) -> int:
    from kore.tasks.registry import all_tasks

    for t in all_tasks():
        print(f"{t.task_id:24s} op={t.operation:16s} dtype={t.dtype:6s} "
              f"backend={t.backend:7s} baseline={t.comparison_baseline} shapes={len(t.shapes)}")
    return 0


def _teacher(args):
    from kore.data.teacher import load_env_local, make_teacher

    load_env_local()
    kw = {}
    if args.teacher in ("claude", "opus", "anthropic") and args.model:
        kw["model"] = args.model
    if args.teacher == "vllm":
        kw["model"] = args.model or "Qwen/Qwen3-32B"
        if args.base_url:
            kw["base_url"] = args.base_url
    return make_teacher(args.teacher, **kw)


def cmd_datagen(args) -> int:
    from kore.data.schemas import write_jsonl
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    task = get_task(args.task)
    teacher = _teacher(args)
    env = KoreEnv(task)

    if args.what == "repair":
        from kore.data.gen_repair import generate_repairs
        recs = generate_repairs(task, teacher, env, n=args.n, seed=args.seed)
    elif args.what == "groups":
        from kore.data.gen_groups import generate_groups
        recs = generate_groups(task, teacher, env, n_parents=args.n_parents, k=args.k, seed=args.seed)
    elif args.what == "wins":
        from kore.data.gen_wins import generate_wins
        recs = generate_wins(task, teacher, env, gens=args.gens)
    else:
        print(f"unknown datagen target: {args.what}", file=sys.stderr)
        return 2

    out = Path(args.out or f"data/{args.what}/{args.task}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, recs)
    print(f"[datagen:{args.what}] {len(recs)} records -> {out}")
    return 0


def cmd_build_datasets(args) -> int:
    from kore.data.build_datasets import build_dpo, build_rft, build_sft
    from kore.data.schemas import read_jsonl
    import json

    records = []
    for p in _expand(args.inputs):
        records += read_jsonl(p, typed=True)
    fn = {"sft": build_sft, "dpo": build_dpo, "rft": build_rft}[args.kind]
    rows = fn(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"[build:{args.kind}] {len(rows)} chat rows -> {out}")
    return 0


def cmd_sft(args) -> int:
    from kore.policy.configs import SFTConfig
    from kore.policy.sft import train_sft

    cfg = SFTConfig(model_id=args.model, output_dir=args.out)
    if args.max_steps:
        cfg.num_train_epochs = 1
    out = train_sft(cfg, Path(args.data))
    print(f"[sft] adapter -> {out}")
    return 0


def cmd_dpo(args) -> int:
    from kore.policy.configs import DPOConfig
    from kore.policy.dpo import train

    cfg = DPOConfig(model_id=args.model, dataset_path=args.data, output_dir=args.out)
    print(train(cfg))
    return 0


def cmd_grpo(args) -> int:
    from kore.policy.configs import GRPOConfig
    from kore.policy.grpo import train_grpo

    cfg = GRPOConfig(model_id=args.model, output_dir=args.out)
    if args.steps:
        cfg.total_steps = args.steps
    tasks = args.tasks.split(",") if args.tasks else None
    print(train_grpo(cfg, tasks=tasks, backend=args.backend))
    return 0


def cmd_value_train(args) -> int:
    from kore.value.train_value import train_from_table

    print(train_from_table(args.table, args.out, seed=args.seed))
    return 0


def cmd_eval(args) -> int:
    from kore.eval.bakeoff import matched_budget_bakeoff
    from kore.eval.policies import model_policy, seed_policy
    from kore.eval.report import format_bakeoff_table
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import all_tasks, get_task

    tasks = [get_task(t) for t in args.tasks.split(",")] if args.tasks else all_tasks()

    # Always score the seed baseline; when a checkpoint is given, ALSO score the
    # trained model (so the bake-off reflects what training produced, not the seed).
    policies = {"seed": seed_policy}
    if args.checkpoint:
        policies["kore"] = model_policy(args.checkpoint, backend=args.backend)

    res = matched_budget_bakeoff(
        policies, tasks, budget=args.budget,
        env_factory=(lambda t: KoreEnv(t)) if not args.dry_run else None,
        dry_run=None,
    )
    print(format_bakeoff_table(res))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kore", description="KORE kernel-RL CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tasks").set_defaults(func=cmd_tasks)

    dg = sub.add_parser("datagen")
    dg.add_argument("what", choices=["repair", "groups", "wins"])
    dg.add_argument("--task", required=True)
    dg.add_argument("--n", type=int, default=50)
    dg.add_argument("--n-parents", type=int, default=20, dest="n_parents")
    dg.add_argument("--k", type=int, default=6)
    dg.add_argument("--gens", type=int, default=8)
    dg.add_argument("--seed", type=int, default=0)
    dg.add_argument("--teacher", default="stub")
    dg.add_argument("--model", default=None)
    dg.add_argument("--base-url", default=None, dest="base_url")
    dg.add_argument("--out", default=None)
    dg.set_defaults(func=cmd_datagen)

    bd = sub.add_parser("build-datasets")
    bd.add_argument("--kind", choices=["sft", "dpo", "rft"], required=True)
    bd.add_argument("--in", nargs="+", dest="inputs", required=True)
    bd.add_argument("--out", required=True)
    bd.set_defaults(func=cmd_build_datasets)

    s = sub.add_parser("sft")
    s.add_argument("--data", required=True)
    s.add_argument("--model", default="Qwen/Qwen3-14B")
    s.add_argument("--out", default="runs/sft")
    s.add_argument("--max-steps", type=int, default=0, dest="max_steps")
    s.set_defaults(func=cmd_sft)

    d = sub.add_parser("dpo")
    d.add_argument("--data", required=True)
    d.add_argument("--model", default="Qwen/Qwen3-14B")
    d.add_argument("--out", default="runs/dpo")
    d.set_defaults(func=cmd_dpo)

    g = sub.add_parser("grpo")
    g.add_argument("--tasks", default=None)
    g.add_argument("--model", default="Qwen/Qwen3-32B")
    g.add_argument("--out", default="runs/grpo")
    g.add_argument("--steps", type=int, default=0)
    g.add_argument("--backend", default="inprocess", choices=["inprocess", "fallback"])
    g.set_defaults(func=cmd_grpo)

    v = sub.add_parser("value-train")
    v.add_argument("--table", required=True)
    v.add_argument("--out", default="runs/value/model.json")
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(func=cmd_value_train)

    e = sub.add_parser("eval")
    e.add_argument("--tasks", default=None)
    e.add_argument("--budget", type=int, default=5)
    e.add_argument("--checkpoint", default=None,
                   help="trained checkpoint to score as the 'kore' policy (else seed-only)")
    e.add_argument("--backend", default="hf")
    e.add_argument("--dry-run", action="store_true", dest="dry_run")
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
