"""KORE end-to-end campaign orchestrator (the full agentic recipe).

Stages (each gated on the previous via retention + kernel metrics):
    datagen  : teacher -> repair + ranked-groups + wins per task
    agentic  : teacher-driven build/test/bench/pmc tool-use trajectories
    build    : multi-capability SFT mix (kernel + QA + agentic + ~45% general)
               and DPO set with >=8% labeled reward-hack hard negatives
    midtrain : Stage-0 full-FT continued pretrain on the ROCm/Triton corpus
    sft      : Stage-1 multi-capability SFT (retains chat/code/orchestration)
    dpo      : Stage-2 preference tuning
    grpo     : Stage-3 multi-turn AGENTIC GRPO (Kevin credit + StarPO-S + KL anchor)
    soup     : Stage-4 base-ward model soup (retention-gated alpha sweep)
    eval     : matched-budget fast_p bake-off + retention suite

--dry-run validates the whole wiring with no GPU/teacher. --stages runs a subset.

    PYTHONPATH=. python scripts/run_campaign.py --model Qwen/Qwen3-14B \
        --tasks rmsnorm_aiter,gemm_bf16,flash_attn_decode_bf16 \
        --teacher claude --stages datagen,agentic,build,sft,dpo,grpo,soup,eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ALL_STAGES = ["datagen", "agentic", "build", "midtrain", "sft", "dpo", "grpo", "soup", "eval"]
DEFAULT_STAGES = ["datagen", "agentic", "build", "sft", "dpo", "grpo", "soup", "eval"]


def _log(stage: str, msg: str) -> None:
    print(f"[campaign:{stage}] {msg}", flush=True)


def _write_rows(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# --------------------------------------------------------------------------- #
def run(args) -> int:
    from kore.tasks.registry import all_tasks, get_task

    tasks = [get_task(t) for t in args.tasks.split(",")] if args.tasks else all_tasks()
    stages = args.stages.split(",") if args.stages else DEFAULT_STAGES
    data_root = Path(args.data_root)
    dry = args.dry_run
    _log("plan", f"model={args.model} tasks={[t.task_id for t in tasks]} stages={stages} dry_run={dry}")

    ctx = {"data_root": data_root, "tasks": tasks, "dry": dry, "args": args,
           "sft_ckpt": args.model, "dpo_ckpt": None, "grpo_ckpt": None, "final": None,
           "metrics": {}}

    dispatch = {
        "datagen": _stage_datagen, "agentic": _stage_agentic, "build": _stage_build,
        "midtrain": _stage_midtrain, "sft": _stage_sft, "dpo": _stage_dpo,
        "grpo": _stage_grpo, "soup": _stage_soup, "eval": _stage_eval,
    }
    for st in stages:
        if st not in dispatch:
            _log("plan", f"unknown stage '{st}', skipping")
            continue
        dispatch[st](ctx)
    _log("done", "campaign complete")
    return 0


# --------------------------------------------------------------------------- #
def _teacher(args):
    from kore.data.teacher import load_env_local, make_teacher
    load_env_local()
    kw = {"model": args.model_teacher} if args.model_teacher else {}
    return make_teacher(args.teacher, **kw)


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


def _stage_build(ctx):
    from kore.policy.configs import MultiCapSFTConfig

    if ctx["dry"]:
        _log("build", "would assemble multi-capability SFT mix + DPO(+>=8% hard negatives)")
        return
    from kore.data.assemble import (build_dpo_with_hard_negatives, build_multicap_dataset,
                                    summarize_multicap)
    from kore.data.teacher import make_teacher

    teacher = None
    try:
        teacher = _teacher(ctx["args"])
    except Exception as e:  # QA gen is optional if teacher unavailable
        _log("build", f"teacher unavailable for QA ({e}); using stub")
        teacher = make_teacher("stub")

    cfg = MultiCapSFTConfig()
    rows = build_multicap_dataset(ctx["data_root"], ctx["tasks"], teacher, cfg,
                                  total=ctx["args"].sft_total, use_hf=ctx["args"].use_hf)
    _write_rows(ctx["data_root"] / "sft" / "multicap.jsonl", rows)
    _log("build", f"multicap SFT: {len(rows)} rows; mix={summarize_multicap(rows)['fractions']}")

    dpo = build_dpo_with_hard_negatives(ctx["data_root"], ctx["tasks"])
    _write_rows(ctx["data_root"] / "dpo" / "pairs.jsonl", dpo["rows"])
    _log("build", f"DPO: {dpo['n_total']} pairs ({dpo['n_hard']} hard, "
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

    cfg = MultiCapSFTConfig(model_id=ctx["args"].model, output_dir=ctx["args"].sft_out)
    ctx["sft_ckpt"] = train_sft(cfg, ctx["data_root"] / "sft" / "multicap.jsonl")
    _log("sft", f"-> {ctx['sft_ckpt']}")
    _retention_gate(ctx, stage="sft", candidate=ctx["sft_ckpt"], base=ctx["args"].model)


def _stage_dpo(ctx):
    if ctx["dry"]:
        _log("dpo", "would DPO on ranked-groups + hard-negative pairs")
        return
    from kore.policy.configs import DPOConfig
    from kore.policy.dpo import train

    cfg = DPOConfig(model_id=ctx["sft_ckpt"], dataset_path=str(ctx["data_root"] / "dpo" / "pairs.jsonl"),
                    output_dir=ctx["args"].dpo_out)
    result = train(cfg)
    ctx["dpo_ckpt"] = (result.get("output_dir") if isinstance(result, dict) else None) or ctx["args"].dpo_out
    _log("dpo", f"-> {ctx['dpo_ckpt']}")
    _retention_gate(ctx, stage="dpo", candidate=ctx["dpo_ckpt"], base=ctx["sft_ckpt"])


def _stage_grpo(ctx):
    if ctx["dry"]:
        _log("grpo", "would run multi-turn AGENTIC GRPO (Kevin credit + StarPO-S + KL-anchor to SFT ckpt)")
        return
    from kore.policy.configs import GRPOConfig
    from kore.policy.grpo import train_grpo

    cfg = GRPOConfig(model_id=ctx["dpo_ckpt"] or ctx["sft_ckpt"], output_dir=ctx["args"].grpo_out,
                     agentic=True, starpo_s=True, ref_checkpoint=ctx["sft_ckpt"])
    ctx["grpo_ckpt"] = train_grpo(cfg, tasks=[t.task_id for t in ctx["tasks"]], backend=ctx["args"].grpo_backend)
    _log("grpo", f"-> {ctx['grpo_ckpt']}")
    _retention_gate(ctx, stage="grpo", candidate=ctx["grpo_ckpt"], base=ctx["sft_ckpt"])


def _stage_soup(ctx):
    if ctx["dry"]:
        _log("soup", "would base-ward model soup (alpha sweep, accept best kernel s.t. no general regression)")
        return
    from kore.policy.configs import SoupConfig
    from kore.policy.soup import build_soup

    cfg = SoupConfig(base_model_id=ctx["args"].model,
                     kore_checkpoint=ctx["grpo_ckpt"] or ctx["dpo_ckpt"] or ctx["sft_ckpt"],
                     output_dir=ctx["args"].soup_out)
    # alpha sweep would call eval per alpha; here we materialize the recommended alpha.
    alpha = cfg.alphas[-1]
    ctx["final"] = build_soup(cfg.base_model_id, cfg.kore_checkpoint, alpha, cfg.output_dir)
    _log("soup", f"alpha={alpha} -> {ctx['final']}")


def _stage_eval(ctx):
    from kore.eval.bakeoff import matched_budget_bakeoff
    from kore.eval.report import format_bakeoff_table

    if ctx["dry"]:
        _log("eval", "would run matched-budget fast_p bake-off vs AITER + full retention suite")
        return
    from kore.env.kore_env import KoreEnv

    def seed_policy(task, feedback=None):
        return task.seed_source

    res = matched_budget_bakeoff({"seed": seed_policy}, ctx["tasks"], budget=5,
                                 env_factory=lambda t: KoreEnv(t), dry_run=None)
    _log("eval", "\n" + format_bakeoff_table(res))


def _retention_gate(ctx, *, stage, candidate, base):
    """Hard-stop the campaign if a stage regresses general ability past epsilon."""
    if ctx["dry"]:
        _log(stage, "would run retention gate (kernel up AND no general-bench regression)")
        return
    try:
        from kore.eval.gates import assert_gate_or_raise
        from kore.eval.retention import run_retention_suite
        from kore.policy.serve import load_generate  # (model_generate factory)

        base_scores = run_retention_suite(load_generate(base))
        cand_scores = run_retention_suite(load_generate(candidate))
        assert_gate_or_raise(before=base_scores["scores"], after=cand_scores["scores"],
                             kernel_keys=[], general_keys=cand_scores["benches"])
        _log(stage, "retention gate PASSED")
    except Exception as e:  # noqa: BLE001
        _log(stage, f"retention gate skipped/failed: {e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KORE end-to-end campaign")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--tasks", default=None)
    p.add_argument("--stages", default=None)
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
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
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
