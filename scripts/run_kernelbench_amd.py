#!/usr/bin/env python
"""Run the real KernelBench Level 1/2 suite through KORE and report ``fast_p``.

Four phases, deliberately separated the same way :mod:`kore.eval.checkpoint_ab`
separates its A/B, because they have incompatible resource profiles:

  ``materialize``  a KernelBench checkout -> runnable KORE task directories
                   (:func:`kore.eval.kernelbench_tasks.materialize`). Touches the
                   GPU only to probe each problem's ABI.
  ``selftest``     submit HAND-WRITTEN Triton kernels for a few problems. This is
                   the POSITIVE CONTROL for the measurement path: without it a
                   ``fast_1`` of 0 is unfalsifiable, because "the model wrote no
                   fast kernel" and "the harness cannot admit one" look identical.
  ``generate``     one model load, batched decode, one completion per problem ->
                   a generations JSONL. Nothing is measured while a 14B model is
                   resident.
  ``measure``      replay those completions through the real
                   :class:`~kore.env.kore_env.KoreEnv` (verified correctness +
                   cold-cache paired timing) and render the field-standard
                   KernelBench ``fast_p`` report.

BASELINE. Every number this script produces is against TORCH-EAGER - the
KernelBench baseline, which is the materialized task's own
``comparison_baseline``. It is NOT the AITER/hipBLASLt production baseline the
rest of KORE grades against, and the two are not comparable. The report says so
in ``baseline``/``baseline_kind``; do not restate a number from here as a
vendor-relative one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Hand-written Triton kernels: the measurement-path positive control.
#
# Three KernelBench Level-1 problems whose correct kernel is short enough to
# author by hand and audit by eye. Two are pure elementwise (a correct kernel can
# at best TIE torch, which is already at the bandwidth roofline - the point is to
# prove the oracle admits a correct kernel and the paired bench measures a real
# ratio), and one is the MinGPT GELU chain, which torch-eager runs as several
# separate kernels and where fusing therefore has genuine headroom.
# --------------------------------------------------------------------------- #
_ELEMENTWISE_TEMPLATE = '''import torch
import triton
import triton.language as tl


@triton.jit
def _kb_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = {expr}
    tl.store(y_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


def kb_forward(x):
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    _kb_kernel[grid](x, y, n, BLOCK=BLOCK, num_warps=8)
    return y
'''

SELFTEST_KERNELS = {
    "kb_l1_19_relu": _ELEMENTWISE_TEMPLATE.format(expr="tl.maximum(x, 0.0)"),
    "kb_l1_21_sigmoid": _ELEMENTWISE_TEMPLATE.format(expr="tl.sigmoid(x)"),
    # 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))), tanh via the sigmoid
    # identity so the kernel needs no libdevice shim.
    "kb_l1_88_mingptnewgelu": _ELEMENTWISE_TEMPLATE.format(
        expr="0.5 * x * (1.0 + (2.0 * tl.sigmoid("
             "2.0 * (0.7978845608028654 * (x + 0.044715 * x * x * x))) - 1.0))"),
}


# --------------------------------------------------------------------------- #
# phase 1: materialize
# --------------------------------------------------------------------------- #
def cmd_materialize(args) -> int:
    from kore.eval.kernelbench_tasks import materialize

    levels = [int(v) for v in str(args.levels).split(",") if v.strip()]
    only = [v for v in str(args.only).split(",") if v.strip()] if args.only else None
    manifest = materialize(
        args.kernelbench_root, args.out, levels=levels, gpu_target=args.gpu_target,
        device=args.device, only=only, limit=args.limit, log=_log)
    _log(f"[materialize] {manifest['n_tasks']} tasks, {manifest['n_skipped']} skipped "
         f"-> {Path(args.out) / 'manifest.json'}")
    for record in manifest["skipped"]:
        _log(f"[materialize] skipped {record['name']}: {record['reason']}")
    return 0


# --------------------------------------------------------------------------- #
# phase 2: generate (model) / selftest (hand-written)
# --------------------------------------------------------------------------- #
def _generation_rows(arm: str, sources: dict, tasks) -> list:
    from kore.eval.checkpoint_ab import GenerationRecord, first_turn_messages, prompt_digest

    rows = []
    for task in tasks:
        source = sources.get(task.task_id)
        if source is None:
            continue
        rows.append(GenerationRecord(
            arm=arm, task_id=task.task_id, sample=0,
            prompt_sha=prompt_digest(first_turn_messages(task)),
            response=source, kernel=source, contract_ok=True,
            response_chars=len(source), gen_seconds=0.0, max_tokens=0, temperature=0.0))
    return rows


def cmd_selftest(args) -> int:
    from kore.eval.checkpoint_ab import write_generations
    from kore.eval.kernelbench_tasks import load_tasks

    tasks = [t for t in load_tasks(args.tasks_dir) if t.task_id in SELFTEST_KERNELS]
    if not tasks:
        _log("[selftest] none of the control problems were materialized; "
             f"expected any of {sorted(SELFTEST_KERNELS)}")
        return 2
    rows = _generation_rows("selftest", SELFTEST_KERNELS, tasks)
    path = write_generations(args.out, rows, arm="selftest",
                             source="hand-written Triton control kernels",
                             task_ids=[t.task_id for t in tasks])
    _log(f"[selftest] wrote {len(rows)} control kernels -> {path}")
    return 0


def _threaded_batch(generate, concurrency: int):
    """Turn a one-at-a-time ``generate`` into ``generate_batch`` over a thread pool.

    A network-backed teacher spends its time waiting, and AMD's gateway allows
    4000 requests/minute, so issuing 200 prompts serially would make the arm take
    hours for no reason. Results are returned in request order, and a failure is
    returned as an empty completion so :func:`generate_arm` records it per task
    instead of losing the batch.
    """
    from concurrent.futures import ThreadPoolExecutor

    def generate_batch(messages_batch, max_tokens: int = 8192,
                       temperature: float = 0.0) -> list:
        def one(messages):
            try:
                return generate(messages, max_tokens=max_tokens, temperature=temperature)
            except TypeError:
                return generate(messages)
            except Exception as exc:  # noqa: BLE001 - one bad call must not lose the batch
                _log(f"[teacher] call failed: {type(exc).__name__}: {str(exc)[:200]}")
                return ""
        with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
            return list(pool.map(one, list(messages_batch)))

    return generate_batch


def cmd_generate_opus(args) -> int:
    """Generate the frontier-teacher arm through :mod:`kore.eval.vs_opus`.

    The teacher is built by :func:`kore.eval.vs_opus.make_opus_teacher`, so the
    Opus side of this benchmark and the Opus side of the head-to-head are the
    same client against the same gateway. The prompt is built by
    :func:`kore.eval.checkpoint_ab.first_turn_messages`, which is
    ``build_transcript(task_prompt(task))`` - byte-identical to what
    :func:`kore.eval.policies.model_policy` sends on its first turn, and
    therefore identical to what the model arm was given.
    """
    from kore.eval.checkpoint_ab import generate_arm, write_generations
    from kore.eval.kernelbench_tasks import load_tasks
    from kore.eval.vs_opus import make_opus_teacher

    levels = [int(v) for v in str(args.levels).split(",") if v.strip()] if args.levels else None
    tasks = load_tasks(args.tasks_dir, levels=levels)
    if args.limit:
        tasks = tasks[:int(args.limit)]
    if not tasks:
        _log("[opus] no tasks resolved")
        return 2

    teacher = make_opus_teacher(args.teacher, model=args.teacher_model,
                                temperature=float(args.temperature))
    if teacher is None:
        _log("[opus] teacher/gateway unavailable; arm SKIPPED (no artifact written)")
        return 3
    _log(f"[opus] {len(tasks)} tasks, model={getattr(teacher, 'model', args.teacher_model)} "
         f"concurrency={args.concurrency}")

    def generate(messages, **_kw):
        return teacher.generate(messages)

    records = generate_arm(
        tasks, arm=args.arm, generate_batch=_threaded_batch(generate, args.concurrency),
        batch_size=int(args.concurrency), max_tokens=int(args.max_tokens),
        temperature=float(args.temperature), log=_log)
    path = write_generations(
        args.out, records, arm=args.arm,
        model=getattr(teacher, "model", args.teacher_model or args.teacher),
        teacher_kind=args.teacher, max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        contract_ok=sum(1 for r in records if r.contract_ok))
    _log(f"[opus] {len(records)} completions "
         f"({sum(1 for r in records if r.contract_ok)} honored the contract) -> {path}")
    return 0


def cmd_headtohead(args) -> int:
    """Paired win-rate between two measured arms, via :mod:`kore.eval.vs_opus`.

    Uses :func:`kore.eval.vs_opus.head_to_head_winrate` unchanged, so a task is
    won only on a CORRECT kernel and only on the timing-integrity-gated speedup
    that ``fast_p`` itself ranks on. Also re-checks that both arms were prompted
    with byte-identical transcripts, which is the precondition that makes the
    comparison about the models rather than about the harness.
    """
    from kore.eval.checkpoint_ab import assert_prompts_matched, read_generations
    from kore.eval.vs_opus import head_to_head_winrate

    left_meta, left_rows = read_generations(args.a_generations)
    right_meta, right_rows = read_generations(args.b_generations)
    # Raises if the two arms did not see byte-identical prompts, which would make
    # any comparison a statement about the harness rather than the models.
    prompts = assert_prompts_matched({left_meta.get("arm") or "a": left_rows,
                                      right_meta.get("arm") or "b": right_rows})
    shared = set.intersection(*(set(v) for v in prompts.values()))

    left = json.loads(Path(args.a_measures).read_text())["eval"]
    right = json.loads(Path(args.b_measures).read_text())["eval"]
    winrate = head_to_head_winrate(left["per_task"], right["per_task"],
                                   margin=float(args.margin))
    payload = {
        "a": {"arm": left.get("arm"), "model": left_meta.get("model")},
        "b": {"arm": right.get("arm"), "model": right_meta.get("model")},
        "baseline": "torch_eager",
        "prompt_parity": {"verified": True, "n_tasks_compared": len(shared)},
        "winrate": winrate,
    }
    out = Path(args.out).with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    counts = winrate["counts"]
    _log(f"[h2h] {payload['a']['arm']} vs {payload['b']['arm']} over n={winrate['n']}: "
         f"a={counts['kore']} b={counts['opus']} tie={counts['tie']} "
         f"neither={counts['neither']} -> {out}")
    return 0


def cmd_generate(args) -> int:
    from kore.eval.checkpoint_ab import generate_arm, load_hf_batch_generate, write_generations
    from kore.eval.kernelbench_tasks import load_tasks

    levels = [int(v) for v in str(args.levels).split(",") if v.strip()] if args.levels else None
    tasks = load_tasks(args.tasks_dir, levels=levels)
    if args.limit:
        tasks = tasks[:int(args.limit)]
    if not tasks:
        _log("[generate] no tasks resolved")
        return 2
    _log(f"[generate] {len(tasks)} tasks, model={args.model} rev={args.revision}")

    started = time.perf_counter()
    backend = load_hf_batch_generate(args.model, dtype=args.dtype, revision=args.revision)
    _log(f"[generate] loaded in {time.perf_counter() - started:.0f}s: {backend['info']}")

    records = generate_arm(
        tasks, arm=args.arm, generate_batch=backend["generate_batch"],
        batch_size=int(args.batch_size), max_tokens=int(args.max_tokens),
        temperature=float(args.temperature), log=_log)
    backend["close"]()

    path = write_generations(
        args.out, records, arm=args.arm, model=args.model, revision=args.revision,
        backend_info=backend["info"], max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        contract_ok=sum(1 for r in records if r.contract_ok))
    _log(f"[generate] {len(records)} completions "
         f"({sum(1 for r in records if r.contract_ok)} honored the contract) -> {path}")
    return 0


# --------------------------------------------------------------------------- #
# phase 3: measure + report
# --------------------------------------------------------------------------- #
def _funnel(result: dict, rows: list) -> dict:
    """Where every task in the denominator ended up.

    ``fast_p`` collapses a whole pipeline into one fraction, so a bare number
    cannot distinguish "the model wrote slow kernels" from "the model wrote prose".
    These counts are disjoint over the same denominator and reconstruct it.
    """
    by_task = {row["task_id"]: row for row in rows}
    observations = {o["task_id"]: o for o in result.get("observations", [])}
    buckets = {"no_contract": 0, "hacked": 0, "compile_failed": 0, "infra_error": 0,
               "incorrect": 0, "correct_untimed": 0, "correct_slower": 0,
               "correct_faster": 0}
    for record in result.get("per_task", []):
        task_id = record["task_id"]
        obs = observations.get(task_id, {})
        row = by_task.get(task_id, {})
        speedup = record.get("best_speedup")
        if obs.get("flagged_hack"):
            buckets["hacked"] += 1
        elif not row.get("contract_ok", True):
            buckets["no_contract"] += 1
        elif obs.get("infra_error"):
            buckets["infra_error"] += 1
        elif not obs.get("compiled", False):
            buckets["compile_failed"] += 1
        elif not obs.get("validation_passed", False):
            buckets["incorrect"] += 1
        elif speedup is None:
            buckets["correct_untimed"] += 1
        elif speedup > 1.0:
            buckets["correct_faster"] += 1
        else:
            buckets["correct_slower"] += 1
    return buckets


def cmd_measure(args) -> int:
    from kore.env.kore_env import KoreEnv
    from kore.eval.checkpoint_ab import measure_arm, read_generations
    from kore.eval.fastp import DEFAULT_PS
    from kore.eval.kernelbench_amd import (
        KERNELBENCH_PS, format_kernelbench_report, to_kernelbench_report,
    )
    from kore.eval.kernelbench_tasks import load_tasks, read_manifest, specs_for_report

    meta, rows = read_generations(args.generations)
    scored = {row["task_id"] for row in rows}
    tasks = [t for t in load_tasks(args.tasks_dir) if t.task_id in scored]
    if not tasks:
        _log("[measure] no materialized task matches the generations file")
        return 2
    _log(f"[measure] {len(tasks)} tasks, arm={meta.get('arm')} budget={args.budget}")

    def env_factory(task):
        return KoreEnv(task, gpu=args.gpu, use_replay=not args.no_replay,
                       correctness_timeout=int(args.correctness_timeout),
                       bench_timeout=int(args.bench_timeout))

    grid = sorted(set(DEFAULT_PS) | set(KERNELBENCH_PS))
    started = time.perf_counter()
    result = measure_arm(rows, tasks, arm=meta.get("arm") or "arm",
                         env_factory=env_factory, budget=int(args.budget),
                         mode=args.mode, ps=grid)
    elapsed = time.perf_counter() - started

    specs = specs_for_report(args.tasks_dir, tasks)
    report = to_kernelbench_report(result, specs, source="full")
    manifest = read_manifest(args.tasks_dir)
    report["provenance"] = {
        "kernelbench_revision": manifest.get("revision"),
        "kernelbench_root": manifest.get("kernelbench_root"),
        "n_materialized": manifest.get("n_tasks"),
        "n_skipped_materialization": manifest.get("n_skipped"),
        "arm": meta.get("arm"),
        "model": meta.get("model"),
        "model_revision": meta.get("revision"),
        "max_tokens": meta.get("max_tokens"),
        "temperature": meta.get("temperature"),
        "budget": int(args.budget),
        "mode": args.mode,
        "gpu_target": manifest.get("gpu_target"),
        "measure_seconds": round(elapsed, 1),
        "verified_correctness": os.environ.get("KORE_VERIFIED_CORRECTNESS"),
        "bench_cold": os.environ.get("KORE_BENCH_COLD"),
    }
    report["funnel"] = _funnel(result, rows)

    stem = Path(args.out).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(
        {"report": report, "eval": result}, indent=2, default=str))
    text = format_kernelbench_report(report)
    stem.with_suffix(".md").write_text(text)
    print()
    print(text)
    print()
    print("funnel: " + json.dumps(report["funnel"]))
    _log(f"[measure] {elapsed:.0f}s -> {stem.with_suffix('.json')}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("materialize", help="KernelBench checkout -> KORE task dirs")
    m.add_argument("--kernelbench-root", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--levels", default="1,2")
    m.add_argument("--gpu-target", default="gfx950")
    m.add_argument("--device", default="cuda")
    m.add_argument("--only", default=None, help="comma-separated problem stems")
    m.add_argument("--limit", type=int, default=None)
    m.set_defaults(func=cmd_materialize)

    s = sub.add_parser("selftest", help="emit hand-written control kernels")
    s.add_argument("--tasks-dir", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_selftest)

    g = sub.add_parser("generate", help="one model load -> one kernel per problem")
    g.add_argument("--tasks-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--revision", default=None)
    g.add_argument("--arm", default="model")
    g.add_argument("--dtype", default="bfloat16")
    g.add_argument("--levels", default=None)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--max-tokens", type=int, default=4096)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--temperature", type=float, default=0.0)
    g.set_defaults(func=cmd_generate)

    o = sub.add_parser("generate-opus", help="frontier-teacher arm through the AMD gateway")
    o.add_argument("--tasks-dir", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--arm", default="opus")
    o.add_argument("--teacher", default="claude")
    o.add_argument("--teacher-model", default=None)
    o.add_argument("--levels", default=None)
    o.add_argument("--limit", type=int, default=None)
    o.add_argument("--max-tokens", type=int, default=8192)
    o.add_argument("--temperature", type=float, default=0.0)
    o.add_argument("--concurrency", type=int, default=16)
    o.set_defaults(func=cmd_generate_opus)

    h = sub.add_parser("headtohead", help="paired win-rate between two measured arms")
    h.add_argument("--a-generations", required=True)
    h.add_argument("--a-measures", required=True)
    h.add_argument("--b-generations", required=True)
    h.add_argument("--b-measures", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--margin", type=float, default=1.0)
    h.set_defaults(func=cmd_headtohead)

    v = sub.add_parser("measure", help="verify + cold-cache bench + fast_p report")
    v.add_argument("--tasks-dir", required=True)
    v.add_argument("--generations", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--budget", type=int, default=1)
    v.add_argument("--mode", default="parallel", choices=("serial", "parallel"))
    v.add_argument("--gpu", default=None)
    v.add_argument("--no-replay", action="store_true")
    v.add_argument("--correctness-timeout", type=int, default=600)
    v.add_argument("--bench-timeout", type=int, default=900)
    v.set_defaults(func=cmd_measure)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
