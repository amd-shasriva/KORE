#!/usr/bin/env python3
"""Multi-policy matched-budget bake-off for the KORE 14B pipeline.

Compares the improvement ladder (seed -> base Qwen3-14B -> midtrain -> SFT -> DPO)
and, optionally, Claude Opus (frontier reference via the AMD gateway) on a task
split, at an EQUAL measurement budget per task (kore.eval.bakeoff). Reports
fast_p (correct AND >=p x baseline), correctness rate, and geomean speedup.

Design for a heavily-shared node:
  * Loads ONE checkpoint at a time and frees VRAM before the next (no stacking).
  * Pins the model AND the kernel-bench subprocess to a single free physical GPU.
  * Saves the combined JSON after EVERY policy, so a mid-run VRAM spike/OOM keeps
    all completed policies' results.

Usage:
  HIP set inside; just pass --gpu <free physical id>.
  python scripts/eval_bakeoff_multi.py --gpu 3 --budget 3 \
      --models seed,base,sft,dpo --claude \
      --tasks flash_attn_decode_bf16,flash_attn_prefill_bf16,paged_attn_decode_bf16 \
      --out runs/full/eval/heldout_attention
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path


MODEL_PATHS = {
    "seed": None,                    # seed_policy — the frozen starter kernel (lower bound)
    "base": "Qwen/Qwen3-14B",        # untrained base model
    "midtrain": "runs/full/midtrain",
    "sft": "runs/full/sft",
    "dpo": "runs/full/dpo",          # DPO v1 (degenerate — kept for comparison)
    "dpo_v2": "runs/full/dpo_v2",    # DPO v2 (RPO-anchored re-train)
    "grpo": "runs/full/grpo",        # if/when present
}
# Any --models entry not in the map is treated as a literal path/HF id, so you can
# also pass e.g. --models seed,sft,runs/full/dpo_v2 directly.


def build_opus_policy(max_tokens: int = 8192, temperature: float = 0.0):
    """Claude Opus as a bake-off PolicyFn (same transcript contract as model_policy)."""
    from kore.data.teacher import ClaudeTeacher, load_env_local
    from kore.eval.policies import _task_id, _task_prompt, _render_feedback
    from kore.policy.format import SYSTEM_PROMPT, build_transcript, parse_response

    load_env_local()
    teacher = ClaudeTeacher(temperature=temperature, max_tokens=max_tokens)
    histories: dict = {}

    def policy(task, feedback=None):
        tid = _task_id(task)
        turns = histories.setdefault(tid, [])
        if feedback is None:
            turns.clear()
        elif turns:
            turns[-1] = {**turns[-1], "feedback": _render_feedback(feedback)}
        messages = build_transcript(_task_prompt(task), turns=turns, system_prompt=SYSTEM_PROMPT)
        out = teacher.generate(messages)
        turns.append({"response": out})
        return parse_response(out).get("kernel") or out

    return policy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="3", help="free PHYSICAL gpu id for model + bench")
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--mode", default="serial", choices=["serial", "parallel"])
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--tasks", default="flash_attn_decode_bf16,flash_attn_prefill_bf16,paged_attn_decode_bf16")
    ap.add_argument("--models", default="seed,base,sft,dpo")
    ap.add_argument("--claude", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="adversarial/metamorphic correctness (KORE_VERIFIED_CORRECTNESS=1); slower")
    ap.add_argument("--fresh", action="store_true",
                    help="disable the replay cache (re-bench identical kernels)")
    ap.add_argument("--out", default="runs/full/eval/bakeoff_multi")
    args = ap.parse_args()

    # Pin the WHOLE process to one physical GPU BEFORE importing torch, so HF
    # device_map="auto" loads the 14B on exactly this GPU (not spread across busy
    # ones). The bench subprocess is pinned to the same physical id via KoreEnv.
    os.environ["HIP_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.strict:
        os.environ["KORE_VERIFIED_CORRECTNESS"] = "1"  # adversarial/metamorphic anti-hack correctness

    import torch  # noqa: E402
    from kore.env.kore_env import KoreEnv  # noqa: E402
    from kore.eval.bakeoff import evaluate_policy  # noqa: E402
    from kore.eval.policies import model_policy, seed_policy  # noqa: E402
    from kore.eval.report import save_report  # noqa: E402
    from kore.tasks.registry import get_task  # noqa: E402

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tasks = [get_task(t) for t in task_ids]
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    # Bench subprocess -> same physical GPU. gpu is the ABSOLUTE physical id; KoreEnv
    # sets HIP_/CUDA_VISIBLE_DEVICES=gpu on the child so it targets that GPU.
    env_factory = lambda t: KoreEnv(t, gpu=args.gpu, use_replay=not args.fresh)  # noqa: E731

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    def _save():
        ranking = sorted(
            (k for k in results if "fast_p" in results[k]),
            key=lambda n: results[n]["fast_p"].get(1.0, 0.0), reverse=True)
        combined = {
            "budget": args.budget, "mode": args.mode, "backend": args.backend,
            "n": len(tasks), "task_ids": task_ids, "gpu": args.gpu,
            "policies": results, "ranking_by_fast1": ranking,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out.with_suffix(".json").write_text(json.dumps(combined, indent=2, default=str))
        return combined

    print(f"[eval] gpu={args.gpu} budget={args.budget} mode={args.mode} backend={args.backend} "
          f"tasks={task_ids} torch_sees={torch.cuda.device_count()} GPU(s)", flush=True)

    def _run(name, policy):
        t0 = time.time()
        try:
            print(f"[eval] === {name}: evaluating {len(tasks)} tasks @ budget {args.budget} ===", flush=True)
            r = evaluate_policy(policy, tasks, env_factory=env_factory,
                                budget=args.budget, mode=args.mode)
            results[name] = r
            print(f"[eval] {name}: fast_1={r['fast_p'].get(1.0):.3f} "
                  f"correct={r['num_correct']}/{len(tasks)} "
                  f"geomean_speedup={r['geometric_mean_speedup']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[eval] {name} FAILED: {e}\n{traceback.format_exc()}", flush=True)
            results[name] = {"error": str(e)}
        _save()

    for name in model_list:
        if name == "seed":
            _run("seed", seed_policy)
            continue
        ckpt = MODEL_PATHS.get(name, name)
        # Skip LOCAL checkpoints that don't exist yet (e.g. dpo_v2 before training);
        # HF hub ids (e.g. "Qwen/Qwen3-14B") are allowed through to download.
        is_local = ckpt.startswith(("runs/", "/", "./", "data/"))
        if is_local and not Path(ckpt).exists():
            print(f"[eval] skip {name}: checkpoint {ckpt} absent", flush=True)
            continue
        pol = None
        try:
            print(f"[eval] loading {name} <- {ckpt} ({args.backend})", flush=True)
            pol = model_policy(ckpt, backend=args.backend, max_tokens=args.max_tokens)
            _run(name, pol)
        except Exception as e:  # noqa: BLE001
            print(f"[eval] {name} load FAILED: {e}\n{traceback.format_exc()}", flush=True)
            results[name] = {"error": f"load: {e}"}
            _save()
        finally:
            del pol
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.claude:
        try:
            print("[eval] loading claude-opus (AMD gateway)", flush=True)
            _run("claude-opus", build_opus_policy(max_tokens=args.max_tokens))
        except Exception as e:  # noqa: BLE001
            print(f"[eval] claude-opus FAILED: {e}\n{traceback.format_exc()}", flush=True)
            results["claude-opus"] = {"error": str(e)}
            _save()

    combined = _save()
    try:
        paths = save_report(combined, out)
        print(f"[eval] report: {paths}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[eval] save_report warn: {e}", flush=True)
    print(f"[eval] DONE -> {out.with_suffix('.json')}", flush=True)
    print(f"[eval] ranking: {combined['ranking_by_fast1']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
