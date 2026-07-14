"""Re-verify + re-baseline EXISTING kernels with v2 rigor (no teacher, no regen).

The v1 datagen produced ~27.6k GPU-verified kernels, but their speedups were measured
against a WEAK baseline (torch-eager) with the adversarial correctness battery OFF.
This module REUSES those kernels (the expensive teacher output) and only RE-MEASURES
them under v2 rigor — the strong ``torch.compile``/vendor baseline + the adversarial
correctness battery — so v1 data gets HONEST v2 numbers WITHOUT re-running datagen.
It is GPU-bound (``KoreEnv.step``) but teacher-free and fully resumable.

Per record type (reusing the verified gen_groups primitives):
  * groups  — re-evaluate every candidate -> re-rank (rank_candidates) -> re-build
    preferences (build_preferences w/ the noise margin). A candidate that no longer
    compiles / passes adversarial correctness sinks in the ranking, so preferences
    stay honest. Optionally attaches rocprof counters for the rank-0 candidate.
  * wins    — re-evaluate final_source vs the STRONG baseline; DROP the win if it no
    longer beats it (speedup <= 1.0) or fails adversarial correctness (an honest
    "was only fast vs eager" cull). Otherwise update speedup/snr/wall.
  * repair  — re-verify the FIXED kernel under adversarial rigor; DROP if it no longer
    passes (a v1 lucky-pass), else keep (the correctness lesson survives).

Derived shards (``_gold_*`` / ``_repair_*``) are skipped — the build stage re-mints
them from the re-verified groups. Rigor is supplied by the environment
(``verify_rigor.set_rigorous_verification`` sets KORE_VERIFIED_CORRECTNESS /
KORE_COMPILE_BASELINE / KORE_SHAPE_AUGMENT, inherited by the verifier subprocess).

The env is built with ``use_replay=False`` so every eval is MEASURED fresh: the
persistent ``runs/replay_*.jsonl`` cache holds v1 WEAK-baseline observations, and
serving those would silently turn the re-baseline into a no-op.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from kore.obs import get_logger

log = get_logger("data.reverify")

# noise-margin bands for re-built preferences (match gen_groups defaults on the box)
_SPEED_BAND = 0.03
_SNR_BAND_DB = 5.0


# --------------------------------------------------------------------------- #
# Per-record re-verification (env is any object with .step(); reuses gen_groups)
# --------------------------------------------------------------------------- #
def reverify_group(group: dict, task, env, cfg, *, speed_band: float = _SPEED_BAND,
                   snr_band: float = _SNR_BAND_DB, ground: bool = False) -> dict:
    """Re-evaluate a ranked group's candidates under rigor -> re-rank + re-prefer."""
    from kore.data.gen_groups import _evaluate, build_preferences, rank_candidates

    cands = group.get("candidates") or []
    results = [_evaluate(env, task, c.get("source", ""), cfg) for c in cands]
    if not results:
        return group
    order = rank_candidates(results)
    rank_of = {idx: pos for pos, idx in enumerate(order)}
    new_cands = [{"source": r["source"], "wall_us": r["wall_us"],
                  "snr_db": r["snr_db"], "rank": rank_of[i],
                  # Persist the baseline-anchoring fields so build_dpo's
                  # candidate_baseline_speedup keeps working post-reverify. Dropping
                  # them silently degraded DPO preference weighting to among-correct
                  # (audit C3); they mirror gen_groups' persisted candidate dict.
                  "speedup": r.get("speedup"),
                  "baseline_wall_us": r.get("baseline_wall_us"),
                  "correct": r.get("correct")}
                 for i, r in enumerate(results)]
    out = dict(group)
    out["candidates"] = new_cands
    out["preferences"] = build_preferences(results, speed_band, snr_band)
    if ground and order and results[order[0]].get("correct") \
            and hasattr(env, "collect_counters"):
        try:
            out["counters"] = env.collect_counters(results[order[0]]["source"])
        except Exception:  # noqa: BLE001 - profiling advisory
            pass
    return out


def reverify_win(win: dict, task, env, cfg) -> Optional[dict]:
    """Re-baseline a win vs the STRONG baseline; drop if it no longer wins/verifies."""
    from kore.data.gen_groups import _evaluate

    src = win.get("final_source") or ""
    if not src:
        return None
    r = _evaluate(env, task, src, cfg)
    if not r.get("correct"):
        return None  # fails adversarial correctness now (v1 lucky-pass)
    sp = r.get("speedup")
    if not sp or sp <= 1.0:
        return None  # no longer beats the strong baseline -> not a win
    out = dict(win)
    out["speedup"] = round(float(sp), 4)
    out["snr_db"] = r["snr_db"]
    out["final_wall_us"] = r["wall_us"]
    return out


def reverify_repair(repair: dict, task, env, cfg) -> Optional[dict]:
    """Re-verify the fixed kernel under adversarial rigor; drop a v1 lucky-pass."""
    from kore.data.prompts import extract_kernel

    fixed = ""
    for m in reversed(repair.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            fixed = extract_kernel(m.get("content", ""))
            break
    if not fixed:
        return None
    try:
        obs = env.step(fixed, full_validation=True, multi_shape=True)
    except Exception:  # noqa: BLE001
        return None
    if not getattr(obs, "validation_passed", False):
        return None  # no longer passes -> drop (honest)
    out = dict(repair)
    out["child_snr_db"] = obs.snr_db
    return out


# --------------------------------------------------------------------------- #
# Per-shard / per-task
# --------------------------------------------------------------------------- #
def _reverify_shard(path: Path, fn: Callable[[dict], Optional[dict]], *,
                    drop_none: bool, backup: bool) -> tuple[int, int]:
    """Apply ``fn`` to each row of a JSONL shard (drop rows -> None when drop_none).

    Returns ``(n_in, n_kept)``. Writes atomically; keeps a ``.pre_reverify.bak``.
    """
    if not path.is_file():
        return 0, 0
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not rows:
        return 0, 0
    kept: list[dict] = []
    for r in rows:
        nr = fn(r)
        if nr is None:
            if not drop_none:
                kept.append(r)
            continue
        kept.append(nr)
    if backup:
        bak = path.with_suffix(path.suffix + ".pre_reverify.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n")
    tmp.replace(path)
    return len(rows), len(kept)


def reverify_task(data_root, task, env, cfg, *, ground: bool = False,
                  backup: bool = True) -> dict:
    """Re-verify all (real, non-derived) shards for one task. Returns stats."""
    data_root = Path(data_root)
    tid = task.task_id
    g_in, g_keep = _reverify_shard(
        data_root / "groups" / f"{tid}.jsonl",
        lambda r: reverify_group(r, task, env, cfg, ground=ground),
        drop_none=False, backup=backup)
    w_in, w_keep = _reverify_shard(
        data_root / "wins" / f"{tid}.jsonl",
        lambda r: reverify_win(r, task, env, cfg), drop_none=True, backup=backup)
    r_in, r_keep = _reverify_shard(
        data_root / "repair" / f"{tid}.jsonl",
        lambda r: reverify_repair(r, task, env, cfg), drop_none=True, backup=backup)
    stats = {"task": tid, "groups": g_in, "wins_in": w_in, "wins_kept": w_keep,
             "repair_in": r_in, "repair_kept": r_keep}
    log.event("reverify_task", **stats)
    return stats


def _marker(data_root: Path, tid: str) -> Path:
    return Path(data_root) / ".reverified" / f"{tid}.done"


def reverify_done(data_root, tid: str) -> bool:
    return _marker(Path(data_root), tid).exists()


# --------------------------------------------------------------------------- #
# GPU-pinned parallel runner (mirrors parallel_datagen; explicit free-GPU ids)
# --------------------------------------------------------------------------- #
def _worker(payload: dict) -> list[tuple]:
    gpu = str(payload["gpu_id"])
    # Pin BEFORE torch import; HIP-only (see parallel_datagen for the double-remap trap).
    os.environ["HIP_VISIBLE_DEVICES"] = gpu
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if payload.get("rigorous", True):
        from kore.data.verify_rigor import set_rigorous_verification
        set_rigorous_verification(True)
    if payload.get("ground"):
        os.environ.setdefault("KORE_GROUND_REASONING", "1")

    from kore.config import CONFIG
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    data_root = Path(payload["data_root"])
    ground = bool(payload.get("ground"))
    out: list[tuple] = []
    for tid in payload["task_ids"]:
        if reverify_done(data_root, tid):
            print(f"[reverify w{gpu}] {tid} skip (resume)", flush=True)
            out.append((tid, "skip"))
            continue
        try:
            task = get_task(tid)
            # Pinning mirrors parallel_datagen's PROVEN pattern: HIP_VISIBLE_DEVICES is
            # already set on os.environ above, so build KoreEnv WITHOUT a gpu arg and let
            # the verifier subprocess inherit that (absolute physical id) string.
            # use_replay=False is ESSENTIAL: the persistent v1 cache holds WEAK-baseline
            # (torch-eager, no adversarial) numbers. Re-verify must MEASURE fresh under
            # rigor, never serve stale cached obs, or the re-baseline is a no-op.
            env = KoreEnv(task, use_replay=False)
            reverify_task(data_root, task, env, CONFIG, ground=ground)
            m = _marker(data_root, tid)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text("ok\n")
            print(f"[reverify w{gpu}] {tid} done", flush=True)
            out.append((tid, "done"))
        except Exception as e:  # noqa: BLE001 - one task never aborts the shard
            print(f"[reverify w{gpu}] {tid} ERROR {type(e).__name__}: {e}", flush=True)
            out.append((tid, "error"))
    return out


def _queue_worker(gpu, task_q, result_q, data_root, ground, rigorous):
    """Persistent GPU-pinned worker: pull tasks from a shared queue until drained.

    DYNAMIC load balancing (vs static pre-sharding): every worker grabs the NEXT
    task the instant it finishes its current one, so no worker sits idle while a
    few others grind the heavy shards (attention/moe/gemm). All workers stay busy
    until the queue is empty. GPU is pinned ONCE (before torch import) for the
    worker's whole life, so it is safe to reuse the process across many tasks.
    """
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu)
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if rigorous:
        from kore.data.verify_rigor import set_rigorous_verification
        set_rigorous_verification(True)
    if ground:
        os.environ.setdefault("KORE_GROUND_REASONING", "1")

    from kore.config import CONFIG
    from kore.env.kore_env import KoreEnv
    from kore.tasks.registry import get_task

    data_root = Path(data_root)
    while True:
        tid = task_q.get()
        if tid is None:  # sentinel: queue drained for this worker
            break
        if reverify_done(data_root, tid):
            print(f"[reverify w{gpu}] {tid} skip (resume)", flush=True)
            result_q.put((tid, "skip"))
            continue
        try:
            task = get_task(tid)
            env = KoreEnv(task, use_replay=False)
            reverify_task(data_root, task, env, CONFIG, ground=ground)
            m = _marker(data_root, tid)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text("ok\n")
            print(f"[reverify w{gpu}] {tid} done", flush=True)
            result_q.put((tid, "done"))
        except Exception as e:  # noqa: BLE001 - one task never aborts the worker
            print(f"[reverify w{gpu}] {tid} ERROR {type(e).__name__}: {e}", flush=True)
            result_q.put((tid, "error"))
    result_q.put(None)  # worker-finished sentinel


def run_reverify(data_root, task_ids, gpu_ids, *, ground: bool = False,
                 rigorous: bool = True, log_fn=print) -> dict:
    """Re-verify ``task_ids`` across the given ``gpu_ids`` (pinned, resumable).

    ``gpu_ids`` is the list of per-worker device ids (e.g. ``[0..7]`` repeated
    ``workers_per_gpu`` times = one entry per worker). Uses a SHARED task queue so
    every worker stays busy pulling the next task until all are done — no tail
    draining where light-task workers idle while heavy shards finish.
    """
    import multiprocessing as mp

    task_ids = list(task_ids)
    gpu_ids = list(gpu_ids) or [0]
    n_workers = len(gpu_ids)
    log_fn(f"reverify: {len(task_ids)} tasks across {n_workers} workers on GPUs "
           f"{sorted(set(gpu_ids))} (dynamic queue; rigor={rigorous}, ground={ground})")

    ctxmp = mp.get_context("spawn")
    task_q: "mp.Queue" = ctxmp.Queue()
    result_q: "mp.Queue" = ctxmp.Queue()
    for tid in task_ids:
        task_q.put(tid)
    for _ in range(n_workers):
        task_q.put(None)  # one drain-sentinel per worker

    procs = [ctxmp.Process(target=_queue_worker,
                           args=(gpu_ids[w], task_q, result_q, str(data_root), ground, rigorous))
             for w in range(n_workers)]
    for p in procs:
        p.start()

    summary = {"done": 0, "skip": 0, "error": 0, "tasks": len(task_ids)}
    finished = 0
    completed = 0
    while finished < n_workers:
        item = result_q.get()
        if item is None:
            finished += 1
            continue
        _tid, status = item
        summary[status] = summary.get(status, 0) + 1
        completed += 1
        if completed % 10 == 0:
            log_fn(f"reverify progress: {completed}/{len(task_ids)} "
                   f"(done={summary['done']} skip={summary['skip']} error={summary['error']})")
    for p in procs:
        p.join()
    log_fn(f"reverify done: {summary}")
    return summary


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Re-verify/re-baseline existing kernels (no teacher)")
    p.add_argument("data_root")
    p.add_argument("--gpus", default="", help="comma-separated physical GPU ids (e.g. 1,3,5)")
    p.add_argument("--tasks", default="", help="comma-separated task ids (default: all with shards)")
    p.add_argument("--ground", action="store_true", help="attach rocprof counters (KORE_GROUND_REASONING)")
    p.add_argument("--no-rigor", action="store_true", help="disable adversarial/strong-baseline rigor")
    a = p.parse_args(argv)
    data_root = Path(a.data_root)
    if a.tasks:
        task_ids = [t for t in a.tasks.split(",") if t]
    else:
        seen: set[str] = set()
        for sub in ("groups", "wins", "repair"):
            d = data_root / sub
            if d.is_dir():
                for pth in d.glob("*.jsonl"):
                    if not pth.stem.startswith("_"):
                        seen.add(pth.stem)
        task_ids = sorted(seen)
    gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""] or [0]
    summary = run_reverify(data_root, task_ids, gpus, ground=a.ground, rigorous=not a.no_rigor)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
