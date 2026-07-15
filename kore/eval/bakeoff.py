"""Matched-measurement-budget bake-off (KORE.pdf Sec 4.7).

A policy is evaluated the way a practitioner actually spends GPU time: under a
fixed measurement budget (max N benches per task). We compare policies at an
*equal* budget so the winner is the one that best converts benches into correct
speedups, not the one that simply benched more.

Levers this module exposes:
  - ``evaluate_policy``       : run one policy over a split under a matched budget.
  - ``matched_budget_bakeoff``: compare several policies at the same budget.
  - ``serial_vs_parallel``    : Kevin's finding (serial refinement > parallel
                                sampling) at equal total budget.
  - ``benches_to_best``       : value-model lever (how quickly ranking by a
                                value model surfaces the truly-best candidate).

Real runs go through ``KoreEnv`` + ``compute_reward``. A ``dry_run`` path
accepts precomputed ``Observation`` objects so the whole module is testable on
CPU with no GPUs.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from kore.config import CONFIG, KoreConfig
from kore.reward.reward import Observation, compute_reward
from kore.eval.fastp import (
    DEFAULT_PS,
    fast_p_curve,
    geometric_mean_speedup,
    mean_ci,
    pass_at_k,
)

# A policy maps (task, feedback) -> kernel source. ``feedback`` is None on the
# first turn (or in parallel mode) and otherwise carries the previous turn's
# observation/reward summary.
PolicyFn = Callable[[object, Optional[dict]], str]

# A measurement source maps (task, kernel_source, turn) -> Observation.
MeasureFn = Callable[[object, str, int], Observation]


def _task_id(task) -> str:
    if isinstance(task, str):
        return task
    tid = getattr(task, "task_id", None)
    return tid if tid is not None else str(task)


def _task_dtype(task, default: str = "bf16") -> str:
    return getattr(task, "dtype", default) or default


def _feedback(obs: Observation, rr) -> dict:
    """Compact summary handed to the policy for the next serial turn."""
    return {
        "compiled": obs.compiled,
        "correct": rr.correct,
        "speedup": rr.speedup,
        "reward": rr.reward,
        "snr_db": obs.snr_db,
        "flags": list(rr.flags),
        "error_text": obs.error_text,
        "detail": rr.detail,
    }


def _make_measure_fn(
    env_factory: Optional[Callable[[object], object]],
    dry_run: Optional[object],
) -> MeasureFn:
    """Build the measurement source.

    ``dry_run`` may be either:
      - a dict ``{task_id: [Observation, ...]}`` (one per turn; the last is
        reused if the budget exceeds the list), or
      - a callable ``(task, turn) -> Observation``.

    Otherwise ``env_factory(task) -> KoreEnv`` supplies live measurements; the
    env is built once per task and reused across turns.
    """
    if dry_run is not None:
        if callable(dry_run):
            def measure(task, kernel_source, turn):
                return dry_run(task, turn)
            return measure

        def measure(task, kernel_source, turn):
            obs_list = dry_run[_task_id(task)]
            idx = min(turn, len(obs_list) - 1)
            return obs_list[idx]
        return measure

    if env_factory is None:
        raise ValueError(
            "evaluate_policy needs env_factory (live GPU runs) or dry_run "
            "(precomputed Observations)"
        )

    _env_cache: dict[str, object] = {}

    def measure(task, kernel_source, turn):
        tid = _task_id(task)
        env = _env_cache.get(tid)
        if env is None:
            env = env_factory(task)
            _env_cache[tid] = env
        return env.step(kernel_source)

    return measure


def _integrity_gated_speedup(rr, cfg: KoreConfig) -> Optional[float]:
    """The timing-INTEGRITY-gated speedup for ranking + fast_p, mirroring the reward's
    ``su_scored`` (reward.py): an ``excessive_speedup`` measurement artifact is capped
    at ``cfg.excessive_speedup_flag`` and a ``high_variance`` (noisy) timing is damped
    to <=1x. Returns ``None`` when there is no measured speedup. Correctness is
    unaffected -- a damped kernel still counts as correct, it just cannot farm fast_p
    with a glitch/noise (audit R2 soup-eval C2)."""
    su = getattr(rr, "speedup", None)
    if su is None:
        return None
    flags = getattr(rr, "flags", ()) or ()
    if "excessive_speedup" in flags:
        su = min(su, float(cfg.excessive_speedup_flag))
    if "high_variance" in flags:
        su = min(su, 1.0)
    return su


def _run_task(
    policy_fn: PolicyFn,
    task,
    measure: MeasureFn,
    budget: int,
    mode: str,
    cfg: KoreConfig,
    torch_baseline: Optional[object] = None,
) -> dict:
    """Run one task under a matched budget; return best-correct-speedup record.

    mode "serial"   : one trajectory of ``budget`` turns; feedback accumulates.
    mode "parallel" : ``budget`` independent single-turn samples; no feedback.

    ``torch_baseline`` (optional) supplies a torch-eager baseline time per task
    (dict ``{task_id: ms}`` or callable ``task -> ms``) so a SECOND fast_p curve
    (vs torch-eager, KernelBench-comparable) can be reported alongside the
    production AITER/hipBLASLt curve. It may also be carried on the Observation
    as a ``torch_baseline_ms`` attribute (e.g. filled by ``driver --impl torch``).
    """
    dtype = _task_dtype(task)
    best_speedup: Optional[float] = None
    best_reward: Optional[float] = None
    best_obs: Optional[Observation] = None
    benches_used = 0
    benches_to_best: Optional[int] = None
    trajectory: list[dict] = []
    feedback: Optional[dict] = None

    for turn in range(max(0, budget)):
        kernel_source = policy_fn(task, feedback if mode == "serial" else None)
        obs = measure(task, kernel_source, turn)
        rr = compute_reward(obs, kernel_source, dtype=dtype, mode="eval", cfg=cfg)
        benches_used += 1
        # Gate the speedup that feeds fast_p on timing INTEGRITY, exactly like the
        # reward's su_scored: an excessive-ratio measurement artifact is capped and a
        # high-variance (noisy) timing is damped to <=1x. rr.speedup itself is the RAW
        # worst-shape ratio, so ranking/fast_p on it would let a timing glitch or a
        # noisy bench farm the headline metric (audit R2 soup-eval C2).
        su_gated = _integrity_gated_speedup(rr, cfg)
        trajectory.append({
            "turn": turn,
            "correct": rr.correct,
            "speedup": rr.speedup,
            "speedup_gated": su_gated,
            "reward": rr.reward,
            "flags": list(rr.flags),
        })
        if rr.correct and su_gated is not None:
            if best_speedup is None or su_gated > best_speedup:
                best_speedup = su_gated
                best_reward = rr.reward
                best_obs = obs
                benches_to_best = benches_used
        # serial mode conditions the next turn on this turn's outcome
        feedback = _feedback(obs, rr)

    correct = best_speedup is not None
    # Normalized times so fast_p is measurement-unit independent: with
    # baseline=1.0, actual=1/speedup, the reconstructed speedup is exact.
    baseline_time = 1.0
    actual_time = (1.0 / best_speedup) if correct else float("inf")

    # torch-eager (KernelBench-comparable) baseline for a SECOND fast_p curve.
    # Expressed relative to the production baseline (=1.0): torch_ms/prod_ms, so
    # speedup_vs_torch = torch_baseline_time / actual_time = (torch/prod)*speedup.
    torch_baseline_time = _torch_baseline_time(task, best_obs, torch_baseline) if correct else None

    return {
        "task_id": _task_id(task),
        "correct": correct,
        "best_speedup": best_speedup,
        "best_reward": best_reward,
        "baseline_time": baseline_time,
        "actual_time": actual_time,
        "torch_baseline_time": torch_baseline_time,
        "benches_used": benches_used,
        "benches_to_best": benches_to_best,
        "trajectory": trajectory,
    }


def _torch_baseline_time(task, best_obs, torch_baseline) -> Optional[float]:
    """Torch-eager baseline time normalized to the production baseline (=1.0)."""
    if best_obs is None:
        return None
    tb = None
    if torch_baseline is not None:
        tb = torch_baseline(task) if callable(torch_baseline) else torch_baseline.get(_task_id(task))
    if tb is None:
        tb = getattr(best_obs, "torch_baseline_ms", None)
    prod_ms = getattr(best_obs, "baseline_ms", None)
    if tb and prod_ms and prod_ms > 0:
        return float(tb) / float(prod_ms)
    return None


def _assemble(per_task: list[dict], budget: int, mode: str, ps: Sequence[float]) -> dict:
    is_correct = [t["correct"] for t in per_task]
    baseline_speed = [t["baseline_time"] for t in per_task]
    actual_speed = [t["actual_time"] for t in per_task]
    n = len(per_task)
    curve = fast_p_curve(is_correct, baseline_speed, actual_speed, n, ps)
    out = {
        "mode": mode,
        "budget": budget,
        "n": n,
        "per_task": per_task,
        "is_correct": is_correct,
        "baseline_speed": baseline_speed,
        "actual_speed": actual_speed,
        "fast_p_curve": curve,
        "fast_p": {p: v for p, v in curve},
        "geometric_mean_speedup": geometric_mean_speedup(is_correct, baseline_speed, actual_speed),
        "num_correct": sum(1 for c in is_correct if c),
    }

    # SECOND fast_p curve vs the torch-eager baseline (KernelBench-comparable),
    # reported only when torch-eager times are available. ``None`` torch times
    # contribute 0 (uncorrected denominator ``n``), exactly like a missing bench.
    torch_speed = [t.get("torch_baseline_time") for t in per_task]
    if any(v is not None for v in torch_speed):
        torch_curve = fast_p_curve(is_correct, torch_speed, actual_speed, n, ps)
        out["fast_p_curve_vs_torch"] = torch_curve
        out["fast_p_vs_torch"] = {p: v for p, v in torch_curve}
        out["torch_baseline_speed"] = torch_speed
        out["geometric_mean_speedup_vs_torch"] = geometric_mean_speedup(
            is_correct, torch_speed, actual_speed
        )
    return out


def evaluate_policy(
    policy_fn: PolicyFn,
    tasks: Sequence,
    env_factory: Optional[Callable[[object], object]] = None,
    budget: int = 5,
    mode: str = "serial",
    *,
    dry_run: Optional[object] = None,
    ps: Sequence[float] = DEFAULT_PS,
    cfg: KoreConfig = CONFIG,
    torch_baseline: Optional[object] = None,
) -> dict:
    """Evaluate one policy over ``tasks`` under a matched measurement budget.

    Each task gets at most ``budget`` benches. Per task we keep the best correct
    speedup; the split-level ``fast_p`` curve is computed over all tasks (``n``
    = number of tasks, uncorrected).

    Provide either ``env_factory`` (live ``KoreEnv`` runs) or ``dry_run``
    (precomputed ``Observation`` objects) for CPU-only testing.

    ``torch_baseline`` optionally supplies torch-eager times (``{task_id: ms}``
    or ``task -> ms``) to add a second fast_p curve vs torch-eager.
    """
    measure = _make_measure_fn(env_factory, dry_run)
    per_task = [
        _run_task(policy_fn, task, measure, budget, mode, cfg, torch_baseline=torch_baseline)
        for task in tasks
    ]
    return _assemble(per_task, budget, mode, ps)


def matched_budget_bakeoff(
    policies: dict,
    tasks: Sequence,
    budget: int = 5,
    env_factory: Optional[Callable[[object], object]] = None,
    mode: str = "serial",
    *,
    dry_run: Optional[object] = None,
    ps: Sequence[float] = DEFAULT_PS,
    cfg: KoreConfig = CONFIG,
) -> dict:
    """Compare multiple policies at an EQUAL budget (the matched-budget bake-off).

    ``policies`` maps name -> ``policy_fn``. Returns a dict of per-policy results
    plus a ranking by ``fast_p`` at p=1.0.
    """
    results = {
        name: evaluate_policy(
            pf, tasks, env_factory=env_factory, budget=budget, mode=mode,
            dry_run=dry_run, ps=ps, cfg=cfg,
        )
        for name, pf in policies.items()
    }
    ranking = sorted(
        results.keys(),
        key=lambda name: results[name]["fast_p"].get(1.0, 0.0),
        reverse=True,
    )
    return {
        "budget": budget,
        "mode": mode,
        "n": len(tasks),
        "policies": results,
        "ranking_by_fast1": ranking,
    }


def serial_vs_parallel(
    policy_fn: PolicyFn,
    task,
    total_budget: int,
    env_factory: Optional[Callable[[object], object]] = None,
    *,
    dry_run: Optional[object] = None,
    cfg: KoreConfig = CONFIG,
) -> dict:
    """Serial refinement vs parallel sampling at equal total budget (Kevin).

    serial  : 1 trajectory x ``total_budget`` turns (feedback accumulates).
    parallel: ``total_budget`` trajectories x 1 turn (independent, best-of-N).

    Returns two comparable best-speedup numbers plus the full sub-results.
    """
    serial = evaluate_policy(
        policy_fn, [task], env_factory=env_factory, budget=total_budget,
        mode="serial", dry_run=dry_run, cfg=cfg,
    )
    parallel = evaluate_policy(
        policy_fn, [task], env_factory=env_factory, budget=total_budget,
        mode="parallel", dry_run=dry_run, cfg=cfg,
    )
    serial_best = serial["per_task"][0]["best_speedup"]
    parallel_best = parallel["per_task"][0]["best_speedup"]
    return {
        "total_budget": total_budget,
        "serial_best_speedup": serial_best,
        "parallel_best_speedup": parallel_best,
        "serial_wins": (serial_best or 0.0) >= (parallel_best or 0.0),
        "serial": serial,
        "parallel": parallel,
    }


def benches_to_best(value_scores: Sequence[float], true_speedups: Sequence[float]) -> dict:
    """Value-model lever: how many benches until the truly-best candidate is hit.

    Candidates are benched in order of descending value-model score; we report
    the rank (1-indexed) at which the candidate with the highest TRUE speedup is
    reached, versus the ``(n+1)/2`` expected under random ordering. Fewer benches
    means a more useful value model.
    """
    n = min(len(value_scores), len(true_speedups))
    if n == 0:
        return {"benches_to_best": 0, "n": 0, "random_expected": 0.0, "best_idx": None}
    order = sorted(range(n), key=lambda i: value_scores[i], reverse=True)
    best_idx = max(range(n), key=lambda i: true_speedups[i])
    benches = order.index(best_idx) + 1
    return {
        "benches_to_best": benches,
        "n": n,
        "random_expected": (n + 1) / 2.0,
        "best_idx": best_idx,
        "speedup_at_best": true_speedups[best_idx],
    }


# --------------------------------------------------------------------------- #
# Multi-seed fast_p with confidence intervals (report mean +/- CI over seeds)
# --------------------------------------------------------------------------- #
def aggregate_fastp_over_seeds(per_seed_results: Sequence[dict],
                               ps: Sequence[float] = DEFAULT_PS) -> dict:
    """Aggregate several single-seed ``evaluate_policy`` results into mean +/- CI.

    For every ``p`` we report the mean fast_p across seeds and its 95% CI (normal
    approx), plus the same for correctness count and geometric-mean speedup.
    """
    results = list(per_seed_results)

    def _fp(res: dict, p: float) -> float:
        fp = res.get("fast_p", {})
        if float(p) in fp:
            return float(fp[float(p)])
        if p in fp:
            return float(fp[p])
        return 0.0

    fast_p_mean_ci = {float(p): mean_ci([_fp(r, p) for r in results]) for p in ps}
    torch_present = any("fast_p_vs_torch" in r for r in results)
    fast_p_vs_torch_mean_ci = None
    if torch_present:
        def _fpt(res: dict, p: float) -> float:
            fp = res.get("fast_p_vs_torch", {})
            return float(fp.get(float(p), fp.get(p, 0.0)))
        fast_p_vs_torch_mean_ci = {float(p): mean_ci([_fpt(r, p) for r in results]) for p in ps}

    return {
        "num_seeds": len(results),
        "fast_p_mean_ci": fast_p_mean_ci,
        "fast_p_vs_torch_mean_ci": fast_p_vs_torch_mean_ci,
        "num_correct_mean_ci": mean_ci([r.get("num_correct", 0) for r in results]),
        "geomean_mean_ci": mean_ci([r.get("geometric_mean_speedup", 0.0) for r in results]),
        "per_seed_fast_p": [r.get("fast_p", {}) for r in results],
    }


def evaluate_policy_multiseed(
    policy_fn: PolicyFn,
    tasks: Sequence,
    seeds: Sequence[int] = (0, 1, 2),
    env_factory: Optional[Callable[[object], object]] = None,
    budget: int = 5,
    mode: str = "serial",
    *,
    dry_run: Optional[object] = None,
    seed_dry_run: Optional[Callable[[int], object]] = None,
    ps: Sequence[float] = DEFAULT_PS,
    cfg: KoreConfig = CONFIG,
    torch_baseline: Optional[object] = None,
) -> dict:
    """Evaluate a policy over >=3 seeds and report fast_p as mean +/- CI.

    Each seed is a full ``evaluate_policy`` run. Use ``seed_dry_run`` (a callable
    ``seed -> dry_run source``) to vary fabricated observations per seed in tests;
    otherwise the same ``dry_run``/``env_factory`` is reused for every seed.
    """
    seeds = list(seeds)
    per_seed: list[dict] = []
    for sd in seeds:
        dr = seed_dry_run(sd) if seed_dry_run is not None else dry_run
        res = evaluate_policy(
            policy_fn, tasks, env_factory=env_factory, budget=budget, mode=mode,
            dry_run=dr, ps=ps, cfg=cfg, torch_baseline=torch_baseline,
        )
        res["seed"] = sd
        per_seed.append(res)
    agg = aggregate_fastp_over_seeds(per_seed, ps)
    agg.update({"seeds": seeds, "mode": mode, "budget": budget,
                "n": len(tasks), "per_seed": per_seed})
    return agg


# --------------------------------------------------------------------------- #
# Unbiased best-of-N: pass@k and fast_p@k over a parallel-sampling eval
# --------------------------------------------------------------------------- #
def best_of_n_pass_at_k(eval_result: dict, ks: Sequence[int] = (1,),
                        ps: Sequence[float] = DEFAULT_PS) -> dict:
    """Unbiased pass@k and fast_p@k from a parallel best-of-N eval result.

    Each task's trajectory is treated as ``n`` independent samples. pass@k is the
    per-task unbiased estimate averaged over the split; fast_p@k additionally
    requires the sample to be faster than the production baseline by a factor p.
    Use ``mode="parallel"`` in ``evaluate_policy`` so samples are independent.
    """
    per_task = eval_result.get("per_task", [])
    ks = list(ks)
    pass_k: dict[int, float] = {}
    fast_pk: dict[str, float] = {}
    for k in ks:
        p_vals = []
        for t in per_task:
            traj = t.get("trajectory", [])
            n = len(traj)
            c = sum(1 for s in traj if s.get("correct"))
            p_vals.append(pass_at_k(n, c, k))
        pass_k[k] = (sum(p_vals) / len(p_vals)) if p_vals else 0.0
        for p in ps:
            f_vals = []
            for t in per_task:
                traj = t.get("trajectory", [])
                n = len(traj)
                c = sum(1 for s in traj
                        if s.get("correct") and (s.get("speedup") or 0.0) > p)
                f_vals.append(pass_at_k(n, c, k))
            fast_pk[f"k={k},p={float(p):g}"] = (sum(f_vals) / len(f_vals)) if f_vals else 0.0
    return {"ks": ks, "pass_at_k": pass_k, "fast_p_at_k": fast_pk, "n": len(per_task)}
