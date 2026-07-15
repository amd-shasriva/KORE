"""KORE vs Opus head-to-head kernel bake-off (does the trained policy beat the teacher?).

KORE's flagship claim is that a *specialized* policy, trained on verified AMD
ROCm/Triton kernels, can out-write a *frontier generalist* (Claude Opus 4.x) at
the one thing it was trained for: fast, correct gfx950/MI350X kernels graded vs
production vendor baselines (AITER / hipBLASLt). This module measures that claim
directly, on the HELD-OUT task split, under a matched measurement budget.

Both sides are scored through the EXACT same path as the campaign's own bake-off:

  * a policy is a ``PolicyFn(task, feedback) -> kernel_source`` (see
    :mod:`kore.eval.policies` / :mod:`kore.eval.bakeoff`);
  * the KORE side wraps the served checkpoint's ``generate`` callable, the Opus
    side wraps the frontier teacher (:func:`kore.data.teacher.make_teacher`, which
    defaults to ``claude-opus-4.8`` through AMD's internal LLM gateway);
  * BOTH are built by :func:`kore.eval.policies.model_policy`, so they share the
    identical prompt contract (``build_transcript`` from :mod:`kore.policy.format`
    + the per-turn verifier-feedback rendering) and the identical response parser
    (``parse_response``). The ONLY thing that differs is the token source;
  * both run under the SAME :class:`~kore.env.kore_env.KoreEnv` (verified
    correctness oracle + cold-cache timing), the SAME budget, and the SAME
    timing-INTEGRITY gate (:func:`kore.eval.bakeoff._integrity_gated_speedup`) that
    caps excessive-ratio artifacts and damps high-variance benches, so neither
    side can farm the headline metric with a glitch or noise.

We then report, with confidence intervals over seeds:

  * ``fast_p`` for each side (mean +/- 95% CI), reusing
    :func:`kore.eval.bakeoff.aggregate_fastp_over_seeds`;
  * the WIN-RATE of KORE over Opus: the fraction of held-out tasks on which KORE's
    best correct kernel is strictly faster than Opus's best correct kernel (a side
    only competes with a CORRECT kernel), plus the reciprocal Opus win-rate and the
    tie-rate, each as mean +/- 95% CI over seeds.

Graceful degradation (like the retention gate): if the teacher / gateway is not
provisioned (no anthropic SDK, no ``AMD_LLM_API_KEY``, or a sustained outage
mid-run) the Opus side is SKIPPED with a loud warning and the KORE-only numbers
are still returned. This module never crashes the caller because the teacher is
down.

Import-safe / offline: nothing heavy is imported at module load. torch / vLLM are
only reached via the injected KORE ``generate`` callable (the caller builds it
with :func:`kore.policy.serve.load_generate`), and the Anthropic SDK only via the
lazily-constructed teacher. The pure logic (win-rate, aggregation, CIs, report)
is unit-testable on CPU with stub policies and no network (see
``tests/test_vs_opus.py``).
"""

from __future__ import annotations

import sys
from typing import Callable, Optional, Sequence

from kore.config import CONFIG, KoreConfig
from kore.eval.bakeoff import aggregate_fastp_over_seeds, evaluate_policy
from kore.eval.fastp import DEFAULT_PS, mean_ci
from kore.eval.policies import PolicyFn, model_policy
from kore.obs import get_logger

_LOG = get_logger("eval.vs_opus")

# The default frontier teacher label. ``make_teacher('claude')`` builds a
# ClaudeTeacher whose model already defaults to claude-opus-4.8 (see
# kore.data.teacher.ClaudeTeacher), so "opus" and "claude" resolve to the same
# frontier model here.
DEFAULT_TEACHER_KIND = "claude"


# --------------------------------------------------------------------------- #
# Loud, non-fatal warning (mirrors the retention gate's "gate NOT enforced").
# --------------------------------------------------------------------------- #
def _loud_warn(log, msg: str, **fields) -> None:
    """Emit a LOUD but non-fatal warning: structured log + a stderr banner.

    The head-to-head must never crash because the teacher/gateway is down; it
    degrades like the retention gate, which logs "gate NOT enforced" and returns.
    """
    (log or _LOG).warn(msg, **fields)
    try:
        print(f"[vs_opus] WARNING: {msg}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - never let logging break the eval
        pass


# --------------------------------------------------------------------------- #
# Policies: wrap a generate callable / a teacher into the SAME model_policy path.
# --------------------------------------------------------------------------- #
def _generate_adapter(generate: Callable[..., str]) -> Callable[..., str]:
    """Adapt any ``generate(messages, ...)`` to the ``model_policy`` generate ABI.

    ``model_policy`` calls ``generate(messages, max_tokens=..., temperature=...)``.
    A KORE served model (``kore.policy.serve.load_generate``) already accepts those
    kwargs; a stub in a test may not. Swallow unknown kwargs so any messages->str
    callable plugs in unchanged.
    """
    def gen(messages, **kw):
        try:
            return generate(messages, **kw)
        except TypeError:
            # The callable does not accept the sampling kwargs; call it plainly.
            return generate(messages)
    return gen


def _teacher_generate(teacher) -> Callable[..., str]:
    """Adapt a :class:`kore.data.teacher.TeacherClient` to the generate ABI.

    A ``TeacherClient.generate(messages)`` carries its OWN decoding params (set at
    construction), so the ``max_tokens`` / ``temperature`` that ``model_policy``
    forwards are intentionally dropped here.
    """
    def gen(messages, **_kw):
        return teacher.generate(messages)
    return gen


def kore_policy(kore_generate_fn: Callable[..., str], *,
                system_prompt: Optional[str] = None,
                max_tokens: int = 8192, temperature: float = 0.0) -> PolicyFn:
    """Build the KORE-side ``PolicyFn`` from a served-model ``generate`` callable.

    This is exactly :func:`kore.eval.policies.model_policy` with the ``generate``
    backend injected, so it shares the transcript contract and parser with the
    campaign's own eval; no torch is imported here (the caller built ``generate``).
    """
    return model_policy("kore", generate=_generate_adapter(kore_generate_fn),
                        system_prompt=system_prompt, max_tokens=max_tokens,
                        temperature=temperature)


def opus_policy(teacher=None, *, kind: str = DEFAULT_TEACHER_KIND,
                system_prompt: Optional[str] = None,
                max_tokens: int = 8192, temperature: float = 0.0,
                **teacher_kwargs) -> PolicyFn:
    """Build the Opus-side ``PolicyFn`` backed by the frontier teacher.

    Given a task, it builds the prompt via :mod:`kore.policy.format`
    (``build_transcript`` inside ``model_policy``), calls the teacher
    (:func:`kore.data.teacher.make_teacher`, defaulting to ``claude-opus-4.8``) to
    generate a kernel, and parses it with ``parse_response`` - i.e. it goes through
    the SAME :func:`kore.eval.policies.model_policy` construction as the KORE side,
    so ``evaluate_policy`` / ``matched_budget_bakeoff`` score both identically.

    ``teacher`` may be supplied (e.g. a shared, already-authenticated client, or a
    ``StubTeacher`` in tests); otherwise it is built lazily with
    ``make_teacher(kind, **teacher_kwargs)`` (which imports the Anthropic SDK).
    """
    if teacher is None:
        from kore.data.teacher import make_teacher  # lazy: anthropic SDK
        teacher = make_teacher(kind, **teacher_kwargs)
    return model_policy("opus", generate=_teacher_generate(teacher),
                        system_prompt=system_prompt, max_tokens=max_tokens,
                        temperature=temperature)


def make_opus_teacher(kind: str = DEFAULT_TEACHER_KIND, *, model: Optional[str] = None,
                      resilient: bool = True, log=None, **teacher_kwargs):
    """Build the frontier teacher, returning ``None`` (not raising) if unavailable.

    This is the graceful-degradation entry point for :func:`head_to_head` and the
    CLI: any provisioning failure (no anthropic SDK, missing ``AMD_LLM_API_KEY``,
    gateway unreachable) is caught and turned into a loud warning + ``None`` so the
    caller can SKIP the Opus side rather than crash - exactly how the retention
    gate tolerates an unprovisioned serving backend.
    """
    try:
        from kore.data.teacher import load_env_local, make_teacher
        load_env_local()
        kw = dict(teacher_kwargs)
        if model:
            kw["model"] = model
        return make_teacher(kind, resilient=resilient, **kw)
    except Exception as e:  # noqa: BLE001 - provisioning failure is non-fatal here
        _loud_warn(log, "frontier teacher NOT provisioned; Opus side will be SKIPPED "
                        "(head-to-head reports KORE-only)",
                   kind=kind, exc_type=type(e).__name__, exc=str(e)[:200])
        return None


def build_policies(kore_generate_fn: Callable[..., str], teacher, *,
                   system_prompt: Optional[str] = None, max_tokens: int = 8192,
                   temperature: float = 0.0) -> dict:
    """Return ``{"kore": PolicyFn, "opus": PolicyFn}`` for ``matched_budget_bakeoff``.

    A convenience so callers can drop BOTH sides straight into the existing
    :func:`kore.eval.bakeoff.matched_budget_bakeoff` at an equal budget. ``opus`` is
    omitted when ``teacher`` is ``None`` (unavailable).
    """
    policies: dict = {
        "kore": kore_policy(kore_generate_fn, system_prompt=system_prompt,
                            max_tokens=max_tokens, temperature=temperature),
    }
    if teacher is not None:
        policies["opus"] = opus_policy(teacher=teacher, system_prompt=system_prompt,
                                       max_tokens=max_tokens, temperature=temperature)
    return policies


# --------------------------------------------------------------------------- #
# PURE win-rate logic (CPU-unit-tested; no GPU, no network).
# --------------------------------------------------------------------------- #
def _competing_speedup(rec: Optional[dict]) -> Optional[float]:
    """The integrity-gated best speedup a side brings to the comparison.

    A side only competes with a CORRECT kernel; ``best_speedup`` is already the
    timing-INTEGRITY-gated speedup that fast_p ranks on (excessive-ratio artifacts
    capped, high-variance benches damped), so comparing the two sides' values is
    apples-to-apples and cannot be farmed by a glitch/noise.
    """
    if not rec or not rec.get("correct"):
        return None
    su = rec.get("best_speedup")
    return float(su) if su is not None else None


def winner_for_task(kore_rec: Optional[dict], opus_rec: Optional[dict],
                    *, margin: float = 1.0) -> str:
    """Decide one matched task: ``"kore"`` | ``"opus"`` | ``"tie"`` | ``"neither"``.

    KORE wins the task iff it has a correct kernel that is faster than Opus's best
    correct kernel by at least the multiplicative ``margin`` (``margin=1.0`` means
    strictly faster). If only one side is correct, that side wins; if neither is
    correct, the task is ``"neither"`` (it still counts in the uncorrected
    denominator, penalizing both sides).
    """
    ks = _competing_speedup(kore_rec)
    os_ = _competing_speedup(opus_rec)
    if ks is None and os_ is None:
        return "neither"
    if os_ is None:
        return "kore"
    if ks is None:
        return "opus"
    m = float(margin) if margin and margin > 0 else 1.0
    if ks > os_ * m:
        return "kore"
    if os_ > ks * m:
        return "opus"
    return "tie"


def head_to_head_winrate(kore_per_task: Sequence[dict], opus_per_task: Sequence[dict],
                         *, margin: float = 1.0) -> dict:
    """Per-task win-rate of KORE over Opus on the matched held-out split.

    Both inputs are ``evaluate_policy(...)["per_task"]`` lists (one record per
    task, carrying ``task_id`` / ``correct`` / ``best_speedup``). Tasks are matched
    by ``task_id``. The denominator ``n`` is the KORE split size (uncorrected, like
    fast_p), so an unattempted / failed task counts against the win-rate.

    Returns win / opus-win / tie / both-incorrect counts and rates plus a per-task
    breakdown. PURE: no GPU, no network, no torch.
    """
    opus_by_id = {r.get("task_id"): r for r in opus_per_task}
    counts = {"kore": 0, "opus": 0, "tie": 0, "neither": 0}
    per_task: list[dict] = []
    for kr in kore_per_task:
        tid = kr.get("task_id")
        orr = opus_by_id.get(tid)
        w = winner_for_task(kr, orr, margin=margin)
        counts[w] += 1
        per_task.append({
            "task_id": tid,
            "winner": w,
            "kore_correct": bool(kr.get("correct")),
            "opus_correct": bool(orr.get("correct")) if orr else False,
            "kore_speedup": _competing_speedup(kr),
            "opus_speedup": _competing_speedup(orr),
        })
    n = len(kore_per_task)
    return {
        "n": n,
        "margin": float(margin),
        "counts": counts,
        "win_rate": counts["kore"] / n if n else 0.0,          # KORE beats Opus
        "opus_win_rate": counts["opus"] / n if n else 0.0,
        "tie_rate": counts["tie"] / n if n else 0.0,
        "both_incorrect_rate": counts["neither"] / n if n else 0.0,
        "per_task": per_task,
    }


def _fastp_at(res: dict, p: float) -> float:
    """Extract fast_p at threshold ``p`` from an ``evaluate_policy`` result."""
    fp = res.get("fast_p", {}) or {}
    if float(p) in fp:
        return float(fp[float(p)])
    if p in fp:
        return float(fp[p])
    return 0.0


# --------------------------------------------------------------------------- #
# The head-to-head driver.
# --------------------------------------------------------------------------- #
def head_to_head(
    tasks: Sequence,
    kore_generate_fn: Callable[..., str],
    teacher,
    budget: int = 5,
    seeds: Sequence[int] = (0, 1, 2),
    *,
    env_factory: Optional[Callable[[object], object]] = None,
    mode: str = "serial",
    margin: float = 1.0,
    ps: Sequence[float] = DEFAULT_PS,
    cfg: KoreConfig = CONFIG,
    system_prompt: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    kore_dry_run: Optional[object] = None,
    opus_dry_run: Optional[object] = None,
    seed_kore_dry_run: Optional[Callable[[int], object]] = None,
    seed_opus_dry_run: Optional[Callable[[int], object]] = None,
    log=None,
) -> dict:
    """Run the KORE-vs-Opus head-to-head on ``tasks`` and report fast_p + win-rate.

    Both sides are built with :func:`kore.eval.policies.model_policy` (so they are
    prompted + parsed identically) and scored with :func:`evaluate_policy` under the
    SAME ``env_factory`` (live :class:`KoreEnv` verify + cold-cache bench), the SAME
    ``budget``, and the SAME timing-integrity gate, once per seed. We report:

      * ``kore`` / ``opus``: fast_p mean +/- 95% CI over seeds (via
        :func:`kore.eval.bakeoff.aggregate_fastp_over_seeds`) plus each seed's raw
        ``evaluate_policy`` result under ``per_seed``;
      * ``win_rate_mean_ci`` (and ``opus_win_rate_mean_ci`` / ``tie_rate_mean_ci``):
        the fraction of held-out tasks KORE wins, mean +/- 95% CI over seeds;
      * ``fast_p_delta_mean_ci``: KORE minus Opus fast_p per threshold, mean +/- CI.

    Graceful degradation: ``teacher=None`` (or a teacher that fails mid-run) SKIPS
    the Opus side with a loud warning and returns ``skipped=True`` with the KORE
    numbers intact - the head-to-head never crashes because the gateway is down.

    Testing hooks (CPU, no GPU/network): pass ``kore_dry_run`` / ``opus_dry_run``
    (precomputed ``Observation`` maps, per :func:`evaluate_policy`) or the per-seed
    ``seed_kore_dry_run(seed)`` / ``seed_opus_dry_run(seed)`` callables to fabricate
    each side's measurements; ``teacher`` can be a ``StubTeacher``.
    """
    seeds = list(seeds)
    n = len(tasks)

    # KORE side (always evaluated; its numbers survive an Opus skip).
    kore_pol = kore_policy(kore_generate_fn, system_prompt=system_prompt,
                           max_tokens=max_tokens, temperature=temperature)
    kore_seed_results: list[dict] = []
    for sd in seeds:
        kdr = seed_kore_dry_run(sd) if seed_kore_dry_run is not None else kore_dry_run
        res = evaluate_policy(kore_pol, tasks, env_factory=env_factory, budget=budget,
                              mode=mode, dry_run=kdr, ps=ps, cfg=cfg)
        res["seed"] = sd
        kore_seed_results.append(res)

    out: dict = {
        "n": n,
        "budget": budget,
        "mode": mode,
        "seeds": seeds,
        "margin": float(margin),
        "skipped": False,
        "skip_reason": None,
        "kore": {**aggregate_fastp_over_seeds(kore_seed_results, ps),
                 "per_seed": kore_seed_results},
        "opus": None,
        "win_rate_mean_ci": None,
        "opus_win_rate_mean_ci": None,
        "tie_rate_mean_ci": None,
        "fast_p_delta_mean_ci": None,
        "per_seed_winrate": [],
    }

    if teacher is None:
        out["skipped"] = True
        out["skip_reason"] = "teacher/gateway unavailable (teacher is None)"
        _loud_warn(log, "Opus side SKIPPED: no frontier teacher provided; "
                        "reporting KORE-only fast_p (head-to-head not computed)")
        return out

    # Opus side (graceful): a provisioning error or a sustained-outage hard-stop
    # from the ResilientTeacher must SKIP the comparison, never crash the eval.
    opus_pol = opus_policy(teacher=teacher, system_prompt=system_prompt,
                           max_tokens=max_tokens, temperature=temperature)
    opus_seed_results: list[dict] = []
    try:
        for sd in seeds:
            odr = seed_opus_dry_run(sd) if seed_opus_dry_run is not None else opus_dry_run
            res = evaluate_policy(opus_pol, tasks, env_factory=env_factory, budget=budget,
                                  mode=mode, dry_run=odr, ps=ps, cfg=cfg)
            res["seed"] = sd
            opus_seed_results.append(res)
    except Exception as e:  # noqa: BLE001 - teacher outage is non-fatal to the eval
        out["skipped"] = True
        out["skip_reason"] = f"teacher failed during eval: {type(e).__name__}: {str(e)[:200]}"
        _loud_warn(log, "Opus side SKIPPED mid-run (teacher/gateway failure); "
                        "reporting KORE-only fast_p (head-to-head not computed)",
                   exc_type=type(e).__name__, exc=str(e)[:200])
        return out

    # Both sides scored: aggregate fast_p + win-rate with CIs over seeds.
    winrate_records: list[dict] = []
    for kres, ores in zip(kore_seed_results, opus_seed_results):
        wr = head_to_head_winrate(kres["per_task"], ores["per_task"], margin=margin)
        wr["seed"] = kres.get("seed")
        winrate_records.append(wr)

    out["opus"] = {**aggregate_fastp_over_seeds(opus_seed_results, ps),
                   "per_seed": opus_seed_results}
    out["win_rate_mean_ci"] = mean_ci([wr["win_rate"] for wr in winrate_records])
    out["opus_win_rate_mean_ci"] = mean_ci([wr["opus_win_rate"] for wr in winrate_records])
    out["tie_rate_mean_ci"] = mean_ci([wr["tie_rate"] for wr in winrate_records])
    out["fast_p_delta_mean_ci"] = {
        float(p): mean_ci([_fastp_at(k, p) - _fastp_at(o, p)
                           for k, o in zip(kore_seed_results, opus_seed_results)])
        for p in ps
    }
    out["per_seed_winrate"] = winrate_records
    return out


# --------------------------------------------------------------------------- #
# Reporting (PURE markdown; ASCII only, no em/en dashes).
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def format_vs_opus_report(res: dict) -> str:
    """Human-readable markdown for a :func:`head_to_head` result."""
    lines: list[str] = []
    lines.append("# KORE vs Opus head-to-head (held-out kernels)")
    lines.append("")
    lines.append(f"- **split size (n)**: {res.get('n', '?')}")
    lines.append(f"- **budget** (max benches/task): {res.get('budget', '?')}")
    lines.append(f"- **mode**: {res.get('mode', '?')}")
    lines.append(f"- **seeds**: {res.get('seeds', [])}")
    lines.append("")

    kore = res.get("kore") or {}
    kore_fp = kore.get("fast_p_mean_ci", {}) or {}

    if res.get("skipped"):
        lines.append(f"> **Opus side SKIPPED**: {res.get('skip_reason')}")
        lines.append("> Reporting KORE-only fast_p (the head-to-head was not computed).")
        lines.append("")
        lines.append("| p | KORE fast_p (mean) | CI95 |")
        lines.append("| --- | --- | --- |")
        for p, mc in sorted(kore_fp.items()):
            lines.append(f"| {float(p):g} | {_pct(mc['mean'])} | +/-{_pct(mc['ci95'])} |")
        lines.append("")
        return "\n".join(lines)

    opus = res.get("opus") or {}
    opus_fp = opus.get("fast_p_mean_ci", {}) or {}
    delta = res.get("fast_p_delta_mean_ci", {}) or {}

    lines.append("## fast_p (mean +/- 95% CI over seeds)")
    lines.append("")
    lines.append("| p | KORE | Opus | KORE - Opus |")
    lines.append("| --- | --- | --- | --- |")
    for p in sorted(kore_fp.keys()):
        k = kore_fp.get(p, {})
        o = opus_fp.get(p, {})
        d = delta.get(p, {})
        lines.append(
            f"| {float(p):g} | {_pct(k.get('mean', 0.0))} +/-{_pct(k.get('ci95', 0.0))} "
            f"| {_pct(o.get('mean', 0.0))} +/-{_pct(o.get('ci95', 0.0))} "
            f"| {_pct(d.get('mean', 0.0))} +/-{_pct(d.get('ci95', 0.0))} |"
        )
    lines.append("")

    wr = res.get("win_rate_mean_ci") or {}
    ow = res.get("opus_win_rate_mean_ci") or {}
    tr = res.get("tie_rate_mean_ci") or {}
    lines.append("## Win-rate (fraction of held-out tasks; mean +/- 95% CI over seeds)")
    lines.append("")
    lines.append(f"- **KORE beats Opus**: {_pct(wr.get('mean', 0.0))} +/-{_pct(wr.get('ci95', 0.0))}")
    lines.append(f"- **Opus beats KORE**: {_pct(ow.get('mean', 0.0))} +/-{_pct(ow.get('ci95', 0.0))}")
    lines.append(f"- **ties**: {_pct(tr.get('mean', 0.0))} +/-{_pct(tr.get('ci95', 0.0))}")
    lines.append("")

    # Headline verdict at p=1.0 (correct AND faster than the production baseline).
    d1 = delta.get(1.0, {}) or delta.get(float(1.0), {})
    win_mean = float(wr.get("mean", 0.0))
    verdict = "WINS" if win_mean > 0.5 else ("TIES" if abs(win_mean - 0.5) < 1e-9 else "LOSES")
    lines.append(f"**Verdict**: KORE {verdict} the head-to-head "
                 f"(win-rate {_pct(win_mean)}; fast_1 delta {_pct(d1.get('mean', 0.0))} "
                 f"+/-{_pct(d1.get('ci95', 0.0))}).")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Thin CLI hook: python -m kore.eval.vs_opus --kore-ckpt <ckpt> [...]
# --------------------------------------------------------------------------- #
def _resolve_tasks(task_ids: Optional[str]):
    """Held-out generalization split by default; else the named task ids."""
    if not task_ids:
        from kore.tasks.registry import heldout_tasks
        return heldout_tasks()
    from kore.tasks.registry import get_task
    out = []
    for tid in [t.strip() for t in task_ids.split(",") if t.strip()]:
        out.append(get_task(tid))
    return out


def main(argv=None) -> int:  # pragma: no cover - CLI wiring (GPU/gateway path)
    import argparse
    import json
    import os
    from pathlib import Path

    p = argparse.ArgumentParser(
        description="KORE vs Opus head-to-head kernel bake-off on the held-out split")
    p.add_argument("--kore-ckpt", required=True,
                   help="the trained KORE checkpoint to serve for the KORE side")
    p.add_argument("--teacher", default=DEFAULT_TEACHER_KIND,
                   help="teacher kind for the Opus side (default: claude -> opus-4.8)")
    p.add_argument("--teacher-model", default=None,
                   help="override the teacher model id (default: claude-opus-4.8)")
    p.add_argument("--backend", default="hf", help="serving backend for --kore-ckpt")
    p.add_argument("--tasks", default=None,
                   help="comma-separated task ids (default: the registry held-out split)")
    p.add_argument("--budget", type=int, default=5, help="max benches per task (matched)")
    p.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    p.add_argument("--mode", default="serial", choices=("serial", "parallel"))
    p.add_argument("--margin", type=float, default=1.0,
                   help="multiplicative speedup margin to win a task (>=1.0)")
    p.add_argument("--out", default=None,
                   help="optional path stem to persist JSON + markdown (default: stdout only)")
    args = p.parse_args(argv)

    # Cold-cache timing for the head-to-head bench (KoreEnv default is already
    # cold; set it explicitly, in THIS process only, for parity with the champion
    # gate). This never touches the live campaign (separate process).
    os.environ.setdefault("KORE_BENCH_COLD", "1")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = _resolve_tasks(args.tasks)
    if not tasks:
        print("[vs_opus] no tasks resolved; nothing to evaluate", file=sys.stderr)
        return 2

    teacher = make_opus_teacher(args.teacher, model=args.teacher_model)

    # KORE served model (lazy torch/vLLM import lives in load_generate).
    from kore.env.kore_env import KoreEnv
    from kore.policy.serve import load_generate
    kore_gen = load_generate(args.kore_ckpt, backend=args.backend)

    res = head_to_head(
        tasks, kore_gen, teacher, budget=args.budget, seeds=seeds,
        env_factory=lambda t: KoreEnv(t), mode=args.mode, margin=args.margin,
    )
    print(format_vs_opus_report(res))

    if args.out:
        stem = Path(args.out).with_suffix("")
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".json").write_text(json.dumps(res, indent=2, default=str))
        stem.with_suffix(".md").write_text(format_vs_opus_report(res))
        print(f"\n[vs_opus] report -> {stem.with_suffix('.json')}")
    return 0


__all__ = [
    "DEFAULT_TEACHER_KIND",
    "kore_policy",
    "opus_policy",
    "make_opus_teacher",
    "build_policies",
    "winner_for_task",
    "head_to_head_winrate",
    "head_to_head",
    "format_vs_opus_report",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
