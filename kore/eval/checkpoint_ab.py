"""Checkpoint A/B: is a trained checkpoint actually better than its base model?

Every other harness in :mod:`kore.eval` answers "how good is this policy". This
one answers the question a training run is judged by: **did the training buy
anything**, measured against the exact weights it started from, on tasks the
training never saw.

The protocol is deliberately boring, because that is what makes it credible:

  * ONE task scope - :func:`kore.tasks.registry.generalization_tasks`, the
    held-out reservation MINUS the tasks whose optimized source leaked into the
    midtrain corpus. A leaked task stays held out for training and stays in the
    decontamination gate, but it cannot carry a zero-shot number, so it is not
    scored here (:func:`scope_report` records the exclusion in the artifact).
  * ONE prompt per task, byte-identical across arms (:func:`first_turn_messages`),
    digested into the record so a prompt drift between arms is detectable after
    the fact rather than assumed away.
  * ONE matched measurement budget - both arms get the same number of benches per
    task, and both are scored by :func:`kore.eval.bakeoff.evaluate_policy`, i.e.
    the same code path, the same correctness oracle, and the same
    timing-INTEGRITY gate (excessive-ratio artifacts capped, high-variance benches
    damped) that the campaign's own bake-off uses.
  * PAIRED statistics - the two arms see the same tasks, so the comparison is
    paired and reported as an effect size with a bootstrap CI and an exact
    distribution-free p-value (:mod:`kore.eval.paired_stats`). For the binary
    outcomes (contract / compile / correct) the exact sign test on the per-task
    deltas *is* McNemar's exact test, which is the right test for paired binary
    data and is honest about how few discordant pairs a small split provides.

GENERATION IS DECOUPLED FROM MEASUREMENT, on purpose:

    phase A  generate_arm()  -> generations JSONL   (model, no GPU timing)
    phase B  measure_arm()   -> measures JSON       (GPU verify + cold-cache bench)
    phase C  compare_arms()  -> report JSON/markdown (pure, CPU, unit-tested)

At ``budget == 1`` (single-shot, no serial feedback) decoupling is EXACT: the
policy cannot condition on a measurement it never sees, so replaying a recorded
completion into :func:`kore.eval.bakeoff.evaluate_policy` is indistinguishable
from generating it inline. It buys three things that matter: the expensive GPU
measurement can be re-run without re-generating, both arms' kernels can be
measured back-to-back in one process on one idle device (so a serving job cannot
perturb the timings it is being compared on), and the completions themselves are
archived as evidence. Serial refinement (``budget > 1``) needs feedback in the
loop and must use :func:`kore.eval.policies.model_policy` live instead.

Import-safe / offline: nothing heavy is imported at module load. torch and
transformers are reached only inside :func:`load_hf_batch_generate`, the GPU env
only through the caller's ``env_factory``, and numpy only inside the statistics.
The whole scoring and comparison layer is pure and is exercised on CPU against a
stub policy by ``tests/test_checkpoint_ab.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from kore.eval.fastp import DEFAULT_PS

SCHEMA_GENERATIONS = "kore.checkpoint-ab-generations.v1"
SCHEMA_MEASURES = "kore.checkpoint-ab-measures.v1"
SCHEMA_REPORT = "kore.checkpoint-ab-report.v1"

#: A generation backend: ``generate(messages, max_tokens=..., temperature=...) -> str``
#: (the ABI of :func:`kore.policy.serve.load_generate`).
GenerateFn = Callable[..., str]
#: An optional batched backend: ``generate_batch([messages, ...], ...) -> [str, ...]``.
GenerateBatchFn = Callable[..., list]


def _task_id(task) -> str:
    if isinstance(task, str):
        return task
    return getattr(task, "task_id", None) or str(task)


def _sha12(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Scope: the held-out tasks a zero-shot claim may actually use.
# --------------------------------------------------------------------------- #
def generalization_scope(task_ids: Optional[Sequence[str]] = None) -> list:
    """The tasks this A/B may score: the generalization scope, in registry order.

    ``task_ids`` restricts to a named subset (each must be IN the scope; naming a
    contaminated or a training task raises, so a subset cannot quietly widen the
    claim). ``None`` means the whole scope.
    """
    from kore.tasks.registry import ContaminatedGeneralizationError, generalization_tasks

    scope = generalization_tasks()
    if task_ids is None:
        return scope
    wanted = [str(t).strip() for t in task_ids if str(t).strip()]
    by_id = {task.task_id: task for task in scope}
    missing = [t for t in wanted if t not in by_id]
    if missing:
        raise ContaminatedGeneralizationError(
            f"tasks are not in the generalization scope: {missing}; the scope is "
            "the held-out reservation minus the contaminated tasks "
            "(kore.tasks.registry.generalization_tasks)"
        )
    return [by_id[t] for t in wanted]


def scope_report(tasks: Sequence) -> dict:
    """Auditable record of the scored scope and what was excluded from it."""
    from kore.tasks.registry import (
        analysis_family,
        generalization_claim_report,
        heldout_tasks,
    )

    claim = generalization_claim_report()
    families: dict[str, list[str]] = {}
    for task in tasks:
        families.setdefault(analysis_family(task), []).append(task.task_id)
    return {
        "taxonomy_version": claim["taxonomy_version"],
        "taxonomy_digest": claim["taxonomy_digest"],
        "n_heldout_reservation": len(heldout_tasks()),
        "n_generalization_scope": claim["scoreable"],
        "n_scored": len(tasks),
        "scored_task_ids": [t.task_id for t in tasks],
        "excluded_contaminated": dict(sorted(claim["excluded_task_ids"].items())),
        "contamination_evidence": dict(claim.get("contamination_evidence", {}) or {}),
        "families": {k: sorted(v) for k, v in sorted(families.items())},
    }


# --------------------------------------------------------------------------- #
# Phase A: prompts + generation (identical prompt for every arm).
# --------------------------------------------------------------------------- #
def first_turn_messages(task, system_prompt: Optional[str] = None) -> list[dict]:
    """The single-turn chat transcript both arms are prompted with.

    This is exactly what :func:`kore.eval.policies.model_policy` sends on its
    first turn (``build_transcript`` over ``policies.task_prompt`` with no prior
    turns), so a completion recorded here is the completion the live bake-off
    would have obtained.
    """
    from kore.eval.policies import task_prompt
    from kore.policy.format import SYSTEM_PROMPT, build_transcript

    return build_transcript(
        task_prompt(task), turns=[], system_prompt=system_prompt or SYSTEM_PROMPT
    )


def prompt_digest(messages: Sequence[dict]) -> str:
    """Stable 12-hex digest of a rendered transcript (arm-equality evidence)."""
    payload = json.dumps(
        [{"role": m.get("role"), "content": m.get("content")} for m in messages],
        sort_keys=True,
        ensure_ascii=False,
    )
    return _sha12(payload)


@dataclass
class GenerationRecord:
    """One completion: what the model was asked, and what it produced."""

    arm: str
    task_id: str
    sample: int
    prompt_sha: str
    response: str
    kernel: str
    contract_ok: bool
    response_chars: int
    gen_seconds: float
    max_tokens: int
    temperature: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def submitted_source(self) -> str:
        """What the bake-off actually benches for this completion.

        Mirrors :func:`kore.eval.policies.model_policy` exactly: the parsed
        ``FULL_KERNEL`` when the model honored the contract, otherwise the raw
        response. A model that cannot produce the contract therefore submits
        prose, fails to compile, and loses the task - which is the honest
        outcome, not a harness error.
        """
        return self.kernel or self.response


def generate_arm(
    tasks: Sequence,
    generate: Optional[GenerateFn] = None,
    *,
    arm: str,
    samples: int = 1,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
    generate_batch: Optional[GenerateBatchFn] = None,
    batch_size: int = 8,
    log: Optional[Callable[[str], None]] = None,
) -> list[GenerationRecord]:
    """Generate ``samples`` completions per task for one arm.

    Supply ``generate`` (one call per completion) and/or ``generate_batch`` (one
    call per chunk of ``batch_size``); the batched path is used when present
    because a 14B model decoding 34 prompts one at a time is dominated by
    per-request latency. A backend failure is captured per record rather than
    aborting the arm, so a single bad completion cannot lose the other 33.
    """
    from kore.policy.format import parse_response

    if generate is None and generate_batch is None:
        raise ValueError("generate_arm needs generate= and/or generate_batch=")

    jobs: list[tuple] = []
    for task in tasks:
        messages = first_turn_messages(task, system_prompt)
        sha = prompt_digest(messages)
        for s in range(max(1, int(samples))):
            jobs.append((_task_id(task), s, messages, sha))

    out: list[GenerationRecord] = []

    def _record(tid, s, sha, text, seconds, error=None) -> GenerationRecord:
        parsed = parse_response(text or "")
        kernel = (parsed.get("kernel") or "").strip()
        return GenerationRecord(
            arm=arm, task_id=tid, sample=s, prompt_sha=sha,
            response=text or "", kernel=kernel, contract_ok=bool(kernel),
            response_chars=len(text or ""), gen_seconds=round(float(seconds), 3),
            max_tokens=int(max_tokens), temperature=float(temperature), error=error,
        )

    if generate_batch is not None:
        for start in range(0, len(jobs), max(1, int(batch_size))):
            chunk = jobs[start:start + max(1, int(batch_size))]
            t0 = time.perf_counter()
            try:
                texts = list(generate_batch(
                    [m for _, _, m, _ in chunk],
                    max_tokens=max_tokens, temperature=temperature,
                ))
                err = None
            except Exception as exc:  # noqa: BLE001 - a bad chunk must not lose the arm
                texts, err = [""] * len(chunk), f"{type(exc).__name__}: {str(exc)[:400]}"
            elapsed = (time.perf_counter() - t0) / max(1, len(chunk))
            if len(texts) < len(chunk):
                texts = list(texts) + [""] * (len(chunk) - len(texts))
            for (tid, s, _m, sha), text in zip(chunk, texts):
                out.append(_record(tid, s, sha, text, elapsed, err))
            if log:
                log(f"[{arm}] generated {len(out)}/{len(jobs)} "
                    f"({elapsed:.1f}s/completion in this chunk)")
        return out

    for tid, s, messages, sha in jobs:
        t0 = time.perf_counter()
        try:
            text = generate(messages, max_tokens=max_tokens, temperature=temperature)
            err = None
        except TypeError:  # a stub that does not accept the sampling kwargs
            text, err = generate(messages), None
        except Exception as exc:  # noqa: BLE001
            text, err = "", f"{type(exc).__name__}: {str(exc)[:400]}"
        rec = _record(tid, s, sha, text, time.perf_counter() - t0, err)
        out.append(rec)
        if log:
            log(f"[{arm}] {tid} sample={s} {rec.response_chars} chars "
                f"contract_ok={rec.contract_ok} in {rec.gen_seconds:.1f}s")
    return out


def write_generations(path, records: Sequence[GenerationRecord], **meta) -> Path:
    """Write a generations JSONL: one metadata header line, then one row each."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema": SCHEMA_GENERATIONS,
                             "n_records": len(records), **meta}) + "\n")
        for rec in records:
            row = rec.to_dict() if isinstance(rec, GenerationRecord) else dict(rec)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def read_generations(path) -> tuple[dict, list[dict]]:
    """Read a generations JSONL written by :func:`write_generations`."""
    rows: list[dict] = []
    meta: dict = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if i == 0 and obj.get("schema") == SCHEMA_GENERATIONS:
                meta = obj
                continue
            rows.append(obj)
    return meta, rows


def assert_prompts_matched(arms: dict) -> dict:
    """Fail loudly unless every arm saw a byte-identical prompt per task.

    ``arms`` maps arm name -> generation rows. A matched-budget claim is
    meaningless if the two arms were prompted differently, and the prompt digest
    is recorded precisely so that this is checkable rather than assumed.
    """
    per_arm: dict[str, dict[str, str]] = {}
    for arm, rows in arms.items():
        shas: dict[str, str] = {}
        for row in rows:
            tid = row["task_id"]
            sha = row["prompt_sha"]
            if shas.setdefault(tid, sha) != sha:
                raise AssertionError(
                    f"arm {arm!r} used two different prompts for task {tid!r}")
        per_arm[arm] = shas
    names = list(per_arm)
    if len(names) > 1:
        ref = per_arm[names[0]]
        for other in names[1:]:
            shared = set(ref) & set(per_arm[other])
            bad = sorted(t for t in shared if ref[t] != per_arm[other][t])
            if bad:
                raise AssertionError(
                    f"arms {names[0]!r} and {other!r} were prompted differently "
                    f"on {len(bad)} task(s): {bad[:5]}")
    return {arm: dict(sorted(shas.items())) for arm, shas in per_arm.items()}


# --------------------------------------------------------------------------- #
# Phase B: measurement through the campaign's own bake-off.
# --------------------------------------------------------------------------- #
def replay_policy(rows: Sequence[dict], *, arm: Optional[str] = None):
    """A ``PolicyFn`` that replays recorded completions in sample order.

    Submits exactly what :func:`kore.eval.policies.model_policy` would submit -
    the parsed ``FULL_KERNEL``, else the raw response - so the measured object is
    the model's own output and not a repaired version of it.
    """
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        if arm is not None and row.get("arm") not in (None, arm):
            continue
        by_task.setdefault(row["task_id"], []).append(row)
    for tid in by_task:
        by_task[tid].sort(key=lambda r: int(r.get("sample", 0)))
    cursor: dict[str, int] = {}

    def policy(task, feedback: Optional[dict] = None) -> str:
        tid = _task_id(task)
        samples = by_task.get(tid)
        if not samples:
            raise KeyError(
                f"no recorded completion for task {tid!r}"
                + (f" in arm {arm!r}" if arm else "")
                + "; generate the arm before measuring it")
        i = cursor.get(tid, 0)
        cursor[tid] = i + 1
        row = samples[min(i, len(samples) - 1)]
        return (row.get("kernel") or "").strip() or (row.get("response") or "")

    return policy


class _RecordingEnv:
    """Delegates to a real env and keeps every ``Observation`` it produced.

    ``evaluate_policy`` returns the reward-level verdict; the raw observation
    carries *why* (compiled at all? infra fault? which baseline was actually
    timed? was the timing publication-grade?), which is the difference between
    reporting a number and being able to defend it.
    """

    def __init__(self, env, sink: list, task_id: str):
        self._env = env
        self._sink = sink
        self._task_id = task_id

    def step(self, source: str, *a, **kw):
        obs = self._env.step(source, *a, **kw)
        self._sink.append((self._task_id, obs))
        return obs

    def __getattr__(self, name):  # pragma: no cover - passthrough for env extras
        return getattr(self._env, name)


def _error_tail(text: Optional[str], limit: int = 400) -> Optional[str]:
    """The DIAGNOSTIC end of an error string, not its beginning.

    ``Observation.error_text`` is usually a Python traceback, whose informative
    part - the exception type and message - is the LAST line. Truncating from the
    front yields a few frames of harness internals and hides the actual failure,
    which is exactly what happened when this eval first tried to explain why a
    kernel did not build.
    """
    if not text:
        return None
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def observation_summary(task_id: str, obs) -> dict:
    """The defensible subset of an ``Observation``, JSON-safe."""
    err = getattr(obs, "error_text", None)
    return {
        "task_id": task_id,
        "compiled": bool(getattr(obs, "compiled", False)),
        "validation_passed": bool(getattr(obs, "validation_passed", False)),
        "snr_db": getattr(obs, "snr_db", None),
        "wall_ms": getattr(obs, "wall_ms", None),
        "baseline_ms": getattr(obs, "baseline_ms", None),
        "baseline_impl": getattr(obs, "baseline_impl", None),
        "timing_grade": getattr(obs, "timing_grade", None),
        "cv_pct": getattr(obs, "cv_pct", None),
        "infra_error": bool(getattr(obs, "infra_error", False)),
        "flagged_hack": bool(getattr(obs, "flagged_hack", False)),
        "error_tail": _error_tail(err),
    }


def measure_arm(
    rows: Sequence[dict],
    tasks: Sequence,
    *,
    arm: str,
    env_factory: Callable[[object], object],
    budget: int = 1,
    mode: str = "parallel",
    ps: Sequence[float] = DEFAULT_PS,
    cfg=None,
) -> dict:
    """Score one arm's recorded completions on hardware, via the real bake-off.

    Returns the :func:`kore.eval.bakeoff.evaluate_policy` result with the raw
    observations attached under ``observations``.
    """
    from kore.config import CONFIG
    from kore.eval.bakeoff import evaluate_policy

    sink: list = []

    def recording_factory(task):
        return _RecordingEnv(env_factory(task), sink, _task_id(task))

    res = evaluate_policy(
        replay_policy(rows, arm=arm), tasks, env_factory=recording_factory,
        budget=budget, mode=mode, ps=ps, cfg=cfg if cfg is not None else CONFIG,
    )
    res["arm"] = arm
    res["observations"] = [observation_summary(tid, obs) for tid, obs in sink]
    return res


# --------------------------------------------------------------------------- #
# Phase C: scoring + the paired comparison (PURE - no GPU, no network).
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = 1.96) -> dict:
    """Wilson score interval for a binomial rate ``k/n``.

    Wilson rather than the normal approximation because these rates are small
    and ``n`` is a few dozen, exactly where ``p +/- z*sqrt(p(1-p)/n)`` produces
    intervals that run below 0 and understate the uncertainty at ``k == 0``.
    """
    n = int(n)
    k = max(0, min(int(k), n))
    if n <= 0:
        return {"k": 0, "n": 0, "rate": 0.0, "lo": 0.0, "hi": 0.0, "z": z}
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {"k": k, "n": n, "rate": p,
            "lo": max(0.0, centre - half), "hi": min(1.0, centre + half), "z": z}


def _fast_p_map(res: dict) -> dict:
    return {float(p): float(v) for p, v in (res.get("fast_p", {}) or {}).items()}


def arm_summary(measure_result: dict, generation_rows: Sequence[dict],
                *, arm: Optional[str] = None) -> dict:
    """Fold one arm's generations + measurements into per-task and split metrics.

    The three binary outcomes form a strict funnel - contract honored, then
    compiled, then correct - which is what makes an unflattering result
    interpretable instead of just small: a model that never emits a parseable
    ``FULL_KERNEL`` scores 0 correct for a reason that has nothing to do with
    kernel skill, and the funnel says so.
    """
    arm = arm or measure_result.get("arm") or ""
    gen_by_task: dict[str, dict] = {}
    for row in generation_rows:
        if row.get("arm") not in (None, arm):
            continue
        gen_by_task.setdefault(row["task_id"], row)
    obs_by_task: dict[str, dict] = {}
    for o in measure_result.get("observations", []) or []:
        obs_by_task.setdefault(o["task_id"], o)

    per_task: list[dict] = []
    for rec in measure_result.get("per_task", []) or []:
        tid = rec.get("task_id")
        gen = gen_by_task.get(tid, {})
        obs = obs_by_task.get(tid, {})
        traj = (rec.get("trajectory") or [{}])[0]
        # ``correct`` is the reward's own SNR/validation gate; ``timed`` adds the
        # requirement that the bench produced an integrity-gated speedup, which
        # is what fast_p scores. A correct-but-screening-graded kernel is a
        # verifier PASS with no defensible timing, and must not be filed as a
        # correctness failure (see kore.eval.bakeoff._run_task).
        per_task.append({
            "task_id": tid,
            "contract_ok": bool(gen.get("contract_ok")),
            "response_chars": gen.get("response_chars"),
            "compiled": bool(obs.get("compiled")),
            "infra_error": bool(obs.get("infra_error")),
            "correct": bool(rec.get("correct_gate", traj.get("correct"))),
            "timed": bool(rec.get("timed", rec.get("correct"))),
            "speedup_gated": rec.get("best_speedup"),
            "speedup_raw": traj.get("speedup"),
            "snr_db": obs.get("snr_db"),
            "baseline_impl": obs.get("baseline_impl"),
            "timing_grade": obs.get("timing_grade"),
            "flags": traj.get("flags") or [],
            "error_tail": obs.get("error_tail"),
        })
    for t in per_task:
        t["faster"] = bool(t["timed"] and (t["speedup_gated"] or 0.0) > 1.0)

    n = len(per_task)
    counts = {
        "contract_ok": sum(1 for t in per_task if t["contract_ok"]),
        "compiled": sum(1 for t in per_task if t["compiled"]),
        "correct": sum(1 for t in per_task if t["correct"]),
        "timed": sum(1 for t in per_task if t["timed"]),
        "faster": sum(1 for t in per_task if t["faster"]),
        "infra_error": sum(1 for t in per_task if t["infra_error"]),
    }
    return {
        "arm": arm,
        "n": n,
        "budget": measure_result.get("budget"),
        "mode": measure_result.get("mode"),
        "counts": counts,
        "rates": {k: wilson_interval(v, n) for k, v in counts.items()},
        "fast_p": _fast_p_map(measure_result),
        "geometric_mean_speedup": measure_result.get("geometric_mean_speedup"),
        "median_response_chars": _median(
            [t["response_chars"] for t in per_task if t["response_chars"] is not None]),
        "per_task": per_task,
    }


def _median(values: Sequence[float]):
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _paired(a_summary: dict, b_summary: dict, key: str) -> tuple[list, list, list[str]]:
    """Aligned per-task values for ``key`` over the tasks BOTH arms attempted."""
    a_by = {t["task_id"]: t for t in a_summary.get("per_task", [])}
    b_by = {t["task_id"]: t for t in b_summary.get("per_task", [])}
    ids = [t for t in (x["task_id"] for x in a_summary.get("per_task", [])) if t in b_by]
    return ([a_by[t][key] for t in ids], [b_by[t][key] for t in ids], ids)


def compare_arms(a_summary: dict, b_summary: dict, *,
                 name_a: Optional[str] = None, name_b: Optional[str] = None,
                 n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired comparison of two arms scored on the same tasks.

    ``a`` is the CANDIDATE (the trained checkpoint) and ``b`` the REFERENCE (its
    base), so a positive effect means the training helped.

    * binary outcomes (contract / compile / correct / faster-than-baseline): the
      per-task delta in ``{-1, 0, +1}``, reported as a mean-delta with a paired
      bootstrap CI and an exact sign test - which on paired binary data is
      exactly McNemar's exact test, with the discordant-pair count reported so
      the reader can see how much evidence there actually is.
    * speedup: the geometric-mean ratio over the tasks where BOTH arms produced
      a correct kernel (the only tasks on which a speedup is comparable), with an
      exponentiated bootstrap CI.
    """
    from kore.eval.paired_stats import (
        paired_comparison,
        paired_speedup_comparison,
        sign_test,
    )

    name_a = name_a or a_summary.get("arm") or "candidate"
    name_b = name_b or b_summary.get("arm") or "base"

    binary: dict[str, dict] = {}
    for key in ("contract_ok", "compiled", "correct", "timed", "faster"):
        a_vals, b_vals, ids = _paired(a_summary, b_summary, key)
        deltas = [float(bool(x)) - float(bool(y)) for x, y in zip(a_vals, b_vals)]
        cmp_ = paired_comparison(deltas=deltas, n_boot=n_boot, seed=seed,
                                 alpha=alpha, headline="sign")
        sgn = sign_test(deltas)
        binary[key] = {
            **cmp_.to_dict(),
            "n_paired": len(ids),
            "a_count": sum(1 for v in a_vals if v),
            "b_count": sum(1 for v in b_vals if v),
            "discordant_pairs": sgn.n_effective,
            "a_only": sgn.n_pos,
            "b_only": sgn.n_neg,
            "test": "exact McNemar (sign test on paired binary deltas)",
        }

    a_su, b_su, ids = _paired(a_summary, b_summary, "speedup_gated")
    both = [i for i, (sa, sb) in enumerate(zip(a_su, b_su))
            if sa and sb and sa > 0 and sb > 0]
    speedup: Optional[dict] = None
    if both:
        cmp_su = paired_speedup_comparison(
            [a_su[i] for i in both], [b_su[i] for i in both],
            n_boot=n_boot, seed=seed, alpha=alpha)
        speedup = {**cmp_su.to_dict(),
                   "task_ids": [ids[i] for i in both],
                   "note": ("geometric-mean ratio of the integrity-gated speedups on "
                            "the tasks where BOTH arms produced a correct AND "
                            "publication-timed kernel")}

    fp_a, fp_b = a_summary.get("fast_p", {}), b_summary.get("fast_p", {})
    fast_p_delta = {p: float(fp_a.get(p, 0.0)) - float(fp_b.get(p, 0.0))
                    for p in sorted(set(fp_a) | set(fp_b))}

    headline = binary["correct"]
    verdict = _verdict(headline, speedup, alpha)
    return {
        "schema": SCHEMA_REPORT,
        "candidate": name_a,
        "reference": name_b,
        "n_paired": len(ids),
        "alpha": alpha,
        "binary": binary,
        "speedup": speedup,
        "fast_p": {"candidate": fp_a, "reference": fp_b, "delta": fast_p_delta},
        "counts": {"candidate": a_summary.get("counts"),
                   "reference": b_summary.get("counts")},
        "rates": {"candidate": a_summary.get("rates"),
                  "reference": b_summary.get("rates")},
        "verdict": verdict,
    }


def _verdict(correct_cmp: dict, speedup: Optional[dict], alpha: float) -> dict:
    """The honest headline: direction, significance, and why - including 'neither'.

    A comparison in which NEITHER arm ever produced a correct kernel is not a
    tie between two equally good models; it is an absence of signal, and it must
    not be reported as "no regression".
    """
    a_n, b_n = correct_cmp["a_count"], correct_cmp["b_count"]
    disc = correct_cmp["discordant_pairs"]
    p = correct_cmp["p_sign"]
    if a_n == 0 and b_n == 0:
        return {
            "direction": "no_signal",
            "significant": False,
            "p_value": 1.0,
            "statement": ("neither arm produced a single correct kernel on this "
                          "scope at this budget, so the comparison has no signal "
                          "to resolve - this is NOT evidence of parity"),
        }
    if disc == 0:
        return {
            "direction": "tie",
            "significant": False,
            "p_value": 1.0,
            "statement": (f"both arms were correct on exactly the same tasks "
                          f"({a_n}/{correct_cmp['n_paired']}); zero discordant "
                          "pairs, so no paired test can separate them"),
        }
    direction = "candidate_better" if a_n > b_n else (
        "reference_better" if b_n > a_n else "tie")
    sig = p < alpha
    stmt = (f"correctness {a_n} vs {b_n} of {correct_cmp['n_paired']}; "
            f"{disc} discordant pair(s); exact McNemar p={p:.3g} "
            f"({'significant' if sig else 'NOT significant'} at alpha={alpha:g})")
    if speedup is not None:
        stmt += (f"; geomean speedup ratio {speedup['effect_size']:.3f}x "
                 f"[{speedup['ci'][0]:.3f}, {speedup['ci'][1]:.3f}] "
                 f"on {speedup['n']} both-correct task(s)")
    return {"direction": direction, "significant": sig, "p_value": p, "statement": stmt}


def build_report(a_summary: dict, b_summary: dict, *, scope: Optional[dict] = None,
                 meta: Optional[dict] = None, **compare_kwargs) -> dict:
    """A complete, self-describing A/B artifact (scope + both arms + comparison)."""
    comparison = compare_arms(a_summary, b_summary, **compare_kwargs)
    return {
        "schema": SCHEMA_REPORT,
        "meta": dict(meta or {}),
        "scope": scope or {},
        "arms": {a_summary.get("arm", "candidate"): a_summary,
                 b_summary.get("arm", "base"): b_summary},
        "comparison": comparison,
    }


def _pct(x) -> str:
    return "-" if x is None else f"{100.0 * float(x):.1f}%"


def format_report(report: dict) -> str:
    """Human-readable markdown for a :func:`build_report` artifact (ASCII only)."""
    cmp_ = report.get("comparison", {}) or {}
    arms = report.get("arms", {}) or {}
    cand_name, ref_name = cmp_.get("candidate", "candidate"), cmp_.get("reference", "base")
    cand, ref = arms.get(cand_name, {}), arms.get(ref_name, {})
    scope = report.get("scope", {}) or {}

    lines = ["# Checkpoint A/B on the held-out generalization scope", ""]
    lines.append(f"- **candidate**: `{cand_name}`")
    lines.append(f"- **reference**: `{ref_name}`")
    lines.append(f"- **tasks scored (n)**: {cmp_.get('n_paired', '?')}"
                 f" of {scope.get('n_generalization_scope', '?')} in scope"
                 f" ({scope.get('n_heldout_reservation', '?')} held out,"
                 f" {len(scope.get('excluded_contaminated', {}) or {})} excluded as contaminated)")
    lines.append(f"- **budget**: {cand.get('budget', '?')} bench(es)/task,"
                 f" mode={cand.get('mode', '?')} (matched)")
    lines.append("")

    lines.append("## Funnel (per-task rate, Wilson 95% CI)")
    lines.append("")
    lines.append(f"| stage | {cand_name} | {ref_name} | delta | discordant | exact McNemar p |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key, label in (("contract_ok", "emitted FULL_KERNEL"),
                       ("compiled", "compiled"),
                       ("correct", "correct (SNR gate)"),
                       ("timed", "correct AND publication-timed"),
                       ("faster", "faster than its baseline")):
        b = (cmp_.get("binary", {}) or {}).get(key, {})
        cr = (cand.get("rates", {}) or {}).get(key, {})
        rr = (ref.get("rates", {}) or {}).get(key, {})
        lines.append(
            f"| {label} | {cr.get('k', '?')}/{cr.get('n', '?')} = {_pct(cr.get('rate'))}"
            f" [{_pct(cr.get('lo'))}, {_pct(cr.get('hi'))}]"
            f" | {rr.get('k', '?')}/{rr.get('n', '?')} = {_pct(rr.get('rate'))}"
            f" [{_pct(rr.get('lo'))}, {_pct(rr.get('hi'))}]"
            f" | {_pct(b.get('effect_size'))}"
            f" | {b.get('discordant_pairs', '?')}"
            f" ({b.get('a_only', '?')}/{b.get('b_only', '?')})"
            f" | {b.get('p_sign', float('nan')):.3g} |")
    lines.append("")
    # An infrastructure fault (OOM, timeout, driver fault) is a statement about
    # the node, not about the kernel, but the reward gate scores it as incorrect.
    # Surface the count so a funnel drop cannot be silently read as a model
    # failure (kore/tasks/README.md makes the same distinction for the corpus
    # verification sweep).
    infra_c = (cand.get("counts", {}) or {}).get("infra_error", 0)
    infra_r = (ref.get("counts", {}) or {}).get("infra_error", 0)
    if infra_c or infra_r:
        lines.append(f"> **Infrastructure faults** (counted as failures above, but a "
                     f"statement about the node rather than the kernel): "
                     f"{cand_name} {infra_c}, {ref_name} {infra_r}.")
        lines.append("")

    fp = cmp_.get("fast_p", {}) or {}
    if fp.get("candidate") or fp.get("reference"):
        lines.append("## fast_p (fraction of the whole scope, uncorrected denominator)")
        lines.append("")
        lines.append(f"| p | {cand_name} | {ref_name} | delta |")
        lines.append("| --- | --- | --- | --- |")
        for p in sorted(set(fp.get("candidate", {})) | set(fp.get("reference", {}))):
            lines.append(f"| {float(p):g} | {_pct(fp['candidate'].get(p))}"
                         f" | {_pct(fp['reference'].get(p))}"
                         f" | {_pct((fp.get('delta') or {}).get(p))} |")
        lines.append("")

    su = cmp_.get("speedup")
    if su:
        lines.append("## Speedup on both-correct tasks")
        lines.append("")
        lines.append(f"- geometric-mean ratio ({cand_name} / {ref_name}): "
                     f"**{su['effect_size']:.3f}x** "
                     f"95% CI [{su['ci'][0]:.3f}, {su['ci'][1]:.3f}] on n={su['n']}")
        lines.append(f"- p (wilcoxon) = {su['p_wilcoxon']:.3g}, "
                     f"p (sign) = {su['p_sign']:.3g}")
        lines.append("")

    v = cmp_.get("verdict", {}) or {}
    lines.append(f"**Verdict: {v.get('direction', '?')}** - {v.get('statement', '')}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# A batched HuggingFace backend (the generation side of a real run).
# --------------------------------------------------------------------------- #
def load_hf_batch_generate(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    enable_thinking: bool = False,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """Load ``model_id`` and return ``(generate, generate_batch, close, info)``.

    Both arms of an A/B must decode under identical conditions, so this pins the
    things that would otherwise differ between a Hub base model and a local
    training output:

      * ``dtype`` is applied to BOTH arms. A full-finetune checkpoint saved
        through FSDP ``FULL_STATE_DICT`` under bf16 mixed precision lands on disk
        as fp32 (55 GiB for 14B - see ``docs/SFT_READINESS.md`` F1) while the Hub
        base is bf16. Loading both at bf16 removes a precision confound that has
        nothing to do with what was trained.
      * ``enable_thinking=False`` matches
        :func:`kore.policy.serve.load_generate`: with the Qwen3 template's
        thinking on, a bounded token budget is spent inside ``<think>`` and the
        kernel never arrives, which reads as a capability difference and is a
        budget artifact.
      * LEFT padding, because a batched decoder-only ``generate`` with right
        padding continues from pad tokens and produces garbage for every
        sequence shorter than the longest one in its batch.
    """
    import torch  # guarded heavy import
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    rev = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, **rev)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device_map,
        trust_remote_code=trust_remote_code, **rev)
    model.eval()

    def _render(messages) -> str:
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        except TypeError:  # template without the kwarg (non-Qwen)
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def generate_batch(messages_batch, max_tokens: int = 4096,
                       temperature: float = 0.0) -> list:
        texts = [_render(m) for m in messages_batch]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        do_sample = bool(temperature and temperature > 0.0)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=None if not do_sample else 1.0,
                pad_token_id=tok.pad_token_id)
        prompt_len = enc["input_ids"].shape[1]
        return [tok.decode(row[prompt_len:], skip_special_tokens=True) for row in out]

    def generate(messages, max_tokens: int = 4096, temperature: float = 0.0) -> str:
        return generate_batch([messages], max_tokens=max_tokens,
                              temperature=temperature)[0]

    def nll(text: str) -> dict:
        """Teacher-forced next-token NLL over raw ``text`` (the pretraining objective).

        Shares the loaded weights with generation so an A/B pays ONE load per arm
        rather than one per measurement. See :mod:`kore.eval.heldout_lm` for what
        the number is for and how it is compared.
        """
        import hashlib as _hashlib

        ids = tok(text, return_tensors="pt", truncation=True, max_length=8192,
                  add_special_tokens=False)["input_ids"]
        if ids.shape[1] < 2:
            return {"sum_nll_nats": 0.0, "n_tokens": 0, "tokens_sha": None}
        ids = ids.to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        # fp32 log-softmax: a bf16 reduction over a 151k-way vocabulary loses
        # precision at the 1e-3 bits/token scale this comparison must resolve.
        logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        target = ids[:, 1:]
        picked = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        sha = _hashlib.sha256(
            ",".join(str(i) for i in ids[0].tolist()).encode("utf-8")
        ).hexdigest()[:12]
        return {"sum_nll_nats": float(-picked.sum().item()),
                "n_tokens": int(target.numel()), "tokens_sha": sha}

    def close() -> None:
        import gc
        nonlocal model, tok
        model = None
        tok = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass

    info = {
        "model_id": model_id,
        "dtype": dtype,
        "revision": revision,
        "enable_thinking": enable_thinking,
        "padding_side": "left",
        "config_dtype": str(getattr(model.config, "dtype",
                                    getattr(model.config, "torch_dtype", None))),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "vocab_size": int(getattr(model.config, "vocab_size", 0)),
        "device": str(model.device),
    }
    return {"generate": generate, "generate_batch": generate_batch, "nll": nll,
            "close": close, "info": info}


def endpoint_generate(base_url: str, model: str = "", *,
                      api_key: Optional[str] = None, timeout: float = 900.0):
    """A ``generate(messages, ...)`` against an OpenAI-compatible chat endpoint.

    The endpoint route (SGLang / vLLM in a container - see
    ``docs/E2E_SERVING_GATE.md``) keeps the engine's torch pin out of the
    training venv. Unlike
    :func:`kore.eval.e2e_sglang_vllm._openai_compatible_generate` this sends a
    full MESSAGE LIST rather than a single user prompt, because the KORE policy
    contract is a system prompt plus a task turn and flattening it would change
    what the model is asked.
    """
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"

    def generate(messages, max_tokens: int = 4096, temperature: float = 0.0) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        payload = {
            "model": model or "default",
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = json.dumps(payload).encode("utf-8")
        try:
            import requests

            resp = requests.post(endpoint, data=data, headers=headers, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
        except ImportError:  # pragma: no cover - stdlib fallback
            import urllib.request

            req = urllib.request.Request(endpoint, data=data, headers=headers,
                                         method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
        msg = body["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or ""

    return generate


# --------------------------------------------------------------------------- #
# CLI: generate -> measure -> compare, as three resumable steps.
# --------------------------------------------------------------------------- #
def _stderr(msg: str) -> None:
    import sys

    print(msg, file=sys.stderr, flush=True)


def _resolve_backend(args) -> dict:
    """Return ``{generate, generate_batch, nll, close, info}`` for one arm.

    ``nll`` is only available for a locally loaded model: a served endpoint
    exposes sampling, not per-token log-likelihood, so the held-out LM-loss
    measurement is skipped rather than faked when the backend is HTTP.
    """
    if args.backend == "endpoint":
        if not args.base_url:
            raise SystemExit("--backend endpoint needs --base-url")
        return {"generate": endpoint_generate(args.base_url, args.served_model),
                "generate_batch": None, "nll": None, "close": (lambda: None),
                "info": {"backend": "endpoint", "base_url": args.base_url,
                         "served_model": args.served_model}}
    if args.backend == "hf-batch":
        backend = load_hf_batch_generate(
            args.model, dtype=args.dtype, revision=args.revision,
            device_map=getattr(args, "device_map", "auto"))
        backend["info"]["backend"] = "hf-batch"
        return backend
    if args.backend == "serve":
        from kore.policy.serve import load_generate

        client = load_generate(args.model, backend="hf", revision=args.revision)
        return {"generate": client, "generate_batch": None, "nll": None,
                "close": client.close,
                "info": {"backend": "serve", "model": args.model}}
    raise SystemExit(f"unknown backend {args.backend!r}")


def _cmd_generate(args) -> int:
    tasks = generalization_scope(
        [t for t in (args.tasks or "").split(",") if t.strip()] or None)
    if args.limit:
        tasks = tasks[: args.limit]
    backend = _resolve_backend(args)
    info = backend["info"]
    try:
        t0 = time.perf_counter()
        records = generate_arm(
            tasks, backend["generate"], arm=args.arm, samples=args.samples,
            max_tokens=args.max_tokens, temperature=args.temperature,
            generate_batch=backend["generate_batch"], batch_size=args.batch_size,
            log=_stderr)
        elapsed = time.perf_counter() - t0
    finally:
        backend["close"]()
    path = write_generations(
        args.out, records, arm=args.arm, backend=info, samples=args.samples,
        max_tokens=args.max_tokens, temperature=args.temperature,
        scope=scope_report(tasks), wall_seconds=round(elapsed, 1))
    ok = sum(1 for r in records if r.contract_ok)
    _stderr(f"[checkpoint_ab] {args.arm}: {len(records)} completions, "
            f"{ok} honored the FULL_KERNEL contract, {elapsed / 60.0:.1f} min -> {path}")
    return 0


def _cmd_run_arm(args) -> int:
    """Everything one arm needs from a loaded model, in ONE load.

    A 14B checkpoint costs minutes to read off shared storage, so generation,
    held-out LM loss and the retention smoke are done back to back rather than in
    three jobs that each pay the load. No GPU timing happens here, so nothing in
    this step is sensitive to sharing the node with the other arm's load.
    """
    from kore.eval import heldout_lm as hl

    tasks = generalization_scope(
        [t for t in (args.tasks or "").split(",") if t.strip()] or None)
    if args.limit:
        tasks = tasks[: args.limit]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    backend = _resolve_backend(args)
    info = backend["info"]
    gen = backend["generate"]
    try:
        _stderr(f"[checkpoint_ab] arm={args.arm} backend={info}")
        (outdir / f"backend_{args.arm}.json").write_text(
            json.dumps(info, indent=2, default=str))
        # LM loss FIRST: it is the cheapest measurement and the one with the most
        # statistical power, so a job that runs out of wall clock during the slow
        # generation pass still leaves a usable result behind.
        if args.lm_loss and backend["nll"] is None:
            _stderr("[checkpoint_ab] held-out LM loss needs per-token "
                    "log-likelihood, which an HTTP endpoint does not expose; "
                    "skipping (use --backend hf-batch)")
        elif args.lm_loss:
            docs = hl.heldout_documents(tasks)
            hl.assert_documents_uncontaminated(docs)
            docs = docs + hl.heldout_documents(tasks, kinds=("reference",)) \
                + hl.general_documents()
            t1 = time.perf_counter()
            rows = hl.score_documents(docs, backend["nll"], arm=args.arm, log=_stderr)
            path = hl.write_scores(
                outdir / f"lm_scores_{args.arm}.json", rows, arm=args.arm,
                model=info, wall_seconds=round(time.perf_counter() - t1, 1))
            for kind in ("seed", "reference", "general"):
                sub = [r for r in rows if r["kind"] == kind]
                if sub:
                    tot = hl.corpus_totals(sub)
                    _stderr(f"[checkpoint_ab] {args.arm}: {kind} bits/token = "
                            f"{tot['bits_per_token']:.4f} over {tot['n_documents']} "
                            f"docs / {tot['n_tokens']} tokens")
            _stderr(f"[checkpoint_ab] {args.arm}: LM scores -> {path}")

        t0 = time.perf_counter()
        records = generate_arm(
            tasks, gen, arm=args.arm, samples=args.samples,
            max_tokens=args.max_tokens, temperature=args.temperature,
            generate_batch=backend["generate_batch"], batch_size=args.batch_size,
            log=_stderr)
        gen_wall = time.perf_counter() - t0
        gen_path = write_generations(
            outdir / f"generations_{args.arm}.jsonl", records, arm=args.arm,
            backend=info, samples=args.samples, max_tokens=args.max_tokens,
            temperature=args.temperature, scope=scope_report(tasks),
            wall_seconds=round(gen_wall, 1))
        ok = sum(1 for r in records if r.contract_ok)
        _stderr(f"[checkpoint_ab] {args.arm}: {ok}/{len(records)} honored the "
                f"FULL_KERNEL contract ({gen_wall / 60.0:.1f} min) -> {gen_path}")

        if args.retention:
            from kore.eval.retention import run_retention_suite

            def model_generate(prompt, max_tokens: int = 512,
                               temperature: float = 0.0, **_kw) -> str:
                return gen([{"role": "user", "content": prompt}],
                           max_tokens=max_tokens, temperature=temperature)

            benches = tuple(b for b in args.retention.split(",") if b.strip())
            t2 = time.perf_counter()
            res = run_retention_suite(model_generate, benches=benches)
            res["arm"] = args.arm
            res["wall_seconds"] = round(time.perf_counter() - t2, 1)
            (outdir / f"retention_{args.arm}.json").write_text(
                json.dumps(res, indent=2, default=str))
            _stderr(f"[checkpoint_ab] {args.arm}: retention {res['scores']}")
    finally:
        backend["close"]()
    return 0


def _cmd_measure(args) -> int:
    from kore.env.kore_env import KoreEnv

    meta, rows = read_generations(args.generations)
    arm = args.arm or meta.get("arm") or (rows[0].get("arm") if rows else "")
    task_ids = sorted({r["task_id"] for r in rows})
    tasks = generalization_scope(task_ids)
    if args.limit:
        tasks = tasks[: args.limit]
    res = measure_arm(rows, tasks, arm=arm,
                      env_factory=lambda t: KoreEnv(t, use_replay=False),
                      budget=args.budget, mode=args.mode)
    summary = arm_summary(res, rows, arm=arm)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"schema": SCHEMA_MEASURES, "arm": arm, "generations_meta": meta,
         "summary": summary}, indent=2, default=str))
    c = summary["counts"]
    _stderr(f"[checkpoint_ab] {arm}: contract {c['contract_ok']}/{summary['n']}, "
            f"compiled {c['compiled']}, correct {c['correct']}, timed {c['timed']}, "
            f"faster {c['faster']} -> {out}")
    return 0


def _load_summary(path) -> dict:
    obj = json.loads(Path(path).read_text())
    return obj["summary"] if "summary" in obj else obj


def _cmd_compare(args) -> int:
    cand = _load_summary(args.candidate)
    ref = _load_summary(args.reference)
    tasks = generalization_scope([t["task_id"] for t in cand["per_task"]])
    report = build_report(cand, ref, scope=scope_report(tasks),
                          meta={"candidate_measures": str(args.candidate),
                                "reference_measures": str(args.reference)})
    stem = Path(args.out).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    md = format_report(report)
    stem.with_suffix(".md").write_text(md)
    print(md)
    _stderr(f"[checkpoint_ab] report -> {stem.with_suffix('.json')}")
    return 0


def main(argv: Optional[list] = None) -> int:  # pragma: no cover - CLI wiring
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m kore.eval.checkpoint_ab",
        description="A/B a trained checkpoint against its base on the held-out "
                    "generalization scope, at a matched measurement budget.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="phase A: one completion per task per arm")
    g.add_argument("--arm", required=True, help="arm label, e.g. midtrain / base")
    g.add_argument("--out", required=True, help="generations JSONL path")
    g.add_argument("--backend", default="hf-batch",
                   choices=("hf-batch", "endpoint", "serve"))
    g.add_argument("--model", default="", help="checkpoint dir or Hub id")
    g.add_argument("--revision", default=None, help="pinned Hub revision")
    g.add_argument("--dtype", default="bfloat16",
                   help="load dtype, applied identically to every arm")
    g.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    g.add_argument("--served-model", default="", help="endpoint 'model' field")
    g.add_argument("--samples", type=int, default=1)
    g.add_argument("--max-tokens", type=int, default=4096)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--tasks", default=None, help="comma-separated subset of the scope")
    g.add_argument("--limit", type=int, default=0)
    g.set_defaults(fn=_cmd_generate)

    r = sub.add_parser("run-arm",
                       help="phase A in one model load: generate + LM loss + retention")
    for opt, kw in (
        ("--arm", dict(required=True)),
        ("--outdir", dict(required=True)),
        ("--backend", dict(default="hf-batch",
                           choices=("hf-batch", "endpoint", "serve"))),
        ("--model", dict(default="")),
        ("--revision", dict(default=None)),
        ("--dtype", dict(default="bfloat16")),
        ("--device-map", dict(default="auto",
                              help="transformers device_map; 'cpu' for a "
                                   "CPU-only allocation")),
        ("--base-url", dict(default=None)),
        ("--served-model", dict(default="")),
        ("--samples", dict(type=int, default=1)),
        ("--max-tokens", dict(type=int, default=4096)),
        ("--temperature", dict(type=float, default=0.0)),
        ("--batch-size", dict(type=int, default=8)),
        ("--tasks", dict(default=None)),
        ("--limit", dict(type=int, default=0)),
        ("--retention", dict(default="mmlu,humaneval",
                             help="comma-separated retention benches, '' to skip")),
    ):
        r.add_argument(opt, **kw)
    r.add_argument("--no-lm-loss", dest="lm_loss", action="store_false",
                   help="skip the held-out LM-loss pass")
    r.set_defaults(fn=_cmd_run_arm, lm_loss=True)

    m = sub.add_parser("measure", help="phase B: verify + cold-cache bench on hardware")
    m.add_argument("generations")
    m.add_argument("--out", required=True)
    m.add_argument("--arm", default=None)
    m.add_argument("--budget", type=int, default=1)
    m.add_argument("--mode", default="parallel", choices=("parallel", "serial"))
    m.add_argument("--limit", type=int, default=0)
    m.set_defaults(fn=_cmd_measure)

    c = sub.add_parser("compare", help="phase C: paired comparison + markdown report")
    c.add_argument("--candidate", required=True, help="candidate measures JSON")
    c.add_argument("--reference", required=True, help="reference measures JSON")
    c.add_argument("--out", required=True, help="output path stem")
    c.set_defaults(fn=_cmd_compare)

    args = p.parse_args(argv)
    return args.fn(args)


__all__ = [
    "SCHEMA_GENERATIONS",
    "SCHEMA_MEASURES",
    "SCHEMA_REPORT",
    "GenerationRecord",
    "generalization_scope",
    "scope_report",
    "first_turn_messages",
    "prompt_digest",
    "generate_arm",
    "write_generations",
    "read_generations",
    "assert_prompts_matched",
    "replay_policy",
    "observation_summary",
    "measure_arm",
    "wilson_interval",
    "arm_summary",
    "compare_arms",
    "build_report",
    "format_report",
    "load_hf_batch_generate",
    "endpoint_generate",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
