"""Bake-off policies for the KORE eval (seed baseline + the *trained* model).

The matched-budget bake-off (:mod:`kore.eval.bakeoff`) scores a
``PolicyFn(task, feedback=None) -> kernel_src``. Two policies matter:

  - :func:`seed_policy`  — returns the task's frozen seed kernel. It measures the
    *starting point*, not anything training produced. Evaluating only this is the
    audited bug: the campaign reported the seed's fast_p as if it were KORE's.
  - :func:`model_policy` — wraps a *real trained checkpoint* via
    :func:`kore.policy.serve.load_generate` + the KORE response parser
    (:func:`kore.policy.format.parse_response`). It builds the multi-turn
    transcript the policy was trained on (task prompt + summarized prior-turn
    verifier feedback), calls the model, and returns the parsed ``FULL_KERNEL``
    source. This is what the bake-off must score to make a trustworthy KORE-vs-seed
    claim.

Import-safe / offline: nothing heavy is imported at module load. ``load_generate``
(and hence torch/vLLM) is imported lazily inside :func:`model_policy`, and only
when a ``generate`` callable is not injected (tests inject a stub).
"""

from __future__ import annotations

from typing import Callable, Optional

# A policy maps (task, feedback) -> kernel source (see kore.eval.bakeoff.PolicyFn).
PolicyFn = Callable[[object, Optional[dict]], str]


def seed_policy(task, feedback: Optional[dict] = None) -> str:
    """Baseline policy: return the task's verified seed kernel unchanged."""
    return task.seed_source


def _task_id(task) -> str:
    if isinstance(task, str):
        return task
    return getattr(task, "task_id", None) or str(task)


def _task_prompt(task) -> str:
    """Render the first-turn user prompt for a task (op + hardware + seed)."""
    tid = _task_id(task)
    op = getattr(task, "operation", tid)
    gpu = getattr(task, "gpu_target", "gfx942")
    dtype = getattr(task, "dtype", "")
    parts = [
        f"Optimize the {op} kernel (task '{tid}', dtype={dtype or 'n/a'}) for AMD "
        f"{gpu}. Keep it numerically correct (pass the SNR gate) and make it as "
        "fast as possible. Output the COMPLETE kernel under the FULL_KERNEL contract."
    ]
    seed = ""
    try:
        seed = task.seed_source
    except Exception:  # noqa: BLE001 - a task without a readable seed still gets a prompt
        seed = ""
    if seed:
        parts.append("Current (seed) kernel:\n```python\n" + seed.strip() + "\n```")
    return "\n\n".join(parts)


def _render_feedback(feedback: dict) -> str:
    """Render the bake-off's compact per-turn feedback dict into policy-visible text."""
    if not feedback:
        return ""
    if feedback.get("compiled") is False:
        err = (feedback.get("error_text") or "")[:800]
        return "RESULT: compile/build FAILED.\n" + (f"COMPILER ERROR:\n{err}\n" if err else "") + \
               "Fix the build error before optimizing further."
    if not feedback.get("correct"):
        lines = ["RESULT: compiled but INCORRECT (failed the SNR gate)."]
        if feedback.get("snr_db") is not None:
            lines.append(f"primary SNR: {feedback['snr_db']:.2f} dB")
        err = (feedback.get("error_text") or "")[:400]
        if err:
            lines.append(f"detail: {err}")
        lines.append("Restore numerical correctness; do not sacrifice accuracy for speed.")
        return "\n".join(lines)
    lines = ["RESULT: CORRECT (passed the SNR gate)."]
    if feedback.get("speedup") is not None:
        lines.append(f"speedup vs reference: {feedback['speedup']:.3f}x")
    lines.append("Propose ONE further optimization to improve the speedup while staying correct.")
    return "\n".join(lines)


def model_policy(
    checkpoint: str,
    *,
    backend: str = "hf",
    generate: Optional[Callable[..., str]] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    **serve_kwargs,
) -> PolicyFn:
    """Build a ``PolicyFn`` backed by a real trained checkpoint.

    The returned policy renders the multi-turn transcript (system + task prompt +
    prior assistant turns with summarized CoT + verifier feedback), calls the
    served model, and returns the parsed ``FULL_KERNEL`` source. It keeps per-task
    history in a closure so serial refinement (the bake-off's serial mode) sees the
    accumulated trajectory; a ``feedback=None`` call starts a fresh trajectory
    (first turn / parallel mode).

    ``generate`` may be injected (e.g. by tests or by the soup sweep, which already
    holds a served model) to avoid re-loading; otherwise it is obtained from
    :func:`kore.policy.serve.load_generate`.
    """
    from kore.policy.format import SYSTEM_PROMPT, build_transcript, parse_response

    gen = generate
    if gen is None:
        from kore.policy.serve import load_generate  # lazy: torch/vLLM only on real runs

        gen = load_generate(checkpoint, backend=backend, **serve_kwargs)

    sys_prompt = system_prompt or SYSTEM_PROMPT
    histories: dict[str, list[dict]] = {}

    def policy(task, feedback: Optional[dict] = None) -> str:
        tid = _task_id(task)
        turns = histories.setdefault(tid, [])
        if feedback is None:
            turns.clear()  # fresh trajectory (first serial turn or parallel sample)
        elif turns:
            turns[-1] = {**turns[-1], "feedback": _render_feedback(feedback)}

        messages = build_transcript(_task_prompt(task), turns=turns, system_prompt=sys_prompt)
        out = gen(messages, max_tokens=max_tokens, temperature=temperature)
        turns.append({"response": out})
        parsed = parse_response(out)
        return parsed.get("kernel") or out

    return policy


__all__ = ["PolicyFn", "seed_policy", "model_policy"]
