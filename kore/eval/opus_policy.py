"""Opus-as-policy: bench the frontier teacher through the SAME policy interface.

KORE's flagship claim is head-to-head: does the *specialized* trained policy beat
the *frontier generalist* (Claude Opus 4.x) at writing fast, correct gfx950/MI350X
kernels graded against the production vendor baselines? To make that comparison
FAIR, Opus must be scored through the EXACT same path KORE is - not a bespoke Opus
harness with its own prompt, its own parser, and its own timing.

This module adapts the Claude/Opus teacher (:func:`kore.data.teacher.make_teacher`,
whose ``claude`` backend defaults to ``claude-opus-4.8`` via AMD's internal LLM
gateway) into a ``PolicyFn(task, feedback) -> kernel_source`` - the identical
interface :func:`kore.eval.bakeoff.evaluate_policy` consumes and the identical
object :func:`kore.eval.policies.model_policy` returns. Concretely Opus is built
BY ``model_policy`` with the teacher's ``generate`` injected as the token source,
so it shares:

  * the prompt contract (``build_transcript`` + the per-turn verifier-feedback
    rendering from :mod:`kore.policy.format`);
  * the response parser (``parse_response`` -> the ``FULL_KERNEL`` source);
  * and therefore, once handed to ``evaluate_policy`` under the same
    :class:`~kore.env.kore_env.KoreEnv`, the identical verified correctness oracle,
    cold-cache timing, matched measurement budget, and timing-INTEGRITY gate.

The ONLY thing that differs between the KORE side and the Opus side is the token
source. That is the whole point.

Turns / agentic variant: the policy interface DOES support turns (the ``feedback``
argument, which ``model_policy`` threads into a multi-turn transcript), so the
default here is the AGENTIC multi-turn variant - Opus sees the prior turn's
compile/SNR/speedup feedback and refines, exactly like the trained policy in the
bake-off's serial mode. Pass ``multi_turn=False`` for the SINGLE-SHOT variant,
where every bench is an independent one-shot generation with no cross-turn memory
(useful for a strict one-attempt comparison or parallel best-of-N).

Graceful degradation (clear error, no crash): if the teacher cannot be provisioned
- the ``anthropic`` SDK is missing, ``AMD_LLM_API_KEY`` is unset, or the gateway is
unreachable - :func:`opus_policy` raises a single, CLEAR
:class:`OpusUnavailableError` (not a deep SDK traceback), and :func:`try_opus_policy`
turns that into a ``(None, reason)`` pair with a loud warning so a caller (the
campaign eval) can SKIP the Opus side rather than crash. No API key is required to
IMPORT this module or to build a KORE-only eval.

Import-safe / offline: nothing heavy is imported at module load. The Anthropic SDK
is reached only lazily inside :func:`build_opus_teacher` (via ``make_teacher``), and
torch/vLLM are never touched (Opus's tokens come from the gateway, not a local
model). The adapter + error handling are unit-testable on CPU with a ``StubTeacher``
and no network (see ``tests/test_opus_head_to_head.py``).
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

from kore.eval.policies import PolicyFn, model_policy
from kore.obs import get_logger

_LOG = get_logger("eval.opus_policy")

# The default frontier teacher kind. ``make_teacher('claude')`` builds a
# ClaudeTeacher whose model already defaults to ``claude-opus-4.8`` (see
# kore.data.teacher.ClaudeTeacher), so "claude" and "opus" resolve to the same
# frontier model. ``make_teacher`` also accepts the aliases "anthropic"/"opus".
DEFAULT_OPUS_KIND = "claude"

# The policy label handed to ``model_policy`` (used only for logging / history
# keys; ``generate`` is injected so no checkpoint is ever loaded).
DEFAULT_OPUS_LABEL = "opus"


class OpusUnavailableError(RuntimeError):
    """The Opus/Claude teacher could not be provisioned.

    Raised by :func:`opus_policy` / :func:`build_opus_teacher` when the frontier
    teacher cannot be built - no ``anthropic`` SDK, missing ``AMD_LLM_API_KEY``,
    or an unreachable gateway. It carries a single, actionable message instead of
    letting a raw ``ImportError`` / SDK error surface, so callers can catch ONE
    exception type and degrade cleanly. :func:`try_opus_policy` converts it into a
    non-raising ``(None, reason)`` result.
    """


def _loud_warn(log, msg: str, **fields) -> None:
    """Emit a LOUD but non-fatal warning: structured log + a stderr banner.

    Mirrors the retention gate's "gate NOT enforced" pattern - the head-to-head
    must degrade (skip the Opus side), never crash, when the gateway is down.
    """
    (log or _LOG).warn(msg, **fields)
    try:
        print(f"[opus_policy] WARNING: {msg}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - never let logging break the eval
        pass


def _teacher_generate_adapter(teacher) -> Callable[..., str]:
    """Adapt a :class:`kore.data.teacher.TeacherClient` to the ``model_policy`` ABI.

    ``model_policy`` calls ``generate(messages, max_tokens=..., temperature=...)``.
    A ``TeacherClient.generate(messages)`` carries its OWN decoding params (fixed at
    construction), so the ``max_tokens`` / ``temperature`` ``model_policy`` forwards
    are intentionally dropped here - the teacher's construction-time config wins
    (see :func:`opus_policy`, which pins the teacher's temperature for a matched
    decode). Any messages->str callable therefore plugs straight in.
    """
    def gen(messages, **_kw):
        return teacher.generate(messages)
    return gen


def build_opus_teacher(
    kind: str = DEFAULT_OPUS_KIND,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    resilient: bool = True,
    load_env: bool = True,
    **teacher_kwargs,
):
    """Construct the frontier teacher, raising a CLEAR error (never a crash) on failure.

    Delegates to :func:`kore.data.teacher.make_teacher` (lazily imported, so this
    module stays import-safe/offline). Any provisioning failure - missing
    ``anthropic`` SDK, unset ``AMD_LLM_API_KEY``, unreachable gateway - is caught and
    re-raised as a single :class:`OpusUnavailableError` with an actionable message.

    ``resilient=True`` wraps the teacher in
    :class:`kore.data.teacher.ResilientTeacher`, which SKIPS a single transient
    total-failure (returns "") but still hard-stops on a SUSTAINED outage - the
    head-to-head driver treats that hard-stop as a clean Opus-side skip.

    ``load_env`` (default True) first loads ``.env.local`` so ``AMD_LLM_API_KEY`` /
    ``AMD_LLM_GATEWAY_URL`` are visible; set it False if the environment is already
    provisioned. ``model`` overrides the teacher model id (default
    ``claude-opus-4.8``); ``temperature`` (when given) is threaded into construction
    so the Opus side can decode at the SAME temperature as the KORE side.
    """
    try:
        from kore.data.teacher import make_teacher  # lazy: anthropic SDK lives here

        if load_env:
            try:
                from kore.data.teacher import load_env_local
                load_env_local()
            except Exception:  # noqa: BLE001 - a missing .env.local is not fatal
                pass

        kw = dict(teacher_kwargs)
        if model:
            kw["model"] = model
        if temperature is not None:
            kw["temperature"] = float(temperature)
        return make_teacher(kind, resilient=resilient, **kw)
    except Exception as e:  # noqa: BLE001 - normalize ANY provisioning failure
        raise OpusUnavailableError(
            f"Opus/Claude teacher unavailable ({type(e).__name__}: {str(e)[:200]}). "
            "Install the 'anthropic' SDK and set AMD_LLM_API_KEY (and, if needed, "
            "AMD_LLM_GATEWAY_URL) in the environment or .env.local, or pass an "
            "explicit teacher=... to opus_policy()."
        ) from e


def opus_policy(
    teacher=None,
    *,
    kind: str = DEFAULT_OPUS_KIND,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    multi_turn: bool = True,
    resilient: bool = True,
    label: str = DEFAULT_OPUS_LABEL,
    load_env: bool = True,
    **teacher_kwargs,
) -> PolicyFn:
    """Build an "Opus-as-policy" ``PolicyFn`` scored identically to the KORE side.

    Returns ``policy(task, feedback=None) -> kernel_source`` built by
    :func:`kore.eval.policies.model_policy` with the teacher's ``generate`` injected
    as the token source. Hand it straight to
    :func:`kore.eval.bakeoff.evaluate_policy` / ``matched_budget_bakeoff`` to bench
    Opus under the identical verified oracle + cold-cache timing + matched budget as
    KORE.

    Variants (the interface supports turns, so both are available):
      * ``multi_turn=True``  (default, AGENTIC): Opus sees the prior turn's
        verifier feedback and refines across the bake-off's serial turns;
      * ``multi_turn=False`` (SINGLE-SHOT): every bench is an independent one-shot
        generation with no cross-turn memory (each call starts a fresh trajectory).

    ``teacher`` may be supplied (a shared authenticated client, or a
    ``StubTeacher`` in tests); otherwise it is built lazily via
    :func:`build_opus_teacher`. To keep the head-to-head fair the teacher's decode
    ``temperature`` is pinned to ``temperature`` here (the teacher normally carries
    its own default of 0.7, which would silently mismatch a greedy KORE side).

    Raises :class:`OpusUnavailableError` (a clear, single error - never a raw SDK
    crash) when no ``teacher`` is given and one cannot be provisioned. Use
    :func:`try_opus_policy` for the non-raising, degrade-gracefully variant.
    """
    if teacher is None:
        teacher = build_opus_teacher(
            kind, model=model, temperature=temperature, resilient=resilient,
            load_env=load_env, **teacher_kwargs,
        )

    # Pin the teacher's decode temperature so Opus decodes at the SAME temperature
    # as the KORE side (a fair matched-decode head-to-head). ResilientTeacher
    # delegates attribute writes to the inner teacher via __getattr__ on read; set
    # it defensively and never let a read-only teacher break construction.
    if temperature is not None and hasattr(teacher, "temperature"):
        try:
            teacher.temperature = float(temperature)
        except Exception:  # noqa: BLE001 - some teachers pin temperature immutably
            pass

    pol = model_policy(
        label, generate=_teacher_generate_adapter(teacher),
        system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature,
    )
    if multi_turn:
        return pol

    # Single-shot: force a fresh trajectory every call so no cross-turn history
    # accumulates (model_policy clears its per-task history on feedback=None).
    def opus_single_shot_policy(task, feedback: Optional[dict] = None) -> str:
        return pol(task, None)

    return opus_single_shot_policy


def try_opus_policy(teacher=None, *, log=None, **kwargs):
    """Degrade-gracefully builder: ``(policy, None)`` on success, ``(None, reason)`` on failure.

    NEVER raises. Wraps :func:`opus_policy` so the campaign eval (or any caller that
    must survive an unprovisioned gateway) can skip the Opus side with a loud warning
    instead of crashing - exactly how the retention gate tolerates a missing serving
    backend. All keyword arguments are forwarded to :func:`opus_policy`.

    Returns a ``(PolicyFn | None, str | None)`` tuple: on failure the second element
    is the human-readable reason (the :class:`OpusUnavailableError` message).
    """
    try:
        return opus_policy(teacher=teacher, **kwargs), None
    except OpusUnavailableError as e:
        reason = str(e)
        _loud_warn(log, "Opus side unavailable; it will be SKIPPED "
                        "(KORE-only numbers still reported)", reason=reason[:200])
        return None, reason
    except Exception as e:  # noqa: BLE001 - any unexpected build failure is non-fatal here
        reason = f"unexpected error building Opus policy: {type(e).__name__}: {str(e)[:200]}"
        _loud_warn(log, "Opus side unavailable (unexpected build error); it will be "
                        "SKIPPED (KORE-only numbers still reported)", reason=reason)
        return None, reason


__all__ = [
    "DEFAULT_OPUS_KIND",
    "DEFAULT_OPUS_LABEL",
    "OpusUnavailableError",
    "build_opus_teacher",
    "opus_policy",
    "try_opus_policy",
]
