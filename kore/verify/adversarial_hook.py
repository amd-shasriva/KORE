"""Safe, throttled GRPO-loop hook for the co-evolved adversarial verifier.

This is the *one-call bridge* between the GRPO training loop (which owns
``kore/policy/grpo.py``) and the coevolutionary adversarial search in
:mod:`kore.verify.adversarial`. The search itself is already built + tested; it
GROWS the deterministic correctness battery by evolving test-cases that BREAK
kernels which currently lucky-pass the SNR gate, then folds the discovered breaks
back into an :func:`~kore.verify.adversarial.adversarial_inputs`-compatible generator
that :func:`~kore.verify.equivalence.verify_equivalence` can consume via its opt-in
``adversarial_inputs_fn=`` argument.

What this module adds
---------------------
A stateful :class:`AdversarialHook` (plus a process-global default + a
:func:`maybe_coevolve` convenience) that the orchestrator can call **once per group,
every N steps**, on a batch of the step's candidate kernels. It runs a *bounded*
coevolution round, screens the discovered breaks for monotone-safety, and
*accumulates* a per-``(op, dtype)`` strengthened ``adversarial_inputs_fn`` in a
process-global :class:`AdversarialRegistry`. The env / a future verify path reads it
back with :func:`get_adversarial_inputs_fn` and passes it to
``verify_equivalence(adversarial_inputs_fn=...)``.

Design guarantees (see the per-guarantee notes below and the module tests)
--------------------------------------------------------------------------
* **OFF by default.** Gated by ``KORE_ADVERSARIAL_COEVOLVE=1`` (mirrors the
  ``adversarial_coevolve`` lever / other ``KORE_*`` levers). Disabled => every entry
  point is a no-op and :func:`get_adversarial_inputs_fn` returns ``None`` - which is
  BYTE-IDENTICAL to the shipped oracle (``verify_equivalence`` treats
  ``adversarial_inputs_fn=None`` exactly like omitting it).
* **FAIL-SAFE.** The entire hot path is wrapped so ANY error -> a no-op that leaves the
  registry (hence the oracle) exactly as it was. The hook never raises into the loop.
* **BOUNDED.** Cost per invocation is hard-capped three ways: (1) small
  rounds x population, (2) a subsample of at most ``max_candidates`` kernels, and (3) a
  wall-clock deadline + evaluation-count budget that *winds the search down* (further
  candidate/reference runs return the current reference output, i.e. "agreement", so no
  more work and no spurious breaks).
* **MONOTONE-SAFE.** Folding only ever ADDS deterministic adversarial inputs to the
  existing battery (``include_base=True``, ``tighten_tolerance=False``): it never
  loosens a bound and never removes a check. Because the reference is ground truth, a
  kernel that matches the reference within tolerance on an input still matches it after
  that input is added - so a correct kernel is *never* falsely rejected. A determinism
  screen additionally drops any candidate case on which the reference is ill-defined or
  non-reproducible, so a flaky reference cannot inject an unsafe check.

Pure CPU. Like :mod:`kore.verify.adversarial`, execution is *injected*
(``run_candidate`` / ``run_reference``): the orchestrator supplies a runner that
dispatches a candidate to the real (GPU) env, while this module stays a pure,
unit-testable search over arrays and never imports torch or touches a GPU/process.

Honest scope. What is PROVEN: (a) disabled/None is byte-identical to stock; (b) folding
is monotone (base battery preserved, tolerance unchanged) so no correct kernel is newly
rejected; (c) cost is bounded by the caps above. What is HEURISTIC: whether a bounded
round actually *finds* a given kernel's defect - that is a directed search over a
repertoire of regimes, not a proof of correctness (same honest limitation as
:func:`~kore.verify.adversarial.coevolve_tests`). Once a break IS found and folded, the
oracle rejects that exact defect with certainty.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from kore.verify.adversarial import (
    TestCase,
    _default_minimal_criterion,
    _default_run,
    coevolve_tests,
    fold_breaking_cases,
    make_strengthened_inputs,
)
from kore.verify.equivalence import Tolerance, compare_pair, tolerance_for

__all__ = [
    "ENV_FLAG",
    "enabled_from_env",
    "HookBudget",
    "HookReport",
    "AdversarialRegistry",
    "AdversarialHook",
    "registry_key",
    "default_registry",
    "default_hook",
    "maybe_coevolve",
    "get_adversarial_inputs_fn",
    "registry_stats",
    "reset_registry",
]

# Consistent with the other KORE levers (KORE_VERIFIED_CORRECTNESS, KORE_SHAPE_AUGMENT,
# ...). The GRPOConfig field is ``adversarial_coevolve``; the orchestrator gates the
# hot path with this env var so it propagates to accelerate-launched training subprocs.
ENV_FLAG = "KORE_ADVERSARIAL_COEVOLVE"

# Post-deadline sentinel: a tiny, finite, self-consistent array. When the per-call
# budget is spent, wound-down runs return this (or the current reference output) so the
# search does no more real work AND can never manufacture a spurious break (both sides
# of the comparison agree). See ``_BoundedRunners``.
_SENTINEL = np.zeros((1,), dtype=np.float64)


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def enabled_from_env(env: Optional[dict] = None) -> bool:
    """True iff the ``adversarial_coevolve`` lever is enabled via ``KORE_ADVERSARIAL_COEVOLVE``.

    Accepts ``1/true/yes/on`` (case-insensitive). Defaults to False (OFF) so the hook
    and :func:`get_adversarial_inputs_fn` are inert unless the orchestrator opts in -
    identical in spirit to ``KORE_VERIFIED_CORRECTNESS`` and friends.
    """
    e = os.environ if env is None else env
    return _truthy(e.get(ENV_FLAG, ""))


def _env_int(e: dict, key: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(e.get(key, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _env_float(e: dict, key: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(e.get(key, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def registry_key(op: str, dtype: str) -> tuple[str, str]:
    """Normalised ``(op, dtype)`` registry key (lower-cased, stripped)."""
    return (str(op or "").strip().lower(), str(dtype or "").strip().lower())


# --------------------------------------------------------------------------- #
# Budget: the hard cost cap for one invocation (all knobs small + env-tunable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HookBudget:
    """Bounded per-invocation cost knobs. Defaults are deliberately small.

    A single ``run`` performs at most ``rounds x population_size`` genome evaluations,
    each running the reference once and up to ``min(len(candidates), max_candidates)``
    candidates - so the runner is invoked at most
    ``rounds * population_size * (1 + max_candidates)`` times, PLUS a bounded fold-screen
    of at most ``2 * fold_max_cases`` reference runs. ``max_seconds`` / ``max_evaluations``
    wind the search down early (see :class:`_BoundedRunners`), so wall-clock is bounded
    even if an injected GPU runner is slow.
    """

    every: int = 50            # throttle: run once per this many steps/calls
    rounds: int = 6            # coevolution rounds (small)
    population_size: int = 24  # genomes per round (small)
    elite_frac: float = 0.25
    max_candidates: int = 6    # subsample of the step's batch actually searched against
    max_seconds: float = 5.0   # wall-clock deadline for the search (soft wind-down)
    max_evaluations: int = 4000  # hard cap on total runner invocations (wind-down)
    families: Optional[tuple] = None    # None => all regime families
    fold_max_cases: int = 16   # max discovered breaks folded per invocation
    max_battery: int = 128     # max accumulated cases retained per (op, dtype)
    seed: int = 0

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "HookBudget":
        """Build a budget, letting the orchestrator tune the caps via env vars.

        All values are clamped to safe ranges; a malformed env value falls back to the
        default. This never widens beyond the hard ceilings, so "bounded" holds even if
        the env is set adversarially.
        """
        e = os.environ if env is None else env
        return cls(
            every=_env_int(e, "KORE_ADVERSARIAL_EVERY", 50, 1, 100_000),
            rounds=_env_int(e, "KORE_ADVERSARIAL_ROUNDS", 6, 1, 64),
            population_size=_env_int(e, "KORE_ADVERSARIAL_POP", 24, 4, 256),
            max_candidates=_env_int(e, "KORE_ADVERSARIAL_MAX_CANDIDATES", 6, 1, 64),
            max_seconds=_env_float(e, "KORE_ADVERSARIAL_MAX_SECONDS", 5.0, 0.0, 600.0),
            max_evaluations=_env_int(e, "KORE_ADVERSARIAL_MAX_EVALS", 4000, 1, 1_000_000),
            fold_max_cases=_env_int(e, "KORE_ADVERSARIAL_FOLD_MAX", 16, 1, 256),
            max_battery=_env_int(e, "KORE_ADVERSARIAL_MAX_BATTERY", 128, 1, 4096),
        )


# --------------------------------------------------------------------------- #
# Report: what one invocation did (never raises; safe to log)
# --------------------------------------------------------------------------- #
@dataclass
class HookReport:
    """Outcome of one :meth:`AdversarialHook.run` (fail-safe; always returned)."""

    ran: bool                       # did a coevolution round actually execute?
    reason: str                     # "ok" | "disabled" | "throttled" | "no-candidates" | "error" | ...
    op: str = ""
    dtype: str = ""
    broke_any: bool = False         # did the search find any breaking case?
    n_breaking: int = 0             # distinct breaking genomes found this round
    n_folded: int = 0               # breaks kept by the fold (pre-screen)
    n_screened_out: int = 0         # folded breaks dropped by the monotone-safety screen
    n_added: int = 0                # cases newly added to the (op, dtype) battery
    battery_size: int = 0           # accumulated battery size after this run
    n_evaluations: int = 0          # coevolution runner invocations (telemetry)
    elapsed_s: float = 0.0
    error: Optional[str] = None     # set (and swallowed) if anything went wrong

    def summary(self) -> str:
        tag = "ran" if self.ran else "skip"
        head = (f"[adv-hook:{tag}] op={self.op!r} dtype={self.dtype!r} reason={self.reason} "
                f"broke_any={self.broke_any} breaking={self.n_breaking} "
                f"folded={self.n_folded} screened_out={self.n_screened_out} "
                f"added={self.n_added} battery={self.battery_size} "
                f"evals={self.n_evaluations} elapsed={self.elapsed_s:.3f}s")
        if self.error:
            head += f" error={self.error}"
        return head


# --------------------------------------------------------------------------- #
# Registry: per-(op, dtype) accumulated adversarial battery (process-global)
# --------------------------------------------------------------------------- #
class AdversarialRegistry:
    """Thread-safe per-``(op, dtype)`` accumulator of folded adversarial test-cases.

    Stores only :class:`~kore.verify.adversarial.TestCase` genomes (pure data). The
    consumable generator is built on demand by :meth:`inputs_fn` via
    :func:`~kore.verify.adversarial.make_strengthened_inputs` with ``include_base=True``,
    so the FIXED enumerated battery is always preserved and the accumulated cases are
    only ever APPENDED - the monotone-safety invariant. Nothing here changes tolerances.
    """

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], list[TestCase]] = {}
        self._lock = threading.RLock()

    def add(self, op: str, dtype: str, cases: Sequence[TestCase],
            *, max_battery: int = 128) -> int:
        """Append ``cases`` to the ``(op, dtype)`` battery (dedup by genome signature).

        Returns the number of NEWLY added cases. The battery is capped at
        ``max_battery`` genomes, retained by descending difficulty (hardest kept).
        Purely additive: existing cases are never removed by an add of new ones (only
        the global cap can evict the easiest when overflowing).
        """
        cases = [c for c in cases if isinstance(c, TestCase)]
        if not cases:
            return 0
        key = registry_key(op, dtype)
        with self._lock:
            cur = self._by_key.get(key, [])
            seen = {c.signature() for c in cur}
            added = 0
            merged = list(cur)
            for c in cases:
                sig = c.signature()
                if sig in seen:
                    continue
                seen.add(sig)
                merged.append(c)
                added += 1
            if len(merged) > max_battery:
                merged.sort(key=lambda c: c.difficulty(dtype), reverse=True)
                merged = merged[:max_battery]
            self._by_key[key] = merged
            return added

    def get(self, op: str, dtype: str) -> list[TestCase]:
        """Snapshot (copy) of the accumulated cases for ``(op, dtype)`` (may be empty)."""
        key = registry_key(op, dtype)
        with self._lock:
            return list(self._by_key.get(key, []))

    def inputs_fn(self, op: str, dtype: str, *, include_base: bool = True
                  ) -> Optional[Callable]:
        """Return an ``adversarial_inputs``-compatible generator for ``(op, dtype)``.

        ``None`` when nothing has accumulated (caller then passes ``None`` to
        ``verify_equivalence``, i.e. the byte-identical stock battery). Otherwise a
        drop-in generator = fixed battery (``include_base``) followed by the accumulated
        breaking regimes, rebuilt at the caller's ``shape``/``dtype``/``arity``.
        """
        cases = self.get(op, dtype)
        if not cases:
            return None
        return make_strengthened_inputs(cases, include_base=include_base)

    def stats(self) -> dict[str, int]:
        """``{"op::dtype": battery_size}`` snapshot for observability."""
        with self._lock:
            return {f"{k[0]}::{k[1]}": len(v) for k, v in self._by_key.items()}

    def keys(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._by_key.keys())

    def clear(self) -> None:
        with self._lock:
            self._by_key.clear()


# --------------------------------------------------------------------------- #
# Bounded runners: wall-clock + evaluation-count wind-down (safe by construction)
# --------------------------------------------------------------------------- #
class _BoundedRunners:
    """Wrap injected ``run_candidate``/``run_reference`` with a hard cost bound.

    Before the deadline/budget is hit, calls pass straight through to the real runner
    (default: an in-process call). Once ``max_seconds`` elapse OR ``max_evaluations``
    runner calls are made, the wrappers stop doing real work:

      * ``reference`` returns the sentinel (and records it as the "current reference"),
      * ``candidate`` returns the CURRENT reference output.

    So every wound-down comparison is (ref vs ref) == agreement: no further reference/
    candidate execution happens and NO spurious break can be created. Breaks found
    *before* wind-down are genuine (real reference vs real candidate). This makes the
    per-invocation cost bounded regardless of how slow an injected GPU runner is.
    """

    def __init__(self, run_candidate: Optional[Callable], run_reference: Optional[Callable],
                 *, max_seconds: float, max_evaluations: int) -> None:
        self._rc = run_candidate or _default_run
        self._rr = run_reference or _default_run
        self._max_seconds = float(max_seconds)
        self._max_evals = int(max_evaluations)
        self._t0 = None
        self.n_calls = 0
        self._last_ref: Any = _SENTINEL

    def start(self) -> "_BoundedRunners":
        self._t0 = time.monotonic()
        return self

    def _expired(self) -> bool:
        if self.n_calls >= self._max_evals:
            return True
        if self._max_seconds <= 0.0:      # 0 => immediate wind-down (no real work)
            return True
        if self._t0 is None:
            return False
        return (time.monotonic() - self._t0) >= self._max_seconds

    def reference(self, fn: Callable, inputs: tuple):
        self.n_calls += 1
        if self._expired():
            self._last_ref = _SENTINEL
            return _SENTINEL
        out = self._rr(fn, inputs)
        self._last_ref = out
        return out

    def candidate(self, fn: Callable, inputs: tuple):
        self.n_calls += 1
        if self._expired():
            # Return THIS case's reference output (set by the preceding reference call
            # in coevolve's per-case evaluation) so the comparison trivially agrees.
            return self._last_ref
        return self._rc(fn, inputs)


# --------------------------------------------------------------------------- #
# The hook
# --------------------------------------------------------------------------- #
class AdversarialHook:
    """Stateful, throttled, fail-safe entry point for the GRPO loop.

    Call :meth:`run` once per group every step; it self-throttles to one real
    coevolution round per ``budget.every`` invocations (or per ``budget.every`` steps
    when a ``step`` index is supplied). Everything is wrapped so a failure degrades to a
    no-op that leaves the registry (and thus the oracle) untouched.
    """

    def __init__(self, budget: Optional[HookBudget] = None,
                 registry: Optional[AdversarialRegistry] = None,
                 *, enabled: Optional[bool] = None) -> None:
        self.budget = budget or HookBudget.from_env()
        self.registry = registry if registry is not None else default_registry()
        # None => resolve from the env var at call time (so toggling the lever works
        # without rebuilding the hook); True/False => hard override (used by tests).
        self._enabled_override = enabled
        self._n_calls = 0
        self._lock = threading.RLock()

    # -- gating ----------------------------------------------------------- #
    def is_enabled(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        return enabled_from_env()

    def _throttle_ok(self, step: Optional[int]) -> bool:
        every = max(1, int(self.budget.every))
        if step is not None:
            return (int(step) % every) == 0
        with self._lock:
            self._n_calls += 1
            return (self._n_calls % every) == 0

    def should_run(self, step: Optional[int] = None, *, force: bool = False) -> bool:
        """Advisory check the orchestrator may use to skip building runner closures.

        NB: with ``step=None`` the internal counter is only advanced inside :meth:`run`,
        so this peek does not consume a tick.
        """
        if not self.is_enabled():
            return False
        if force:
            return True
        every = max(1, int(self.budget.every))
        if step is not None:
            return (int(step) % every) == 0
        with self._lock:
            return ((self._n_calls + 1) % every) == 0

    # -- the one call ----------------------------------------------------- #
    def run(self, *, op: str, dtype: str,
            reference_fn: Callable,
            candidate_fns: Any,
            step: Optional[int] = None,
            shape: Any = None,
            op_class: str = "generic",
            arity: Optional[int] = None,
            run_candidate: Optional[Callable] = None,
            run_reference: Optional[Callable] = None,
            tol: Optional[Tolerance] = None,
            force: bool = False) -> HookReport:
        """Maybe run a bounded coevolution round for one ``(op, dtype)`` on a candidate batch.

        Parameters mirror :func:`~kore.verify.adversarial.coevolve_tests`. ``op`` /
        ``dtype`` are the task's ``operation`` / ``dtype`` (the registry key).
        ``reference_fn`` is the ground-truth oracle callable; ``candidate_fns`` is the
        step's batch of candidate kernels (any opaque handle understood by
        ``run_candidate``; if ``run_candidate`` is omitted they must be plain callables).
        ``run_candidate`` is the INJECTION point the orchestrator uses to dispatch a
        candidate to the real (GPU) env - this module never runs a kernel itself.

        Returns a :class:`HookReport`. NEVER raises: any error is captured in
        ``report.error`` and the registry is left unchanged.
        """
        t0 = time.monotonic()
        op_s, dtype_s = str(op or ""), str(dtype or "")
        try:
            if not self.is_enabled():
                return HookReport(ran=False, reason="disabled", op=op_s, dtype=dtype_s)
            if not (force or self._throttle_ok(step)):
                return HookReport(ran=False, reason="throttled", op=op_s, dtype=dtype_s)

            cand_list = self._as_candidate_list(candidate_fns)
            if not cand_list:
                return HookReport(ran=False, reason="no-candidates", op=op_s, dtype=dtype_s)
            if reference_fn is None:
                return HookReport(ran=False, reason="no-reference", op=op_s, dtype=dtype_s)

            b = self.budget
            cand_list = cand_list[: max(1, int(b.max_candidates))]
            tol = tol or tolerance_for(dtype_s)
            ar = int(arity) if arity else 1

            # --- bounded coevolution (pure CPU; execution is injected) --- #
            runners = _BoundedRunners(run_candidate, run_reference,
                                      max_seconds=b.max_seconds,
                                      max_evaluations=b.max_evaluations).start()
            result = coevolve_tests(
                reference_fn, cand_list, shape=shape, dtype=dtype_s, arity=ar,
                seed=int(b.seed), rounds=int(b.rounds),
                population_size=int(b.population_size), elite_frac=float(b.elite_frac),
                families=list(b.families) if b.families else None, tol=tol, device="cpu",
                run_candidate=runners.candidate, run_reference=runners.reference,
            )

            report = HookReport(ran=True, reason="ok", op=op_s, dtype=dtype_s,
                                broke_any=bool(result.broke_any),
                                n_evaluations=int(result.n_evaluations),
                                elapsed_s=round(time.monotonic() - t0, 4))
            if not result.breaking_cases:
                report.battery_size = len(self.registry.get(op_s, dtype_s))
                return report

            # --- fold (monotone: add-only, tolerance unchanged) --- #
            fold = fold_breaking_cases(result.breaking_cases, base_tol=tol, dtype=dtype_s,
                                       tighten_tolerance=False,   # NEVER tighten: add-only
                                       max_cases=int(b.fold_max_cases), include_base=True)
            report.n_breaking = len(result.breaking_cases)
            report.n_folded = fold.n_folded

            # --- monotone-safety screen (drop unsafe/ill-defined cases) --- #
            safe_cases, dropped = self._screen_cases(
                fold.cases, reference_fn=reference_fn, run_reference=run_reference,
                shape=shape, dtype=dtype_s, arity=ar, tol=tol)
            report.n_screened_out = dropped

            report.n_added = self.registry.add(op_s, dtype_s, safe_cases,
                                                max_battery=int(b.max_battery))
            report.battery_size = len(self.registry.get(op_s, dtype_s))
            report.elapsed_s = round(time.monotonic() - t0, 4)
            return report
        except Exception as exc:      # noqa: BLE001 - the hook must NEVER break the loop
            return HookReport(ran=False, reason="error", op=op_s, dtype=dtype_s,
                              error=f"{type(exc).__name__}: {exc}",
                              elapsed_s=round(time.monotonic() - t0, 4))

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _as_candidate_list(candidate_fns: Any) -> list:
        if candidate_fns is None:
            return []
        if callable(candidate_fns):
            return [candidate_fns]
        try:
            return [c for c in candidate_fns if c is not None]
        except TypeError:
            return [candidate_fns]

    def _screen_cases(self, cases: Sequence[TestCase], *, reference_fn: Callable,
                      run_reference: Optional[Callable], shape: Any, dtype: str,
                      arity: int, tol: Tolerance) -> tuple[list, int]:
        """Keep only cases proven MONOTONE-SAFE to add; drop the rest.

        A case is admitted iff, on the input it materialises, the reference (a) builds,
        (b) defines a non-empty finite truth (the minimal criterion), and (c) is
        REPRODUCIBLE - two reference evaluations agree within ``tol``. Given a
        deterministic reference this holds for every genuine breaking case, so nothing
        useful is lost; but it guarantees we never fold an input on which even a
        reference-matching (correct) kernel could be rejected. Bounded: at most
        ``fold_max_cases`` cases, each 2 reference runs, wound down by its own deadline.
        """
        rr = run_reference or _default_run
        screen = _BoundedRunners(None, run_reference,
                                 max_seconds=self.budget.max_seconds,
                                 max_evaluations=max(4, 2 * len(cases) + 2)).start()
        kept: list[TestCase] = []
        dropped = 0
        for c in cases:
            try:
                inputs = c.build(shape, dtype, device="cpu")
            except Exception:      # noqa: BLE001 - unbuildable => not safe to fold
                dropped += 1
                continue
            if screen._expired():
                dropped += 1        # ran out of screening budget => conservatively drop
                continue
            r1 = rr(reference_fn, inputs)
            r2 = rr(reference_fn, inputs)
            if not _default_minimal_criterion(r1, inputs):
                dropped += 1
                continue
            try:
                same = compare_pair(r1, r2, tol).ok
            except Exception:      # noqa: BLE001 - un-comparable reference => drop
                same = False
            if not same:            # non-reproducible reference => unsafe to fold
                dropped += 1
                continue
            kept.append(c)
        return kept, dropped


# --------------------------------------------------------------------------- #
# Process-global default registry + hook + one-call convenience
# --------------------------------------------------------------------------- #
_REGISTRY = AdversarialRegistry()
_DEFAULT_HOOK: Optional[AdversarialHook] = None
_DEFAULT_LOCK = threading.RLock()


def default_registry() -> AdversarialRegistry:
    """The process-global :class:`AdversarialRegistry` the env / verify path reads."""
    return _REGISTRY


def default_hook() -> AdversarialHook:
    """The process-global :class:`AdversarialHook` used by :func:`maybe_coevolve`."""
    global _DEFAULT_HOOK
    with _DEFAULT_LOCK:
        if _DEFAULT_HOOK is None:
            _DEFAULT_HOOK = AdversarialHook(registry=_REGISTRY)
        return _DEFAULT_HOOK


def maybe_coevolve(**kwargs) -> HookReport:
    """THE one-call the orchestrator adds to the GRPO loop (fail-safe, throttled).

    Thin wrapper over ``default_hook().run(**kwargs)``. Call it unconditionally once per
    group per step with ``op``, ``dtype``, ``reference_fn``, ``candidate_fns``, ``step``
    and (for the live loop) an injected ``run_candidate`` that dispatches to the env; it
    self-gates on the lever + throttle and never raises. See :meth:`AdversarialHook.run`.
    """
    return default_hook().run(**kwargs)


def get_adversarial_inputs_fn(op: str, dtype: str, *, enabled: Optional[bool] = None,
                              registry: Optional[AdversarialRegistry] = None
                              ) -> Optional[Callable]:
    """Accumulated ``adversarial_inputs_fn`` for ``(op, dtype)``, or ``None`` if inert.

    This is the ENV-SIDE consumption point. Pass the result straight to
    ``verify_equivalence(..., adversarial_inputs_fn=get_adversarial_inputs_fn(op, dtype))``:
    when the lever is off or nothing has accumulated it returns ``None``, which
    ``verify_equivalence`` treats identically to the stock enumerated battery (proven
    byte-identical in the coevolution test-suite). So wiring this in is always safe -
    it can only ever ADD discovered regimes, and only when explicitly enabled.
    """
    is_on = enabled_from_env() if enabled is None else bool(enabled)
    if not is_on:
        return None
    reg = registry if registry is not None else _REGISTRY
    return reg.inputs_fn(op, dtype)


def registry_stats(registry: Optional[AdversarialRegistry] = None) -> dict[str, int]:
    """``{"op::dtype": battery_size}`` for the (default) registry - observability."""
    reg = registry if registry is not None else _REGISTRY
    return reg.stats()


def reset_registry(registry: Optional[AdversarialRegistry] = None) -> None:
    """Clear the (default) registry. Primarily for tests / a fresh campaign."""
    reg = registry if registry is not None else _REGISTRY
    reg.clear()
