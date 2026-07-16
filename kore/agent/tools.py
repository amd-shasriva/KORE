"""Agentic tool layer: OpenAI/Hermes tool schemas + executors over ``KoreEnv``.

The policy drives its own optimize loop by emitting tool calls. Each tool wraps
the *verified* environment for a single task:

    build(kernel_src)                 -> compile-only check
    test(kernel_src, shape=None)      -> correctness (SNR gate) on one/all shapes
    bench(kernel_src, shape=None)     -> timed vs the production baseline
    pmc(kernel_src)                   -> hardware counters (from Observation or stub)
    keep()                            -> commit the current candidate as the kernel
    revert()                          -> discard the current candidate

``ToolExecutor`` is *pure/deterministic given the env*: it holds only the
episode's rolling state (current candidate, committed kernel, best-correct
kernel) and dispatches a parsed tool call to the real env, returning a compact
JSON-serializable dict. ``tool_use_reward`` is a ToolRL-style pure reward
shaping function over a finished episode.

Nothing here imports torch/vllm; the only GPU contact is the injected env.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from kore.reward.physics import compute_kernel_reward

# --------------------------------------------------------------------------- #
# Tool schemas (OpenAI function-calling JSON; also rendered into a Hermes
# system prompt by kore.agent.format.build_agent_system_prompt).
# --------------------------------------------------------------------------- #
_KERNEL_SRC = {
    "type": "string",
    "description": "COMPLETE Triton kernel source (full file, ready to run).",
}
_SHAPE = {
    "type": "string",
    "description": "Optional shape name to target; omit to run all validation shapes.",
}


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _fn(
        "build",
        "Compile-check a candidate kernel WITHOUT correctness/timing. Fast gate "
        "to catch syntax/compile errors before spending a full validation.",
        {"kernel_src": _KERNEL_SRC},
        ["kernel_src"],
    ),
    _fn(
        "test",
        "Run the full four-prong correctness oracle (random, adversarial, "
        "metamorphic, determinism) with the SNR gate on a candidate. "
        "Returns whether it is numerically correct on the target shape(s).",
        {"kernel_src": _KERNEL_SRC, "shape": _SHAPE},
        ["kernel_src"],
    ),
    _fn(
        "bench",
        "Benchmark a candidate against the production baseline (AITER/hipBLASLt) "
        "and report the worst-shape speedup. Only meaningful once correct.",
        {"kernel_src": _KERNEL_SRC, "shape": _SHAPE},
        ["kernel_src"],
    ),
    _fn(
        "pmc",
        "Collect hardware performance counters (rocprofv3) for a candidate to "
        "diagnose compute- vs memory-bound behavior (registers, occupancy, "
        "wait/MFMA ratio).",
        {"kernel_src": _KERNEL_SRC},
        ["kernel_src"],
    ),
    _fn(
        "keep",
        "Commit the current candidate as the working kernel (a deliberate, "
        "trainable decision to accept the last change).",
        {},
        [],
    ),
    _fn(
        "revert",
        "Discard the current candidate and roll back to the last committed "
        "kernel (a deliberate, trainable decision to reject the last change).",
        {},
        [],
    ),
]

TOOL_NAMES: list[str] = [s["function"]["name"] for s in TOOL_SCHEMAS]
_SCHEMA_BY_NAME: dict[str, dict] = {s["function"]["name"]: s for s in TOOL_SCHEMAS}


# --------------------------------------------------------------------------- #
# Tool-call validation (ToolRL: name + params + format correctness)
# --------------------------------------------------------------------------- #
_TYPE_MAP = {"string": str, "number": (int, float), "integer": int,
             "boolean": bool, "object": dict, "array": list}


def validate_tool_call(call: dict) -> dict:
    """Validate a parsed tool call against ``TOOL_SCHEMAS``.

    Returns ``{"valid_name", "valid_params", "errors": [...]}``. Pure; never
    raises. ``call`` is ``{"name": str, "arguments": dict, ...}``.
    """
    name = call.get("name")
    args = call.get("arguments")
    errors: list[str] = []

    valid_name = name in _SCHEMA_BY_NAME
    if not valid_name:
        return {"valid_name": False, "valid_params": False,
                "errors": [f"unknown tool {name!r}"]}

    params = _SCHEMA_BY_NAME[name]["function"]["parameters"]
    props = params.get("properties", {})
    required = params.get("required", [])

    if not isinstance(args, dict):
        return {"valid_name": True, "valid_params": False,
                "errors": ["arguments is not an object"]}

    for req in required:
        if req not in args or args[req] in (None, ""):
            errors.append(f"missing required arg {req!r}")
    for key, val in args.items():
        if key not in props:
            errors.append(f"unexpected arg {key!r}")
            continue
        expected = props[key].get("type")
        py = _TYPE_MAP.get(expected)
        if py is not None and val is not None and not isinstance(val, py):
            errors.append(f"arg {key!r} should be {expected}")

    return {"valid_name": True, "valid_params": not errors, "errors": errors}


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #
def _round(x: Optional[float], n: int = 4) -> Optional[float]:
    return round(x, n) if isinstance(x, (int, float)) else None


class ToolExecutor:
    """Dispatch parsed tool calls to a task-bound :class:`KoreEnv`.

    Holds the rolling episode state:
      - ``committed_src``  : the accepted working kernel (seeded from the task).
      - ``candidate_src``  : the last kernel passed to build/test/bench/pmc.
      - ``best_src``       : the best *correct* kernel seen so far (by reward);
                             the trajectory is scored by this (Kevin: score by
                             BEST kernel, not the last one).
    """

    def __init__(self, env, task, seed_src: Optional[str] = None):
        self.env = env
        self.task = task
        self.dtype = getattr(task, "dtype", "fp32")

        self.seed_src: Optional[str] = seed_src
        self.committed_src: Optional[str] = seed_src
        self.committed_reward: Optional[float] = None
        self.candidate_src: Optional[str] = None
        self.candidate_reward: Optional[float] = None
        self.candidate_correct: bool = False
        # Per-turn MEASURED speedup of the candidate evaluated this turn (vendor-
        # relative, from the verified Observation). None unless the candidate was
        # BENCHED and correct, so a build/test turn never fabricates a timing
        # signal. Consumed by the harness's per-turn trace -> GRPO ``speedups``.
        self.candidate_speedup: Optional[float] = None

        self.best_src: Optional[str] = None
        self.best_reward: float = float("-inf")
        self.best_turn: Optional[int] = None
        self.best_obs: dict = {}
        # Best MEASURED speedup seen so far (max over benched-correct candidates),
        # tracked independently of best_reward so the ``bench`` tool can report an
        # honest frontier delta ("did THIS change beat my fastest correct kernel?").
        self.best_speedup: Optional[float] = None

        self.keep_decisions: list[dict] = []
        self._turn: int = 0

    # -- helpers ---------------------------------------------------------- #
    def set_turn(self, turn: int) -> None:
        self._turn = turn

    def reseed_lineage(self) -> dict:
        """Abandon the current candidate lineage and roll back to the seed.

        Debugging-trap escape (P1): after too many non-improving turns the
        harness discards the dead candidate/committed lineage and restarts from
        the task's seed kernel. The best *correct* kernel found so far is
        PRESERVED (``best_src``/``best_reward`` are untouched) so a reset can
        never lose real progress - it only stops the policy from patching a
        kernel that is going nowhere.
        """
        self.committed_src = self.seed_src
        self.committed_reward = None
        self.candidate_src = None
        self.candidate_reward = None
        self.candidate_correct = False
        self.candidate_speedup = None
        return {"ok": True, "reseeded": True,
                "seeded_from": "task_seed" if self.seed_src else "empty",
                "preserved_best_reward": _round(
                    self.best_reward if self.best_reward != float("-inf") else None)}

    def _obs_dict(self, obs) -> dict:
        return {
            "compiled": bool(getattr(obs, "compiled", False)),
            "validation_passed": bool(getattr(obs, "validation_passed", False)),
            "snr_db": _round(getattr(obs, "snr_db", None), 2),
            "wall_ms": _round(getattr(obs, "wall_ms", None)),
            "baseline_ms": _round(getattr(obs, "baseline_ms", None)),
            "profile_efficiency": _round(getattr(obs, "profile_efficiency", None), 3),
        }

    def _evaluate(self, src: str, do_bench: bool, multi_shape: bool):
        """Run the env and compute the reward; update candidate + best state."""
        obs = self.env.step(src, full_validation=do_bench, multi_shape=multi_shape)
        # Reward mode honors KORE_REWARD_MODE ("residual" -> physics residual reward,
        # else vendor speedup); the non-agentic GRPO path is driven by config.reward_mode.
        # KORE_REWARD_PHASE bridges the correctness->latency curriculum so a correct
        # kernel is credited correctness-only in the correctness phase, matching the
        # serial path's apply_reward_phase (audit R2 grpo C1/C2).
        _mode = os.environ.get("KORE_REWARD_MODE", "speedup")
        _phase = os.environ.get("KORE_REWARD_PHASE", "all")
        rr = compute_kernel_reward(obs, src, self.task, mode=_mode, dtype=self.dtype,
                                   reward_phase=_phase)
        self.candidate_src = src
        self.candidate_reward = rr.reward
        self.candidate_correct = rr.correct
        # Only a benched, correct candidate has a trustworthy measured speedup; a
        # compile/correctness-only eval leaves it None (no timing was taken).
        self.candidate_speedup = (float(rr.speedup)
                                  if (rr.correct and do_bench and rr.speedup is not None)
                                  else None)
        if rr.correct and rr.reward > self.best_reward:
            self.best_reward = rr.reward
            self.best_src = src
            self.best_turn = self._turn
            self.best_obs = self._obs_dict(obs)
        # Frontier of measured speedup advances on any faster benched-correct
        # candidate (decoupled from best_reward, which is the Kevin scoring key).
        if self.candidate_speedup is not None and (
            self.best_speedup is None or self.candidate_speedup > self.best_speedup):
            self.best_speedup = self.candidate_speedup
        return obs, rr

    # -- dispatch --------------------------------------------------------- #
    def dispatch(self, call: dict, turn: Optional[int] = None) -> dict:
        """Execute one parsed tool call. Never raises; returns a compact dict."""
        if turn is not None:
            self._turn = turn
        name = call.get("name")
        args = call.get("arguments") or {}
        if name not in _SCHEMA_BY_NAME:
            return {"ok": False, "tool": name, "error": f"unknown tool {name!r}"}
        v = validate_tool_call(call)
        if not v["valid_params"]:
            return {"ok": False, "tool": name, "error": "; ".join(v["errors"])}
        try:
            handler = getattr(self, f"_tool_{name}")
            return handler(args)
        except Exception as e:  # noqa: BLE001 - tools must never crash the loop
            return {"ok": False, "tool": name, "error": f"executor error: {e}"}

    # -- individual tools ------------------------------------------------- #
    def _tool_build(self, args: dict) -> dict:
        src = args["kernel_src"]
        obs, rr = self._evaluate(src, do_bench=False, multi_shape=False)
        return {
            "ok": bool(obs.compiled),
            "tool": "build",
            "compiled": bool(obs.compiled),
            "error": (obs.error_text or None) if not obs.compiled else None,
        }

    def _tool_test(self, args: dict) -> dict:
        src = args["kernel_src"]
        multi = args.get("shape") is None
        obs, rr = self._evaluate(src, do_bench=False, multi_shape=multi)
        return {
            "ok": bool(rr.correct),
            "tool": "test",
            "compiled": bool(obs.compiled),
            "correct": bool(rr.correct),
            "snr_db": _round(getattr(obs, "snr_db", None), 2),
            "snr_by_shape": {k: _round(v, 2) for k, v in
                             (getattr(obs, "snr_by_shape", {}) or {}).items()},
            "tier": rr.tier,
            "reward": _round(rr.reward),
            "error": (obs.error_text or None) if not rr.correct else None,
        }

    def _tool_bench(self, args: dict) -> dict:
        src = args["kernel_src"]
        multi = args.get("shape") is None
        # Snapshot the measured-speedup frontier BEFORE this candidate is folded in
        # so the delta reflects "did THIS turn's change beat my fastest correct
        # kernel so far?" - the per-turn latency feedback the policy optimizes.
        prev_best_su = self.best_speedup
        obs, rr = self._evaluate(src, do_bench=True, multi_shape=multi)
        cur_su = self.candidate_speedup
        delta = (round(cur_su - prev_best_su, 3)
                 if (cur_su is not None and prev_best_su is not None) else None)
        improved = bool(cur_su is not None and (prev_best_su is None or cur_su > prev_best_su))
        return {
            "ok": bool(rr.correct),
            "tool": "bench",
            "compiled": bool(obs.compiled),
            "correct": bool(rr.correct),
            "speedup": _round(rr.speedup, 3),
            # Per-turn measured-latency feedback: the model reads its own kernel's
            # speedup, the running best, the signed delta vs that best, and whether
            # it pushed the frontier. Pure context (the trained reward is still the
            # verified compute_kernel_reward), so it cannot be gamed.
            "best_speedup_so_far": _round(self.best_speedup, 3),
            "delta_vs_best": delta,
            "improved_frontier": improved,
            "wall_ms": _round(getattr(obs, "wall_ms", None)),
            "baseline_ms": _round(getattr(obs, "baseline_ms", None)),
            "tier": rr.tier,
            "reward": _round(rr.reward),
            "error": (obs.error_text or None) if not rr.correct else None,
        }

    def _tool_pmc(self, args: dict) -> dict:
        src = args["kernel_src"]
        obs, rr = self._evaluate(src, do_bench=True, multi_shape=False)
        # Surface the real hardware-counter signal KoreEnv computes: the
        # baseline-relative roofline efficiency (rocprofv3). Available only when
        # profiling is enabled (KORE_PROFILE_REWARD_WEIGHT>0), else honest stub.
        eff = getattr(obs, "profile_efficiency", None)
        available = eff is not None
        return {
            "ok": bool(obs.compiled),
            "tool": "pmc",
            "available": bool(available),
            "profile_efficiency": _round(eff, 3),
            "diagnosis": (f"roofline efficiency vs baseline: {eff:.2f} "
                          "(1.0 = as efficient as the vendor kernel)" if available
                          else "hardware-counter profiling disabled on this env "
                               "(set KORE_PROFILE_REWARD_WEIGHT>0 to enable)"),
        }

    def _tool_keep(self, args: dict) -> dict:
        if self.candidate_src is None:
            return {"ok": False, "tool": "keep", "kept": False,
                    "reason": "no candidate to keep"}
        prev = self.committed_reward
        improved = (prev is None) or (
            self.candidate_reward is not None and self.candidate_reward > prev
        )
        self.committed_src = self.candidate_src
        self.committed_reward = self.candidate_reward
        decision = {
            "turn": self._turn, "action": "keep",
            "candidate_reward": _round(self.candidate_reward),
            "prev_committed_reward": _round(prev),
            "correct": self.candidate_correct,
            "improved": bool(improved),
        }
        self.keep_decisions.append(decision)
        return {"ok": True, "tool": "keep", "kept": True,
                "improved": bool(improved),
                "committed_reward": _round(self.committed_reward),
                "correct": self.candidate_correct}

    def _tool_revert(self, args: dict) -> dict:
        # Was the discarded candidate actually a regression vs. what we keep?
        was_regression = (
            self.committed_reward is not None
            and self.candidate_reward is not None
            and self.candidate_reward < self.committed_reward
        ) or (self.candidate_src is not None and not self.candidate_correct)
        decision = {
            "turn": self._turn, "action": "revert",
            "candidate_reward": _round(self.candidate_reward),
            "committed_reward": _round(self.committed_reward),
            "correct": self.candidate_correct,
            "was_regression": bool(was_regression),
        }
        self.keep_decisions.append(decision)
        self.candidate_src = None
        self.candidate_reward = None
        self.candidate_correct = False
        return {"ok": True, "tool": "revert", "reverted": True,
                "committed_reward": _round(self.committed_reward),
                "was_regression": bool(was_regression)}


# --------------------------------------------------------------------------- #
# ToolRL-style reward shaping (PURE function of a finished episode)
# --------------------------------------------------------------------------- #
# Weights: outcome dominates (correctness/speed of the BEST kernel); the tool-use
# terms shape *how* the policy got there. Penalties are folded in (negative).
W_NAME = 0.15        # +correct tool name
W_PARAM = 0.15       # +valid params / schema
W_FORMAT = 0.10      # +well-formed tool-call syntax
W_OUTCOME = 1.00     # +best-kernel reward (correctness first, then speed)
W_KEEP_REVERT = 0.20  # +correct keep/revert decisions
W_REFLECT = 0.10     # +reflection quality (BOUNDED so it can't dominate outcome)
P_MALFORMED = 0.30   # -malformed call fraction
P_FAILED = 0.10      # -failed (build/test) call fraction


def _episode_field(episode: Any, name: str, default=None):
    if isinstance(episode, dict):
        return episode.get(name, default)
    return getattr(episode, name, default)


def tool_use_reward(episode: Any) -> dict:
    """ToolRL-style reward shaping over a finished :class:`AgentEpisode`.

    Returns the individual components plus a weighted ``total`` so the parent RL
    loop can fold it into the trajectory reward. Pure and deterministic.

    Components
      tool_name    fraction of calls that named a real tool
      param        fraction of calls with schema-valid params
      format       fraction of calls that were well-formed (parsed cleanly)
      outcome      reward of the BEST correct kernel (Kevin: score by best)
      keep_revert  signed score for correct keep/revert decisions
      penalty      malformed + failed call penalties (<= 0)
      total        weighted sum
    """
    trace = list(_episode_field(episode, "tool_trace", []) or [])
    best_reward = _episode_field(episode, "best_reward", None)
    keep_decisions = list(_episode_field(episode, "keep_decisions", []) or [])
    reflections = list(_episode_field(episode, "reflections", []) or [])
    success = bool(_episode_field(episode, "success", False))

    n = len(trace)
    n_malformed = sum(1 for t in trace if t.get("malformed"))
    n_bad_name = sum(1 for t in trace if not t.get("valid_name"))
    n_bad_param = sum(1 for t in trace if not t.get("valid_params"))
    n_failed = sum(1 for t in trace if _is_failed_call(t))

    def frac_ok(bad: int) -> float:
        return 1.0 - (bad / n) if n else 0.0

    name_score = frac_ok(n_bad_name)
    param_score = frac_ok(n_bad_param)
    format_score = frac_ok(n_malformed)

    outcome = float(best_reward) if isinstance(best_reward, (int, float)) \
        and best_reward != float("-inf") else 0.0

    kr_score = _keep_revert_score(keep_decisions)
    reflect_score = _reflection_score(reflections, trace)

    malformed_pen = -P_MALFORMED * (n_malformed / n) if n else 0.0
    failed_pen = -P_FAILED * (n_failed / n) if n else 0.0
    penalty = malformed_pen + failed_pen

    total = (
        W_NAME * name_score
        + W_PARAM * param_score
        + W_FORMAT * format_score
        + W_OUTCOME * outcome
        + W_KEEP_REVERT * kr_score
        + W_REFLECT * reflect_score
        + penalty
    )

    return {
        "tool_name": round(name_score, 4),
        "param": round(param_score, 4),
        "format": round(format_score, 4),
        "outcome": round(outcome, 4),
        "keep_revert": round(kr_score, 4),
        "reflection": round(reflect_score, 4),
        "penalty": round(penalty, 4),
        "total": round(total, 4),
        "n_calls": n,
        "n_malformed": n_malformed,
        "n_bad_param": n_bad_param,
        "n_failed": n_failed,
        "n_reflections": len(reflections),
        "success": success,
    }


def _is_failed_call(t: dict) -> bool:
    """A build/test/bench/pmc call that ran but reported a failure."""
    name = t.get("name")
    if name not in ("build", "test", "bench", "pmc"):
        return False
    res = t.get("result") or {}
    return res.get("ok") is False


_REFLECT_FIELDS = ("root_cause", "evidence", "planned_fix")


def _reflection_score(reflections: list[dict], trace: list[dict]) -> float:
    """Bounded [0,1] quality of the episode's reflections (GEAK/Reflexion).

    Each reflection scores on two halves: (a) COMPLETENESS - the fraction of
    ``root_cause``/``evidence``/``planned_fix`` fields that are non-empty; and
    (b) GROUNDING - whether it references the ACTUAL error text surfaced by a
    failed tool call this episode (not a generic guess). Averaged over
    reflections; 0.0 when there were none. Pure and bounded so, at
    ``W_REFLECT``, it can never dominate the verified kernel reward.
    """
    if not reflections:
        return 0.0
    error_texts = _episode_error_texts(trace)
    scores: list[float] = []
    for r in reflections:
        if not isinstance(r, dict):
            continue
        present = [str(r.get(k, "") or "").strip() for k in _REFLECT_FIELDS]
        completeness = sum(1 for p in present if p) / len(_REFLECT_FIELDS)
        grounded = 1.0 if _references_error(" ".join(present), error_texts) else 0.0
        scores.append(0.5 * completeness + 0.5 * grounded)
    return sum(scores) / len(scores) if scores else 0.0


def _episode_error_texts(trace: list[dict]) -> list[str]:
    out: list[str] = []
    for t in trace:
        res = t.get("result") if isinstance(t, dict) else None
        if isinstance(res, dict):
            err = res.get("error")
            if isinstance(err, str) and err.strip():
                out.append(err.lower())
    return out


def _references_error(text: str, error_texts: list[str]) -> bool:
    """True if ``text`` shares a meaningful token with any observed error."""
    if not text or not error_texts:
        return False
    tokens = {w for w in re.findall(r"[a-z0-9_]{4,}", text.lower())}
    if not tokens:
        return False
    for err in error_texts:
        err_tokens = set(re.findall(r"[a-z0-9_]{4,}", err))
        if tokens & err_tokens:
            return True
    return False


def _keep_revert_score(decisions: list[dict]) -> float:
    """Signed, normalized score for keep/revert quality.

    +1 for a keep that improved (or a revert of a regression), -1 for the
    opposite. Averaged over decisions; 0 if there were none.
    """
    if not decisions:
        return 0.0
    s = 0.0
    for d in decisions:
        if d.get("action") == "keep":
            s += 1.0 if d.get("improved") else -1.0
        elif d.get("action") == "revert":
            s += 1.0 if d.get("was_regression") else -1.0
    return s / len(decisions)
