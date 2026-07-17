"""AgentHarness: the multi-turn tool-orchestration loop.

The policy (or a teacher) drives its own optimize loop. Each turn:

  render messages -> model.generate() -> parse Hermes/OpenAI tool calls
    -> execute each via ToolExecutor -> append role:"tool" result -> repeat

The trajectory is scored by the BEST correct kernel reached (Kevin), and
keep/revert are recorded as first-class (trainable) decisions. The GPU-bound
work lives entirely in the injected ``env``; the harness itself is CPU-only and
deterministic given a deterministic model + env.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional

from kore.agent.format import (
    arch_desc,
    build_agent_system_prompt,
    parse_reflection,
    parse_tool_calls,
    render_tool_result,
)
from kore.agent.tools import TOOL_SCHEMAS, ToolExecutor, validate_tool_call
from kore.config import CONFIG
from kore.data.mutate import infer_family
from kore.data.schemas import read_jsonl
from kore.obs import get_logger

log = get_logger("agent.harness")

# Phase labels for the correctness->optimization split.
PHASE_CORRECTNESS = "correctness"
PHASE_OPTIMIZE = "optimize"


def _summarize_tool_result(result) -> str:
    """Compact one-line summary of a tool result dict (for DEBUG turn logs)."""
    if not isinstance(result, dict):
        return str(result)[:80]
    keys = ("tool", "ok", "compiled", "correct", "tier", "speedup", "reward",
            "kept", "improved", "reverted", "was_regression", "error")
    parts = [f"{k}={result[k]}" for k in keys if k in result]
    return " ".join(parts)[:200]


@dataclass
class AgentEpisode:
    """The full record of one agentic optimize episode.

    RL CONTRACT (for the GRPO agentic path): ``turn_rewards``/``turn_correct``
    are per-turn parallel arrays (length == ``turns_used``) carrying the
    ToolExecutor's *verified* reward and correctness of the current
    candidate/best kernel at the end of each turn. The GRPO loop reads these to
    apply per-turn Kevin credit; ``best_reward``/``turns_to_best``/``success``
    remain the trajectory-level summary.
    """

    task_id: str
    messages: list[dict] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)
    keep_decisions: list[dict] = field(default_factory=list)
    best_kernel: Optional[str] = None
    best_reward: Optional[float] = None
    turns_to_best: Optional[int] = None
    turns_used: int = 0
    committed_kernel: Optional[str] = None
    success: bool = False
    # --- GEAK-style cognition additions --- #
    turn_rewards: list[float] = field(default_factory=list)   # per-turn verified reward
    turn_correct: list[bool] = field(default_factory=list)    # per-turn correctness
    turn_speedups: list = field(default_factory=list)         # per-turn MEASURED speedup (or None)
    turn_phis: list = field(default_factory=list)             # per-turn roofline potential Phi=rho (or None)
    turn_codes: list[str] = field(default_factory=list)       # per-turn candidate kernel source
    reflections: list[dict] = field(default_factory=list)     # structured reflect turns
    phase_trace: list[dict] = field(default_factory=list)     # [{turn, phase}]
    reseeds: list[dict] = field(default_factory=list)         # trap-avoidance restarts

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "messages": self.messages,
            "tool_trace": self.tool_trace,
            "keep_decisions": self.keep_decisions,
            "best_kernel": self.best_kernel,
            "best_reward": self.best_reward,
            "turns_to_best": self.turns_to_best,
            "turns_used": self.turns_used,
            "committed_kernel": self.committed_kernel,
            "success": self.success,
            "turn_rewards": self.turn_rewards,
            "turn_correct": self.turn_correct,
            "turn_speedups": self.turn_speedups,
            "turn_phis": self.turn_phis,
            "turn_codes": self.turn_codes,
            "reflections": self.reflections,
            "phase_trace": self.phase_trace,
            "reseeds": self.reseeds,
        }


# --------------------------------------------------------------------------- #
# Inference-time knowledge base (GEAK KB): retrieve prior winning kernels
# --------------------------------------------------------------------------- #
def _dtype_from_task_id(task_id: str) -> str:
    """Best-effort dtype extraction from a task id (e.g. gemm_bf16 -> bf16)."""
    s = (task_id or "").lower()
    for d in ("bf16", "fp16", "fp8", "mxfp8", "mxfp4", "fp4", "fp32", "int8", "int4"):
        if d in s:
            return d
    return ""


class WinsKB:
    """Offline, deterministic retriever over prior winning kernels.

    Reads ``WinRecord``s from ``data/wins/*.jsonl`` (produced by ``gen_wins``)
    and indexes them by ``(op_family, dtype)`` so the harness can inject the
    top-k fastest prior wins for the *same kind* of kernel as few-shot context
    into the opening user prompt. No network; pure filesystem. When no wins
    exist the KB is simply empty and every lookup is a safe no-op.
    """

    def __init__(self, entries: Optional[list[dict]] = None):
        self.entries: list[dict] = entries or []

    @classmethod
    def from_dir(cls, wins_dir) -> "WinsKB":
        entries: list[dict] = []
        if not wins_dir or not os.path.isdir(wins_dir):
            return cls(entries)
        for path in sorted(glob.glob(os.path.join(wins_dir, "*.jsonl"))):
            for rec in read_jsonl(path, typed=False):
                if not isinstance(rec, dict) or rec.get("type") != "win":
                    continue
                src = rec.get("final_source") or ""
                if not src:
                    continue
                op = rec.get("operation") or rec.get("task_id") or ""
                entries.append({
                    "task_id": rec.get("task_id", ""),
                    "family": infer_family(op or rec.get("task_id", "")),
                    "dtype": _dtype_from_task_id(rec.get("task_id", "")),
                    "speedup": rec.get("speedup") or 0.0,
                    "final_source": src,
                    "snr_db": rec.get("snr_db"),
                })
        # Deterministic order: fastest first, then by task_id for tie-break.
        entries.sort(key=lambda e: (-(e["speedup"] or 0.0), e["task_id"]))
        return cls(entries)

    def retrieve(self, op: str, dtype: str, k: int = 2) -> list[dict]:
        """Top-k prior wins for this op family (+dtype preferred). Deterministic."""
        if not self.entries or k <= 0:
            return []
        family = infer_family(op or "")
        dtype = (dtype or "").lower()
        exact = [e for e in self.entries if e["family"] == family and e["dtype"] == dtype]
        fam_only = [e for e in self.entries if e["family"] == family]
        # Prefer op+dtype matches, then same-family, then nothing (no cross-op leak).
        ranked = exact + [e for e in fam_only if e not in exact]
        return ranked[:k]


def _render_kb_context(wins: list[dict]) -> str:
    """Render retrieved wins as a few-shot 'prior winning kernels' block."""
    if not wins:
        return ""
    parts = ["\n\n## Prior winning kernels (reference - adapt, do not copy blindly)"]
    for i, w in enumerate(wins, 1):
        su = w.get("speedup")
        head = f"### Example {i}: {w.get('task_id', 'win')}"
        if isinstance(su, (int, float)):
            head += f" (≈{su:.2f}x vs baseline)"
        parts.append(head + f"\n```python\n{w['final_source']}\n```")
    return "\n".join(parts)


def build_agent_user_prompt(task, seed_src: str = "", kb_context: str = "") -> str:
    """The opening user turn: the task + the seed kernel + optional KB few-shots."""
    op = getattr(task, "operation", None) or getattr(task, "task_id", "kernel")
    dtype = getattr(task, "dtype", "fp32")
    seed_block = f"\n\n## Seed kernel (optimize this)\n```python\n{seed_src}\n```" if seed_src else ""
    return (
        f"Optimize the `{op}` Triton kernel ({dtype}) for {arch_desc(getattr(task, 'gpu_target', None))}.\n"
        "Use your tools to build, test, and bench candidates. Keep improvements "
        "and revert regressions. Reach a correct kernel first, then maximize the "
        f"speedup vs the production baseline.{seed_block}{kb_context}"
    )


class AgentHarness:
    """Run the multi-turn tool loop for a single task with an injected env.

    Beyond the plain build/test/bench/keep/revert loop, the harness now adds
    GEAK-style cognition (all CPU-only / deterministic given a deterministic
    model + env):

      * per-turn verified reward trace (RL contract for the GRPO agentic path),
      * a structured reflection turn parsed after failures,
      * correctness->optimization phase split (system prompt swaps on the first
        correct kernel),
      * debugging-trap avoidance (re-seed a fresh lineage after ``reseed_patience``
        consecutive non-improving turns),
      * an inference-time knowledge base that injects prior winning kernels as
        few-shot context into the opening user prompt.
    """

    def __init__(
        self,
        task,
        model_or_teacher,
        env,
        max_turns: int = 8,
        tools: Optional[list[dict]] = None,
        seed_src: Optional[str] = None,
        system_prompt: Optional[str] = None,
        reseed_patience: int = 3,
        kb: Optional[WinsKB] = None,
        wins_dir: Optional[str] = None,
        kb_top_k: int = 2,
        use_kb: bool = True,
    ):
        self.task = task
        self.model = model_or_teacher
        self.env = env
        self.max_turns = max_turns
        self.tools = tools or TOOL_SCHEMAS
        # A fixed system_prompt (if provided) disables the phase split; otherwise
        # the harness swaps correctness/optimize prompts as the episode evolves.
        self.fixed_system_prompt = system_prompt
        if seed_src is None:
            seed_src = _safe_seed(task)
        self.seed_src = seed_src or ""
        self.executor = ToolExecutor(env, task, seed_src=self.seed_src or None)
        self.reseed_patience = max(1, int(reseed_patience))
        self.kb_top_k = kb_top_k
        if kb is not None:
            self.kb = kb
        elif use_kb:
            wins_dir = wins_dir or str(CONFIG.data_dir / "wins")
            self.kb = WinsKB.from_dir(wins_dir)
        else:
            self.kb = WinsKB([])

    def _system_prompt_for(self, phase: str) -> str:
        if self.fixed_system_prompt is not None:
            return self.fixed_system_prompt
        return build_agent_system_prompt(
            self.tools, phase=phase,
            arch=getattr(self.task, "gpu_target", None))

    def _kb_context(self) -> str:
        wins = self.kb.retrieve(
            getattr(self.task, "operation", "") or getattr(self.task, "task_id", ""),
            getattr(self.task, "dtype", ""),
            k=self.kb_top_k,
        )
        return _render_kb_context(wins)

    def run(self) -> AgentEpisode:
        task_id = getattr(self.task, "task_id", "task")
        ex = self.executor
        phase = PHASE_CORRECTNESS
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt_for(phase)},
            {"role": "user",
             "content": build_agent_user_prompt(self.task, self.seed_src, self._kb_context())},
        ]
        tool_trace: list[dict] = []
        turn_rewards: list[float] = []
        turn_correct: list[bool] = []
        turn_speedups: list = []
        turn_phis: list = []
        turn_codes: list[str] = []
        reflections: list[dict] = []
        phase_trace: list[dict] = []
        reseeds: list[dict] = []
        turns_used = 0
        stall = 0

        for turn in range(self.max_turns):
            phase_trace.append({"turn": turn, "phase": phase})
            best_before = ex.best_reward

            text = self.model.generate(messages)
            messages.append({"role": "assistant", "content": text})
            turns_used = turn + 1

            # Structured reflection (GEAK/Reflexion): parsed block, not a tool.
            refl = parse_reflection(text)
            if refl is not None:
                refl_entry = {"turn": turn, **refl}
                reflections.append(refl_entry)
                log.debug("agent_reflection", task=task_id, turn=turn,
                          root_cause=refl.get("root_cause", "")[:120])

            calls = parse_tool_calls(text)
            if calls:
                ex.set_turn(turn)
                for call in calls:
                    v = validate_tool_call(call)
                    result = ex.dispatch(call, turn=turn)
                    name = call.get("name") or "unknown"
                    tool_trace.append({
                        "turn": turn,
                        "name": call.get("name"),
                        "arguments": call.get("arguments", {}),
                        "valid_name": v["valid_name"],
                        "valid_params": v["valid_params"],
                        "malformed": bool(call.get("malformed")),
                        "result": result,
                    })
                    messages.append(render_tool_result(name, result))
                    _br = ex.best_reward
                    log.debug("agent_turn", task=task_id, turn=turn, tool=name,
                              valid=bool(v["valid_name"] and v["valid_params"]),
                              malformed=bool(call.get("malformed")),
                              result_summary=_summarize_tool_result(result),
                              best_reward=(_br if _br != float("-inf") else None))

            # --- per-turn verified reward trace (RL contract): one entry/turn --- #
            self._record_turn(ex, turn_rewards, turn_correct, turn_speedups,
                              turn_codes, turn_phis)

            # --- correctness -> optimization phase split --- #
            if phase == PHASE_CORRECTNESS and ex.best_src is not None:
                phase = PHASE_OPTIMIZE
                messages[0] = {"role": "system",
                               "content": self._system_prompt_for(phase)}
                log.event("agent_phase_switch", task=task_id, turn=turn,
                          phase=phase, best_reward=ex.best_reward)

            # A turn with neither a tool call nor a reflection -> model is done.
            if not calls and refl is None:
                break

            # --- debugging-trap avoidance --- #
            improved = ex.best_reward > best_before
            stall = 0 if improved else stall + 1
            if stall >= self.reseed_patience and turn < self.max_turns - 1:
                info = ex.reseed_lineage()
                info.update({"turn": turn, "stall": stall})
                reseeds.append(info)
                messages.append({
                    "role": "user",
                    "content": (
                        f"No progress after {stall} turns - abandon the current "
                        "candidate lineage. Re-seed a FRESH 1-shot design from the "
                        "task/seed below and try a structurally different approach."
                        + build_agent_user_prompt(self.task, self.seed_src)
                    ),
                })
                log.event("agent_reseed", task=task_id, turn=turn, stall=stall)
                stall = 0

        best_reward = ex.best_reward if ex.best_reward != float("-inf") else None
        success = ex.best_src is not None
        log.event("agent_episode_done", task=task_id, turns=turns_used, success=success,
                  best_reward=best_reward, turns_to_best=ex.best_turn,
                  n_reflections=len(reflections), n_reseeds=len(reseeds),
                  final_phase=phase)
        return AgentEpisode(
            task_id=task_id,
            messages=messages,
            tool_trace=tool_trace,
            keep_decisions=list(ex.keep_decisions),
            best_kernel=ex.best_src,
            best_reward=best_reward,
            turns_to_best=ex.best_turn,
            turns_used=turns_used,
            committed_kernel=ex.committed_src,
            success=success,
            turn_rewards=turn_rewards,
            turn_correct=turn_correct,
            turn_speedups=turn_speedups,
            turn_phis=turn_phis,
            turn_codes=turn_codes,
            reflections=reflections,
            phase_trace=phase_trace,
            reseeds=reseeds,
        )

    @staticmethod
    def _record_turn(ex: ToolExecutor, turn_rewards: list[float],
                     turn_correct: list[bool], turn_speedups: list,
                     turn_codes: list[str], turn_phis: list) -> None:
        """Append the verified reward/correctness/speedup/source of this turn.

        Uses the candidate evaluated this turn when present; otherwise carries
        the best-so-far reward (0.0 before any candidate). This is the per-turn
        signal the GRPO agentic path folds in as Kevin credit AND (new) surfaces
        as per-turn ``speedups`` + ``codes`` so agentic wins reach co-evolution
        distillation and the open-ended controller exactly like the serial path.
        The four arrays are populated in lockstep (one entry per turn), so they
        stay index-aligned with ``_HFChatPolicy.turn_inputs``.
        """
        if ex.candidate_reward is not None:
            r = float(ex.candidate_reward)
            c = bool(ex.candidate_correct)
            su = ex.candidate_speedup
            phi = ex.candidate_phi
            code = ex.candidate_src or ""
        elif ex.best_reward != float("-inf"):
            r = float(ex.best_reward)
            c = True
            su = ex.best_speedup
            phi = None  # best-so-far carries no fresh potential this turn
            code = ex.best_src or ""
        else:
            r = 0.0
            c = False
            su = None
            phi = None
            code = ""
        # EXACT reward (no display rounding): turn_rewards is the per-turn GRPO
        # Kevin-credit signal, so it must match best_reward bit-for-bit. Rounding
        # here silently discards reward contrast the advantage estimator needs.
        turn_rewards.append(float(r))
        turn_correct.append(c)
        # Measured speedup only when the turn actually benched a correct candidate
        # (else None), so a correctness-only turn never invents a timing number.
        turn_speedups.append(float(su) if su is not None else None)
        turn_phis.append(float(phi) if phi is not None else None)
        turn_codes.append(code)


def _safe_seed(task) -> str:
    try:
        return task.seed_source
    except Exception:  # noqa: BLE001 - FakeEnv tasks may have no seed file
        return ""
