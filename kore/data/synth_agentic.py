"""Synthesize native agentic (Hermes tool-use) SFT trajectories from ALREADY-
VERIFIED KORE records - CPU-only, GPU-free, grounded in real measurements.

Why this exists
---------------
The agentic SFT slice teaches the multi-turn tool-use *skill*: call
build/test/bench, read the ``role:"tool"`` result, reflect on a failure, and
keep/revert. Generating it with the live :class:`~kore.agent.harness.AgentHarness`
re-executes real GPU tools for every turn of every trajectory
(``n_traj x turns x tasks``) - tens of GPU-hours.

But the *results* the agent would read are exactly the measurements we ALREADY
have on disk from datagen:

  * ``repair`` - a broken kernel + the verifier error + the FIXED kernel's SNR.
  * ``wins``   - a seed kernel + measured seed/final walltimes + speedup + SNR.
  * ``groups`` - several ranked candidates, each with a measured walltime + SNR.

This module reconstructs faithful Hermes tool-use trajectories from those
verified records using the EXACT executor result schema
(:mod:`kore.agent.tools`) and the EXACT renderers (:mod:`kore.agent.format`).
No teacher, no GPU, and no fabricated correctness: every ``role:"tool"`` result
carries a real measured number (walltime / SNR / error text) taken from the
record. The reconstructed trajectories map 1:1 onto the live agentic curriculum
categories, so the SFT mixer sees the same behavior distribution it would from
live rollouts, minus the cost:

  * repair -> "repair"  : test(broken) -> fail -> reflect -> test(fixed) -> keep
  * wins   -> "success" : bench(seed) -> bench(optimized) -> keep
  * groups -> "search"  : bench several candidates -> keep the fastest correct

Output ``AgenticTrajectoryRecord`` shards are written into ``<data_root>/agentic``
so :func:`kore.data.assemble._agentic_rows` picks them up automatically; the
existing web tool-use blend (xLAM / ToolACE, ~40% of the slice) supplies breadth.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterator, Optional

from kore.agent.format import arch_desc, build_agent_system_prompt, render_reflection, render_tool_result
from kore.agent.schema import AgenticTrajectoryRecord
from kore.data.schemas import (
    read_jsonl,
    stamp_production_record,
    write_jsonl,
)
from kore.obs import get_logger

log = get_logger("data.synth_agentic")

# Fallback ONLY when a source record carries no arch. Matches the registry's
# TRAIN_ARCH so the synthesized slice stays consistent with the rest of the
# corpus (kernel/repair/QA prompts) - the KORE target is gfx950/CDNA4 (MI350X).
DEFAULT_ARCH = "gfx950"

# --------------------------------------------------------------------------- #
# Small, pure helpers
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_SNR_RE = re.compile(r"snr_db\s*=\s*(-?\d+\.?\d*)")


def _fences(text: str) -> list[str]:
    return [m.group(1).strip() for m in _FENCE_RE.finditer(text or "")]


def _last_fence(text: str) -> Optional[str]:
    f = _fences(text)
    return f[-1] if f else None


def _largest_fence(text: str) -> Optional[str]:
    """The biggest code block - robust way to grab the full kernel out of a
    prompt that also contains a signature/reference snippet."""
    f = _fences(text)
    return max(f, key=len) if f else None


def _extract_think(text: str) -> str:
    m = _THINK_RE.search(text or "")
    return m.group(1).strip() if m else ""


def _parse_snr(text: str) -> Optional[float]:
    m = _SNR_RE.search(text or "")
    return round(float(m.group(1)), 2) if m else None


def _round(v: Any, nd: int = 4) -> Optional[float]:
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _ms(us: Any) -> Optional[float]:
    """Microseconds -> milliseconds (the harness reports wall_ms)."""
    v = _round(us, 6)
    return None if v is None else round(v / 1000.0, 6)


def _op_of(rec: dict) -> str:
    return str(rec.get("operation") or rec.get("operator") or rec.get("task_id") or "kernel")


def _arch_of(rec: dict, arch: Optional[str]) -> str:
    return str(arch or rec.get("arch") or rec.get("gpu") or DEFAULT_ARCH)


def _dtype_of(rec: dict) -> str:
    return str(rec.get("dtype") or "")


# --------------------------------------------------------------------------- #
# Prompt + message construction (mirrors the live harness format exactly)
# --------------------------------------------------------------------------- #
_GOALS = {
    "repair": (
        "The seed kernel below FAILS the verifier. Reproduce the failure with a "
        "tool, diagnose the exact cause, apply a targeted fix, re-verify "
        "correctness, then keep it."
    ),
    "optimize": (
        "Use your tools to build, test, and bench candidates. Reach a correct "
        "kernel first, then maximize the speedup vs the seed baseline; keep "
        "improvements and revert regressions."
    ),
    "search": (
        "Several candidate designs are available. Bench them with your tools, "
        "compare the walltimes, and keep the fastest correct one."
    ),
}


def _user_prompt(op: str, dtype: str, arch: str, seed_src: str, mode: str) -> str:
    desc = arch_desc(arch)
    dt = f" ({dtype})" if dtype else ""
    seed_block = (
        f"\n\n## Seed kernel\n```python\n{seed_src}\n```" if seed_src else ""
    )
    return (
        f"Optimize the `{op}` Triton kernel{dt} for {desc}.\n"
        f"{_GOALS.get(mode, _GOALS['optimize'])}{seed_block}"
    )


def _assistant(name: str, arguments: dict, *, thinking: str = "",
               reflection: Optional[dict] = None) -> dict:
    """One assistant turn: optional <think>, optional <reflect>, one <tool_call>."""
    parts: list[str] = []
    if thinking:
        parts.append(f"<think>\n{thinking.strip()}\n</think>")
    if reflection:
        parts.append(render_reflection(reflection))
    payload = {"name": name, "arguments": arguments}
    parts.append(f"<tool_call>\n{json.dumps(payload)}\n</tool_call>")
    return {"role": "assistant", "content": "\n".join(parts)}


def _step(messages: list[dict], trace: list[dict], turn: int, name: str,
          arguments: dict, result: dict, *, thinking: str = "",
          reflection: Optional[dict] = None) -> None:
    messages.append(_assistant(name, arguments, thinking=thinking, reflection=reflection))
    messages.append(render_tool_result(name, result))
    trace.append({
        "turn": turn, "name": name, "arguments": arguments,
        "valid_name": True, "valid_params": True, "malformed": False,
        "result": result,
    })


# --------------------------------------------------------------------------- #
# Grounded reflection text (derived ONLY from the real error/failure class)
# --------------------------------------------------------------------------- #
def _root_cause(failure_class: str, err: str) -> str:
    if failure_class == "compile_fail":
        return "the kernel did not compile - a build/type/indexing error, not a numeric one"
    if failure_class == "snr_fail":
        return "the kernel compiled but the output is numerically wrong (low SNR vs reference)"
    return err[:160] or "the kernel failed the verifier"


def _planned_fix(failure_class: str, real_think: str) -> str:
    if real_think:
        # Prefer the real teacher's stated plan when it's concise.
        line = real_think.strip().splitlines()[-1].strip()
        if 12 <= len(line) <= 240:
            return line
    if failure_class == "compile_fail":
        return "fix the compile error (types/masks/indexing) without changing the algorithm"
    return "restore fp32 accumulation and correct masking/indexing to fix the numerics"


# --------------------------------------------------------------------------- #
# Per-record synthesizers
# --------------------------------------------------------------------------- #
def synth_from_repair(rec: dict, arch: Optional[str] = None) -> Optional[AgenticTrajectoryRecord]:
    """repair record -> a test(broken)->reflect->test(fixed)->keep trajectory."""
    msgs = rec.get("messages") or []
    if len(msgs) < 3:
        return None
    broken = _largest_fence(msgs[1].get("content", ""))
    fixed = _last_fence(msgs[-1].get("content", ""))
    if not broken or not fixed:
        return None
    a = _arch_of(rec, arch)
    op = _op_of(rec)
    fclass = str(rec.get("failure_class") or "snr_fail")
    err = (rec.get("error_text") or "").strip()
    child_snr = rec.get("child_snr_db")
    real_think = _extract_think(msgs[-1].get("content", ""))

    messages = [
        {"role": "system", "content": build_agent_system_prompt(phase="correctness", arch=a)},
        {"role": "user", "content": _user_prompt(op, _dtype_of(rec), a, broken, "repair")},
    ]
    trace: list[dict] = []

    # Turn 0 - reproduce the failure (real measured failure signal).
    if fclass == "compile_fail":
        r0 = {"ok": False, "tool": "test", "compiled": False, "correct": False,
              "error": err or "compilation failed"}
    else:
        r0 = {"ok": False, "tool": "test", "compiled": True, "correct": False,
              "snr_db": _parse_snr(err), "error": err or "correctness failed (low SNR)"}
    _step(messages, trace, 0, "test", {"kernel_src": broken}, r0,
          thinking="Reproduce the reported failure on the current kernel before changing anything.")

    # Turn 1 - reflect on the real error, then verify the fixed kernel (real SNR).
    reflection = {
        "root_cause": _root_cause(fclass, err),
        "evidence": (err or "verifier rejected the kernel")[:200],
        "planned_fix": _planned_fix(fclass, real_think),
    }
    r1 = {"ok": True, "tool": "test", "compiled": True, "correct": True,
          "snr_db": _round(child_snr, 2)}
    _step(messages, trace, 1, "test", {"kernel_src": fixed}, r1,
          thinking=(real_think[:800] or "Apply the targeted fix and re-verify correctness."),
          reflection=reflection)

    # Turn 2 - keep.
    _step(messages, trace, 2, "keep", {},
          {"ok": True, "tool": "keep", "kept": True, "improved": True, "correct": True},
          thinking="Correctness restored on all shapes - commit this kernel.")

    return AgenticTrajectoryRecord(
        task_id=str(rec.get("task_id", "repair")),
        messages=messages, tool_trace=trace, best_kernel=fixed,
        best_reward=None, turns_to_best=1, success=True,
        reflections=[{"turn": 1, **reflection}],
        phase_trace=[{"turn": t, "phase": "correctness"} for t in range(3)],
        provenance={"category": "repair", "source": "synth", "synth_from": "repair",
                    "failure_class": fclass, "arch": a, "teacher": "synthetic-verified"},
        type="agentic", gpu=a,
    )


def synth_from_win(rec: dict, arch: Optional[str] = None) -> Optional[AgenticTrajectoryRecord]:
    """win record -> a bench(seed)->bench(optimized)->keep trajectory."""
    final = rec.get("final_source") or ""
    if "def " not in final:
        return None
    traj = rec.get("trajectory") or []
    seed = None
    first_think = ""
    for t in traj:
        if t.get("role") == "assistant":
            seed = _last_fence(t.get("content", ""))
            first_think = _extract_think(t.get("content", ""))
            if seed:
                break
    if not seed:
        return None
    a = _arch_of(rec, arch)
    op = _op_of(rec)
    iw, fw = rec.get("initial_wall_us"), rec.get("final_wall_us")
    sp, snr = rec.get("speedup"), rec.get("snr_db")

    messages = [
        {"role": "system", "content": build_agent_system_prompt(phase="optimize", arch=a)},
        {"role": "user", "content": _user_prompt(op, _dtype_of(rec), a, seed, "optimize")},
    ]
    trace: list[dict] = []

    r0 = {"ok": True, "tool": "bench", "compiled": True, "correct": True,
          "wall_ms": _ms(iw), "snr_db": _round(snr, 2)}
    _step(messages, trace, 0, "bench", {"kernel_src": seed}, r0,
          thinking=(first_think[:600] or "Bench the seed to establish the reference walltime."))

    r1 = {"ok": True, "tool": "bench", "compiled": True, "correct": True,
          "speedup": _round(sp, 3), "wall_ms": _ms(fw), "baseline_ms": _ms(iw),
          "snr_db": _round(snr, 2)}
    _step(messages, trace, 1, "bench", {"kernel_src": final}, r1,
          thinking="Apply the optimization and bench again; keep it only if it stays correct AND is faster.")

    _step(messages, trace, 2, "keep", {},
          {"ok": True, "tool": "keep", "kept": True, "improved": True, "correct": True},
          thinking="Faster and still correct - commit it.")

    return AgenticTrajectoryRecord(
        task_id=str(rec.get("task_id", "win")),
        messages=messages, tool_trace=trace, best_kernel=final,
        best_reward=None, turns_to_best=1, success=True, reflections=[],
        phase_trace=[{"turn": t, "phase": "optimize"} for t in range(3)],
        provenance={"category": "success", "source": "synth", "synth_from": "wins",
                    "speedup": _round(sp, 3), "arch": a, "teacher": "synthetic-verified"},
        type="agentic", gpu=a,
    )


def synth_from_group(rec: dict, arch: Optional[str] = None,
                     max_bench: int = 3) -> Optional[AgenticTrajectoryRecord]:
    """group record -> a bench(several candidates)->keep(best) trajectory."""
    cands = [c for c in (rec.get("candidates") or []) if c.get("source")]
    if len(cands) < 2:
        return None

    def _key(c: dict) -> tuple:
        r = c.get("rank")
        return (r if r is not None else 1_000_000, c.get("wall_us") or 1e18)

    ordered = sorted(cands, key=_key)
    chosen = ordered[:max_bench]
    seq = list(reversed(chosen))  # explore worse->better, keep the best last
    best = chosen[0]
    ref_wall = seq[0].get("wall_us")

    a = _arch_of(rec, arch)
    op = _op_of(rec)
    messages = [
        {"role": "system", "content": build_agent_system_prompt(phase="optimize", arch=a)},
        {"role": "user", "content": _user_prompt(op, _dtype_of(rec), a, seq[0].get("source", ""), "search")},
    ]
    trace: list[dict] = []
    for i, c in enumerate(seq):
        wall = c.get("wall_us")
        res = {"ok": True, "tool": "bench", "compiled": True, "correct": True,
               "wall_ms": _ms(wall), "snr_db": _round(c.get("snr_db"), 2)}
        if i > 0 and ref_wall and wall:
            res["speedup"] = _round(ref_wall / wall, 3)
            res["baseline_ms"] = _ms(ref_wall)
        think = ("Bench the first candidate design to set a reference walltime."
                 if i == 0 else f"Bench candidate design #{i + 1} and compare against the best so far.")
        _step(messages, trace, i, "bench", {"kernel_src": c.get("source", "")}, res, thinking=think)

    _step(messages, trace, len(seq), "keep", {},
          {"ok": True, "tool": "keep", "kept": True, "improved": True, "correct": True},
          thinking="This candidate has the lowest walltime and is still correct - commit it.")

    return AgenticTrajectoryRecord(
        task_id=str(rec.get("task_id", "group")),
        messages=messages, tool_trace=trace, best_kernel=best.get("source", ""),
        best_reward=None, turns_to_best=len(seq) - 1, success=True, reflections=[],
        phase_trace=[{"turn": t, "phase": "optimize"} for t in range(len(seq) + 1)],
        provenance={"category": "search", "source": "synth", "synth_from": "groups",
                    "n_candidates": len(chosen), "arch": a, "teacher": "synthetic-verified"},
        type="agentic", gpu=a,
    )


# --------------------------------------------------------------------------- #
# Validation + driver
# --------------------------------------------------------------------------- #
def _valid(rec: AgenticTrajectoryRecord) -> bool:
    """A synthesized record must be a clean multi-turn tool-use transcript."""
    msgs = rec.messages
    if len(msgs) < 4:
        return False
    if msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
        return False
    for m in msgs:
        if not isinstance(m.get("content"), str) or not m["content"]:
            return False
    # Must end on a kept decision and carry at least one parseable tool call.
    from kore.agent.format import parse_tool_calls
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    if len(assistants) < 2:
        return False
    if not any(parse_tool_calls(a["content"]) for a in assistants):
        return False
    last_tool = [m for m in msgs if m.get("role") == "tool"][-1]
    return '"tool": "keep"' in last_tool.get("content", "")


def _iter_records(d: Path) -> Iterator[dict]:
    if not d.exists():
        return
    for p in sorted(d.glob("*.jsonl")):
        # Skip synthetic/derived shards ("_gold_from_groups", "_repair_pairs",
        # "_synth_*") so the synthesizers never re-ingest their own or each other's
        # outputs (which would, e.g., turn repair-DPO broken kernels into agentic
        # bench candidates). Only real per-task datagen shards are read.
        if p.name.startswith("_"):
            continue
        try:
            if p.stat().st_size == 0:
                continue
        except OSError:
            continue
        for rec in read_jsonl(
            p, typed=False, mode="generic_training_row"):
            if isinstance(rec, dict):
                yield rec


_SYNTH_FN = {
    "repair": synth_from_repair,
    "wins": synth_from_win,
    "groups": synth_from_group,
}
DEFAULT_MIX = (("repair", 0.5), ("groups", 0.3), ("wins", 0.2))


def synthesize_agentic(
    data_root: Any, *, cap: int = 4000, seed: int = 0,
    arch: Optional[str] = None, mix: tuple = DEFAULT_MIX,
    write: bool = True, prefix: str = "_synth",
) -> dict:
    """Build up to ``cap`` native agentic trajectories from verified records.

    Reads ``<data_root>/{repair,wins,groups}`` (never touched) and writes
    ``<data_root>/agentic/{prefix}_{kind}.jsonl`` so the SFT build picks them up.
    ``arch=None`` stamps each trajectory with its source record's arch (keeping
    the slice consistent with the rest of the corpus); pass an explicit slug to
    override. Returns a per-category count summary. Deterministic given ``seed``.
    """
    data_root = Path(data_root)
    rng = random.Random(seed)
    budgets = {k: max(0, int(round(cap * frac))) for k, frac in mix}
    built: dict[str, list[AgenticTrajectoryRecord]] = {k: [] for k in _SYNTH_FN}
    # Held-out generalization tasks (paged-KV decode, MLA) must NEVER be synthesized
    # into agentic SFT trajectories - otherwise they leak into training and invalidate
    # the eval split (audit C2). The kernel repair/win/group slices are already
    # train-filtered; this closes the agentic bypass at the source.
    from kore.tasks.registry import HELDOUT_TASKS
    n_heldout_skipped = 0

    for kind in _SYNTH_FN:
        want = budgets.get(kind, 0)
        if want <= 0:
            continue
        recs = list(_iter_records(data_root / kind))
        rng.shuffle(recs)
        fn = _SYNTH_FN[kind]
        for r in recs:
            if len(built[kind]) >= want:
                break
            if isinstance(r, dict) and r.get("task_id") in HELDOUT_TASKS:
                n_heldout_skipped += 1
                continue
            try:
                out = fn(r, arch)
            except Exception as e:  # noqa: BLE001 - one bad record must not abort
                log.debug("synth_skip", kind=kind, err=str(e)[:120])
                out = None
            if out is not None and _valid(out):
                built[kind].append(out)

    summary: dict[str, int] = {}
    agdir = data_root / "agentic"
    if write:
        agdir.mkdir(parents=True, exist_ok=True)
    for kind, recs in built.items():
        summary[kind] = len(recs)
        if write and recs:
            rows = [
                stamp_production_record(
                    record,
                    provenance_id="synth_agentic_v1",
                    evaluation_id=f"synth_agentic:{kind}:{index}",
                )
                for index, record in enumerate(recs)
            ]
            write_jsonl(agdir / f"{prefix}_{kind}.jsonl", rows)
    summary["total"] = sum(len(v) for v in built.values())
    summary["heldout_skipped"] = n_heldout_skipped
    log.event("agentic_synth_done", cap=cap, arch=str(arch), **summary)
    return summary


__all__ = [
    "synth_from_repair", "synth_from_win", "synth_from_group",
    "synthesize_agentic",
]
