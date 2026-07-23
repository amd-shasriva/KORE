"""Generate winning trajectories (KORE Stage 3 seed / SFT-on-wins).

A short greedy evolve loop: the parent is the best-so-far kernel, the teacher
proposes a rewrite conditioned on the last verifier feedback, we verify it, and
keep it as the new best only if it is correct AND meaningfully faster
(wall < best_wall * 0.98). The full multi-turn chat is stored as a ``WinRecord``
if the loop achieved any net speedup over the initial kernel.

CRITICAL (audited): the *stored* trajectory is NOT the raw search log. Raw greedy
search oscillates - the wall bounces, most turns are dead-ends/regressions, and a
naive dump stores non-convergent noise as a "win", with footer metrics that don't
even multiply out and analyses that describe changes the emitted code never made.
:func:`build_convergent_trajectory` reconstructs a CLEAN win from the recorded
turns:

  1. keep ONLY the turns on the strictly-improving path to the best kept kernel;
  2. drop any kept turn whose ANALYSIS claims a change its FULL_KERNEL diff does
     not actually implement ("describe but don't implement");
  3. regenerate every feedback string from the KEPT measurements and recompute
     the footer (initial / final / speedup) so they are internally consistent and
     multiply out exactly;
  4. optionally fold one real regression into an explicit 2-turn "tried X,
     measured slower, reverted" lesson - genuine negative signal, not noise.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from typing import Optional

from kore.config import CONFIG
from kore.data.amd_knowledge import ExperienceLedger, live_system_prompt
from kore.data.grounded_reasoning import (
    _transform_hint,
    collect_counters as _collect_counters,
    diagnose_bottleneck_rich as _diagnose_rich,
)
from kore.data.prompts import (
    SYSTEM_PROMPT,
    build_turn_prompt,
    extract_kernel,
    normalize_assistant,
)
from kore.data.schemas import WinRecord
from kore.obs import get_logger
from kore.policy.format import format_assistant_turn, parse_response
from kore.reward.reward import compute_reward

log = get_logger("data.gen_wins")

_IMPROVE_FACTOR = 0.98  # a kept step must beat best wall by >= 2%

# Tier 2 (PMC-guided feedback): collecting rocprofv3 counters is a few extra
# profiled replays, so it is gated + budgeted per trajectory. On by default;
# degrades to wall-only feedback when the profiler is unavailable (CPU box / no
# rocprof) or the flag is cleared.
_PMC_ON = os.environ.get("KORE_WINS_PMC", "1") != "0"
_MAX_PMC_COLLECTS = int(os.environ.get("KORE_WINS_PMC_MAX", "4"))


# --------------------------------------------------------------------------- #
# Feedback formatting (single source of truth so live + reconstructed strings
# are byte-identical and every "speedup=Xx" is consistent with its "wall=Yus").
# --------------------------------------------------------------------------- #
def _us(w: Optional[float]) -> str:
    return f"{w:.1f}us" if w is not None else "n/a"


def _x(s: Optional[float]) -> str:
    return f"{s:.3f}x" if s is not None else "n/a"


def _self_speedup(initial: Optional[float], wall: Optional[float]) -> Optional[float]:
    """Speedup of ``wall`` relative to the trajectory's own initial/seed wall.

    Every stored per-turn speedup is measured against the SAME initial baseline,
    so the numbers are internally consistent and the footer (initial/final) is
    just the last turn's reported speedup."""
    if initial and wall and wall > 0:
        return initial / wall
    return None


def _fmt_correct_feedback(wall_us: Optional[float], speedup: Optional[float],
                          suffix: str = "Now make it faster with one more change.") -> str:
    return f"Correct? YES. wall={_us(wall_us)} speedup={_x(speedup)}. {suffix}"


def _bottleneck_feedback(counters: Optional[dict]) -> str:
    """Tier 2: turn measured rocprofv3 counters into a targeted next-move hint so the
    teacher optimizes the REAL limiter (memory / LDS / matrix-cores / occupancy)
    instead of guessing "make it faster". Empty string when no counters / unknown."""
    if not counters:
        return ""
    try:
        label, evidence = _diagnose_rich(counters)
        if not label or label == "unknown":
            return ""
        return (f"\nHARDWARE COUNTERS (rocprofv3): {label} - {evidence}. "
                f"Target this next: {_transform_hint(label)}.")
    except Exception:  # noqa: BLE001 - counter feedback is advisory; never fatal
        return ""


def _feedback(obs, rr, counters: Optional[dict] = None) -> str:
    # error_text is Optional[str]: it is None for a compiled-but-incorrect kernel
    # (an SNR failure carries no error string), so guard before slicing - otherwise
    # a correctness miss (common on the tighter fp16 SNR thresholds) crashes the
    # whole wins shard with 'NoneType' is not subscriptable.
    err = obs.error_text or ""
    if not obs.compiled:
        return f"FAILED to compile: {err[:400]}"
    if not rr.correct:
        return (
            f"Correct? NO. snr_db={obs.snr_db}. {err[:200]}\n"
            "Fix correctness before optimizing further."
        )
    wall_us = obs.wall_ms * 1000.0 if obs.wall_ms is not None else None
    # wall_ms/speedup can be None when timing is unmeasurable on this stack
    # (e.g. fp8 on ROCm) - format defensively so the wins shard isn't lost.
    base = _fmt_correct_feedback(wall_us, rr.speedup)
    bn = _bottleneck_feedback(counters)  # Tier 2: append counter-grounded diagnosis
    return f"{base}{bn}" if bn else base


# --------------------------------------------------------------------------- #
# Structured per-turn record + trajectory reconstruction
# --------------------------------------------------------------------------- #
@dataclass
class WinTurn:
    """One raw evolve turn as measured by the verifier (pre-reconstruction)."""

    response: str            # raw teacher text
    cand_src: str            # extracted candidate kernel
    correct: bool            # passed the correctness/SNR gate
    wall_us: Optional[float]  # measured wall time (us), or None if unmeasurable
    snr_db: Optional[float] = None
    mode: str = "exploit"


def _src_tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", s or "")


def _changed_token_set(a: str, b: str) -> set[str]:
    """Tokens added or removed between two kernel sources."""
    at, bt = _src_tokens(a), _src_tokens(b)
    sm = difflib.SequenceMatcher(a=at, b=bt, autojunk=False)
    out: set[str] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out.update(at[i1:i2])
        out.update(bt[j1:j2])
    return out


_KNOB_NAMES = ("num_warps", "num_stages", "block_m", "block_n", "block_k", "group_m")
_KNOB_RE = re.compile(
    r"\b(num_warps|num_stages|BLOCK_M|BLOCK_N|BLOCK_K|GROUP_M)\b"
    r"\s*(?::\s*tl\.constexpr\s*)?=\s*(\d+)", re.IGNORECASE)


def _knob_values(src: str) -> dict[str, int]:
    """Parse tunable-knob values (kwargs, single, and tuple assignment forms)."""
    out: dict[str, int] = {}
    for m in _KNOB_RE.finditer(src or ""):
        out[m.group(1).lower()] = int(m.group(2))
    for m in re.finditer(r"^[ \t]*([A-Za-z_][\w,\s]*?)=\s*([\d,\s]+)$", src or "",
                         re.MULTILINE):
        names = [n.strip().lower() for n in m.group(1).split(",")]
        vals = [v.strip() for v in m.group(2).split(",")]
        if len(names) == len(vals):
            for n, v in zip(names, vals):
                if n in _KNOB_NAMES and v.isdigit():
                    out[n] = int(v)
    return out


def _changed_knobs(prev_src: str, cur_src: str) -> set[str]:
    """Knobs whose value (or presence) differs between two sources."""
    pv, cv = _knob_values(prev_src), _knob_values(cur_src)
    return {k for k in set(pv) | set(cv) if pv.get(k) != cv.get(k)}


# A claimed change -> the concrete diff tokens that prove it was implemented.
_CLAIM_KNOBS: dict[str, set[str]] = {
    "num_warps": {"num_warps"},
    "num_stages": {"num_stages"},
    "block": {"block_m", "block_n", "block_k", "group_m"},
    "vectorize": {"block_k"},
    "group": {"group_m"},
}
_CLAIM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "num_warps": ("num_warps", "warp", "warps"),
    "num_stages": ("num_stages", "num stages", "stage", "stages", "pipeline"),
    "block": ("block_m", "block_n", "block_k", "block size", "block-size",
              "tile", "tiling", "blocking"),
    "vectorize": ("vectoriz", "vector width", "coalesc"),
    "group": ("group_m", "grouping", "group size"),
}


def _claim_implemented(response: str, prev_src: str, cur_src: str) -> bool:
    """Is the turn's stated change actually present in its FULL_KERNEL diff?

    Guards against "describe but don't implement" turns: if the ANALYSIS/
    PROPOSED_CHANGE names a CONCRETE knob (num_warps, num_stages, BLOCK_*, ...) but
    the prev->cur diff touches none of them, the claim is unsupported and the turn
    is dropped. A vague claim (no concrete knob) cannot be disproved, so it is kept.
    """
    parsed = parse_response(response or "")
    claim = f"{parsed.get('analysis', '')} {parsed.get('proposed_change', '')}".lower()
    if not claim.strip():
        return True
    claimed = [k for k, kws in _CLAIM_KEYWORDS.items() if any(w in claim for w in kws)]
    if not claimed:
        return True
    # Evidence a knob was actually touched: its parsed value changed, OR the knob
    # identifier itself appears in the raw token diff (covers newly-introduced knobs
    # / structural rewrites where parsing the assignment form may miss it).
    changed = _changed_knobs(prev_src, cur_src)
    tokens = {t.lower() for t in _changed_token_set(prev_src, cur_src)}
    return any((_CLAIM_KNOBS[k] & changed) or (_CLAIM_KNOBS[k] & tokens) for k in claimed)


def _pick_regression(regressions: list[dict], slower_than: Optional[float]) -> Optional[dict]:
    """The most clearly-slower correct regression (for the revert lesson)."""
    cands = [r for r in regressions if r.get("wall") is not None and r.get("src")]
    if slower_than is not None:
        cands = [r for r in cands if r["wall"] > slower_than]
    if not cands:
        return None
    # deterministic: the slowest (most obviously worse); tie-break by source text.
    return max(cands, key=lambda r: (r["wall"], r["src"]))


def _lesson_messages(pivot_src: str, pivot_wall: Optional[float],
                     initial_wall: Optional[float], regression: dict) -> list[dict]:
    """A 2-exchange "tried X, measured slower, reverted" lesson that starts and
    ends at ``pivot_src`` (so it never changes the final kernel / footer)."""
    r_src = regression["src"]
    r_wall = regression["wall"]
    pivot_sp = _self_speedup(initial_wall, pivot_wall)
    # 1. explore prompt from the pivot kernel.
    u1 = build_turn_prompt(
        parent_source=pivot_src,
        feedback=_fmt_correct_feedback(
            pivot_wall, pivot_sp,
            "Incremental tuning has plateaued - try a structural change."),
        mode="explore",
    )
    # 2. assistant emits the (real) regression candidate. Keep the teacher's own
    #    proposed change only if it is grounded in the pivot->R diff, else stay vague.
    parsed = parse_response(regression.get("response", ""))
    proposed = (parsed.get("proposed_change") or "").strip()
    if not proposed or not _claim_implemented(regression.get("response", ""), pivot_src, r_src):
        proposed = "Apply a structural change to escape the plateau."
    a1 = format_assistant_turn(
        "Incremental tuning has plateaued; trying a structural change to look for a "
        "faster region of the search space.",
        proposed, r_src)
    # 3. feedback: it measured slower -> revert.
    r_sp = _self_speedup(initial_wall, r_wall)
    u2 = (f"Correct? YES. wall={_us(r_wall)} speedup={_x(r_sp)}. That is SLOWER than "
          f"the previous {_us(pivot_wall)} - the structural change regressed. Revert to "
          f"the faster kernel and try a different optimization.")
    # 4. assistant reverts to the pivot kernel (net no change to the best).
    a2 = format_assistant_turn(
        f"The structural change measured {_us(r_wall)} vs {_us(pivot_wall)} (slower), so "
        f"it is a regression. Reverting to the faster kernel before trying another idea.",
        "Revert to the previous faster kernel.", pivot_src)
    return [
        {"role": "user", "content": u1},
        {"role": "assistant", "content": a1},
        {"role": "user", "content": u2},
        {"role": "assistant", "content": a2},
    ]


def build_convergent_trajectory(
    seed_src: str,
    initial_wall: Optional[float],
    initial_snr: Optional[float],
    turns: list[WinTurn],
    *,
    improve_factor: float = _IMPROVE_FACTOR,
    include_regression_lesson: bool = True,
) -> Optional[dict]:
    """Reconstruct a clean, convergent win from raw evolve ``turns``.

    Returns a dict with ``messages`` (the stored chat), ``initial_wall_us``,
    ``final_wall_us``, ``speedup``, ``final_source`` and ``snr_db`` - all mutually
    consistent (``initial == final * speedup`` exactly, feedback numbers match the
    kept turns) - or ``None`` when there is no net, verified, convergent win.
    """
    # 1. Walk the raw turns; keep the strictly-improving path; log regressions.
    best_wall, best_src, best_snr = initial_wall, seed_src, initial_snr
    improving: list[dict] = []
    regressions: list[dict] = []
    for t in turns:
        if not t.correct or not t.cand_src:
            continue
        w = t.wall_us
        if w is not None and best_wall is not None and w < best_wall * improve_factor:
            improving.append({"response": t.response, "src": t.cand_src,
                              "wall": w, "snr": t.snr_db, "prev_wall": best_wall,
                              "prev_src": best_src})
            best_wall, best_src, best_snr = w, t.cand_src, t.snr_db
        elif w is not None and best_wall is not None and w > best_wall:
            regressions.append({"response": t.response, "src": t.cand_src, "wall": w})

    # 2. Claim-vs-diff validation: drop turns that describe-but-don't-implement,
    #    re-parenting each kept turn to the previous KEPT kernel.
    kept: list[dict] = []
    prev_src = seed_src
    for step in improving:
        if step["src"] == prev_src:
            continue  # no real change (can't have improved without changing code)
        if not _claim_implemented(step["response"], prev_src, step["src"]):
            continue
        kept.append({**step, "prev_src": prev_src})
        prev_src = step["src"]
    if not kept:
        return None

    # 3. Footer recomputed from the kept turns (guaranteed to multiply out).
    final_src = kept[-1]["src"]
    final_wall = kept[-1]["wall"]
    final_snr = kept[-1]["snr"]
    speedup = _self_speedup(initial_wall, final_wall)
    if speedup is None or speedup <= 1.0:
        return None

    # 4. Assemble the chat with regenerated, consistent feedback.
    lesson = None
    if include_regression_lesson:
        pivot_wall = kept[-1]["prev_wall"]
        lesson = _pick_regression(regressions, pivot_wall)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    prev_src, prev_wall = seed_src, initial_wall
    for i, step in enumerate(kept):
        last = i == len(kept) - 1
        if last and lesson is not None:
            # Fold the regression in right before the final improving turn; the
            # lesson returns to prev_src, so the final turn proceeds unchanged.
            messages += _lesson_messages(prev_src, prev_wall, initial_wall, lesson)
        user_fb = _fmt_correct_feedback(prev_wall, _self_speedup(initial_wall, prev_wall))
        messages.append({"role": "user",
                         "content": build_turn_prompt(parent_source=prev_src,
                                                      feedback=user_fb, mode="exploit")})
        messages.append({"role": "assistant",
                         "content": normalize_assistant(step["response"])})
        prev_src, prev_wall = step["src"], step["wall"]

    return {
        "messages": messages,
        "initial_wall_us": initial_wall,
        "final_wall_us": final_wall,
        "speedup": speedup,
        "final_source": final_src,
        "snr_db": final_snr,
    }


def generate_wins(
    task,
    teacher,
    env,
    gens: int,
    cfg=CONFIG,
    *,
    include_regression_lesson: bool = True,
    ledger: Optional[ExperienceLedger] = None,
) -> list[WinRecord]:
    """Run a single evolve trajectory of ``gens`` turns; return [WinRecord] if it
    produced a net, verified, convergent speedup, else [].

    Search-intelligence layers (adapted from Hyperloom-Forge), all improving the
    LIVE generation only - the STORED SFT trajectory keeps the canonical contract:
      * Tier 1 - the live teacher context is primed with the AMD-Triton playbook
        (:func:`kore.data.amd_knowledge.live_system_prompt`) so the teacher applies
        gfx950/CDNA4 discipline from the first move.
      * Tier 2 - the seed and every KEPT improvement are re-profiled and the
        counter-diagnosed bottleneck is fed back so the next change targets the real
        limiter (budgeted + gated by ``KORE_WINS_PMC``; degrades to wall-only).
      * Tier 3 - failed / regressed attempts distill into a deduped 'do-NOT-repeat'
        ``ledger`` injected into later turns; pass a shared ledger across a task's
        ``deepen_wins`` trajectories so no dead-end is re-walked.
    """
    if ledger is None:
        ledger = ExperienceLedger()
    _pmc_left = [_MAX_PMC_COLLECTS if _PMC_ON else 0]
    with log.stage("generate_wins", task=task.task_id, gens=gens):
        seed_src = task.seed_source
        best_src = seed_src

        def _counters(src):
            """Budgeted, fail-safe rocprofv3 counter collection (Tier 2)."""
            if _pmc_left[0] <= 0:
                return None
            c = _collect_counters(env, src)
            if c:
                _pmc_left[0] -= 1
            return c

        # Measure the seed as the starting point.
        obs = env.step(seed_src, full_validation=True, multi_shape=True)
        rr = compute_reward(obs, seed_src, dtype=task.dtype, cfg=cfg)
        initial_wall = obs.wall_ms * 1000.0 if obs.wall_ms is not None else None
        initial_snr = obs.snr_db
        best_wall = initial_wall
        best_snr = obs.snr_db
        # Tier 2: profile the seed so turn 1 already targets the real bottleneck.
        seed_counters = _counters(seed_src) if rr.correct else None

        # ``context`` is the LIVE chat fed to the teacher (multi-turn generation is
        # unchanged); ``turns`` records the structured, verifier-measured outcome of
        # each turn so the STORED trajectory can be reconstructed cleanly afterwards.
        # Tier 1: prime the LIVE system prompt with the AMD-Triton playbook; the
        # STORED trajectory (build_convergent_trajectory) keeps the plain SYSTEM_PROMPT.
        context: list[dict] = [{"role": "system", "content": live_system_prompt(SYSTEM_PROMPT)}]
        feedback = _feedback(obs, rr, counters=seed_counters)
        mode = "exploit"
        turns: list[WinTurn] = []

        def _emit_turn(turn: int, turn_mode: str, improved: bool) -> None:
            sp = _self_speedup(initial_wall, best_wall)
            log.event(
                "win_turn", task=task.task_id, turn=turn, mode=turn_mode,
                improved=improved, best_wall_us=best_wall, best_snr=best_snr,
                speedup=sp,
            )
            log.progress(turn + 1, gens, "wins", best_wall_us=best_wall,
                         best_snr=best_snr, speedup=sp)

        for turn in range(gens):
            turn_mode = mode
            improved = False
            prompt = build_turn_prompt(parent_source=best_src, feedback=feedback,
                                       tuning_hints=ledger.render(), mode=mode)
            context.append({"role": "user", "content": prompt})
            response = teacher.generate(context)
            # Store the assistant turn in the CANONICAL contract (Pillar 0): the raw
            # teacher text may be loosely shaped; normalize_assistant re-renders it to
            # ANALYSIS/PROPOSED_CHANGE/FULL_KERNEL so the live context never drifts.
            context.append({"role": "assistant", "content": normalize_assistant(response)})

            cand_src = extract_kernel(response)
            if not cand_src:
                feedback = "No kernel found in your response. Output a full FULL_KERNEL block."
                ledger.record(outcome="no kernel emitted")
                mode = "repair"
                _emit_turn(turn, turn_mode, improved)
                continue

            try:
                c_obs = env.step(cand_src, full_validation=True, multi_shape=True)
            except Exception as e:
                feedback = f"Verifier crashed: {str(e)[:200]}"
                ledger.record(error_text=str(e), outcome="verifier crashed")
                mode = "repair"
                _emit_turn(turn, turn_mode, improved)
                continue

            c_rr = compute_reward(c_obs, cand_src, dtype=task.dtype, cfg=cfg)
            cand_wall = c_obs.wall_ms * 1000.0 if c_obs.wall_ms is not None else None
            turns.append(WinTurn(response=response, cand_src=cand_src,
                                 correct=bool(c_rr.correct), wall_us=cand_wall,
                                 snr_db=c_obs.snr_db, mode=turn_mode))

            if not c_rr.correct:
                feedback = _feedback(c_obs, c_rr)
                ledger.record(error_text=c_obs.error_text or "",
                              outcome=f"incorrect (snr_db={c_obs.snr_db})")  # Tier 3
                mode = "repair"
                _emit_turn(turn, turn_mode, improved)
                continue

            improved = (
                cand_wall is not None
                and best_wall is not None
                and cand_wall < best_wall * _IMPROVE_FACTOR
            )
            if improved:
                best_src = cand_src
                best_wall = cand_wall
                best_snr = c_obs.snr_db
                mode = "exploit"
                # Tier 2: re-profile the new best so the next turn targets its bottleneck.
                feedback = _feedback(c_obs, c_rr, counters=_counters(cand_src))
            else:
                # correct but NOT faster: record the dead-end, pivot dimension (Tier 3).
                feedback = _feedback(c_obs, c_rr)
                ledger.record(outcome="correct but not faster than the current best")
                mode = "explore"  # plateau -> try a structural change next
            _emit_turn(turn, turn_mode, improved)

        built = build_convergent_trajectory(
            seed_src, initial_wall, initial_snr, turns,
            improve_factor=_IMPROVE_FACTOR,
            include_regression_lesson=include_regression_lesson,
        )
        is_win = built is not None
        log.metric(
            "wins_summary", task=task.task_id, turns=gens, is_win=is_win,
            speedup=(built["speedup"] if is_win else None),
            initial_wall_us=initial_wall,
            final_wall_us=(built["final_wall_us"] if is_win else best_wall),
            best_snr=(built["snr_db"] if is_win else best_snr),
            kept_turns=(sum(1 for m in built["messages"] if m["role"] == "assistant")
                        if is_win else 0),
        )
        if not is_win:
            return []

        return [
            WinRecord(
                task_id=task.task_id,
                trajectory=built["messages"],
                initial_wall_us=built["initial_wall_us"],
                final_wall_us=built["final_wall_us"],
                speedup=built["speedup"],
                final_source=built["final_source"],
                snr_db=built["snr_db"],
                gpu=task.gpu_target,
                operation=getattr(task, "operation", None),
                arch=getattr(task, "gpu_target", None),
            )
        ]
