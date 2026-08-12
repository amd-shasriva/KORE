#!/usr/bin/env python
"""Stage 4 of the v5 build: assemble the mixture.

WHAT WENT WRONG IN V4, AND WHAT THIS FIXES

v4 scored 55.1% on AgentKernelArena against 55.9% for the base model it was
fine-tuned from. Its 69,851 rows carried 38,460 kernel examples in four shapes:
repair, multi-turn refine, step-centric, and kernel QA. Every one of them asks a
variant of one question -- "here is a kernel, improve or fix it".

The arena asks five questions, and on three of them v4 had never seen the shape:

    torch2hip      57 tasks   PyTorch module -> HIP      v4  7.5%, 70% miscompile
    torch2flydsl   45 tasks   PyTorch module -> FlyDSL   v4  0.0%
    triton2flydsl  51 tasks   port between dialects      v4 72.0%, 14pp BELOW base
    instruction2*  31 tasks   written spec -> kernel     v4 61.3%

The results track that exactly: v4 gained where training resembled the task
(instruction2triton +9.7, triton2triton +3.0) and lost precisely where the shape
was missing. So v5's job is not more kernels, it is the missing questions, and
three measured properties of the corpus set the weights.

**Correctness is what the arena pays for.** 92% of v4's passes scored exactly
120 -- the flat correct-but-no-speedup value -- because the harness ran without
reference latencies and cannot award a speedup term for most categories. Data
that teaches a kernel to be *right* outranks data that teaches it to be fast.

**Redundancy, not scarcity, was the constraint.** 35% of mined rows were
cross-root duplicates and repair carried 9.24 answers per distinct problem. v4
trained on that unthinned, so its 19,587 repair rows held far fewer than 19,587
lessons.

**FlyDSL is 24.7% of the benchmark and ~1% of the corpus.** This mixture cannot
close that with data that does not exist; it upsamples what there is, which is
defensible for a category where the model actively regressed, and reports the
residual gap honestly rather than hiding it behind a ratio.

Replay is held near v4's 44.9%, inside the 30-50% band that prevents catastrophic
forgetting, and reuses v4's own general rows -- they are not kernel data, so they
carry none of the redundancy this build removed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import random
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

REPO = Path("/home/shasriva/Kore-RL/KORE")
sys.path.insert(0, str(REPO))

#: The rebuilt replay mixture, budgeted in tokens. Falls back to v4's general
#: slices only if it has not been built.
V5_REPLAY = REPO / "data/v5_replay.jsonl"
V4_SFT = REPO / "data/b05factory/sft/multicap_v3.jsonl"

#: kernel_qa is deliberately ABSENT. It is GPU domain content that was filed as
#: replay, and counting ~20M tokens of it as general capability is most of why the
#: replay share looked healthy while the genuinely general part was 17% of tokens.
#: It is loaded on the kernel side instead.
REPLAY_SOURCES = ("general_chat", "general_code", "math_reasoning",
                  "instruction_following", "agentic_tooluse")

#: Scarce slices are repeated to reach a floor share. Repetition is a real cost --
#: it trades diversity for coverage -- so it is capped and applied only where the
#: benchmark demands a language the corpus barely contains.
FLYDSL_MAX_REPEAT = 6


def as_dict(rec) -> dict:
    if isinstance(rec, dict):
        return rec
    if is_dataclass(rec):
        try:
            return asdict(rec)
        except Exception:  # noqa: BLE001
            pass
    return {k: getattr(rec, k) for k in dir(rec)
            if not k.startswith("_") and not callable(getattr(rec, k, None))}


def content_hash(messages) -> str:
    body = "\n".join(
        f"{m.get('role')}:{m.get('content')}" for m in messages
        if isinstance(m, dict))
    return hashlib.sha1(body.encode("utf-8", "ignore")).hexdigest()


def to_row(messages, source: str, task_id: str, **meta) -> dict:
    from kore.data.v5_policy import dialect
    return {"messages": messages, "_source": source, "_task_id": task_id,
            "_dialect": dialect(task_id), **meta}


#: A target this short is a token, not a demonstration.
MIN_TARGET_CHARS = 24

#: No single answer may appear more than this often. Found by measurement: the
#: most-repeated target in the first build was a bare ``revert`` tool call
#: appearing 543 times, and the replay side contributed 304 copies of the single
#: word "arnold". Neither teaches anything, and repeated identical targets are the
#: duplication that costs capability rather than merely wasting tokens.
MAX_TARGET_REPEATS = 12

_CODEBLOCK = re.compile(r"```[a-zA-Z+]*\n(.*?)```", re.S)

_REVERT = re.compile(r'"name"\s*:\s*"(?:revert|undo|noop|no_op|pass|skip)"', re.I)


def degenerate(target: str) -> str:
    """Why this target is not worth imitating, or '' if it is fine."""
    t = (target or "").strip()
    if len(t) < MIN_TARGET_CHARS:
        return "too_short"
    if _REVERT.search(t):
        return "revert_or_noop"
    return ""


def sanitize(rows: list[dict]) -> tuple[list[dict], collections.Counter]:
    """Enforce the trainer's contract and drop targets not worth imitating.

    Four rules, each answering a specific way a row can be accepted by the loader
    and still be wrong:

    * ``_provenance`` must be an object. The loader calls ``.get("kind")`` on it
      whenever ``repair_loss_weight >= 1.5``, and production sets 2.0, so a bare
      string raises after the 30B model has loaded on every rank.
    * Every assistant turn carries full loss and there is no per-turn opt-out, so
      a multi-turn row is collapsed until exactly one assistant turn remains.
    * An empty assistant turn trains the stop token alone and trips no guard --
      it teaches the model to answer immediately with nothing.
    * A target that delegates to torch passed the numerical harness and would
      score 20, not 120, at evaluation while teaching the fallback.
    """
    from kore.data.v5_emit import assistant_turns, cheats, flatten_history

    out: list[dict] = []
    stats: collections.Counter = collections.Counter()
    for r in rows:
        msgs = r.get("messages") or []
        if not msgs:
            stats["no_messages"] += 1
            continue
        if assistant_turns(msgs) > 1:
            msgs = flatten_history(msgs)
            stats["flattened"] += 1
        if assistant_turns(msgs) != 1:
            stats["no_single_assistant"] += 1
            continue
        bad = False
        for m in msgs:
            if not isinstance(m, dict) or "content" not in m or "role" not in m:
                bad = True
                break
            if m.get("role") not in ("system", "user", "assistant", "tool"):
                bad = True
                break
            if not str(m.get("content") or "").strip():
                bad = True
                break
        if bad:
            stats["malformed_message"] += 1
            continue
        target = msgs[-1].get("content") or ""
        why = cheats(target)
        if why:
            stats[f"cheat::{why.split(':')[0]}"] += 1
            continue
        why2 = degenerate(target)
        if why2:
            stats[f"degenerate::{why2}"] += 1
            continue
        prov = r.get("_provenance")
        if not isinstance(prov, dict):
            r["_provenance"] = ({"kind": str(prov)} if isinstance(prov, str) and prov
                                else {"kind": r.get("_source") or "v5"})
            stats["provenance_coerced"] += 1
        # "repair" is the one value the loader acts on -- it duplicates the row.
        # Slices that are not repair must not accidentally claim to be.
        if r["_provenance"].get("kind") == "repair" and r.get("_source") != "kernel_repair":
            r["_provenance"]["kind"] = str(r.get("_source") or "v5")
        r["messages"] = msgs
        out.append(r)
    return out, stats


def enforce_length(rows: list[dict], model_id: str, revision: str, limit: int
                   ) -> tuple[list[dict], collections.Counter]:
    """Drop rows over ``limit`` real tokens, measured the way the loader measures.

    The v3 build gated on a characters/3.6 estimate, and 209 of its rows still
    exceeded the true cap and were silently dropped at train time. Gating here
    with the actual tokenizer means the count lands in the build receipt instead
    of one aggregate log line, and it is a drop rather than a truncation --
    truncation is right-side, which decapitates the answer.
    """
    stats: collections.Counter = collections.Counter()
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    except Exception as exc:  # noqa: BLE001 - no tokenizer is a warning, not a stop
        print(f"  WARN: tokenizer unavailable ({type(exc).__name__}); "
              f"skipping length gate", file=sys.stderr)
        stats["tokenizer_unavailable"] = 1
        return rows, stats
    kept = []
    for r in rows:
        try:
            n = len(tok.apply_chat_template(r["messages"], tokenize=True,
                                            add_generation_prompt=False))
        except Exception:  # noqa: BLE001 - a row that will not render is a row that will crash
            stats["render_failed"] += 1
            continue
        if n > limit:
            stats[f"over_limit::{r.get('_source')}"] += 1
            continue
        r["_tokens"] = n
        kept.append(r)
    return kept, stats


def screen(rows: list[dict], policy: str, arena) -> tuple[list[dict], collections.Counter]:
    """The authoritative final gate, applied to every row whatever its origin.

    Upstream stages each filter, but they were written at different times against
    different rules and stage 1's cache predates the benchmark screen entirely.
    Re-asking here means one place decides what is trainable, and a row that slips
    a stage cannot reach the mixture. Rows carrying no task identity -- the general
    replay slices -- are not kernel tasks and are passed through.
    """
    from kore.data.v5_policy import admits

    kept: list[dict] = []
    blocked: collections.Counter = collections.Counter()
    for r in rows:
        tid = str(r.get("_task_id") or "")
        if not tid:
            kept.append(r)
            continue
        ok, why = admits({"task_id": tid, "arch": "gfx950"}, policy, arena)
        if ok:
            kept.append(r)
        else:
            blocked[why.split(":")[0]] += 1
    return kept, blocked


#: Extra slices recovered outside the main stages, each already emitted in final
#: row form. ``language`` is the FlyDSL anchor; ``modality`` and ``dpo_chosen``
#: carry their own shape tags.
EXTRA_SLICES = ("v5_modality.jsonl", "v5_dpo_chosen.jsonl", "v5_flydsl_lang.jsonl")


def shape_for(source: str) -> str:
    """The skill a slice teaches, from its source name.

    Prefix matching rather than a fixed-width slice: an earlier version keyed on
    the first seven characters, which turned ``kernel_torch2hip`` into
    ``"torch2h"``, missed the ``"torch2"`` key, and silently routed the largest
    and most benchmark-relevant slice into "other" where the planner dropped it.
    """
    name = source.replace("kernel_", "", 1)
    if name.startswith("torch2"):
        return "torch2kernel"
    if name.startswith(("triton2", "translate")):
        return "port"
    if name.startswith("instruction"):
        return "instruction"
    if name.startswith(("flydsl_language", "language")):
        return "language"
    if name.startswith("repair"):
        return "repair"
    return "optimize"


def load_extra(rows: list[dict]) -> list[dict]:
    for name in EXTRA_SLICES:
        p = REPO / "data" / name
        if not p.is_file():
            continue
        with p.open(errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if not d.get("messages"):
                    continue
                src = str(d.get("_source") or "")
                d["_shape"] = ("language" if "flydsl_language" in src
                               else d.get("_shape") or shape_for(src))
                rows.append(d)
    return rows


def load_kernel_rows(stage1: dict, stage3: dict, translate_path: Path) -> list[dict]:
    """Every kernel-specific row, tagged by the skill it teaches."""
    rows: list[dict] = []

    for r in stage1["repairs"]:
        d = as_dict(r)
        msgs = d.get("messages") or d.get("trajectory") or []
        if msgs:
            rows.append(to_row(msgs, "kernel_repair", str(d.get("task_id") or ""),
                               _shape="repair",
                               _provenance={"kind": "repair"}))

    for r in stage1["wins"]:
        d = as_dict(r)
        msgs = d.get("trajectory") or d.get("messages") or []
        if msgs:
            rows.append(to_row(msgs, "kernel_win", str(d.get("task_id") or ""),
                               _shape="optimize", _speedup=d.get("speedup")))

    for r in stage3.get("gold_wins") or []:
        d = as_dict(r)
        msgs = d.get("trajectory") or d.get("messages") or []
        if msgs:
            rows.append(to_row(msgs, "kernel_gold_win", str(d.get("task_id") or ""),
                               _shape="optimize", _speedup=d.get("speedup")))

    for r in stage3.get("step_rows") or []:
        msgs = r.get("messages") or []
        if msgs:
            rows.append(to_row(
                msgs, "kernel_step_centric", str(r.get("_task_id") or ""),
                _shape="optimize", _kind=r.get("_kind"), _gain=r.get("_gain")))

    # kernel_qa: GPU domain Q&A, previously counted as replay.
    v4 = REPO / "data/b05factory/sft/multicap_v3.jsonl"
    if v4.is_file():
        with v4.open(errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if str(d.get("_source") or "") == "kernel_qa" and d.get("messages"):
                    rows.append({"messages": d["messages"], "_source": "kernel_qa",
                                 "_task_id": "", "_dialect": "Triton",
                                 "_shape": "language",
                                 "_provenance": {"kind": "kernel_qa"}})

    if translate_path.is_file():
        with translate_path.open(errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                src = str(d.get("_source") or "")
                rows.append(to_row(d.get("messages") or [], src,
                                   str(d.get("_task_id") or ""),
                                   _shape=shape_for(src),
                                   _provenance=d.get("_provenance")
                                   or {"kind": "verified_twin"}))
    return load_extra(rows)


def load_replay(path: Path) -> list[dict]:
    """Prefer the rebuilt token-budgeted mixture; fall back to v4's slices."""
    out: list[dict] = []
    if V5_REPLAY.is_file():
        with V5_REPLAY.open(errors="ignore") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if d.get("messages"):
                    out.append(d)
        if out:
            return out
    if not path.is_file():
        return out
    with path.open(errors="ignore") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if str(d.get("_source") or "") in REPLAY_SOURCES and d.get("messages"):
                out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("strict", "audited"), default="strict")
    ap.add_argument("--out", default=str(REPO / "data/v5_sft.jsonl"))
    ap.add_argument("--replay-target", type=float, default=0.14,
                    help="replay share of TOKENS. Measured evidence puts the "
                         "forgetting-mitigation plateau at 5-10% of tokens with no "
                         "further benefit above it, and anchoring at ~1%; the 25% "
                         "figure comes from continued pretraining under a much "
                         "larger distribution shift. 14% sits above the plateau "
                         "without spending budget the kernel side can use.")
    ap.add_argument("--max-upsample", type=float, default=2.0,
                    help="cap on repeating a scarce slice; 2x keeps repetition "
                         "mild while 4x drove it to 45%")
    ap.add_argument("--use-all", action="store_true", default=True,
                    help="keep every distinct row; balance by upsampling only")
    ap.add_argument("--fixed-body", dest="use_all", action="store_false",
                    help="downsample to --body instead (discards distinct rows)")
    ap.add_argument("--flydsl-analysis-max", type=float, default=0.5,
                    help="max share of FlyDSL TOKENS that may be repair-shaped "
                         "(ANALYSIS preamble); measured at 81.8% before this")
    ap.add_argument("--flydsl-target", type=float, default=0.05,
                    help="FlyDSL share of the kernel body; an anchor, not a cure")
    ap.add_argument("--body", type=int, default=55000,
                    help="kernel-specific rows; 55k keeps repetition near 10%")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arena-index", default=str(REPO / "data/arena_contamination.json"))
    ap.add_argument("--allow-unscreened", action="store_true",
                    help="emit without benchmark screening (not for training)")
    ap.add_argument("--model-id", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--revision", default="b2cff646eb4bb1d68355c01b18ae02e7cf42d120")
    # 17,408 is the configured cap; the margin leaves room for a later template
    # tweak to add tokens without silently deleting rows at train time.
    ap.add_argument("--token-limit", type=int, default=16896)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    build = REPO / "runs/v5_build"
    with (build / "stage1.pkl").open("rb") as fh:
        stage1 = pickle.load(fh)
    s3p = build / f"stage3_{args.policy}.pkl"
    stage3 = {}
    if s3p.is_file():
        with s3p.open("rb") as fh:
            stage3 = pickle.load(fh)
    tpath = REPO / f"data/v5_translate_{args.policy}.jsonl"

    from kore.data.arena_index import ArenaIndex
    arena = None
    if Path(args.arena_index).is_file():
        arena = ArenaIndex.load(args.arena_index)
        print(f"arena screen         : {arena}")
    elif not args.allow_unscreened:
        print(f"ERROR: no arena index at {args.arena_index}; refusing to emit "
              f"unscreened training data.", file=sys.stderr)
        return 2

    kernel = load_kernel_rows(stage1, stage3, tpath)
    print(f"kernel rows gathered : {len(kernel):,}")

    kernel, blocked = screen(kernel, args.policy, arena)
    print(f"after final screen   : {len(kernel):,} "
          f"({sum(blocked.values()):,} blocked)")
    for k, v in blocked.most_common():
        print(f"    blocked::{k:<28} {v:,}")

    kernel, sstats = sanitize(kernel)
    print(f"after sanitize       : {len(kernel):,}")
    for k, v in sstats.most_common():
        print(f"    {k:<32} {v:,}")

    # Final content-level dedup. The slices are built from overlapping artifacts
    # -- a gold win and a step row can land on the same kernel -- and identical
    # message bodies are the one duplication that survives every earlier key.
    seen: set[str] = set()
    deduped = []
    for r in kernel:
        h = content_hash(r["messages"])
        if h in seen:
            continue
        seen.add(h)
        deduped.append(r)
    print(f"after content dedup  : {len(deduped):,} "
          f"({len(kernel) - len(deduped):,} exact-body duplicates removed)")
    kernel = deduped

    # Solve the composition rather than accept whatever the pipeline produced.
    # v4's mixture was the latter and it was a shape monoculture.
    from kore.data.v5_plan import (BENCHMARK_SHAPES, GENERAL_SHAPES,
                                   describe_gap, solve, solve_use_all)

    by_shape_rows: dict[str, list[dict]] = collections.defaultdict(list)
    for r in kernel:
        by_shape_rows[str(r.get("_shape") or "other")].append(r)
    available = {k: len(v) for k, v in by_shape_rows.items()}
    plan = (solve_use_all(available, GENERAL_SHAPES, args.max_upsample)
            if args.use_all
            else solve(available, GENERAL_SHAPES, args.body))
    print(f"\n=== composition plan ({'use-all' if args.use_all else f'body {args.body:,}'}) ===")
    print(plan.table())
    print(f"repetition {100 * plan.repetition_rate:.1f}%  "
          f"unique {plan.unique_total:,}")

    # A dialect floor, applied after the shape plan. FlyDSL is a quarter of the
    # evaluation surface and under one percent of the corpus, and the base model
    # scores 86% on porting to it while v4 scores 72% -- so what is needed is
    # enough presence to stop the fine-tune driving the language out, not enough
    # to teach it. Training longer on a small in-language set has been measured to
    # hurt rather than help, so this is capped hard and reported honestly rather
    # than inflated to look like the benchmark.
    def apply_dialect_floor(pool: list[dict], want: float, cap: float = 4.0
                            ) -> list[dict]:
        have = [r for r in pool if r.get("_dialect") == "FlyDSL"]
        if not have or want <= 0:
            return []
        need = int(want * len(pool) / max(1e-9, 1 - want))
        # `have` is already upsampled by the shape plan, so a further 4x here
        # compounded to 6-8x and left FlyDSL as ~273 distinct items repeated 13
        # times. Budget against DISTINCT rows and cap the total multiple.
        distinct = len({content_hash(r["messages"]) for r in have})
        mult = min(cap, max(1.0, need / max(1, len(have))))
        if distinct and (len(have) * mult) / distinct > cap:
            mult = max(1.0, (cap * distinct) / len(have))
        print(f"  (FlyDSL distinct rows {distinct:,}; total repetition capped "
              f"at {cap:g}x of distinct)")
        extra = int(len(have) * (mult - 1))
        print(f"FlyDSL floor         : {len(have):,} distinct x{mult:.2f} "
              f"-> +{extra:,} (target {100*want:.1f}%)")
        return [dict(have[i % len(have)], _repeat=1) for i in range(extra)]

    def rebalance_flydsl_format(pool: list[dict], max_analysis_share: float = 0.5
                                ) -> list[dict]:
        """Hold repair-shaped FlyDSL rows to a share of FlyDSL tokens.

        Measured on the previous build, 81.8% of FlyDSL tokens opened with an
        ``ANALYSIS:`` preamble, because every FlyDSL repair and gold-win row has
        that shape while only the translate and language rows are direct code. The
        arena's FlyDSL tasks ask for a port -- code, not a preamble about a broken
        kernel -- so the dominant FlyDSL gradient was training the wrong output
        format for the task it is evaluated on. That is a better explanation of the
        observed regression than share, since FlyDSL already sits above the share
        at which the literature says an ability is anchored.

        Dropping the excess rather than adding volume: the fix is the ratio.
        """
        fly = [r for r in pool if r.get("_dialect") == "FlyDSL"]
        if not fly:
            return pool
        def is_analysis(r):
            body = (r["messages"][-1].get("content") or "").lstrip()
            return body.upper().startswith("ANALYSIS")
        def toks(r):
            return int(r.get("_tokens") or 0) or max(
                1, len(r["messages"][-1].get("content") or "") // 4)
        ana = [r for r in fly if is_analysis(r)]
        direct = [r for r in fly if not is_analysis(r)]
        t_ana = sum(toks(r) for r in ana)
        t_dir = sum(toks(r) for r in direct)
        if not ana or t_ana + t_dir == 0:
            return pool
        share = t_ana / (t_ana + t_dir)
        if share <= max_analysis_share:
            return pool
        # Largest first: the longest ANALYSIS rows carry the most of the skew.
        allowed = max_analysis_share / (1 - max_analysis_share) * t_dir
        keep, running = [], 0
        for r in sorted(ana, key=toks):
            if running + toks(r) > allowed:
                continue
            keep.append(r)
            running += toks(r)
        dropped = len(ana) - len(keep)
        print(f"FlyDSL format        : ANALYSIS-shaped {100*share:.1f}% -> "
              f"{100*running/max(1,running+t_dir):.1f}% of FlyDSL tokens "
              f"(dropped {dropped:,} of {len(ana):,} repair-shaped rows)")
        drop_ids = {id(r) for r in ana} - {id(r) for r in keep}
        return [r for r in pool if id(r) not in drop_ids]

    selected: list[dict] = []
    for res in plan.results:
        pool = by_shape_rows.get(res.name, [])
        if not pool or res.emitted <= 0:
            continue
        rng.shuffle(pool)
        take = pool[:res.emitted]
        # Any shortfall is made up by repeating the slice, which is why the plan
        # caps the multiple: repeated targets are the duplication that costs
        # capability, not the harmless kind.
        while len(take) < res.emitted:
            take += pool[:res.emitted - len(take)]
        for i, r in enumerate(take):
            selected.append(dict(r) if i < len(pool) else dict(r, _repeat=1))
    selected = rebalance_flydsl_format(selected, args.flydsl_analysis_max)
    selected += apply_dialect_floor(selected, args.flydsl_target)
    kernel = selected

    # Measure the kernel side before budgeting replay against it.
    kernel, klen = enforce_length(kernel, args.model_id, args.revision,
                                  args.token_limit)
    if klen:
        print(f"kernel length gate   : {len(kernel):,} kept")
        for k, v in klen.most_common(6):
            print(f"    {k:<38} {v:,}")

    replay_pool = load_replay(V4_SFT)
    rng.shuffle(replay_pool)
    # Select replay by TOKENS, not rows. Kernel rows average ~3,500 tokens and
    # replay rows ~730, so matching a row share undershoots the token share by
    # roughly five times: an earlier build hit 42% of rows and only 13.5% of
    # tokens, below the range where replay reliably prevents forgetting. Tokens
    # are the unit the optimizer sees and the unit every mixture result is stated
    # in, so that is what the target means here.
    kernel_tok = sum(int(r.get("_tokens") or 0) for r in kernel)
    if kernel_tok <= 0:
        kernel_tok = sum(len(str(m.get("content") or "")) for r in kernel
                         for m in r["messages"]) // 4
    want_tok = int(args.replay_target / (1 - args.replay_target) * kernel_tok)
    replay, got = [], 0
    for r in replay_pool:
        if got >= want_tok:
            break
        n = int(r.get("_tokens") or 0) or max(1, sum(
            len(str(m.get("content") or "")) for m in r["messages"]) // 4)
        replay.append(r)
        got += n
    print(f"replay               : {len(replay):,} rows / {got:,} tokens "
          f"of {len(replay_pool):,} available "
          f"(target {100*args.replay_target:.0f}% of tokens, kernel {kernel_tok:,})")

    final = kernel + replay
    # Cap how often any one answer may appear, across kernel and replay alike.
    # Upsampling and public corpora both produce runs of an identical target, and
    # the earlier build shipped one answer 543 times.
    tally: collections.Counter = collections.Counter()
    capped, dropped_rep = [], collections.Counter()
    for r in final:
        msgs = r.get("messages") or []
        body = (msgs[-1].get("content") or "") if msgs else ""
        # Key on the extracted code. Hashing the whole message lets one kernel
        # appear many times under different prompts and still look distinct --
        # which is how a 12x cap still shipped a target 42 times.
        m = _CODEBLOCK.findall(body)
        code = max(m, key=len) if m else body
        key = hashlib.sha1(re.sub(r"\s+", " ", code).strip()
                           .encode("utf-8", "ignore")).hexdigest()
        tally[key] += 1
        if tally[key] > MAX_TARGET_REPEATS:
            dropped_rep[str(r.get("_source"))] += 1
            continue
        capped.append(r)
    if dropped_rep:
        print(f"target-repeat cap (<= {MAX_TARGET_REPEATS}x): dropped "
              f"{sum(dropped_rep.values()):,}")
        for k, v in dropped_rep.most_common(6):
            print(f"    {k:<30} {v:,}")
    final = capped

    final, lstats = enforce_length(final, args.model_id, args.revision,
                                   args.token_limit)
    if lstats:
        print(f"length gate (<= {args.token_limit:,} tok): {len(final):,} kept")
        for k, v in lstats.most_common(8):
            print(f"    {k:<38} {v:,}")
    rng.shuffle(final)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in final:
            fh.write(json.dumps(r) + "\n")

    by_source = collections.Counter(r.get("_source") for r in final)
    by_shape = collections.Counter(r.get("_shape") for r in kernel)
    by_dial = collections.Counter(r.get("_dialect") for r in kernel)
    n = len(final)
    print(f"\n=== v5 ({args.policy}): {n:,} rows -> {out} ===")
    print(f"  kernel {len(kernel):,} ({100*len(kernel)/n:.1f}%)   "
          f"replay {len(replay):,} ({100*len(replay)/n:.1f}%)")
    print("\n  by source:")
    for k, v in by_source.most_common():
        print(f"    {str(k):<28} {v:>8,}  {100*v/n:5.1f}%")
    print("\n  kernel body by arena shape:")
    for k, v in by_shape.most_common():
        print(f"    {str(k):<28} {v:>8,}  {100*v/len(kernel):5.1f}%")
    print("\n  kernel body by dialect (benchmark: Triton 50.8 / FlyDSL 24.7 / HIP 24.2):")
    for k, v in by_dial.most_common():
        print(f"    {str(k):<28} {v:>8,}  {100*v/len(kernel):5.1f}%")

    receipt = {
        "policy": args.policy, "rows": n,
        "kernel": len(kernel), "replay": len(replay),
        "by_source": dict(by_source), "by_shape": dict(by_shape),
        "by_dialect": dict(by_dial), "seed": args.seed,
        "sanitize": dict(sstats), "length_gate": dict(lstats),
        "token_limit": args.token_limit, "model_id": args.model_id,
    }
    out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
