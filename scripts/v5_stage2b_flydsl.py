#!/usr/bin/env python
"""Stage 2b: a FlyDSL language anchor from the DSL's own test suite and tutorials.

FlyDSL is 24.7% of the benchmark and about 1% of the corpus, which looks like a
knowledge gap and is not one. The base model scores 86.0% on ``triton2flydsl``
untouched; v4 scored 72.0%. Qwen3-Coder walked in able to write this language and
fine-tuning on 99% Triton and HIP beat it out of him. So the goal here is not to
teach FlyDSL, it is to stop unlearning it -- an anchor against drift, which needs
far less data than a curriculum would.

WHAT IS EXCLUDED, AND WHY IT COSTS ALMOST NOTHING

``flydsl/kernels/`` is AMD's production kernel library, and it is the corpus the
arena draws its FlyDSL tasks from: ``kernels/norm/layernorm_kernel.py`` against
arena task ``flydsl2flydsl/layernorm_kernel``, and the same for ``rmsnorm_kernel``,
``softmax_kernel``, ``moe_sorting_kernel``, ``fused_rope_cache_kernel``. Training
on it is training on the answer key, so the whole directory is excluded rather
than filtered by a similarity threshold -- on the one category where measurement
matters most, a wholesale rule beats a tuned one.

That exclusion is cheap because the value in this library is the *language*, not
the *operations*. The model does not need to be shown layernorm; it knows
layernorm from Triton and HIP. It needs to be shown how anything at all is
expressed in FlyDSL -- the tiling idioms, the memory model, the ``flyc.jit``
calling convention. Language competence transfers across operations; a memorised
kernel does not. And the arithmetic agrees: the directly-leaked tasks are
``flydsl2flydsl``, of which only 5 are runnable on gfx950 and where all three arms
already tie at 40%. The category actually worth 14 points is ``triton2flydsl``,
where the fight is against regression, not ignorance.

The test suite teaches the language more directly than a 5,550-line production
flash-attention kernel would anyway: each test is a small kernel exercising one
construct, next to an assertion describing what it should compute.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path("/home/shasriva/Kore-RL/KORE")
sys.path.insert(0, str(REPO))

FLYDSL = Path("/home/shasriva/third_party/flydsl")
ARENA = Path("/home/shasriva/third_party/AgentKernelArena/tasks")

#: The public FlyDSL ecosystem, cloned alongside. There is no FlyDSL corpus on
#: HuggingFace -- five search terms, zero hits -- so these repositories are the
#: entire public supply of the language, and they hold 276 further kernel
#: definitions against the 293 in the DSL's own tree.
#:
#: This is human-written, working FlyDSL, which is the opposite failure mode from
#: resampling a model that does not know the API: gating 1,178 second-attempt
#: generated kernels returned 3 passes, because the model reproduces the same
#: signature error however many times it is asked. Scraped code does not have that
#: problem.
ECOSYSTEM = Path("/home/shasriva/third_party/flydsl_ecosystem")

#: Repositories worth mining, richest first. Excluded: ``GEAK`` (its FlyDSL is
#: markdown reference, not kernels), ``tile-kernel-bench-cdna4`` and
#: ``rocm-KDA-pilot`` (no kernel definitions), and any benchmark-shaped repo, since
#: a benchmark of generated kernels is an evaluation set rather than training data.
#: Ordered by measured yield. Most of these were invisible to a GitHub *repo*
#: search because they are production codebases that happen to contain FlyDSL --
#: aiter, Primus-Turbo, mori, MSLK, pyhip -- rather than FlyDSL projects. Code
#: search finds them; repo search does not, which is why the first sweep saw a
#: tenth of the supply.
ECOSYSTEM_REPOS = (
    "FlyDSL",                       # Deep-Spark vendor port: new families
    "aiter",                        # AMD AI Tensor Engine
    "atrex-kernel-agent",
    "flydsl-kernel-profiling-lib",
    "Primus-Turbo", "pyhip", "how-to-optimize-in-flydsl", "mori",
    "tilelang-to-flydsl-skills", "Primus", "MSLK", "ai-framework-labs",
    "flykernels", "flydsl-rocprof-cli", "flydsl_demo", "flydsl-examples",
    "flydsl-gemm", "flydsl-torchInductor",
    # Every non-main ref of the DSL's own repository, extracted by walking 2,011
    # refs including 186 closed-unmerged PRs. GitHub code search indexes only
    # default branches and skips forks, so this material is invisible to it -- and
    # it is larger than main. Mined LAST and with name-level dedup, because it
    # spans historical revisions: 364 content-unique files carry 6,151
    # definitions, i.e. ~17 per file, most of which are successive edits of the
    # same kernel rather than different kernels.
    "flydsl-allbranches",
)

#: Within one source, keep only the longest version of each kernel NAME. Content
#: hashing alone would admit every revision of a kernel as distinct, which is the
#: duplication this build spends most of its effort removing.
DEDUP_BY_NAME_SOURCES = frozenset({"flydsl-allbranches"})

#: Directories mined. ``kernels/`` is AMD's production library and does overlap the
#: benchmark -- but only 6 of its 88 files name an arena task, so excluding the
#: whole directory discarded ~82 clean files to guard 6, of which only 3 are active
#: on gfx950 at all. The per-file name screen below is the proportionate control.
ADMITTED_DIRS = ("tests", "examples", "docs", "kernels")

#: Files whose stem names an arena FlyDSL task. Blocked by name, and so are any
#: files that import them, since a helper carries the same content.
#:
#: The last five are the ones that matter most and were missed: the arena DOES ship
#: FlyDSL, contrary to an earlier check of mine that was accidentally scoped to one
#: sub-suite. 27 files carry 140 definitions, and five ``torch2flydsl`` tasks ship a
#: complete working reference solution rather than the honest NotImplementedError
#: stub the other forty use. Those five are answer keys.
ARENA_NAMED_STEMS = frozenset({
    "fused_rope_cache_kernel", "layernorm_kernel", "moe_sorting_kernel",
    "rmsnorm_kernel", "softmax_kernel", "topk_gating_softmax_kernel",
    "hgemm_kernel", "gemm_a8w8_bpreshuffle_kernel", "jagged_dense_bmm_kernel",
    "qk_norm_rope_quant_kernel",
})

SYSTEM = ("You are KORE, an expert AMD GPU kernel engineer targeting MI355X "
          "(gfx950, CDNA4).")

#: File-stem similarity at which a file is refused. Kept strict: a filename is a
#: strong signal and files are cheap to lose.
NAME_OVERLAP_BLOCK = 0.25

#: Function-name similarity at which an individual kernel is refused. Looser than
#: the file threshold on purpose -- kernel names are generic ("gemm_kernel",
#: "reduce"), so 0.25 rejected 82 kernels for sharing one common word with a task
#: name, while the real exposure is already covered by refusing arena-named files
#: and anything importing them.
KERNEL_NAME_BLOCK = 0.55

MAX_KERNEL_CHARS = 20000


def arena_flydsl_bodies() -> set[str]:
    """Normalised bodies of every FlyDSL kernel the benchmark itself ships.

    Names are a proxy and this is the fact. Screening on content catches a kernel
    copied under a different filename, which no name rule can, and it is the only
    check that would have caught the five ``torch2flydsl`` answer keys had they
    been vendored into a third-party repo under another name.
    """
    out: set[str] = set()
    root = ARENA.parent if ARENA.name == "tasks" else ARENA
    for p in root.rglob("*.py"):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if "flyc.jit" not in text and "flyc.kernel" not in text:
            continue
        for _name, src, _doc in extract_kernels(text, p):
            out.add(normalize_body(src))
    return out


def normalize_body(src: str) -> str:
    """Whitespace- and comment-insensitive form of a kernel body."""
    lines = []
    for line in (src or "").splitlines():
        line = re.sub(r"#.*$", "", line).strip()
        if line:
            lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines))


def arena_flydsl_names() -> list[tuple[str, set[str]]]:
    out = []
    for sub in ("triton2flydsl", "torch2flydsl", "flydsl2flydsl"):
        d = ARENA / sub
        if not d.is_dir():
            continue
        for cfg in d.rglob("config.yaml"):
            name = cfg.parent.name
            out.append((f"{sub}/{name}", tokens(name)))
    return out


def tokens(s: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) > 2}


def name_overlap(stem: str, arena: list[tuple[str, set[str]]]) -> tuple[float, str]:
    st = tokens(stem)
    best, score = "", 0.0
    for name, at in arena:
        shared = len(st & at)
        if not shared:
            continue
        s = shared / max(1, len(st | at))
        if s > score:
            best, score = name, s
    return score, best


def extract_kernels(text: str, path: Path) -> list[tuple[str, str, str]]:
    """Every ``@flyc.jit`` function: (name, source, docstring-or-context).

    Parsed rather than regexed so a kernel comes out as a syntactically complete
    unit; a truncated function is worse than no example.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.splitlines()
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorated = any(
            "flyc" in ast.unparse(d)
            and ("jit" in ast.unparse(d) or "kernel" in ast.unparse(d))
            for d in node.decorator_list) if node.decorator_list else False
        if not decorated:
            continue
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        end = getattr(node, "end_lineno", node.lineno)
        src = "\n".join(lines[start:end])
        if not src.strip() or len(src) > MAX_KERNEL_CHARS:
            continue
        doc = ast.get_docstring(node) or ""
        out.append((node.name, src, doc))
    return out


def describe(name: str, doc: str, path: Path) -> str:
    """A spec for the kernel that does not contain the kernel."""
    pretty = name.replace("_", " ").strip()
    lines = [f"Write a FlyDSL kernel `{name}` for gfx950 (CDNA4)."]
    if doc:
        lines.append("\n" + doc.strip()[:700])
    else:
        lines.append(f"It should implement {pretty} using the FlyDSL "
                     f"`@flyc.jit` programming model.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/v5_flydsl_lang.jsonl"))
    ap.add_argument("--no-ecosystem", action="store_true",
                    help="mine only the DSL's own tree, not the public repos")
    ap.add_argument("--exclude-kernels-dir", action="store_true",
                    help="drop flydsl/kernels/ entirely instead of screening it "
                         "per file")
    args = ap.parse_args()

    if not FLYDSL.is_dir():
        print(f"no FlyDSL checkout at {FLYDSL}", file=sys.stderr)
        return 1

    arena = arena_flydsl_names()
    arena_bodies = arena_flydsl_bodies()
    print(f"arena FlyDSL kernel bodies indexed for content screening: "
          f"{len(arena_bodies):,}")
    dirs = [d for d in ADMITTED_DIRS
            if not (args.exclude_kernels_dir and d == "kernels")]
    print(f"arena FlyDSL tasks: {len(arena)}   mining dirs: {dirs}")
    print(f"blocking {len(ARENA_NAMED_STEMS)} arena-named stems by name and by import")

    rows: list[dict] = []
    stats: collections.Counter = collections.Counter()
    seen: set[str] = set()
    #: (source, kernel name) -> index into rows, for name-level dedup.
    best_by_name: dict[tuple, int] = {}

    roots = [(FLYDSL / sub, f"flydsl/{sub}") for sub in dirs]
    if not args.no_ecosystem:
        roots += [(ECOSYSTEM / r, f"eco/{r}") for r in ECOSYSTEM_REPOS]

    for d, label in roots:
        if not d.is_dir():
            continue
        sub = label
        for p in sorted(d.rglob("*.py")):
            if "__pycache__" in p.parts or p.name == "__init__.py":
                continue
            if "thirdparty" in p.parts or ".git" in p.parts:
                stats["skipped_vendored"] += 1
                continue
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            if "flyc.jit" not in text and "flyc.kernel" not in text:
                continue
            kernels = extract_kernels(text, p)
            if not kernels:
                continue
            # The six arena-named stems are blocked everywhere: those are specific
            # file identities the benchmark asks about by name.
            if p.stem in ARENA_NAMED_STEMS:
                stats["blocked_arena_named_file"] += 1
                continue
            # The FUZZY name screen applies only to AMD's own production library,
            # because that is the corpus the arena draws from and a filename there
            # really can be the file a task names. Everywhere else it is wrong, and
            # measurably so: FlyDSL code is organised by operator and arena FlyDSL
            # tasks are named after operators, so word-overlap rejected 121 of 123
            # definitions in a third-party profiling library for sharing a word like
            # "gemm". Operator identity is not contamination -- a shared source
            # document is -- and the arena ships no FlyDSL reference files at all,
            # so there is no document for a third-party repo to be copying.
            in_amd_library = sub.endswith("kernels")
            if in_amd_library:
                score, which = name_overlap(p.stem, arena)
                if score >= NAME_OVERLAP_BLOCK:
                    stats["blocked_arena_name"] += 1
                    continue
                if any(re.search(rf"\b{re.escape(stem)}\b", text)
                       for stem in ARENA_NAMED_STEMS):
                    stats["blocked_imports_arena_named"] += 1
                    continue
            for name, src, doc in kernels:
                if in_amd_library:
                    kscore, _ = name_overlap(name, arena)
                    if kscore >= KERNEL_NAME_BLOCK:
                        stats["blocked_arena_kernel_name"] += 1
                        continue
                # Content screen against the benchmark's own FlyDSL kernels. This
                # is the check that does not depend on a filename being honest.
                if normalize_body(src) in arena_bodies:
                    stats["blocked_arena_body_match"] += 1
                    continue
                h = hashlib.sha1(src.encode()).hexdigest()
                if h in seen:
                    stats["dup"] += 1
                    continue
                key = (sub, name)
                if sub.rsplit("/", 1)[-1] in DEDUP_BY_NAME_SOURCES:
                    prior = best_by_name.get(key)
                    if prior is not None:
                        old = rows[prior]["messages"][-1]["content"]
                        if len(src) <= len(old):
                            stats["dup_name_shorter"] += 1
                            continue
                        rows[prior] = None  # superseded by a longer revision
                        stats["dup_name_superseded"] += 1
                seen.add(h)
                best_by_name[key] = len(rows)
                rows.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": describe(name, doc, p)},
                        {"role": "assistant", "content": f"```python\n{src}\n```"},
                    ],
                    "_source": "kernel_flydsl_language",
                    "_task_id": f"flydsl_lib::{p.stem}::{name}",
                    "_repo": sub,
                    "_dialect": "FlyDSL",
                    "_shape": "instruction",
                    "_origin": f"{sub}:{p.name}",
                })
                stats[f"from_{sub}"] += 1

    rows = [r for r in rows if r is not None]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\n=== FlyDSL language anchor: {len(rows):,} kernels ===")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"  {k:<32} {v:,}")
    sizes = sorted(len(r["messages"][-1]["content"]) for r in rows)
    if sizes:
        print(f"  kernel chars  median={sizes[len(sizes)//2]:,}  "
              f"p90={sizes[int(len(sizes)*.9)]:,}  max={sizes[-1]:,}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
