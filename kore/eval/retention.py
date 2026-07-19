"""KORE retention eval suite - general-capability harnesses (KORE.pdf Sec 5).

KORE's headline claim is *not* "best kernel numbers"; it is "best kernel numbers
WHILE matching-or-beating the base model on every general benchmark". Kernel
specialization must never silently regress general chat / code / reasoning. This
module provides the *general* half of that contract: a pluggable, dependency-free
harness that runs a battery of standard general benchmarks against any model and
reports the standard metric per bench plus an aggregate.

Pairing:
  - kernel numbers come from :mod:`kore.eval.bakeoff` / :mod:`kore.eval.fastp`
    (KernelBench L1/L2 correctness + speedup + fast_p, TritonBench-revised,
    ROCm-bench).
  - general numbers come from here (MMLU, HumanEval, LiveCodeBench-style, IFEval,
    BFCL, MT-Bench).
  - :mod:`kore.eval.gates` combines the two into a PASS/FAIL stage gate.

Design constraints:
  - **CPU-only + stub-friendly.** Every scorer takes a ``model_generate`` callable
    ``model_generate(prompt: str, **kw) -> str`` (see :class:`Scorer`). A trivial
    deterministic stub exercises the whole pipeline with no GPU; a real HF / vLLM
    model plugs in unchanged (see ``__doc__`` of :func:`run_retention_suite`).
  - **No heavy imports at module top.** Only stdlib is imported here. Executable
    code benches (HumanEval / LiveCodeBench) run generated code in a *separate*
    sandboxed subprocess with a wall-clock timeout (see :func:`_run_python_program`).
  - **Bundled samples are SMOKE-sized.** Each bench ships ~10-20 canonical items
    under ``kore/eval/data/<bench>.jsonl``, clearly marked. Pass ``full=True`` /
    ``n=`` (or set ``KORE_EVAL_FULL=1``) to pull the REAL HuggingFace splits
    (see :data:`FULL_HF_SOURCES`) mapped onto the same per-row schema; any
    failure (no ``datasets``, offline, schema drift) falls back to smoke and the
    source used is reported per bench.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# The single interface a model must satisfy to be evaluated: turn a prompt into
# a completion string. Keyword args (temperature, max_tokens, stop, ...) are
# forwarded to the backend. A chat model is adapted by rendering messages to a
# prompt (see ``chat_model_generate`` below).
ModelGenerate = Callable[..., str]

# An MT-Bench-style judge: score a single answer on a 1-10 scale. ``reference``
# is the optional gold answer for reference-guided judging.
JudgeFn = Callable[..., float]


@runtime_checkable
class Scorer(Protocol):
    """A benchmark scorer.

    ``name`` identifies the bench; ``score(model_generate)`` runs the bench and
    returns a metric dict that always contains a ``"score"`` key in ``[0, 1]``
    (the bench's primary, higher-is-better number) plus bench-specific detail.
    """

    name: str

    def score(self, model_generate: ModelGenerate) -> dict:  # pragma: no cover - protocol
        ...


_DATA_DIR = Path(__file__).resolve().parent / "data"


def load_bench(name: str, data_dir: Optional[Path] = None) -> list[dict]:
    """Load a bundled JSONL bench sample.

    Blank lines and ``#`` comment lines are ignored so the smoke files can carry
    a human-readable header. Each remaining line is one JSON object.
    """
    d = Path(data_dir) if data_dir is not None else _DATA_DIR
    path = d / f"{name}.jsonl"
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(json.loads(s))
    return items


# ---------------------------------------------------------------------------
# FULL benchmark loading (HuggingFace ``datasets``, lazily / offline-safe)
# ---------------------------------------------------------------------------
#
# The bundled JSONL files above are SMOKE-sized. For a *real* retention run we
# want the full public splits. These are fetched via HuggingFace ``datasets``
# - but that import (and the network round-trip) is deferred to call time and
# wrapped so that ANY failure (missing dep, offline box, schema drift) falls
# back to the bundled smoke set. That keeps this module import-clean and keeps
# CI / this CPU box fully functional with no network.
#
# Each loader maps the upstream row schema onto the SAME per-row schema the
# smoke JSONL uses, so every scorer and every metric definition is unchanged
# regardless of which source produced the items. Selection is controlled by
# ``full: bool`` / ``n: int|None`` on each scorer (or the ``KORE_EVAL_FULL``
# env var); ``n`` caps how many items are pulled (``None`` == the whole split).

# Upstream HuggingFace sources per bench (documented; used by the loaders).
FULL_HF_SOURCES: dict[str, str] = {
    "mmlu": "cais/mmlu",
    "humaneval": "openai_humaneval",
    "livecodebench": "livecodebench/code_generation_lite",
    "ifeval": "google/IFEval",
    "bfcl": "gorilla-llm/berkeley-function-calling-leaderboard",
    "mtbench": "lmsys/mt_bench",
}


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _hf_load_split(
    path: str,
    config: Optional[str] = None,
    split: str = "test",
    n: Optional[int] = None,
) -> list[dict]:
    """Return up to ``n`` rows of a HF dataset split as plain dicts.

    ``import datasets`` is intentionally done here (not at module top) so the
    heavy dependency is only paid when a full run is explicitly requested.
    Streaming is used when ``n`` is set so we do not materialize a huge split.
    """
    import datasets  # heavy + optional; guarded on purpose

    stream = n is not None
    # some benches (e.g. livecodebench) ship a custom loader script.
    kw = dict(split=split, streaming=stream, trust_remote_code=True)
    if config is not None:
        ds = datasets.load_dataset(path, config, **kw)
    else:
        ds = datasets.load_dataset(path, **kw)
    rows: list[dict] = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        rows.append(dict(row))
    return rows


def _full_mmlu(n: Optional[int]) -> list[dict]:
    rows = _hf_load_split("cais/mmlu", "all", "test", n)
    out: list[dict] = []
    for i, r in enumerate(rows):
        out.append(
            {
                "id": r.get("id", f"mmlu-full-{i}"),
                "subject": r.get("subject"),
                "question": r["question"],
                "choices": list(r["choices"]),
                "answer": r["answer"],  # int index or letter; _norm_letter handles both
            }
        )
    return out


def _full_humaneval(n: Optional[int]) -> list[dict]:
    rows = _hf_load_split("openai_humaneval", None, "test", n)
    out: list[dict] = []
    for i, r in enumerate(rows):
        entry = r["entry_point"]
        test = r["test"]
        # openai_humaneval ships ``def check(candidate): ...`` and expects the
        # entry point to be passed in; wire that call so the program actually runs.
        if "check(" not in test.split("def check", 1)[-1]:
            test = test.rstrip() + f"\n\ncheck({entry})\n"
        out.append(
            {
                "task_id": r.get("task_id", f"HE-full-{i}"),
                "entry_point": entry,
                "prompt": r["prompt"],
                "canonical_solution": r.get("canonical_solution", ""),
                "test": test,
            }
        )
    return out


def _map_lcb_row(r: dict, i: int) -> Optional[dict]:
    # Passthrough if a mirror already exposes our smoke schema.
    if r.get("test") and r.get("prompt"):
        return {
            "task_id": r.get("task_id", f"LCB-full-{i}"),
            "entry_point": r.get("entry_point", ""),
            "prompt": r["prompt"],
            "reference_solution": r.get("reference_solution", ""),
            "test": r["test"],
            "time_limit_s": float(r.get("time_limit_s", 6.0)),
        }
    # Native livecodebench/code_generation_lite functional mapping.
    try:
        meta = r.get("metadata")
        if isinstance(meta, str):
            meta = json.loads(meta) if meta else {}
        func_name = (meta or {}).get("func_name")
        pub = r.get("public_test_cases")
        if isinstance(pub, str):
            pub = json.loads(pub)
        if not (func_name and pub):
            return None
        test_lines = []
        for tc in pub:
            if tc.get("testtype") != "functional":
                return None  # stdin/stdout problems need a different harness
            args = [ln for ln in str(tc["input"]).split("\n") if ln.strip()]
            call = ", ".join(args)
            test_lines.append(f"assert {func_name}({call}) == {str(tc['output']).strip()}")
        prompt = (r.get("question_content", "") + "\n\n" + (r.get("starter_code") or "")).strip()
        return {
            "task_id": r.get("question_id", f"LCB-full-{i}"),
            "entry_point": func_name,
            "prompt": prompt,
            "test": "\n".join(test_lines) + "\n",
            "time_limit_s": 6.0,
        }
    except Exception:
        return None


def _full_livecodebench(n: Optional[int]) -> list[dict]:
    rows = _hf_load_split("livecodebench/code_generation_lite", None, "test", n)
    out: list[dict] = []
    for i, r in enumerate(rows):
        it = _map_lcb_row(r, i)
        if it:
            out.append(it)
    return out


# Map the subset of official IFEval instruction ids we can check programmatically
# (see IFEVAL_CHECKS) onto our checker specs. Unsupported ids are skipped.
def _map_ifeval_instructions(id_list, kwargs_list) -> list[dict]:
    specs: list[dict] = []
    kwargs_list = kwargs_list or [{}] * len(id_list)
    for iid, kw in zip(id_list, kwargs_list):
        kw = kw or {}
        if iid == "change_case:english_lowercase":
            specs.append({"type": "all_lowercase"})
        elif iid == "change_case:english_capital":
            specs.append({"type": "all_uppercase"})
        elif iid == "punctuation:no_comma":
            specs.append({"type": "no_commas"})
        elif iid == "keywords:existence":
            for w in kw.get("keywords", []) or []:
                specs.append({"type": "keyword_include", "keyword": w})
        elif iid == "keywords:forbidden_words":
            for w in kw.get("forbidden_words", []) or []:
                specs.append({"type": "keyword_forbidden", "keyword": w})
        elif iid == "startend:end_checker":
            ep = kw.get("end_phrase")
            if ep:
                specs.append({"type": "ends_with", "suffix": ep})
        elif iid == "detectable_format:number_bullet_lists":
            nb = kw.get("num_bullets")
            if nb is not None:
                specs.append({"type": "num_bullets", "n": nb})
        # else: instruction type we cannot verify offline -> skip it
    return specs


def _full_ifeval(n: Optional[int]) -> list[dict]:
    rows = _hf_load_split("google/IFEval", None, "train", n)
    out: list[dict] = []
    for i, r in enumerate(rows):
        specs = _map_ifeval_instructions(r.get("instruction_id_list", []), r.get("kwargs", []))
        if not specs:
            continue  # nothing checkable in this row
        out.append({"id": r.get("key", f"if-full-{i}"), "prompt": r["prompt"], "instructions": specs})
    return out


def _bfcl_extract_question(q) -> Optional[str]:
    """Pull the (last) user turn text out of BFCL's nested message structure."""

    def walk(x):
        if isinstance(x, dict):
            if x.get("role") in (None, "user") and "content" in x:
                return x["content"]
            return None
        if isinstance(x, list):
            found = None
            for e in x:
                r = walk(e)
                if r:
                    found = r
            return found
        if isinstance(x, str):
            return x
        return None

    return walk(q)


def _map_bfcl_row(r: dict, i: int) -> Optional[dict]:
    tools = r.get("tools") or r.get("function") or r.get("functions")
    if isinstance(tools, dict):
        tools = [tools]
    q = r.get("question")
    if not isinstance(q, str):
        q = _bfcl_extract_question(q)
    ans = r.get("answer") or r.get("ground_truth")
    if isinstance(ans, list) and ans:
        ans = ans[0]
    if not (q and isinstance(ans, dict) and ans.get("name")):
        return None
    return {
        "id": r.get("id", f"bfcl-full-{i}"),
        "question": q,
        "tools": tools or [],
        "answer": {"name": ans["name"], "arguments": ans.get("arguments", {})},
    }


def _full_bfcl(n: Optional[int]) -> list[dict]:
    rows = _hf_load_split("gorilla-llm/berkeley-function-calling-leaderboard", "simple", "train", n)
    out: list[dict] = []
    for i, r in enumerate(rows):
        it = _map_bfcl_row(r, i)
        if it:
            out.append(it)
    return out


def _full_mtbench(n: Optional[int]) -> list[dict]:
    try:
        rows = _hf_load_split("lmsys/mt_bench", None, "train", n)
    except Exception:
        rows = _hf_load_split("HuggingFaceH4/mt_bench_prompts", None, "train", n)
    out: list[dict] = []
    for i, r in enumerate(rows):
        q = r.get("question")
        if q is None:
            p = r.get("prompt") or r.get("turns")
            q = p[0] if isinstance(p, list) and p else p
        ref = r.get("reference")
        if isinstance(ref, list):
            ref = ref[0] if ref else None
        if not q:
            continue
        out.append({"id": r.get("question_id", f"mt-full-{i}"), "question": q, "reference": ref})
    return out


_FULL_LOADERS: dict[str, Callable[[Optional[int]], list[dict]]] = {
    "mmlu": _full_mmlu,
    "humaneval": _full_humaneval,
    "livecodebench": _full_livecodebench,
    "ifeval": _full_ifeval,
    "bfcl": _full_bfcl,
    "mtbench": _full_mtbench,
}


def load_full_bench(name: str, n: Optional[int] = None) -> list[dict]:
    """Load the full (HF) split for ``name`` mapped to the smoke row schema.

    Raises on any failure (missing ``datasets``, offline, schema drift) - callers
    (:func:`_resolve_bench_items`) treat that as the signal to fall back to smoke.
    """
    loader = _FULL_LOADERS.get(name)
    if loader is None:
        raise ValueError(f"unknown bench: {name!r} (known: {tuple(_FULL_LOADERS)})")
    return loader(n)


def _resolve_bench_items(scorer) -> list[dict]:
    """Shared item resolution for every scorer.

    Order of precedence: explicit ``items=`` > full HF split (if requested and it
    succeeds) > bundled smoke set. Records the chosen source on ``scorer.source``
    so :meth:`score` can report it (``"explicit"`` / ``"full-hf"`` / ``"smoke"``).
    """
    if scorer.items is not None:
        scorer.source = "explicit"
        return scorer.items
    if scorer.full or _truthy_env("KORE_EVAL_FULL"):
        try:
            # cap the pulled split: explicit scorer.n, else KORE_EVAL_N (keeps the
            # retention GATE fast - a few hundred items per bench is plenty of signal).
            n = scorer.n
            if n is None:
                try:
                    n = int(os.environ.get("KORE_EVAL_N", "") or 0) or None
                except ValueError:
                    n = None
            full = load_full_bench(scorer.name, n)
            if full:
                scorer.source = "full-hf"
                return full
        except Exception:
            pass  # offline / missing dep / schema drift -> smoke fallback
    scorer.source = "smoke"
    return load_bench(scorer.name, scorer.data_dir)


# ---------------------------------------------------------------------------
# Sandboxed code execution (HumanEval / LiveCodeBench)
# ---------------------------------------------------------------------------

def _limit_resources(mem_bytes: int, cpu_seconds: int):  # pragma: no cover - POSIX child
    """preexec hook: cap address space + CPU time in the child (best effort)."""
    try:
        import resource  # POSIX only

        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except Exception:
        pass


def _run_python_program(
    program: str,
    timeout: float = 10.0,
    mem_mb: int = 512,
) -> dict:
    """Run ``program`` as a standalone script in an isolated subprocess.

    Safety: fresh temp cwd, isolated interpreter (``-I``, ignores env / user
    site), hard wall-clock ``timeout`` (process killed on expiry), and a best-
    effort address-space / CPU cap on POSIX. Returns
    ``{"passed", "detail", "elapsed_s", "timed_out"}``. ``passed`` iff the script
    exits 0 (i.e. no assertion/exception).

    This is deliberately a *subprocess* rather than ``exec`` in-process: model-
    generated code is untrusted and can loop forever, crash, or leak state.
    """
    preexec = None
    if os.name == "posix":
        preexec = lambda: _limit_resources(mem_mb * 1024 * 1024, int(timeout) + 1)  # noqa: E731
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
                preexec_fn=preexec,
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "detail": "timeout",
                "elapsed_s": time.perf_counter() - start,
                "timed_out": True,
            }
        except Exception as e:  # pragma: no cover - defensive
            return {
                "passed": False,
                "detail": f"exec-error: {e}",
                "elapsed_s": time.perf_counter() - start,
                "timed_out": False,
            }
        elapsed = time.perf_counter() - start
        if proc.returncode == 0:
            return {"passed": True, "detail": "ok", "elapsed_s": elapsed, "timed_out": False}
        return {
            "passed": False,
            "detail": (proc.stderr or proc.stdout or "nonzero exit")[-2000:],
            "elapsed_s": elapsed,
            "timed_out": False,
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """If the model wrapped code in a markdown fence, return the fenced body."""
    m = _CODE_FENCE.search(text)
    if m:
        return m.group(1)
    return text


def _extract_first_json(text: str) -> Optional[dict]:
    """Extract the first balanced ``{...}`` JSON object from free-form text."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[start : i + 1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
        start = text.find("{", start + 1)
    return None


def _norm_letter(ans) -> str:
    """Normalize an MMLU answer (index 0-3 or letter) to an uppercase letter."""
    if isinstance(ans, int):
        return "ABCD"[ans] if 0 <= ans < 4 else "?"
    s = str(ans).strip().upper()
    return s[0] if s else "?"


# ---------------------------------------------------------------------------
# MMLU - multiple-choice accuracy
# ---------------------------------------------------------------------------

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def format_mmlu_prompt(item: dict) -> str:
    lines = [f"Question: {item['question']}", ""]
    for i, ch in enumerate(item["choices"]):
        lines.append(f"{_LETTERS[i]}. {ch}")
    lines.append("")
    lines.append("Respond with only the letter of the correct choice.")
    return "\n".join(lines)


def parse_mmlu_answer(text: str, n_choices: int) -> str:
    """Parse the chosen option letter. Prefers an explicit 'answer: X', else the
    last standalone valid letter, matching common MMLU parsing heuristics."""
    valid = set(_LETTERS[:n_choices])
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([A-H])\)?", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    found = [c for c in re.findall(r"[A-H]", text.upper()) if c in valid]
    return found[-1] if found else "?"


@dataclass
class MMLUScorer:
    name: str = "mmlu"
    items: Optional[list[dict]] = None
    data_dir: Optional[Path] = None
    full: bool = False
    n: Optional[int] = None
    source: Optional[str] = None

    def _load(self) -> list[dict]:
        return _resolve_bench_items(self)

    def score(self, model_generate: ModelGenerate) -> dict:
        items = self._load()
        correct = 0
        per_item = []
        for it in items:
            prompt = format_mmlu_prompt(it)
            # 32 (not 8) so a brief "The answer is B" lead-in still reaches the
            # letter; the real fix is thinking OFF in load_generate so the budget
            # isn't spent on a <think> trace (audit R2 soup-eval C1).
            out = model_generate(prompt, max_tokens=32, temperature=0.0)
            pred = parse_mmlu_answer(out, len(it["choices"]))
            gold = _norm_letter(it["answer"])
            ok = pred == gold
            correct += int(ok)
            per_item.append({"id": it.get("id"), "pred": pred, "gold": gold, "correct": ok})
        n = len(items)
        acc = correct / n if n else 0.0
        return {"score": acc, "accuracy": acc, "n": n, "correct": correct, "source": self.source, "per_item": per_item}


# ---------------------------------------------------------------------------
# HumanEval - pass@1 via sandboxed exec
# ---------------------------------------------------------------------------

def format_humaneval_prompt(item: dict) -> str:
    return (
        "Complete the following Python function. Return only the function body "
        "(the code after the signature).\n\n" + item["prompt"]
    )


@dataclass
class HumanEvalScorer:
    name: str = "humaneval"
    items: Optional[list[dict]] = None
    data_dir: Optional[Path] = None
    timeout: float = 10.0
    full: bool = False
    n: Optional[int] = None
    source: Optional[str] = None

    def _load(self) -> list[dict]:
        return _resolve_bench_items(self)

    @staticmethod
    def build_program(item: dict, completion: str) -> str:
        """Assemble prompt + completion + test into a runnable program, tolerant
        of chat-style output.

        A model SFT'd on multi-turn dialogues answers conversationally (prose
        before/after the code, with or without a markdown fence). The old parser
        only handled a bare body or a fenced full-def, so any prose outside a
        fence made the assembled program a SyntaxError -> pass@1 scored 0 even
        when the code was correct (the post-SFT ``humaneval 0.30->0.018`` gate
        artifact). This robustly extracts the entry-point function block.
        """
        comp = _strip_code_fences(completion)
        entry = item["entry_point"]
        m = re.search(rf"^[ \t]*def\s+{re.escape(entry)}\b", comp, re.MULTILINE)
        if m:
            # Drop leading prose before the def; keep only the contiguous function
            # block (def line + indented/blank lines), dropping trailing prose.
            lines = comp[m.start():].splitlines()
            block = [lines[0]]
            for ln in lines[1:]:
                if ln.strip() == "" or ln[:1] in (" ", "\t"):
                    block.append(ln)
                else:
                    break
            body = "\n".join(block)
        else:
            # Body-only completion: strip leading non-indented prose, then graft
            # the body onto the signature from the prompt.
            lines = comp.splitlines()
            while lines and lines[0].strip() and lines[0][:1] not in (" ", "\t"):
                lines.pop(0)
            body = item["prompt"] + "\n".join(lines)
        return body + "\n\n" + item["test"] + "\n"

    def score(self, model_generate: ModelGenerate) -> dict:
        items = self._load()
        passed = 0
        per_item = []
        for it in items:
            comp = model_generate(format_humaneval_prompt(it), max_tokens=512, temperature=0.0)
            program = self.build_program(it, comp)
            res = _run_python_program(program, timeout=self.timeout)
            passed += int(res["passed"])
            per_item.append({"task_id": it.get("task_id"), "passed": res["passed"], "detail": res["detail"]})
        n = len(items)
        p1 = passed / n if n else 0.0
        return {"score": p1, "pass@1": p1, "n": n, "passed": passed, "source": self.source, "per_item": per_item}


# ---------------------------------------------------------------------------
# LiveCodeBench-style - timed code correctness
# ---------------------------------------------------------------------------

def format_livecodebench_prompt(item: dict) -> str:
    return (
        "Solve the following problem by writing a complete Python function. "
        "Return only the function definition.\n\n" + item["prompt"]
    )


@dataclass
class LiveCodeBenchScorer:
    name: str = "livecodebench"
    items: Optional[list[dict]] = None
    data_dir: Optional[Path] = None
    full: bool = False
    n: Optional[int] = None
    source: Optional[str] = None

    def _load(self) -> list[dict]:
        return _resolve_bench_items(self)

    def score(self, model_generate: ModelGenerate) -> dict:
        items = self._load()
        passed = 0
        times: list[float] = []
        per_item = []
        for it in items:
            comp = _strip_code_fences(
                model_generate(format_livecodebench_prompt(it), max_tokens=1024, temperature=0.0)
            )
            program = comp + "\n\n" + it["test"] + "\n"
            limit = float(it.get("time_limit_s", 5.0))
            res = _run_python_program(program, timeout=limit)
            ok = res["passed"]  # correct AND within the wall-clock limit
            passed += int(ok)
            times.append(res["elapsed_s"])
            per_item.append(
                {
                    "task_id": it.get("task_id"),
                    "passed": ok,
                    "timed_out": res["timed_out"],
                    "elapsed_s": round(res["elapsed_s"], 4),
                }
            )
        n = len(items)
        rate = passed / n if n else 0.0
        return {
            "score": rate,
            "pass_rate": rate,
            "n": n,
            "passed": passed,
            "mean_elapsed_s": (sum(times) / len(times)) if times else 0.0,
            "source": self.source,
            "per_item": per_item,
        }


# ---------------------------------------------------------------------------
# IFEval-style - checkable instruction following
# ---------------------------------------------------------------------------

def _check_instruction(spec: dict, response: str) -> bool:
    t = spec.get("type")
    r = response
    if t == "all_lowercase":
        letters = [c for c in r if c.isalpha()]
        return bool(letters) and all(c.islower() for c in letters)
    if t == "all_uppercase":
        letters = [c for c in r if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)
    if t == "keyword_include":
        return spec["keyword"].lower() in r.lower()
    if t == "keyword_forbidden":
        return spec["keyword"].lower() not in r.lower()
    if t == "ends_with":
        return r.rstrip().endswith(spec["suffix"])
    if t == "starts_with":
        return r.lstrip().startswith(spec["prefix"])
    if t == "no_commas":
        return "," not in r
    if t == "word_count_min":
        return len(r.split()) >= int(spec["n"])
    if t == "word_count_max":
        return len(r.split()) <= int(spec["n"])
    if t == "num_bullets":
        bullets = [ln for ln in r.splitlines() if ln.strip().startswith(("* ", "- ", "• "))]
        return len(bullets) == int(spec["n"])
    if t == "num_sentences_max":
        sentences = [s for s in re.split(r"[.!?]+", r) if s.strip()]
        return len(sentences) <= int(spec["n"])
    if t == "json_format":
        try:
            json.loads(r.strip())
            return True
        except Exception:
            return False
    raise ValueError(f"unknown IFEval instruction type: {t!r}")


# Public registry of supported instruction checkers (for docs / extension).
IFEVAL_CHECKS = (
    "all_lowercase",
    "all_uppercase",
    "keyword_include",
    "keyword_forbidden",
    "ends_with",
    "starts_with",
    "no_commas",
    "word_count_min",
    "word_count_max",
    "num_bullets",
    "num_sentences_max",
    "json_format",
)


@dataclass
class IFEvalScorer:
    name: str = "ifeval"
    items: Optional[list[dict]] = None
    data_dir: Optional[Path] = None
    full: bool = False
    n: Optional[int] = None
    source: Optional[str] = None

    def _load(self) -> list[dict]:
        return _resolve_bench_items(self)

    def score(self, model_generate: ModelGenerate) -> dict:
        items = self._load()
        prompt_strict = 0  # all constraints satisfied for the prompt
        inst_total = 0
        inst_ok = 0
        per_item = []
        for it in items:
            out = model_generate(it["prompt"], max_tokens=256, temperature=0.0)
            results = [_check_instruction(spec, out) for spec in it["instructions"]]
            inst_total += len(results)
            inst_ok += sum(1 for x in results if x)
            all_ok = all(results)
            prompt_strict += int(all_ok)
            per_item.append({"id": it.get("id"), "all_ok": all_ok, "per_instruction": results})
        n = len(items)
        prompt_acc = prompt_strict / n if n else 0.0
        inst_acc = inst_ok / inst_total if inst_total else 0.0
        return {
            "score": prompt_acc,
            "prompt_strict_accuracy": prompt_acc,
            "instruction_accuracy": inst_acc,
            "n": n,
            "source": self.source,
            "per_item": per_item,
        }


# ---------------------------------------------------------------------------
# BFCL-style - function/tool-calling accuracy
# ---------------------------------------------------------------------------

def format_bfcl_prompt(item: dict) -> str:
    tools = json.dumps(item.get("tools", []))
    return (
        "You are a function-calling assistant. Given the user request and the "
        "available tools, respond with a single JSON object of the form "
        '{"name": <tool name>, "arguments": {<arg>: <value>, ...}} and nothing else.\n'
        f"Tools: {tools}\n"
        f"User: {item['question']}"
    )


def _args_match(expected: dict, got: dict) -> bool:
    """Every expected arg must be present with an equal value (string-normalized)."""
    for k, v in expected.items():
        if k not in got:
            return False
        if isinstance(v, (int, float)) and isinstance(got[k], (int, float)):
            if float(v) != float(got[k]):
                return False
        elif str(got[k]).strip() != str(v).strip():
            return False
    return True


@dataclass
class BFCLScorer:
    name: str = "bfcl"
    items: Optional[list[dict]] = None
    data_dir: Optional[Path] = None
    full: bool = False
    n: Optional[int] = None
    source: Optional[str] = None

    def _load(self) -> list[dict]:
        return _resolve_bench_items(self)

    def score(self, model_generate: ModelGenerate) -> dict:
        items = self._load()
        name_ok = 0
        full_ok = 0
        per_item = []
        for it in items:
            out = model_generate(format_bfcl_prompt(it), max_tokens=256, temperature=0.0)
            call = _extract_first_json(out)
            exp = it["answer"]
            got_name = (call or {}).get("name")
            got_args = (call or {}).get("arguments", {}) if call else {}
            nm = got_name == exp["name"]
            am = nm and isinstance(got_args, dict) and _args_match(exp.get("arguments", {}), got_args)
            name_ok += int(nm)
            full_ok += int(am)
            per_item.append({"id": it.get("id"), "name_ok": nm, "args_ok": am, "parsed": call is not None})
        n = len(items)
        acc = full_ok / n if n else 0.0
        return {
            "score": acc,
            "accuracy": acc,
            "name_accuracy": (name_ok / n) if n else 0.0,
            "n": n,
            "source": self.source,
            "per_item": per_item,
        }


# ---------------------------------------------------------------------------
# MT-Bench-style - LLM-judge (injectable)
# ---------------------------------------------------------------------------

def default_stub_judge(question: str, answer: str, reference: Optional[str] = None) -> float:
    """Deterministic stand-in judge for tests (1-10 scale).

    Real runs inject a strong-model judge via ``judge=`` (see
    :func:`run_retention_suite`). This stub is a *proxy*: empty answers score 1;
    with a reference it scores by unigram overlap; otherwise by (bounded) length.
    Deterministic so CI is stable.
    """
    ans = (answer or "").strip()
    if not ans:
        return 1.0
    if reference:
        ref_words = set(re.findall(r"\w+", reference.lower()))
        ans_words = set(re.findall(r"\w+", ans.lower()))
        if not ref_words:
            return 5.0
        overlap = len(ref_words & ans_words) / len(ref_words)
        return round(1.0 + 9.0 * overlap, 3)
    n_words = len(ans.split())
    return round(min(10.0, 3.0 + n_words / 10.0), 3)


@dataclass
class MTBenchScorer:
    name: str = "mtbench"
    items: Optional[list[dict]] = None
    data_dir: Optional[Path] = None
    judge: JudgeFn = default_stub_judge
    full: bool = False
    n: Optional[int] = None
    source: Optional[str] = None

    def _load(self) -> list[dict]:
        return _resolve_bench_items(self)

    def score(self, model_generate: ModelGenerate) -> dict:
        items = self._load()
        scores: list[float] = []
        per_item = []
        for it in items:
            out = model_generate(it["question"], max_tokens=512, temperature=0.0)
            raw = float(self.judge(it["question"], out, it.get("reference")))
            raw = max(1.0, min(10.0, raw))
            scores.append(raw)
            per_item.append({"id": it.get("id"), "judge_score": raw})
        n = len(items)
        mean_raw = (sum(scores) / n) if n else 0.0
        return {
            "score": mean_raw / 10.0,  # normalized to [0, 1] for the aggregate
            "mean_judge_score": mean_raw,  # raw 1-10 MT-Bench scale
            "n": n,
            "source": self.source,
            "per_item": per_item,
        }


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

# Canonical general-capability battery (KORE.pdf Sec 5 retention set).
DEFAULT_BENCHES: tuple[str, ...] = (
    "mmlu",
    "humaneval",
    "livecodebench",
    "ifeval",
    "bfcl",
    "mtbench",
)


def build_scorer(
    name: str,
    *,
    judge: Optional[JudgeFn] = None,
    data_dir: Optional[Path] = None,
    full: bool = False,
    n: Optional[int] = None,
    **kw,
) -> Scorer:
    """Instantiate a scorer by bench name. ``judge`` applies only to MT-Bench.

    ``full``/``n`` select the FULL HuggingFace split (with offline fallback to
    smoke) - see :func:`load_full_bench`.
    """
    common = dict(data_dir=data_dir, full=full, n=n)
    if name == "mmlu":
        return MMLUScorer(**common, **kw)
    if name == "humaneval":
        return HumanEvalScorer(**common, **kw)
    if name == "livecodebench":
        return LiveCodeBenchScorer(**common, **kw)
    if name == "ifeval":
        return IFEvalScorer(**common, **kw)
    if name == "bfcl":
        return BFCLScorer(**common, **kw)
    if name == "mtbench":
        return MTBenchScorer(judge=judge or default_stub_judge, **common, **kw)
    raise ValueError(f"unknown bench: {name!r} (known: {DEFAULT_BENCHES})")


def chat_model_generate(chat_fn: Callable[[list[dict]], str], system: Optional[str] = None) -> ModelGenerate:
    """Adapt a chat model (``chat_fn(messages) -> str``) to the ``model_generate``
    prompt interface, so a chat-only backend plugs into every scorer unchanged."""

    def generate(prompt: str, **kw) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return chat_fn(messages)

    return generate


def run_retention_suite(
    model_generate: ModelGenerate,
    benches: Sequence[str] = DEFAULT_BENCHES,
    judge: Optional[JudgeFn] = None,
    *,
    full: bool = False,
    n: Optional[int] = None,
    data_dir: Optional[Path] = None,
    scorers: Optional[Sequence[Scorer]] = None,
    cache_dir: Optional[Path] = None,
    cache_tag: str = "",
) -> dict:
    """Run the general-retention battery and return per-bench + aggregate scores.

    ``model_generate(prompt, **kw) -> str`` is the only model interface required.

    Wiring a real model (non-smoke run)::

        # vLLM-ROCm (see kore.policy.serve.VLLMPolicy)
        policy = VLLMPolicy(model="Qwen/Qwen3-32B", tensor_parallel_size=8)
        gen = lambda p, **kw: policy.generate([p], **kw)[0]

        # or plain HuggingFace transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
        def gen(prompt, max_tokens=512, temperature=0.0, **kw):
            ids = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**ids, max_new_tokens=max_tokens,
                                  do_sample=temperature > 0, temperature=max(temperature, 1e-5))
            return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

        base = run_retention_suite(gen, judge=my_gpt4_judge)

    For the full (non-smoke) benchmark, pass ``full=True`` (or set the
    ``KORE_EVAL_FULL=1`` env var) to pull each bench's real HuggingFace split
    (see :data:`FULL_HF_SOURCES`); ``n`` caps items per bench. If ``datasets`` is
    missing, the box is offline, or any upstream schema drifts, each bench
    silently falls back to its bundled smoke set - the chosen source is reported
    per bench under ``"sources"``. You can also point a scorer at an arbitrary
    split by constructing it with ``items=<rows>`` and passing ``scorers=``. A
    strong judge is passed via ``judge=`` for MT-Bench.

    Returns::

        {
          "benches": [...],
          "scores": {bench: primary_score in [0,1]},   # <- feed to StageGate general_keys
          "aggregate": mean(primary scores),
          "per_bench": {bench: full metric dict},
          "sources": {bench: "full-hf" | "smoke" | "explicit"},
          "full": bool,   # whether a full run was requested (flag or env)
        }
    """
    if scorers is None:
        scorers = [build_scorer(b, judge=judge, data_dir=data_dir, full=full, n=n) for b in benches]

    # Optional per-benchmark score cache: on a contended/flaky node the ~1.75h gate
    # can be SIGKILLed mid-suite; caching each bench's result to disk lets a relaunch
    # SKIP the benches already scored instead of redoing them from scratch. base and
    # candidate are stable across restarts, so the caller encodes the model
    # fingerprint in ``cache_tag`` to keep the cache valid + role-separated.
    _cache = Path(cache_dir) if cache_dir else None
    if _cache is not None:
        _cache.mkdir(parents=True, exist_ok=True)

    per_bench: dict[str, dict] = {}
    scores: dict[str, float] = {}
    sources: dict[str, Optional[str]] = {}
    for sc in scorers:
        _cf = (_cache / f"{cache_tag}__{sc.name}.json") if _cache is not None else None
        result = None
        if _cf is not None and _cf.exists():
            try:
                result = json.loads(_cf.read_text())
            except Exception:  # noqa: BLE001 - corrupt/partial cache -> recompute
                result = None
        if result is None:
            result = sc.score(model_generate)
            if _cf is not None:
                try:
                    _cf.write_text(json.dumps(result))
                except Exception:  # noqa: BLE001 - caching is best-effort, never fatal
                    pass
        per_bench[sc.name] = result
        scores[sc.name] = float(result.get("score", 0.0))
        sources[sc.name] = result.get("source", getattr(sc, "source", None))

    aggregate = (sum(scores.values()) / len(scores)) if scores else 0.0
    return {
        "benches": [sc.name for sc in scorers],
        "scores": scores,
        "aggregate": aggregate,
        "per_bench": per_bench,
        "sources": sources,
        "full": bool(full or _truthy_env("KORE_EVAL_FULL")),
    }


__all__ = [
    "ModelGenerate",
    "JudgeFn",
    "Scorer",
    "load_bench",
    "MMLUScorer",
    "HumanEvalScorer",
    "LiveCodeBenchScorer",
    "IFEvalScorer",
    "BFCLScorer",
    "MTBenchScorer",
    "default_stub_judge",
    "IFEVAL_CHECKS",
    "DEFAULT_BENCHES",
    "FULL_HF_SOURCES",
    "load_full_bench",
    "build_scorer",
    "chat_model_generate",
    "run_retention_suite",
    "format_mmlu_prompt",
    "parse_mmlu_answer",
    "format_humaneval_prompt",
    "format_livecodebench_prompt",
    "format_bfcl_prompt",
]
