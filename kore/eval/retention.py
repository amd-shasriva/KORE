"""KORE retention eval suite — general-capability harnesses (KORE.pdf Sec 5).

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
    under ``kore/eval/data/<bench>.jsonl``, clearly marked. Swap in the full set
    by pointing the scorer at the real split (each JSONL row uses the same schema).
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
# MMLU — multiple-choice accuracy
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

    def _load(self) -> list[dict]:
        return self.items if self.items is not None else load_bench("mmlu", self.data_dir)

    def score(self, model_generate: ModelGenerate) -> dict:
        items = self._load()
        correct = 0
        per_item = []
        for it in items:
            prompt = format_mmlu_prompt(it)
            out = model_generate(prompt, max_tokens=8, temperature=0.0)
            pred = parse_mmlu_answer(out, len(it["choices"]))
            gold = _norm_letter(it["answer"])
            ok = pred == gold
            correct += int(ok)
            per_item.append({"id": it.get("id"), "pred": pred, "gold": gold, "correct": ok})
        n = len(items)
        acc = correct / n if n else 0.0
        return {"score": acc, "accuracy": acc, "n": n, "correct": correct, "per_item": per_item}


# ---------------------------------------------------------------------------
# HumanEval — pass@1 via sandboxed exec
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

    def _load(self) -> list[dict]:
        return self.items if self.items is not None else load_bench("humaneval", self.data_dir)

    @staticmethod
    def build_program(item: dict, completion: str) -> str:
        """Assemble prompt + completion + test into a runnable program.

        Accepts either a bare function body (indented) or a full ``def`` — if the
        completion already re-declares the entry point we use it standalone.
        """
        comp = _strip_code_fences(completion)
        entry = item["entry_point"]
        if re.search(rf"^\s*def\s+{re.escape(entry)}\b", comp, re.MULTILINE):
            body = comp
        else:
            body = item["prompt"] + comp
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
        return {"score": p1, "pass@1": p1, "n": n, "passed": passed, "per_item": per_item}


# ---------------------------------------------------------------------------
# LiveCodeBench-style — timed code correctness
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

    def _load(self) -> list[dict]:
        return self.items if self.items is not None else load_bench("livecodebench", self.data_dir)

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
            "per_item": per_item,
        }


# ---------------------------------------------------------------------------
# IFEval-style — checkable instruction following
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

    def _load(self) -> list[dict]:
        return self.items if self.items is not None else load_bench("ifeval", self.data_dir)

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
            "per_item": per_item,
        }


# ---------------------------------------------------------------------------
# BFCL-style — function/tool-calling accuracy
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

    def _load(self) -> list[dict]:
        return self.items if self.items is not None else load_bench("bfcl", self.data_dir)

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
            "per_item": per_item,
        }


# ---------------------------------------------------------------------------
# MT-Bench-style — LLM-judge (injectable)
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

    def _load(self) -> list[dict]:
        return self.items if self.items is not None else load_bench("mtbench", self.data_dir)

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


def build_scorer(name: str, *, judge: Optional[JudgeFn] = None, data_dir: Optional[Path] = None, **kw) -> Scorer:
    """Instantiate a scorer by bench name. ``judge`` applies only to MT-Bench."""
    if name == "mmlu":
        return MMLUScorer(data_dir=data_dir, **kw)
    if name == "humaneval":
        return HumanEvalScorer(data_dir=data_dir, **kw)
    if name == "livecodebench":
        return LiveCodeBenchScorer(data_dir=data_dir, **kw)
    if name == "ifeval":
        return IFEvalScorer(data_dir=data_dir, **kw)
    if name == "bfcl":
        return BFCLScorer(data_dir=data_dir, **kw)
    if name == "mtbench":
        return MTBenchScorer(judge=judge or default_stub_judge, data_dir=data_dir, **kw)
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
    data_dir: Optional[Path] = None,
    scorers: Optional[Sequence[Scorer]] = None,
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

    For the full (non-smoke) benchmark, point each scorer at the real split by
    constructing scorers explicitly with ``items=<full split>`` (same row schema
    as the bundled JSONL) and passing them via ``scorers=``. A strong judge is
    passed via ``judge=`` for MT-Bench.

    Returns::

        {
          "benches": [...],
          "scores": {bench: primary_score in [0,1]},   # <- feed to StageGate general_keys
          "aggregate": mean(primary scores),
          "per_bench": {bench: full metric dict},
        }
    """
    if scorers is None:
        scorers = [build_scorer(b, judge=judge, data_dir=data_dir) for b in benches]

    per_bench: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for sc in scorers:
        result = sc.score(model_generate)
        per_bench[sc.name] = result
        scores[sc.name] = float(result.get("score", 0.0))

    aggregate = (sum(scores.values()) / len(scores)) if scores else 0.0
    return {
        "benches": [sc.name for sc in scorers],
        "scores": scores,
        "aggregate": aggregate,
        "per_bench": per_bench,
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
    "build_scorer",
    "chat_model_generate",
    "run_retention_suite",
    "format_mmlu_prompt",
    "parse_mmlu_answer",
    "format_humaneval_prompt",
    "format_livecodebench_prompt",
    "format_bfcl_prompt",
]
