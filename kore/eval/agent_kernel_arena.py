"""Run AgentKernelArena tasks without its Docker runner, and score them its way.

AgentKernelArena is AMD's own kernel-agent benchmark: 412 tasks over hip2hip,
triton2triton, torch2hip and related types, with published numbers for frontier
agents on MI355X (gfx950) -- the hardware we train for. That makes it the only
benchmark where "we beat Opus" is a checkable claim rather than an assertion
about our own held-out split.

The published bar, from the AgentKernelArena paper:

    PyTorch-to-HIP    6.89x   (Claude Code, Opus 4.6)
    HIP-to-HIP        6.69x   (Claude Code, Opus 4.6)
    Triton-to-Triton  2.13x   (Cursor Agent, Opus 4.7 High)

Triton-to-Triton is the soft target and it is not close: Opus scores 2.13x there
against 6.69x on HIP-to-HIP, so it is markedly weaker at IMPROVING an existing
Triton kernel than at writing one from PyTorch. Improving an existing kernel
under execution feedback is exactly what our multi-turn data teaches, and 165 of
the 412 tasks are that type.

Why we re-implement the runner rather than call theirs: AKA drives tasks through
Docker, and this cluster has no container runtime at all -- docker, podman,
apptainer, singularity and enroot are all absent. But a task's contract is
declared in its own config.yaml as plain argv, so the Docker image is a
reproducibility convenience rather than a dependency. We honour the same
contract, in the same order, and use the same scoring formula, so the numbers
line up with the published ones.

Scoring is AKA's, unchanged (src/score.py):

    compile fails                  ->   0
    compiles, correctness fails    ->  20
    both pass                      -> 120 + speedup * 100

Deviating from it would produce a number that looks comparable and is not.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from kore.obs import get_logger

log = get_logger("eval.aka")

# Points from AgentKernelArena's default scoring policy.
COMPILE_POINTS = 20
CORRECT_POINTS = 100
SPEEDUP_POINTS = 100

# The task types we can score against a published Opus number.
PUBLISHED_OPUS_MEAN_SPEEDUP = {
    "torch2hip": 6.89,        # Claude Code, Opus 4.6
    "hip2hip": 6.69,          # Claude Code, Opus 4.6
    "triton2triton": 2.13,    # Cursor Agent, Opus 4.7 High
}


#: Harness and packaging files. Never the answer, and never the problem
#: statement -- writing a kernel into the test runner would be scored as the
#: model's work, and showing it as context just wastes the window.
_SCAFFOLDING = frozenset({
    "test_kernel_harness.py", "performance_utils_pytest.py", "task_runner.py",
    "conftest.py", "__init__.py", "setup.py", "utils.py", "compile.py",
    "correctness_check.py", "cal_kernel_perf.py", "kernel_loader_template.py",
})


class ArenaError(RuntimeError):
    """A task cannot be loaded or run under the arena contract."""


@dataclass
class ArenaTask:
    task_id: str
    task_type: str
    root: Path
    source_files: list[str]
    target_functions: list[str]
    compile_command: list[list[str]]
    correctness_command: list[list[str]]
    performance_command: list[list[str]]
    instructions: str = ""
    required_arch: Optional[str] = None
    # Where the answer belongs, from the task's own `target_file_path`. For a
    # translation task this is a DIFFERENT file, in a different language, from the
    # one you read: torch2hip reads pytorch_code_module/*.py and its
    # compile_command builds hip/*.hip. Writing back over the source leaves that
    # target empty, and the build then yields an extension with no init symbol --
    # "dynamic module does not define module export function" -- identically on
    # every task in the category.
    target_file: Optional[str] = None

    def answer_path(self) -> str:
        """Relative path the generated code must be written to."""
        if self.target_file:
            return self.target_file
        if self.source_files:
            # Same-language tasks (triton2triton, hip2hip) optimize in place.
            return self.source_files[0]
        # instruction2triton declares no source at all -- the task IS the
        # instruction. The file to write is the one its commands operate on, and
        # it differs per task (gemm.py, layernorm.py, ...), so a fixed fallback
        # writes the answer somewhere nothing reads and every task scores the
        # shipped stub instead of the model.
        inferred = self._file_named_in_commands()
        return inferred or "kernel.py"

    def context_files(self) -> list[str]:
        """Files that state the problem without being the answer.

        torch2flydsl ships model.py (the PyTorch to translate) beside kernel.py
        (the empty target), and declares only kernel.py as its source. Showing the
        model just its own blank target is why that category compiles almost
        everything and gets none of it right.
        """
        answer = self.answer_path()
        out = []
        for p in sorted(self.root.glob("*.py")):
            rel = p.name
            if rel == answer or rel in _SCAFFOLDING or rel in self.source_files:
                continue
            out.append(rel)
        return out

    def _file_named_in_commands(self) -> Optional[str]:
        """The .py file this task's own commands act on, if any.

        Read from the commands rather than guessed, so it stays right when the
        task is named layernorm.py instead of gemm.py.
        """
        text = " ".join(c[0] for c in
                        (self.correctness_command + self.compile_command
                         + self.performance_command) if c)
        for name in re.findall(r"[\w./-]+\.py", text):
            base = name.split("/")[-1]
            if base in _SCAFFOLDING:
                continue
            if (self.root / base).exists():
                return base
        return None

    def source_text(self) -> str:
        parts = []
        for rel in self.source_files:
            p = self.root / rel
            if p.exists():
                parts.append(f"# ---- {rel} ----\n{p.read_text()}")
        return "\n\n".join(parts)


@dataclass
class ArenaResult:
    task_id: str
    task_type: str
    compiled: bool = False
    correct: bool = False
    baseline_seconds: Optional[float] = None
    optimized_seconds: Optional[float] = None
    speedup: Optional[float] = None
    score: float = 0.0
    error: str = ""
    seconds: float = 0.0
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "task_type": self.task_type,
            "compiled": self.compiled, "correct": self.correct,
            "baseline_seconds": self.baseline_seconds,
            "optimized_seconds": self.optimized_seconds,
            "speedup": self.speedup, "score": self.score,
            # Wide enough to diagnose a failure from the ledger alone. The
            # in-memory .error carries the full diagnostic for retry feedback;
            # this is only what gets persisted.
            "error": self.error[:4000], "seconds": round(self.seconds, 2),
            "detail": self.detail,
        }


def _as_argv_list(value: Any) -> list[list[str]]:
    """AKA declares commands as a LIST of shell strings; keep them as shell.

    The commands embed quoting (``python3 -c "import ast; ..."``), so splitting
    them ourselves would corrupt them. They run through the shell exactly as
    written, which is also how the reference runner executes them.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [[value]]
    return [[str(v)] for v in value]


def load_task(config_path: str | os.PathLike) -> ArenaTask:
    import yaml

    p = Path(config_path)
    cfg = yaml.safe_load(p.read_text()) or {}
    plat = cfg.get("platform_support") or {}
    prompt = cfg.get("prompt") or {}
    root = p.parent
    task_type = str(cfg.get("task_type") or "")
    if not task_type:
        raise ArenaError(f"{p}: no task_type")
    return ArenaTask(
        # Absolute here; discover_tasks rewrites it to the tasks/-relative path
        # so ids match the ones the published results are reported under.
        task_id=str(root),
        task_type=task_type,
        root=root,
        source_files=[str(s) for s in (cfg.get("source_file_path") or [])],
        target_functions=[str(s) for s in (cfg.get("target_kernel_functions") or [])],
        compile_command=_as_argv_list(cfg.get("compile_command")),
        correctness_command=_as_argv_list(cfg.get("correctness_command")),
        performance_command=_as_argv_list(cfg.get("performance_command")),
        instructions=str(prompt.get("instructions") or ""),
        required_arch=(str(plat.get("required_arch")) if plat.get("required_arch") else None),
        target_file=(str(cfg["target_file_path"])
                     if cfg.get("target_file_path") else None),
    )


def discover_tasks(
    arena_root: str | os.PathLike,
    task_types: Optional[Sequence[str]] = None,
    gpu_arch: str = "gfx950",
) -> list[ArenaTask]:
    """Find tasks runnable on this architecture.

    A task with ``status: skip`` or a different ``required_arch`` is filtered
    before any GPU time is spent, which is what the reference runner does in
    preflight.
    """
    import yaml

    root = Path(arena_root)
    tasks_dir = root / "tasks"
    if not tasks_dir.is_dir():
        raise ArenaError(f"no tasks/ under {root}; clone AMD-AGI/AgentKernelArena")
    out: list[ArenaTask] = []
    for cfg_path in sorted(tasks_dir.rglob("config.yaml")):
        try:
            raw = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            continue
        plat = raw.get("platform_support") or {}
        if str(plat.get("status") or "active") == "skip":
            continue
        req = plat.get("required_arch")
        if req and str(req) != gpu_arch:
            continue
        ttype = str(raw.get("task_type") or "")
        if task_types and ttype not in task_types:
            continue
        try:
            task = load_task(cfg_path)
        except ArenaError:
            continue
        task.task_id = str(cfg_path.parent.relative_to(tasks_dir))
        out.append(task)
    return out


def _run(cmds: list[list[str]], cwd: Path, timeout: int) -> tuple[bool, str]:
    """Run each declared command in order; the first failure stops the gate."""
    for argv in cmds:
        try:
            proc = subprocess.run(
                argv[0], shell=True, cwd=str(cwd), timeout=timeout,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s: {argv[0][:120]}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        if proc.returncode != 0:
            return False, f"exit {proc.returncode}: {_diagnostic(proc)}"
    return True, ""


#: How much of a failing command's output to keep. A tail alone is close to
#: useless as retry feedback: hipcc and C++ template errors print the diagnostic
#: FIRST and the build system's "build stopped" summary LAST, so 400 trailing
#: characters reliably capture the summary and never the error. Keeping both ends
#: is what makes an attempt-2 fix possible.
_DIAG_HEAD = 3000
_DIAG_TAIL = 1500


def _diagnostic(proc) -> str:
    """The useful part of a failed command's output: the errors, then both ends."""
    out = ((proc.stderr or "") + ("\n" + proc.stdout if proc.stdout else "")).strip()
    if len(out) <= _DIAG_HEAD + _DIAG_TAIL:
        return out
    errs = [ln for ln in out.splitlines()
            if "error:" in ln.lower() or "Error:" in ln][:20]
    parts = []
    if errs:
        parts.append("ERRORS:\n" + "\n".join(errs))
    parts.append(out[:_DIAG_HEAD])
    parts.append(f"...[{len(out) - _DIAG_HEAD - _DIAG_TAIL} chars omitted]...")
    parts.append(out[-_DIAG_TAIL:])
    return "\n".join(parts)


def _parse_speedup(text: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Pull baseline/optimized/speedup out of a performance harness's output.

    Harnesses differ across suites, so accept a JSON blob if one is present and
    otherwise fall back to labelled numbers. Returning None rather than guessing
    matters: a fabricated speedup would enter the score directly.
    """
    import re

    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            sp = d.get("speedup_ratio", d.get("speedup"))
            return (
                _num(d.get("baseline_time", d.get("baseline_seconds"))),
                _num(d.get("optimized_time", d.get("optimized_seconds"))),
                _num(sp),
            )
    # The gpumode harnesses print a per-case "speedup=" line for every shape and
    # THEN an "Average: ... speedup=" line. Taking the first match records case 0
    # and calls it the task's speedup -- wrong on all 79 hip2hip/torch2hip tasks,
    # which are the two categories carrying the 6.69x and 6.89x bars, and wrong in
    # an unpredictable direction because it depends on whether the first shape
    # happened to be favourable. AKA averages the per-case ratios, so prefer an
    # explicitly-labelled average and fall back to the mean of the per-case lines.
    avg = re.search(r"average[^\n]*?speedup[_ ]?(?:ratio)?\s*[:=]\s*"
                    r"([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", text, re.I)
    if avg:
        return None, None, _num(avg.group(1))
    per_case = re.findall(r"case\s+\d+[^\n]*?speedup[_ ]?(?:ratio)?\s*[:=]\s*"
                          r"([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", text, re.I)
    vals = [v for v in (_num(x) for x in per_case) if v]
    if vals:
        return None, None, sum(vals) / len(vals)
    m = re.search(r"speedup[_ ]?(?:ratio)?\s*[:=]\s*"
                  r"([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", text, re.I)
    if m:
        return None, None, _num(m.group(1))

    # The GEAK suites -- which are most of the arena, all of triton2triton and
    # triton2flydsl -- report an absolute geomean latency and no ratio at all,
    # because a single run has nothing to compare against. Their number is still
    # the measurement we want; the denominator just has to come from the baseline
    # run of the same task. Returning it as the optimized time lets the caller
    # divide, instead of discarding a timing that was taken successfully.
    #
    # This is why 92 of 95 correct triton2triton kernels scored as if they had no
    # speedup: the harness measured them fine and the parser only knew one format.
    m = re.search(r"GEAK_RESULT_LATENCY_MS\s*=\s*([0-9]*\.?[0-9]+)", text)
    if m:
        return None, _num(m.group(1)), None
    return None, None, None


#: Where the harnesses write their timings, in AKA's own precedence order
#: (src/performance.py: performance_report_candidates).
_REPORT_CANDIDATES = (
    ("build", "performance_report.json"), ("performance_report.json",),
    ("build", "perf_report.json"), ("perf_report.json",),
    ("perf", "benchmark_results.json"),
)

#: Device-time keys, preferred over host/wall time. AKA's comment is that host
#: timings "can be gamed by editing the test harness", so a report offering only
#: host_time_ms is treated as no measurement rather than a weak one.
_DEVICE_TIME_KEYS = ("execution_time_ms", "device_time_ms", "gpu_time_ms",
                     "elapsed_ms", "time_ms")


def _case_time(case: dict) -> Optional[float]:
    for k in _DEVICE_TIME_KEYS:
        if k in case:
            return _num(case[k])
    timing = case.get("timing_ms")
    if isinstance(timing, dict):
        return _num(timing.get("mean"))
    return None


def _parse_report_files(workspace: Path, task_type: str):
    """Timings from the JSON report a harness wrote, if any.

    Returns (baseline, optimized, speedup) in the same shape as _parse_speedup.
    An explicit ratio in the report wins; otherwise the per-case device times are
    returned as an aggregate so the caller can divide by a baseline run.
    """
    for parts in _REPORT_CANDIDATES:
        p = workspace.joinpath(*parts)
        if not p.is_file():
            continue
        try:
            doc = json.loads(p.read_text())
        except Exception:  # noqa: BLE001 - a torn report is not a measurement
            continue
        if isinstance(doc, dict):
            sp = _num(doc.get("speedup_ratio", doc.get("speedup")))
            if sp:
                return (_num(doc.get("ori_time")), _num(doc.get("opt_time")), sp)
            cases = doc.get("test_cases") or doc.get("cases") or []
        elif isinstance(doc, list):
            cases = doc
        else:
            continue
        times = [t for t in (_case_time(c) for c in cases
                             if isinstance(c, dict)) if t]
        if times:
            # Mean over cases, matching how the same quantity is aggregated on
            # the baseline side, so the eventual ratio compares like with like.
            return None, sum(times) / len(times), None
    return None, None, None


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f > 0 else None


def score_result(compiled: bool, correct: bool, speedup: Optional[float]) -> float:
    """AgentKernelArena's default scoring policy, unchanged."""
    if not compiled:
        return 0.0
    if not correct:
        return float(COMPILE_POINTS)
    return float(COMPILE_POINTS + CORRECT_POINTS + (speedup or 0.0) * SPEEDUP_POINTS)


def evaluate_task(
    task: ArenaTask,
    workspace: Path,
    timeout: int = 900,
    reference_latency: Optional[float] = None,
) -> ArenaResult:
    """Compile, check, then time -- gated in that order, as AKA does.

    Gating matters for honesty: timing a kernel that failed correctness would
    reward exactly the shortcut we filter out of the training data.

    ``reference_latency`` is the same task's latency from a baseline run, in the
    units its own harness reports. Suites that print an absolute latency rather
    than a ratio -- the GEAK ones, which are most of the arena -- can only yield a
    speedup by comparison with it, so without it a perfectly good measurement is
    thrown away and a correct kernel scores as though it were no faster.
    """
    started = time.time()
    res = ArenaResult(task_id=task.task_id, task_type=task.task_type)

    ok, err = _run(task.compile_command, workspace, timeout)
    res.compiled = ok
    if not ok:
        res.error = err
        res.seconds = time.time() - started
        res.score = score_result(False, False, None)
        return res

    ok, err = _run(task.correctness_command, workspace, timeout)
    res.correct = ok
    if not ok:
        res.error = err
        res.seconds = time.time() - started
        res.score = score_result(True, False, None)
        return res

    if task.performance_command:
        try:
            proc = subprocess.run(
                task.performance_command[0][0], shell=True, cwd=str(workspace),
                timeout=timeout, capture_output=True, text=True,
            )
            base, opt, sp = _parse_speedup((proc.stdout or "") + (proc.stderr or ""))
            # AKA reads the JSON report FIRST and stdout only as a fallback, and
            # most harnesses write their timings there and print only a summary
            # line. Reading stdout alone leaves 301 of 402 tasks with no speedup
            # at all, capping their score at 120 -- including most of
            # triton2triton, the category any Opus comparison rests on.
            if sp is None and opt is None:
                base, opt, sp = _parse_report_files(workspace, task.task_type)
            if base is None and reference_latency:
                base = reference_latency
            res.baseline_seconds, res.optimized_seconds = base, opt
            if sp is None and base and opt:
                sp = base / opt
            res.speedup = sp
        except Exception as exc:  # noqa: BLE001 - a timing failure is not a correctness failure
            res.error = f"performance: {type(exc).__name__}: {exc}"

    res.score = score_result(res.compiled, res.correct, res.speedup)
    res.seconds = time.time() - started
    return res


# Our HIP training tasks were written independently -- no AgentKernelArena
# source, references or shapes were copied, because training on the benchmark
# would destroy the only checkable "we beat Opus on HIP" claim we have. But some
# of them share an OPERATOR with an AKA task, which is unavoidable: every kernel
# corpus contains gelu and layernorm, and excluding the operators AKA happens to
# test would cripple the model rather than make the comparison cleaner.
#
# These counts are MEASURED, by scripts/audit_hip_tasks.py against the AKA
# checkout, and they are held here as data rather than prose so the test below
# can tie ``hip_tasks`` to the live registry.  That coupling is the point: when
# the HIP family grew from 20 to 188 tasks the prose said "20/20" and nothing
# failed, which is exactly how a disclosure goes stale while still being quoted.
OPERATOR_OVERLAP: dict = {
    "hip_tasks": 188,       # backend == "hip" tasks in the registry
    "shared": 86,           # of those, how many share an operator with any AKA task
    "hip2hip": 78,
    "torch2hip": 84,
    "measured_by": "scripts/audit_hip_tasks.py",
}

# The disclosure rides on the summary rather than living in a script someone
# has to remember to run, so a number cannot be quoted without it.
OPERATOR_OVERLAP_DISCLOSURE = (
    "KORE HIP training tasks were authored independently; no AgentKernelArena "
    "source, reference implementation or shape set was used. "
    f"{OPERATOR_OVERLAP['shared']}/{OPERATOR_OVERLAP['hip_tasks']} share an "
    f"operator with an AKA task ({OPERATOR_OVERLAP['hip2hip']} hip2hip, "
    f"{OPERATOR_OVERLAP['torch2hip']} torch2hip). See "
    f"{OPERATOR_OVERLAP['measured_by']} for the per-task breakdown."
)


def summarize(results: Sequence[ArenaResult]) -> dict:
    """Per-type aggregates next to the published Opus number for that type."""
    import statistics

    by_type: dict[str, list[ArenaResult]] = {}
    for r in results:
        by_type.setdefault(r.task_type, []).append(r)

    out: dict[str, Any] = {
        "n": len(results),
        "training_overlap_disclosure": OPERATOR_OVERLAP_DISCLOSURE,
        "by_type": {},
    }
    for ttype, rs in sorted(by_type.items()):
        speeds = [r.speedup for r in rs if r.correct and r.speedup]
        row = {
            "n": len(rs),
            "compiled": sum(r.compiled for r in rs),
            "correct": sum(r.correct for r in rs),
            "mean_speedup": (statistics.fmean(speeds) if speeds else None),
            "geomean_speedup": (
                statistics.geometric_mean(speeds) if speeds else None),
            "mean_score": statistics.fmean([r.score for r in rs]) if rs else 0.0,
        }
        bar = PUBLISHED_OPUS_MEAN_SPEEDUP.get(ttype)
        if bar is not None:
            row["opus_published_mean_speedup"] = bar
            row["beats_opus"] = (
                row["mean_speedup"] is not None and row["mean_speedup"] > bar)
        out["by_type"][ttype] = row
    return out
