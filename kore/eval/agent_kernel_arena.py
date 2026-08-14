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
import threading
import sys
from functools import lru_cache
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

#: Fraction of a type's CORRECT tasks that must carry a measured speedup before
#: its mean may be compared against the published bar above.
#:
#: The bars are full-category means. A mean over a handful of measured tasks is a
#: different quantity, and the bias runs one way: correct-but-unmeasured tasks
#: are dropped from our mean rather than counted as 1.0x, so sparse coverage
#: flatters us. The 2026-08-10 sweep reported "triton2triton mean_speedup 4.42,
#: beats_opus true" from 5 measurements across 159 correct tasks -- 3% coverage,
#: and nothing in the output said so.
#:
#: 0.8 rather than 1.0 because a few tasks legitimately refuse to yield a
#: comparable number (a benchmark-method mismatch, a harness that times only one
#: shape), and demanding perfection would suppress an otherwise sound comparison.
MIN_SPEEDUP_COVERAGE = 0.8


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
    #: Per-case timing rows exactly as the harness reported them. Persisted so a
    #: baseline run's cases can be paired shape-by-shape with an optimized run's
    #: later, which is how AKA computes a speedup; an aggregate cannot be
    #: re-matched after the fact.
    perf_cases: list = field(default_factory=list)
    #: How the speedup was obtained, or why it is missing or suspect. Without it a
    #: correct-but-untimed task and a correct-but-not-faster task are both a bare
    #: 120 and cannot be told apart from the ledger.
    speedup_note: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "task_type": self.task_type,
            "compiled": self.compiled, "correct": self.correct,
            "baseline_seconds": self.baseline_seconds,
            "optimized_seconds": self.optimized_seconds,
            "speedup": self.speedup, "score": self.score,
            "perf_cases": self.perf_cases,
            "speedup_note": self.speedup_note,
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


@lru_cache(maxsize=1)
def _task_env() -> dict:
    """The environment a task's own commands need.

    Tasks invoke ``python3``, not an interpreter we choose, and with the ambient
    environment that resolves to /usr/bin/python3 -- which has no torch. Whole
    families failed on that alone and none was judged on its code: a repository task
    died with ``ModuleNotFoundError: No module named 'torch'`` inside its own runner,
    and 19 torch2flydsl tasks failed every case with ``No module named 'aiter'``.

    So put the interpreter that has torch first on PATH, and add the aiter checkout
    to PYTHONPATH, since several tasks import it as a library rather than editing it.

    The GPU architecture is the third thing they need and the one we were not
    giving them. AKA sets it in ``src/preprocessing.setup_rocm_env``, which we
    never call because we drive the tasks directly rather than through their
    runner; its own docstring says that without it "PyTorch and CMake will fall
    back to their built-in arch lists", and gfx950 is new enough that falling back
    is not safe. Every torch2hip candidate -- 62 of 62 across both arms -- failed
    to compile while the models were producing plausible HIP.
    """
    env = dict(os.environ)
    venv_bin = str(Path(sys.executable).parent)
    if env.get("PATH", "").split(os.pathsep)[:1] != [venv_bin]:
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    aiter = Path.home() / "third_party" / "aiter"
    if aiter.is_dir():
        parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        if str(aiter) not in parts:
            env["PYTHONPATH"] = os.pathsep.join([str(aiter), *parts])
    arch = _detect_gfx_arch()
    if arch:
        # All three, together, exactly as AKA does: PyTorch reads
        # PYTORCH_ROCM_ARCH and CMake-based HIP builds read the other two, and a
        # task that mixes both must not see two different architectures.
        for var in ("PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS"):
            env.setdefault(var, arch)
    return env


@lru_cache(maxsize=1)
def _detect_gfx_arch() -> str:
    """The gfx target of the GPU this worker was given, or "" if unknowable.

    Asks torch before rocminfo: torch is what compiles the extension, so its view
    of the device is the one that has to be satisfied, and it is already loaded
    here. The feature suffix is stripped -- torch reports
    ``gfx950:sramecc+:xnack-`` and PYTORCH_ROCM_ARCH wants the bare target.

    Returns "" rather than guessing. A wrong architecture compiles cleanly and
    then fails at launch, which is far harder to diagnose than the missing-arch
    fallback it would be papering over.
    """
    try:
        import torch  # noqa: PLC0415 - heavy, and only needed on a GPU worker

        if torch.cuda.is_available() and torch.cuda.device_count():
            name = torch.cuda.get_device_properties(0).gcnArchName or ""
            if name.startswith("gfx"):
                return name.split(":")[0]
    except Exception:  # noqa: BLE001 - fall through to rocminfo
        pass
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True,
                             timeout=30)
        for line in (out.stdout or "").splitlines():
            if "gfx" in line and "Name:" in line:
                tok = line.split()[-1].strip()
                if tok.startswith("gfx"):
                    return tok.split(":")[0]
    except Exception:  # noqa: BLE001 - no rocminfo on a login node
        pass
    return ""


#: Held while a task times its kernel. Generation and compilation overlap
#: freely; timing must not. Several kernels benchmarked at once on one GPU
#: all read slower than they are, which biases every speedup downward
#: without looking wrong.
_BENCH_LOCK = threading.Lock()


def _run(cmds: list[list[str]], cwd: Path, timeout: int) -> tuple[bool, str]:
    """Run each declared command in order; the first failure stops the gate."""
    for argv in cmds:
        try:
            proc = subprocess.run(
                argv[0], shell=True, cwd=str(cwd), timeout=timeout,
                capture_output=True, text=True, env=_task_env(),
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
    # A GEAK harness computes its own geometric mean across every shape it timed
    # and prints it as GEAK_RESULT_GEOMEAN_SPEEDUP. That number is authoritative:
    # it is derived in-process from all shapes, by the same code that produced the
    # per-shape lines. Preferring it is not a nicety -- 32 harnesses print it, and
    # without this branch the generic "speedup=" search below matches the FIRST
    # per-shape line instead. Measured against a real harness transcript, that
    # reported 1.2x where the harness's own geomean was 4.5789x, understating the
    # kernel by 3.8x and feeding it straight into 120 + speedup*100.
    #
    # Checked after the JSON blob (an explicit speedup_ratio still wins) and before
    # the per-case regexes (which are the gpumode dialect and match the wrong line
    # here). A non-positive value is the harness's failure sentinel, e.g.
    # GEAK_RESULT_GEOMEAN_SPEEDUP=-1 in flydsl2flydsl/pa_decode_swa_kernel, so it
    # deliberately falls through rather than being reported as a measurement --
    # the latency branch below may still yield a usable optimized time.
    geo = re.search(r"GEAK_RESULT_GEOMEAN_SPEEDUP\s*=\s*"
                    r"(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", text)
    if geo is not None:
        val = _num(geo.group(1))
        if val:
            return None, None, val

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


def _case_method(case: dict) -> Optional[str]:
    """How this case was timed, if the harness said.

    Kept because the method dominates the number when it differs between the two
    sides being divided. Measured on gfx950, the same torch.relu call reads
    0.00514 ms captured in a CUDA graph and 0.01232 ms with per-launch events --
    a 2.4x gap from the measurement alone. AKA carries this field and warns on a
    mismatch (src/evaluator.py); dividing a graph-timed baseline by an
    event-timed optimized run manufactures a 2.4x regression out of nothing.
    """
    m = case.get("benchmark_method")
    return str(m) if m else None


def _report_cases(workspace: Path) -> list[dict]:
    """Per-case rows from whichever report file a harness wrote.

    Returns the raw case dicts rather than an aggregate so the caller can match
    baseline against optimized case-by-case, which is how AKA computes a speedup.
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
            cases = doc.get("test_cases") or doc.get("cases") or []
        elif isinstance(doc, list):
            cases = doc
        else:
            continue
        rows = [c for c in cases if isinstance(c, dict)]
        if rows:
            return rows
    return []


def _parse_report_files(workspace: Path, task_type: str = ""):
    """Timings from the JSON report a harness wrote, if any.

    Returns (baseline, optimized, speedup) in the same shape as _parse_speedup.
    An explicit ratio in the report wins; otherwise the per-case device times are
    returned as an aggregate so the caller can divide by a baseline run.

    The aggregate here is a fallback for the case where per-case matching is not
    possible (a baseline with no stored cases). Prefer speedup_from_cases, which
    matches cases and averages their ratios the way AKA does; a mean of times
    divided by a mean of times is dominated by the largest shape and is not the
    quantity the published bars were computed with.
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


def _case_key(case: dict, index: int) -> str:
    """Identity used to pair a baseline case with an optimized one.

    Mirrors AKA's precedence (src/testcases.py: match_test_cases): params, then
    shape, then an explicit id, then position. Position is last because a harness
    that skips a failing shape shifts every later index, which would silently
    divide one shape's time by another's.
    """
    for k in ("params", "shape"):
        v = case.get(k)
        if v not in (None, "", [], {}):
            return f"{k}={json.dumps(v, sort_keys=True)}"
    for k in ("test_case_id", "case_id", "name", "op_name"):
        v = case.get(k)
        if v not in (None, ""):
            return f"id={v}"
    return f"index={index}"


def speedup_from_cases(baseline: list[dict], optimized: list[dict]):
    """Mean of per-case speedup ratios, the way AKA computes it.

    Returns (speedup, note). ``note`` is None when the number is clean, otherwise
    a short reason the caller should surface rather than silently dropping.

    Two rules are taken from AKA rather than invented here:

    * Match cases by identity and require the sets to agree. AKA refuses outright
      on an incomplete match ("Incomplete test case match, refusing to calculate
      speedup") because a partial pairing compares different shapes.
    * Average the per-case RATIOS, not the ratio of the averages. With a baseline
      of [1.0, 100.0] ms against an optimized [0.1, 100.0] ms, mean-of-ratios is
      5.5x and ratio-of-means is 1.005x. The published bars are the former.
    """
    if not baseline or not optimized:
        return None, "no per-case timings on one side"
    b = {_case_key(c, i): c for i, c in enumerate(baseline)}
    o = {_case_key(c, i): c for i, c in enumerate(optimized)}
    shared = set(b) & set(o)
    if not shared:
        return None, "no case identities in common"
    if len(shared) != len(b) or len(shared) != len(o):
        return None, (f"incomplete case match: {len(shared)} shared of "
                      f"{len(b)} baseline / {len(o)} optimized")
    ratios, methods_b, methods_o = [], set(), set()
    for k in sorted(shared):
        tb, to = _case_time(b[k]), _case_time(o[k])
        if not tb or not to:
            return None, f"non-positive timing for case {k}"
        ratios.append(tb / to)
        mb, mo = _case_method(b[k]), _case_method(o[k])
        if mb:
            methods_b.add(mb)
        if mo:
            methods_o.add(mo)
    sp = sum(ratios) / len(ratios)
    if methods_b and methods_o and methods_b != methods_o:
        # Reported, not suppressed: the number is real but is measuring the
        # method delta as much as the kernel, and a caller that cannot see that
        # will read it as kernel quality.
        return sp, (f"benchmark method mismatch: baseline={sorted(methods_b)} "
                    f"optimized={sorted(methods_o)}")
    return sp, None


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
    reference_cases: Optional[list] = None,
) -> ArenaResult:
    """Compile, check, then time -- gated in that order, as AKA does.

    Gating matters for honesty: timing a kernel that failed correctness would
    reward exactly the shortcut we filter out of the training data.

    ``reference_latency`` is the same task's latency from a baseline run, in the
    units its own harness reports. Suites that print an absolute latency rather
    than a ratio -- the GEAK ones, which are most of the arena -- can only yield a
    speedup by comparison with it, so without it a perfectly good measurement is
    thrown away and a correct kernel scores as though it were no faster.

    ``reference_cases`` is the same run's PER-CASE rows. When present it is
    preferred over ``reference_latency``, because pairing shapes and averaging
    their ratios is what AKA does and what the published bars were computed with;
    dividing one aggregate by another is dominated by the largest shape.
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
            # One benchmark on this GPU at a time.
            #
            # Generation, compilation and correctness all overlap happily -- that is
            # what makes several tasks in flight worthwhile. Timing does not: a
            # kernel benchmarked while three other kernels share the same GPU reads
            # slower than it is, and speedup is one of the three things scored here.
            # Raising concurrency without this lock buys throughput by corrupting the
            # measurement the run exists to produce, and the corruption is invisible
            # -- the numbers look plausible, just wrong, and wrong in the direction
            # that understates every kernel.
            # Delete any report file BEFORE timing, so a perf command that fails
            # to write one cannot be scored against a stale file. 17 vLLM tasks
            # ship a build/performance_report.json committed to the AKA repo
            # (force-added past its own .gitignore), and _workspace() copies the
            # task tree wholesale. Without this, a failed benchmark silently
            # adopts AMD's committed number as ours -- and on the baseline side
            # that number becomes the denominator for every later comparison.
            # AKA clears these itself (src/performance.py:
            # clear_performance_report_files); we were not.
            for parts in _REPORT_CANDIDATES:
                stale = workspace.joinpath(*parts)
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
                except OSError:  # noqa: PERF203 - a locked report is not fatal
                    pass

            with _BENCH_LOCK:
                proc = subprocess.run(
                    task.performance_command[0][0], shell=True, cwd=str(workspace),
                    timeout=timeout, capture_output=True, text=True,
                    env=_task_env(),
                )
            base, opt, sp = _parse_speedup((proc.stdout or "") + (proc.stderr or ""))
            # Keep the per-case rows whatever else happens: they are what lets a
            # later run pair shape-by-shape instead of dividing two aggregates.
            res.perf_cases = _report_cases(workspace)
            note = ""

            # AKA reads the JSON report FIRST and stdout only as a fallback, and
            # most harnesses write their timings there and print only a summary
            # line. Reading stdout alone leaves 301 of 402 tasks with no speedup
            # at all, capping their score at 120 -- including most of
            # triton2triton, the category any Opus comparison rests on.
            if sp is None and opt is None:
                base, opt, sp = _parse_report_files(workspace, task.task_type)

            # Matched per-case ratios beat any aggregate. Only reachable when the
            # baseline run stored its own cases; falls through quietly otherwise.
            if sp is None and reference_cases and res.perf_cases:
                sp_cases, why = speedup_from_cases(reference_cases, res.perf_cases)
                if sp_cases:
                    sp, note = sp_cases, (why or "per-case matched ratios")
                elif why:
                    note = f"per-case matching declined: {why}"

            if base is None and reference_latency:
                base = reference_latency
            res.baseline_seconds, res.optimized_seconds = base, opt
            if sp is None and base and opt:
                sp = base / opt
                note = note or "aggregate ratio (no per-case match available)"
            res.speedup = sp
            if sp is None and not note:
                note = ("timed but no denominator -- run `baseline` for this task"
                        if opt else "no timing parsed from harness output")
            res.speedup_note = note
        except Exception as exc:  # noqa: BLE001 - a timing failure is not a correctness failure
            res.error = f"performance: {type(exc).__name__}: {exc}"
            # Distinguish "the timing blew up" from "correct but not faster".
            # Both scored a bare 120 before, which is how a systematic timing
            # failure hid behind a plausible-looking correctness result.
            res.speedup_note = f"performance failed: {type(exc).__name__}"
    else:
        res.speedup_note = "task declares no performance_command"

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
        correct = [r for r in rs if r.correct]
        speeds = [r.speedup for r in correct if r.speedup]
        # Coverage is the denominator the mean has always been missing. Reporting
        # a mean without it is how "triton2triton mean_speedup 4.42, beats_opus
        # true" was read as a result when it was computed over 5 of 159 correct
        # tasks. A mean over 3% of a category is not that category's speedup.
        coverage = (len(speeds) / len(correct)) if correct else 0.0
        row = {
            "n": len(rs),
            "compiled": sum(r.compiled for r in rs),
            "correct": len(correct),
            "speedup_samples": len(speeds),
            "speedup_coverage": round(coverage, 4),
            "mean_speedup": (statistics.fmean(speeds) if speeds else None),
            "geomean_speedup": (
                statistics.geometric_mean(speeds) if speeds else None),
            "mean_score": statistics.fmean([r.score for r in rs]) if rs else 0.0,
        }
        # Why correct kernels went untimed, so a systematic harness problem is
        # visible in the summary instead of only in a per-task ledger nobody
        # reads. A timing failure and a genuinely un-improved kernel both scored
        # a bare 120 before and were indistinguishable here.
        notes: dict[str, int] = {}
        for r in correct:
            if not r.speedup and r.speedup_note:
                notes[r.speedup_note.split(":")[0]] = (
                    notes.get(r.speedup_note.split(":")[0], 0) + 1)
        if notes:
            row["unmeasured_reasons"] = dict(sorted(
                notes.items(), key=lambda kv: -kv[1]))
        bar = PUBLISHED_OPUS_MEAN_SPEEDUP.get(ttype)
        if bar is not None:
            row["opus_published_mean_speedup"] = bar
            # Comparable only against a mean over most of the category. The
            # published bar is a full-category mean, so claiming to beat it from
            # a handful of measured tasks compares two different quantities --
            # and it is the favourable direction, since unmeasured-but-correct
            # tasks are excluded rather than counted as 1.0x.
            if row["mean_speedup"] is None:
                row["beats_opus"] = None
                row["beats_opus_note"] = "no speedup measured"
            elif coverage < MIN_SPEEDUP_COVERAGE:
                row["beats_opus"] = None
                row["beats_opus_note"] = (
                    f"speedup coverage {coverage:.0%} is below the "
                    f"{MIN_SPEEDUP_COVERAGE:.0%} needed to compare against a "
                    f"full-category published mean ({len(speeds)}/{len(correct)} "
                    f"correct tasks measured)")
            else:
                row["beats_opus"] = row["mean_speedup"] > bar
        out["by_type"][ttype] = row
    return out
