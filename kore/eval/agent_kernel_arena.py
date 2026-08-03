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
            "error": self.error[:500], "seconds": round(self.seconds, 2),
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
            tail = (proc.stderr or proc.stdout or "").strip()[-400:]
            return False, f"exit {proc.returncode}: {tail}"
    return True, ""


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
    m = re.search(r"speedup[_ ]?(?:ratio)?\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.I)
    return None, None, (_num(m.group(1)) if m else None)


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
) -> ArenaResult:
    """Compile, check, then time -- gated in that order, as AKA does.

    Gating matters for honesty: timing a kernel that failed correctness would
    reward exactly the shortcut we filter out of the training data.
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
            res.baseline_seconds, res.optimized_seconds = base, opt
            if sp is None and base and opt:
                sp = base / opt
            res.speedup = sp
        except Exception as exc:  # noqa: BLE001 - a timing failure is not a correctness failure
            res.error = f"performance: {type(exc).__name__}: {exc}"

    res.score = score_result(res.compiled, res.correct, res.speedup)
    res.seconds = time.time() - started
    return res


def summarize(results: Sequence[ArenaResult]) -> dict:
    """Per-type aggregates next to the published Opus number for that type."""
    import statistics

    by_type: dict[str, list[ArenaResult]] = {}
    for r in results:
        by_type.setdefault(r.task_type, []).append(r)

    out: dict[str, Any] = {"n": len(results), "by_type": {}}
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
