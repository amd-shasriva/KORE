"""KoreEnv: the verified evaluation environment.

Wraps the KernelForge verifier contract into a single ``evaluate(task, source)``
call that returns a reward :class:`Observation`. Every candidate is run in an
*isolated* per-eval workdir (driver.py + reference.py + kernel.py copied in) so
the model's kernel can never mutate the task sources, and a hung/OOM/illegal-
memory kernel is contained in its own subprocess with a hard timeout.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from kore.config import CONFIG
from kore.env.replay import ReplayCache
from kore.reward.reward import Observation, scan_for_hacks
from kore.tasks.base import Shape, Task

_SNR = re.compile(r"SNR:\s*([-\d.eE]+)")
_ALLCLOSE = re.compile(r"allclose:\s*(True|False)", re.IGNORECASE)
_MEDIAN = re.compile(r"median_ms:\s*([-\d.eE]+)")
_COMPILE_ERR = re.compile(
    r"(SyntaxError|CompilationError|triton\..*Error|IndentationError|ImportError|"
    r"ModuleNotFoundError|NameError while compiling|out of resource|OutOfResources)",
    re.IGNORECASE,
)


class KoreEnv:
    """Task-bound verified environment. Construct one per task, then call
    ``step(source, ...)`` for each candidate kernel."""

    def __init__(self, task: Task, config=CONFIG, use_replay: bool = True,
                 correctness_timeout: int = 300, bench_timeout: int = 300):
        self.task = task
        self.cfg = config
        self.correctness_timeout = correctness_timeout
        self.bench_timeout = bench_timeout
        self.use_replay = use_replay
        self._cache_obj = ReplayCache(self.cfg.runs_dir / f"replay_{task.task_id}.jsonl") \
            if use_replay else None

    # ------------------------------------------------------------------ #
    def step(self, source: str, full_validation: bool = True,
             multi_shape: bool = True) -> Observation:
        """Evaluate one candidate. ``multi_shape`` runs the full shape set
        (primary + validation); otherwise only the primary/first shape."""
        return self.evaluate(self.task, source, shapes=self._shapes(multi_shape),
                             do_bench=full_validation)

    def _shapes(self, multi_shape: bool) -> list[Shape]:
        shapes = self.task.shapes or [Shape("default", {})]
        if multi_shape:
            return shapes
        primary = self.task.shape("primary") or self.task.shape("minimal") or shapes[0]
        return [primary]

    def _cache(self, task_id: str):
        return self._cache_obj

    # ------------------------------------------------------------------ #
    def evaluate(self, task: Task, source: str, shapes: Optional[list[Shape]] = None,
                 do_bench: bool = True) -> Observation:
        # anti-hack scan is free and must gate everything.
        hack = scan_for_hacks(source)
        if hack:
            return Observation(compiled=False, dtype=task.dtype, flagged_hack=True,
                               hack_reason=hack, error_text=f"reward-hack: {hack}")

        if self.use_replay and self._cache_obj is not None:
            cached = self._cache_obj.get(task.task_id, source)
            if cached is not None:
                return cached

        shapes = shapes or task.shapes or [Shape("default", {})]
        workdir = Path(tempfile.mkdtemp(prefix=f"kore_{task.task_id}_"))
        try:
            obs = self._run(task, source, shapes, workdir, do_bench)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        if self.use_replay and self._cache_obj is not None and (obs.compiled or obs.error_text):
            self._cache_obj.put(task.task_id, source, obs)
        return obs

    # ------------------------------------------------------------------ #
    def _env(self) -> dict:
        env = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[2])  # /root/Kore-rl/kore
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        env["HIP_VISIBLE_DEVICES"] = env.get("HIP_VISIBLE_DEVICES", "0")
        env["GPU_TARGET"] = self.cfg.gpu_target
        return env

    def _run(self, task: Task, source: str, shapes: list[Shape], workdir: Path,
             do_bench: bool) -> Observation:
        # stage isolated sources
        for p in task.dir.glob("*.py"):
            shutil.copy(p, workdir / p.name)
        (workdir / "kernel.py").write_text(source)
        driver = workdir / "driver.py"
        env = self._env()

        snr_by_shape: dict[str, float] = {}
        compiled = True
        validation_passed = True
        last_err: Optional[str] = None

        for sh in shapes:
            cmd = [sys.executable, str(driver), *sh.as_args()]
            try:
                r = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True,
                                   text=True, timeout=self.correctness_timeout)
            except subprocess.TimeoutExpired:
                return Observation(compiled=True, dtype=task.dtype, validation_passed=False,
                                   error_text=f"correctness timeout on shape {sh.name}")
            out = (r.stdout or "") + "\n" + (r.stderr or "")
            if r.returncode != 0:
                if _COMPILE_ERR.search(out):
                    compiled = False
                validation_passed = False
                last_err = _tail(out)
                # a compile failure on any shape is terminal
                if not compiled:
                    return Observation(compiled=False, dtype=task.dtype,
                                       validation_passed=False, error_text=last_err)
                continue
            m = _SNR.search(out)
            ac = _ALLCLOSE.search(out)
            if m:
                snr_by_shape[sh.name] = float(m.group(1))
            if ac and ac.group(1).lower() == "false":
                validation_passed = False
            if not m and not ac:
                validation_passed = False
                last_err = _tail(out)

        thr = self.cfg.snr_threshold_for(task.dtype)
        correct = validation_passed and bool(snr_by_shape) and all(v >= thr for v in snr_by_shape.values())

        obs = Observation(
            compiled=compiled, dtype=task.dtype,
            snr_by_shape=snr_by_shape,
            snr_db=min(snr_by_shape.values()) if snr_by_shape else None,
            validation_passed=correct, error_text=last_err if not correct else None,
        )
        if not (correct and do_bench):
            return obs

        # benchmark candidate + real production baseline on every shape
        wall_by_shape: dict[str, float] = {}
        base_by_shape: dict[str, float] = {}
        for sh in shapes:
            cand = self._bench_one(driver, sh, "candidate", workdir, env)
            ref = self._bench_one(driver, sh, "reference", workdir, env)
            if cand is not None:
                wall_by_shape[sh.name] = cand
            if ref is not None:
                base_by_shape[sh.name] = ref
        obs.wall_by_shape = wall_by_shape
        obs.baseline_by_shape = base_by_shape
        if wall_by_shape:
            obs.wall_ms = max(wall_by_shape.values())
        if base_by_shape:
            obs.baseline_ms = max(base_by_shape.values())
        return obs

    def _bench_one(self, driver: Path, sh: Shape, impl: str, workdir: Path,
                   env: dict) -> Optional[float]:
        cmd = [sys.executable, str(driver), "--bench-mode", "--impl", impl, *sh.as_args()]
        try:
            r = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True,
                               text=True, timeout=self.bench_timeout)
        except subprocess.TimeoutExpired:
            return None
        m = _MEDIAN.search((r.stdout or "") + (r.stderr or ""))
        return float(m.group(1)) if m else None


def _tail(s: str, n: int = 800) -> str:
    s = s.strip()
    return s[-n:] if len(s) > n else s
