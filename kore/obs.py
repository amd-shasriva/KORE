"""KORE observability: one structured logger for the whole pipeline.

Goal: when a run is going, you can watch *everything* move — every datagen
attempt, teacher call (latency/tokens/retries), per-shape compile/SNR/bench,
every RL step (reward mean/std, advantages, KL, grad-norm, GPU mem), and every
campaign stage with timers/progress/ETA — as both human-readable console lines
and machine-readable JSONL events.

Design:
* Zero heavy deps (stdlib only); safe to import anywhere.
* Dual sink: aligned console (optional color on a tty) + append-only JSONL.
* Every line carries a wall timestamp, elapsed-since-run-start, level, logger
  name, and the active stage stack.
* Rich helpers: ``log``/``debug``/``info``/``warn``/``error``, ``event`` (pure
  JSONL), ``stage`` (timed context), ``timer`` (timed context), ``progress``
  (i/N + rate + ETA), ``metric`` (key/value scalars), ``heartbeat``.
* Level via ``KORE_LOG_LEVEL`` (DEBUG/INFO/WARN/ERROR), run dir via
  ``KORE_RUN_DIR`` or :func:`configure`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "WARNING": 30, "ERROR": 40}
_COLORS = {"DEBUG": "\033[2;37m", "INFO": "\033[0;36m", "WARN": "\033[0;33m",
           "ERROR": "\033[0;31m", "STAGE": "\033[1;35m", "METRIC": "\033[0;32m"}
_RESET = "\033[0m"


class _Run:
    """Process-wide logging context (start time, sinks, stage stack, level)."""

    def __init__(self) -> None:
        self.t0 = time.time()
        self.level = _LEVELS.get(os.environ.get("KORE_LOG_LEVEL", "INFO").upper(), 20)
        self.stage_stack: list[tuple[str, float]] = []
        self.lock = threading.RLock()
        self.jsonl_path: Optional[Path] = None
        self._jsonl_fh = None
        self.color = sys.stderr.isatty() and os.environ.get("KORE_LOG_COLOR", "1") != "0"
        self.counters: dict[str, float] = {}
        rd = os.environ.get("KORE_RUN_DIR")
        if rd:
            self.set_run_dir(rd)

    def set_run_dir(self, run_dir) -> None:
        with self.lock:
            p = Path(run_dir)
            p.mkdir(parents=True, exist_ok=True)
            self.jsonl_path = p / "events.jsonl"
            if self._jsonl_fh:
                try:
                    self._jsonl_fh.close()
                except Exception:
                    pass
            self._jsonl_fh = self.jsonl_path.open("a", encoding="utf-8")

    def write_jsonl(self, rec: dict) -> None:
        with self.lock:
            if self._jsonl_fh is None:
                return
            self._jsonl_fh.write(json.dumps(rec, default=str) + "\n")
            self._jsonl_fh.flush()


_RUN = _Run()


def configure(run_dir=None, level: Optional[str] = None, color: Optional[bool] = None) -> None:
    if run_dir is not None:
        _RUN.set_run_dir(run_dir)
    if level is not None:
        _RUN.level = _LEVELS.get(level.upper(), _RUN.level)
    if color is not None:
        _RUN.color = bool(color)


def _fmt_elapsed(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _fmt_fields(fields: dict) -> str:
    out = []
    for k, v in fields.items():
        if isinstance(v, float):
            v = f"{v:.4g}"
        out.append(f"{k}={v}")
    return " ".join(out)


class KoreLogger:
    def __init__(self, name: str) -> None:
        self.name = name

    # -- core ------------------------------------------------------------- #
    def _emit(self, level: str, msg: str, fields: dict, kind: str = "log") -> None:
        lvl = _LEVELS.get(level, 20)
        stage = _RUN.stage_stack[-1][0] if _RUN.stage_stack else "-"
        rec = {"ts": time.time(), "elapsed_s": round(time.time() - _RUN.t0, 3),
               "level": level, "kind": kind, "logger": self.name, "stage": stage,
               "msg": msg, **({"fields": fields} if fields else {})}
        _RUN.write_jsonl(rec)
        if lvl < _RUN.level:
            return
        el = _fmt_elapsed(time.time() - _RUN.t0)
        tag = level if kind == "log" else kind.upper()
        prefix = f"[{el}] {tag:<6} {self.name}"
        if stage != "-":
            prefix += f" ({stage})"
        line = f"{prefix}: {msg}"
        if fields:
            line += "  " + _fmt_fields(fields)
        if _RUN.color:
            c = _COLORS.get(tag, "")
            line = f"{c}{line}{_RESET}" if c else line
        print(line, file=sys.stderr, flush=True)

    def debug(self, msg: str, **fields) -> None: self._emit("DEBUG", msg, fields)
    def info(self, msg: str, **fields) -> None: self._emit("INFO", msg, fields)
    def warn(self, msg: str, **fields) -> None: self._emit("WARN", msg, fields)
    def warning(self, msg: str, **fields) -> None: self._emit("WARN", msg, fields)
    def error(self, msg: str, **fields) -> None: self._emit("ERROR", msg, fields)

    def event(self, name: str, **fields) -> None:
        """Structured event (JSONL always; console at INFO).

        ``name`` is positional so callers may freely pass a ``kind=`` data field
        (e.g. ``event("verify_shape", kind="ok")``) without colliding with the
        record type.
        """
        self._emit("INFO", name, fields, kind="event")

    def metric(self, msg: str = "metrics", **kv) -> None:
        self._emit("INFO", msg, kv, kind="metric")

    # -- timed contexts --------------------------------------------------- #
    @contextmanager
    def stage(self, name: str, **fields):
        _RUN.stage_stack.append((name, time.time()))
        self._emit("INFO", f"stage start: {name}", fields, kind="stage")
        t0 = time.time()
        try:
            yield self
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            self._emit("ERROR", f"stage FAILED: {name} ({type(e).__name__}: {e})",
                       {"elapsed_s": round(dt, 2)}, kind="stage")
            raise
        else:
            dt = time.time() - t0
            self._emit("INFO", f"stage done: {name}", {"elapsed_s": round(dt, 2), **fields},
                       kind="stage")
        finally:
            if _RUN.stage_stack:
                _RUN.stage_stack.pop()

    @contextmanager
    def timer(self, label: str, **fields):
        t0 = time.time()
        try:
            yield
        finally:
            self._emit("DEBUG", f"{label}", {"took_s": round(time.time() - t0, 3), **fields},
                       kind="timer")

    # -- progress --------------------------------------------------------- #
    def progress(self, i: int, n: int, label: str = "", *, t_start: Optional[float] = None,
                 **fields) -> None:
        pct = 100.0 * i / n if n else 0.0
        extra = {}
        if t_start is not None and i > 0:
            rate = i / max(time.time() - t_start, 1e-9)
            eta = (n - i) / rate if rate > 0 else float("inf")
            extra = {"rate_per_s": round(rate, 3), "eta": _fmt_elapsed(eta)}
        self._emit("INFO", f"{label} {i}/{n} ({pct:.0f}%)", {**extra, **fields},
                   kind="progress")

    def heartbeat(self, label: str = "alive", **fields) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                fields.setdefault("gpu_mem_gb", round(torch.cuda.max_memory_allocated() / 1e9, 2))
        except Exception:
            pass
        self._emit("DEBUG", label, fields, kind="heartbeat")


def get_logger(name: str) -> KoreLogger:
    return KoreLogger(name)


def gpu_mem_snapshot() -> dict:
    """Per-visible-GPU peak allocated GB (best-effort)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {f"gpu{i}_gb": round(torch.cuda.max_memory_allocated(i) / 1e9, 2)
                for i in range(torch.cuda.device_count())}
    except Exception:
        return {}
