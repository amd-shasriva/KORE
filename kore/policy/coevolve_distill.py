"""Co-evolution DISTILLATION SINK: route winning kernels into an on-disk RFT set.

The open-ended co-evolution loop (:mod:`kore.openended.coevolve`) discovers
kernels that are correct AND faster-than-baseline during GRPO. Those verified
wins are exactly the trajectories a later expert-iteration / RFT stage wants to
distill from (cf. :mod:`kore.data.rejection`, :mod:`kore.data.gen_wins`).

:class:`DistillationSink` is the ``distill_fn`` hook side of that loop. It is a
callable matching ``coevolve.DistillFn`` (``list[dict] -> int``): it converts each
qualifying win dict into a :class:`kore.data.schemas.WinRecord` and persists it to
a JSONL log via :func:`kore.data.schemas.write_jsonl`, so it round-trips losslessly
back through :func:`~kore.data.schemas.read_jsonl`.

Design (mirrors the datagen conventions):
  * FILTER  - keep only verified (if required) wins that beat baseline by
    ``min_speedup`` (a "win" elsewhere in KORE = correct AND speedup > 1).
  * MAP     - win dict ``{"descriptor", "kernel_src", "speedup", ...}`` ->
    ``WinRecord`` with a minimal but valid two-turn chat ``trajectory`` (a user
    turn stating the task + an assistant turn wrapping the final kernel in the
    repo's ``FULL_KERNEL:`` code-block convention, so it re-parses with
    :func:`kore.data.prompts.extract_kernel`).
  * DEDUP   - by a stable hash of ``(task_id, final_source)`` so re-emitting the
    same kernel never bloats the set; the highest-``speedup`` record per hash wins
    (like :func:`kore.data.rejection.stratified_rft_select`'s fastest-instance
    dedup), deduping across the existing file's contents on load.

Everything is pure/CPU: no ``torch`` import (not even lazily - the descriptor is
read purely via attribute access), so it is fully unit-testable without a GPU.
"""

from __future__ import annotations

import hashlib
import statistics
from pathlib import Path
from typing import Any, Optional, Union

from kore.data.schemas import (
    GPU_DEFAULT,
    WinRecord,
    read_jsonl,
    stamp_production_record,
    write_jsonl,
)


def _win_key(task_id: str, final_source: str) -> str:
    """Stable content hash of ``(task_id, final_source)`` for dedup.

    Independent of process / PYTHONHASHSEED so the same kernel always collapses to
    the same key across runs and re-instantiations of the sink."""
    h = hashlib.sha256()
    h.update(task_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(final_source.encode("utf-8"))
    return h.hexdigest()


def _desc_get(descriptor: Any, win: dict, name: str) -> Any:
    """Read ``name`` from the descriptor (attribute or dict) then the win dict.

    ``descriptor`` is a :class:`kore.openended.task_space.TaskDescriptor` in the
    real loop (``.task_id`` / ``.op`` are attributes), but we stay duck-typed so
    the sink is testable with plain dicts / stand-ins."""
    if descriptor is not None:
        val = getattr(descriptor, name, None)
        if val is not None:
            return val
        if isinstance(descriptor, dict) and descriptor.get(name) is not None:
            return descriptor.get(name)
    return win.get(name)


def _build_trajectory(task_id: str, operation: Optional[str], final_source: str) -> list[dict]:
    """A minimal but valid two-turn chat: task statement + final kernel.

    The assistant turn wraps the kernel in the repo's ``FULL_KERNEL:`` fenced
    convention (see :mod:`kore.data.prompts`), so it re-extracts cleanly with
    :func:`kore.data.prompts.extract_kernel`."""
    op_str = f" (operation: {operation})" if operation else ""
    user = (
        "Write and optimize a Triton kernel for the AMD Instinct MI350X "
        f"(gfx950 / CDNA4) for task `{task_id}`{op_str}. "
        "Output the complete kernel under the FULL_KERNEL: contract."
    )
    assistant = f"FULL_KERNEL:\n```python\n{final_source}\n```"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


class DistillationSink:
    """A ``distill_fn`` that funnels verified co-evolution wins into an RFT JSONL.

    Pass an instance directly as ``coevolve.run_generation(..., distill_fn=sink)``:
    :meth:`__call__` is :meth:`record`. Each call filters, maps, dedups and
    persists; it returns the number of NEW unique kernels written.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        min_speedup: float = 1.0,
        require_verified: bool = True,
        gpu: str = GPU_DEFAULT,
    ) -> None:
        self.path = Path(path)
        self.min_speedup = float(min_speedup)
        self.require_verified = bool(require_verified)
        self.gpu = gpu
        # key -> best (highest-speedup) WinRecord seen for that (task, source).
        self._best: dict[str, WinRecord] = {}
        self._load_existing()

    # -- persistence ------------------------------------------------------- #
    def _load_existing(self) -> None:
        """Seed the dedup table from any pre-existing file so we append + dedup
        across prior contents instead of clobbering or double-writing them."""
        for rec in read_jsonl(
            self.path, mode="production_strict"):
            if not isinstance(rec, WinRecord) or not rec.final_source:
                continue
            key = _win_key(rec.task_id, rec.final_source)
            cur = self._best.get(key)
            if cur is None or (rec.speedup or 0.0) > (cur.speedup or 0.0):
                self._best[key] = rec

    def _flush(self) -> None:
        """Rewrite the JSONL with the current best-per-hash set (creates parents)."""
        rows = []
        for key, record in self._best.items():
            rows.append(stamp_production_record(
                record,
                provenance_id="coevolve_distillation_sink_v1",
                evaluation_id=f"coevolve:{key}",
            ))
        write_jsonl(self.path, rows)

    # -- mapping ----------------------------------------------------------- #
    def _to_winrecord(self, win: Any) -> Optional[WinRecord]:
        """Map one win dict -> ``WinRecord``, or ``None`` if malformed/filtered.

        Filter: must be verified (when ``require_verified``) and beat baseline by
        ``>= min_speedup``. Malformed dicts (missing source / task_id / bad
        speedup) are skipped rather than raising."""
        if not isinstance(win, dict):
            return None

        final_source = win.get("kernel_src") or win.get("final_source")
        if not isinstance(final_source, str) or not final_source.strip():
            return None

        raw_speedup = win.get("speedup")
        try:
            speedup = float(raw_speedup) if raw_speedup is not None else None
        except (TypeError, ValueError):
            return None

        # win filter (a "win" = verified & faster-than-baseline).
        if self.require_verified and not bool(win.get("verified")):
            return None
        if speedup is None or speedup < self.min_speedup:
            return None

        descriptor = win.get("descriptor")
        task_id = _desc_get(descriptor, win, "task_id")
        if not task_id or not isinstance(task_id, str):
            return None

        operation = _desc_get(descriptor, win, "op") or _desc_get(descriptor, win, "operation")
        # shape provenance: prefer an explicit shape, else the descriptor's regime.
        shape = win.get("shape")
        if shape is None and descriptor is not None:
            shape = getattr(descriptor, "shape_regime", None)
        arch = win.get("arch") or _desc_get(descriptor, win, "arch") or self.gpu

        snr_db = win.get("snr_db")
        initial_wall_us = win.get("initial_wall_us")
        final_wall_us = win.get("final_wall_us")

        return WinRecord(
            task_id=task_id,
            trajectory=_build_trajectory(task_id, operation, final_source),
            initial_wall_us=initial_wall_us,
            final_wall_us=final_wall_us,
            speedup=speedup,
            final_source=final_source,
            snr_db=snr_db,
            gpu=self.gpu,
            operation=operation,
            arch=arch,
            shape=str(shape) if shape is not None else None,
        )

    # -- the DistillFn hook ------------------------------------------------- #
    def record(self, wins: list[dict]) -> int:
        """Convert + persist qualifying wins; return the count of NEW kernels.

        Deduplicates by ``(task_id, final_source)`` hash, keeping only the highest
        ``speedup`` record per hash (across this batch AND the existing file). A
        NEW record is a hash not previously present; an improved speedup for an
        existing hash updates it in place but is not counted as new."""
        if not wins:
            return 0

        new_count = 0
        changed = False
        for win in wins:
            try:
                rec = self._to_winrecord(win)
            except Exception:
                # never let one malformed win abort the whole batch.
                rec = None
            if rec is None:
                continue
            key = _win_key(rec.task_id, rec.final_source)
            cur = self._best.get(key)
            if cur is None:
                self._best[key] = rec
                new_count += 1
                changed = True
            elif (rec.speedup or 0.0) > (cur.speedup or 0.0):
                self._best[key] = rec
                changed = True

        if changed:
            self._flush()
        return new_count

    __call__ = record

    # -- introspection ----------------------------------------------------- #
    def stats(self) -> dict:
        """Summary of the current on-disk set: count, unique tasks, speedups."""
        recs = list(self._best.values())
        speeds = [r.speedup for r in recs if r.speedup is not None]
        return {
            "count": len(recs),
            "unique_tasks": len({r.task_id for r in recs}),
            "mean_speedup": (statistics.fmean(speeds) if speeds else None),
            "median_speedup": (statistics.median(speeds) if speeds else None),
        }
