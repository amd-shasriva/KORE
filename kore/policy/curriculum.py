"""Deterministic registered-task curriculum for trustworthy GRPO runs.

Selection is stratified by ``(operator_family, dtype)``.  Strata are served in a
fixed round-robin, while tasks within each stratum are permuted with SHA-256 and
an explicit epoch counter.  There is no dependence on Python's randomized
``hash()`` or on rank-local RNG state.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = "CurriculumStateV1"
SCHEDULER_VERSION = "registered-stratified-sha256-v1"


class CurriculumError(ValueError):
    """Raised when a task set or scheduler state violates an invariant."""


@dataclass(frozen=True)
class RegisteredTaskInfo:
    task_id: str
    operator_family: str
    dtype: str

    @property
    def stratum(self) -> tuple[str, str]:
        return (self.operator_family, self.dtype)


@dataclass(frozen=True)
class CurriculumStateV1:
    schema_version: str
    scheduler_version: str
    seed: int
    draw_index: int
    task_set_digest: str
    stratum_draw_counts: tuple[tuple[str, str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scheduler_version": self.scheduler_version,
            "seed": self.seed,
            "draw_index": self.draw_index,
            "task_set_digest": self.task_set_digest,
            "stratum_draw_counts": [
                {
                    "operator_family": family,
                    "dtype": dtype,
                    "draws": draws,
                }
                for family, dtype, draws in self.stratum_draw_counts
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CurriculumStateV1":
        if not isinstance(raw, Mapping):
            raise CurriculumError("curriculum state must be a mapping")
        expected = {
            "schema_version",
            "scheduler_version",
            "seed",
            "draw_index",
            "task_set_digest",
            "stratum_draw_counts",
        }
        missing = sorted(expected - set(raw))
        unknown = sorted(set(raw) - expected)
        if missing:
            raise CurriculumError(f"curriculum state missing: {', '.join(missing)}")
        if unknown:
            raise CurriculumError(f"unknown curriculum state: {', '.join(unknown)}")
        if raw["schema_version"] != SCHEMA_VERSION:
            raise CurriculumError(
                f"unsupported curriculum schema: {raw['schema_version']!r}"
            )
        if raw["scheduler_version"] != SCHEDULER_VERSION:
            raise CurriculumError(
                f"unsupported scheduler version: {raw['scheduler_version']!r}"
            )
        seed = raw["seed"]
        draw_index = raw["draw_index"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise CurriculumError("curriculum seed must be an integer")
        if (
            isinstance(draw_index, bool)
            or not isinstance(draw_index, int)
            or draw_index < 0
        ):
            raise CurriculumError("curriculum draw_index must be non-negative")
        digest = raw["task_set_digest"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise CurriculumError("invalid task_set_digest")
        counts_raw = raw["stratum_draw_counts"]
        if not isinstance(counts_raw, list):
            raise CurriculumError("stratum_draw_counts must be a list")
        counts: list[tuple[str, str, int]] = []
        for item in counts_raw:
            if not isinstance(item, Mapping) or set(item) != {
                "operator_family",
                "dtype",
                "draws",
            }:
                raise CurriculumError("malformed stratum draw-count entry")
            family, dtype, draws = (
                item["operator_family"],
                item["dtype"],
                item["draws"],
            )
            if not isinstance(family, str) or not family:
                raise CurriculumError("operator_family must be non-empty")
            if not isinstance(dtype, str) or not dtype:
                raise CurriculumError("dtype must be non-empty")
            if isinstance(draws, bool) or not isinstance(draws, int) or draws < 0:
                raise CurriculumError("stratum draw count must be non-negative")
            counts.append((family, dtype, draws))
        if counts != sorted(counts):
            raise CurriculumError("stratum draw counts must be canonically sorted")
        return cls(
            schema_version=SCHEMA_VERSION,
            scheduler_version=SCHEDULER_VERSION,
            seed=seed,
            draw_index=draw_index,
            task_set_digest=digest,
            stratum_draw_counts=tuple(counts),
        )


def _default_registry_functions():
    from kore.tasks.registry import get_task, is_heldout, operator_family

    return get_task, is_heldout, operator_family


def _load_task_infos(
    task_ids: Sequence[str],
    *,
    task_loader: Optional[Callable[[str], Any]] = None,
    is_heldout_fn: Optional[Callable[[Any], bool]] = None,
    operator_family_fn: Optional[Callable[[Any], str]] = None,
    explicit_heldout_ids: Iterable[str] = (),
) -> tuple[RegisteredTaskInfo, ...]:
    if not isinstance(task_ids, Sequence) or isinstance(task_ids, (str, bytes)):
        raise CurriculumError("task_ids must be a sequence of registered task IDs")
    ids = list(task_ids)
    if not ids:
        raise CurriculumError("registered curriculum requires a non-empty task set")
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in ids):
        raise CurriculumError("every task ID must be a non-empty string")
    if len(ids) != len(set(ids)):
        raise CurriculumError("registered curriculum task IDs must be unique")

    default_loader, default_heldout, default_family = _default_registry_functions()
    loader = task_loader or default_loader
    heldout = is_heldout_fn or default_heldout
    family_of = operator_family_fn or default_family
    explicit_heldout = set(explicit_heldout_ids)

    overlap = sorted(set(ids) & explicit_heldout)
    if overlap:
        raise CurriculumError(
            f"training task set overlaps held-out IDs: {', '.join(overlap)}"
        )

    infos: list[RegisteredTaskInfo] = []
    for task_id in ids:
        try:
            task = loader(task_id)
        except Exception as exc:
            raise CurriculumError(f"task is not registered: {task_id}") from exc
        canonical_id = getattr(task, "task_id", None)
        if canonical_id != task_id:
            raise CurriculumError(
                f"registry returned task_id={canonical_id!r} for requested {task_id!r}"
            )
        if bool(heldout(task)):
            raise CurriculumError(f"held-out task cannot enter training: {task_id}")
        family = str(family_of(task) or "").strip().lower()
        dtype = str(getattr(task, "dtype", "") or "").strip().lower()
        if not family or not dtype:
            raise CurriculumError(
                f"task {task_id!r} lacks operator_family/dtype stratification metadata"
            )
        infos.append(
            RegisteredTaskInfo(
                task_id=task_id,
                operator_family=family,
                dtype=dtype,
            )
        )
    return tuple(sorted(infos, key=lambda info: info.task_id))


def _task_set_digest(infos: Sequence[RegisteredTaskInfo]) -> str:
    payload = [
        {
            "task_id": info.task_id,
            "operator_family": info.operator_family,
            "dtype": info.dtype,
        }
        for info in sorted(infos, key=lambda item: item.task_id)
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def registered_task_set_digest(
    task_ids: Sequence[str],
    *,
    task_loader: Optional[Callable[[str], Any]] = None,
    is_heldout_fn: Optional[Callable[[Any], bool]] = None,
    operator_family_fn: Optional[Callable[[Any], str]] = None,
    explicit_heldout_ids: Iterable[str] = (),
) -> str:
    """Validate a registered train set and return its immutable digest."""

    infos = _load_task_infos(
        task_ids,
        task_loader=task_loader,
        is_heldout_fn=is_heldout_fn,
        operator_family_fn=operator_family_fn,
        explicit_heldout_ids=explicit_heldout_ids,
    )
    return _task_set_digest(infos)


class RegisteredStratifiedScheduler:
    """Fair SHA/counter scheduler over an immutable registered train set."""

    schema_version = SCHEMA_VERSION
    scheduler_version = SCHEDULER_VERSION

    def __init__(
        self,
        task_ids: Sequence[str],
        *,
        seed: int = 0,
        task_loader: Optional[Callable[[str], Any]] = None,
        is_heldout_fn: Optional[Callable[[Any], bool]] = None,
        operator_family_fn: Optional[Callable[[Any], str]] = None,
        explicit_heldout_ids: Iterable[str] = (),
        state: Optional[CurriculumStateV1 | Mapping[str, Any]] = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise CurriculumError("scheduler seed must be an integer")
        self.seed = seed
        self._infos = _load_task_infos(
            task_ids,
            task_loader=task_loader,
            is_heldout_fn=is_heldout_fn,
            operator_family_fn=operator_family_fn,
            explicit_heldout_ids=explicit_heldout_ids,
        )
        self.task_set_digest = _task_set_digest(self._infos)
        by_stratum: dict[tuple[str, str], list[str]] = {}
        for info in self._infos:
            by_stratum.setdefault(info.stratum, []).append(info.task_id)
        self._by_stratum = {
            key: tuple(sorted(values)) for key, values in by_stratum.items()
        }
        self.strata = tuple(sorted(self._by_stratum))
        if not self.strata:
            raise CurriculumError("registered curriculum has no non-empty strata")
        self.draw_index = 0
        if state is not None:
            self.restore(state)

    def _stratum_counts(self, draw_index: Optional[int] = None) -> dict[tuple[str, str], int]:
        draw = self.draw_index if draw_index is None else draw_index
        cycles, remainder = divmod(draw, len(self.strata))
        return {
            stratum: cycles + (1 if index < remainder else 0)
            for index, stratum in enumerate(self.strata)
        }

    def _permutation(self, stratum: tuple[str, str], epoch: int) -> tuple[str, ...]:
        family, dtype = stratum

        def key(task_id: str) -> tuple[str, str]:
            material = (
                f"{SCHEDULER_VERSION}\0{self.seed}\0{family}\0{dtype}\0"
                f"{epoch}\0{task_id}"
            )
            return hashlib.sha256(material.encode("utf-8")).hexdigest(), task_id

        return tuple(sorted(self._by_stratum[stratum], key=key))

    def _task_for_draw(self, draw_index: int) -> str:
        if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index < 0:
            raise CurriculumError("draw index must be a non-negative integer")
        stratum = self.strata[draw_index % len(self.strata)]
        local_draw = draw_index // len(self.strata)
        tasks = self._by_stratum[stratum]
        epoch, position = divmod(local_draw, len(tasks))
        return self._permutation(stratum, epoch)[position]

    def next_task_id(self) -> str:
        """Select one task locally (single-process or rank 0 only)."""

        task_id = self._task_for_draw(self.draw_index)
        self.draw_index += 1
        return task_id

    def state(self) -> CurriculumStateV1:
        counts = self._stratum_counts()
        return CurriculumStateV1(
            schema_version=SCHEMA_VERSION,
            scheduler_version=SCHEDULER_VERSION,
            seed=self.seed,
            draw_index=self.draw_index,
            task_set_digest=self.task_set_digest,
            stratum_draw_counts=tuple(
                (family, dtype, counts[(family, dtype)])
                for family, dtype in self.strata
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return self.state().to_dict()

    def restore(self, state: CurriculumStateV1 | Mapping[str, Any]) -> None:
        parsed = (
            state
            if isinstance(state, CurriculumStateV1)
            else CurriculumStateV1.from_dict(state)
        )
        if parsed.seed != self.seed:
            raise CurriculumError(
                f"curriculum seed mismatch: state={parsed.seed}, scheduler={self.seed}"
            )
        if parsed.task_set_digest != self.task_set_digest:
            raise CurriculumError("curriculum task-set digest mismatch")
        expected_counts = self._stratum_counts(parsed.draw_index)
        expected = tuple(
            (family, dtype, expected_counts[(family, dtype)])
            for family, dtype in self.strata
        )
        if parsed.stratum_draw_counts != expected:
            raise CurriculumError(
                "curriculum stratum counters do not match the global draw index"
            )
        self.draw_index = parsed.draw_index

    def next_for_rank(
        self,
        *,
        rank: int,
        world_size: int,
        broadcast: Callable[[Optional[dict[str, Any]], int], dict[str, Any]],
    ) -> str:
        """Return rank 0's broadcast decision; followers never select locally.

        ``broadcast(payload, src)`` must return rank 0's payload on every rank.
        The payload carries both the chosen ID and post-draw scheduler state, so a
        follower can resume exactly without reconstructing rank 0's decision.
        """

        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size < 1
            or rank < 0
            or rank >= world_size
        ):
            raise CurriculumError(f"invalid rank/world_size: {rank}/{world_size}")
        pre_draw = self.draw_index
        payload: Optional[dict[str, Any]]
        if rank == 0:
            task_id = self.next_task_id()
            payload = {
                "scheduler_version": SCHEDULER_VERSION,
                "task_set_digest": self.task_set_digest,
                "pre_draw_index": pre_draw,
                "task_id": task_id,
                "post_state": self.state_dict(),
            }
        else:
            payload = None
        received = broadcast(payload, 0)
        if not isinstance(received, Mapping):
            raise CurriculumError("rank-0 curriculum broadcast returned no payload")
        required = {
            "scheduler_version",
            "task_set_digest",
            "pre_draw_index",
            "task_id",
            "post_state",
        }
        if set(received) != required:
            raise CurriculumError("malformed rank-0 curriculum payload")
        if received["scheduler_version"] != SCHEDULER_VERSION:
            raise CurriculumError("rank-0 scheduler version mismatch")
        if received["task_set_digest"] != self.task_set_digest:
            raise CurriculumError("rank-0 task-set digest mismatch")
        if received["pre_draw_index"] != pre_draw:
            raise CurriculumError(
                "rank-local curriculum state diverged before broadcast"
            )
        task_id = received["task_id"]
        if task_id not in {info.task_id for info in self._infos}:
            raise CurriculumError(f"rank 0 broadcast an unregistered task: {task_id!r}")
        if rank == 0:
            if dict(received["post_state"]) != self.state_dict():
                raise CurriculumError("rank 0 received a mutated curriculum payload")
        else:
            self.restore(received["post_state"])
            if self.draw_index != pre_draw + 1:
                raise CurriculumError("broadcast did not advance exactly one draw")
        return task_id

    def save_json(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.state_dict(), sort_keys=True, indent=2, allow_nan=False
        ) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return target

    @classmethod
    def from_json(
        cls,
        task_ids: Sequence[str],
        path: str | os.PathLike[str],
        **kwargs: Any,
    ) -> "RegisteredStratifiedScheduler":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            raise CurriculumError(f"cannot load curriculum state from {path}") from exc
        state = CurriculumStateV1.from_dict(raw)
        if "seed" in kwargs and kwargs["seed"] != state.seed:
            raise CurriculumError("explicit seed disagrees with curriculum state")
        kwargs["seed"] = state.seed
        return cls(task_ids, state=state, **kwargs)
