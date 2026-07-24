"""Declarative HSACO launch-plan contract.

The models describe data; they do not load code, allocate devices, or implement
privileged GPU policy in the repository process.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from kore.sandbox.errors import InvalidLaunchPlan


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_.$@]{0,255}$")
_HEX256 = re.compile(r"^[0-9a-f]{64}$")


class BufferAccess(str, Enum):
    READ_ONLY = "read-only"
    WRITE_ONLY = "write-only"
    READ_WRITE = "read-write"


class ScalarType(str, Enum):
    BOOL = "bool"
    I8 = "i8"
    U8 = "u8"
    I16 = "i16"
    U16 = "u16"
    I32 = "i32"
    U32 = "u32"
    I64 = "i64"
    U64 = "u64"
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"


class ArgumentKind(str, Enum):
    BUFFER = "buffer"
    SCALAR = "scalar"


@dataclass(frozen=True)
class LaunchCaps:
    max_modules: int = 8
    max_symbols_per_module: int = 128
    max_hsaco_bytes: int = 64 * 1024 * 1024
    max_buffers: int = 128
    max_buffer_bytes: int = 8 * 1024 * 1024 * 1024
    max_total_buffer_bytes: int = 16 * 1024 * 1024 * 1024
    max_scalars: int = 256
    max_launches: int = 128
    max_arguments_per_launch: int = 256
    max_dependencies_per_launch: int = 64
    max_scratch_bytes_per_launch: int = 256 * 1024 * 1024
    max_total_scratch_bytes: int = 1024 * 1024 * 1024
    max_grid: tuple[int, int, int] = (2**31 - 1, 65535, 65535)
    max_block: tuple[int, int, int] = (1024, 1024, 64)
    max_threads_per_block: int = 1024

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name in ("max_grid", "max_block"):
                if len(value) != 3 or any(
                    isinstance(v, bool) or not isinstance(v, int) or v <= 0
                    for v in value
                ):
                    raise ValueError(f"{name} must contain three positive dimensions")
            elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class HsacoImage:
    name: str
    target: str
    sha256: str
    size_bytes: int
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(self.symbols))
        _valid_name(self.name, "module name")
        if not _NAME.fullmatch(self.target):
            raise InvalidLaunchPlan(f"invalid HSACO target: {self.target!r}")
        if not _HEX256.fullmatch(self.sha256):
            raise InvalidLaunchPlan("HSACO sha256 must be a lowercase digest")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise InvalidLaunchPlan("HSACO size_bytes must be positive")
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise InvalidLaunchPlan("HSACO symbols must be non-empty and unique")
        for symbol in self.symbols:
            if not _SYMBOL.fullmatch(symbol):
                raise InvalidLaunchPlan(f"invalid HSACO symbol: {symbol!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "symbols": list(self.symbols),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HsacoImage":
        return cls(
            name=str(value["name"]),
            target=str(value["target"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            symbols=tuple(str(v) for v in value["symbols"]),
        )


@dataclass(frozen=True)
class BufferSpec:
    name: str
    size_bytes: int
    access: BufferAccess | str
    alignment: int = 16
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        _valid_name(self.name, "buffer name")
        try:
            access = self.access if isinstance(self.access, BufferAccess) else BufferAccess(self.access)
        except (TypeError, ValueError) as exc:
            raise InvalidLaunchPlan(f"invalid buffer access: {self.access!r}") from exc
        object.__setattr__(self, "access", access)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise InvalidLaunchPlan("buffer size_bytes must be positive")
        if (
            isinstance(self.alignment, bool)
            or not isinstance(self.alignment, int)
            or self.alignment <= 0
            or self.alignment > 4096
            or self.alignment & (self.alignment - 1)
        ):
            raise InvalidLaunchPlan("buffer alignment must be a power of two <= 4096")
        if self.content_sha256 is not None and not _HEX256.fullmatch(self.content_sha256):
            raise InvalidLaunchPlan("buffer content_sha256 must be a lowercase digest")
        if access is BufferAccess.WRITE_ONLY and self.content_sha256 is not None:
            raise InvalidLaunchPlan("write-only buffers cannot have initial content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "access": self.access.value,
            "alignment": self.alignment,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BufferSpec":
        return cls(
            name=str(value["name"]),
            size_bytes=int(value["size_bytes"]),
            access=str(value["access"]),
            alignment=int(value.get("alignment", 16)),
            content_sha256=(
                str(value["content_sha256"])
                if value.get("content_sha256") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ScalarSpec:
    name: str
    dtype: ScalarType | str
    value: bool | int | float

    def __post_init__(self) -> None:
        _valid_name(self.name, "scalar name")
        try:
            dtype = self.dtype if isinstance(self.dtype, ScalarType) else ScalarType(self.dtype)
        except (TypeError, ValueError) as exc:
            raise InvalidLaunchPlan(f"invalid scalar type: {self.dtype!r}") from exc
        object.__setattr__(self, "dtype", dtype)
        _validate_scalar(dtype, self.value)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype.value, "value": self.value}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScalarSpec":
        return cls(name=str(value["name"]), dtype=str(value["dtype"]), value=value["value"])


@dataclass(frozen=True)
class GridSpec:
    grid: tuple[int, int, int]
    block: tuple[int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", tuple(self.grid))
        object.__setattr__(self, "block", tuple(self.block))
        for name, dims in (("grid", self.grid), ("block", self.block)):
            if len(dims) != 3 or any(
                isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in dims
            ):
                raise InvalidLaunchPlan(f"{name} must contain three positive integers")

    def to_dict(self) -> dict[str, Any]:
        return {"grid": list(self.grid), "block": list(self.block)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GridSpec":
        return cls(
            grid=tuple(int(v) for v in value["grid"]),
            block=tuple(int(v) for v in value["block"]),
        )


@dataclass(frozen=True)
class LaunchArgument:
    kind: ArgumentKind | str
    name: str

    def __post_init__(self) -> None:
        try:
            kind = self.kind if isinstance(self.kind, ArgumentKind) else ArgumentKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise InvalidLaunchPlan(f"invalid argument kind: {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        _valid_name(self.name, "argument name")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "name": self.name}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LaunchArgument":
        return cls(kind=str(value["kind"]), name=str(value["name"]))


@dataclass(frozen=True)
class KernelLaunch:
    launch_id: str
    module: str
    symbol: str
    arguments: tuple[LaunchArgument, ...]
    grid: GridSpec
    dependencies: tuple[str, ...] = ()
    scratch_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        _valid_name(self.launch_id, "launch id")
        _valid_name(self.module, "launch module")
        if not _SYMBOL.fullmatch(self.symbol):
            raise InvalidLaunchPlan(f"invalid launch symbol: {self.symbol!r}")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise InvalidLaunchPlan(f"duplicate dependencies for launch {self.launch_id}")
        if self.launch_id in self.dependencies:
            raise InvalidLaunchPlan(f"launch {self.launch_id} depends on itself")
        if (
            isinstance(self.scratch_bytes, bool)
            or not isinstance(self.scratch_bytes, int)
            or self.scratch_bytes < 0
        ):
            raise InvalidLaunchPlan("scratch_bytes cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "launch_id": self.launch_id,
            "module": self.module,
            "symbol": self.symbol,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "grid": self.grid.to_dict(),
            "dependencies": list(self.dependencies),
            "scratch_bytes": self.scratch_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelLaunch":
        return cls(
            launch_id=str(value["launch_id"]),
            module=str(value["module"]),
            symbol=str(value["symbol"]),
            arguments=tuple(LaunchArgument.from_dict(v) for v in value.get("arguments", ())),
            grid=GridSpec.from_dict(value["grid"]),
            dependencies=tuple(str(v) for v in value.get("dependencies", ())),
            scratch_bytes=int(value.get("scratch_bytes", 0)),
        )


@dataclass(frozen=True)
class LaunchPlan:
    modules: tuple[HsacoImage, ...]
    buffers: tuple[BufferSpec, ...]
    scalars: tuple[ScalarSpec, ...]
    launches: tuple[KernelLaunch, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(self, "buffers", tuple(self.buffers))
        object.__setattr__(self, "scalars", tuple(self.scalars))
        object.__setattr__(self, "launches", tuple(self.launches))
        if self.schema_version != 1:
            raise InvalidLaunchPlan(f"unsupported launch-plan schema: {self.schema_version}")
        self.validate(LaunchCaps())

    def validate(self, caps: LaunchCaps) -> None:
        if not self.modules or len(self.modules) > caps.max_modules:
            raise InvalidLaunchPlan("launch plan has an invalid module count")
        if len(self.buffers) > caps.max_buffers:
            raise InvalidLaunchPlan("launch plan exceeds buffer count cap")
        if len(self.scalars) > caps.max_scalars:
            raise InvalidLaunchPlan("launch plan exceeds scalar count cap")
        if not self.launches or len(self.launches) > caps.max_launches:
            raise InvalidLaunchPlan("launch plan has an invalid launch count")

        module_map = _unique_map(self.modules, "name", "module")
        buffer_map = _unique_map(self.buffers, "name", "buffer")
        scalar_map = _unique_map(self.scalars, "name", "scalar")
        launch_map = _unique_map(self.launches, "launch_id", "launch")

        total_hsaco = 0
        for module in self.modules:
            if len(module.symbols) > caps.max_symbols_per_module:
                raise InvalidLaunchPlan(f"module {module.name} exceeds symbol cap")
            total_hsaco += module.size_bytes
        if total_hsaco > caps.max_hsaco_bytes:
            raise InvalidLaunchPlan("launch plan exceeds total HSACO byte cap")

        total_buffers = 0
        for buffer in self.buffers:
            if buffer.size_bytes > caps.max_buffer_bytes:
                raise InvalidLaunchPlan(f"buffer {buffer.name} exceeds byte cap")
            total_buffers += buffer.size_bytes
        if total_buffers > caps.max_total_buffer_bytes:
            raise InvalidLaunchPlan("launch plan exceeds total buffer byte cap")

        total_scratch = 0
        for launch in self.launches:
            module = module_map.get(launch.module)
            if module is None:
                raise InvalidLaunchPlan(
                    f"launch {launch.launch_id} references unknown module {launch.module}"
                )
            if launch.symbol not in module.symbols:
                raise InvalidLaunchPlan(
                    f"launch {launch.launch_id} references undeclared symbol {launch.symbol}"
                )
            if len(launch.arguments) > caps.max_arguments_per_launch:
                raise InvalidLaunchPlan(f"launch {launch.launch_id} exceeds argument cap")
            if len(launch.dependencies) > caps.max_dependencies_per_launch:
                raise InvalidLaunchPlan(f"launch {launch.launch_id} exceeds dependency cap")
            for dependency in launch.dependencies:
                if dependency not in launch_map:
                    raise InvalidLaunchPlan(
                        f"launch {launch.launch_id} has unknown dependency {dependency}"
                    )
            for argument in launch.arguments:
                target = buffer_map if argument.kind is ArgumentKind.BUFFER else scalar_map
                if argument.name not in target:
                    raise InvalidLaunchPlan(
                        f"launch {launch.launch_id} references unknown "
                        f"{argument.kind.value} {argument.name}"
                    )
            for axis, (actual, maximum) in enumerate(zip(launch.grid.grid, caps.max_grid)):
                if actual > maximum:
                    raise InvalidLaunchPlan(
                        f"launch {launch.launch_id} grid axis {axis} exceeds cap"
                    )
            for axis, (actual, maximum) in enumerate(zip(launch.grid.block, caps.max_block)):
                if actual > maximum:
                    raise InvalidLaunchPlan(
                        f"launch {launch.launch_id} block axis {axis} exceeds cap"
                    )
            threads = math.prod(launch.grid.block)
            if threads > caps.max_threads_per_block:
                raise InvalidLaunchPlan(
                    f"launch {launch.launch_id} exceeds threads-per-block cap"
                )
            if launch.scratch_bytes > caps.max_scratch_bytes_per_launch:
                raise InvalidLaunchPlan(f"launch {launch.launch_id} exceeds scratch cap")
            total_scratch += launch.scratch_bytes
        if total_scratch > caps.max_total_scratch_bytes:
            raise InvalidLaunchPlan("launch plan exceeds total scratch cap")

        _validate_acyclic_dependencies(launch_map)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "modules": [module.to_dict() for module in self.modules],
            "buffers": [buffer.to_dict() for buffer in self.buffers],
            "scalars": [scalar.to_dict() for scalar in self.scalars],
            "launches": [launch.to_dict() for launch in self.launches],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LaunchPlan":
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            modules=tuple(HsacoImage.from_dict(v) for v in value.get("modules", ())),
            buffers=tuple(BufferSpec.from_dict(v) for v in value.get("buffers", ())),
            scalars=tuple(ScalarSpec.from_dict(v) for v in value.get("scalars", ())),
            launches=tuple(KernelLaunch.from_dict(v) for v in value.get("launches", ())),
        )


def _valid_name(value: str, label: str) -> None:
    if not _NAME.fullmatch(value):
        raise InvalidLaunchPlan(f"invalid {label}: {value!r}")


def _unique_map(items: tuple[Any, ...], attr: str, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        key = getattr(item, attr)
        if key in result:
            raise InvalidLaunchPlan(f"duplicate {label}: {key}")
        result[key] = item
    return result


def _validate_scalar(dtype: ScalarType, value: bool | int | float) -> None:
    if dtype is ScalarType.BOOL:
        if not isinstance(value, bool):
            raise InvalidLaunchPlan("bool scalar requires a bool value")
        return
    if dtype.value.startswith(("i", "u")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidLaunchPlan(f"{dtype.value} scalar requires an integer")
        bits = int(dtype.value[1:])
        lower = 0 if dtype.value.startswith("u") else -(2 ** (bits - 1))
        upper = 2**bits - 1 if dtype.value.startswith("u") else 2 ** (bits - 1) - 1
        if not lower <= value <= upper:
            raise InvalidLaunchPlan(f"{dtype.value} scalar is out of range")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidLaunchPlan(f"{dtype.value} scalar requires a numeric value")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise InvalidLaunchPlan("floating-point scalar is not representable") from exc
    if not math.isfinite(numeric):
        raise InvalidLaunchPlan("floating-point scalar must be finite")
    maximum = {
        ScalarType.F16: 65504.0,
        ScalarType.BF16: 3.38953139e38,
        ScalarType.F32: 3.40282347e38,
        ScalarType.F64: 1.7976931348623157e308,
    }[dtype]
    if abs(numeric) > maximum:
        raise InvalidLaunchPlan(f"{dtype.value} scalar is out of range")


def _validate_acyclic_dependencies(launches: Mapping[str, KernelLaunch]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(launch_id: str) -> None:
        if launch_id in visiting:
            raise InvalidLaunchPlan("launch dependencies contain a cycle")
        if launch_id in visited:
            return
        visiting.add(launch_id)
        for dependency in launches[launch_id].dependencies:
            visit(dependency)
        visiting.remove(launch_id)
        visited.add(launch_id)

    for launch_id in launches:
        visit(launch_id)
