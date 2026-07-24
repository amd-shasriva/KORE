from __future__ import annotations

from dataclasses import replace

import pytest

from kore.sandbox.errors import InvalidLaunchPlan
from kore.sandbox.launch_plan import (
    ArgumentKind,
    BufferAccess,
    BufferSpec,
    GridSpec,
    HsacoImage,
    KernelLaunch,
    LaunchArgument,
    LaunchCaps,
    LaunchPlan,
    ScalarSpec,
    ScalarType,
)
from kore.sandbox.models import (
    ExecutionKind,
    IsolationMode,
    IsolationPolicy,
    SandboxRequest,
    TrustLevel,
)


def _module() -> HsacoImage:
    return HsacoImage(
        name="candidate",
        target="gfx950",
        sha256="a" * 64,
        size_bytes=4096,
        symbols=("kernel_main",),
    )


def _launch(
    launch_id: str = "main",
    *,
    symbol: str = "kernel_main",
    dependencies: tuple[str, ...] = (),
    scratch_bytes: int = 0,
) -> KernelLaunch:
    return KernelLaunch(
        launch_id=launch_id,
        module="candidate",
        symbol=symbol,
        arguments=(
            LaunchArgument(ArgumentKind.BUFFER, "input"),
            LaunchArgument(ArgumentKind.BUFFER, "output"),
            LaunchArgument(ArgumentKind.SCALAR, "count"),
        ),
        grid=GridSpec(grid=(16, 1, 1), block=(256, 1, 1)),
        dependencies=dependencies,
        scratch_bytes=scratch_bytes,
    )


def _plan(*launches: KernelLaunch) -> LaunchPlan:
    return LaunchPlan(
        modules=(_module(),),
        buffers=(
            BufferSpec("input", 4096, BufferAccess.READ_ONLY, content_sha256="b" * 64),
            BufferSpec("output", 4096, BufferAccess.WRITE_ONLY),
        ),
        scalars=(ScalarSpec("count", ScalarType.U32, 1024),),
        launches=launches or (_launch(),),
    )


def test_valid_hsaco_launch_plan_round_trips():
    plan = _plan()

    restored = LaunchPlan.from_dict(plan.to_dict())

    assert restored == plan
    assert restored.launches[0].grid.block == (256, 1, 1)


def test_production_request_accepts_only_declarative_launch_controls():
    policy = IsolationPolicy(
        mode=IsolationMode.EXTERNAL_BROKER,
        trust_level=TrustLevel.UNTRUSTED,
        production=True,
        require_signed_verdict=True,
        allow_legacy_python=False,
        approved_broker_id="prod-broker",
    )

    request = SandboxRequest.create(
        task_id="launch-test",
        task_descriptor={"task": "launch-test"},
        source="candidate source",
        policy=policy,
        toolchain_descriptor={"compiler": "amdclang"},
        runtime_descriptor={"target": "gfx950"},
        execution_kind=ExecutionKind.HSACO_LAUNCH_PLAN,
        launch_plan=_plan().to_dict(),
    )

    assert request.execution_kind is ExecutionKind.HSACO_LAUNCH_PLAN
    assert request.argv == ()
    assert request.environment == {}


def test_launch_plan_rejects_undeclared_symbol():
    with pytest.raises(InvalidLaunchPlan, match="undeclared symbol"):
        _plan(_launch(symbol="not_exported"))


def test_launch_plan_rejects_unknown_buffer():
    bad = replace(
        _launch(),
        arguments=(LaunchArgument(ArgumentKind.BUFFER, "missing"),),
    )
    with pytest.raises(InvalidLaunchPlan, match="unknown buffer"):
        _plan(bad)


def test_launch_plan_rejects_dependency_cycle():
    first = _launch("first", dependencies=("second",))
    second = _launch("second", dependencies=("first",))
    with pytest.raises(InvalidLaunchPlan, match="cycle"):
        _plan(first, second)


def test_launch_plan_enforces_grid_scratch_and_launch_caps():
    plan = _plan(_launch(scratch_bytes=128))
    with pytest.raises(InvalidLaunchPlan, match="scratch cap"):
        plan.validate(
            LaunchCaps(
                max_scratch_bytes_per_launch=64,
                max_total_scratch_bytes=64,
            )
        )

    too_wide = replace(_launch(), grid=GridSpec(grid=(1, 1, 1), block=(1024, 2, 1)))
    with pytest.raises(InvalidLaunchPlan, match="threads-per-block"):
        _plan(too_wide)

    two_launches = _plan(_launch("one"), _launch("two"))
    with pytest.raises(InvalidLaunchPlan, match="launch count"):
        two_launches.validate(LaunchCaps(max_launches=1))


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (ScalarType.U8, -1),
        (ScalarType.I8, 128),
        (ScalarType.BOOL, 1),
        (ScalarType.F32, float("nan")),
    ],
)
def test_launch_plan_rejects_invalid_scalars(dtype, value):
    with pytest.raises(InvalidLaunchPlan):
        ScalarSpec("bad", dtype, value)
