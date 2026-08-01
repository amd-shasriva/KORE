"""CPU regressions for the PRODUCTION wiring of the correctness oracle.

``kore/verify`` describes a four-prong oracle, but for a long time only three of
them ran in production: the driver's reseeded random trials, its enumerated
adversarial battery, and the environment's determinism re-check. These tests pin
down the fourth - the metamorphic prong - now that
:class:`kore.env.kore_env.KoreEnv` runs it, and pin down the properties that make
it safe to run on every RL candidate:

* it is planned ONLY where the operator contract proves the relation set, so an
  honest kernel in an unproven family is never judged by it;
* a prong that cannot run is never counted as a pass (fail-closed);
* its gate is honest, and a run whose gate the replay contract cannot describe
  is never cached;
* the false-accept bound is reported from a known comparison count or not at all;
* wiring it in does not regress the noisy-timing -> ``screening`` demotion.

No GPU: every subprocess boundary is stubbed, and the runner itself is driven on
CPU tensors (the hardware behaviour is demonstrated separately).
"""

from __future__ import annotations

import importlib.util
import json
import math
import types
from pathlib import Path

import pytest

from kore.env.kore_env import KoreEnv, _metamorphic_gate
from kore.tasks.base import Shape, Task
from kore.verify.metamorphic import metamorphic_relations
from kore.verify.production import (
    DEFAULT_MAX_ELEMENTS,
    GENOPS_METAMORPHIC_OP_CLASS,
    METAMORPHIC_DTYPES,
    RUNNER_SHIM_NAME,
    OracleReport,
    ProngStatus,
    build_oracle_report,
    expected_output_elements,
    format_metamorphic_report,
    metamorphic_plan_for_task,
    parse_metamorphic_report,
    sanitize_detail,
    select_metamorphic_shape,
    task_output_op_class,
)

TASKS_ROOT = Path(__file__).resolve().parents[1] / "kore" / "tasks"


# =========================================================================== #
# 1. Planning: which families may be judged by a metamorphic identity
# =========================================================================== #
@pytest.mark.parametrize(
    "task_id,op_class",
    [
        ("gen_relu_fp32", "elementwise"),      # unary
        ("gen_add_fp32", "elementwise"),       # binary
        ("gen_silu_mul_fp16", "elementwise"),  # 2-operand fusion
        ("gen_fma_bf16", "elementwise"),       # 3-operand fusion
        ("gen_row_sum_bf16", "reduction"),
        ("gen_row_max_fp32", "reduction"),
    ],
)
def test_genops_pointwise_and_reduction_families_are_planned(task_id, op_class):
    plan = metamorphic_plan_for_task(Task.from_dir(TASKS_ROOT / task_id))
    assert plan.applicable, plan.reason
    assert plan.op_class == op_class
    assert plan.relations == tuple(r.name for r in metamorphic_relations(op_class))


@pytest.mark.parametrize(
    "task_id,marker",
    [
        # A K-contraction satisfies none of the generic relations: permuting every
        # operand along one axis is simply not an identity of matmul.
        ("gen_gemm_bias_relu_bf16", "contraction"),
        # Hand-authored: the operator contract is not fixed by a generator spec.
        ("softmax_bf16", "hand-authored"),
        ("gelu_tanh_bf16", "hand-authored"),
        # Breadth/vendor generators emit arbitrary layouts (attention, SSM, ...).
        ("genb_ssm_lru_bf16", "genops"),
        ("genv_rmsnorm_bf16", "genops"),
    ],
)
def test_unproven_families_are_not_planned(task_id, marker):
    plan = metamorphic_plan_for_task(Task.from_dir(TASKS_ROOT / task_id))
    assert not plan.applicable
    assert marker in plan.reason
    assert plan.op_class == "generic"


def test_every_registry_task_classifies_without_raising():
    """The planner runs on every candidate, so it must never raise or guess."""
    checked = applicable = 0
    for task_dir in sorted(TASKS_ROOT.glob("*")):
        if not (task_dir / "task.yaml").is_file():
            continue
        try:
            task = Task.from_dir(task_dir)
        except ValueError:  # not a loadable task directory
            continue
        checked += 1
        plan = metamorphic_plan_for_task(task)
        assert plan.reason, task.task_id
        if plan.applicable:
            applicable += 1
            # Applicability implies EVERY precondition, not just the family.
            assert task.task_id.startswith("gen_")
            assert task.raw.get("op_family") in GENOPS_METAMORPHIC_OP_CLASS
            assert task.dtype in METAMORPHIC_DTYPES
            assert plan.op_class in ("elementwise", "reduction")
        else:
            assert plan.op_class == "generic"
    assert checked > 1000, "expected the full committed task registry"
    assert applicable > 100, "the pointwise/reduction generated families"


def _synthetic_task(tmp_path: Path, **overrides) -> Task:
    task_dir = tmp_path / overrides.get("task_id", "gen_relu_fp32")
    task_dir.mkdir(parents=True, exist_ok=True)
    for name in ("task.yaml", "reference.py", "driver.py", "seed_triton.py"):
        (task_dir / name).write_text("# stub\n")
    fields = {
        "task_id": "gen_relu_fp32",
        "operation": "relu",
        "dtype": "fp32",
        "backend": "triton",
        "gpu_target": "gfx950",
        "dir": task_dir,
        "seed_kernel_name": "seed_triton.py",
        "snr_threshold": 40.0,
        "comparison_baseline": "torch_relu",
        "shapes": [Shape("minimal", {"M": 64, "N": 512}),
                   Shape("primary", {"M": 4096, "N": 8192})],
        "raw": {"generated": True, "op_family": "unary"},
    }
    fields.update(overrides)
    return Task(**fields)


def test_plan_is_fail_closed_on_every_missing_precondition(tmp_path):
    assert metamorphic_plan_for_task(_synthetic_task(tmp_path)).applicable

    # not generator-emitted -> the operator contract is not fixed
    plan = metamorphic_plan_for_task(
        _synthetic_task(tmp_path, raw={"op_family": "unary"}))
    assert not plan.applicable and "hand-authored" in plan.reason

    # unknown source family
    plan = metamorphic_plan_for_task(
        _synthetic_task(tmp_path, raw={"generated": True, "op_family": "sorcery"}))
    assert not plan.applicable and "no proven metamorphic identity" in plan.reason

    # no declared family at all
    plan = metamorphic_plan_for_task(
        _synthetic_task(tmp_path, raw={"generated": True}))
    assert not plan.applicable and "no op_family" in plan.reason

    # quantized storage: the plain float relations do not respect scale structure
    for dtype in ("fp8", "int8", "mxfp4"):
        plan = metamorphic_plan_for_task(_synthetic_task(tmp_path, dtype=dtype))
        assert not plan.applicable and "tolerances" in plan.reason

    # wrong generator prefix (a minted/breadth id with genops-looking metadata)
    plan = metamorphic_plan_for_task(
        _synthetic_task(tmp_path, task_id="genb_relu_fp32"))
    assert not plan.applicable and "genops" in plan.reason

    # metadata that is not a mapping
    plan = metamorphic_plan_for_task(_synthetic_task(tmp_path, raw=None))
    assert not plan.applicable and "metadata" in plan.reason


def test_plan_defers_to_the_taxonomy_authority(tmp_path, monkeypatch):
    """If the family authority disagrees with the generator contract, do not run."""
    import kore.tasks.taxonomy as taxonomy

    task = _synthetic_task(tmp_path)
    monkeypatch.setattr(taxonomy, "product_family_for_task",
                        lambda *a, **k: "attention")
    plan = metamorphic_plan_for_task(task)
    assert not plan.applicable and "does not match" in plan.reason

    def _boom(*_a, **_k):
        raise taxonomy.TaxonomyError("unclassifiable")

    monkeypatch.setattr(taxonomy, "product_family_for_task", _boom)
    plan = metamorphic_plan_for_task(task)
    assert not plan.applicable and "could not classify" in plan.reason


# =========================================================================== #
# 2. Shape selection: cheapest usable evidence, explicitly bounded
# =========================================================================== #
def test_shape_selection_prefers_the_cheapest_declared_shape():
    shapes = [Shape("primary", {"M": 4096, "N": 8192}),
              Shape("minimal", {"M": 64, "N": 512})]
    used, declared, note = select_metamorphic_shape(shapes)
    assert used == declared == {"M": 64, "N": 512}
    assert "declared" in note


def test_shape_selection_caps_rows_and_says_so():
    shapes = [Shape("huge", {"M": 8192, "N": 4096})]
    used, declared, note = select_metamorphic_shape(shapes, max_elements=1 << 20)
    assert declared == {"M": 8192, "N": 4096}
    assert used["N"] == 4096 and used["M"] == (1 << 20) // 4096
    assert "row-capped" in note


def test_shape_selection_rejects_shapes_the_relations_cannot_use():
    for shapes in ([Shape("s", {"M": 1, "N": 512})],        # cannot split rows
                   [Shape("s", {"M": 64, "N": 1})],          # cannot permute cols
                   [Shape("s", {"M": 8, "N": 8, "K": 8})],   # not a 2-D [M, N]
                   []):
        used, declared, note = select_metamorphic_shape(shapes)
        assert used is None and declared is None and "no requested shape" in note


# =========================================================================== #
# 3. The false-accept bound is reported, and only when it is exact
# =========================================================================== #
def test_bound_matches_the_closed_form_for_a_known_comparison_count():
    report = build_oracle_report(
        task_id="gen_relu_fp32", verified=True, prongs=(),
        op_class="elementwise", shape_dims=[{"M": 64, "N": 512}],
        trials_per_shape=5, defect_fraction=1e-4)
    assert report.random_elements == 5 * 64 * 512
    assert report.false_accept_bound == pytest.approx(
        (1 - 1e-4) ** report.random_elements, rel=1e-9)
    assert report.false_accept_bound_log10 == pytest.approx(
        math.log10(report.false_accept_bound), rel=1e-9)
    assert "1-p" in report.bound_basis and "LUCKY RANDOM MISSES" in report.bound_basis


def test_reduction_bound_counts_output_rows_not_input_elements():
    """A row reduction compares M values per trial; the bound must say so."""
    report = build_oracle_report(
        task_id="gen_row_sum_fp32", verified=True, prongs=(),
        op_class="reduction", shape_dims=[{"M": 64, "N": 512}],
        trials_per_shape=5)
    assert report.random_elements == 5 * 64
    # ... which is a far weaker statistical guarantee than the elementwise case,
    # and the whole point of publishing the number.
    assert report.false_accept_bound > 0.9


def test_bound_is_withheld_rather_than_invented_when_the_count_is_unknown():
    report = build_oracle_report(
        task_id="flash_attn_prefill_bf16", verified=True, prongs=(),
        op_class="generic", shape_dims=[{"B": 4, "H": 8}], trials_per_shape=5)
    assert report.random_elements is None
    assert report.false_accept_bound is None
    assert report.false_accept_bound_log10 is None
    assert "no number is invented" in report.bound_basis
    assert "n/a" in report.summary()


def test_underflowing_bound_still_reports_an_exponent():
    report = build_oracle_report(
        task_id="gen_relu_fp32", verified=True, prongs=(),
        op_class="elementwise", shape_dims=[{"M": 4096, "N": 8192}],
        trials_per_shape=5)
    assert report.false_accept_bound == 0.0  # underflows float64
    assert report.false_accept_bound_log10 < -1000
    assert "10^" in report.summary()


@pytest.mark.parametrize(
    "task_id,op_class",
    [("gen_relu_fp32", "elementwise"), ("gen_row_sum_bf16", "reduction"),
     ("gen_gemm_bias_relu_bf16", "elementwise"), ("softmax_bf16", "generic")],
)
def test_output_extent_is_known_wherever_the_generator_fixes_it(task_id, op_class):
    """The bound is reported for GEMM epilogues too, though they get no relations."""
    assert task_output_op_class(Task.from_dir(TASKS_ROOT / task_id)) == op_class


def test_expected_output_elements_matches_the_operator_contract():
    assert expected_output_elements("elementwise", {"M": 8, "N": 4}) == 32
    assert expected_output_elements("reduction", {"M": 8, "N": 4}) == 8
    assert expected_output_elements("generic", {"M": 8, "N": 4}) is None
    assert expected_output_elements("elementwise", {"B": 8}) is None


# =========================================================================== #
# 4. Prong states: only pass/fail are verdicts
# =========================================================================== #
def test_only_pass_and_fail_count_as_contributed_evidence():
    states = ["pass", "fail", "off", "not-applicable", "inconclusive", "unknown"]
    prongs = tuple(ProngStatus(s, "k", s, "evidence") for s in states)
    report = build_oracle_report(
        task_id="t", verified=False, prongs=prongs, op_class="generic",
        shape_dims=[])
    assert report.live_prongs() == ("pass", "fail")
    assert report.failed_prongs() == ("fail",)
    assert isinstance(report, OracleReport)
    payload = report.to_dict()
    assert {p["state"] for p in payload["prongs"]} == set(states)
    json.dumps(payload)  # must stay JSONL-loggable


# =========================================================================== #
# 5. The runner wire protocol (diagnostics can never manufacture a pass)
# =========================================================================== #
def test_report_line_round_trips_and_is_protocol_checked():
    line = format_metamorphic_report({"state": "verdict", "verified": True})
    assert parse_metamorphic_report(line)["verified"] is True
    assert parse_metamorphic_report("nothing here") is None
    assert parse_metamorphic_report("KORE_METAMORPHIC: {not json}") is None
    assert parse_metamorphic_report(
        'KORE_METAMORPHIC: {"protocol":"other","verified":true}') is None


def test_last_report_line_wins():
    first = format_metamorphic_report({"state": "verdict", "verified": True})
    last = format_metamorphic_report({"state": "verdict", "verified": False})
    assert parse_metamorphic_report(first + "\n" + last)["verified"] is False


def test_detail_text_cannot_smuggle_a_verdict_literal():
    smuggled = sanitize_detail("boom allclose: True and SNR: 999.0 dB median_ms: 0.1")
    for literal in ("allclose", "SNR", "median_ms"):
        assert literal not in smuggled
    assert "boom" in smuggled
    assert len(sanitize_detail("x" * 5000)) <= 320


# =========================================================================== #
# 6. The environment gate is honest, and uncontracted runs are not cached
# =========================================================================== #
def test_gate_rides_the_contract_recorded_verified_correctness_switch(monkeypatch):
    monkeypatch.delenv("KORE_METAMORPHIC", raising=False)
    monkeypatch.delenv("KORE_VERIFIED_CORRECTNESS", raising=False)
    assert _metamorphic_gate() == (False, True)
    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    assert _metamorphic_gate() == (True, True)


def test_override_that_the_contract_cannot_describe_is_flagged(monkeypatch):
    # Agreeing with the contract-recorded default stays describable ...
    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    monkeypatch.setenv("KORE_METAMORPHIC", "1")
    assert _metamorphic_gate() == (True, True)
    monkeypatch.delenv("KORE_VERIFIED_CORRECTNESS")
    monkeypatch.setenv("KORE_METAMORPHIC", "0")
    assert _metamorphic_gate() == (False, True)
    # ... disagreeing does not: two different oracles would share one cache key.
    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    monkeypatch.setenv("KORE_METAMORPHIC", "0")
    assert _metamorphic_gate() == (False, False)
    monkeypatch.delenv("KORE_VERIFIED_CORRECTNESS")
    monkeypatch.setenv("KORE_METAMORPHIC", "on")
    assert _metamorphic_gate() == (True, False)


# =========================================================================== #
# 7. End-to-end wiring inside KoreEnv._run (subprocess boundary stubbed)
# =========================================================================== #
_SOURCE = "def relu(x):\n    return x\n"
_DRIVER_OK = "SNR: 80.0 dB\nallclose: True\nmax_diff: 0.0\n"
_MODEL_FINGERPRINT = None


def _config(tmp_path: Path, **overrides):
    cfg = types.SimpleNamespace(
        runs_dir=tmp_path / "runs",
        gpu_target="gfx950",
        rocm_path=str(tmp_path / "missing-rocm"),
        shape_augment=False,
        shape_augment_max=6,
        snr_threshold_for=lambda _dtype: 40.0,
        atol=1e-2, rtol=1e-2,
        verifier_determinism_check=False,
        determinism_snr_tol_db=10.0,
        warmup_iters=10, bench_iters=30,
        min_variance_runs=3, max_variance_runs=5,
        cv_threshold_pct=3.0, baseline_cv_threshold_pct=3.0,
        paired_ratio_cv_threshold_pct=3.0, paired_ci_threshold_pct=3.0,
        paired_confidence_z=1.96, noise_floor_pct=2.0,
        profile_reward_weight=0.0,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def _metamorphic_out(passed: bool, extra: str = "") -> str:
    payload = format_metamorphic_report({
        "state": "verdict", "verified": passed, "op_class": "elementwise",
        "shape_used": "M=64,N=512", "output_elements": 64 * 512,
        "failures": [] if passed else ["elem_locality: per-element rel-err 1e+00"],
    })
    return f"{payload}\n{extra}SNR: 90.00 dB\nallclose: {passed}\n"


def _env_and_task(tmp_path: Path, **cfg_overrides):
    task = _synthetic_task(tmp_path)
    env = KoreEnv(task, config=_config(tmp_path, **cfg_overrides),
                  use_replay=False, gpu="0")
    return env, task


def _stub(env: KoreEnv, metamorphic_result):
    """Route driver execs to a passing verdict and the shim to ``metamorphic_result``."""
    seen: list[list[str]] = []

    def fake_exec(cmd, workdir, environ, timeout):
        parts = [str(p) for p in cmd]
        seen.append(parts)
        if any(RUNNER_SHIM_NAME in p for p in parts):
            return metamorphic_result
        return 0, _DRIVER_OK, False

    env._exec = fake_exec
    env._env = lambda *a, **k: {}
    return seen


def _run(env, task, tmp_path, do_bench=False):
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    return env._run(task, _SOURCE, list(task.shapes), workdir, do_bench)


def test_metamorphic_pass_keeps_the_verdict_and_reports_four_live_prongs(
        tmp_path, monkeypatch):
    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    env, task = _env_and_task(tmp_path, verifier_determinism_check=True)
    seen = _stub(env, (0, _metamorphic_out(True), False))

    obs = _run(env, task, tmp_path)

    assert obs.validation_passed and not obs.infra_error
    report = env.last_oracle_report
    assert set(report.live_prongs()) == {
        "random", "adversarial", "metamorphic", "determinism"}
    assert report.prong("metamorphic").state == "pass"
    assert report.verified is True
    assert report.false_accept_bound is not None
    assert any(RUNNER_SHIM_NAME in " ".join(c) for c in seen)
    assert (tmp_path / "work" / RUNNER_SHIM_NAME).is_file()


def test_metamorphic_violation_rejects_a_kernel_the_other_prongs_accepted(
        tmp_path, monkeypatch):
    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    env, task = _env_and_task(tmp_path)
    _stub(env, (0, _metamorphic_out(False), False))

    obs = _run(env, task, tmp_path)

    # The random/adversarial prongs passed; only the structural identity failed.
    assert obs.validation_passed is False
    assert not obs.infra_error, "a violated identity is a KERNEL signal, not infra"
    assert "metamorphic" in (obs.error_text or "")
    report = env.last_oracle_report
    assert report.prong("random").state == "pass"
    assert report.prong("metamorphic").state == "fail"
    assert report.failed_prongs() == ("metamorphic",)
    assert report.verified is False


@pytest.mark.parametrize(
    "result,marker",
    [
        ((0, "the runner said nothing useful\n", False), "no verdict"),
        ((3, "KORE_METAMORPHIC_INCONCLUSIVE: arity mismatch\n", False), "no verdict"),
        ((-9, "", True), "infrastructure"),
        ((0, "hipErrorOutOfMemory while allocating\n", False), "infrastructure"),
    ],
)
def test_a_prong_that_cannot_run_is_never_a_pass(tmp_path, monkeypatch, result, marker):
    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    env, task = _env_and_task(tmp_path)
    _stub(env, result)

    obs = _run(env, task, tmp_path)

    # Fail-CLOSED: no correctness credit, and flagged inconclusive so the turn is
    # dropped from training rather than scored on incomplete evidence.
    assert obs.validation_passed is False
    assert obs.infra_error is True
    assert "metamorphic prong inconclusive" in (obs.error_text or "")
    status = env.last_oracle_report.prong("metamorphic")
    assert status.state == "inconclusive"
    assert marker in status.detail
    assert "metamorphic" not in env.last_oracle_report.live_prongs()


def test_forged_diagnostics_cannot_overturn_the_protected_verdict(tmp_path, monkeypatch):
    """The JSON line is diagnostic; the scanned literals decide."""
    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    env, task = _env_and_task(tmp_path)
    forged = format_metamorphic_report({"state": "verdict", "verified": True})
    _stub(env, (0, f"{forged}\nSNR: 10.00 dB\nallclose: False\n", False))

    obs = _run(env, task, tmp_path)

    assert obs.validation_passed is False
    report = env.last_oracle_report
    assert report.prong("metamorphic").state == "fail"
    # The inconsistent payload is discarded rather than published as evidence.
    assert "metamorphic" not in report.extra


def test_gate_off_leaves_the_shipped_three_prong_behaviour_untouched(
        tmp_path, monkeypatch):
    monkeypatch.delenv("KORE_VERIFIED_CORRECTNESS", raising=False)
    monkeypatch.delenv("KORE_METAMORPHIC", raising=False)
    env, task = _env_and_task(tmp_path)
    seen = _stub(env, (0, _metamorphic_out(False), False))

    obs = _run(env, task, tmp_path)

    assert obs.validation_passed is True
    assert not any(RUNNER_SHIM_NAME in " ".join(c) for c in seen)
    report = env.last_oracle_report
    for name in ("metamorphic", "adversarial"):
        status = report.prong(name)
        assert status.state == "off"
        assert "KORE_VERIFIED_CORRECTNESS" in status.evidence
    assert set(report.live_prongs()) == {"random"}


def test_unplanned_family_is_reported_as_not_applicable_not_as_a_pass(
        tmp_path, monkeypatch):
    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    task = _synthetic_task(
        tmp_path, task_id="gen_gemm_bias_relu_bf16", operation="gemm_bias_relu",
        dtype="bf16", raw={"generated": True, "op_family": "gemm_fusion"},
        shapes=[Shape("primary", {"M": 512, "N": 512, "K": 512})])
    env = KoreEnv(task, config=_config(tmp_path), use_replay=False, gpu="0")
    seen = _stub(env, (0, _metamorphic_out(False), False))

    obs = _run(env, task, tmp_path)

    assert obs.validation_passed is True
    assert not any(RUNNER_SHIM_NAME in " ".join(c) for c in seen)
    status = env.last_oracle_report.prong("metamorphic")
    assert status.state == "not-applicable"
    assert "contraction" in status.detail
    assert "metamorphic" not in env.last_oracle_report.live_prongs()


def test_noisy_timing_still_demotes_to_screening_with_the_prong_live(
        tmp_path, monkeypatch):
    """The prong must not regress the correct-but-unmeasurable -> screening path."""
    from kore.tasks._genops import (
        DRIVER_CAPABILITY_PROTOCOL,
        DRIVER_PROTOCOL_ID,
        PUBLICATION_GUARANTEES,
    )

    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    env, task = _env_and_task(tmp_path)
    _stub(env, (0, _metamorphic_out(True), False))
    env._driver_caps_cache = {
        "protocol": DRIVER_CAPABILITY_PROTOCOL,
        "protocol_id": DRIVER_PROTOCOL_ID,
        "performance_eligible": True,
        **PUBLICATION_GUARANTEES,
    }
    noisy = [
        {"pair": i, "order": "AB" if i % 2 == 0 else "BA",
         "candidate_ms": c, "baseline_ms": 2.0, "ratio": 2.0 / c}
        for i, c in enumerate([1.0, 1.6, 0.6, 1.4, 0.8])
    ]
    env._bench_all = lambda *a, **k: ({sh.name: list(noisy) for sh in task.shapes},
                                      False)

    obs = _run(env, task, tmp_path, do_bench=True)

    assert obs.validation_passed is True
    assert obs.infra_error is False
    assert obs.timing_grade == "screening"
    assert env.last_oracle_report.prong("metamorphic").state == "pass"


def test_uncontracted_gate_override_disables_the_replay_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    monkeypatch.setenv("KORE_METAMORPHIC", "0")
    task = _synthetic_task(tmp_path)
    env = KoreEnv(task, config=_config(tmp_path), use_replay=True, gpu="0")
    reads, writes = [], []
    env._cache_obj = types.SimpleNamespace(
        get=lambda *a, **k: reads.append(a) or None,
        put=lambda *a, **k: writes.append(a),
    )
    env._run = lambda *a, **k: __import__(
        "kore.reward.reward", fromlist=["Observation"]).Observation(
            compiled=True, validation_passed=True, dtype="fp32",
            snr_by_shape={s.name: 90.0 for s in task.shapes},
            requested_shapes=[s.name for s in task.shapes])

    env.evaluate(task, _SOURCE, shapes=list(task.shapes), do_bench=False)

    assert reads == [] and writes == []


def test_report_is_not_carried_over_from_a_previous_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("KORE_VERIFIED_CORRECTNESS", "1")
    env, task = _env_and_task(tmp_path)
    _stub(env, (0, _metamorphic_out(True), False))
    env.evaluate(task, _SOURCE, shapes=list(task.shapes), do_bench=False)
    assert env.last_oracle_report is not None
    env._run = lambda *a, **k: __import__(
        "kore.reward.reward", fromlist=["Observation"]).Observation(compiled=False)
    env.evaluate(task, _SOURCE + "\n# changed\n", shapes=list(task.shapes),
                 do_bench=False)
    assert env.last_oracle_report is None


# =========================================================================== #
# 8. The runner itself, driven on CPU tensors
# =========================================================================== #
_REFERENCE_SRC = '''
import torch


def parse_shape(spec):
    out = {}
    for kv in spec.split(","):
        key, _, value = kv.partition("=")
        out[key.strip()] = int(value)
    return out


def get_inputs(shape, device="cuda", seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn((shape["M"], shape["N"]), generator=g,
                        dtype=torch.float32),)


def ref_fn(x):
    return torch.nn.functional.silu(x)


arity = __ARITY__
entry_name = "silu"
family = "unary"
dtype_name = "fp32"
mutates_input = False
'''

_HONEST = "import torch\n\n\ndef silu(x):\n    return x * torch.sigmoid(x)\n"
# Elementwise in value but not in structure: each row leaks 1% of the row above,
# which no random VALUE check can distinguish from silu at this magnitude.
_ROW_LEAK = (
    "import torch\n\n\n"
    "def silu(x):\n"
    "    z = x + 0.01 * torch.roll(x, 1, 0)\n"
    "    return z * torch.sigmoid(z)\n"
)
_CRASH = "def silu(x):\n    raise RuntimeError('kernel launch failed')\n"


def _drive_runner(tmp_path: Path, kernel_src: str, *, arity: int = 1,
                  op_class: str = "elementwise", shape: str = "M=16,N=8"):
    pytest.importorskip("torch")
    from kore.verify.runner import runner_main

    (tmp_path / "reference.py").write_text(
        _REFERENCE_SRC.replace("__ARITY__", str(arity)))
    (tmp_path / "kernel.py").write_text(kernel_src)
    spec = importlib.util.spec_from_file_location(
        f"ref_{tmp_path.name}", tmp_path / "reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    rc = runner_main(ref, str(tmp_path), argv=[
        "--shape", shape, "--op-class", op_class,
        "--source-family", "unary", "--dtype", "fp32"])
    return rc


def test_runner_accepts_a_genuinely_elementwise_kernel(tmp_path, capsys):
    rc = _drive_runner(tmp_path, _HONEST)
    out = capsys.readouterr().out
    assert rc == 0
    assert "allclose: True" in out
    payload = parse_metamorphic_report(out)
    assert payload["verified"] is True and payload["failures"] == []
    assert payload["output_elements"] == 16 * 8
    assert payload["candidate_calls"] >= 8  # two evaluations per relation


def test_runner_rejects_a_kernel_that_is_not_a_function_of_its_own_element(
        tmp_path, capsys):
    rc = _drive_runner(tmp_path, _ROW_LEAK)
    out = capsys.readouterr().out
    assert rc == 0
    assert "allclose: False" in out
    payload = parse_metamorphic_report(out)
    assert payload["verified"] is False
    assert payload["failures"], "the violated relation must be named"


def test_runner_turns_a_crashing_candidate_into_a_verdict_not_a_skip(
        tmp_path, capsys):
    rc = _drive_runner(tmp_path, _CRASH)
    out = capsys.readouterr().out
    assert rc == 0 and "allclose: False" in out
    assert "RuntimeError" in parse_metamorphic_report(out)["failures"][0]


def test_runner_refuses_to_judge_when_the_reference_contradicts_the_plan(
        tmp_path, capsys):
    """Metadata drift must produce NO verdict, not an unproven one."""
    rc = _drive_runner(tmp_path, _HONEST, arity=2)
    out = capsys.readouterr().out
    assert rc == 3
    assert "allclose:" not in out, "an unrunnable prong must publish no verdict"
    assert parse_metamorphic_report(out)["state"] == "inconclusive"


def test_runner_refuses_an_op_class_with_no_proven_relations(tmp_path, capsys):
    rc = _drive_runner(tmp_path, _HONEST, op_class="generic")
    out = capsys.readouterr().out
    assert rc == 3 and "allclose:" not in out
    assert "no metamorphic relations" in parse_metamorphic_report(out)["reason"]


def test_runner_reduction_relations_hold_for_a_true_row_reduction(tmp_path, capsys):
    rc = _drive_runner(
        tmp_path, "def silu(x):\n    return x.sum(-1)\n", op_class="reduction")
    out = capsys.readouterr().out
    assert rc == 0 and "allclose: True" in out


def test_runner_reduction_relations_reject_a_row_dependent_reduction(
        tmp_path, capsys):
    rc = _drive_runner(
        tmp_path,
        "import torch\n\n\ndef silu(x):\n"
        "    return x.sum(-1) + 0.05 * torch.roll(x.sum(-1), 1, 0)\n",
        op_class="reduction")
    out = capsys.readouterr().out
    assert rc == 0 and "allclose: False" in out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
