"""Real-hardware tests for the paired timing protocol and cold-cache timing.

The speed half of every KORE claim rests on the ``kore-paired-v2`` protocol:
the driver emits exactly ``repeat`` raw ``KORE_TIMING_PAIR`` lines in balanced
AB/BA order with self-consistent ratios, the environment validates them, and
``kore.reward.reward._publication_timing_error`` recomputes the whole thing
independently before any speed credit is paid. On CPU that chain is only ever
exercised against hand-written strings, because ``_bench_all`` is monkeypatched
away. Here the lines come from a real timed subprocess on a real card.

Also covered: ``KORE_BENCH_COLD``. The docs claim timing is cold-cache because
the driver flushes L2 between timed iterations, and the roofline speed-of-light
integrity gate is only sound under that assumption. So the flush has to be
*observable* in the measurement, not merely coded.

Run with ``python -m pytest -m gpu -q``.
"""

from __future__ import annotations

import json
import math
import re

import pytest

from kore.config import CONFIG
from kore.env.kore_env import (
    _cold_cache_timing,
    _parse_driver_capabilities,
    _parse_timing_pairs,
    _supports_batch_bench,
)
from kore.reward.reward import _publication_timing_error
from kore.reward.stats import paired_timing_stats, publication_admission_error
from kore.tasks._genops import (
    DRIVER_CAPABILITY_PROTOCOL,
    DRIVER_PROTOCOL_ID,
    PUBLICATION_GUARANTEES,
)

pytestmark = pytest.mark.gpu

_TIMING_PAIR_LINE = re.compile(r"^KORE_TIMING_PAIR:\s*(\{.*\})\s*$", re.MULTILINE)
_MEDIAN_LINE = re.compile(r"median_ms:\s*([-\d.eE]+)")

#: Shape used for the cold-vs-warm comparison. It has to be big enough that HBM
#: traffic dominates the kernel (so an L2 flush changes the time) yet small enough
#: to sit in the last-level cache when NOT flushed - a 64x512 tile is pure launch
#: overhead and shows no cache effect at all.
COLD_CACHE_SHAPE = "validation_1"

#: Repeats to demand from the driver. ``1`` is the degenerate case the balance
#: check must still accept; the others must alternate and stay balanced.
REPEAT_CASES = (1, 2, 3)


def _shape_spec(shape) -> str:
    return ",".join(f"{k}={v}" for k, v in shape.dims.items()) if shape.dims else "default"


@pytest.fixture(scope="session")
def staged_seed(gpu_harness):
    """A staged workdir holding the task's seed kernel, for direct driver runs."""
    return gpu_harness.stage(gpu_harness.task.seed_source)


@pytest.fixture(scope="session")
def driver_capabilities(gpu_harness, staged_seed):
    rc, out, timed_out = gpu_harness.run_driver(
        staged_seed, "--kore-driver-capabilities", timeout=300)
    assert timed_out is False and rc == 0, f"capability probe failed: {out[-800:]}"
    caps = _parse_driver_capabilities(out)
    assert caps, f"driver emitted no parsable capability handshake: {out[-800:]}"
    return caps


@pytest.fixture(scope="session")
def bench_both_output(gpu_harness, staged_seed):
    """One real ``--bench-both`` run of the publication protocol, reused widely."""
    spec = _shape_spec(gpu_harness.shape("minimal"))
    repeat = max(1, int(CONFIG.max_variance_runs))
    rc, out, timed_out = gpu_harness.run_driver(
        staged_seed, "--bench-both", "--shapes", spec,
        "--warmup", str(CONFIG.warmup_iters), "--iters", str(CONFIG.bench_iters),
        "--repeat", str(repeat), timeout=600)
    assert timed_out is False and rc == 0, f"bench-both failed: {out[-800:]}"
    return {"output": out, "repeat": repeat, "spec": spec}


# --------------------------------------------------------------------------- #
# the versioned handshake
# --------------------------------------------------------------------------- #
def test_driver_advertises_the_paired_v2_publication_protocol(
        driver_capabilities) -> None:
    """The environment pays speed credit only for this exact handshake."""
    caps = driver_capabilities
    assert caps["protocol"] == DRIVER_CAPABILITY_PROTOCOL
    assert caps["protocol_id"] == DRIVER_PROTOCOL_ID
    assert caps["performance_eligible"] is True
    missing = {
        name: caps.get(name)
        for name, expected in PUBLICATION_GUARANTEES.items()
        if caps.get(name) is not expected
    }
    assert not missing, f"driver does not guarantee {missing}"
    assert _supports_batch_bench(caps) is True


# --------------------------------------------------------------------------- #
# 3. the paired timing protocol
# --------------------------------------------------------------------------- #
def test_bench_both_emits_exactly_the_requested_pair_count(
        bench_both_output) -> None:
    lines = _TIMING_PAIR_LINE.findall(bench_both_output["output"])
    assert len(lines) == bench_both_output["repeat"], (
        f"driver emitted {len(lines)} KORE_TIMING_PAIR lines for "
        f"--repeat {bench_both_output['repeat']}")


def test_timing_pairs_validate_against_the_environment_parser(
        bench_both_output) -> None:
    """The verifier's own parser accepts the driver's real output.

    ``_parse_timing_pairs`` is the gate: it enforces exact count, ascending pair
    indices, alternating and balanced AB/BA ordering, finite positive times, and
    ``ratio``/``log_speedup`` recomputed from the raw times. Feeding it live
    output is the only way to know the driver and the parser agree.
    """
    pairs, error = _parse_timing_pairs(bench_both_output["output"],
                                       bench_both_output["repeat"])
    assert error is None, f"driver output rejected by the verifier parser: {error}"
    assert len(pairs) == bench_both_output["repeat"]
    assert [pair["pair"] for pair in pairs] == list(range(len(pairs)))


def test_timing_pair_ordering_is_alternating_and_balanced(
        bench_both_output) -> None:
    """Candidate-first and reference-first runs must be interleaved and even.

    Timing A then B is biased: whichever runs second inherits a warmed clock and
    a settled power state. The protocol cancels that by alternating the order and
    keeping the counts within one of each other.
    """
    orders = [json.loads(line)["order"]
              for line in _TIMING_PAIR_LINE.findall(bench_both_output["output"])]
    assert orders, "no pairs to check"
    assert set(orders) <= {"AB", "BA"}, f"unexpected orders {set(orders)}"
    assert all(a != b for a, b in zip(orders, orders[1:])), (
        f"pair order does not alternate: {orders}")
    assert abs(orders.count("AB") - orders.count("BA")) <= 1, (
        f"pair order is not balanced: {orders}")


def test_timing_pair_fields_are_internally_consistent(bench_both_output) -> None:
    """``ratio`` and ``log_speedup`` must be derivable from the raw times.

    A driver that reported a ratio not implied by its own ``candidate_ms`` /
    ``baseline_ms`` would let a speedup be asserted rather than measured.
    """
    payloads = [json.loads(line)
                for line in _TIMING_PAIR_LINE.findall(bench_both_output["output"])]
    assert payloads
    for payload in payloads:
        cand = float(payload["candidate_ms"])
        base = float(payload["baseline_ms"])
        assert math.isfinite(cand) and cand > 0.0
        assert math.isfinite(base) and base > 0.0
        assert float(payload["ratio"]) == pytest.approx(base / cand, rel=1e-9)
        assert float(payload["log_speedup"]) == pytest.approx(
            math.log(base / cand), rel=1e-9, abs=1e-12)
        assert payload["baseline_kind"] is not None, (
            "every pair must record which baseline it was measured against")


@pytest.mark.parametrize("repeat", REPEAT_CASES)
def test_requested_repeat_count_is_honored(gpu_harness, staged_seed,
                                           repeat) -> None:
    """The pair count follows ``--repeat`` rather than a driver-side default.

    ``KoreEnv._bench_all`` asks for exactly ``max_variance_runs`` pairs and
    rejects the whole measurement on any other count, so the driver honoring the
    request is load-bearing for the admission gate.
    """
    spec = _shape_spec(gpu_harness.shape("minimal"))
    rc, out, timed_out = gpu_harness.run_driver(
        staged_seed, "--bench-both", "--shapes", spec,
        "--warmup", "4", "--iters", "8", "--repeat", str(repeat), timeout=600)
    assert timed_out is False and rc == 0, f"bench-both failed: {out[-800:]}"
    pairs, error = _parse_timing_pairs(out, repeat)
    assert error is None, f"repeat={repeat}: {error}"
    assert len(pairs) == repeat


def test_multi_shape_bench_emits_one_verified_block_per_shape(gpu_harness,
                                                             staged_seed) -> None:
    """Each requested shape gets its own pair block and its own late verdict.

    ``_bench_all`` splits on ``SHAPE_BEGIN`` and validates every block
    separately, precisely so an early failing shape cannot hide behind a later
    passing one.
    """
    shapes = [gpu_harness.shape("minimal"), gpu_harness.shape("validation_0")]
    specs = [_shape_spec(shape) for shape in shapes]
    repeat = 2
    rc, out, timed_out = gpu_harness.run_driver(
        staged_seed, "--bench-both", "--shapes", ";".join(specs),
        "--warmup", "4", "--iters", "8", "--repeat", str(repeat), timeout=600)
    assert timed_out is False and rc == 0, f"bench-both failed: {out[-800:]}"

    blocks = out.split("SHAPE_BEGIN")[1:]
    assert len(blocks) == len(specs), (
        f"expected {len(specs)} shape blocks, got {len(blocks)}")
    for spec, block in zip(specs, blocks):
        marker, _, _body = block.partition("\n")
        assert marker.strip() == spec, (
            f"shape block marker {marker.strip()!r} != requested {spec!r}")
        pairs, error = _parse_timing_pairs(block, repeat)
        assert error is None, f"{spec}: {error}"
        assert len(pairs) == repeat
        assert re.search(r"allclose:\s*(True|False)", block), (
            f"{spec}: block carries no post-timing correctness verdict")


def test_publication_recompute_agrees_with_the_driver(seed_observation,
                                                      gpu_harness) -> None:
    """The reward's independent recompute must reproduce the driver's numbers.

    ``_publication_timing_error`` re-derives every ratio, log-speedup, CI and
    classification from the raw samples and refuses to pay speed credit on any
    disagreement. This asserts the two agree on live measurements, and (when the
    node is too noisy for vendor-grade admission) that the demotion to
    ``screening`` is itself justified by the recomputed statistics rather than
    being an unexplained downgrade.
    """
    obs = seed_observation
    assert obs.validation_passed is True, obs.error_text
    name = gpu_harness.shape("minimal").name

    if obs.timing_grade == "screening":
        stats = paired_timing_stats(
            list(obs.candidate_samples_by_shape[name]),
            list(obs.baseline_samples_by_shape[name]),
            noise_floor_pct=float(CONFIG.noise_floor_pct),
            z=float(CONFIG.paired_confidence_z),
        )
        admission = publication_admission_error(
            stats,
            min_pairs=max(2, int(CONFIG.min_variance_runs)),
            candidate_cv_threshold_pct=float(CONFIG.cv_threshold_pct),
            baseline_cv_threshold_pct=float(CONFIG.baseline_cv_threshold_pct),
            paired_ratio_cv_threshold_pct=float(CONFIG.paired_ratio_cv_threshold_pct),
            paired_ci_threshold_pct=float(CONFIG.paired_ci_threshold_pct),
        )
        assert admission is not None, (
            "timing was demoted to screening but the recomputed statistics pass "
            f"every admission gate: {stats}")
        pytest.skip(f"node too noisy for vendor-grade admission: {admission}")

    assert obs.timing_grade == "publication", obs.error_text
    assert obs.timing_protocol == DRIVER_PROTOCOL_ID
    assert obs.timing_protocol_version == DRIVER_CAPABILITY_PROTOCOL
    assert obs.timing_pair_count == max(1, int(CONFIG.max_variance_runs))
    assert _publication_timing_error(obs, CONFIG) is None

    candidate = list(obs.candidate_samples_by_shape[name])
    baseline = list(obs.baseline_samples_by_shape[name])
    assert len(candidate) == obs.timing_pair_count
    assert len(baseline) == obs.timing_pair_count
    stats = paired_timing_stats(
        candidate, baseline,
        noise_floor_pct=float(CONFIG.noise_floor_pct),
        z=float(CONFIG.paired_confidence_z),
    )
    assert obs.paired_ratio_samples_by_shape[name] == pytest.approx(
        stats["paired_ratios"], rel=1e-9)
    assert obs.paired_log_speedup_samples_by_shape[name] == pytest.approx(
        stats["paired_log_speedups"], rel=1e-9, abs=1e-12)
    assert obs.paired_log_ci_by_shape[name] == pytest.approx(
        [stats["log_ci_lo"], stats["log_ci_hi"]], rel=1e-9, abs=1e-12)
    assert obs.timing_classification_by_shape[name] == stats["classification"]
    assert obs.candidate_cv_by_shape[name] == pytest.approx(
        stats["candidate_cv_pct"], rel=1e-9)
    assert obs.baseline_cv_by_shape[name] == pytest.approx(
        stats["baseline_cv_pct"], rel=1e-9)


# --------------------------------------------------------------------------- #
# 5. cold-cache vs warm timing
# --------------------------------------------------------------------------- #
def _bench_median(gpu_harness, workdir, shape, *, cold: bool, impl: str = "candidate",
                  warmup: int = 10, iters: int = 50):
    rc, out, timed_out = gpu_harness.run_driver(
        workdir, "--bench-mode", "--impl", impl,
        "--warmup", str(warmup), "--iters", str(iters), *shape.as_args(),
        timeout=600, KORE_BENCH_COLD="1" if cold else "0")
    assert timed_out is False and rc == 0, f"bench failed: {out[-800:]}"
    medians = _MEDIAN_LINE.findall(out)
    assert medians, f"driver printed no median_ms: {out[-800:]}"
    return float(medians[-1])


def test_cold_cache_timing_is_measurably_slower_than_warm(gpu_harness,
                                                          staged_seed) -> None:
    """``KORE_BENCH_COLD`` must change the number, not just the code path.

    Cold timing is what makes a reported speedup honest (a warm-cache measurement
    can beat the HBM roofline and would look like a superhuman kernel), and it is
    a precondition for the speed-of-light integrity gate. So the L2 flush has to
    show up as real added latency on a memory-bound shape. Both sides take the
    best of several runs: a busy neighbour can only inflate a measurement, so the
    minimum is the least contaminated estimate, and the assertion is on their
    ratio rather than on any absolute latency.
    """
    shape = gpu_harness.shape(COLD_CACHE_SHAPE)
    cold = min(_bench_median(gpu_harness, staged_seed, shape, cold=True)
               for _ in range(3))
    warm = min(_bench_median(gpu_harness, staged_seed, shape, cold=False)
               for _ in range(3))
    assert warm > 0.0 and cold > 0.0
    assert cold > warm * 1.10, (
        f"cold-cache timing ({cold:.4f} ms) is not measurably slower than warm "
        f"({warm:.4f} ms) on shape {shape.dims}: the documented L2 flush between "
        "timed iterations is not affecting the measurement, so cold-cache "
        "provenance and the roofline integrity floor are both unfounded")


def test_cold_cache_provenance_tracks_the_bench_cold_setting(
        gpu_harness, driver_capabilities, monkeypatch) -> None:
    """``cold_cache_verified`` must reflect the environment actually handed out.

    The physics integrity gate reads this bit, so it has to follow the real child
    environment - here the one ``KoreEnv._env`` builds - and not a default
    assumption.
    """
    shape = gpu_harness.shape("minimal")

    monkeypatch.delenv("KORE_BENCH_COLD", raising=False)
    default_env = gpu_harness.child_env()
    assert _cold_cache_timing(default_env, driver_capabilities) is True
    obs = gpu_harness.evaluate(gpu_harness.task.seed_source, [shape])
    assert obs.infra_error is False, f"infra failure: {obs.error_text}"
    assert obs.validation_passed is True, obs.error_text
    assert obs.cold_cache_verified is True, (
        "a default evaluation must record cold-cache provenance")

    monkeypatch.setenv("KORE_BENCH_COLD", "0")
    warm_env = gpu_harness.child_env()
    assert warm_env["KORE_BENCH_COLD"] == "0"
    assert _cold_cache_timing(warm_env, driver_capabilities) is False
    warm_obs = gpu_harness.evaluate(gpu_harness.task.seed_source, [shape])
    assert warm_obs.validation_passed is True, warm_obs.error_text
    assert warm_obs.cold_cache_verified is False, (
        "timing measured with KORE_BENCH_COLD=0 must not claim cold-cache "
        "provenance - the HBM branch of the speed-of-light floor depends on it")


def test_unrecognized_protocol_never_claims_cold_cache(gpu_harness,
                                                       monkeypatch) -> None:
    """Cold-cache provenance also requires the driver that owns the flush.

    The flush lives in the driver's timing helpers, so a driver that does not
    advertise the KORE timing protocol cannot vouch for it however the
    environment is configured.
    """
    monkeypatch.delenv("KORE_BENCH_COLD", raising=False)
    env = gpu_harness.child_env()
    assert _cold_cache_timing(env, {"protocol": 0}) is False
    assert _cold_cache_timing(
        env, {"protocol": DRIVER_CAPABILITY_PROTOCOL - 1}) is False
