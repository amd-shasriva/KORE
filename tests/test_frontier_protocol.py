"""Deterministic CPU-only tests for the frontier claim protocol."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from kore.eval.frontier_protocol import (
    CLAIM_PROFILES,
    PROTOCOL_SCHEMA_VERSION,
    ArmKind,
    ArmObservation,
    ArmSpec,
    ArtifactSignature,
    BaselineKind,
    BaselineObservation,
    BudgetEnvelope,
    DataQuality,
    Disclosure,
    EvidenceRef,
    Fingerprint,
    FrontierArtifact,
    HarnessTrack,
    PairedDatum,
    PairedTaskSample,
    Preregistration,
    ResourceUsage,
    TaskManifestEntry,
    TrackSpec,
    artifact_payload_sha256,
    evaluate_claim,
    hierarchical_paired_bootstrap,
    holm_adjust,
    manifest_fingerprint,
    sha256_text,
)


_REVISION = "0123456789abcdef0123456789abcdef01234567"
_RUN_IDS = ("run-0", "run-1")


def _fp(kind: str, name: str) -> Fingerprint:
    return Fingerprint.from_payload(
        kind,
        {"name": name, "revision": _REVISION},
    )


def _evidence(label: str) -> EvidenceRef:
    return EvidenceRef(
        raw_trace_sha256=sha256_text(f"trace:{label}"),
        sample_sha256=sha256_text(f"sample:{label}"),
        source_url=f"https://example.invalid/evidence/{label}",
        source_revision=_REVISION,
    )


def _budget() -> BudgetEnvelope:
    return BudgetEnvelope(
        output_tokens=1_024,
        context_policy=_fp("context-policy", "frozen-context-v1"),
        tool_calls=4,
        correctness_calls=5,
        fresh_timed_calls=7,
        profiler_calls=2,
        gpu_seconds=60.0,
        wall_time_seconds=120.0,
        cost_usd=2.0,
    )


def _usage() -> ResourceUsage:
    return ResourceUsage(
        output_tokens=512,
        tool_calls=2,
        correctness_calls=4,
        fresh_timed_calls=6,
        profiler_calls=1,
        gpu_seconds=30.0,
        wall_time_seconds=70.0,
        cost_usd=1.0,
    )


def _arm(track: HarnessTrack, side: str) -> ArmSpec:
    budget = _budget()
    is_model = track is HarnessTrack.CLOSED_MODEL
    shared = track.value
    tool_name = f"{shared}-tool" if is_model else f"{shared}-{side}-tool"
    prompt_name = f"{shared}-prompt" if is_model else f"{shared}-{side}-prompt"
    return ArmSpec(
        arm_id=f"{track.value}-{side}",
        display_name=f"{track.value} {side}",
        arm_kind=ArmKind.MODEL if is_model else ArmKind.SYSTEM,
        disclosure=Disclosure.CLOSED if is_model else Disclosure.OPEN,
        artifact_fingerprint=_fp("artifact", f"{track.value}-{side}-artifact"),
        checkpoint_fingerprint=_fp("checkpoint", f"{track.value}-{side}-checkpoint"),
        tool_fingerprint=_fp("tool", tool_name),
        prompt_fingerprint=_fp("prompt", prompt_name),
        verifier_fingerprint=_fp("verifier", f"{shared}-verifier"),
        hardware_fingerprint=_fp("hardware", f"{shared}-hardware"),
        budget_fingerprint=budget.fingerprint(),
        budget=budget,
        source_url=f"https://example.invalid/arms/{track.value}/{side}",
        source_revision=_REVISION,
    )


def _track(track: HarnessTrack) -> TrackSpec:
    return TrackSpec(
        track=track,
        candidate=_arm(track, "candidate"),
        comparator=_arm(track, "comparator"),
        run_ids=_RUN_IDS,
        started_at_utc="2026-01-02T00:00:00Z",
    )


def _manifest() -> tuple[TaskManifestEntry, ...]:
    return tuple(
        TaskManifestEntry(
            task_id=f"task-{index}",
            family_id=f"family-{index // 2}",
            source_url=f"https://example.invalid/tasks/task-{index}",
            source_revision=_REVISION,
            task_sha256=sha256_text(f"task-spec:{index}"),
        )
        for index in range(4)
    )


def _observation(
    label: str,
    *,
    correct: bool,
    time_ms: float | None,
) -> ArmObservation:
    return ArmObservation(
        public_correct=correct,
        hidden_correct=correct,
        time_ms=time_ms,
        usage=_usage(),
        evidence=_evidence(label),
    )


def _baselines(label: str) -> tuple[BaselineObservation, ...]:
    times = {
        BaselineKind.VENDOR_PRODUCTION: 2.0,
        BaselineKind.BEST_VENDOR: 1.8,
        BaselineKind.COMPILER: 2.4,
        BaselineKind.EAGER: 3.0,
    }
    return tuple(
        BaselineObservation(
            kind=kind,
            hidden_correct=True,
            time_ms=time_ms,
            evidence=_evidence(f"{label}-{kind.value}"),
        )
        for kind, time_ms in times.items()
    )


def _resign(artifact: FrontierArtifact) -> FrontierArtifact:
    unsigned = replace(artifact, signature=None)
    signature = ArtifactSignature(
        algorithm="ed25519",
        signature="detached-test-signature",
        signer_key_sha256=sha256_text("test-signing-key"),
        payload_sha256=artifact_payload_sha256(unsigned),
        verified=True,
    )
    return replace(unsigned, signature=signature)


def _artifact(
    *,
    candidate_correct=lambda _index: True,
    comparator_time=lambda index: 1.5 if index % 2 == 0 else 2.5,
) -> FrontierArtifact:
    manifest = _manifest()
    tracks = tuple(_track(track) for track in HarnessTrack)
    samples = []
    for track in tracks:
        for run_id in track.run_ids:
            for index, task in enumerate(manifest):
                candidate_ok = bool(candidate_correct(index))
                samples.append(
                    PairedTaskSample(
                        track=track.track,
                        run_id=run_id,
                        task_id=task.task_id,
                        family_id=task.family_id,
                        candidate=_observation(
                            f"{track.track.value}-{run_id}-{task.task_id}-candidate",
                            correct=candidate_ok,
                            time_ms=0.5 if candidate_ok else None,
                        ),
                        comparator=_observation(
                            f"{track.track.value}-{run_id}-{task.task_id}-comparator",
                            correct=True,
                            time_ms=float(comparator_time(index)),
                        ),
                        baselines=_baselines(
                            f"{track.track.value}-{run_id}-{task.task_id}"
                        ),
                        sol_time_ms=0.25,
                        sol_evidence=_evidence(
                            f"{track.track.value}-{run_id}-{task.task_id}-sol"
                        ),
                    )
                )
    profile = CLAIM_PROFILES["development"]
    artifact = FrontierArtifact(
        schema_version=PROTOCOL_SCHEMA_VERSION,
        artifact_id="synthetic-frontier-artifact",
        source_url="https://example.invalid/artifacts/frontier",
        source_revision=_REVISION,
        preregistration=Preregistration(
            profile_name=profile.name,
            profile_fingerprint=profile.fingerprint,
            manifest_fingerprint=manifest_fingerprint(manifest),
            registered_at_utc="2026-01-01T00:00:00Z",
        ),
        manifest=manifest,
        tracks=tracks,
        samples=tuple(samples),
    )
    return _resign(artifact)


def test_profiles_and_fingerprints_are_preregistered_and_immutable():
    assert set(CLAIM_PROFILES) == {
        "development",
        "frontier-competitive",
        "best-in-class-model",
        "best-in-class-system",
    }
    for name, profile in CLAIM_PROFILES.items():
        assert profile.name == name
        assert profile.required_tracks == tuple(HarnessTrack)
        assert profile.n_boot > 0
        assert len(profile.fingerprint) == 64
    assert CLAIM_PROFILES["best-in-class-model"].required_superiority_tracks == (
        HarnessTrack.CLOSED_MODEL,
    )
    assert CLAIM_PROFILES["best-in-class-system"].required_superiority_tracks == (
        HarnessTrack.OPEN_SYSTEM,
    )

    fingerprint = _fp("tool", "immutable")
    with pytest.raises(FrozenInstanceError):
        fingerprint.sha256 = "0" * 64


def test_valid_artifact_reports_two_tracks_full_denominator_and_separate_metrics():
    report = evaluate_claim(_artifact(), seed=17)
    assert report.passed is True
    assert report.errors == []
    assert set(report.tracks) == {track.value for track in HarnessTrack}
    assert report.primary_holm
    assert report.secondary_holm

    for track in report.tracks.values():
        assert track.denominator == len(_manifest()) * len(_RUN_IDS)
        assert track.candidate.canonical_fast_p == 1.0
        assert track.comparator.canonical_fast_p == 0.5
        assert (
            track.candidate.certified_lower_ci_fast_p
            <= track.candidate.canonical_fast_p
        )
        assert set(track.baseline_fast_p) == {kind.value for kind in BaselineKind}
        assert "sol" not in track.baseline_fast_p
        assert (
            track.sol_attainment["candidate"]["interpretation"]
            == "fraction of physical SoL ceiling attained; not a comparator"
        )
        assert track.hidden_correctness["candidate"]["denominator"] == track.denominator
        assert track.secondary_tests["mcnemar"]["holm_p_value"] is not None
        assert set(track.fingerprints["comparator"]) >= {
            "artifact",
            "checkpoint",
            "tool",
            "prompt",
            "verifier",
            "hardware",
            "budget",
        }

    serialized = report.to_dict()
    assert serialized["evidence_manifest"]
    first_evidence = serialized["evidence_manifest"][0]["candidate"]
    assert len(first_evidence["raw_trace_sha256"]) == 64
    assert first_evidence["source_url"].startswith("https://")
    json.dumps(serialized, allow_nan=False)


def test_hierarchical_bootstrap_is_task_family_run_paired_and_deterministic():
    data = [
        PairedDatum(
            run_id=f"run-{run}",
            family_id=f"family-{task // 2}",
            task_id=f"task-{task}",
            candidate=1.0,
            comparator=float(task % 2 == 0),
        )
        for run in range(2)
        for task in range(4)
    ]
    first = hierarchical_paired_bootstrap(
        data,
        n_boot=500,
        ci_level=0.90,
        seed=99,
        noninferiority_margin=0.1,
    )
    second = hierarchical_paired_bootstrap(
        data,
        n_boot=500,
        ci_level=0.90,
        seed=99,
        noninferiority_margin=0.1,
    )
    assert first.to_dict() == second.to_dict()
    assert first.n == 8 and first.n_runs == 2 and first.n_families == 2
    assert first.delta_estimate == 0.5
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


def test_candidate_cannot_win_by_omitting_a_task():
    artifact = _artifact()
    samples = tuple(
        sample
        for sample in artifact.samples
        if not (
            sample.track is HarnessTrack.CLOSED_MODEL
            and sample.run_id == "run-0"
            and sample.task_id == "task-3"
        )
    )
    report = evaluate_claim(_resign(replace(artifact, samples=samples)), seed=7)
    closed = report.tracks[HarnessTrack.CLOSED_MODEL.value]

    assert report.passed is False
    assert any("coverage: missing sample" in error for error in report.errors)
    assert closed.denominator == 8
    assert closed.candidate.successes == 7
    assert closed.candidate.canonical_fast_p == 7 / 8


def test_candidate_cannot_win_by_increasing_any_matched_budget():
    artifact = _artifact()
    model_track = next(
        track for track in artifact.tracks if track.track is HarnessTrack.CLOSED_MODEL
    )
    larger_budget = replace(
        model_track.candidate.budget,
        output_tokens=model_track.candidate.budget.output_tokens + 1,
    )
    larger_candidate = replace(
        model_track.candidate,
        budget=larger_budget,
        budget_fingerprint=larger_budget.fingerprint(),
    )
    changed_track = replace(model_track, candidate=larger_candidate)
    tracks = tuple(
        changed_track if track.track is HarnessTrack.CLOSED_MODEL else track
        for track in artifact.tracks
    )
    report = evaluate_claim(_resign(replace(artifact, tracks=tracks)), seed=8)

    assert report.passed is False
    assert any(
        "candidate/comparator envelopes are not exactly matched" in error
        for error in report.errors
    )


def test_candidate_cannot_win_with_sol_impossible_timing():
    artifact = _artifact()
    samples = list(artifact.samples)
    sample = samples[0]
    impossible = replace(sample.candidate, time_ms=sample.sol_time_ms / 2.0)
    samples[0] = replace(sample, candidate=impossible)
    report = evaluate_claim(
        _resign(replace(artifact, samples=tuple(samples))),
        seed=9,
    )
    track = report.tracks[sample.track.value]

    assert report.passed is False
    assert any("sol-integrity" in error for error in report.errors)
    assert track.candidate.successes == track.denominator - 1
    assert track.sol_attainment["candidate"]["max"] <= 1.0
    assert "sol" not in track.baseline_fast_p


def test_candidate_cannot_condition_claim_on_both_correct_survivors():
    artifact = _artifact(
        candidate_correct=lambda index: index % 2 == 0,
        comparator_time=lambda _index: 1.5,
    )
    report = evaluate_claim(artifact, seed=10)
    for track in report.tracks.values():
        survivor = track.survivor_analysis
        assert survivor["geomean_candidate_over_comparator_speed"] == pytest.approx(3.0)
        assert survivor["gate_eligible"] is False
        assert survivor["excluded_cells"] == 4
        assert track.candidate.canonical_fast_p == 0.5
        assert track.comparator.canonical_fast_p == 1.0
        assert track.gates["noninferiority"]["passed"] is False

    assert report.passed is False
    assert any("survivor-bias" in warning for warning in report.warnings)


@pytest.mark.parametrize(
    ("variant", "expected_error"),
    (
        ("unsigned", "artifact is unsigned"),
        ("smoke", "uses smoke data"),
        ("fallback", "uses fallback data"),
        ("contaminated", "reports contamination"),
        ("skipped", "comparator track was skipped"),
    ),
)
def test_disqualifying_provenance_never_passes(variant: str, expected_error: str):
    artifact = _artifact()
    if variant == "unsigned":
        changed = replace(artifact, signature=None)
    elif variant == "smoke":
        changed = _resign(
            replace(artifact, quality=replace(artifact.quality, smoke_data=True))
        )
    elif variant == "fallback":
        first = replace(
            artifact.tracks[0],
            quality=replace(artifact.tracks[0].quality, fallback_data=True),
        )
        changed = _resign(replace(artifact, tracks=(first,) + artifact.tracks[1:]))
    elif variant == "contaminated":
        changed = _resign(
            replace(
                artifact,
                quality=replace(artifact.quality, contamination_detected=True),
            )
        )
    else:
        skipped = replace(
            artifact.tracks[1],
            skipped=True,
            skip_reason="synthetic outage",
        )
        changed = _resign(replace(artifact, tracks=(artifact.tracks[0], skipped)))

    report = evaluate_claim(changed, seed=11)
    assert report.passed is False
    assert any(expected_error in error for error in report.errors)
    if variant == "skipped":
        assert any("required confidence intervals missing" in error for error in report.errors)


def test_nonfinite_measurement_fails_closed_instead_of_crashing():
    artifact = _artifact()
    samples = list(artifact.samples)
    sample = samples[0]
    baselines = tuple(
        replace(baseline, time_ms=float("nan"))
        if baseline.kind is BaselineKind.COMPILER
        else baseline
        for baseline in sample.baselines
    )
    samples[0] = replace(sample, baselines=baselines)
    report = evaluate_claim(
        _resign(replace(artifact, samples=tuple(samples))),
        seed=12,
    )

    assert report.passed is False
    assert any("finite-input" in error for error in report.errors)
