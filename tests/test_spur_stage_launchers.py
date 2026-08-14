"""SPUR sbatch launcher coverage for every full-FT training stage.

The cluster has no GPUs on the login node, so a stage that ships no
``scripts/spur_*.sbatch`` cannot run here at all - which is exactly the state
``sft``, ``dpo`` and ``grpo`` were in while ``midtrain`` had six launchers.
These tests pin the coverage and the SBATCH directives that make a launcher
survive contact with this scheduler:

* every stage ``scripts/launch_distributed.sh`` accepts has a launcher, and that
  launcher actually drives that stage;
* the resource request matches the verified midtrain precedent (full-node
  8x mi355x GRES, the amd-general account/partition/qos, a time limit inside the
  cluster maximum);
* the job is requeue-safe (``--requeue`` + append output + an explicit
  self-requeue whose success is checked rather than assumed, because SPUR's
  ``scontrol`` exits 0 when it cannot reach the controller);
* the offline/PATH environment the ``exec accelerate`` handoff needs;
* ``KORE_RESOURCE_PREFLIGHT`` is never ``strict`` - no measured peak profile
  exists for these workloads, so strict mode would refuse to start.

Everything here is CPU-only and reads the scripts as text or drives the
resolver directly; nothing submits a job.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
LAUNCH_DISTRIBUTED = SCRIPTS / "launch_distributed.sh"

#: Upper sanity bound on --time, not a cluster ceiling.
#:
#: This was 23*3600, inferred from the midtrain job that produced
#: runs/midtrain_14b_frontier having run at 23:00:00. That was a description of
#: one job, not a limit: `scontrol show partition amd-spur` reports
#: MaxTime=UNLIMITED, a probe at --time=7-00:00:00 was accepted and ran, and the
#: v5 SFT job is running at this moment with limit=3-00:00:00. Keeping 23h here
#: made the suite red for a launcher that works.
#:
#: 8 days rather than no bound at all, so a typo like --time=300-00:00:00 (which
#: would sit unschedulable forever against a QoS cap) still fails.
MAX_TIME_SECONDS = 8 * 24 * 3600

#: Stage -> the launcher that runs it on SPUR. midtrain has several (it also
#: scales to 3/4/8 nodes); the other three are single-node by design.
STAGE_LAUNCHERS = {
    "midtrain": "scripts/spur_midtrain_1node.sbatch",
    "sft": "scripts/spur_sft_1node.sbatch",
    "dpo": "scripts/spur_dpo_1node.sbatch",
    "grpo": "scripts/spur_grpo_1node.sbatch",
}

#: SFT, DPO, and GRPO share the scheduler-launcher mechanics even though only
#: the direct-instruct SFT launcher is currently a production operation.
NEW_STAGE_LAUNCHERS = ("sft", "dpo", "grpo")

def _text(relative: str) -> str:
    return (REPO / relative).read_text()


def _directives(source: str) -> dict[str, str]:
    """``#SBATCH --key=value`` / ``#SBATCH --key`` pairs (flags map to "")."""
    found: dict[str, str] = {}
    for line in source.splitlines():
        line = line.strip()
        if not line.startswith("#SBATCH "):
            continue
        token = line[len("#SBATCH "):].strip()
        key, _, value = token.partition("=")
        found[key.strip()] = value.strip()
    return found


def _seconds(limit: str) -> int:
    """Parse a Slurm ``[days-]HH:MM:SS`` time limit."""
    days, _, clock = limit.partition("-")
    if not clock:
        days, clock = "0", days
    parts = [int(p) for p in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def _stages_launch_distributed_accepts() -> list[str]:
    """The stage names ``launch_distributed.sh`` validates, read from the script.

    Deriving them keeps this test honest: adding a stage to the launcher without
    a matching sbatch file makes the coverage test below fail.
    """
    for line in LAUNCH_DISTRIBUTED.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("midtrain|sft|dpo|grpo)"):
            return stripped.split(")")[0].split("|")
    pytest.fail("could not read the stage list out of launch_distributed.sh")


# --------------------------------------------------------------------------- #
# 1. stage coverage
# --------------------------------------------------------------------------- #
def test_every_launchable_stage_has_an_sbatch_launcher():
    stages = _stages_launch_distributed_accepts()
    assert set(stages) == set(STAGE_LAUNCHERS), (
        "launch_distributed.sh and STAGE_LAUNCHERS disagree about the stage set"
    )
    for stage, relative in STAGE_LAUNCHERS.items():
        assert (REPO / relative).is_file(), (
            f"stage {stage!r} has no SPUR launcher; the login node has no GPUs, "
            f"so without {relative} that stage cannot run on this cluster"
        )


@pytest.mark.parametrize("stage", sorted(STAGE_LAUNCHERS))
def test_each_launcher_drives_its_own_stage(stage):
    source = _text(STAGE_LAUNCHERS[stage])
    assert f"kore.policy.{stage}" in source or f"launch_distributed.sh {stage}" in source, (
        f"{STAGE_LAUNCHERS[stage]} never launches the {stage} stage"
    )
    for other in STAGE_LAUNCHERS:
        if other == stage:
            continue
        assert f"launch_distributed.sh {other}" not in source, (
            f"{STAGE_LAUNCHERS[stage]} also launches {other}"
        )


def test_stage_launchers_distinguish_live_sft_from_legacy_14b_stages():
    registry = json.loads((SCRIPTS / "operations_registry.json").read_text())
    records = {record["path"]: record for record in registry["scripts"]}

    sft = records[STAGE_LAUNCHERS["sft"]]
    assert sft["classification"] == "active"
    assert sft["production"] is True

    for stage in ("dpo", "grpo"):
        record = records[STAGE_LAUNCHERS[stage]]
        assert record["classification"] == "active"
        assert record.get("production") is not True
        assert "14B" in record["role"]


# --------------------------------------------------------------------------- #
# 2. SBATCH directive coherence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stage", sorted(STAGE_LAUNCHERS))
def test_launcher_requests_a_full_node_of_mi355x(stage):
    directives = _directives(_text(STAGE_LAUNCHERS[stage]))
    assert directives.get("--account") == "amd-general"
    assert directives.get("--partition") == "amd-spur"
    assert directives.get("--qos") == "amd-general-qos"
    # Full node: either the GRES spelling or the gpus-per-node one, always 8.
    #
    # The count is what matters, NOT the type. This asserted mi355x:8 and had to
    # be relaxed, because the typed spelling does not dispatch on this
    # controller: it is accepted, a node is picked, and the node never confirms,
    # leaving the job in JobLaunchFailure forever. Measured head to head, three
    # trials each, identical otherwise:
    #
    #   --gres=gpu:mi355x:8   0 of 3 dispatched
    #   --gres=gpu:8          2 of 3 dispatched
    #   --gpus-per-node=8     0 of 3 dispatched
    #
    # Every node in amd-spur is MI355X, so naming the model buys nothing, and the
    # launcher verifies it received eight usable devices at runtime -- which is
    # the check that actually protects the run. Pinning the typed form here made
    # the suite demand a spelling that provably never starts.
    gpus = directives.get("--gres") or directives.get("--gpus-per-node")
    assert gpus is not None, "no GPU request at all"
    assert gpus.endswith(":8") or gpus == "8", gpus
    assert "--exclusive" in directives


@pytest.mark.parametrize("stage", sorted(STAGE_LAUNCHERS))
def test_launcher_time_limit_is_positive_and_within_the_cluster_maximum(stage):
    limit = _directives(_text(STAGE_LAUNCHERS[stage])).get("--time")
    assert limit, "no --time; a job with no limit is not schedulable here"
    assert 0 < _seconds(limit) <= MAX_TIME_SECONDS, limit


@pytest.mark.parametrize("stage", sorted(STAGE_LAUNCHERS))
def test_launcher_is_requeue_safe(stage):
    """No launcher may set --requeue: on this controller it is a permanent hold.

    This assertion was originally the opposite. Measured paired in one
    scheduling window on identical scripts differing only in this directive:
    WITH --requeue the job went straight to PENDING(JobHoldMaxRequeue); WITHOUT
    it the job was scheduled and ran. The controller appears to trip MaxRequeue
    on the FIRST requeue, so any transient NODE_FAIL -- and this cluster reports
    them in the hundreds -- became an unrecoverable hold rather than a retry.

    Recovery lives in scripts/spur_pipeline_driver.sh instead: it resubmits the
    stage and the trainer resumes from its last complete checkpoint, which is
    strictly more capable because it also survives the job id changing.
    """
    directives = _directives(_text(STAGE_LAUNCHERS[stage]))
    assert "--requeue" not in directives, (
        f"{stage}: --requeue causes an immediate JobHoldMaxRequeue on this "
        "controller; the pipeline driver handles resubmission instead"
    )
    # Harmless and still correct: a stage that is resubmitted under the same id
    # would otherwise erase the first attempt's log.
    assert directives.get("--open-mode") == "append"


@pytest.mark.parametrize("stage", sorted(STAGE_LAUNCHERS))
def test_launcher_writes_stdout_and_stderr_under_runs(stage):
    directives = _directives(_text(STAGE_LAUNCHERS[stage]))
    out, err = directives.get("--output"), directives.get("--error")
    assert out and err
    runs = f"{REPO}/runs/"
    assert out.startswith(runs) and err.startswith(runs), (out, err)
    assert out.endswith("-%j.out") and err.endswith("-%j.err"), (out, err)
    # Distinct streams, same job-scoped stem.
    assert out[: -len(".out")] == err[: -len(".err")]


def test_stage_launchers_do_not_share_an_output_stem():
    """Each stage's logs must be findable without disambiguating by job id."""
    stems = {}
    for stage, relative in STAGE_LAUNCHERS.items():
        out = _directives(_text(relative)).get("--output", "")
        stems.setdefault(Path(out).name, []).append(stage)
    collisions = {stem: stages for stem, stages in stems.items() if len(stages) > 1}
    assert not collisions, f"stages share an output filename: {collisions}"


# --------------------------------------------------------------------------- #
# 3. runtime environment
# --------------------------------------------------------------------------- #
def test_no_launcher_requests_strict_resource_preflight():
    """Strict preflight refuses to start without a measured peak profile.

    None of these workloads has one, and preflight additionally cannot join DRM
    cards to HIP ordinals without ``KORE_HIP_INVENTORY_JSON``, so strict mode
    would fail a run that is otherwise healthy.
    """
    for path in sorted(SCRIPTS.glob("spur_*.sbatch")):
        source = path.read_text()
        assert "KORE_RESOURCE_PREFLIGHT=strict" not in source, path.name
        assert "PREFLIGHT_STRICT" not in source, path.name


@pytest.mark.parametrize("stage", NEW_STAGE_LAUNCHERS)
def test_added_launchers_pin_preflight_to_report_after_sourcing_env_local(stage):
    source = _text(STAGE_LAUNCHERS[stage])
    assert "export KORE_RESOURCE_PREFLIGHT=report" in source
    # The export must come AFTER .env.local is sourced, or the shared env file
    # (read with `set -a`) could override the launch's choice.
    assert source.index("kore_secure_source_env") < source.index(
        "export KORE_RESOURCE_PREFLIGHT=report"
    )


@pytest.mark.parametrize("stage", NEW_STAGE_LAUNCHERS)
def test_added_launchers_are_offline_and_resolve_the_venv_accelerate(stage):
    source = _text(STAGE_LAUNCHERS[stage])
    assert "export HF_HUB_OFFLINE=1" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "export PYTHONPATH=" in source
    # launch_distributed.sh execs BARE `accelerate`; without the venv on PATH the
    # job dies with "accelerate: not found" after the allocation is granted.
    assert 'export PATH="/home/shasriva/kore-venv/bin:$PATH"' in source


@pytest.mark.parametrize("stage", NEW_STAGE_LAUNCHERS)
def test_added_launchers_fail_closed_on_a_partial_gpu_allocation(stage):
    """The 8-GPU check must run BEFORE the physical masks are dropped.

    Once ``ROCR_VISIBLE_DEVICES`` is unset, torch reports every device on the
    node, so a SPUR partial allocation becomes invisible and the run would
    silently train a 14B on fewer than eight devices.
    """
    source = _text(STAGE_LAUNCHERS[stage])
    assert 'if [[ "$VISIBLE_GPUS" != "8" ]]' in source
    assert source.index("VISIBLE_GPUS=") < source.index("unset ROCR_VISIBLE_DEVICES")


@pytest.mark.parametrize("stage", NEW_STAGE_LAUNCHERS)
def test_added_launchers_verify_their_own_requeue_landed(stage):
    """``scontrol requeue`` exits 0 even when the controller is unreachable.

    Trusting that status would retire the job with training unfinished, so the
    helper has to inspect the output too - the same guard the datagen and
    frontier-stage arrays use.
    """
    source = _text(STAGE_LAUNCHERS[stage])
    assert "requeue_self()" in source
    assert "scontrol requeue" in source
    assert "no leader" in source, "requeue output is not checked for a failure string"
    assert "handle_requeue_warning" in source and "trap handle_requeue_warning USR1" in source
    # spurctld is not reachable at the CLI's localhost default on every node, so
    # the job says up front whether its own requeue can land, and retries against
    # an explicit controller address before giving up.
    assert "scontrol ping" in source
    assert "KORE_SPUR_CONTROLLER_ADDR" in source
    # The local timer must fire before the wall clock, since Slurm does not
    # auto-requeue a job that merely times out.
    assert "KORE_REQUEUE_AFTER_SECONDS" in source
    limit = _seconds(_directives(source)["--time"])

    # Two designs are legitimate here, and the requirement is the same for both:
    # whatever the drain resolves to, it must fire BEFORE the wall clock, because
    # Slurm does not auto-requeue a job that merely times out.
    #
    #   dpo/grpo: a numeric default -- sleep "${KORE_REQUEUE_AFTER_SECONDS:-81000}"
    #   sft:      an EMPTY default, because the drain is DERIVED from the job's
    #             real TimeLimit at runtime and so tracks --time automatically
    #             instead of being a constant to update in two places.
    #
    # This test used to int() whatever followed `:-`, which raised ValueError on
    # the empty one -- i.e. it demanded the worse design. Accept either, and
    # check the property that matters.
    numeric = re.search(r"KORE_REQUEUE_AFTER_SECONDS:-(\d+)", source)
    if numeric:
        default = int(numeric.group(1))
        assert 0 < default < limit, (default, limit)
    else:
        assert "squeue" in source and "%l" in source, (
            "an empty KORE_REQUEUE_AFTER_SECONDS default is only safe if the "
            "drain reads the real TimeLimit instead of assuming one")
        # The numeric fallback matters more than it looks: this controller often
        # does not report TimeLimit at all, so the fallback is the live path. One
        # set later than --time means a hard kill with no graceful drain and no
        # resubmit marker for the supervisor to follow.
        fallbacks = [int(f) for f in re.findall(r"WALL_SECONDS:-(\d+)", source)]
        assert fallbacks, "no numeric fallback for a derived drain timer"
        for fb in fallbacks:
            assert 0 < fb <= limit, (fb, limit)


@pytest.mark.parametrize("stage", NEW_STAGE_LAUNCHERS)
def test_added_launchers_keep_a_stable_output_dir_for_resume(stage):
    """Resume works only if ``output_dir`` is identical on the requeued child.

    Every stage discovers its own resume point from ``output_dir`` (the Trainer
    stages via ``latest_checkpoint``, GRPO via ``_find_grpo_resume_checkpoint``),
    so the launcher must not derive it from anything job-specific.
    """
    source = _text(STAGE_LAUNCHERS[stage])
    default_out = source.split('OUT_DIR="${3:-')[1].split('}')[0]
    assert default_out.startswith("runs/"), default_out
    for volatile in ("$SLURM_JOB_ID", "$(date", "$$", "$RANDOM"):
        assert volatile not in default_out, (stage, default_out)


@pytest.mark.shell
@pytest.mark.parametrize("stage", sorted(STAGE_LAUNCHERS))
def test_launcher_is_syntactically_valid_bash(stage):
    result = subprocess.run(["bash", "-n", str(REPO / STAGE_LAUNCHERS[stage])],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# 4. the stage chain the launchers encode
# --------------------------------------------------------------------------- #
def _launcher_defaults(stage: str) -> dict[str, str]:
    """The positional defaults (``CFG`` / ``FROM_STAGE`` / ``OUT_DIR``)."""
    source = _text(STAGE_LAUNCHERS[stage])
    out = {}
    for name, marker in (("cfg", 'CFG="${1:-'),
                         ("from", 'FROM_STAGE="${2:-'),
                         ("out", 'OUT_DIR="${3:-')):
        out[name] = source.split(marker)[1].split('}')[0]
    return out


def test_launcher_defaults_chain_midtrain_to_sft_to_dpo_to_grpo():
    """Each stage must default to training the previous stage's output.

    The shipped configs all name the RAW base model, so a launcher that failed
    to override ``model_id`` would silently re-fine-tune Qwen3-14B from scratch
    and look completely healthy while doing it.
    """
    sft, dpo, grpo = (_launcher_defaults(s) for s in ("sft", "dpo", "grpo"))
    midtrain_out = json.loads(
        (REPO / "configs" / "midtrain_14b_full.json").read_text())["output_dir"]
    assert sft["from"].startswith("runs/midtrain"), sft["from"]
    assert midtrain_out.startswith("runs/midtrain")
    assert dpo["from"] == sft["out"], (dpo["from"], sft["out"])
    assert grpo["from"] == dpo["out"], (grpo["from"], dpo["out"])
    assert len({sft["out"], dpo["out"], grpo["out"]}) == 3


@pytest.mark.parametrize("stage", NEW_STAGE_LAUNCHERS)
def test_launcher_default_config_is_the_shipped_stage_config(stage):
    cfg = _launcher_defaults(stage)["cfg"]
    assert cfg == f"$REPO/configs/{stage}_14b_full.json", cfg
    assert (REPO / "configs" / f"{stage}_14b_full.json").is_file()


@pytest.mark.parametrize("stage", NEW_STAGE_LAUNCHERS)
def test_launcher_resolves_its_config_through_the_shared_resolver(stage):
    """All three must use one resolver, or the handoff rules drift per stage."""
    source = _text(STAGE_LAUNCHERS[stage])
    assert "scripts/spur_resolve_launch_config.py" in source
    assert f"--stage {stage}" in source
    assert '--from-stage "$FROM_STAGE"' in source
    assert '--output-dir "$OUT_DIR"' in source
    # And it must launch the RESOLVED file, never the shipped one.
    assert f'launch_distributed.sh {stage} "$RESOLVED"' in source


# --------------------------------------------------------------------------- #
# 5. the resolver itself
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(SCRIPTS))


def _resolver():
    import spur_resolve_launch_config

    return spur_resolve_launch_config


def _fake_checkpoint(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (directory / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 8)
    return directory


def _stage_config(stage: str) -> dict:
    return json.loads((REPO / "configs" / f"{stage}_14b_full.json").read_text())


@pytest.mark.parametrize("stage", ["sft", "dpo"])
def test_resolver_overrides_model_id_to_the_previous_stage_output(stage, tmp_path):
    resolver = _resolver()
    previous = _fake_checkpoint(tmp_path / "runs" / "prev")
    dataset = tmp_path / "data" / "rows.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n")

    config = dict(_stage_config(stage), dataset_path=str(dataset))
    resolved, changes = resolver.resolve(
        stage, config, from_stage=str(previous), output_dir="runs/next",
        repo_root=tmp_path)

    assert resolved["model_id"] == str(previous)
    assert resolved["output_dir"] == "runs/next"
    assert any("model_id" in change for change in changes)
    # The stale Hub revision is deliberately kept: model_spec ignores it for a
    # local directory and logs that it did, so it remains the lineage record.
    assert resolved["model_revision"] == config["model_revision"]


def test_resolver_rejects_a_previous_stage_that_never_consolidated_weights(tmp_path):
    resolver = _resolver()
    empty = tmp_path / "runs" / "midtrain_unfinished"
    empty.mkdir(parents=True)
    (empty / "config.json").write_text("{}")          # no safetensors shard

    with pytest.raises(ValueError, match="safetensors"):
        resolver.resolve("sft", _stage_config("sft"), from_stage=str(empty),
                         repo_root=tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        resolver.resolve("sft", _stage_config("sft"),
                         from_stage=str(tmp_path / "nope"), repo_root=tmp_path)


def test_resolver_dash_leaves_the_config_untouched(tmp_path):
    resolver = _resolver()
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("{}\n")
    config = dict(_stage_config("sft"), dataset_path=str(dataset))

    resolved, changes = resolver.resolve("sft", config, from_stage="-",
                                         output_dir="-", repo_root=tmp_path)
    assert resolved["model_id"] == config["model_id"]
    assert resolved["output_dir"] == config["output_dir"]
    assert changes == []


def test_resolver_rejects_a_dataset_path_nothing_produces(tmp_path):
    resolver = _resolver()
    config = dict(_stage_config("sft"), dataset_path="data/does/not/exist.jsonl")
    with pytest.raises(ValueError, match="dataset_path"):
        resolver.resolve("sft", config, repo_root=tmp_path)


def test_resolver_reroots_the_grpo_coevolution_archive(tmp_path):
    """``_build_opus_scores`` derives the archive root from the distill path.

    Leaving the shipped ``data/full14b/...`` in place would point the
    regret-vs-Opus curriculum at a different data root than the run trains
    against - and it fails SAFE, so it would go unnoticed.
    """
    resolver = _resolver()
    previous = _fake_checkpoint(tmp_path / "runs" / "dpo_out")
    resolved, _ = resolver.resolve(
        "grpo", _stage_config("grpo"), from_stage=str(previous),
        output_dir="runs/grpo_out", data_root="data/b05factory", repo_root=tmp_path)

    assert resolved["coevolve_distill_path"] == "data/b05factory/coevolve_wins.jsonl"
    assert resolved["coevolve_opus_scores_path"] == "data/b05factory/opus_scores.json"
    from pathlib import PurePosixPath

    assert str(PurePosixPath(resolved["coevolve_distill_path"]).parent) == "data/b05factory"


def test_resolver_strict_parse_rejects_a_key_no_stage_config_has(tmp_path):
    """A misspelt key must cost milliseconds here, not an 8-rank 14B load."""
    resolver = _resolver()
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("{}\n")
    config = dict(_stage_config("sft"), dataset_path=str(dataset),
                  max_seq_lenght=17408)
    with pytest.raises(TypeError):
        resolver.resolve("sft", config, repo_root=tmp_path)


def test_resolver_rejects_an_unknown_stage():
    with pytest.raises(ValueError, match="unknown stage"):
        _resolver().resolve("midtrain", {})


@pytest.mark.parametrize("stage", ["sft", "dpo", "grpo"])
def test_resolver_cli_writes_a_config_the_stage_loader_accepts(stage, tmp_path):
    """End-to-end through ``main``: the written file must load as that stage."""
    resolver = _resolver()
    previous = _fake_checkpoint(tmp_path / "prev")
    config = _stage_config(stage)
    if stage in ("sft", "dpo"):
        dataset = tmp_path / "rows.jsonl"
        dataset.write_text("{}\n")
        config["dataset_path"] = str(dataset)
    source = tmp_path / f"{stage}.json"
    source.write_text(json.dumps(config))
    out = tmp_path / "resolved" / f"{stage}.resolved.json"

    rc = resolver.main([
        "--stage", stage, "--config", str(source), "--out", str(out),
        "--from-stage", str(previous), "--output-dir", f"runs/{stage}_out",
        "--repo-root", str(tmp_path),
    ])
    assert rc == 0
    written = json.loads(out.read_text())
    assert written["model_id"] == str(previous)
    assert written["output_dir"] == f"runs/{stage}_out"
    assert not list(out.parent.glob("*.partial.*")), "staging file was left behind"
    resolver._stage_loader(stage)(dict(written))


def test_resolver_cli_reports_a_bad_handoff_without_writing_anything(tmp_path):
    resolver = _resolver()
    source = tmp_path / "sft.json"
    source.write_text(json.dumps(_stage_config("sft")))
    out = tmp_path / "sft.resolved.json"

    rc = resolver.main([
        "--stage", "sft", "--config", str(source), "--out", str(out),
        "--from-stage", str(tmp_path / "missing"), "--repo-root", str(tmp_path),
    ])
    assert rc == 2
    assert not out.exists()


if __name__ == "__main__":  # pragma: no cover - convenience for ad-hoc runs
    sys.exit(pytest.main([__file__, "-v"]))
