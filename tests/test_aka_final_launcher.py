"""Contract for the final measured arena sweep and its submitter.

Three failures already cost real allocations, and each has a test here:

1. ``--account=amd-general`` with a QoS from another family is a phantom
   association. The controller accepts the submission and never dispatches it, so
   the job pends forever with nothing to read. Every arena launcher carried that
   pairing, which is why nothing submitted through them ever landed.
2. Running an arm whose ``--out`` holds no baseline produces a full sweep of
   correct kernels with null denominators. It happened on 2026-08-10 to 253 of
   302 correct kernels and is unrecoverable, because task workspaces are deleted
   after scoring. The baseline phase must therefore come first, in the same
   directory.
3. Defaulting the output directory to something keyed on ``SLURM_JOB_ID`` gives
   every requeue a fresh empty directory, which silently restarts a multi-day
   sweep from task 1 and discards the baseline the arms depend on.

Everything here reads the scripts as text. Nothing submits a job and nothing
needs a GPU.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
FINAL = SCRIPTS / "spur_aka_final.sbatch"
ONE_NODE = SCRIPTS / "spur_aka_1node.sbatch"
SUBMITTER = SCRIPTS / "queue_aka_final.sh"
PREFLIGHT = SCRIPTS / "aka_gateway_preflight.py"
RUNNER = SCRIPTS / "run_agent_kernel_arena.py"

#: Account -> the QoS family it is actually associated with. Pairing across
#: families is accepted at submit and never dispatches.
ACCOUNT_QOS = {
    "amd-primus": "amd-primus-qos",
    "amd-burst": "amd-burst-qos",
    "amd-general": "amd-general-qos",
}

ARENA_LAUNCHERS = (FINAL, ONE_NODE)


def _directives(source: str) -> dict[str, str]:
    """``#SBATCH --key=value`` / ``#SBATCH --key`` pairs (flags map to "")."""
    found: dict[str, str] = {}
    for line in source.splitlines():
        line = line.strip()
        if not line.startswith("#SBATCH "):
            continue
        key, _, value = line[len("#SBATCH "):].strip().partition("=")
        found[key.strip()] = value.strip()
    return found


# --------------------------------------------------------------------------- #
# 1. the files exist and are executable as scripts
# --------------------------------------------------------------------------- #
def test_every_piece_of_the_sweep_exists():
    for path in (FINAL, ONE_NODE, SUBMITTER, PREFLIGHT):
        assert path.is_file(), f"{path.name} is missing"


# --------------------------------------------------------------------------- #
# 2. scheduler directives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("launcher", ARENA_LAUNCHERS, ids=lambda p: p.name)
def test_account_and_qos_are_from_the_same_family(launcher):
    directives = _directives(launcher.read_text())
    account = directives.get("--account")
    qos = directives.get("--qos")
    assert account in ACCOUNT_QOS, f"unknown account {account!r}"
    assert qos == ACCOUNT_QOS[account], (
        f"{launcher.name} pairs account {account!r} with QoS {qos!r}. Cross-family "
        f"pairings are phantom associations: accepted at submit, never dispatched."
    )


@pytest.mark.parametrize("launcher", ARENA_LAUNCHERS, ids=lambda p: p.name)
def test_arena_launchers_target_primus(launcher):
    # burst was abandoned after six consecutive node deaths on idle hardware.
    assert _directives(launcher.read_text()).get("--account") == "amd-primus"


@pytest.mark.parametrize("launcher", ARENA_LAUNCHERS, ids=lambda p: p.name)
def test_requests_a_whole_node_with_eight_untyped_gpus(launcher):
    directives = _directives(launcher.read_text())
    gres = directives.get("--gres")
    # The count matters, the type must NOT be named: gpu:mi355x:8 is accepted,
    # picks a node, and never confirms. Measured 0/3 vs 2/3 dispatched.
    assert gres == "gpu:8", gres
    assert "--exclusive" in directives, (
        "without --exclusive the scheduler co-locates this on a GPU already "
        "holding hundreds of GiB of someone else's VRAM, and a timing run on a "
        "shared GPU is not a measurement"
    )
    assert directives.get("--partition") == "amd-spur"


def test_final_launcher_walltime_matches_a_limit_this_qos_accepts():
    limit = _directives(FINAL.read_text()).get("--time")
    # 3 days is the limit amd-primus-qos demonstrably accepts: the v5 SFT job
    # runs under it. A longer value fails the same silent way a wrong account
    # does, so this is pinned rather than merely bounded.
    assert limit == "3-00:00:00", limit


def test_final_launcher_does_not_set_requeue():
    """--requeue turns the first requeue into a permanent hold on this controller.

    This assertion was the exact opposite until 2026-08-18, contradicting
    tests/test_spur_stage_launchers.py::test_launcher_is_requeue_safe.

    Be precise about the failure mode, because the loose version of this rule
    misleads in both directions. The directive does NOT stop a job being scheduled:
    job 13907 was submitted twelve minutes after the commit that added it and ran
    for days. Reasoning from "a requeue job never lands" would misattribute a queue
    stall that is actually a QOS group node cap.

    What it does is trip MaxRequeue on the FIRST requeue: the stage-launcher
    measurement paired identical scripts differing only in this directive, and the
    one carrying it ended in PENDING(JobHoldMaxRequeue) while the other ran. So it
    is harmless until a transient NODE_FAIL -- reported in the hundreds on this
    cluster -- after which the sweep is dead in a hold rather than retried.

    It also buys nothing: resubmitting against the same --out resumes from the
    ledgers and survives the job id changing, which a requeue does not.
    """
    directives = _directives(FINAL.read_text())
    assert "--requeue" not in directives, (
        "--requeue causes an immediate JobHoldMaxRequeue on this controller; "
        "resubmit against the same --out instead, which resumes from the ledgers"
    )
    # Still required: a resubmitted sweep must not truncate the previous attempt's
    # log, which is the only record that it ran at all.
    assert directives.get("--open-mode") == "append"


def test_no_arena_launcher_sets_requeue():
    """The rule has to hold for every arena launcher, not just the final one."""
    for launcher in ARENA_LAUNCHERS:
        directives = _directives(launcher.read_text())
        assert "--requeue" not in directives, (
            f"{launcher.name}: --requeue is an immediate JobHoldMaxRequeue here"
        )


# --------------------------------------------------------------------------- #
# 3. phase ordering -- the baseline must precede every arm
# --------------------------------------------------------------------------- #
def test_baseline_phase_runs_before_any_arm():
    source = FINAL.read_text()
    baseline_at = source.index("phase baseline")
    arm_loop_at = source.index("for arm in $RESOLVED_ARMS")
    assert baseline_at < arm_loop_at, (
        "arms would run before the baseline, which is how 253 correct kernels "
        "ended up with null denominators"
    )


def test_baseline_is_unconditional_rather_than_marker_gated():
    # cmd_baseline_merge writes baseline_results.json even from partial ledgers,
    # so its existence does not mean "complete" and must not be used to skip the
    # phase. The phase is idempotent instead: it skips task ids already timed.
    source = FINAL.read_text()
    assert re.search(r"^phase baseline ", source, re.M), (
        "the baseline phase must be invoked unconditionally"
    )
    assert "-f \"$OUT/baseline_results.json\"" not in source, (
        "baseline_results.json is not a completeness marker; gating on it would "
        "skip an incomplete baseline"
    )


def test_all_arms_share_one_output_directory():
    source = FINAL.read_text()
    # A single OUT, used by the baseline and by every arm, is what makes the
    # denominators identical across arms so the baseline cancels in arm-vs-arm
    # ratios.
    assert source.count("--out $OUT") >= 1
    assert "aka_${SLURM_JOB_ID" not in source, (
        "keying the output directory on the job id gives every requeue an empty "
        "directory and restarts the sweep from task 1"
    )
    assert "KORE_AKA_OUT" in source


# --------------------------------------------------------------------------- #
# 4. the API-arm preflight
# --------------------------------------------------------------------------- #
def test_api_arms_are_preflighted_before_the_baseline_burns_the_allocation():
    source = FINAL.read_text()
    preflight_at = source.index("aka_gateway_preflight.py")
    baseline_at = source.index("phase baseline")
    assert preflight_at < baseline_at, (
        "an unreachable gateway must be detected before hours of baseline work, "
        "not after"
    )


def test_launcher_api_detection_matches_the_runner():
    """The sbatch decides concurrency and preflight from the model name; the
    runner decides whether to inject an API generate. If the two disagree, an arm
    is either preflighted and then served locally, or served over the network
    with no preflight at all."""
    runner = RUNNER.read_text()
    match = re.search(r"_API_MODEL_PREFIXES\s*=\s*\(([^)]*)\)", runner)
    assert match, "could not find _API_MODEL_PREFIXES in the runner"
    prefixes = set(re.findall(r'"([^"]+)"', match.group(1)))
    assert prefixes, "no prefixes parsed"

    case = re.search(r"is_api_model\(\)\s*\{.*?case.*?in\s*(.*?)\)\s*return 0",
                     FINAL.read_text(), re.S)
    assert case, "could not find is_api_model in the launcher"
    shell_globs = {p.strip().rstrip("*") for p in case.group(1).split("|")}
    assert prefixes == shell_globs, (
        f"runner prefixes {sorted(prefixes)} != launcher globs "
        f"{sorted(shell_globs)}"
    )


def test_preflight_distinguishes_config_faults_from_unreachable_gateways():
    source = PREFLIGHT.read_text()
    # The sbatch does not branch on these today, but an operator reading a log
    # needs to know whether to fix .env.local or wait for the gateway.
    assert "return 3" in source and "return 4" in source
    # It must force the model the way _api_generate does, or it would preflight
    # whatever KORE_TEACHER_MODEL happens to say.
    assert 'os.environ["KORE_TEACHER_MODEL"] = args.model' in source
    assert "teacher.model != args.model" in source
    # An empty reply is the failure that looks exactly like a bad model.
    assert "empty reply" in source


def test_a_failed_api_preflight_skips_that_arm_without_discarding_the_node():
    source = FINAL.read_text()
    assert "SKIPPED_ARMS" in source
    assert "INCOMPLETE:" in source, (
        "a skipped arm must make the job exit loudly, or a missing arm reads as "
        "a complete result set"
    )


# --------------------------------------------------------------------------- #
# 5. the submitter must not be able to touch SFT
# --------------------------------------------------------------------------- #
def _code(path: Path) -> str:
    """Executable lines only.

    The prose in these scripts names the commands it promises not to run, so a
    substring search over the whole file would flag its own documentation.
    """
    return "\n".join(line for line in path.read_text().splitlines()
                     if not line.lstrip().startswith("#"))


def test_submitter_cannot_disturb_a_running_job():
    source = _code(SUBMITTER)
    for forbidden in ("scancel", "scontrol update", "scontrol requeue"):
        assert forbidden not in source, (
            f"the submitter contains {forbidden!r}; it must be read-only with "
            f"respect to every job it did not create"
        )


def test_submitter_uses_only_flags_this_sbatch_implements():
    """SPUR's sbatch is a reimplementation with a narrower flag set than Slurm's.

    ``--nice`` does not exist here: passing it aborts the submission with
    "unexpected argument", which is how the first attempt to queue this sweep
    failed. Anything the submitter passes must be in the supported set.
    """
    supported = {"--account", "--qos", "--export", "--dependency", "--requeue",
                 "--exclusive", "--hold", "--partition", "--job-name", "--time",
                 "--gres", "--nodes", "--cpus-per-task", "--open-mode",
                 "--output", "--error", "--begin"}
    passed = set(re.findall(r"(--[a-z-]+)=", _code(SUBMITTER)))
    # KORE_* assignments are environment, not flags.
    unsupported = {f for f in passed if f not in supported}
    assert not unsupported, f"SPUR's sbatch does not implement {sorted(unsupported)}"


def test_submitter_passes_the_matching_account_and_qos_explicitly():
    """Both flags must be passed, and they must be a matching family pair.

    These were literal ``--account=amd-primus --qos=amd-primus-qos`` until
    2026-08-18, when amd-primus-qos turned out to carry a group node cap that was
    16/16 with 32 jobs queued behind it. They are now variables so a sweep can be
    routed to the uncapped amd-burst pair, and the pairing rule the controller
    enforces is checked inside the submitter instead of being spelled literally
    here. The invariant this test defends is unchanged: never submit one without
    the other, and never cross families.
    """
    source = _code(SUBMITTER)
    assert '--account="$ACCOUNT"' in source
    assert '--qos="$QOS"' in source
    assert 'ACCOUNT="${KORE_AKA_ACCOUNT:-amd-primus}"' in source, \
        "the default must stay primus so existing invocations are unchanged"
    for account, qos in ACCOUNT_QOS.items():
        if account in source:
            assert qos in source, f"{account} must map to {qos}"


def test_submitter_does_not_smuggle_space_bearing_values_through_export():
    """``--export`` takes a comma-separated list, and the default arm list is
    ``base opus``. Putting that inside the list is how the opus arm gets dropped
    from a three-day sweep with nothing in the log to show it."""
    source = _code(SUBMITTER)
    assert "--export=ALL," not in source, (
        "pass KORE_AKA_* through the environment with a plain --export=ALL"
    )
    assert 'export KORE_AKA_ARMS="$ARMS"' in source
    assert 'export KORE_AKA_OUT="$OUT"' in source


def test_submitter_refuses_to_queue_a_second_copy_of_the_same_sweep():
    # Two sweeps sharing one --out would delete each other's task workspaces
    # while the other was still evaluating in them. The guard keys on the job
    # name so a differently-named sweep with its own --out can queue alongside.
    source = _code(SUBMITTER)
    assert "already queued" in source
    assert 'awk -v n="$NAME"' in source
    assert '--job-name="$NAME"' in source


def test_submitter_validates_the_v5_checkpoint_before_the_queue_wait():
    """A missing or mistyped checkpoint must fail at submit.

    The arena exits 2 on an unloadable model, and discovering that after a
    multi-day queue wait spends a scarce exclusive allocation on a typo.
    """
    source = _code(SUBMITTER)
    assert "KORE_AKA_V5_MODEL is unset" in source
    assert "config.json" in source
    assert "safetensors" in source
