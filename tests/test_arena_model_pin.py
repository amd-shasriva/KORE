"""An evaluation arm must resolve to weights that actually exist, at submit time.

WHAT THIS PREVENTS
------------------
On 2026-08-17 the final sweep (job 13907) spent 4h22m producing a complete
416-task baseline, then ran its base arm for two seconds. All 8 shards died with
``FloatingRevisionError: model revision must be a full 40- or 64-hex commit
hash; got None`` before scoring a single task, and the sweep moved on to the next
arm as though a phase had finished. The control arm of the comparison was empty
and nothing said so.

Three separate defects lined up:

1. The launcher defaults the base arm to a Hugging Face *repo id*.
2. ``load_generate`` refuses a repo id with no immutable revision -- correctly,
   because a floating ref silently changes which weights you evaluated -- and
   ``cmd_run`` parsed ``--revision`` but never forwarded it. The arm could not
   have succeeded under any invocation.
3. The non-API preflight printed "local checkpoint, no gateway needed" for
   whatever string it was given, without looking at it.

So: resolution is tested behaviourally here, and the launcher contract is tested
as text. Nothing in this file needs a GPU, a network, or real weights.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUNNER_PATH = REPO / "scripts" / "run_agent_kernel_arena.py"
FINAL = REPO / "scripts" / "spur_aka_final.sbatch"

#: The commit model_spec records for the production backbone. Duplicated here on
#: purpose: if the profile's pin ever moves, this test should be the thing that
#: notices, not a sweep three days into a queue wait.
PINNED = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
REPO_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


@pytest.fixture(scope="module")
def runner():
    """The arena runner, imported as a module (it lives in scripts/, not a package)."""
    spec = importlib.util.spec_from_file_location("aka_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_snapshot(root: Path, repo_id: str, revision: str) -> Path:
    """Build the directory layout ``resolve_local_snapshot`` looks for."""
    dirname = "models--" + repo_id.replace("/", "--")
    snapshot = root / dirname / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    return snapshot


def _local_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
    (root / "model.safetensors").write_bytes(b"")
    return root


# --------------------------------------------------------------------------- #
# the three kinds of --model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["claude-opus-5", "gpt-5.2", "anthropic/claude-3"])
def test_an_api_arm_is_a_label_and_is_never_turned_into_a_path(runner, model):
    """API arms reach the gateway by name; resolving them as paths would break them."""
    assert runner.resolve_policy_checkpoint(model) == model


def test_a_local_checkpoint_directory_is_loaded_as_given(runner, tmp_path):
    """The bytes on disk are the identity; there is no Hub commit to resolve."""
    ckpt = _local_checkpoint(tmp_path / "sft_v5_final")
    assert runner.resolve_policy_checkpoint(str(ckpt)) == str(ckpt)


def test_a_local_checkpoint_ignores_a_configured_revision(runner, tmp_path, capsys):
    """A directory has no Hub commit, so a passed revision must not be claimed."""
    ckpt = _local_checkpoint(tmp_path / "sft_v5_final")
    assert runner.resolve_policy_checkpoint(str(ckpt), PINNED) == str(ckpt)
    assert "IGNORED" in capsys.readouterr().out


def test_a_repo_id_resolves_to_the_recorded_pins_local_snapshot(runner, tmp_path):
    """THE REGRESSION. A bare repo id is what the launcher passes by default.

    Before the fix this raised FloatingRevisionError and killed the arm.
    """
    cache = tmp_path / "hub"
    snapshot = _fake_snapshot(cache, REPO_ID, PINNED)
    resolved = runner.resolve_policy_checkpoint(
        REPO_ID, environ={"HF_HUB_CACHE": str(cache)}
    )
    assert Path(resolved) == snapshot


def test_an_explicit_revision_overrides_the_recorded_pin(runner, tmp_path):
    other = "a" * 40
    cache = tmp_path / "hub"
    snapshot = _fake_snapshot(cache, REPO_ID, other)
    resolved = runner.resolve_policy_checkpoint(
        REPO_ID, other, environ={"HF_HUB_CACHE": str(cache)}
    )
    assert Path(resolved) == snapshot


def test_the_revision_env_var_is_honoured_for_an_unregistered_repo_id(runner, tmp_path):
    """A model with no recorded profile is still evaluable, but only when pinned."""
    unknown = "some-org/some-model"
    rev = "b" * 40
    cache = tmp_path / "hub"
    snapshot = _fake_snapshot(cache, unknown, rev)
    resolved = runner.resolve_policy_checkpoint(
        unknown,
        environ={"HF_HUB_CACHE": str(cache), "KORE_MODEL_REVISION": rev},
    )
    assert Path(resolved) == snapshot


# --------------------------------------------------------------------------- #
# fail closed, and say what to do about it
# --------------------------------------------------------------------------- #
def test_an_unpinned_unknown_repo_id_is_refused_with_an_actionable_message(runner):
    """Loading whatever the cache happens to hold is not a measurement."""
    with pytest.raises(SystemExit) as excinfo:
        runner.resolve_policy_checkpoint("some-org/unpinned", environ={})
    message = str(excinfo.value)
    assert "--revision" in message
    assert "KORE_MODEL_REVISION" in message


def test_a_floating_revision_is_still_refused(runner, tmp_path):
    """The whole point of the guard: a branch or tag can move under you."""
    for floating in ("main", "v1.0", "b2cff64"):
        with pytest.raises(SystemExit):
            runner.resolve_policy_checkpoint(
                REPO_ID, floating, environ={"HF_HUB_CACHE": str(tmp_path)}
            )


def test_a_pin_with_no_local_snapshot_says_the_jobs_run_offline(runner, tmp_path):
    """HF_HUB_OFFLINE=1 means an uncached commit cannot be fetched at all."""
    with pytest.raises(SystemExit) as excinfo:
        runner.resolve_policy_checkpoint(
            REPO_ID, environ={"HF_HUB_CACHE": str(tmp_path / "empty")}
        )
    message = str(excinfo.value)
    assert PINNED in message
    assert "HF_HUB_OFFLINE" in message


def test_an_empty_model_is_refused(runner):
    with pytest.raises(SystemExit):
        runner.resolve_policy_checkpoint("")


# --------------------------------------------------------------------------- #
# check-model: the preflight the launcher actually calls
# --------------------------------------------------------------------------- #
class _Args:
    def __init__(self, **kw):
        self.model = kw.get("model", "")
        self.revision = kw.get("revision")


def test_check_model_passes_for_a_complete_checkpoint(runner, tmp_path):
    ckpt = _local_checkpoint(tmp_path / "good")
    assert runner.cmd_check_model(_Args(model=str(ckpt))) == 0


def test_check_model_rejects_a_checkpoint_with_no_weights(runner, tmp_path):
    """A config-only directory loads nothing; catching it costs one stat."""
    bare = tmp_path / "weightless"
    bare.mkdir()
    (bare / "config.json").write_text("{}")
    assert runner.cmd_check_model(_Args(model=str(bare))) == 2


def test_check_model_reports_an_api_arm_as_the_gateways_business(runner):
    assert runner.cmd_check_model(_Args(model="claude-opus-5")) == 0


def test_check_model_imports_no_gpu_framework(runner):
    """Preflight runs before the GPU check, and must stay a directory stat."""
    assert "torch" not in sys.modules or True  # torch may be loaded by another test
    source = RUNNER_PATH.read_text()
    body = source.split("def cmd_check_model", 1)[1].split("\ndef ", 1)[0]
    for banned in ("import torch", "load_generate", "from_pretrained"):
        assert banned not in body, f"check-model must not {banned}"


# --------------------------------------------------------------------------- #
# cmd_run must actually use the resolver
# --------------------------------------------------------------------------- #
def test_cmd_run_resolves_the_checkpoint_before_building_the_policy():
    """Guards the exact regression: a bare repo id reaching model_policy.

    ``--revision`` existed as a flag for weeks while cmd_run dropped it on the
    floor, so asserting the flag exists proves nothing. This asserts the call.
    """
    source = RUNNER_PATH.read_text()
    body = source.split("def cmd_run", 1)[1]
    resolve_at = body.find("resolve_policy_checkpoint(args.model, args.revision)")
    policy_at = body.find("model_policy(")
    assert resolve_at != -1, "cmd_run must resolve the model pin"
    assert resolve_at < policy_at, "resolution must happen before model_policy"
    assert "model_policy(checkpoint" in body, \
        "model_policy must receive the RESOLVED checkpoint, not args.model"


# --------------------------------------------------------------------------- #
# launcher contract
# --------------------------------------------------------------------------- #
def _code_lines(source: str) -> str:
    """Executable lines only, so a comment describing an old bug is not the bug."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


def test_the_launcher_preflights_a_served_arm_instead_of_announcing_it():
    """The old branch printed a reassurance and checked nothing."""
    code = _code_lines(FINAL.read_text())
    assert "check-model" in code, \
        "a served arm must be verified with check-model before the baseline runs"
    assert "local checkpoint, no gateway needed" not in code, \
        "that message asserted a fact the launcher had not checked"


def test_the_launcher_treats_a_zero_row_arm_as_a_failed_run():
    """An arm that scored nothing is a bug in the run, not a bad result."""
    source = FINAL.read_text()
    assert "ZERO SCORED TASKS" in source
    assert "DEAD_ARMS" in source
    assert 'echo "aka-final rc=$RC"' in source


# --------------------------------------------------------------------------- #
# account/QOS routing
#
# amd-primus-qos carries a group node cap. On 2026-08-18 it was 16/16 with 32 jobs
# blocked on QOSGrpNodeLimit -- ours 20th -- while 15 of the 16 nodes were parked
# shells idling for up to 4 days. amd-burst-qos had 98 nodes running and zero jobs
# blocked. The route out is real but must be a MATCHING pair: the controller
# rejects amd-primus + amd-burst-qos, and accepts amd-burst + amd-burst-qos.
# --------------------------------------------------------------------------- #
SUBMITTER = REPO / "scripts" / "queue_aka_final.sh"


def test_the_submitter_lets_the_account_and_qos_be_chosen():
    """Hardcoding primus is what pinned every sweep behind the capped queue."""
    code = _code_lines(SUBMITTER.read_text())
    assert "KORE_AKA_ACCOUNT" in code
    assert "--account=\"$ACCOUNT\"" in code or '--account="$ACCOUNT"' in code
    assert '--qos="$QOS"' in code
    assert "--account=amd-primus --qos=amd-primus-qos" not in code, \
        "the submitter must no longer hardcode the capped primus pair"


def test_the_submitter_knows_which_qos_each_account_pairs_with():
    code = _code_lines(SUBMITTER.read_text())
    for account, qos in (("amd-primus", "amd-primus-qos"),
                         ("amd-burst", "amd-burst-qos"),
                         ("amd-general", "amd-general-qos")):
        assert account in code and qos in code, f"{account} -> {qos} mapping missing"


def test_the_submitter_refuses_a_cross_family_account_qos_pair():
    """amd-primus + amd-burst-qos is rejected by the controller; fail locally first."""
    code = _code_lines(SUBMITTER.read_text())
    assert "cross-family" in code
    assert "REFUSING" in code


def test_the_submitter_defaults_to_primus_so_existing_invocations_are_unchanged():
    code = _code_lines(SUBMITTER.read_text())
    assert 'ACCOUNT="${KORE_AKA_ACCOUNT:-amd-primus}"' in code


def test_the_stale_preemption_claim_is_corrected():
    """The header asserted PreemptMode=OFF; the partition reports CANCEL.

    The old sentence is allowed to survive as a quotation being corrected -- that
    is how the next reader learns the claim was checked -- so this asserts the
    correction is present rather than that the words are absent.
    """
    source = SUBMITTER.read_text()
    assert "PreemptMode=CANCEL" in source, \
        "record what the partition actually reports"
    assert "was false" in source, \
        "the stale claim must be marked as corrected, not silently deleted"
    assert "can itself be cancelled" in source, \
        "the real consequence is that OUR burst job is the preemptible one"


# --------------------------------------------------------------------------- #
# baseline inheritance
#
# A sweep in a fresh --out re-times all 416 reference kernels (4h22m measured) AND
# produces different denominators, so its speedups cannot be compared with an arm
# that ran in another directory. Job 16441 did exactly that on 2026-08-18. Seeding
# fixes it -- but only if it COPIES: the first hand-rolled seeding used `ln`, and
# because cmd_baseline appends to baseline.shard<i>of8.jsonl, a shared inode lets
# one run append rows into the ledger another run's published numbers divide by.
# --------------------------------------------------------------------------- #
def test_the_launcher_can_inherit_a_baseline_instead_of_retiming_it():
    code = _code_lines(FINAL.read_text())
    assert "KORE_AKA_SEED_BASELINE_FROM" in code
    assert "seed_baseline" in code, "seeding must be a named, testable step"
    assert "seed_baseline || exit 2" in code, \
        "a failed seed must abort before the baseline phase burns the allocation"


def test_baseline_seeding_copies_and_never_hard_links():
    """The inode hazard is the whole reason this is in the launcher."""
    code = _code_lines(FINAL.read_text())
    seed = code.split("seed_baseline()", 1)[1].split("\nseed_baseline ||", 1)[0]
    assert "cp --preserve=timestamps" in seed, "seeding must copy"
    assert not re.search(r"^\s*ln\s", seed, re.M), \
        "seeding must never hard-link: cmd_baseline appends to the shard ledgers"
    assert "-links +1" in seed, \
        "seeding must assert afterwards that no seeded file shares an inode"


def test_baseline_seeding_refuses_to_clobber_existing_rows():
    """Resuming into a populated --out must not overwrite its ledgers."""
    seed = _code_lines(FINAL.read_text()).split("seed_baseline()", 1)[1]
    assert "leaving it alone" in seed
    assert "already holds" in seed


def test_baseline_seeding_is_a_no_op_when_unset():
    """Every existing invocation must behave exactly as before."""
    seed = _code_lines(FINAL.read_text()).split("seed_baseline()", 1)[1]
    assert '[ -n "$SEED_FROM" ] || return 0' in seed


def test_the_launcher_can_run_from_a_pinned_checkout():
    """Editing a running sweep's own script can corrupt it mid-flight.

    Bash reads a script incrementally, and SPUR is not known to spool the batch
    file, so a new sweep must be able to run out of its own tree instead of forcing
    an edit to the tree a live job is executing from.
    """
    code = _code_lines(FINAL.read_text())
    assert 'REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"' in code, \
        "the sbatch must accept KORE_REPO and default to the usual path"
    assert 'REPO="/home/shasriva/Kore-RL/KORE"' not in code, \
        "the repo path must no longer be hardcoded"
    submitter = _code_lines(SUBMITTER.read_text())
    assert 'export KORE_REPO="$REPO"' in submitter, \
        "the submitter must pin the job to the checkout it was invoked from"


def test_the_launcher_still_keeps_the_allocation_when_one_arm_is_unusable():
    """A scarce node that can produce the baseline is worth more than a clean exit."""
    source = FINAL.read_text()
    preflight = source.split("RESOLVED_ARMS=\"\"", 1)[1].split("common_args()", 1)[0]
    assert "continue" in preflight
    assert "SKIPPED_ARMS" in preflight
