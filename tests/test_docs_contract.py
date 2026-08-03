"""Mechanical anti-drift contract for the project's documentation.

Documentation in this repository has repeatedly gone false without anything
failing. A systematic audit found, among others: a launch command pointing at an
11 MB / 1,360-row development stub while the real corpus is 683 MB / 86k rows; a
preflight command that exits 1 doing nothing; a "recommended path" launcher that
refuses to run; a retracted physics result (``R2 ~ 0.98``) still asserted as
validated in seven separate READMEs; a shipped-config transcription whose
``max_seq_length`` silently discarded 53.6% of a capability slice; and a
"Release prerequisites" section claiming no license had been chosen months after
one was.

Every one of those is mechanically checkable, and none of them was checked. This
module checks them. It deliberately does NOT try to validate prose -- it covers
five decidable classes:

1. **Paths.** Every repo-relative path named in a ``.md`` resolves, from the repo
   root or from the doc's own directory. Absences must be listed in
   :data:`ABSENT_PATHS` *with a reason* -- cluster-only corpus, external repo, or
   an output the documented command itself creates. An unexplained absence fails.
2. **Documented config values.** The JSON blocks in ``configs/README.md`` and
   ``docs/DISTRIBUTED.md`` are parsed and compared key-by-key against the real
   config files, and every key must be a real dataclass field. The doc IS the
   pin, so nothing is duplicated here: changing a config forces a doc edit.
3. **Pinned artifact counts.** Counts stated in prose are compared against the
   artifact that produces them. Pinned deliberately, in the style of
   ``tests/test_task_integrity_gates.py``: these are measurements, so a change to
   an artifact must be an explicit edit here and in the prose rather than a
   silent re-baseline.
4. **Retracted claims.** No doc may reassert the falsified ``R2 ~ 0.98``
   residual-decomposition result as validated, quote a P0 verdict string the
   adjudicator cannot emit, or present the dead ``KORE_PEAK_*`` overrides as
   effective. These are the specific claims that came back after being retracted.
5. **Deprecated entrypoints.** No doc may present a script that
   ``scripts/operations_registry.json`` classifies as deprecated as a recommended
   or production path, because those scripts exit 64 rather than running.

What this deliberately does NOT cover, so nobody reads a green run as "the docs
are true":

* **Whether a documented command works.** Only the fail-closed campaign-mode
  contract is checked textually. Nothing here executes a launcher, an sbatch, a
  serve command, or a docker run.
* **Numbers that need a GPU or a trained checkpoint.** The measured figures in
  ``docs/SFT_READINESS.md`` (step timings, checkpoint sizes, peak memory) and
  ``docs/E2E_SERVING_GATE.md`` (throughput, accuracy) cannot be re-derived on a
  CPU box. They are marked in-document as measurements from a named run.
* **``runs/`` and ``logs/`` references.** Gitignored trainer output; see the note
  on :data:`_PATH_ROOTS`.
* **Prose that is simply wrong.** "X is enforced" is only caught when it names a
  value or a path. A false behavioural claim about code neither of those touches
  still needs a reader.

CPU-only and import-light: no torch, no GPU, no network.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted(
    path
    for path in REPO_ROOT.rglob("*.md")
    if not any(
        part in {".git", ".pytest_cache", "node_modules", "build", "dist"}
        for part in path.relative_to(REPO_ROOT).parts
    )
    # runs/ holds gitignored trainer output, including TRL's auto-generated model
    # cards. They are not project documentation and nobody maintains them.
    and path.relative_to(REPO_ROOT).parts[0] != "runs"
)


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def _load_json(rel_path: str) -> dict:
    return json.loads(_read(rel_path))


def _settings(rel_path: str) -> dict:
    """A config's real fields, minus the ``_comment_*`` justification keys."""
    return {k: v for k, v in _load_json(rel_path).items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# 1. every path a doc names must exist, or be explained
# --------------------------------------------------------------------------- #
#: Repo-relative-looking references that legitimately do not resolve locally.
#: Each entry carries the reason, because "it's fine" is how the 1,360-row stub
#: survived. A path NOT listed here and not on disk fails the suite.
ABSENT_PATHS: dict[str, str] = {
    # ---- cluster-only: present on SPUR, never in a fresh checkout ---------- #
    "data/b05factory/midtrain/corpus.jsonl":
        "cluster-only (SPUR: 683,463,071 B / 86,010 rows); materialize with "
        "data/release/reassemble.sh",
    "data/b05factory/sft/multicap.jsonl":
        "cluster-only (SPUR: 630,488,937 B / 56,493 rows); materialize with "
        "data/release/reassemble.sh",
    "data/b05factory/sft/multicap_v2.jsonl":
        "cluster-only (SPUR: 864,715,542 B / 61,122 rows); the mixture SFT "
        "trains on -- base mix plus 4,629 multi-turn refinement trajectories; "
        "materialize with data/release/reassemble.sh",
    "data/b05factory/sft/multicap_v3.jsonl":
        "cluster-only output of scripts/build_sft_v3_mixture.py; it is the "
        "review-gated production SFT mixture and is not materialized in a fresh checkout",
    "data/b05factory/sft/kernel_multiturn_refine.jsonl":
        "cluster-only; the filtered trajectory slice reassemble.sh concatenates "
        "onto multicap.jsonl to rebuild multicap_v2.jsonl offline",
    "data/b05factory/dpo/pairs.jsonl":
        "cluster-only preference corpus (SPUR: 1,093,321,730 B); materialize with "
        "data/release/reassemble.sh. Named in DATASET_STATUS.md; the check passed "
        "only where the artifact happened to be materialized already, so a fresh "
        "checkout failed on it",
    # ---- outputs the documented command itself creates --------------------- #
    "data/b05factory/sft/hipkittens.jsonl":
        "written by scripts/build_hipkittens_sft.py, the command "
        "docs/HIPKITTENS_INGEST.md documents; data/b05factory is gitignored so it "
        "is never in a fresh checkout",
    "data/b05factory/sft/hipkittens_report.json":
        "the gate report scripts/build_hipkittens_sft.py writes alongside the "
        "slice; contamination, dedup and token counts live there rather than being "
        "pinned in prose",
    "configs/sft_14b_full.resolved.json":
        "written by the resolve step in docs/SFT_READINESS.md's own launch command",
    "data/full14b/coevolve_wins.jsonl":
        "quoted in configs/README.md as the superseded value of "
        "coevolve_distill_path; the shipped root is data/b05factory",
    "data/b05factory/coevolve_wins.jsonl":
        "GRPO distill sink; created by the run",
    "data/b05factory/opus_scores.json":
        "co-evolution score cache; populated by the run's first step",
    # ---- quoted as historic defects, deliberately kept in the text --------- #
    "data/sft/multicap.jsonl":
        "the path nothing produces; quoted in docs/SFT_READINESS.md Blocker 1 as "
        "the defect it was",
    "data/sft/": "same defect, quoted as prose",
    "data/synthetic.py":
        "explicitly described in docs/DATASET_SPEC.md as removed",
    "kore/verifier/test.py":
        "deleted; docs/DATASET_SPEC.md names it only to say correctness moved to "
        "kore/tasks/_genops.py and kore/env/kore_env.py",
    "kore/verifier/bench.py": "deleted; same note as kore/verifier/test.py",
    # ---- external repositories, not part of this checkout ------------------ #
    "repos/vllm": "external checkout (docs/KORE_BENCH_BLUEPRINT.md source list)",
    "repos/GEAK-eval": "external checkout",
    "repos/KernelBench": "external checkout",
    "repos/KernelForge-main": "external checkout",
    "docs/model_ops_guide.md":
        "ROCm/ATOM's doc, always cited with that repo as the prefix in "
        "docs/KORE_BENCH_BLUEPRINT.md",
}

#: Anything containing one of these is a glob, a placeholder, or a dotted symbol
#: reference rather than a path, so it is not resolvable and not checked.
_UNRESOLVABLE = ("*", "{", "}", "<", ">", "…", "...")

#: Directory prefixes that make a token look like a repo-relative path.
#:
#: ``runs/`` and ``logs/`` are deliberately excluded: they are gitignored trainer
#: output, so every reference to one is a path that exists only while some run's
#: output is kept, and checking them would fail on a cleaned-up checkout rather
#: than on a documentation defect. The claims that matter about a run directory
#: -- "this stage writes here", "the next stage reads that" -- are semantic, and
#: are covered by comparing documented `output_dir` values to the configs instead.
_PATH_ROOTS = ("kore/", "tests/", "scripts/", "configs/", "docs/", "data/",
               "repos/", "figures/")

_PATH_RE = re.compile(
    r"(?<![\w./-])((?:" + "|".join(re.escape(r) for r in _PATH_ROOTS) +
    r")[A-Za-z0-9_./*{}<>,-]+)"
)


def _path_references(doc: Path) -> set[str]:
    """Repo-relative-looking path tokens named in ``doc``."""
    found: set[str] = set()
    for match in _PATH_RE.finditer(doc.read_text(encoding="utf-8")):
        token = match.group(1).rstrip(".,;:)`'\"")
        if any(bad in token for bad in _UNRESOLVABLE):
            continue
        found.add(token)
    return found


def _resolves(doc: Path, token: str) -> bool:
    """A markdown reference resolves from the repo root or the doc's directory.

    Both are legitimate: package READMEs link relatively (``kore/README.md``
    naming ``data/README.md`` means ``kore/data/README.md``), while top-level docs
    use repo-relative paths.
    """
    if (REPO_ROOT / token).exists():
        return True
    if (doc.parent / token).exists():
        return True
    # A module reference without the .py suffix, e.g. `kore/tasks/_genops`.
    if (REPO_ROOT / (token + ".py")).exists():
        return True
    # A dotted symbol hung off a real module, e.g.
    # `scripts/run_campaign.py._stage_eval`.
    head = token.split(".py", 1)[0] + ".py"
    return ".py." in token and (REPO_ROOT / head).exists()


def test_every_path_named_in_a_doc_exists_or_is_explained():
    """A doc that names a path nothing produces is the worst kind of wrong.

    ``docs/DISTRIBUTED.md`` shipped ``configs/midtrain_14b_full.json`` pointing at
    an 11 MB development stub while the real corpus is 683 MB, and
    ``docs/SFT_READINESS.md`` gave a launch command resolving a config against a
    checkpoint directory nothing writes. Both looked authoritative.
    """
    unexplained: list[str] = []
    for doc in DOCS:
        for token in sorted(_path_references(doc)):
            if _resolves(doc, token):
                continue
            if token in ABSENT_PATHS or token.rstrip("/") in ABSENT_PATHS:
                continue
            if any(token.startswith(prefix + "/") for prefix in ABSENT_PATHS):
                continue
            unexplained.append(f"{_rel(doc)}: {token}")

    assert not unexplained, (
        "these docs name paths that do not exist. Fix the reference, or add it to "
        "ABSENT_PATHS with the reason (cluster-only / external / run output):\n- "
        + "\n- ".join(unexplained)
    )


def test_absent_path_allowlist_has_no_dead_entries():
    """An allowlist entry for a path that now exists hides the next real absence."""
    stale = [
        token
        for token in ABSENT_PATHS
        if (REPO_ROOT / token).exists()
        # These legitimately come and go: `reassemble.sh` materializes the
        # cluster corpora locally, the run writes its own sinks, and the launch
        # command writes its own resolved config.
        and not token.startswith((
            "data/b05factory/", "data/full14b/", "configs/sft_14b_full.resolved"))
    ]
    assert not stale, f"ABSENT_PATHS entries that now resolve: {stale}"


# --------------------------------------------------------------------------- #
# 2. documented config values must equal the shipped configs
# --------------------------------------------------------------------------- #
#: (doc, fenced-block language, config the block claims to describe, dataclass).
DOCUMENTED_CONFIG_BLOCKS = (
    ("docs/DISTRIBUTED.md", "json", "configs/sft_coder30b_a3b.json", "SFTConfig"),
    ("docs/GRPO_READINESS.md", "jsonc", "configs/grpo_14b_full.json", "GRPOConfig"),
)


def _fenced_block(text: str, language: str) -> str:
    match = re.search(rf"```{language}\n(.*?)\n```", text, re.DOTALL)
    assert match, f"no ```{language} block found"
    return match.group(1)


def _parse_jsonc(body: str) -> dict:
    """JSON with ``//`` line comments stripped (outside strings)."""
    cleaned: list[str] = []
    for line in body.splitlines():
        in_string = False
        escaped = False
        cut = len(line)
        for i, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = not in_string
            elif ch == "/" and not in_string and line[i:i + 2] == "//":
                cut = i
                break
        cleaned.append(line[:cut].rstrip())
    return json.loads("\n".join(cleaned))


@pytest.mark.parametrize(
    "doc_rel,language,config_rel,dataclass_name", DOCUMENTED_CONFIG_BLOCKS
)
def test_documented_config_block_matches_the_shipped_config(
    doc_rel, language, config_rel, dataclass_name
):
    """The doc is the pin: every value it quotes must equal the live config.

    ``configs/README.md`` transcribed ``physics_shaping_weight: 0.15``,
    ``value_prefilter: true`` and ``coevolve_distill_path:
    data/full14b/coevolve_wins.jsonl`` after all three had changed, and
    ``docs/DISTRIBUTED.md`` labelled a block "as actually shipped" while its
    ``dataset_path`` named a file nothing produces and its ``max_seq_length`` was
    the value that dropped half the math slice.

    Nothing is duplicated in this test, so the only way to satisfy it is to keep
    the prose true.
    """
    documented = _parse_jsonc(_fenced_block(_read(doc_rel), language))
    shipped = _settings(config_rel)

    mismatches = [
        f"{key}: doc says {value!r}, {config_rel} has "
        f"{shipped.get(key, '<ABSENT>')!r}"
        for key, value in documented.items()
        if key not in shipped or shipped[key] != value
    ]
    assert not mismatches, (
        f"{doc_rel} has drifted from {config_rel}:\n- " + "\n- ".join(mismatches)
    )


@pytest.mark.parametrize(
    "doc_rel,language,config_rel,dataclass_name", DOCUMENTED_CONFIG_BLOCKS
)
def test_documented_config_keys_are_real_dataclass_fields(
    doc_rel, language, config_rel, dataclass_name
):
    """A doc must not advertise a knob the config dataclass does not have.

    Some keys in a launch JSON are deliberately consumed by the launcher rather
    than the dataclass (``fsdp*``, ``zero_stage``, ``synced_gpus``, ...), so those
    are exempted by name; everything else must be a field.
    """
    from kore.policy import configs as policy_configs

    cls = getattr(policy_configs, dataclass_name)
    fields = {f.name for f in dataclasses.fields(cls)}
    launcher_owned = {
        "fsdp", "fsdp_version", "fsdp_transformer_layer_cls", "fsdp_cpu_offload",
        "zero_stage", "synced_gpus", "cpu_offload",
        # Consumed by the model-identity split rather than the stage dataclass:
        # `sft_config_from_dict` returns (config, model_spec) and the revision
        # travels in the second.
        "model_revision",
    }

    documented = _parse_jsonc(_fenced_block(_read(doc_rel), language))
    unknown = sorted(set(documented) - fields - launcher_owned)
    assert not unknown, (
        f"{doc_rel} documents keys that are not {dataclass_name} fields "
        f"(and are not launcher-owned): {unknown}"
    )


def test_grpo_config_comment_keys_name_a_real_knob():
    """Every ``_comment_<field>`` justifies a knob that actually exists.

    The shipped GRPO recipe explains each deliberately-off lever in a
    ``_comment_<field>`` key. A comment naming a renamed or deleted knob is a
    justification for nothing, and reads as though the lever is still governed.

    A comment may name a field the JSON deliberately leaves unset -- that is a
    documented omission, not drift -- so the field only has to exist on
    ``GRPOConfig``.
    """
    from kore.policy.configs import GRPOConfig

    raw = _load_json("configs/grpo_14b_full.json")
    known = {k for k in raw if not k.startswith("_")}
    known |= {f.name for f in dataclasses.fields(GRPOConfig)}

    orphans = sorted(
        key for key in raw
        if key.startswith("_comment_") and key[len("_comment_"):] not in known
    )
    assert not orphans, (
        "configs/grpo_14b_full.json has _comment_ keys naming no real knob: "
        f"{orphans}"
    )


# --------------------------------------------------------------------------- #
# 3. counts stated in prose must match the artifact they describe
#
# Pinned deliberately (see tests/test_task_integrity_gates.py): these are
# measurements, so changing an artifact must force an explicit edit here AND in
# the prose, rather than silently invalidating the prose.
# --------------------------------------------------------------------------- #
# Task registry composition. 63 + 4 + 33 + 6 = 106 vendor-lane tasks; the
# remaining 1,248 (92%) are torch-anchored. The total includes the 20-task HIP C++
# family, all of which are torch-baselined (their production baseline is the eager
# torch path they have to beat).
EXPECTED_TOTAL_TASKS = 1_522
EXPECTED_GENB_TASKS = 1_052
EXPECTED_AITER_DECLARED = 63
EXPECTED_HIPBLASLT_DECLARED = 4
# 1,279 + the 188-task HIP C++ family, every one of which declares a torch or
# torch_compile bar: no HIP task is graded against a vendor kernel, because the
# vendor GEMM is the one baseline whose own CV cannot clear the publication gate.
EXPECTED_TORCH_DECLARED = 1_447
EXPECTED_GEMM_FUSION_UPGRADED = 33
EXPECTED_GATED_ACT_UPGRADED = 6
EXPECTED_VENDOR_LANE = 106
# Metamorphic prong coverage: tasks whose operator contract is fixed by the
# _genops generator spec, so a proven relation set exists.
EXPECTED_METAMORPHIC_PLANNED = 168
# data/gfx950_task_verification.json, measured on real gfx950.
EXPECTED_BREADTH_PASS = 1_052
# Cluster-verified corpus sizes (data/b05factory/** is not present locally).
EXPECTED_MIDTRAIN_ROWS = 86_010
EXPECTED_SFT_ROWS = 56_493
EXPECTED_DPO_PAIRS = 96_675
# tests/test_sft_launch_readiness.py, default suite (excludes the release test).
EXPECTED_SFT_READINESS_TESTS = 36


def _registry_baseline_counts() -> dict[str, int]:
    from kore.tasks import _genops
    from kore.tasks import registry

    tasks = registry.all_tasks()
    declared = [str(getattr(t, "comparison_baseline", "")) for t in tasks]
    generated = _genops._registry()

    gemm_fusion = 0
    gated = 0
    for task in tasks:
        entry = generated.get(task.operation)
        if not entry:
            continue
        family = entry[0]
        kind = _genops._vendor_baseline_kind(task.operation, family, task.dtype)
        if kind != "vendor":
            continue
        # Already counted by its declaration; only the runtime UPGRADES are new.
        if declared[tasks.index(task)].startswith(("aiter", "hipblaslt")):
            continue
        if family == "gemm_fusion":
            gemm_fusion += 1
        elif family == "fusion":
            gated += 1

    return {
        "total": len(tasks),
        "genb": sum(1 for t in tasks if t.task_id.startswith("genb_")),
        "aiter": sum(1 for x in declared if x.startswith("aiter")),
        "hipblaslt": sum(1 for x in declared if x.startswith("hipblaslt")),
        "torch": sum(1 for x in declared if x.startswith("torch")),
        "gemm_fusion_upgraded": gemm_fusion,
        "gated_act_upgraded": gated,
    }


def test_task_registry_composition_is_what_the_docs_claim():
    """The baseline-lane split is the single most cited number in this repo.

    ``README.md``, ``docs/DATASET_SPEC.md`` and ``kore/tasks/README.md`` all state
    it, and all three were off by one in both directions (64/3 against a real
    63/4) while claiming "2 gated activations" where there are 6 tasks.
    """
    counts = _registry_baseline_counts()

    assert counts["total"] == EXPECTED_TOTAL_TASKS
    assert counts["genb"] == EXPECTED_GENB_TASKS
    assert counts["aiter"] == EXPECTED_AITER_DECLARED
    assert counts["hipblaslt"] == EXPECTED_HIPBLASLT_DECLARED
    assert counts["torch"] == EXPECTED_TORCH_DECLARED
    assert counts["gemm_fusion_upgraded"] == EXPECTED_GEMM_FUSION_UPGRADED
    assert counts["gated_act_upgraded"] == EXPECTED_GATED_ACT_UPGRADED

    vendor = (counts["aiter"] + counts["hipblaslt"]
              + counts["gemm_fusion_upgraded"] + counts["gated_act_upgraded"])
    assert vendor == EXPECTED_VENDOR_LANE
    # The "remaining ~93%" figure in README.md / DATASET_SPEC.md. The vendor lane
    # is a fixed 106 tasks, so growing the HIP family dilutes the share; the prose
    # has to move with it.
    breadth_share = (counts["total"] - vendor) / counts["total"]
    assert 0.925 <= breadth_share <= 0.935, breadth_share


@pytest.mark.parametrize("doc_rel", ["README.md", "docs/DATASET_SPEC.md"])
def test_baseline_lane_prose_states_the_measured_counts(doc_rel):
    """The prose has to carry the real numbers, not just be near them."""
    text = _read(doc_rel)
    for number in (
        f"{EXPECTED_TOTAL_TASKS:,}",
        f"{EXPECTED_GENB_TASKS:,}",
        str(EXPECTED_AITER_DECLARED),
        str(EXPECTED_HIPBLASLT_DECLARED),
        str(EXPECTED_GEMM_FUSION_UPGRADED),
        str(EXPECTED_VENDOR_LANE),
    ):
        assert number in text, (
            f"{doc_rel} does not state {number!r}; the baseline-lane counts "
            "changed and the prose was not updated"
        )


def test_breadth_verification_artifact_matches_the_docs():
    """``data/gfx950_task_verification.json`` is 1,052/1,052 PASS."""
    artifact = _load_json("data/gfx950_task_verification.json")
    results = artifact["results"]
    passing = [r for r in results if r.get("status") == "PASS"]

    assert len(results) == EXPECTED_BREADTH_PASS
    assert len(passing) == EXPECTED_BREADTH_PASS
    assert artifact["summary"]["counts"] == {"PASS": EXPECTED_BREADTH_PASS}


def test_metamorphic_prong_coverage_matches_the_readmes():
    """``kore/verify/README.md`` and ``README.md`` both state 168 planned tasks."""
    from kore.tasks import registry
    from kore.verify.production import metamorphic_plan_for_task

    planned = sum(
        1 for task in registry.all_tasks()
        if metamorphic_plan_for_task(task).applicable
    )
    assert planned == EXPECTED_METAMORPHIC_PLANNED

    for doc_rel in ("README.md", "kore/verify/README.md"):
        assert str(EXPECTED_METAMORPHIC_PLANNED) in _read(doc_rel), doc_rel


def test_dataset_status_states_the_cluster_verified_counts():
    """``DATASET_STATUS.md`` row counts, verified on SPUR.

    The corpora live only on the cluster, so this pins the numbers rather than
    reading the files. Re-verify with::

        ssh amd-spur-tunnel 'cd ~/Kore-RL/KORE && wc -l data/b05factory/*/*.jsonl'
    """
    text = _read("DATASET_STATUS.md")
    for rows in (EXPECTED_MIDTRAIN_ROWS, EXPECTED_SFT_ROWS, EXPECTED_DPO_PAIRS):
        assert f"{rows:,}" in text, (
            f"DATASET_STATUS.md no longer states {rows:,}; re-verify on the "
            "cluster and update both the doc and this test"
        )
    # The superseded snapshot must keep saying so rather than contradicting it.
    snapshot = _read("data/DATASET_STATUS.md")
    assert "SUPERSEDED" in snapshot
    assert f"{EXPECTED_SFT_ROWS:,}" in snapshot


def test_sft_readiness_states_its_own_test_count():
    """Keep the live SFT readiness regression inventory visible in its guide."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider",
         "tests/test_sft_launch_readiness.py"],
        cwd=REPO_ROOT,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    match = re.search(
        r"tests/test_sft_launch_readiness\.py: (\d+)", result.stdout)
    assert match, f"could not read the collected count:\n{result.stdout[-2000:]}"
    collected = int(match.group(1))

    assert collected == EXPECTED_SFT_READINESS_TESTS, (
        f"tests/test_sft_launch_readiness.py now collects {collected} tests in "
        f"the default suite, not {EXPECTED_SFT_READINESS_TESTS}; update "
        "EXPECTED_SFT_READINESS_TESTS and docs/SFT_READINESS.md together"
    )
    text = _read("docs/SFT_READINESS.md")
    assert f"{EXPECTED_SFT_READINESS_TESTS} pass" in text, (
        "docs/SFT_READINESS.md must state the collected count, currently "
        f"{collected}"
    )


def test_p0_peak_table_matches_the_calibration_artifact():
    """``docs/P0_RESULTS.md``'s peak table quoted a superseded calibration file.

    It said HBM 4.60 TB/s at 57% and bf16 1.27 PF/s at 51% against a 2.5 PF/s
    datasheet, while the studies it documents were run with
    ``data/calibration_v1.json`` -- 4.763 TB/s at 60%, 1.296 PF/s at 56%, against
    the MI350X 2.3 PF/s datasheet. The same document stated the correct pair four
    sections earlier.
    """
    calibration = _load_json("data/calibration_v1.json")
    text = _read("docs/P0_RESULTS.md")

    hbm_tb = calibration["measured"]["hbm_bytes_per_s"] / 1e12
    bf16_pf = calibration["measured"]["bf16_flops_per_s"] / 1e15
    hbm_share = calibration["measured_over_datasheet"]["hbm"]
    bf16_share = calibration["measured_over_datasheet"]["bf16"]

    for value in (f"{hbm_tb:.3f}", f"{bf16_pf:.3f}"):
        assert value in text, f"docs/P0_RESULTS.md does not state {value}"
    for share in (hbm_share, bf16_share):
        assert f"{round(share * 100)}%" in text, (
            f"docs/P0_RESULTS.md does not state {round(share * 100)}% attained"
        )
    # The datasheet column: MI350X, not MI355X. Dividing by an MI355X 2.5 PF/s
    # ceiling lowers the speed-of-light integrity floor by 8% and admits
    # physically impossible timings, so the table must not quote it as ours.
    datasheet_pf = calibration["datasheet"]["bf16_flops_per_s"] / 1e15
    assert f"{datasheet_pf:.1f} PF/s" in text
    stale_datasheet = [
        f"line {lineno}: {sentence.strip()[:120]}"
        for lineno, sentence in _sentences(text)
        if "2.5 PF/s" in sentence and not _RETRACTION_MARKERS.search(sentence)
        and "MI355X" not in sentence
    ]
    assert not stale_datasheet, (
        "docs/P0_RESULTS.md quotes the MI355X 2.5 PF/s bf16 datasheet peak as if "
        f"it were this node's; rooflines.DEFAULT_SKU pins mi350x at "
        f"{datasheet_pf:.1f} PF/s:\n- " + "\n- ".join(stale_datasheet)
    )


def test_p0_calibration_fingerprint_in_the_reproduce_command_is_current():
    """The documented ``--expect-model-fingerprint`` must be the one that matches.

    ``data/calibration_v1.json`` was reissued when the model fingerprint widened
    from v1 to v2, which left the reproduce command in ``docs/P0_RESULTS.md``
    exiting 1 with ``physical-model fingerprint mismatch``.
    """
    fingerprint = _load_json("data/calibration_v1.json")["model_fingerprint"]
    text = _read("docs/P0_RESULTS.md")
    command = re.search(
        r"--expect-model-fingerprint (sha256:[0-9a-f]+)", text
    )
    assert command, "no --expect-model-fingerprint in docs/P0_RESULTS.md"
    assert command.group(1) == fingerprint, (
        f"docs/P0_RESULTS.md's reproduce command pins {command.group(1)}, but "
        f"data/calibration_v1.json carries {fingerprint}"
    )


# --------------------------------------------------------------------------- #
# 4. retracted claims must not come back
# --------------------------------------------------------------------------- #
#: Wording that marks a sentence as *retracting* a claim rather than making it.
#: A retraction has to name the figure to be readable, so the checks below are
#: sentence-scoped and skip anything carrying one of these.
_RETRACTION_MARKERS = re.compile(
    r"earlier revision|previously reported|retract|withdraw|superseded|"
    r"not validated|is not a validated|does not survive|artifact|must not|"
    r"no longer|shared[- ]denominator|in-sample only|is not evidence|"
    r"falsif|would lower|disagreed with|reads as|an earlier",
    re.IGNORECASE,
)

#: Phrasings that assert the figure rather than retracting it.
_VALIDATED_R2_PATTERNS = (
    r"validated (?:offline )?(?:at|to|by) [Rr].{0,3}\s*.{0,3}\s*0\.9",
    r"validated [Rr].{0,3}.{0,3}0\.9",
    r"[Rr].{0,3}\s*.{0,3}\s*0\.98[^.\n]{0,30}is the validated",
    r"the [Rr].{0,3}\s*.{0,3}\s*0\.98 residual-decomposition result",
    r"reconstructs the runtime residual with [Rr].{0,3}\s*.{0,3}\s*0\.98",
)


def _sentences(text: str):
    """(line number, sentence) pairs, so a check can be sentence-scoped.

    Markdown table cells are split too: a row like ``| ... cross-validated R2 |
    **0.978** |`` would otherwise read as one sentence asserting the second cell
    of the first, which is how a table of *controls* looked like a claim.
    """
    for lineno, line in enumerate(text.splitlines(), start=1):
        for cell in line.split("|"):
            for sentence in re.split(r"(?<=[.!?;])\s+", cell):
                if sentence.strip():
                    yield lineno, sentence


_SHELL_FENCE = re.compile(r"```(?:bash|sh|shell|console)\n(.*?)\n```", re.DOTALL)


def _shell_blocks(text: str):
    """(first line number, body) for every fenced shell block.

    Commands are only checked inside these: prose that names a script, and
    ``run_campaign.py:2130``-style source citations, are not invocations.
    """
    for match in _SHELL_FENCE.finditer(text):
        yield text[:match.start(1)].count("\n") + 1, match.group(1)


def test_the_retracted_r2_result_is_not_asserted_anywhere():
    """``R2 ~ 0.98`` as a *validated* result is retracted; it kept coming back.

    It was asserted in seven separate READMEs after ``docs/P0_RESULTS.md``
    withdrew it. The figure is reproducible but is a shared-denominator artifact:
    a ``T_candidate``-only predictor scores 0.997, and on the preregistered
    normalized target over held-out clusters the named model scores -0.458.

    Mentioning it is fine -- asserting it as validated is not, so this matches
    per sentence and exempts sentences that carry retraction wording.
    """
    offenders: list[str] = []
    for doc in DOCS:
        for lineno, sentence in _sentences(doc.read_text(encoding="utf-8")):
            if _RETRACTION_MARKERS.search(sentence):
                continue
            for pattern in _VALIDATED_R2_PATTERNS:
                match = re.search(pattern, sentence)
                if match:
                    offenders.append(
                        f"{_rel(doc)}:{lineno}: {match.group(0)!r}")

    assert not offenders, (
        "the retracted R2 ~ 0.98 residual-decomposition result is asserted as "
        "validated. It is not; see docs/P0_RESULTS.md (verdict INTEGRITY_ONLY, "
        "all three checks FAIL, no authorized family):\n- " + "\n- ".join(offenders)
    )


def test_docs_only_quote_p0_verdicts_the_adjudicator_can_emit():
    """``PARTIAL`` / ``FALLBACK`` / ``PIVOT`` are from a superseded revision.

    ``kore/analysis/README.md`` reported "verdict **PARTIAL**" long after
    ``p0_sol.decide`` stopped being able to return it, which made a falsified
    result read as a qualified pass.
    """
    from kore.analysis import p0_sol

    source = Path(p0_sol.__file__).read_text(encoding="utf-8")
    decide = source.split("def decide(", 1)[1].split("\ndef ", 1)[0]
    emittable = set(re.findall(r'return "([A-Z_]+)"', decide))
    assert emittable, "could not read p0_sol.decide's return values"

    # Verdict strings a superseded revision used, that decide() cannot return.
    retired = {"PARTIAL", "FALLBACK", "PIVOT"} - emittable
    offenders: list[str] = []
    for doc in DOCS:
        for lineno, sentence in _sentences(doc.read_text(encoding="utf-8")):
            if _RETRACTION_MARKERS.search(sentence):
                continue
            for verdict in sorted(retired):
                # Only flag it where the text calls it a verdict/decision.
                match = re.search(
                    rf"verdict\W{{0,4}}\*{{0,2}}{verdict}\b", sentence, re.IGNORECASE)
                if match:
                    offenders.append(f"{_rel(doc)}:{lineno}: {match.group(0)!r}")

    assert not offenders, (
        f"these docs quote a P0 verdict p0_sol.decide cannot emit "
        f"(it returns {sorted(emittable)}):\n- " + "\n- ".join(offenders)
    )


def test_no_doc_presents_the_dead_peak_overrides_as_effective():
    """``KORE_PEAK_*`` are a no-op; two docs told the reader to set them.

    ``resolve_peaks`` ignores them and raises a ``RuntimeWarning`` naming any that
    are exported, so "set on-node ``KORE_PEAK_BF16``" was advice that silently did
    nothing -- and the value it was meant to fix (eta being ~1.7x off) stayed
    wrong.
    """
    from kore.analysis.rooflines import LEGACY_PEAK_ENV_VARS

    # The vars really are inert: datasheet peaks come back regardless.
    assert LEGACY_PEAK_ENV_VARS  # named, so the warning can be emitted

    imperative = re.compile(
        r"(?:^|[^a-z])(?:set|export|setting|use|pass)\b[^.\n]{0,60}"
        r"KORE_PEAK_(?:BF16|FP8|HBM_BW)",
        re.IGNORECASE,
    )
    # Sentences that make the deadness explicit are the point, not the problem.
    exonerating = re.compile(
        r"no[- ]op|dead|no effect|does nothing|ignored|unsupported|removed|"
        r"rejected|RuntimeWarning|has NO effect|to no effect|deliberately",
        re.IGNORECASE,
    )

    offenders: list[str] = []
    for doc in DOCS:
        for lineno, line in enumerate(
            doc.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if imperative.search(line) and not exonerating.search(line):
                offenders.append(f"{_rel(doc)}:{lineno}: {line.strip()[:140]}")

    assert not offenders, (
        "these lines tell the reader to set a KORE_PEAK_* override, which does "
        "nothing. Point at a kore.runtime-calibration.v1 document instead:\n- "
        + "\n- ".join(offenders)
    )


# --------------------------------------------------------------------------- #
# 5. deprecated entrypoints must not be documented as the way to run something
# --------------------------------------------------------------------------- #
def _deprecated_scripts() -> set[str]:
    registry = _load_json("scripts/operations_registry.json")
    return {
        entry["path"]
        for entry in registry["scripts"]
        if entry.get("lifecycle") == "deprecated"
    }


def test_operations_registry_paths_all_exist():
    """The registry is the authority on lifecycle; it cannot name ghosts."""
    registry = _load_json("scripts/operations_registry.json")
    missing = [
        entry["path"] for entry in registry["scripts"]
        if not (REPO_ROOT / entry["path"]).exists()
    ]
    assert not missing, f"operations_registry.json names absent scripts: {missing}"


def test_no_doc_recommends_a_deprecated_entrypoint():
    """A deprecated script exits 64; documenting it as the path is a dead end.

    ``scripts/README.md`` headed a section "Running the full campaign
    (recommended path)" over ``bash scripts/tmux_campaign.sh``, which prints
    ``is deprecated and disabled for production`` and exits 64.
    """
    deprecated = _deprecated_scripts()
    assert deprecated, "no deprecated scripts in the registry -- check the parse"

    # Words that turn a mention into an instruction.
    recommending = re.compile(
        r"recommended|production path|use this|the way to|start with|"
        r"^\s*(?:bash|sh|python)\s", re.IGNORECASE
    )
    # Words that mark it as history or a dev-only escape hatch.
    exonerating = re.compile(
        r"deprecat|disabled|exits 64|KORE_ALLOW_DEPRECATED_DEV|superseded|"
        r"earlier revision|not portable|no effect|history", re.IGNORECASE
    )

    offenders: list[str] = []
    for doc in DOCS:
        lines = doc.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            named = [s for s in deprecated if Path(s).name in line]
            if not named or not recommending.search(line):
                continue
            # A nearby caveat counts: a fenced example under a paragraph that
            # already says "deprecated" is not a false recommendation.
            window = "\n".join(lines[max(0, lineno - 12):lineno + 3])
            if exonerating.search(window):
                continue
            offenders.append(f"{_rel(doc)}:{lineno}: {line.strip()[:140]}")

    assert not offenders, (
        "these lines present a deprecated entrypoint as a way to run something. "
        "Point at the scheduler path (scripts/spur_*) and say the wrapper is "
        "deprecated:\n- " + "\n- ".join(offenders)
    )


# --------------------------------------------------------------------------- #
# 6. the test suite's own map must stay complete
# --------------------------------------------------------------------------- #
#: Modules whose whole purpose is to freeze a number, a path, or a behavioural
#: contract. An undocumented one is a real hazard: when the thing it pins
#: changes, the failure is the only pointer to it, and nobody knows to look at
#: the map. Ordinary unit-test modules are deliberately NOT required here -- see
#: the note in the docstring.
_PIN_BEARING = re.compile(
    r"test_.*(?:contract|integrity|honesty|readiness|_gates|registry|"
    r"provenance|durability|certification|protocol)\w*\.py$"
)


def test_tests_readme_names_every_pin_bearing_test_module():
    """An unlisted contract test is one nobody looks at when its pin changes.

    ``tests/README.md``'s coverage map named 74 modules while 123 existed, so
    every contract test added over the previous months -- dataloader, packaging,
    task integrity, operations registry, GRPO capabilities -- was invisible.

    Only the pin-bearing modules are *required* to appear, because requiring every
    module would make this test fail on any newly added unit test rather than on a
    documentation defect. The reverse direction is checked universally: the map may
    not name a module that no longer exists.
    """
    named = set(re.findall(r"`(test_[A-Za-z0-9_]+\.py)`", _read("tests/README.md")))
    on_disk = {path.name for path in (REPO_ROOT / "tests").glob("test_*.py")}
    pins = {name for name in on_disk if _PIN_BEARING.search(name)}

    assert not (pins - named), (
        "tests/README.md's coverage map does not mention these contract tests: "
        f"{sorted(pins - named)}"
    )
    assert not (named - on_disk), (
        "tests/README.md names test modules that no longer exist: "
        f"{sorted(named - on_disk)}"
    )


# --------------------------------------------------------------------------- #
# 7. documented commands: the fail-closed campaign contract
# --------------------------------------------------------------------------- #
def test_campaign_invocations_in_docs_carry_a_campaign_mode():
    """``run_campaign.py`` is fail-closed: without a mode flag it exits 1.

    Three documented invocations (two in ``README.md``, one in each of the
    Quick-start and Resume sections) omitted ``--use-hf`` and so exited 1 with
    ``missing --use-hf (full retention sources are mandatory)`` -- printing no
    plan at all, which is exactly what they claimed to do.
    """
    required = ("--use-hf", "--campaign-mode", "--help")
    offenders: list[str] = []
    for doc in DOCS:
        for lineno, block in _shell_blocks(doc.read_text(encoding="utf-8")):
            # One command per backslash-joined logical line.
            for offset, command in enumerate(
                block.replace("\\\n", " ").splitlines()
            ):
                if "run_campaign.py" not in command:
                    continue
                if command.lstrip().startswith("#"):
                    continue
                if any(flag in command for flag in required):
                    continue
                offenders.append(
                    f"{_rel(doc)}:~{lineno + offset}: {command.strip()[:140]}")

    assert not offenders, (
        "these documented run_campaign.py invocations omit --use-hf and "
        "--campaign-mode, so they exit 1 without doing anything:\n- "
        + "\n- ".join(offenders)
    )


def test_release_prerequisites_do_not_claim_the_license_is_unchosen():
    """The licensing gap is closed; ``README.md`` said it was open.

    ``LICENSE`` declares the repository proprietary and AMD-internal, and telling
    a reader an owner still has to "select the license" invites exactly the wrong
    action.
    """
    assert (REPO_ROOT / "LICENSE").is_file()
    assert (REPO_ROOT / "THIRD_PARTY.md").is_file()

    section = _read("README.md").split("## Release prerequisites", 1)
    assert len(section) == 2, "README.md has no Release prerequisites section"
    body = section[1].split("\n## ", 1)[0]

    forbidden = (
        "does not yet contain owner-approved licensing",
        "until an authorized owner has selected the license",
    )
    for phrase in forbidden:
        assert phrase not in body, (
            f"README.md still says {phrase!r}, but LICENSE exists and pyproject "
            "pins the matching metadata"
        )
    assert "LICENSE" in body and "THIRD_PARTY" in body
