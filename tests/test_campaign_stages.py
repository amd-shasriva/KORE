"""CPU-only tests for the five campaign stages that had NO coverage at all.

``tests/test_campaign_wiring.py`` covers midtrain/sft/dpo/grpo/evolve. The stages
below were never exercised by a test, and they are the ones that decide what the
run trains on and what it is allowed to claim:

  * ``_stage_build``   - takes the AUTHORITATIVE registry train/held-out split and
    assembles the train-only SFT mix + the DPO pairs. Every later stage inherits
    whatever it emits, so a leak here is unrecoverable. This repo has a MEASURED
    contamination history (11 held-out tasks whose optimized kernel source reached
    the mid-train corpus), which is why section 1 spends most of its effort proving
    that a held-out record cannot reach either training product.
  * ``_stage_soup`` / ``_stage_eval`` - the two stages that publish a CLAIM. Their
    gates are only worth as much as their strictness, so sections 2 and 3 build the
    adversarial cases (kernel improves but a general metric regresses; kernel flat;
    a general metric missing; a smoke retention source; a bundled KernelBench
    fallback) and assert the promotion is REFUSED.
  * ``_stage_datagen`` / ``_stage_agentic`` (+ ``_stage_reverify``) - sections 4-6:
    only TRAIN tasks are generated/synthesized/re-verified, synthesis reads only
    VERIFIED records, and a failure cannot masquerade as empty-but-successful.

Everything heavy is stubbed the same way the wiring suite stubs it: no GPU, no
teacher, no torch, no trained checkpoint. Where a stage's real decision lives in a
pure helper (``soup_sweep_materialized``, ``StageGate``, the artifact contract),
the helper is driven directly as well, so a passing test means the DECISION was
made correctly and not merely that a stub was called.

Two real defects were found while writing this and are pinned here:
``test_build_refuses_a_dpo_set_below_the_hard_negative_floor`` (the >=8% floor was
computed and logged but never enforced) and
``test_sequential_datagen_publishes_a_resumable_receipted_shard`` (the sequential
datagen path wrote shards that its own resume check and the production build
reader both reject). Both are fixed in ``scripts/run_campaign.py``.

A third, latent one is deliberately NOT asserted as behavior, because it is
currently unreachable: ``_stage_build`` reads its raw records with ``typed=True``,
and the typed dataclasses drop the ``provenance_root`` / ``_provenance`` keys that
``registry.is_heldout_record`` uses for the ``heldout_lineage`` decision. Today
every generated record belongs to a REGISTERED task, so the root is resolved from
the registry instead and nothing leaks - the registry currently has zero lineage
descendants. ``test_the_campaign_holdout_filter_is_the_registry_authority_verbatim``
pins the authority itself so the gap is visible if records ever start carrying
their own provenance root.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import scripts.run_campaign as rc
from kore.data.schemas import (
    RankedGroupRecord,
    RepairRecord,
    WinRecord,
    stamp_production_record,
    write_jsonl,
)
from kore.tasks.registry import (
    get_task,
    heldout_tasks,
    is_heldout_record,
    split_decision,
)

# The two campaign-selected TRAIN tasks every fixture below uses.
TRAIN_IDS = ["rmsnorm_aiter", "gemm_bf16"]
# A marker that only ever appears in HELD-OUT records. Section 1 greps the emitted
# training products for it: the corpus-level assertion, not a filter-level one.
HELDOUT_MARK = "HELDOUT_LEAK_CANARY"


def _heldout_task(reason: str):
    """A held-out task reserved for the given reason, resolved from the registry.

    The two reservation mechanisms fail differently, so both are exercised:
    ``near_probe`` reserves exact task identities (and anything sharing their
    provenance root) while their product family keeps training; ``whole_family``
    reserves an entire product leaf, so even an unregistered variant of it is
    eval-only.
    """
    for task in heldout_tasks():
        if split_decision(task).reason == reason:
            return task
    raise AssertionError(f"the registry reserves no task for reason {reason!r}")


def _args(argv):
    return rc.build_parser().parse_args(argv)


def _no_torch(monkeypatch):
    """Make the stages' opportunistic ``import torch`` a no-op.

    ``_stage_soup``/``_release_model_memory`` call ``torch.cuda.empty_cache()``
    behind ``try/except``. On this box that would touch GPUs other jobs are using,
    so the import is denied outright rather than mocked.
    """
    monkeypatch.setitem(sys.modules, "torch", None)


# --------------------------------------------------------------------------- #
# Record fixtures (typed, production-stamped, distinguishable by source text)
# --------------------------------------------------------------------------- #
def _kernel(tag: str) -> str:
    return (
        "import triton\nimport triton.language as tl\n\n\n"
        f"def {tag}(x):\n"
        f"    # {tag}\n"
        "    return x + 1\n"
    )


def _repair(task_id: str, operation: str, tag: str, arch: str = "gfx950") -> RepairRecord:
    return RepairRecord(
        task_id=task_id,
        failure_class="snr_fail",
        parent_hash=f"parent-{tag}",
        error_text="snr_db = 3.0",
        messages=[
            {"role": "system", "content": "you optimize AMD kernels"},
            {"role": "user",
             "content": f"repair this kernel\n```python\n{_kernel(tag + '_broken')}\n```"},
            {"role": "assistant",
             "content": f"FULL_KERNEL:\n```python\n{_kernel(tag + '_fixed')}\n```"},
        ],
        child_snr_db=45.0,
        operation=operation,
        arch=arch,
        gpu=arch,
    )


def _win(task_id: str, operation: str, tag: str, *, speedup: float = 2.0,
         arch: str = "gfx950") -> WinRecord:
    return WinRecord(
        task_id=task_id,
        trajectory=[
            {"role": "user",
             "content": f"optimize\n```python\n{_kernel(tag + '_seed')}\n```"},
            {"role": "assistant",
             "content": f"FULL_KERNEL:\n```python\n{_kernel(tag + '_final')}\n```"},
        ],
        initial_wall_us=200.0,
        final_wall_us=200.0 / speedup,
        speedup=speedup,
        final_source=_kernel(tag + "_final"),
        snr_db=44.0,
        operation=operation,
        arch=arch,
        gpu=arch,
    )


def _group(task_id: str, operation: str, tag: str, *, n_candidates: int = 4,
           arch: str = "gfx950") -> RankedGroupRecord:
    candidates = [
        {"source": _kernel(f"{tag}_cand{i}"), "wall_us": 100.0 * (i + 1),
         "snr_db": 40.0, "rank": i}
        for i in range(n_candidates)
    ]
    return RankedGroupRecord(
        task_id=task_id,
        parent_id=f"parent-{tag}",
        candidates=candidates,
        preferences=[[0, i] for i in range(1, n_candidates)],
        operation=operation,
        arch=arch,
        gpu=arch,
        baseline_wall_us=150.0,
        baseline_type="aiter",
        baseline_kind="vendor",
    )


def _write_shard(root, sub: str, task_id: str, records) -> None:
    """Publish a production-envelope shard exactly where the build stage looks."""
    rows = [
        stamp_production_record(rec, provenance_id="test-datagen-v1",
                                evaluation_id=f"test:{sub}:{task_id}:{index}")
        for index, rec in enumerate(records)
    ]
    write_jsonl(root / sub / f"{task_id}.jsonl", rows)


def _seed_train_shards(root, task_id: str, operation: str, tag: str) -> None:
    _write_shard(root, "repair", task_id, [_repair(task_id, operation, tag + "_r")])
    _write_shard(root, "wins", task_id, [_win(task_id, operation, tag + "_w")])
    _write_shard(root, "groups", task_id, [_group(task_id, operation, tag + "_g")])


def _seed_heldout_shards(root, task) -> None:
    """Datagen output for a held-out task - the exact thing that must never train."""
    tag = HELDOUT_MARK
    _write_shard(root, "repair", task.task_id,
                 [_repair(task.task_id, task.operation, tag + "_r")])
    _write_shard(root, "wins", task.task_id,
                 [_win(task.task_id, task.operation, tag + "_w")])
    _write_shard(root, "groups", task.task_id,
                 [_group(task.task_id, task.operation, tag + "_g")])


# --------------------------------------------------------------------------- #
# 1. _stage_build - the data-integrity chokepoint
# --------------------------------------------------------------------------- #
def _build_ctx(tmp_path, monkeypatch, *extra_argv, tasks=None):
    """A ctx for ``_stage_build`` with the AUTHORITATIVE split already applied."""
    from kore.data.teacher import StubTeacher

    # The mixer decontaminates the general slices against the retention benchmarks;
    # offline that needs the explicit development reference set.
    monkeypatch.setenv("KORE_DECONTAM_DEVELOPMENT", "1")
    monkeypatch.setattr(rc, "_teacher", lambda args: StubTeacher())
    task_ids = list(tasks or TRAIN_IDS)
    args = _args([
        "--tasks", ",".join(task_ids),
        "--data-root", str(tmp_path),
        "--sft-total", "200",
        "--no-gold-wins", "--no-repair-dpo",
        "--campaign-mode", "development",
        *extra_argv,
    ])
    ctx = {"data_root": tmp_path, "args": args, "dry": False,
           "tasks": [get_task(t) for t in task_ids]}
    rc._apply_split(ctx)
    return ctx


def _products(tmp_path) -> tuple[list[dict], list[dict]]:
    sft = (tmp_path / "sft" / "multicap.jsonl").read_text()
    dpo = (tmp_path / "dpo" / "pairs.jsonl").read_text()
    return ([json.loads(line) for line in sft.splitlines() if line.strip()],
            [json.loads(line) for line in dpo.splitlines() if line.strip()])


def test_build_never_lets_a_heldout_task_reach_either_training_product(
        tmp_path, monkeypatch):
    """The single most consequential invariant in the campaign.

    A whole operator family is reserved eval-only, and this repo has already been
    burned once: 11 held-out tasks had their optimized kernel source leak into the
    mid-train corpus. So this asserts on the EMITTED CORPUS, not on the filter -
    the held-out kernels carry a canary string, and neither the SFT mix nor the DPO
    pairs may contain it anywhere.

    Both reservation mechanisms are seeded, because they are enforced by different
    code paths: a near-generalization probe and a whole reserved product family.
    """
    probe = _heldout_task("near_probe")
    family = _heldout_task("whole_family")
    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    _seed_train_shards(tmp_path, "gemm_bf16", "gemm", "gemm")
    _seed_heldout_shards(tmp_path, probe)
    _seed_heldout_shards(tmp_path, family)

    ctx = _build_ctx(tmp_path, monkeypatch)
    for task in (probe, family):
        assert task.task_id in ctx["eval_task_ids"]
        assert task.task_id not in ctx["train_task_ids"]

    rc._stage_build(ctx)

    sft_rows, dpo_rows = _products(tmp_path)
    sft_text = json.dumps(sft_rows)
    dpo_text = json.dumps(dpo_rows)
    assert HELDOUT_MARK not in sft_text, "held-out kernel source reached the SFT mix"
    assert HELDOUT_MARK not in dpo_text, "held-out kernel source reached the DPO pairs"
    for task in (probe, family):
        assert task.task_id not in sft_text and task.task_id not in dpo_text
    # ...and the run was not vacuously clean: the TRAIN records did get through.
    assert "rms" in sft_text or "gemm" in sft_text
    assert sft_rows and dpo_rows


def test_build_holds_out_a_reserved_family_variant_that_is_not_a_registered_task(
        tmp_path, monkeypatch):
    """A whole product leaf is reserved, so an UNREGISTERED variant of it is still
    eval-only. An id-only filter would let this one straight through."""
    family = _heldout_task("whole_family")
    variant_id = f"{family.task_id}_variant_x"
    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    _write_shard(tmp_path, "wins", variant_id,
                 [_win(variant_id, family.operation, HELDOUT_MARK + "_variant")])

    ctx = _build_ctx(tmp_path, monkeypatch, tasks=["rmsnorm_aiter"])
    # The campaign never named this id, so only the registry's family authority
    # can keep it out.
    assert variant_id not in ctx["eval_task_ids"]
    assert is_heldout_record({"type": "win", "task_id": variant_id,
                              "operation": family.operation, "arch": "gfx950"}) is True

    rc._stage_build(ctx)

    sft_rows, dpo_rows = _products(tmp_path)
    assert HELDOUT_MARK not in json.dumps(sft_rows)
    assert HELDOUT_MARK not in json.dumps(dpo_rows)


def test_the_campaign_holdout_filter_is_the_registry_authority_verbatim():
    """``_rec_is_heldout`` must delegate, never re-implement.

    A near probe reserves its LINEAGE (anything sharing its provenance root), a
    whole family reserves every member, and an unclassifiable identity is eval-only
    by default. All three are registry decisions; the campaign owning a second copy
    of them is how the two drift apart.
    """
    probe = _heldout_task("near_probe")
    family = _heldout_task("whole_family")
    records = [
        {"type": "win", "task_id": probe.task_id, "operation": probe.operation,
         "arch": "gfx950"},
        {"type": "win", "task_id": "fresh_id", "operation": "rmsnorm",
         "arch": "gfx950", "provenance_root": probe.task_id},
        {"type": "win", "task_id": f"{family.task_id}_variant",
         "operation": family.operation, "arch": "gfx950"},
        {"type": "win", "task_id": "rmsnorm_aiter", "operation": "rmsnorm",
         "arch": "gfx1100"},
        {"type": "win", "task_id": "rmsnorm_aiter", "operation": "rmsnorm",
         "arch": "gfx950"},
    ]
    for record in records:
        assert rc._rec_is_heldout(record, set()) is is_heldout_record(record)
    assert [rc._rec_is_heldout(r, set()) for r in records] == [
        True, True, True, True, False]
    # ...and an id the campaign reserved for eval this run is honored on top.
    assert rc._rec_is_heldout(records[-1], {"rmsnorm_aiter"}) is True


def test_build_holds_out_a_foreign_arch_record_from_a_trainable_family(
        tmp_path, monkeypatch):
    """A trainable op family measured on a FOREIGN arch is still eval-only: the
    claim is a gfx950 claim, so an RDNA/NVIDIA-tagged kernel is not evidence."""
    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    _write_shard(tmp_path, "wins", "rmsnorm_foreign",
                 [_win("rmsnorm_foreign", "rmsnorm", HELDOUT_MARK + "_foreign",
                       arch="gfx1100")])

    ctx = _build_ctx(tmp_path, monkeypatch, tasks=["rmsnorm_aiter"])
    rc._stage_build(ctx)

    sft_rows, dpo_rows = _products(tmp_path)
    assert HELDOUT_MARK not in json.dumps(sft_rows)
    assert HELDOUT_MARK not in json.dumps(dpo_rows)


def test_build_reads_the_split_from_the_registry_not_from_the_selection(
        tmp_path, monkeypatch):
    """``--tasks`` selects what to TRAIN on; it can never widen the train split.

    Naming a held-out task on the command line routes it to eval, and its records
    still get dropped from the training products.
    """
    held = _heldout_task("near_probe")
    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    _seed_heldout_shards(tmp_path, held)

    ctx = _build_ctx(tmp_path, monkeypatch, tasks=["rmsnorm_aiter", held.task_id])
    assert ctx["train_task_ids"] == ["rmsnorm_aiter"]
    assert ctx["eval_task_ids"] == [held.task_id]

    rc._stage_build(ctx)

    sft_rows, dpo_rows = _products(tmp_path)
    assert HELDOUT_MARK not in json.dumps(sft_rows)
    assert HELDOUT_MARK not in json.dumps(dpo_rows)


def test_build_keeps_every_trainable_family_it_was_given(tmp_path, monkeypatch):
    """The holdout filter is the ONLY thing allowed to remove a record.

    A previous revision ran a random 80/10/10 leakage_split here and silently
    exiled ~20% of trainable op families into partitions nothing consumed. Both
    selected families must survive into the DPO product.
    """
    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rmscanary")
    _seed_train_shards(tmp_path, "gemm_bf16", "gemm", "gemmcanary")

    ctx = _build_ctx(tmp_path, monkeypatch)
    rc._stage_build(ctx)

    _sft_rows, dpo_rows = _products(tmp_path)
    text = json.dumps(dpo_rows)
    assert "rmscanary" in text and "gemmcanary" in text


# --- the >=8% reward-hack hard-negative floor ------------------------------- #
def _dpo_hard_fraction(dpo_rows: list[dict]) -> float:
    hard = sum(1 for row in dpo_rows if row.get("negative_kind") == "reward_hack")
    return hard / len(dpo_rows) if dpo_rows else 0.0


def test_build_meets_the_hard_negative_floor_at_the_default_target(
        tmp_path, monkeypatch):
    """With the default ``--dpo-hard-fraction`` the emitted pairs clear >=8%.

    The abundant ranked-group pairs are subsampled so the anti-reward-hack
    contrast is not diluted; this asserts on the file the DPO stage will read.
    """
    from kore.data.hard_negatives import HARD_NEGATIVE_DPO_TARGET

    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    _write_shard(tmp_path, "groups", "rmsnorm_bulk",
                 [_group("rmsnorm_bulk", "rmsnorm", f"bulk{i}", n_candidates=30)
                  for i in range(4)])

    ctx = _build_ctx(tmp_path, monkeypatch, tasks=["rmsnorm_aiter"])
    rc._stage_build(ctx)

    _sft_rows, dpo_rows = _products(tmp_path)
    assert _dpo_hard_fraction(dpo_rows) >= HARD_NEGATIVE_DPO_TARGET


def test_build_refuses_a_dpo_set_below_the_hard_negative_floor(tmp_path, monkeypatch):
    """REGRESSION for a real defect: the floor was computed and logged, never enforced.

    ``build_dpo_with_hard_negatives`` returns ``meets_target``; the build stage
    only interpolated it into a log line, so a diluted set trained anyway. A
    shipped 14B run emitted 4.1% (``n_hard=1854 / n_total=44709`` in
    ``data/full14b/events.jsonl``) and nothing stopped it.

    Here ``--dpo-hard-fraction 0`` turns the subsampling off and one very wide
    ranked group floods the base pairs, so the emitted set lands at ~4% - the same
    dilution the shipped run had.
    """
    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    _write_shard(tmp_path, "groups", "rmsnorm_flood",
                 [_group("rmsnorm_flood", "rmsnorm", "flood", n_candidates=200)])

    ctx = _build_ctx(tmp_path, monkeypatch, "--dpo-hard-fraction", "0",
                     tasks=["rmsnorm_aiter"])
    with pytest.raises(SystemExit, match="hard-negative floor NOT met"):
        rc._stage_build(ctx)


def test_build_refuses_a_dpo_set_with_no_hard_negatives_at_all(tmp_path, monkeypatch):
    """Zero hard negatives is the degenerate case of the same failure, and the
    one an empty/failed curation actually produces."""
    from kore.data import assemble

    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    monkeypatch.setattr(
        assemble, "build_dpo_with_hard_negatives",
        lambda *a, **k: {"rows": [{"prompt": [], "chosen": [], "rejected": []}] * 50,
                         "n_hard": 0, "n_total": 50, "meets_target": False})

    ctx = _build_ctx(tmp_path, monkeypatch, tasks=["rmsnorm_aiter"])
    with pytest.raises(SystemExit, match="hard-negative floor"):
        rc._stage_build(ctx)


def test_build_accepts_a_dpo_set_that_clears_the_floor(tmp_path, monkeypatch):
    """The complement: the enforcement is a floor, not a blanket refusal."""
    from kore.data import assemble

    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    rows = [{"prompt": [{"role": "user", "content": "p"}],
             "chosen": [{"role": "assistant", "content": "c"}],
             "rejected": [{"role": "assistant", "content": "r"}]}] * 100
    monkeypatch.setattr(
        assemble, "build_dpo_with_hard_negatives",
        lambda *a, **k: {"rows": rows, "n_hard": 12, "n_total": 100,
                         "meets_target": True})

    ctx = _build_ctx(tmp_path, monkeypatch, tasks=["rmsnorm_aiter"])
    rc._stage_build(ctx)

    assert len(_products(tmp_path)[1]) == 100


def test_the_hard_negative_floor_is_the_spec_floor_not_the_requested_target(
        tmp_path, monkeypatch):
    """``--dpo-hard-fraction`` tunes how hard the abundant base pairs are thinned;
    it is not a lever for opting out of the 8% floor. A set at 10% clears the floor
    even though it missed the requested 12% target."""
    from kore.data import assemble

    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    rows = [{"prompt": [{"role": "user", "content": "p"}],
             "chosen": [{"role": "assistant", "content": "c"}],
             "rejected": [{"role": "assistant", "content": "r"}]}] * 100
    monkeypatch.setattr(
        assemble, "build_dpo_with_hard_negatives",
        lambda *a, **k: {"rows": rows, "n_hard": 10, "n_total": 100,
                         "meets_target": True})

    ctx = _build_ctx(tmp_path, monkeypatch, "--dpo-hard-fraction", "0.12",
                     tasks=["rmsnorm_aiter"])
    rc._stage_build(ctx)
    assert len(_products(tmp_path)[1]) == 100


# --- fail closed on a collapsed / empty mix --------------------------------- #
def test_build_refuses_an_sft_mix_whose_general_slice_collapsed(tmp_path, monkeypatch):
    """The ~45% general slice is the anti-catastrophic-forgetting backbone. A mix
    that water-filled it away is a forgetting ACCELERATOR, so it is refused rather
    than trained."""
    from kore.data import assemble

    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    kernel_only = [{"messages": [{"role": "user", "content": f"k{i}"}],
                    "_source": "kernel_repair_opt"} for i in range(50)]
    monkeypatch.setattr(assemble, "build_multicap_dataset",
                        lambda *a, **k: kernel_only)

    ctx = _build_ctx(tmp_path, monkeypatch, tasks=["rmsnorm_aiter"])
    with pytest.raises(SystemExit, match="general-retention slice collapsed"):
        rc._stage_build(ctx)


def test_build_refuses_a_completely_empty_sft_mix(tmp_path, monkeypatch):
    from kore.data import assemble

    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    monkeypatch.setattr(assemble, "build_multicap_dataset", lambda *a, **k: [])

    ctx = _build_ctx(tmp_path, monkeypatch, tasks=["rmsnorm_aiter"])
    with pytest.raises(SystemExit, match="general-retention slice collapsed"):
        rc._stage_build(ctx)


def test_the_build_artifact_contract_rejects_an_empty_or_malformed_product(tmp_path):
    """Second layer: even if a stage wrote something, the campaign refuses to mark
    build complete unless both products parse and carry their required fields."""
    ctx = {"data_root": tmp_path, "args": _args(["--tasks", "rmsnorm_aiter"]),
           "dry": False, "train_task_ids": ["rmsnorm_aiter"], "eval_task_ids": [],
           "lineage": _lineage()}
    sft = tmp_path / "sft" / "multicap.jsonl"
    dpo = tmp_path / "dpo" / "pairs.jsonl"
    sft.parent.mkdir(parents=True, exist_ok=True)
    dpo.parent.mkdir(parents=True, exist_ok=True)

    # exactly what _write_rows emits for an empty product: a single blank line.
    sft.write_text("\n")
    dpo.write_text("\n")
    with pytest.raises(RuntimeError, match="invalid JSONL artifact"):
        rc._capture_stage_artifact(ctx, "build")

    # a well-formed SFT mix but DPO rows missing the preference fields
    sft.write_text(json.dumps({"messages": [{"role": "user", "content": "q"}]}) + "\n")
    dpo.write_text(json.dumps({"prompt": [{"role": "user", "content": "p"}]}) + "\n")
    with pytest.raises(RuntimeError, match="misses keys"):
        rc._capture_stage_artifact(ctx, "build")

    # and a complete product is accepted, so the rejections above are meaningful
    dpo.write_text(json.dumps({
        "prompt": [{"role": "user", "content": "p"}],
        "chosen": [{"role": "assistant", "content": "c"}],
        "rejected": [{"role": "assistant", "content": "r"}]}) + "\n")
    artifact = rc._capture_stage_artifact(ctx, "build")
    assert artifact["stage"] == "build" and artifact["digest"]


# --------------------------------------------------------------------------- #
# Shared lineage / retention scaffolding for the claim-publishing stages
# --------------------------------------------------------------------------- #
def _lineage() -> dict:
    return {
        "compatibility_digest": "sha256:test-lineage",
        "verifier_gate_contract": {"version": 1, "digest": "sha256:test-gate"},
    }


def _scores(**overrides) -> dict:
    scores = dict.fromkeys(rc._GENERAL_GATE_KEYS, 0.60)
    scores.update(overrides)
    return scores


def _suite(scores: dict, source: str = "full-hf") -> dict:
    return {"scores": dict(scores), "full": True,
            "sources": {key: source for key in rc._GENERAL_GATE_KEYS}}


# --------------------------------------------------------------------------- #
# 2. _stage_soup - base-ward interpolation behind a retention-gated alpha sweep
# --------------------------------------------------------------------------- #
def _soup_ctx(tmp_path, *extra_argv):
    args = _args(["--tasks", "rmsnorm_aiter", "--data-root", str(tmp_path),
                  "--soup-out", str(tmp_path / "soup"), "--eval-budget", "1",
                  "--campaign-mode", "development", *extra_argv])
    return {"data_root": tmp_path, "args": args, "dry": False,
            "base": "base_model", "grpo_ckpt": "runs/grpo",
            "tasks": [get_task("rmsnorm_aiter")],
            "eval_tasks": [get_task("rmsnorm_aiter")],
            "eval_task_ids": ["rmsnorm_aiter"], "lineage": _lineage()}


def _stub_soup(monkeypatch, *, kernel_by_alpha, general_by_alpha=None):
    """Wire the soup stage to a scripted per-alpha evaluation.

    The stage's REAL decision logic (``soup_sweep_materialized`` + ``StageGate``
    + the receipt it writes) is left untouched; only checkpoint materialization,
    serving and measurement are replaced.
    """
    import kore.eval.bakeoff as bakeoff
    import kore.eval.policies as policies
    import kore.policy.soup as soup

    _no_torch(monkeypatch)
    seen: dict = {"alphas": [], "current": None}

    def fake_build_soup(base, kore_ckpt, alpha, out_dir, *a, **k):
        seen["alphas"].append(float(alpha))
        seen["current"] = float(alpha)
        return str(out_dir)

    def fake_retention(ctx, generate, *, stage, role, expected_sources=None,
                       cache_tag=None):
        alpha = seen["current"]
        overrides = (general_by_alpha or {}).get(alpha, {})
        return _suite(_scores(**overrides))

    monkeypatch.setattr(soup, "build_soup", fake_build_soup)
    monkeypatch.setattr(rc, "_evaluate_model_retention",
                        lambda ctx, model, **kw: _suite(_scores()))
    monkeypatch.setattr(rc, "_load_generate_or_fail",
                        lambda ctx, model, *, stage: object())
    monkeypatch.setattr(rc, "_run_retention_suite_checked", fake_retention)
    monkeypatch.setattr(policies, "model_policy",
                        lambda checkpoint, **kw: (lambda task, feedback=None: "src"))
    monkeypatch.setattr(
        bakeoff, "evaluate_policy",
        lambda pol, tasks, **kw: {"fast_p": {1.0: kernel_by_alpha[seen["current"]]}})
    return seen


def test_soup_refuses_to_soup_the_untrained_base(tmp_path):
    ctx = _soup_ctx(tmp_path)
    ctx["grpo_ckpt"] = ctx["dpo_ckpt"] = ctx["sft_ckpt"] = None
    with pytest.raises(SystemExit, match="refusing to soup the base"):
        rc._stage_soup(ctx)


def test_soup_sweep_always_evaluates_the_alpha_zero_safety_point(tmp_path, monkeypatch):
    """alpha=0 is literally the base model. It is prepended to every sweep even
    though ``SoupConfig.alphas`` never lists it, because it is the reference the
    nonzero alphas are gated against."""
    from kore.policy.configs import SoupConfig

    seen = _stub_soup(monkeypatch,
                      kernel_by_alpha={0.0: 0.40, 0.7: 0.55, 0.8: 0.60, 0.9: 0.50})
    ctx = _soup_ctx(tmp_path)
    rc._stage_soup(ctx)

    assert 0.0 not in SoupConfig().alphas, "the sweep must ADD the safety point"
    assert seen["alphas"][0] == 0.0
    sweep = json.loads((tmp_path / "eval" / "soup_sweep.json").read_text())
    assert sweep["alpha_zero_included"] is True
    assert any(row["alpha"] == 0.0 and row["safety_only"] for row in sweep["sweep"])


def test_soup_promotes_the_best_strictly_improving_alpha(tmp_path, monkeypatch):
    _stub_soup(monkeypatch,
               kernel_by_alpha={0.0: 0.40, 0.7: 0.55, 0.8: 0.62, 0.9: 0.50})
    ctx = _soup_ctx(tmp_path)
    rc._stage_soup(ctx)

    sweep = json.loads((tmp_path / "eval" / "soup_sweep.json").read_text())
    assert sweep["best_alpha"] == 0.8          # best kernel among the passing alphas
    assert sweep["gate_satisfied"] is True
    assert sweep["nonzero_promoted"] is True
    assert ctx["final"] == str(tmp_path / "soup")


def test_soup_refuses_an_alpha_that_wins_the_kernel_but_regresses_a_general_metric(
        tmp_path, monkeypatch):
    """The promotion contract is conjunctive. The alpha with the BEST kernel here
    also drops mmlu well past epsilon, so it must not be the one promoted - and if
    it is the only nonzero candidate, nothing is promoted at all."""
    _stub_soup(
        monkeypatch,
        kernel_by_alpha={0.0: 0.40, 0.7: 0.90, 0.8: 0.90, 0.9: 0.90},
        general_by_alpha={0.7: {"mmlu": 0.20}, 0.8: {"mmlu": 0.20},
                          0.9: {"mmlu": 0.20}},
    )
    ctx = _soup_ctx(tmp_path)
    with pytest.raises(SystemExit, match="soup promotion aborted"):
        rc._stage_soup(ctx)

    sweep = json.loads((tmp_path / "eval" / "soup_sweep.json").read_text())
    assert sweep["gate_satisfied"] is False
    assert sweep["nonzero_promoted"] is False
    assert ctx.get("final") is None
    regressed = [row for row in sweep["sweep"] if row["alpha"] > 0.0]
    assert regressed and all("mmlu" in row["gate"]["regressions"] for row in regressed)


def test_soup_prefers_the_retaining_alpha_over_a_faster_regressing_one(
        tmp_path, monkeypatch):
    """The same conjunction, with a survivor: the regressing alpha has the higher
    kernel score, so a max-by-kernel that ignored retention would pick it."""
    _stub_soup(
        monkeypatch,
        kernel_by_alpha={0.0: 0.40, 0.7: 0.50, 0.8: 0.55, 0.9: 0.99},
        general_by_alpha={0.9: {"humaneval": 0.10}},
    )
    ctx = _soup_ctx(tmp_path)
    rc._stage_soup(ctx)

    sweep = json.loads((tmp_path / "eval" / "soup_sweep.json").read_text())
    assert sweep["best_alpha"] == 0.8
    assert sweep["best"]["kernel"] == 0.55


def test_soup_refuses_when_no_nonzero_alpha_beats_the_base_kernel(
        tmp_path, monkeypatch):
    """Interpolating back toward the base must BUY something. Equal-to-base is not
    an improvement, so the sweep aborts rather than shipping a no-op soup."""
    _stub_soup(monkeypatch,
               kernel_by_alpha={0.0: 0.40, 0.7: 0.40, 0.8: 0.40, 0.9: 0.39})
    ctx = _soup_ctx(tmp_path)
    with pytest.raises(SystemExit, match="no nonzero alpha improved the kernel"):
        rc._stage_soup(ctx)

    sweep = json.loads((tmp_path / "eval" / "soup_sweep.json").read_text())
    assert sweep["gate_satisfied"] is False
    assert sweep["alpha_zero_included"] is True


def test_soup_aborts_when_the_alpha_zero_safety_point_itself_fails_retention(
        tmp_path, monkeypatch):
    """alpha=0 reproduces the base; if it cannot retain against the base's own
    scores the measurement is untrustworthy and no alpha may be promoted."""
    _stub_soup(monkeypatch,
               kernel_by_alpha={0.0: 0.40, 0.7: 0.90, 0.8: 0.90, 0.9: 0.90},
               general_by_alpha={0.0: {"bfcl": 0.05}})
    ctx = _soup_ctx(tmp_path)
    with pytest.raises(SystemExit, match="alpha=0 base safety evaluation failed"):
        rc._stage_soup(ctx)

    sweep = json.loads((tmp_path / "eval" / "soup_sweep.json").read_text())
    # It aborted at the safety point, before any nonzero alpha was materialized.
    assert [row["alpha"] for row in sweep["sweep"]] == [0.0]


def test_soup_refuses_a_non_finite_kernel_measurement(tmp_path, monkeypatch):
    """A NaN fast_p is not a small number, it is an unusable measurement - and it
    compares False against everything, so a silent one would quietly exclude an
    alpha instead of failing the sweep."""
    _stub_soup(monkeypatch,
               kernel_by_alpha={0.0: 0.40, 0.7: float("nan"), 0.8: 0.5, 0.9: 0.5})
    ctx = _soup_ctx(tmp_path)
    with pytest.raises(RuntimeError, match="non-finite"):
        rc._stage_soup(ctx)
    assert ctx.get("final") is None


def test_the_soup_artifact_contract_refuses_an_alpha_zero_promotion(tmp_path):
    """A receipt that promoted alpha=0 promoted the BASE. The campaign refuses to
    record soup as complete on it, whatever the sweep claimed."""
    ctx = {"data_root": tmp_path, "args": _args(["--tasks", "rmsnorm_aiter"]),
           "dry": False, "final": str(tmp_path / "final"), "lineage": _lineage()}
    receipt = tmp_path / "eval" / "soup_sweep.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)

    for payload in (
        {"best_alpha": 0.0, "gate_satisfied": True, "nonzero_promoted": True},
        {"best_alpha": 0.8, "gate_satisfied": False, "nonzero_promoted": True},
        {"best_alpha": 0.8, "gate_satisfied": True, "nonzero_promoted": False},
        {"best_alpha": None, "gate_satisfied": True, "nonzero_promoted": True},
    ):
        receipt.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError, match="does not authorize nonzero promotion"):
            rc._capture_stage_artifact(ctx, "soup")


# --------------------------------------------------------------------------- #
# 3. _stage_eval - the matched-budget bake-off + the final promotion StageGate
# --------------------------------------------------------------------------- #
def _policy_result(fast1: float, *, per_task=None) -> dict:
    return {
        "n": 1, "budget": 1, "mode": "serial",
        "fast_p": {1.0: fast1, 1.5: fast1 / 2},
        "fast_p_curve": [(1.0, fast1), (1.5, fast1 / 2)],
        "geometric_mean_speedup": 1.0 + fast1,
        "num_correct": 1,
        "per_task": per_task or [],
    }


def _bakeoff(seed_fast1: float, kore_fast1: float, **kw) -> dict:
    return {
        "budget": 1, "n": 1,
        "policies": {"seed": _policy_result(seed_fast1, **kw),
                     "kore": _policy_result(kore_fast1, **kw)},
        "ranking_by_fast1": ["kore", "seed"],
    }


def _eval_ctx(tmp_path, *extra_argv):
    args = _args(["--tasks", "rmsnorm_aiter", "--data-root", str(tmp_path),
                  "--eval-budget", "1", "--campaign-mode", "development",
                  *extra_argv])
    return {"data_root": tmp_path, "args": args, "dry": False,
            "base": "base_model", "grpo_ckpt": "runs/grpo",
            "tasks": [get_task("rmsnorm_aiter")],
            "eval_tasks": [get_task("rmsnorm_aiter")],
            "eval_task_ids": ["rmsnorm_aiter"], "lineage": _lineage()}


def _stub_eval(monkeypatch, *, bakeoff_result, base_scores, candidate_scores,
               tracks=None):
    import kore.eval.bakeoff as bakeoff_mod
    import kore.eval.policies as policies

    _no_torch(monkeypatch)
    monkeypatch.setattr(bakeoff_mod, "matched_budget_bakeoff",
                        lambda policies_, tasks, **kw: bakeoff_result)
    monkeypatch.setattr(policies, "model_policy",
                        lambda checkpoint, **kw: (lambda task, feedback=None: "src"))
    monkeypatch.setattr(
        rc, "_evaluate_retention_pair",
        lambda ctx, *, stage, base, candidate: (_suite(base_scores),
                                                _suite(candidate_scores)))
    verdicts = tracks if tracks is not None else {
        "paired_significance": True, "kernelbench_amd": True,
        "opus_head_to_head": True,
    }
    for name, fn in (("paired_significance", "_eval_paired_significance"),
                     ("kernelbench_amd", "_eval_kernelbench_amd"),
                     ("opus_head_to_head", "_eval_opus_head_to_head")):
        monkeypatch.setattr(rc, fn,
                            (lambda passed: lambda *a, **k: {"passed": passed})(
                                verdicts[name]))


def test_eval_refuses_to_promote_the_untrained_base(tmp_path):
    ctx = _eval_ctx(tmp_path)
    ctx["grpo_ckpt"] = None
    with pytest.raises(SystemExit, match="refusing to promote the base"):
        rc._stage_eval(ctx)


def test_eval_refuses_an_empty_heldout_split(tmp_path):
    ctx = _eval_ctx(tmp_path)
    ctx["eval_tasks"] = []
    ctx["eval_task_ids"] = []
    ctx["tasks"] = []
    with pytest.raises(SystemExit, match="empty held-out task split"):
        rc._stage_eval(ctx)


def test_eval_promotes_only_on_a_strict_kernel_win_with_full_retention(
        tmp_path, monkeypatch):
    _stub_eval(monkeypatch, bakeoff_result=_bakeoff(0.20, 0.45),
               base_scores=_scores(), candidate_scores=_scores())
    ctx = _eval_ctx(tmp_path)
    rc._stage_eval(ctx)

    promotion = json.loads((tmp_path / "eval" / "promotion_gate.json").read_text())
    assert promotion["passed"] is True
    assert rc._SOUP_KERNEL_KEY in promotion["improvements"]
    assert promotion["candidate"] == "runs/grpo"
    claim = json.loads((tmp_path / "eval" / "claim_status.json").read_text())
    assert claim["passed"] is True and claim["profile"] == "core"


def test_eval_refuses_a_kernel_win_that_regressed_a_general_metric(
        tmp_path, monkeypatch):
    """The headline case: KORE more than doubles fast_p@1 but loses 25 points of
    mmlu. A kernel-only gate would promote it."""
    _stub_eval(monkeypatch, bakeoff_result=_bakeoff(0.20, 0.45),
               base_scores=_scores(),
               candidate_scores=_scores(mmlu=0.35))
    ctx = _eval_ctx(tmp_path)
    with pytest.raises(SystemExit, match="final promotion gate"):
        rc._stage_eval(ctx)

    promotion = json.loads((tmp_path / "eval" / "promotion_gate.json").read_text())
    assert promotion["passed"] is False
    assert "mmlu" in promotion["regressions"]
    # the kernel objective DID improve - the refusal is purely the retention half
    assert rc._SOUP_KERNEL_KEY in promotion["improvements"]
    assert not (tmp_path / "eval" / "claim_status.json").exists()


def test_eval_refuses_a_flat_kernel_metric_even_with_perfect_retention(
        tmp_path, monkeypatch):
    """"Did not get worse" is not a claim: the kernel metric must STRICTLY improve."""
    _stub_eval(monkeypatch, bakeoff_result=_bakeoff(0.30, 0.30),
               base_scores=_scores(), candidate_scores=_scores(mmlu=0.99))
    ctx = _eval_ctx(tmp_path)
    with pytest.raises(SystemExit, match="final promotion gate"):
        rc._stage_eval(ctx)

    promotion = json.loads((tmp_path / "eval" / "promotion_gate.json").read_text())
    assert promotion["passed"] is False
    assert rc._SOUP_KERNEL_KEY in promotion["regressions"]


def test_eval_refuses_when_a_general_metric_was_never_measured(tmp_path, monkeypatch):
    """A metric we promised to track but did not measure is a failure, not a
    silent pass - otherwise dropping a benchmark would be the cheapest way to
    clear the gate."""
    unmeasured = _scores()
    unmeasured.pop("bfcl")
    _stub_eval(monkeypatch, bakeoff_result=_bakeoff(0.20, 0.45),
               base_scores=_scores(), candidate_scores=unmeasured)
    ctx = _eval_ctx(tmp_path)
    with pytest.raises(SystemExit, match="final promotion gate"):
        rc._stage_eval(ctx)

    promotion = json.loads((tmp_path / "eval" / "promotion_gate.json").read_text())
    assert "bfcl" in promotion["regressions"]


def test_eval_refuses_a_bakeoff_that_is_missing_a_policy(tmp_path, monkeypatch):
    """A one-sided bake-off cannot support a seed-vs-KORE claim."""
    broken = _bakeoff(0.20, 0.45)
    broken["policies"].pop("seed")
    broken["ranking_by_fast1"] = ["kore"]
    _stub_eval(monkeypatch, bakeoff_result=broken, base_scores=_scores(),
               candidate_scores=_scores())
    ctx = _eval_ctx(tmp_path)
    with pytest.raises(RuntimeError, match="missing seed/kore policy results"):
        rc._stage_eval(ctx)


def test_eval_refuses_a_bakeoff_with_no_fast_p_metrics(tmp_path, monkeypatch):
    empty = _bakeoff(0.20, 0.45)
    empty["policies"]["kore"]["fast_p"] = {}
    _stub_eval(monkeypatch, bakeoff_result=empty, base_scores=_scores(),
               candidate_scores=_scores())
    ctx = _eval_ctx(tmp_path)
    with pytest.raises(RuntimeError, match="empty fast_p metrics"):
        rc._stage_eval(ctx)


def test_a_required_frontier_track_is_blocking_for_the_profile_that_claims_it(
        tmp_path, monkeypatch):
    _stub_eval(monkeypatch, bakeoff_result=_bakeoff(0.20, 0.45),
               base_scores=_scores(), candidate_scores=_scores(),
               tracks={"paired_significance": False, "kernelbench_amd": True,
                       "opus_head_to_head": True})
    ctx = _eval_ctx(tmp_path, "--claim-profile", "kernel-frontier")
    with pytest.raises(SystemExit, match="failed required frontier tracks"):
        rc._stage_eval(ctx)

    claim = json.loads((tmp_path / "eval" / "claim_status.json").read_text())
    assert claim["passed"] is False
    assert claim["failed_required_tracks"] == ["paired_significance"]
    # the core promotion gate itself passed; the profile is what blocked
    assert json.loads(
        (tmp_path / "eval" / "promotion_gate.json").read_text())["passed"] is True


def test_the_same_failing_track_is_reported_but_not_blocking_for_the_core_profile(
        tmp_path, monkeypatch):
    _stub_eval(monkeypatch, bakeoff_result=_bakeoff(0.20, 0.45),
               base_scores=_scores(), candidate_scores=_scores(),
               tracks={"paired_significance": False, "kernelbench_amd": False,
                       "opus_head_to_head": False})
    ctx = _eval_ctx(tmp_path)
    rc._stage_eval(ctx)

    claim = json.loads((tmp_path / "eval" / "claim_status.json").read_text())
    assert claim["passed"] is True and claim["required_tracks"] == []
    assert all(track["passed"] is False for track in claim["tracks"].values())
    assert all(track["required"] is False for track in claim["tracks"].values())


def test_the_eval_artifact_contract_requires_a_passing_gate_and_matching_profile(
        tmp_path):
    ctx = {"data_root": tmp_path,
           "args": _args(["--tasks", "rmsnorm_aiter"]), "dry": False,
           "lineage": _lineage()}
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "bakeoff.json").write_text(json.dumps(_bakeoff(0.2, 0.4)))
    promotion = eval_dir / "promotion_gate.json"
    claim = eval_dir / "claim_status.json"
    good_claim = {"profile": "core", "passed": True}

    promotion.write_text(json.dumps({"passed": False}))
    claim.write_text(json.dumps(good_claim))
    with pytest.raises(RuntimeError, match="no valid bakeoff or passing StageGate"):
        rc._capture_stage_artifact(ctx, "eval")

    promotion.write_text(json.dumps({"passed": True}))
    claim.write_text(json.dumps({"profile": "flagship", "passed": True}))
    with pytest.raises(RuntimeError, match="claim profile does not match"):
        rc._capture_stage_artifact(ctx, "eval")

    claim.write_text(json.dumps({"profile": "core", "passed": False}))
    with pytest.raises(RuntimeError, match="frontier track contract did not pass"):
        rc._capture_stage_artifact(ctx, "eval")

    claim.write_text(json.dumps(good_claim))
    assert rc._capture_stage_artifact(ctx, "eval")["stage"] == "eval"


def test_eval_scores_the_final_promoted_checkpoint_not_an_earlier_stage(
        tmp_path, monkeypatch):
    """When soup promoted a checkpoint, THAT is what gets published. Falling back
    to the GRPO checkpoint would publish a model nobody shipped."""
    _stub_eval(monkeypatch, bakeoff_result=_bakeoff(0.20, 0.45),
               base_scores=_scores(), candidate_scores=_scores())
    ctx = _eval_ctx(tmp_path)
    ctx["final"] = "runs/soup"
    rc._stage_eval(ctx)

    promotion = json.loads((tmp_path / "eval" / "promotion_gate.json").read_text())
    assert promotion["candidate"] == "runs/soup"


# --- neither gate may pass on a smoke / bundled fallback -------------------- #
def _stub_retention_backend(monkeypatch, source: str):
    """Serve a retention suite from ``source`` through the REAL validation path."""
    import kore.eval.retention as retention

    _no_torch(monkeypatch)
    monkeypatch.setattr(rc, "_load_generate_or_fail",
                        lambda ctx, model, *, stage: object())
    monkeypatch.setattr(retention, "run_retention_suite",
                        lambda generate, **kw: _suite(_scores(), source=source))


def test_the_final_gate_cannot_pass_on_a_smoke_retention_fallback(
        tmp_path, monkeypatch):
    """End to end: the bake-off is a landslide (fast_p 0.20 -> 0.90), so the ONLY
    thing that can stop promotion is the retention half noticing that its general
    scores came from the bundled smoke pool rather than the real splits."""
    import kore.eval.bakeoff as bakeoff_mod
    import kore.eval.policies as policies

    monkeypatch.setattr(bakeoff_mod, "matched_budget_bakeoff",
                        lambda policies_, tasks, **kw: _bakeoff(0.20, 0.90))
    monkeypatch.setattr(policies, "model_policy",
                        lambda checkpoint, **kw: (lambda task, feedback=None: "src"))
    _stub_retention_backend(monkeypatch, "bundled-smoke")

    ctx = _eval_ctx(tmp_path)
    ctx["args"].campaign_mode = "production"
    with pytest.raises(SystemExit, match="production retention rejected smoke/fallback"):
        rc._stage_eval(ctx)

    assert not (tmp_path / "eval" / "promotion_gate.json").exists()


def test_the_soup_gate_cannot_pass_on_a_smoke_retention_fallback(
        tmp_path, monkeypatch):
    """Same for soup, and it aborts on the BASE suite - before a single alpha is
    materialized, so no work is done against an unusable no-regression floor."""
    _stub_retention_backend(monkeypatch, "bundled-smoke")
    ctx = _soup_ctx(tmp_path)
    ctx["args"].campaign_mode = "production"
    with pytest.raises(SystemExit, match="production retention rejected smoke/fallback"):
        rc._stage_soup(ctx)

    assert not (tmp_path / "eval" / "soup_sweep.json").exists()


def test_a_production_gate_refuses_a_smoke_retention_source(tmp_path):
    """The general half of both gates is only meaningful on the real benchmark
    splits. A bundled smoke source scores high and cheap, so production rejects it
    before it can be compared."""
    ctx = _eval_ctx(tmp_path)
    ctx["args"].campaign_mode = "production"
    smoke = _suite(_scores(), source="bundled-smoke")

    with pytest.raises(SystemExit, match="rejected smoke/fallback sources"):
        rc._validate_retention_suite(ctx, smoke, stage="eval", role="candidate")

    # and a full-hf suite that merely forgot to declare itself full is refused too
    not_full = _suite(_scores())
    not_full["full"] = False
    with pytest.raises(SystemExit, match="rejected smoke/fallback sources"):
        rc._validate_retention_suite(ctx, not_full, stage="eval", role="candidate")

    assert rc._validate_retention_suite(ctx, _suite(_scores()), stage="eval",
                                        role="candidate")


def test_a_gate_refuses_a_candidate_measured_against_different_sources(tmp_path):
    """Base and candidate must be scored on the SAME sources or the comparison is
    not a comparison."""
    ctx = _eval_ctx(tmp_path)
    base = _suite(_scores())
    candidate = _suite(_scores(), source="bundled-smoke")
    with pytest.raises(SystemExit, match="retention source mismatch"):
        rc._validate_retention_suite(ctx, candidate, stage="eval", role="candidate",
                                     expected_sources=base["sources"])


def test_a_gate_refuses_a_non_finite_or_non_numeric_retention_metric(tmp_path):
    ctx = _eval_ctx(tmp_path)
    with pytest.raises(SystemExit, match="non-finite"):
        rc._validate_retention_suite(ctx, _suite(_scores(mmlu=float("inf"))),
                                     stage="eval", role="candidate")
    with pytest.raises(SystemExit, match="non-numeric"):
        rc._validate_retention_suite(ctx, _suite(_scores(mmlu="n/a")),
                                     stage="eval", role="candidate")


def test_the_kernelbench_track_cannot_be_satisfied_by_bundled_specs_in_production(
        tmp_path, monkeypatch):
    """The bundled offline specs are a smoke fixture. Under a profile that CLAIMS
    KernelBench-AMD, a bundled report can never pass however good it looks."""
    import kore.eval.kernelbench_amd as kb

    report = {"n": 25, "correct_rate": 1.0, "fast_1": 1.0,
              "fast_p": {1.0: 1.0, 1.5: 1.0}}
    monkeypatch.setattr(kb, "bundled_specs", lambda: [{"spec": "smoke"}])
    monkeypatch.setattr(kb, "format_kernelbench_report", lambda r: "report")
    monkeypatch.setattr(kb, "run_kernelbench_amd",
                        lambda policy, specs, **kw: {"report": report})

    ctx = _eval_ctx(tmp_path, "--claim-profile", "kernel-frontier")
    ctx["args"].campaign_mode = "production"
    result = rc._eval_kernelbench_amd(ctx, object(), object())

    assert result["source"] == "bundled-smoke"
    assert result["source_ok"] is False
    assert result["passed"] is False


# --- the paired-significance track ------------------------------------------ #
def _pair(task_id: str, *, correct: bool, speedup: float) -> dict:
    return {"task_id": task_id, "correct": correct, "best_speedup": speedup}


def test_paired_significance_needs_at_least_two_matched_correct_tasks(tmp_path):
    res = _bakeoff(0.2, 0.4)
    res["policies"]["kore"]["per_task"] = [_pair("t1", correct=True, speedup=2.0)]
    res["policies"]["seed"]["per_task"] = [_pair("t1", correct=True, speedup=1.0)]
    with pytest.raises(RuntimeError, match="need >=2"):
        rc._eval_paired_significance(_eval_ctx(tmp_path), res)


def test_paired_significance_ignores_tasks_only_one_policy_solved(tmp_path):
    """An unpaired task is not evidence: if seed failed it, there is no pair."""
    res = _bakeoff(0.2, 0.4)
    res["policies"]["kore"]["per_task"] = [
        _pair("t1", correct=True, speedup=2.0),
        _pair("t2", correct=True, speedup=2.0),
    ]
    res["policies"]["seed"]["per_task"] = [
        _pair("t1", correct=True, speedup=1.0),
        _pair("t2", correct=False, speedup=1.0),
    ]
    with pytest.raises(RuntimeError, match="only 1 matched-correct"):
        rc._eval_paired_significance(_eval_ctx(tmp_path), res)


def test_paired_significance_passes_only_when_kore_is_better_and_significant(
        tmp_path):
    ctx = _eval_ctx(tmp_path)
    tasks = [f"t{i}" for i in range(12)]
    res = _bakeoff(0.2, 0.4)
    res["policies"]["kore"]["per_task"] = [
        _pair(t, correct=True, speedup=2.0) for t in tasks]
    res["policies"]["seed"]["per_task"] = [
        _pair(t, correct=True, speedup=1.0) for t in tasks]
    won = rc._eval_paired_significance(ctx, res)
    assert won["passed"] is True and won["direction"] == "kore_better"
    assert won["n"] == len(tasks)
    assert json.loads((tmp_path / "eval" / "paired_seed_vs_kore.json").read_text())

    # the mirror image: the SAME machinery must refuse when the seed is ahead.
    reversed_res = _bakeoff(0.4, 0.2)
    reversed_res["policies"]["kore"]["per_task"] = [
        _pair(t, correct=True, speedup=1.0) for t in tasks]
    reversed_res["policies"]["seed"]["per_task"] = [
        _pair(t, correct=True, speedup=2.0) for t in tasks]
    lost = rc._eval_paired_significance(ctx, reversed_res)
    assert lost["passed"] is False and lost["direction"] != "kore_better"


# --------------------------------------------------------------------------- #
# 4. _stage_datagen - shard generation, sharding, resumability
# --------------------------------------------------------------------------- #
def _datagen_ctx(tmp_path, *extra_argv, tasks=None):
    task_ids = list(tasks or TRAIN_IDS)
    args = _args(["--tasks", ",".join(task_ids), "--data-root", str(tmp_path),
                  "--teacher", "stub", "--n-repair", "1", "--n-parents", "2",
                  "--k", "2", "--wins-gens", "1", "--n-agentic", "1",
                  "--campaign-mode", "development", *extra_argv])
    ctx = {"data_root": tmp_path, "args": args, "dry": False,
           "tasks": [get_task(t) for t in task_ids]}
    rc._apply_split(ctx)
    return ctx


def _sequential(monkeypatch):
    """Force the single-process datagen path (no GPU pinning, one worker)."""
    monkeypatch.setattr(rc, "_gpu_ids", lambda ctx: [])
    monkeypatch.setattr(rc, "_datagen_plan", lambda ctx: (1, 1))


def _stub_generators(monkeypatch, calls: list):
    import kore.data.gen_groups as gen_groups
    import kore.data.gen_repair as gen_repair
    import kore.data.gen_wins as gen_wins
    import kore.env.kore_env as kore_env

    monkeypatch.setattr(rc, "_teacher", lambda args: SimpleNamespace(kind="stub"))
    monkeypatch.setattr(kore_env, "KoreEnv", lambda task, **kw: SimpleNamespace(task=task))

    def repairs(task, teacher, env, n=1, **kw):
        calls.append((task.task_id, "repair"))
        return [_repair(task.task_id, task.operation, f"{task.task_id}_r")]

    def groups(task, teacher, env, n_parents=1, k=2, **kw):
        calls.append((task.task_id, "groups"))
        return [_group(task.task_id, task.operation, f"{task.task_id}_g")
                for _ in range(n_parents)]

    def wins(task, teacher, env, gens=1, **kw):
        calls.append((task.task_id, "wins"))
        return [_win(task.task_id, task.operation, f"{task.task_id}_w")]

    monkeypatch.setattr(gen_repair, "generate_repairs", repairs)
    monkeypatch.setattr(gen_groups, "generate_groups", groups)
    monkeypatch.setattr(gen_wins, "generate_wins", wins)


def test_datagen_dry_run_touches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "_teacher",
                        lambda args: pytest.fail("a dry run contacted the teacher"))
    ctx = _datagen_ctx(tmp_path)
    ctx["dry"] = True
    rc._stage_datagen(ctx)
    assert list(tmp_path.iterdir()) == []


def test_datagen_generates_only_train_split_tasks(tmp_path, monkeypatch):
    """Held-out tasks get no datagen at all - the split is enforced before a
    single teacher call, not after."""
    held = heldout_tasks()[0]
    calls: list = []
    _sequential(monkeypatch)
    _stub_generators(monkeypatch, calls)

    ctx = _datagen_ctx(tmp_path, tasks=["rmsnorm_aiter", held.task_id])
    rc._stage_datagen(ctx)

    generated = {task_id for task_id, _kind in calls}
    assert generated == {"rmsnorm_aiter"}
    assert held.task_id not in generated
    assert not (tmp_path / "repair" / f"{held.task_id}.jsonl").exists()
    for kind in ("repair", "groups", "wins"):
        assert (tmp_path / kind / "rmsnorm_aiter.jsonl").exists()


def test_sequential_datagen_publishes_a_resumable_receipted_shard(
        tmp_path, monkeypatch):
    """REGRESSION for a real defect: the sequential path claimed to be resumable
    ("matches the parallel path") but wrote bare JSONL with no completion receipt.
    Its own resume check (``shard_done``) therefore always said False, so every
    rerun regenerated every shard, and the production build reader
    (``read_jsonl(mode='production_strict')``) rejected the unstamped records.
    """
    from kore.data.parallel_datagen import shard_done
    from kore.data.schemas import read_jsonl

    calls: list = []
    _sequential(monkeypatch)
    _stub_generators(monkeypatch, calls)

    ctx = _datagen_ctx(tmp_path, tasks=["rmsnorm_aiter"])
    rc._stage_datagen(ctx)
    first = list(calls)
    assert first, "nothing was generated"

    for kind in ("repair", "groups", "wins"):
        assert shard_done(tmp_path, "rmsnorm_aiter", kind, gate="quota_only") is True
        # the production build stage reads these back under the strict envelope
        assert read_jsonl(tmp_path / kind / "rmsnorm_aiter.jsonl", typed=True,
                          mode="production_strict")

    calls.clear()
    rc._stage_datagen(ctx)
    assert calls == [], f"a rerun regenerated completed shards: {first}"


def test_datagen_regenerates_a_shard_that_was_deleted(tmp_path, monkeypatch):
    """Resume must fill HOLES, so deleting a shard is the documented way to force
    its regeneration."""
    calls: list = []
    _sequential(monkeypatch)
    _stub_generators(monkeypatch, calls)

    ctx = _datagen_ctx(tmp_path, tasks=["rmsnorm_aiter"])
    rc._stage_datagen(ctx)
    (tmp_path / "wins" / "rmsnorm_aiter.jsonl").unlink()
    (tmp_path / "wins" / "rmsnorm_aiter.jsonl.complete.json").unlink()

    calls.clear()
    rc._stage_datagen(ctx)
    assert calls == [("rmsnorm_aiter", "wins")]


def test_datagen_parallel_path_shards_the_train_split_across_pinned_gpus(
        tmp_path, monkeypatch):
    import kore.data.parallel_datagen as pdg

    held = heldout_tasks()[0]
    seen: dict = {}

    def fake_run(task_ids, kinds, data_root, counts, **kw):
        seen["task_ids"] = list(task_ids)
        seen["kinds"] = tuple(kinds)
        seen["kw"] = kw
        return {"done": len(task_ids), "skip": 0, "error": 0, "records": 3}

    monkeypatch.setattr(pdg, "run_parallel_datagen", fake_run)
    monkeypatch.setattr(rc, "_datagen_plan", lambda ctx: (4, 8))

    ctx = _datagen_ctx(tmp_path, "--gpu-ids", "2,5",
                       tasks=["rmsnorm_aiter", "gemm_bf16", held.task_id])
    rc._stage_datagen(ctx)

    assert set(seen["task_ids"]) == {"rmsnorm_aiter", "gemm_bf16"}
    assert held.task_id not in seen["task_ids"]
    assert seen["kinds"] == pdg.DATAGEN_KINDS
    assert seen["kw"]["gpu_ids"] == [2, 5]
    # only a quota-satisfied receipt counts as done; a partial shard is redone
    assert seen["kw"]["completion_gate"] == "quota_only"


def test_datagen_does_not_swallow_a_worker_failure(tmp_path, monkeypatch):
    """A datagen run that lost workers must not return an empty-but-successful
    stage; the campaign has to see the failure."""
    import kore.data.parallel_datagen as pdg

    def boom(*a, **k):
        raise pdg.DatagenRunError("worker 0 died with exit code 1",
                                  summary={"done": 0, "error": 1})

    monkeypatch.setattr(pdg, "run_parallel_datagen", boom)
    monkeypatch.setattr(rc, "_datagen_plan", lambda ctx: (4, 8))
    monkeypatch.setattr(rc, "_gpu_ids", lambda ctx: [0])

    ctx = _datagen_ctx(tmp_path)
    with pytest.raises(pdg.DatagenRunError, match="worker 0 died"):
        rc._stage_datagen(ctx)


def test_the_datagen_artifact_contract_requires_every_train_shard(tmp_path):
    """An empty or missing shard cannot be recorded as a completed datagen."""
    ctx = {"data_root": tmp_path, "args": _args(["--tasks", "rmsnorm_aiter"]),
           "dry": False, "train_task_ids": ["rmsnorm_aiter"], "eval_task_ids": [],
           "lineage": _lineage()}
    with pytest.raises(RuntimeError, match="missing"):
        rc._capture_stage_artifact(ctx, "datagen")

    for kind in ("repair", "groups", "wins"):
        path = tmp_path / kind / "rmsnorm_aiter.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"task_id": "rmsnorm_aiter", "type": kind}) + "\n")
    artifact = rc._capture_stage_artifact(ctx, "datagen")
    assert len(artifact["outputs"]) == 3

    # a shard belonging to the WRONG task is rejected, not counted
    (tmp_path / "wins" / "rmsnorm_aiter.jsonl").write_text(
        json.dumps({"task_id": "someone_else", "type": "win"}) + "\n")
    with pytest.raises(RuntimeError, match="expected 'rmsnorm_aiter'"):
        rc._capture_stage_artifact(ctx, "datagen")


# --------------------------------------------------------------------------- #
# 5. _stage_agentic - CPU synthesis from VERIFIED records, or the live GPU path
# --------------------------------------------------------------------------- #
def _agentic_ctx(tmp_path, *extra_argv, tasks=None):
    task_ids = list(tasks or ["rmsnorm_aiter"])
    args = _args(["--tasks", ",".join(task_ids), "--data-root", str(tmp_path),
                  "--teacher", "stub", "--n-agentic", "1",
                  "--campaign-mode", "development", *extra_argv])
    ctx = {"data_root": tmp_path, "args": args, "dry": False,
           "tasks": [get_task(t) for t in task_ids]}
    rc._apply_split(ctx)
    return ctx


def _agentic_rows(tmp_path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((tmp_path / "agentic").glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines()
                 if line.strip()]
    return rows


def _forbid_live_agentic(monkeypatch):
    import kore.data.gen_agentic as gen_agentic
    import kore.data.parallel_datagen as pdg

    monkeypatch.setattr(
        gen_agentic, "generate_agentic_trajectories",
        lambda *a, **k: pytest.fail("synth mode ran the live GPU harness"))
    monkeypatch.setattr(
        pdg, "run_parallel_datagen",
        lambda *a, **k: pytest.fail("synth mode dispatched GPU workers"))
    monkeypatch.setattr(
        rc, "_teacher", lambda args: pytest.fail("synth mode contacted the teacher"))


def test_agentic_synth_builds_trajectories_only_from_verified_records(
        tmp_path, monkeypatch):
    """The synth path exists so the tool-use slice carries REAL measurements. It
    reads the verified repair/wins/groups shards and nothing else - no teacher,
    no GPU, no fabricated tool results."""
    _forbid_live_agentic(monkeypatch)
    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")

    ctx = _agentic_ctx(tmp_path)
    rc._stage_agentic(ctx)

    rows = _agentic_rows(tmp_path)
    assert rows, "synth produced no trajectories from verified records"
    for row in rows:
        assert row["task_id"] == "rmsnorm_aiter"
        assert row["messages"] and row["tool_trace"]
        # every tool turn is answered by a rendered result, ending on a keep
        assert any(msg.get("role") == "tool" for msg in row["messages"])
        assert '"tool": "keep"' in row["messages"][-1]["content"]


def test_agentic_synth_never_synthesizes_a_heldout_task(tmp_path, monkeypatch):
    """The agentic slice is the one SFT source that is read straight off disk, so
    it is the easiest place for a held-out record to sneak into training."""
    from kore.data.synth_agentic import synthesize_agentic

    held = heldout_tasks()[0]
    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    _seed_heldout_shards(tmp_path, held)

    summary = synthesize_agentic(tmp_path, cap=200, seed=0)
    assert summary["total"] > 0
    assert summary["heldout_skipped"] > 0

    rows = _agentic_rows(tmp_path)
    assert rows
    assert {row["task_id"] for row in rows} == {"rmsnorm_aiter"}
    assert HELDOUT_MARK not in json.dumps(rows)


def test_agentic_synth_with_no_verified_records_produces_no_artifact(tmp_path,
                                                                     monkeypatch):
    """Nothing to synthesize from must not read as a completed agentic stage."""
    _forbid_live_agentic(monkeypatch)
    ctx = _agentic_ctx(tmp_path)
    ctx["lineage"] = _lineage()
    ctx["train_task_ids"] = ["rmsnorm_aiter"]
    rc._stage_agentic(ctx)

    assert _agentic_rows(tmp_path) == []
    with pytest.raises(RuntimeError, match="produced no JSONL trajectories"):
        rc._capture_stage_artifact(ctx, "agentic")


def test_the_agentic_artifact_contract_requires_a_real_tool_trace(tmp_path):
    """A trajectory with an empty ``tool_trace`` taught no tool use; the contract
    refuses it rather than counting it."""
    ctx = {"data_root": tmp_path, "args": _args(["--tasks", "rmsnorm_aiter"]),
           "dry": False, "train_task_ids": ["rmsnorm_aiter"], "eval_task_ids": [],
           "lineage": _lineage()}
    path = tmp_path / "agentic" / "_synth_repair.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_id": "rmsnorm_aiter",
                                "messages": [{"role": "user", "content": "q"}],
                                "tool_trace": []}) + "\n")
    with pytest.raises(RuntimeError, match="empty required fields"):
        rc._capture_stage_artifact(ctx, "agentic")

    path.write_text(json.dumps({"task_id": "rmsnorm_aiter",
                                "messages": [{"role": "user", "content": "q"}],
                                "tool_trace": [{"turn": 0, "name": "test"}]}) + "\n")
    assert rc._capture_stage_artifact(ctx, "agentic")["outputs"]


def test_agentic_live_mode_shards_only_train_tasks_over_the_gpu_workers(
        tmp_path, monkeypatch):
    import kore.data.parallel_datagen as pdg

    held = heldout_tasks()[0]
    seen: dict = {}
    monkeypatch.setattr(pdg, "run_parallel_datagen",
                        lambda task_ids, kinds, *a, **kw: seen.update(
                            task_ids=list(task_ids), kinds=tuple(kinds), kw=kw)
                        or {"done": 1, "skip": 0, "error": 0, "records": 1})
    monkeypatch.setattr(rc, "_datagen_plan", lambda ctx: (4, 8))

    ctx = _agentic_ctx(tmp_path, "--agentic", "live",
                       tasks=["rmsnorm_aiter", held.task_id])
    rc._stage_agentic(ctx)

    assert seen["task_ids"] == ["rmsnorm_aiter"]
    assert seen["kinds"] == pdg.AGENTIC_KINDS
    assert seen["kw"]["completion_gate"] == "quota_only"


def test_agentic_both_mode_runs_synth_first_then_the_live_path(tmp_path, monkeypatch):
    import kore.data.parallel_datagen as pdg

    order: list = []
    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    monkeypatch.setattr(rc, "_stage_agentic_synth",
                        lambda ctx: order.append("synth"))
    monkeypatch.setattr(pdg, "run_parallel_datagen",
                        lambda *a, **k: order.append("live") or
                        {"done": 1, "skip": 0, "error": 0, "records": 1})
    monkeypatch.setattr(rc, "_datagen_plan", lambda ctx: (4, 8))

    rc._stage_agentic(_agentic_ctx(tmp_path, "--agentic", "both"))
    assert order == ["synth", "live"]


def test_agentic_dry_run_touches_nothing(tmp_path, monkeypatch):
    _forbid_live_agentic(monkeypatch)
    ctx = _agentic_ctx(tmp_path)
    ctx["dry"] = True
    rc._stage_agentic(ctx)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# 6. _stage_reverify - re-measure EXISTING kernels, never the held-out ones
# --------------------------------------------------------------------------- #
def test_reverify_never_touches_a_heldout_task(tmp_path, monkeypatch):
    """Re-verification re-measures kernels on the GPU. Doing that for a held-out
    task would make its measurements part of the training loop."""
    import kore.data.reverify as reverify

    held = heldout_tasks()[0]
    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    _seed_heldout_shards(tmp_path, held)
    seen: dict = {}
    monkeypatch.setattr(reverify, "run_reverify",
                        lambda root, task_ids, gpus, **kw: seen.update(
                            task_ids=list(task_ids), gpus=list(gpus)) or {"n": 1})
    monkeypatch.setattr(rc, "_gpu_ids", lambda ctx: [3])
    monkeypatch.setattr(rc, "_log_datagen_coverage", lambda ctx: None)

    ctx = _datagen_ctx(tmp_path, tasks=["rmsnorm_aiter"])
    ctx["eval_task_ids"] = [held.task_id]
    rc._stage_reverify(ctx)

    assert seen["task_ids"] == ["rmsnorm_aiter"]
    assert held.task_id not in seen["task_ids"]
    assert seen["gpus"] == [3]


def test_reverify_skips_derived_and_evolve_shards(tmp_path, monkeypatch):
    """``_``-prefixed shards are minted derivatives and ``.evolve`` shards are
    already-verified evolutionary output; neither is a task to re-verify."""
    import kore.data.reverify as reverify

    _seed_train_shards(tmp_path, "rmsnorm_aiter", "rmsnorm", "rms")
    _write_shard(tmp_path, "wins", "_gold_from_groups",
                 [_win("rmsnorm_aiter", "rmsnorm", "gold")])
    _write_shard(tmp_path, "wins", "rmsnorm_aiter.evolve",
                 [_win("rmsnorm_aiter", "rmsnorm", "evolved")])
    seen: dict = {}
    monkeypatch.setattr(reverify, "run_reverify",
                        lambda root, task_ids, gpus, **kw: seen.update(
                            task_ids=list(task_ids)) or {"n": 1})
    monkeypatch.setattr(rc, "_log_datagen_coverage", lambda ctx: None)

    ctx = _datagen_ctx(tmp_path, tasks=["rmsnorm_aiter"])
    rc._stage_reverify(ctx)

    assert seen["task_ids"] == ["rmsnorm_aiter"]


def test_reverify_on_a_fresh_run_is_a_clean_no_op(tmp_path, monkeypatch):
    import kore.data.reverify as reverify

    monkeypatch.setattr(
        reverify, "run_reverify",
        lambda *a, **k: pytest.fail("reverify ran with nothing on disk"))
    rc._stage_reverify(_datagen_ctx(tmp_path, tasks=["rmsnorm_aiter"]))
