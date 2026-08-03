"""The external task pool: it must grow the task space without moving the registry.

Three claims are load-bearing and each is tested against the real registry and
the real decontamination module rather than a stub.

1. **The registry is untouched.** The whole point of a pool is that
   ``taxonomy_digest()`` -- which ``validate_split_manifest`` compares a
   serialized manifest against -- does not move when the trainable task count
   grows. A pool task that leaked into ``registry.all_tasks()`` would silently
   invalidate every in-flight campaign's manifest.

2. **A pool task means what a registry task means.** Its family comes from
   ``registry.operator_family`` and its split from the same taxonomy authority,
   so "trainable" is not a second, weaker notion.

3. **Admission is fail-closed.** Unsafe, unclassifiable, reserved-family,
   contaminated, duplicate, and non-deterministic modules are each dropped with
   a recorded reason. The gates are tested by feeding them the thing they exist
   to catch -- including a real held-out task's own seed source.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from kore.data import task_mining as mining
from kore.tasks import external
from kore.tasks import registry, taxonomy

torch = pytest.importorskip("torch")


SIMPLE_MODULE = '''
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Linear(c, c)
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(torch.nn.functional.gelu(self.fc(x)))


def get_inputs():
    return [torch.rand([4, 32, 64])]


def get_init_inputs():
    return [[], {'c': 64}]
'''


def _candidate(source=SIMPLE_MODULE, name="TinyMLP", entry="TinyMLP", source_id="kernelbook"):
    return mining.Candidate(source_id, "row-1", name, entry, source, {"dataset": "test"})


@pytest.fixture(scope="module")
def accepted_spec():
    outcome = mining.screen_candidate(_candidate())
    assert outcome.accepted, f"{outcome.reason}: {outcome.detail}"
    return outcome.spec


# --------------------------------------------------------------------------- #
# Claim 1: the registry does not move
# --------------------------------------------------------------------------- #
def test_building_a_pool_task_does_not_touch_the_registry(accepted_spec, tmp_path):
    before_digest = registry.taxonomy_digest()
    before_ids = set(registry.task_ids())

    external.materialize_external_task(accepted_spec, tmp_path)

    assert registry.taxonomy_digest() == before_digest
    assert set(registry.task_ids()) == before_ids
    assert accepted_spec.task_id not in before_ids
    # And the pool lives outside the directory the registry globs.
    assert registry.TASKS_DIR not in (tmp_path / accepted_spec.task_id).parents


def test_pool_ids_cannot_collide_with_a_registry_generator_namespace(accepted_spec):
    assert accepted_spec.task_id.startswith("kbk_")
    for prefix in ("gen_", "genb_", "genv_"):
        assert not accepted_spec.task_id.startswith(prefix)
    assert set(external.SOURCE_PREFIXES.values()) == {"kbk", "syn"}


def test_identity_is_content_addressed_so_a_rebuild_reproduces_it():
    first = external.make_identity("kernelbook", "TinyMLP", SIMPLE_MODULE, "fp32")
    again = external.make_identity("kernelbook", "TinyMLP", SIMPLE_MODULE, "fp32")
    other = external.make_identity("kernelbook", "TinyMLP", SIMPLE_MODULE + "\n#x", "fp32")

    assert first == again
    assert first != other, "a different module must not reuse an existing identity"


# --------------------------------------------------------------------------- #
# Claim 2: a pool task is classified by the registry's own authority
# --------------------------------------------------------------------------- #
def test_materialized_task_classifies_through_operator_family(accepted_spec, tmp_path):
    task = external.materialize_external_task(accepted_spec, tmp_path)

    # The same function the registry uses for its own tasks, with no special case.
    assert registry.operator_family(task) == accepted_spec.family
    assert registry.analysis_family(task) == taxonomy.analysis_family(
        accepted_spec.family
    )
    assert not registry.is_heldout(task)
    assert registry.split_decision(task).reason == "train"
    assert accepted_spec.family in taxonomy.PRODUCT_FAMILIES


def test_pool_task_declares_a_trainable_architecture_and_dtype(accepted_spec):
    assert external.POOL_GPU_TARGET in taxonomy.TRAIN_ARCHITECTURES
    assert accepted_spec.dtype in taxonomy.TRAIN_DTYPES
    assert external.split_decision_for_spec(accepted_spec).split == "train"


def test_a_reserved_family_declaration_cannot_be_overridden():
    """The minted branch refuses to relabel a reserved leaf as something trainable."""
    with pytest.raises(taxonomy.TaxonomyError):
        taxonomy.product_family_for_source("minted", "kbk_mla_decode", "gemm")


# --------------------------------------------------------------------------- #
# Claim 3: admission is fail-closed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("snippet,expected", [
    ("import os\nos.system('ls')\n", "forbidden"),
    ("import subprocess\n", "forbidden_import"),
    ("import torch\nx = eval('1')\n", "forbidden_call"),
    ("import torch\ntorch.load('w.pt')\n", "forbidden_call"),
    ("import requests\n", "forbidden_import"),
    ("def f(:\n", "syntax_error"),
])
def test_unsafe_modules_are_refused_before_execution(snippet, expected):
    reason = external.module_safety_reason(snippet)
    assert reason is not None and expected in reason


def test_execution_refuses_an_unsafe_module_even_from_a_written_spec():
    with pytest.raises(external.ExternalTaskError, match="refusing to execute"):
        external.exec_module_source("import socket\n")


def test_input_constructor_randomness_is_allowed_but_forward_randomness_is_not():
    """``get_inputs`` draws random inputs by design; ``forward`` may not."""
    assert external.nondeterminism_reason(SIMPLE_MODULE) is None

    leaky = SIMPLE_MODULE.replace(
        "return self.norm(", "return self.norm(torch.rand_like(x) * "
    )
    assert "nondeterministic_call" in (external.nondeterminism_reason(leaky) or "")
    outcome = mining.screen_candidate(_candidate(source=leaky))
    assert not outcome.accepted and outcome.reason == "nondeterministic_oracle"


def test_dropout_in_forward_is_caught_by_the_measured_determinism_probe():
    """``Module.eval()`` disables ``nn.Dropout``, so the static scan cannot see it.

    The functional form is what actually breaks an oracle, and it is caught by
    running the module twice rather than by reading it.
    """
    source = SIMPLE_MODULE.replace(
        "return self.norm(",
        "return self.norm(x.new_empty(x.shape).uniform_() * ",
    )
    outcome = mining.screen_candidate(_candidate(source=source))
    assert not outcome.accepted
    assert outcome.reason == "nondeterministic_oracle"


def test_an_unclassifiable_module_is_dropped_rather_than_given_a_raw_family():
    source = SIMPLE_MODULE.replace(
        "return self.norm(torch.nn.functional.gelu(self.fc(x)))", "return x"
    ).replace("self.fc = nn.Linear(c, c)", "pass").replace(
        "self.norm = nn.LayerNorm(c)", "pass"
    )
    assert external.classify_module(source, "Mystery") is None
    outcome = mining.screen_candidate(_candidate(source=source, name="Mystery"))
    assert not outcome.accepted and outcome.reason == "unclassifiable_operation"


def test_reserved_families_are_refused_on_source_evidence_not_just_names():
    reserved = SIMPLE_MODULE.replace("TinyMLP", "Block").replace(
        "def forward(self, x):", "def forward(self, x):\n        paged_attn = 1"
    )
    assert external.reserved_family_marker(reserved) == "paged_attn"
    assert external.classify_module(reserved, "Block") is None
    # A word that merely contains the marker is not a reserved family.
    assert external.reserved_family_marker("mlanguage model") is None


def test_constructor_arguments_must_be_plain_json_data():
    source = SIMPLE_MODULE.replace(
        "return [[], {'c': 64}]", "return [[], {'c': 64, 'w': torch.ones(4)}]"
    )
    outcome = mining.screen_candidate(_candidate(source=source))
    assert not outcome.accepted and outcome.reason == "unserializable_init_args"


def test_a_toy_shape_is_refused_as_an_optimization_target():
    """KernelBook's default input is 256 elements; a speedup there is launch overhead."""
    tiny = SIMPLE_MODULE.replace(
        "torch.rand([4, 32, 64])", "torch.rand([2, 2])"
    ).replace("{'c': 64}", "{'c': 2}").replace("nn.LayerNorm(c)", "nn.LayerNorm(c)")
    outcome = mining.screen_candidate(_candidate(source=tiny))
    assert not outcome.accepted
    assert outcome.reason in {"shape_too_small", "no_runnable_scale"}


def test_the_accepted_primary_shape_clears_the_optimization_floor(accepted_spec):
    elements = sum(
        int(torch.tensor(spec.sized(accepted_spec.primary_scale)).prod())
        for spec in accepted_spec.input_specs
    )
    assert elements >= external.MIN_PRIMARY_ELEMENTS
    assert accepted_spec.primary_scale > 1, "the toy upstream shape was not scaled"


# --------------------------------------------------------------------------- #
# Decontamination, against the real held-out set
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def decontaminator():
    return mining.Decontaminator()


def test_the_gate_refuses_to_run_with_an_empty_holdout_set(monkeypatch):
    monkeypatch.setattr(
        "kore.data.decontam.heldout_task_ids", lambda: frozenset()
    )
    with pytest.raises(RuntimeError, match="held-out gate is empty"):
        mining.Decontaminator()


def test_a_heldout_task_seed_is_recognized_as_contaminated(decontaminator):
    """The strongest available evidence: a real reserved task's own source."""
    heldout = sorted(registry.heldout_tasks(), key=lambda t: t.task_id)[0]
    source = (heldout.dir / "seed_triton.py").read_text()

    verdict = decontaminator.check(_candidate(source=source, name=heldout.operation))

    assert verdict is not None
    reason, evidence = verdict
    assert reason.startswith("decontam:")
    assert reason != "decontam:heldout_task_id_literal" or evidence["task_id"]


def test_a_literal_heldout_task_id_is_refused(decontaminator):
    heldout_id = sorted(registry.heldout_tasks())[0].task_id
    source = SIMPLE_MODULE + f"\n# ported from {heldout_id}\n"

    verdict = decontaminator.check(_candidate(source=source))

    assert verdict is not None
    assert verdict[0] == "decontam:heldout_task_id_literal"
    assert verdict[1]["task_id"] == heldout_id


def test_an_ordinary_module_is_not_flagged_by_boilerplate(decontaminator):
    assert decontaminator.check(_candidate()) is None


def test_benchmark_problems_are_indexed_as_reference_documents():
    problems = [{"code": SIMPLE_MODULE, "name": "1_Tiny", "level": "1"}]
    references = mining.benchmark_references(problems)
    assert references[0]["source_id"] == "kernelbench"
    assert references[0]["reference_id"] == "kernelbench:1:1_Tiny"

    gate = mining.Decontaminator(extra_references=references)
    verdict = gate.check(_candidate())
    assert verdict is not None, "a mined copy of a benchmark problem must be refused"
    assert verdict[0].startswith("decontam:")


def test_evidence_never_carries_the_contaminated_text(decontaminator):
    heldout_id = sorted(registry.heldout_tasks())[0].task_id
    verdict = decontaminator.check(
        _candidate(source=SIMPLE_MODULE + f"\n# {heldout_id}\n")
    )
    assert verdict is not None
    assert "text" not in verdict[1]


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #
def test_dedup_collapses_a_renamed_and_reformatted_copy():
    dedup = mining.Deduplicator()
    assert dedup.check(SIMPLE_MODULE, "a") is None

    renamed = SIMPLE_MODULE.replace("TinyMLP", "SmallMLP").replace(
        "self.fc", "self.dense"
    ).replace("def forward(self, x):", "def forward(self, inp):\n        x = inp")
    verdict = dedup.check(renamed, "b")

    assert verdict is not None and verdict[0].startswith("dedup:")
    assert verdict[1]["duplicate_of"] == "a"


def test_dedup_keeps_a_structurally_different_module():
    dedup = mining.Deduplicator()
    dedup.check(SIMPLE_MODULE, "a")
    other = SIMPLE_MODULE.replace(
        "self.norm(torch.nn.functional.gelu(self.fc(x)))",
        "self.norm(torch.softmax(self.fc(x), dim=-1)) + x",
    )
    assert dedup.check(other, "b") is None


def test_dedup_is_seeded_with_the_registry_so_a_pool_task_cannot_restate_one():
    source = (registry.all_tasks()[0].dir / "reference.py").read_text()
    dedup = mining.Deduplicator([source])
    verdict = dedup.check(source, "pool-task")
    assert verdict is not None
    assert verdict[1]["duplicate_of"] == "registry"


def test_admit_applies_both_shared_state_gates_and_counts_every_drop(decontaminator):
    good = mining.screen_candidate(_candidate())
    duplicate = mining.screen_candidate(
        _candidate(source=SIMPLE_MODULE.replace("TinyMLP", "OtherMLP"), name="OtherMLP")
    )
    rejected = mining.Outcome(_candidate(), False, "safety", "forbidden_import: os")

    accepted, report = mining.admit(
        [good, duplicate, rejected], decontaminator, mining.Deduplicator()
    )

    assert len(accepted) == 1
    assert report.considered == 3 and report.accepted == 1
    assert report.drops["safety"] == 1
    assert sum(v for k, v in report.drops.items() if k.startswith("dedup:")) == 1
    assert report.as_dict()["dropped"] == 2


# --------------------------------------------------------------------------- #
# The emitted task satisfies the driver ABI
# --------------------------------------------------------------------------- #
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_emitted_reference_satisfies_the_generic_driver_contract(accepted_spec, tmp_path):
    task = external.materialize_external_task(accepted_spec, tmp_path)
    reference = _load(task.reference_path, "pool_reference_contract")

    for attribute in ("parse_shape", "get_inputs", "ref_fn", "baseline_fn",
                      "entry_name", "mutates_input"):
        assert hasattr(reference, attribute), attribute
    assert reference.entry_name == accepted_spec.operation
    assert reference.mutates_input is False

    from kore.tasks import _genops

    shape = reference.parse_shape("S=4")
    inputs = reference.get_inputs(shape, device="cpu", seed=0)
    oracle = reference.ref_fn(*inputs)
    # ``_build_bench_pair`` refuses a reference that does not declare
    # ``mutates_input``; the tolerance/adversarial defaults must also resolve.
    assert _genops._tolerance_declarations(reference)["grid"] == (None,)
    assert _genops._adversarial_sets(reference, shape) is None
    assert _genops._output_pairs(reference.baseline_fn(*inputs), oracle) is not None


def test_the_seed_is_a_correct_starting_point_under_the_entry_name(accepted_spec, tmp_path):
    task = external.materialize_external_task(accepted_spec, tmp_path)
    reference = _load(task.reference_path, "pool_reference_seed")
    seed = _load(task.seed_path, "pool_seed")

    entry = getattr(seed, reference.entry_name)
    inputs = reference.get_inputs(reference.parse_shape("S=2"), device="cpu", seed=0)
    assert torch.equal(entry(*inputs), reference.ref_fn(*inputs))


def test_the_oracle_is_deterministic_and_scales_with_the_declared_shape(
    accepted_spec, tmp_path
):
    task = external.materialize_external_task(accepted_spec, tmp_path)
    reference = _load(task.reference_path, "pool_reference_shapes")

    def run(scale):
        return reference.ref_fn(
            *reference.get_inputs({"S": scale}, device="cpu", seed=0)
        )

    assert torch.equal(run(2), run(2))
    small, large = run(1), run(4)
    assert large.shape[0] == small.shape[0] * 4

    declared = {shape.name: shape.dims for shape in task.shapes}
    assert declared["minimal"] == {"S": 1}
    assert declared["primary"] == {"S": accepted_spec.primary_scale}


def test_the_embedded_spec_is_valid_python_for_every_field_type(accepted_spec, tmp_path):
    """JSON's ``true``/``null`` are not Python literals; the spec must be decoded."""
    task = external.materialize_external_task(accepted_spec, tmp_path)
    text = task.reference_path.read_text()
    assert "_SPEC = json.loads(" in text
    compile(text, "reference.py", "exec")

    restored = external.ExternalTaskSpec.from_dict(
        json.loads(json.dumps(accepted_spec.to_dict()))
    )
    assert restored.to_dict() == accepted_spec.to_dict()


# --------------------------------------------------------------------------- #
# The pool on disk
# --------------------------------------------------------------------------- #
def test_index_round_trips_and_only_trainable_ids_are_offered(accepted_spec, tmp_path):
    external.write_pool_index([accepted_spec], tmp_path / external.POOL_INDEX_NAME)
    restored = external.load_pool_specs(tmp_path)

    assert [spec.to_dict() for spec in restored] == [accepted_spec.to_dict()]
    assert external.pool_train_task_ids(tmp_path) == (accepted_spec.task_id,)

    external.materialize_pool(tmp_path, restored)
    assert [task.task_id for task in external.load_pool(tmp_path)] == [
        accepted_spec.task_id
    ]


def test_resolve_prefers_the_pool_and_falls_back_to_the_registry(accepted_spec, tmp_path):
    external.materialize_pool(tmp_path, [accepted_spec])

    pooled = external.resolve_task(accepted_spec.task_id, tmp_path)
    assert pooled.task_id == accepted_spec.task_id

    known = registry.task_ids()[0]
    assert external.resolve_task(known, tmp_path).task_id == known
    with pytest.raises(KeyError):
        external.resolve_task("definitely_not_a_task", tmp_path)


def test_a_hand_edited_index_cannot_widen_the_train_set(accepted_spec, tmp_path):
    """Re-deriving the split at read time is what makes the index untrusted input."""
    payload = accepted_spec.to_dict()
    payload["task_id"] = sorted(registry.heldout_tasks())[0].task_id
    payload["operation"] = "mla_decode"
    path = tmp_path / external.POOL_INDEX_NAME
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    assert external.pool_train_task_ids(tmp_path) == ()


def test_pool_root_prefers_the_explicit_argument_then_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("KORE_TASK_POOL", str(tmp_path / "from-env"))
    assert external.pool_root(tmp_path / "explicit") == tmp_path / "explicit"
    assert external.pool_root() == tmp_path / "from-env"
    monkeypatch.delenv("KORE_TASK_POOL")
    assert external.pool_root().name == "task_pool"


# --------------------------------------------------------------------------- #
# The synthetic source
# --------------------------------------------------------------------------- #
def test_synthesis_is_deterministic_and_every_module_is_admissible():
    from kore.tasks import synth_modules

    first = synth_modules.synthesize(40, seed=3)
    assert first == synth_modules.synthesize(40, seed=3)
    assert first != synth_modules.synthesize(40, seed=4)

    for _, name, source in first:
        assert external.module_safety_reason(source) is None, name
        assert external.nondeterminism_reason(source) is None, name
        assert external.classify_module(source, name) is not None, name


def test_synthesized_modules_survive_the_full_screen():
    from kore.tasks import synth_modules

    outcomes = [
        mining.screen_candidate(
            mining.Candidate("synthetic", name, name, name, source, {})
        )
        for _, name, source in synth_modules.synthesize(12, seed=1)
    ]
    assert all(outcome.accepted for outcome in outcomes), [
        (o.candidate.module_name, o.reason, o.detail)
        for o in outcomes if not o.accepted
    ]
    assert all(o.spec.task_id.startswith("syn_") for o in outcomes)
