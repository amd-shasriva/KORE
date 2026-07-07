"""CPU-only tests for on-policy relabeling (DAgger / iterative-DPO).

No GPU, no torch/trl/transformers. A ``FakePolicy`` (the TeacherClient duck type:
``generate(messages) -> str``) returns scripted kernels wrapped in the
FULL_KERNEL contract; a ``FakeEnv`` returns scripted verifier ``Observation``s
keyed by markers in the kernel source. Covers:

  * relabel_groups_on_policy uses the POLICY (not a teacher) to build groups
  * dagger_repairs mines the policy's OWN failures and gets verified expert fixes
    (and drops failures the teacher can't fix)
  * the teacher_frac beta schedule + teacher-only mixing
  * iterative_dpo aggregates the union of rounds and refreshes the reference
  * the IPO / cDPO DPOConfig path constructs on CPU
  * rft can consume on-policy wins
"""

from __future__ import annotations

from kore.data.onpolicy import (
    relabel_groups_on_policy,
    dagger_repairs,
    dagger_teacher_frac,
    iterative_dpo,
)
from kore.data.schemas import RankedGroupRecord, RepairRecord, WinRecord
from kore.data.teacher import StubTeacher, TeacherClient
from kore.reward.reward import Observation


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeTask:
    task_id = "fake_gemm_bf16"
    operation = "gemm"
    dtype = "bf16"
    gpu_target = "gfx942"
    seed_source = "def seed():\n    return 0"


def _wrap(src: str) -> str:
    return f"FULL_KERNEL:\n```python\n{src}\n```\n"


def _obs_bad_compile():
    return Observation(compiled=False, dtype="bf16", error_text="SyntaxError: bad")


def _obs_incorrect():
    return Observation(compiled=True, dtype="bf16", validation_passed=False,
                       snr_by_shape={"primary": 5.0}, snr_db=5.0,
                       error_text="worst SNR 5.0 < 25.0")


def _obs_correct_slow():
    return Observation(compiled=True, dtype="bf16", validation_passed=True,
                       snr_by_shape={"primary": 40.0}, snr_db=40.0,
                       wall_by_shape={"primary": 2.0}, baseline_by_shape={"primary": 2.0},
                       wall_ms=2.0, baseline_ms=2.0)


def _obs_correct_fast():
    return Observation(compiled=True, dtype="bf16", validation_passed=True,
                       snr_by_shape={"primary": 41.0}, snr_db=41.0,
                       wall_by_shape={"primary": 1.0}, baseline_by_shape={"primary": 2.0},
                       wall_ms=1.0, baseline_ms=2.0)


class FakeEnv:
    """Markers: ``__BAD__`` compile-fail, ``__WRONG__`` incorrect, ``__FAST__``
    correct 2x, anything else correct-but-slow."""

    def __init__(self):
        self.calls: list = []

    def step(self, source, full_validation=True, multi_shape=True):
        self.calls.append((source, full_validation, multi_shape))
        if "__BAD__" in source:
            return _obs_bad_compile()
        if "__WRONG__" in source:
            return _obs_incorrect()
        if "__FAST__" in source:
            return _obs_correct_fast()
        return _obs_correct_slow()


class FakePolicy:
    """A policy = the TeacherClient duck type. Returns ``script[i % len]`` wrapped
    in the FULL_KERNEL contract, recording every call."""

    def __init__(self, script: list[str]):
        self.script = script
        self.calls: list = []

    def generate(self, messages: list[dict]) -> str:
        i = len(self.calls)
        self.calls.append(list(messages))
        return _wrap(self.script[i % len(self.script)])


# --------------------------------------------------------------------------- #
# 1. relabel_groups_on_policy
# --------------------------------------------------------------------------- #
def test_fake_policy_is_teacherclient():
    assert isinstance(FakePolicy(["x"]), TeacherClient)


def test_relabel_groups_uses_policy_not_teacher():
    policy = FakePolicy(["cand __FAST__", "cand slow", "cand __WRONG__"])
    env = FakeEnv()
    groups = relabel_groups_on_policy(FakeTask(), policy, env, n_parents=2, k=3, seed=0)

    assert len(groups) == 2
    assert all(isinstance(g, RankedGroupRecord) for g in groups)
    # the POLICY generated every candidate (2 parents x 3 = 6 generate() calls)
    assert len(policy.calls) == 6
    # at least one real preference (fast beats wrong / slow)
    assert any(g.preferences for g in groups)
    # candidates carry the policy's own sources
    srcs = {c["source"] for g in groups for c in g.candidates}
    assert "cand __FAST__" in srcs


# --------------------------------------------------------------------------- #
# 2. dagger_repairs
# --------------------------------------------------------------------------- #
def test_dagger_repairs_collects_policy_failures():
    policy = FakePolicy(["cand __WRONG__", "cand __BAD__"])
    teacher = StubTeacher(fn=lambda m: _wrap("cand good"))  # expert fix validates
    env = FakeEnv()
    recs = dagger_repairs(FakeTask(), policy, teacher, env, n=2, seed=1)

    assert len(recs) >= 1
    assert len(policy.calls) > 0  # the policy was rolled to find failures
    classes = set()
    for r in recs:
        assert isinstance(r, RepairRecord)
        assert r.failure_class in ("snr_fail", "compile_fail")
        # verified fix stored in the diagnose-then-fix format
        assert "<answer>" in r.messages[-1]["content"]
        classes.add(r.failure_class)
    # both a compile fail (__BAD__) and an snr fail (__WRONG__) get repaired
    assert classes == {"snr_fail", "compile_fail"}


def test_dagger_repairs_drops_unfixed_failures():
    policy = FakePolicy(["cand __WRONG__"])
    # the teacher's "fix" is ALSO wrong -> never validates -> no record emitted
    teacher = StubTeacher(fn=lambda m: _wrap("cand __WRONG__"))
    env = FakeEnv()
    recs = dagger_repairs(FakeTask(), policy, teacher, env, n=1, seed=1)
    assert recs == []


def test_dagger_teacher_frac_decays_30_to_0():
    assert abs(dagger_teacher_frac(0, 4) - 0.30) < 1e-9
    assert abs(dagger_teacher_frac(3, 4) - 0.0) < 1e-9
    assert 0.0 < dagger_teacher_frac(1, 4) < 0.30
    assert dagger_teacher_frac(0, 1) == 0.0  # single round -> end value


def test_dagger_teacher_frac_mixes_teacher_only_failures():
    # teacher_frac=1.0 -> n_policy=0, so the policy is never rolled; all failures
    # come from mining the teacher's own natural failures.
    policy = FakePolicy(["cand good"])
    state = {"i": 0}

    def tfn(_msgs):
        i = state["i"]
        state["i"] = i + 1
        # first call: a failing candidate; then: the validated fix
        return _wrap("cand __WRONG__") if i == 0 else _wrap("cand good")

    teacher = StubTeacher(fn=tfn)
    env = FakeEnv()
    recs = dagger_repairs(FakeTask(), policy, teacher, env, n=1, seed=1, teacher_frac=1.0)
    assert len(recs) == 1
    assert len(policy.calls) == 0  # teacher-only round


# --------------------------------------------------------------------------- #
# 3. iterative_dpo aggregation + reference refresh
# --------------------------------------------------------------------------- #
def _fast_slow_policy_factory(_round, _prev):
    return FakePolicy(["cand __FAST__", "cand slow", "cand __WRONG__"])


def test_iterative_dpo_aggregation_grows_the_set():
    rounds = iterative_dpo(3, _fast_slow_policy_factory, FakeTask(),
                           lambda t: FakeEnv(), n_parents=2, k=3, seed=0)
    assert len(rounds) == 3

    # union with prior rounds: 2 groups added per round -> 2, 4, 6
    agg_counts = [len(rd.groups_agg) for rd in rounds]
    assert agg_counts == [2, 4, 6]
    assert [len(rd.groups_new) for rd in rounds] == [2, 2, 2]

    # DPO pairs (built from the aggregated union) are non-decreasing and positive
    pairs = [rd.n_pairs for rd in rounds]
    assert pairs[0] > 0
    assert pairs[0] <= pairs[1] <= pairs[2]
    assert rounds[-1].dpo_pairs and len(rounds[-1].dpo_pairs) == pairs[-1]

    # no train_fn -> reference never refreshed
    assert all(rd.ref_model_id is None for rd in rounds)


def test_iterative_dpo_no_aggregation_uses_latest_round_only():
    rounds = iterative_dpo(3, _fast_slow_policy_factory, FakeTask(),
                           lambda t: FakeEnv(), n_parents=2, k=3, seed=0,
                           aggregate=False)
    assert [len(rd.groups_agg) for rd in rounds] == [2, 2, 2]


def test_iterative_dpo_train_fn_refreshes_reference():
    seen_prev: list = []

    def policy_factory(round_idx, prev_ckpt):
        seen_prev.append(prev_ckpt)
        return FakePolicy(["cand __FAST__", "cand slow"])

    def train_fn(rd):
        return f"ckpt-round-{rd.round}"

    rounds = iterative_dpo(2, policy_factory, FakeTask(), lambda t: FakeEnv(),
                           n_parents=1, k=2, seed=0, train_fn=train_fn)

    # round 0 starts from None; round 1's policy is built from round 0's checkpoint
    assert seen_prev == [None, "ckpt-round-0"]
    assert rounds[0].ref_model_id is None
    assert rounds[1].ref_model_id == "ckpt-round-0"  # reference refresh
    assert rounds[0].policy_ckpt == "ckpt-round-0"
    assert rounds[1].policy_ckpt == "ckpt-round-1"


# --------------------------------------------------------------------------- #
# 4. IPO / cDPO config path
# --------------------------------------------------------------------------- #
def test_ipo_config_path_constructs():
    from kore.policy.configs import DPOConfig
    from kore.policy.dpo import build_trl_dpo_kwargs, dpo_config_from_dict

    cfg = DPOConfig(model_id="m", dataset_path="p.jsonl")
    cfg.loss_type = "ipo"  # attribute, no schema change
    kw = build_trl_dpo_kwargs(cfg)
    assert kw["loss_type"] == "ipo"
    assert abs(kw["beta"] - 0.1) < 1e-9
    assert kw["max_length"] == cfg.max_length

    # cDPO via label smoothing, threaded through the JSON config path
    cfg2 = dpo_config_from_dict({"model_id": "m", "dataset_path": "p",
                                 "loss_type": "sigmoid", "label_smoothing": 0.1})
    assert cfg2.loss_type == "sigmoid"
    assert abs(cfg2.label_smoothing - 0.1) < 1e-9
    kw2 = build_trl_dpo_kwargs(cfg2)
    assert abs(kw2["label_smoothing"] - 0.1) < 1e-9

    # a plain config (no IPO/cDPO fields) still builds and omits the keys
    kw3 = build_trl_dpo_kwargs(DPOConfig(model_id="m", dataset_path="p"))
    assert "loss_type" not in kw3 and "label_smoothing" not in kw3


def test_dpo_reference_refresh_field():
    from kore.policy.configs import DPOConfig

    # per-round reference refresh: point ref at the previous round's checkpoint
    cfg = DPOConfig(model_id="round1_ckpt", ref_model_id="round0_ckpt", dataset_path="p")
    assert cfg.ref_model_id == "round0_ckpt"


# NB: on-policy >1x win selection for RFT now lives in kore.data.rejection
# (stratified_rft_select), tested in tests/test_rejection.py. The old
# kore.policy.rft module was removed as superseded dead code.
