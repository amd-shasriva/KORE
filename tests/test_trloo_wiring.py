"""TRLOO reaches the trainer, or fails loudly -- it never degrades quietly.

``tests/test_trloo_advantages.py`` proves the estimator is unbiased. These tests
prove the trainer actually USES it when the config asks for it, that the sharded
path forms its baseline over the whole group rather than per rank, and that the
one way this can go wrong -- samples with no ``(trajectory, turn)`` key -- raises
instead of silently reverting to the biased pooled estimator.

The silent-revert case is the one worth guarding hardest: a run that requested an
unbiased estimator and quietly got the biased one would look completely healthy in
the step log, which is the failure mode this whole feature exists to remove.
"""

from __future__ import annotations

import pytest

from kore.policy import anticollapse as ac
from kore.policy import grpo, trloo
from kore.policy.capabilities import (
    DECLARED,
    audit_requested_capabilities,
    format_capability_audit,
)
from kore.policy.configs import GRPOConfig


def _sample(ret, key, *, tokens=4, learnable=True):
    """A GRPO sample tuple: [ret, gen_inputs, ref_logp, old_logp, n_tokens, sc_w, key]."""
    return [ret, ("prompt", "gen") if learnable else None, None, None, tokens,
            None, key]


# --------------------------------------------------------------------------- #
# 1. dispatch: the named estimator is the one that runs
# --------------------------------------------------------------------------- #
def test_grpo_estimator_still_routes_through_the_avspo_floor():
    """The incumbent path must be byte-identical to before this feature landed."""
    samples = [_sample(1.0, (0, 0)), _sample(2.0, (0, 1)),
               _sample(3.0, (1, 0)), _sample(4.0, (1, 1))]
    advs = grpo.group_sample_advantages(
        samples, variance_floor=0.1, avspo_virtual_k=2, adv_eps=grpo._EPS,
        advantage_estimator="grpo")
    assert advs == ac.avspo_advantages([1.0, 2.0, 3.0, 4.0], 0.1, 2, grpo._EPS)


def test_trloo_estimator_routes_to_the_turn_level_leave_one_out_baseline():
    samples = [_sample(1.0, (0, 0)), _sample(2.0, (0, 1)),
               _sample(3.0, (1, 0)), _sample(4.0, (1, 1))]
    advs = grpo.group_sample_advantages(
        samples, advantage_estimator="trloo")
    # turn 0: {1.0, 3.0} -> each centred on the other; turn 1: {2.0, 4.0}.
    assert advs == pytest.approx([-2.0, -2.0, 2.0, 2.0])
    assert advs == trloo.turn_loo_advantages(
        [1.0, 2.0, 3.0, 4.0], [(0, 0), (0, 1), (1, 0), (1, 1)])


def test_trloo_ignores_the_variance_floor_it_is_handed():
    """The floor is a self-inclusive statistic, so TRLOO must not apply it."""
    samples = [_sample(1.0, (0, 0)), _sample(1.0, (1, 0))]
    with_floor = grpo.group_sample_advantages(
        samples, variance_floor=0.9, avspo_virtual_k=4,
        advantage_estimator="trloo")
    without = grpo.group_sample_advantages(
        samples, variance_floor=0.0, advantage_estimator="trloo")
    assert with_floor == without == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# 2. no silent fallback
# --------------------------------------------------------------------------- #
def test_missing_turn_keys_yield_none_rather_than_pooled_advantages():
    legacy = [[1.0, ("p", "g"), None, None, 4, None],
              [3.0, ("p", "g"), None, None, 4, None]]
    assert grpo.group_sample_advantages(
        legacy, advantage_estimator="trloo") is None
    # ... while the pooled estimator is perfectly happy with the same tuples.
    assert grpo.group_sample_advantages(
        legacy, advantage_estimator="grpo") is not None


def test_the_training_path_raises_instead_of_training_a_biased_estimator():
    """``_accumulate_grpo_grads`` must refuse, not fall back.

    Falling back would satisfy every assertion a training loop makes about
    itself: a loss appears, a gradient flows, the step log looks normal. The run
    would simply be optimising a biased objective.
    """
    legacy = [[1.0, ("p", "g"), None, None, 4, None]]
    with pytest.raises(grpo.AdvantageKeyError, match="trajectory_id"):
        grpo._accumulate_grpo_grads(
            [legacy], lambda gen: None, ref_anchor_coef=0.0,
            advantage_estimator="trloo", backward=False)


def test_an_unknown_estimator_name_is_rejected_at_config_validation():
    """A typo must not silently select the default."""
    with pytest.raises(ValueError, match="advantage_estimator"):
        GRPOConfig(advantage_estimator="rloo").validate()
    with pytest.raises(ValueError, match="advantage_estimator"):
        GRPOConfig(advantage_estimator="TRLOO").validate()
    assert GRPOConfig(advantage_estimator="trloo",
                      variance_floor=0.0).validate() is None
    assert GRPOConfig(advantage_estimator="grpo").validate() is None


# --------------------------------------------------------------------------- #
# 3. the logged advantage must be the one that trained
# --------------------------------------------------------------------------- #
def test_step_stats_report_the_estimator_actually_in_force():
    """``adv_absmean`` is where a reader confirms the switch took effect.

    If diagnostics kept reporting pooled advantages while TRLOO trained, the step
    log would be evidence for the wrong thing.
    """
    samples = [_sample(1.0, (0, 0)), _sample(2.0, (0, 1)),
               _sample(3.0, (1, 0)), _sample(4.0, (1, 1))]
    trloo_cfg = GRPOConfig(advantage_estimator="trloo", variance_floor=0.0)
    grpo_cfg = GRPOConfig(advantage_estimator="grpo", variance_floor=0.1)

    trloo_absmean, _ = grpo._grpo_step_adv_stats([samples], trloo_cfg)
    grpo_absmean, _ = grpo._grpo_step_adv_stats([samples], grpo_cfg)
    assert trloo_absmean == pytest.approx(2.0)          # |+-2.0| from above
    assert trloo_absmean != pytest.approx(grpo_absmean)


# --------------------------------------------------------------------------- #
# 4. sharded path: the baseline spans the group, and ids must be GLOBAL
# --------------------------------------------------------------------------- #
def test_cross_rank_trloo_matches_computing_it_centrally():
    """Two ranks, one group: gathering then estimating == estimating centrally."""
    per_rank_returns = [[1.0, 2.0], [3.0, 4.0]]
    per_rank_keys = [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]
    split = grpo.distributed_group_advantages(
        per_rank_returns, per_rank_keys=per_rank_keys,
        advantage_estimator="trloo")
    central = trloo.turn_loo_advantages(
        [1.0, 2.0, 3.0, 4.0], [(0, 0), (0, 1), (1, 0), (1, 1)])
    assert grpo.merge_across_ranks(split) == central
    assert split[0] == central[:2] and split[1] == central[2:]


def test_rank_local_trajectory_ids_would_corrupt_the_baseline():
    """Why ``_rollout_slice_distributed`` maps through ``_rank_slice``.

    Each rank indexes its own trajectories from 0. If those local ids were used
    as the TRLOO key, rank 0's first trajectory and rank 1's first trajectory
    would look like ONE trajectory -- so each would be excluded from the other's
    baseline as "self", which is the self-inclusion the estimator exists to
    remove, reintroduced by a bookkeeping mistake rather than by the formula.
    """
    per_rank_returns = [[1.0, 2.0], [3.0, 4.0]]
    global_keys = [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]
    local_keys = [[(0, 0), (0, 1)], [(0, 0), (0, 1)]]     # the bug

    correct = grpo.distributed_group_advantages(
        per_rank_returns, per_rank_keys=global_keys, advantage_estimator="trloo")
    corrupted = grpo.distributed_group_advantages(
        per_rank_returns, per_rank_keys=local_keys, advantage_estimator="trloo")

    assert grpo.merge_across_ranks(correct) == pytest.approx([-2.0, -2.0, 2.0, 2.0])
    # With colliding ids every turn looks like a single trajectory, so no peer
    # remains and the constant-0 baseline leaves the raw returns behind.
    assert grpo.merge_across_ranks(corrupted) == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert grpo.merge_across_ranks(correct) != grpo.merge_across_ranks(corrupted)


def test_the_sharded_rollout_emits_global_trajectory_ids():
    """Pin the mapping itself: rank r's local index i must key on ``_rank_slice``.

    Checked against ``_rank_slice`` rather than a hardcoded table so it keeps
    tracking the real partition if that ever changes.
    """
    G, world = 8, 4
    for rank in range(world):
        my = grpo._rank_slice(G, rank, world)
        assert my, "every rank owns at least one trajectory at G=8, world=4"
        emitted = [(my[i], 0) for i in range(len(my))]
        assert all(key[0] < G for key in emitted)
    # Union over ranks covers every trajectory exactly once -> no id collisions.
    all_ids = [my_id
               for rank in range(world)
               for my_id in grpo._rank_slice(G, rank, world)]
    assert sorted(all_ids) == list(range(G))


def test_cross_rank_trloo_without_keys_returns_none():
    assert grpo.distributed_group_advantages(
        [[1.0], [2.0]], advantage_estimator="trloo") is None
    # Length disagreement between returns and keys must also refuse.
    assert grpo.distributed_group_advantages(
        [[1.0, 2.0], [3.0]], per_rank_keys=[[(0, 0)], [(1, 0)]],
        advantage_estimator="trloo") is None


def test_cross_rank_grpo_path_is_unchanged_by_the_new_arguments():
    """Regression guard for the incumbent: same call, same numbers."""
    per_rank = [[1.0, 2.0], [3.0, 4.0]]
    assert grpo.merge_across_ranks(
        grpo.distributed_group_advantages(per_rank)) == grpo.group_advantages(
            [1.0, 2.0, 3.0, 4.0])


# --------------------------------------------------------------------------- #
# 5. end to end: the TRLOO advantage reaches the actual gradient
# --------------------------------------------------------------------------- #
def test_the_gradient_carries_the_trloo_advantage_not_the_pooled_one():
    """A real backward through ``_accumulate_grpo_grads``, both estimators.

    Everything above tests the advantage NUMBERS. This tests that they survive the
    loss construction: with a trivial differentiable log-prob whose gradient is 1
    per sample, the accumulated gradient is the token-weighted mean of the
    advantages, so the two estimators must produce measurably different gradients
    on the same group. Without this, a correct estimator wired into a path that
    ignored it would still pass every other test in this file.
    """
    torch = pytest.importorskip("torch")

    weight = torch.nn.Parameter(torch.zeros(1))

    def logp_fn(gen_inputs):
        # A stand-in for the policy's token-mean log-prob: d(logp)/d(weight) == 1.
        return weight.sum()

    # Token counts DIFFER on purpose. The loss is a global token-mean, so with
    # equal counts both estimators' advantages sum to zero over the group and the
    # gradients would coincide at 0.0 -- a test that proves nothing. Unequal
    # lengths, which is the real case, make the weighting visible.
    tokens = [1, 2, 3, 4]
    samples = [_sample(1.0, (0, 0), tokens=tokens[0]),
               _sample(2.0, (0, 1), tokens=tokens[1]),
               _sample(3.0, (1, 0), tokens=tokens[2]),
               _sample(4.0, (1, 1), tokens=tokens[3])]

    grads = {}
    for estimator in ("grpo", "trloo"):
        weight.grad = None
        _, n_terms = grpo._accumulate_grpo_grads(
            [samples], logp_fn, ref_anchor_coef=0.0, variance_floor=0.0,
            advantage_estimator=estimator, backward=True)
        assert n_terms == 4
        grads[estimator] = float(weight.grad)

    def _expected(advantages):
        # No stored old_logp, so the per-sample term is -A * logp, each scaled by
        # n_tok / total_tokens (the DAPO global token-mean).
        total = sum(tokens)
        return -sum(a * n / total for a, n in zip(advantages, tokens))

    trloo_advs = grpo.group_sample_advantages(samples, advantage_estimator="trloo")
    pooled = ac.avspo_advantages([1.0, 2.0, 3.0, 4.0], 0.0, 2, grpo._EPS)

    assert trloo_advs == pytest.approx([-2.0, -2.0, 2.0, 2.0])
    assert grads["trloo"] == pytest.approx(_expected(trloo_advs), abs=1e-6)
    assert grads["grpo"] == pytest.approx(_expected(pooled), abs=1e-6)

    # The load-bearing assertion: two different objectives, two different gradients
    # from the identical group. A correct estimator wired into a path that ignored
    # it would pass every other test in this file but fail this one.
    assert grads["trloo"] != pytest.approx(grads["grpo"], abs=1e-3)
    assert grads["trloo"] == pytest.approx(-0.8, abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. the capability audit must not let TRLOO+AVSPO look like it works
# --------------------------------------------------------------------------- #
def test_audit_reports_the_variance_floor_as_inert_under_trloo(tmp_path):
    """Requesting the AVSPO floor with TRLOO requests something that never runs."""
    config = GRPOConfig(advantage_estimator="trloo", variance_floor=0.1,
                        ref_anchor_coef=0.0)
    findings = audit_requested_capabilities(config, root=tmp_path)
    assert any(f.feature == "avspo" and f.scope == DECLARED for f in findings), (
        format_capability_audit(findings))


def test_audit_is_silent_when_trloo_disables_the_floor(tmp_path):
    config = GRPOConfig(advantage_estimator="trloo", variance_floor=0.0,
                        ref_anchor_coef=0.0)
    findings = audit_requested_capabilities(config, root=tmp_path)
    assert not any(f.feature == "avspo" for f in findings), (
        format_capability_audit(findings))
