"""CPU-only tests for the publishable-EVAL frontier modules.

Covers the three new modules end to end, entirely on CPU with fake candidate
functions and fabricated Observations (no GPU, no campaign, no training):

  * kore.eval.kernelbench_amd  - KernelBench spec <-> KORE Task, fast_p roundtrip,
    backend tagging, and the wider held-out protocol + leakage check.
  * kore.eval.robust_eval      - the anti-hack correctness battery: it must catch a
    constant/memset kernel and a precision-downgraded kernel that naive allclose
    waves through, and its metamorphic relations must flag a permutation-breaking
    reduction.
  * kore.eval.paired_stats     - paired bootstrap CI + sign/Wilcoxon p-values on
    synthetic paired deltas (correct sign, coverage, and significance behavior).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from kore.eval import kernelbench_amd as kb
from kore.eval import paired_stats as ps
from kore.eval import robust_eval as re_
from kore.reward.reward import Observation


# --------------------------------------------------------------------------- #
# Shared fakes.
# --------------------------------------------------------------------------- #
def _benign_policy(task, feedback=None):
    # A neutral kernel string that trips NONE of the anti-hack patterns (no torch
    # ops, no @ operator, no vendor imports, no oracle call).
    return "def kernel(*args):\n    return compute(*args)\n"


def _obs(speedup: float, snr: float = 90.0) -> Observation:
    # Correct + timed: baseline_ms=1.0, wall=1/speedup -> worst-shape speedup exact.
    return Observation(
        compiled=True, snr_db=snr, wall_ms=1.0 / speedup, baseline_ms=1.0,
        wall_by_shape={"s": 1.0 / speedup}, baseline_by_shape={"s": 1.0},
        snr_by_shape={"s": snr}, validation_passed=True,
    )


# =========================================================================== #
# 1. KernelBench-AMD adapter
# =========================================================================== #
def test_bundled_specs_cover_the_three_classes():
    specs = kb.bundled_specs()
    families = {s.family for s in specs}
    # elementwise + gemm + a fusion class are all represented.
    assert "elementwise" in families
    assert "gemm" in families
    assert families & {"fusion", "gemm_fusion"}
    # levels 1 and 2 both present (single-op + fused), like KernelBench.
    assert {s.level for s in specs} == {1, 2}


def test_spec_to_task_backend_tagging():
    spec = next(s for s in kb.bundled_specs() if s.family == "gemm")

    t950 = kb.spec_to_task(spec)                       # default arch
    assert t950.gpu_target == "gfx950"
    assert t950.backend == "triton"
    assert t950.comparison_baseline == kb.KERNELBENCH_BASELINE == "torch_eager"
    assert t950.raw["level"] == spec.level
    assert t950.raw["family"] == spec.family
    assert getattr(t950, "kernelbench_spec", None) is spec

    t942 = kb.spec_to_task(spec, gpu_target="gfx942")  # CDNA3 override
    assert t942.gpu_target == "gfx942"
    assert t942.dtype == spec.dtype


def test_kernelbench_spec_to_task_fastp_roundtrip():
    # spec -> KORE Task -> matched-budget fast_p -> field-standard KB report.
    spec = next(s for s in kb.bundled_specs() if s.family == "gemm")
    task = kb.spec_to_task(spec, gpu_target="gfx942")

    from kore.eval.bakeoff import evaluate_policy
    dry = {task.task_id: [_obs(2.0)]}   # correct + exactly 2x
    res = evaluate_policy(_benign_policy, [task], budget=1, dry_run=dry,
                          ps=(1.0, 1.5, 2.0))

    report = kb.to_kernelbench_report(res, [spec])
    assert report["benchmark"] == "KernelBench-AMD"
    assert report["baseline"] == "torch_eager"
    assert report["n"] == 1 and report["num_correct"] == 1
    # 2x is correct AND > 1x and > 1.5x, but NOT > 2x -> field-standard fast_p.
    assert report["fast_p"][1.0] == 1.0
    assert report["fast_p"][1.5] == 1.0
    assert report["fast_p"][2.0] == 0.0
    assert report["fast_1"] == 1.0
    # per-level breakdown carries the level-1 gemm.
    assert report["per_level"]["level_1"]["fast_p"][1.0] == 1.0
    # formatting is non-empty ascii markdown.
    md = kb.format_kernelbench_report(report)
    assert "KernelBench-AMD" in md and "fast_p" in md


def test_run_kernelbench_amd_end_to_end_dry_run():
    specs = kb.bundled_specs()
    tasks = kb.specs_to_tasks(specs)
    # relu 2x (fast), matmul 1.2x, add_mul 0.8x (slower), matmul_relu 3x.
    speeds = {"kb_l1_relu": 2.0, "kb_l1_matmul": 1.2,
              "kb_l2_add_mul": 0.8, "kb_l2_matmul_relu": 3.0}
    dry = {t.task_id: [_obs(speeds[t.task_id])] for t in tasks}
    out = kb.run_kernelbench_amd(_benign_policy, specs, gpu_target="gfx950",
                                 budget=1, dry_run=dry)
    rep = out["report"]
    assert out["gpu_target"] == "gfx950"
    assert rep["n"] == 4 and rep["num_correct"] == 4
    # 3 of 4 beat 1x (2.0, 1.2, 3.0); 0.8 does not -> fast_1 = 3/4.
    assert abs(rep["fast_1"] - 0.75) < 1e-9
    # only matmul_relu (3x) and relu (2x) clear >1.5x... 2.0>1.5 and 3.0>1.5 -> 2/4.
    assert abs(rep["fast_p"][1.5] - 0.5) < 1e-9
    # only 3x clears >2x -> 1/4.
    assert abs(rep["fast_p"][2.0] - 0.25) < 1e-9


# =========================================================================== #
# 2. robust_eval anti-hack correctness battery
# =========================================================================== #
def _relu_ref(x):
    return torch.relu(x)


def _relu_inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(64, 96, generator=g, dtype=torch.float32),)


def test_robust_good_kernel_passes_full_battery():
    rep = re_.robust_correctness(_relu_ref, _relu_ref, _relu_inputs, family="elementwise")
    assert rep.passed is True
    assert rep.failed_check is None
    # random inits + adversarial + non-contiguous + differential all ran.
    assert rep.n_checks >= 4


def test_robust_catches_constant_memset_kernel():
    # A memset "kernel" that returns a constant regardless of input.
    def cand_const(x):
        return torch.zeros_like(x)

    rep = re_.robust_correctness(cand_const, _relu_ref, _relu_inputs, family="elementwise")
    assert rep.passed is False
    # relu of random input is not all-zero, so it dies at the very first check.
    assert rep.failed_check == "random_inits"


def test_robust_catches_precision_downgrade_that_allclose_passes():
    # A kernel that quantizes the output to a coarse (1/64) grid: it stays within a
    # loose allclose tolerance of the reference, but is FAR less accurate than fp32
    # warrants -> only the differential fp64 oracle catches it.
    def cand_downgrade(x):
        y = torch.relu(x)
        return torch.round(y * 64.0) / 64.0

    x0 = _relu_inputs(0)[0]
    # Naive allclose (the weak gate) is fooled ...
    assert torch.allclose(cand_downgrade(x0), _relu_ref(x0), atol=1e-2, rtol=1e-2)
    # ... and so is a plain reseeded-SNR check.
    rnd = re_.check_random_inits(cand_downgrade, _relu_ref, _relu_inputs)
    assert rnd.passed is True
    # ... but the differential oracle rejects the precision downgrade.
    diff = re_.check_differential_oracle(cand_downgrade, _relu_ref, _relu_inputs)
    assert diff.passed is False
    assert diff.metrics["cand_rel_err"] > diff.metrics["ref_rel_err"]
    # and the full battery therefore rejects it too.
    rep = re_.robust_correctness(cand_downgrade, _relu_ref, _relu_inputs, family="elementwise")
    assert rep.passed is False


def test_metamorphic_flags_permutation_breaking_reduction():
    # Reference is a permutation-INVARIANT reduction (row sum).
    def ref_sum(x):
        return x.sum(-1)

    def red_inputs(seed: int):
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(32, 40, generator=g, dtype=torch.float32),)

    # A candidate that depends on element ORDER (returns the first element) breaks
    # permutation-invariance while a genuine reduction preserves it.
    def cand_perm_break(x):
        return x[..., 0]

    # The metamorphic permutation relation flags the order-dependent candidate ...
    bad = re_.metamorphic_permutation_invariance(cand_perm_break, ref_sum, red_inputs)
    assert bad.passed is False
    # ... and confirms a genuine reduction preserves permutation-invariance.
    good = re_.metamorphic_permutation_invariance(ref_sum, ref_sum, red_inputs)
    assert good.passed is True

    # It is wired into the family battery for reductions; the battery rejects the
    # permutation-breaker (this candidate is gross enough to also fail the random
    # check, so we only assert overall rejection, not which check fired first).
    rep = re_.robust_correctness(cand_perm_break, ref_sum, red_inputs, family="reduction")
    assert rep.passed is False
    # The permutation relation is part of the reduction battery.
    assert re_.metamorphic_permutation_invariance in re_.metamorphic_relations_for("reduction")


def test_adversarial_and_noncontiguous_pass_for_faithful_kernel():
    adv = re_.check_adversarial_regimes(_relu_ref, _relu_ref, _relu_inputs)
    assert adv.passed is True
    nc = re_.check_noncontiguous(_relu_ref, _relu_ref, _relu_inputs)
    assert nc.passed is True


def test_metamorphic_homogeneity_for_linear_op():
    def ref_mm(a, b):
        return a @ b

    def mm_inputs(seed: int):
        g = torch.Generator().manual_seed(seed)
        k = 32
        sc = 1.0 / math.sqrt(k)
        return (torch.randn(16, k, generator=g) * sc, torch.randn(k, 20, generator=g) * sc)

    # matmul is homogeneous: f(a*x)=a*f(x); a faithful kernel preserves it ...
    good = re_.metamorphic_homogeneity(ref_mm, ref_mm, mm_inputs)
    assert good.passed is True

    # ... a kernel that adds a constant bias breaks homogeneity.
    def cand_biased(a, b):
        return a @ b + 3.0

    bad = re_.metamorphic_homogeneity(cand_biased, ref_mm, mm_inputs)
    assert bad.passed is False


def test_robust_over_kernelbench_spec_bridge():
    # Integration: run the battery directly over a bundled KernelBench spec via the
    # inputs-factory bridge (spec.reference is its own faithful kernel).
    spec = next(s for s in kb.bundled_specs() if s.family == "elementwise")
    factory = re_.inputs_factory_from_spec(spec)
    rep = re_.robust_correctness(spec.reference, spec.reference, factory, family=spec.family)
    assert rep.passed is True


# =========================================================================== #
# 3. paired_stats
# =========================================================================== #
def test_paired_bootstrap_positive_effect_significant():
    rng = np.random.default_rng(0)
    deltas = rng.normal(0.30, 0.10, size=40)   # clearly-positive per-task deltas
    cmp = ps.paired_comparison(deltas=deltas, seed=1)

    # effect size ~ sample mean, and the bootstrap mean brackets it.
    assert abs(cmp.effect_size - float(np.mean(deltas))) < 0.02
    assert cmp.ci[0] <= cmp.effect_size <= cmp.ci[1]
    # a real positive effect: CI excludes 0, direction + significance agree.
    assert cmp.ci[0] > 0.0
    assert cmp.direction == "kore_better"
    assert cmp.significant is True
    assert cmp.p_value < 0.05
    # all three paired tests agree it is significant.
    assert cmp.p_bootstrap < 0.05 and cmp.p_sign < 0.05 and cmp.p_wilcoxon < 0.05


def test_paired_null_not_significant():
    # A genuine null: deltas symmetric about 0 (equal +/- magnitudes) -> no effect,
    # so a correct paired test must NOT reject. Symmetric-exact keeps this
    # deterministic (a fixed random draw could spuriously clear alpha ~5% of runs).
    mags = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    deltas = np.array(mags + [-m for m in mags])
    cmp = ps.paired_comparison(deltas=deltas, seed=2)
    assert abs(cmp.effect_size) < 1e-9         # mean is exactly 0
    assert cmp.ci[0] < 0.0 < cmp.ci[1]         # CI straddles 0
    assert cmp.significant is False
    assert cmp.p_wilcoxon > 0.05
    assert cmp.p_sign == 1.0                   # balanced signs


def test_sign_test_exact_closed_form():
    # 8 all-positive deltas: two-sided p = 2 * (0.5)**8 = 0.0078125.
    st = ps.sign_test([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert st.n_pos == 8 and st.n_neg == 0
    assert abs(st.p_value - 2.0 * (0.5 ** 8)) < 1e-12
    # a balanced split is maximally non-significant (p capped at 1.0).
    st2 = ps.sign_test([1.0, -1.0, 2.0, -2.0])
    assert st2.p_value == 1.0


def test_wilcoxon_direction_and_ties():
    # symmetric with a zero (dropped) -> not significant.
    w0 = ps.wilcoxon_signed_rank([1.0, -1.0, 2.0, -2.0, 0.0])
    assert w0.n_effective == 4
    assert w0.p_value > 0.5
    # strongly positive -> small p, W+ near the max n(n+1)/2.
    w1 = ps.wilcoxon_signed_rank([0.5, 0.9, 1.2, 1.6, 2.0, 2.4, 3.1, 3.9])
    assert w1.p_value < 0.05
    assert w1.statistic == 8 * 9 / 2


def test_paired_speedup_ratio_effect():
    kore = np.array([1.5, 1.8, 2.1, 1.2, 1.9, 1.6, 2.3, 1.4, 1.7, 2.0])
    base = np.array([1.1, 1.2, 1.0, 1.05, 1.3, 1.15, 1.25, 1.0, 1.1, 1.2])
    cmp = ps.paired_speedup_comparison(kore, base, seed=3)
    assert cmp.effect_kind == "geomean_speedup_ratio"
    # geometric-mean ratio matches the direct computation.
    expected = float(np.exp(np.mean(np.log(kore) - np.log(base))))
    assert abs(cmp.effect_size - expected) < 0.02
    # KORE is faster: ratio CI excludes 1.0 and it reads significant.
    assert cmp.ci[0] > 1.0
    assert cmp.significant is True
    assert cmp.direction == "kore_better"


def test_paired_bootstrap_is_deterministic():
    deltas = [0.1, -0.2, 0.3, 0.05, 0.4, -0.1, 0.25, 0.15]
    a = ps.paired_bootstrap(deltas, seed=42, n_boot=2000)
    b = ps.paired_bootstrap(deltas, seed=42, n_boot=2000)
    assert a.ci_lo == b.ci_lo and a.ci_hi == b.ci_hi and a.p_value == b.p_value


def test_paired_bootstrap_ci_coverage_calibrated():
    # Deterministic coverage sanity: over many fixed-seed synthetic paired samples
    # drawn from a known-mean population, the 90% bootstrap CI should cover the true
    # mean roughly 90% of the time (percentile bootstrap under-covers slightly at
    # small n, so we assert a generous, but non-trivial, band).
    rng = np.random.default_rng(2024)
    true_mu, n, trials = 0.25, 25, 200
    covered = 0
    for i in range(trials):
        sample = rng.normal(true_mu, 0.3, size=n)
        r = ps.paired_bootstrap(sample, n_boot=400, ci_level=0.90, seed=i)
        if r.ci_lo <= true_mu <= r.ci_hi:
            covered += 1
    coverage = covered / trials
    assert 0.80 <= coverage <= 0.98


# =========================================================================== #
# 4. Wider held-out protocol (family x shape-regime x dtype) + leakage check
# =========================================================================== #
def test_heldout_protocol_excludes_families_and_nonempty():
    from kore.tasks import registry as reg

    all_tasks = reg.all_tasks()
    proto = kb.propose_heldout_protocol(all_tasks)

    # non-empty and dozens of tasks (a real generalization eval, not two tasks).
    assert proto.n_heldout > 0
    assert proto.n_heldout >= 12

    # the registry's authoritative reserved families are held out ...
    assert "mla" in proto.heldout_families
    assert "paged_attention" in proto.heldout_families

    # ... and NO held-out family leaks into train (family-level cleanliness).
    fam_of = reg.operator_family
    by_id = {t.task_id: t for t in all_tasks}
    train_fams = {fam_of(by_id[t]) for t in proto.train_tasks}
    for f in proto.heldout_families:
        assert f not in train_fams

    rep = kb.leakage_check(proto, all_tasks)
    assert rep["ok"] is True
    assert rep["task_overlap"] == []
    assert rep["family_overlap"] == []


def test_heldout_protocol_spans_all_three_axes():
    proto = kb.propose_heldout_protocol()
    axes = proto.axes_summary()
    # family, shape-regime and dtype axes are all populated.
    assert axes["n_families"] >= 3
    assert axes["n_shape_regimes"] >= 1
    assert axes["n_dtypes"] >= 2
    # train + held-out partition the universe with no double counting.
    assert set(proto.heldout_tasks).isdisjoint(set(proto.train_tasks))


def test_heldout_protocol_deterministic():
    a = kb.propose_heldout_protocol().as_dict()
    b = kb.propose_heldout_protocol().as_dict()
    assert a["heldout_tasks"] == b["heldout_tasks"]
    assert a["heldout_families"] == b["heldout_families"]


def test_leakage_check_detects_injected_overlap():
    # A protocol that puts the same task in both splits must be flagged.
    bad = kb.HeldoutProtocol(
        heldout_families=["gemm"], heldout_tasks=["x", "y"], train_tasks=["y", "z"],
    )
    rep = kb.leakage_check(bad)
    assert rep["ok"] is False
    assert "y" in rep["task_overlap"]


def test_shape_regime_classification():
    from kore.tasks.base import Shape

    class _T:
        def __init__(self, shapes):
            self.shapes = shapes

    small = _T([Shape("s", {"M": 64, "N": 128})])          # 8192 elements
    large = _T([Shape("s", {"M": 4096, "N": 8192})])       # 33.5M elements
    odd = _T([Shape("s", {"M": 4096, "N": 8191})])         # non-aligned
    assert kb.shape_regime(small) == "small"
    assert kb.shape_regime(large) == "large"
    assert kb.shape_regime(odd).endswith("_odd")
