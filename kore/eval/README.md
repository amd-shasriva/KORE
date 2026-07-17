# `kore/eval` - evaluation, gates & generalization

KORE's claim is conjunctive: **best kernel numbers _while_ matching-or-beating the base model on every general benchmark, _and_ generalizing to held-out operator families.** This package measures all three, plus a maximum-scrutiny anti-hack re-eval of champion kernels.

A **publishability layer** (unit-tested in `kore/eval/tests/test_eval_frontier.py`, now **wired into the campaign eval stage**) makes those numbers comparable to the wider literature and defensible under review: a recognized-benchmark adapter (`kernelbench_amd.py`), a robust-KernelBench-style anti-hack correctness battery (`robust_eval.py`), and paired significance statistics for KORE-vs-baseline/Opus (`paired_stats.py`).

---

## Files

| File | Purpose |
| --- | --- |
| `fastp.py` | The KernelBench `fast_p` metric (+ pass@k, bootstrap CIs) |
| `bakeoff.py` | Matched-measurement-budget bake-off of policies (seed vs. trained) |
| `retention.py` | Six-benchmark general-capability suite |
| `gates.py` | PASS/FAIL stage gates (`retention_gate`, `StageGate`) |
| `generalization.py` | Zero-shot cross-family transfer harness |
| `champion.py` | Champion kernel re-eval under harder-than-training scrutiny |
| `kernelbench_amd.py` | **(new)** Recognized-benchmark adapter: KernelBench ⇄ KORE, `fast_p` at `p ∈ {1, 1.5, 2}`, wider held-out protocol |
| `robust_eval.py` | **(new)** Robust-KernelBench anti-hack battery: adversarial regimes, fp64 differential oracle, metamorphic relations, halt-on-first |
| `paired_stats.py` | **(new)** Paired bootstrap CI + sign/Wilcoxon tests for KORE-vs-baseline/Opus |
| `korebench.py`, `policies.py`, `report.py`, `vs_opus.py`, `e2e_sglang_vllm.py` | standardized report, policy wrappers, markdown/JSON output, KORE-vs-Opus harness, E2E serving gate |

---

## fast_p and the bake-off

```
fast_p = (1/n) · #{ tasks that are correct AND baseline/actual > p }
```

`n` is the **full** split size (uncorrected denominator) - failed or unattempted tasks contribute 0, penalizing wasted budget. `bakeoff.matched_budget_bakeoff` compares policies at an **equal bench budget** per task and ranks by `fast_p[1.0]`. Two policies matter: `seed_policy` (the frozen starter) and `model_policy(checkpoint)` (the trained model). Reporting only the seed, or comparing at unequal budgets, is a documented audit trap.

```mermaid
flowchart LR
  SEED[seed_policy] --> EV[evaluate_policy budget=N]
  MODEL[model_policy checkpoint] --> EV
  TASKS[task split] --> EV
  EV --> FP["fast_p curve (p=0,0.5,1,1.5,2)"]
  FP --> CMP[matched_budget_bakeoff · rank by fast_1]
  CMP --> RPT[report → JSON + markdown]
```

---

## Retention gate

```mermaid
flowchart TD
  BASE[base model] --> BS[run_retention_suite → base scores]
  CAND[candidate ckpt] --> CS[run_retention_suite → candidate scores]
  BS --> RG{retention_gate ε}
  CS --> RG
  RG -->|no bench drops > ε| PASS[PASS → promote]
  RG -->|any bench drops > ε| FAIL[FAIL → hard-stop campaign]
```

`run_retention_suite` scores six benches - **MMLU, HumanEval, LiveCodeBench, IFEval, BFCL, MT-Bench** - each normalized to `[0,1]`. `KORE_EVAL_FULL=1` pulls the real HuggingFace splits (capped by `KORE_EVAL_N`, default 300/bench); otherwise bundled smoke subsets are used, and the `sources` field records which (a PASS on smoke is *not* comparable to a PASS on full HF). The campaign calls `retention_gate` after midtrain/sft/dpo/grpo; a FAIL raises and stops the run. `StageGate` additionally requires kernel metrics to strictly improve for promotion.

> MT-Bench is the highest-variance gate key (LLM-as-judge noise); it needs a strong injected judge and enough items, or `ε` will trip on judge noise. The campaign default `ε=0.02` is chosen with this in mind.

---

## Generalization (held-out families)

`generalization.py` classifies tasks into 8 families (`attention, moe, gemm, norm, positional, quant, reduction, activation`, first-match-wins rules), builds a leakage-checked split by **entire families**, and evaluates the physics residual reward on held-out families from a P0 measures JSON - **offline, no training**. It gates aggregation on the reward's own correctness verdict (`rr.correct`), not just the raw measure flag.

> **Two family taxonomies, by design.** The 8-family `classify` here is the richer analysis / leave-one-family-out grouping. The AUTHORITATIVE product split is `kore.tasks.registry` (`operator_family` + `HELDOUT_FAMILIES`): the model **trains** core attention (flash prefill/decode/varlen/fp8) and reserves the structurally-distinct **MLA** (latent attention) and **paged-KV decode** families (plus any foreign-arch task) as the never-trained set. `korebench.py`'s per-family view uses the registry taxonomy; don't conflate the two.

---

## Champion re-eval (anti-hack)

`champion.py` re-benchmarks the best-per-task kernels under **harder** conditions than training - `KORE_VERIFIED_CORRECTNESS=1`, `KORE_COMPILE_BASELINE=1`, `KORE_BENCH_COLD=1`, `KORE_CORRECTNESS_TRIALS=10`, and augmented held-out shapes, with the replay cache disabled. A kernel is certified only if it is correct on unseen shapes, hack-free, low-variance, and its measured speedup hasn't *collapsed* below `0.7×` of the claimed value - catching kernels that overfit training shapes or the timing setup.

---

## Publishable-eval frontier (wired into `_stage_eval`)

Three modules that make a KORE result *publishable* - comparable to the field, hardened against correctness-hacks, and statistically defensible. All are import-safe (torch/numpy imported lazily where needed) and unit-tested in `kore/eval/tests/test_eval_frontier.py`. Two of them are now **wired into the campaign eval stage** (`scripts/run_campaign.py._stage_eval`), each fail-safe (wrapped so a failure logs and skips, never breaking eval): the **paired-significance** track (`_eval_paired_significance` → `paired_speedup_comparison`: bootstrap CI + Wilcoxon + sign test on KORE-vs-seed per-task speedups → `eval/paired_seed_vs_kore.json`) and the **KernelBench-AMD `fast_p`** track (`_eval_kernelbench_amd` → `run_kernelbench_amd`, bundled offline specs by default or a real checkout via `--kernelbench-root` → `eval/kernelbench_amd.json`). `robust_eval.py` stays exposed for maximum-scrutiny anti-hack audits (import-checked in the campaign preflight).

### `kernelbench_amd.py` - recognized-benchmark adapter

Bridges KORE and the field-standard **KernelBench** in both directions so a KORE number drops straight into a KernelBench-style leaderboard, measured on the KORE target AMD arch (**gfx950**/CDNA4 by default, gfx942/CDNA3 accepted; every task is backend-tagged):

- **Forward** (`spec_to_task`): a KernelBench-style problem (a PyTorch `Model.forward` + input generator + named shapes, Level 1 single-ops / Level 2 fusions) becomes a genuine KORE `Task`, graded through KORE's own verified, timing-integrity-gated matched-budget bake-off. The PyTorch reference becomes the correctness oracle; the baseline is **torch-eager** (KernelBench's baseline), labeled as such.
- **Reverse** (`to_kernelbench_report`): renders a KORE `evaluate_policy` result as the field-standard **`fast_p` at `p ∈ {1.0, 1.5, 2.0}`** (fraction of the *whole* split that is correct AND >p× faster than baseline), with a per-**level** (1 vs 2) breakdown, correct rate, and geomean speedup. `fast_1` is the headline.
- **Bundled fixtures** (`bundled_specs`) span the three canonical classes (elementwise L1, GEMM L1, pointwise + GEMM-epilogue L2) so the whole path is CPU-testable offline; `load_real_kernelbench` documents/implements loading the real Level 1/2 problem files from a checkout.
- **Wider held-out protocol** (`propose_heldout_protocol`): the registry reserves only a couple of families (MLA, paged-KV) - too thin for a publishable generalization claim. This proposes a **dozens-of-tasks** split stratified over three axes - operator **family × shape-regime × dtype** - reserving *whole* families with a strict `leakage_check` / `assert_no_leakage` (no task in both splits, no family straddling the boundary). It only computes a proposal; it never mutates the registry.

### `robust_eval.py` - robust-KernelBench anti-hack battery

Hardens the EVAL-time correctness verdict against kernels that pass a naive `allclose` on a few random inputs. Each check is a pure function of a candidate callable + a torch reference + a deterministic input factory (CPU/torch-testable with fake kernels):

- **Adversarial regimes** (`check_adversarial_regimes`): enumerated hard fills - zeros / ones / neg-ones / large / neg-large / small / sign-alternating / NaN-Inf - with **non-finite-structure-aware** comparison (a correct kernel must reproduce the reference's NaN/Inf positions and inf signs exactly).
- **fp64 differential oracle** (`check_differential_oracle`): recompute the reference in fp64 (the high-precision truth) and reject a candidate that is materially **less accurate than its own dtype warrants** - catching a precision downgrade that `allclose` waves through.
- **Metamorphic relations**: `permutation_invariance` (reductions/softmax), `homogeneity` `f(ax)=a·f(x)` (linear/GEMM), and `additive_response` (fusions) - each applied only when the *reference* itself satisfies it, then required of the candidate (flags constant/memset and mis-fused kernels). Plus reseeded random inits and non-contiguous/strided inputs.
- **Halt-on-first**: `robust_correctness(...)` runs the applicable battery in order and **halts on the first mismatch**, returning a `RobustReport` naming the failing check - the maximum-scrutiny verdict a published claim needs. `inputs_factory_from_spec` bridges it directly onto a `kernelbench_amd` spec.

### `paired_stats.py` - paired significance for KORE-vs-baseline/Opus

Because KORE and the other side (seed baseline or the Opus teacher) are scored on the **same** held-out tasks under a matched budget, the comparison is **paired** - far more powerful than unpaired, since it cancels task-to-task difficulty variance. Pure numpy/python (no scipy, no torch):

- **Effect size + 95% CI**: `paired_bootstrap` gives a percentile bootstrap CI (and a recentred-bootstrap two-sided p-value) on the mean per-task delta; `paired_speedup_comparison` works in the log domain so the effect is a **geometric-mean speedup ratio** with an exponentiated CI ("KORE is X× faster, 95% CI …").
- **Non-parametric p-values**: an exact two-sided **sign test** (binomial, robust to outliers) and the **Wilcoxon signed-rank** test (normal approx with tie + continuity correction). `paired_comparison` bundles all three, with the headline test (Wilcoxon by default) deciding significance at `alpha`.

See also: [`kore/policy`](../policy/README.md), [`kore/reward`](../reward/README.md), [`kore/analysis`](../analysis/README.md), [`docs/KORE_BENCH_BLUEPRINT.md`](../../docs/KORE_BENCH_BLUEPRINT.md).
