# `tests/` — the test suite

The default `pytest` command discovers both this top-level suite and every
package-local `kore/**/tests/test_*.py` module. It exercises science, reward,
data, RL math, breadth generators, and campaign wiring on CPU; use collection
itself as the live inventory rather than maintaining a count in prose.

```bash
pip install -e ".[test]"                              # CPU test environment
python -m pytest                                      # default CPU suite
python -m pytest --collect-only -q                    # live module/item inventory
python -m pytest tests                                # top-level CI split
python -m pytest kore                                 # package-local CI split
python -m pytest tests/test_campaign_wiring.py        # one file
python -m pytest -m gpu                               # opt-in: real GPU + ROCm
```

## Marker groups

Two markers gate tests out of the default run, and `pyproject.toml` deselects
exactly those two:

| Marker | Meaning |
| --- | --- |
| `gpu` | needs a supported AMD GPU (gfx950) and ROCm. Run with `python -m pytest -m gpu`. |
| `release` | release-blocking licensing / regenerated-artifact checks, excluded from the default suite. |

`cpu` is applied automatically by the root `conftest.py` to every test that
carries no resource marker, so `-m cpu` is the complement of the opt-in groups.
`packaging` and `shell` label subsets that are cheap enough to stay in the
default run.

**A declared marker must select at least one test.**
`test_marker_contract.py` enforces that, because an audit found `gpu`, `model`,
`network` and `dependency` all registered, all documented as opt-in groups, and
all selecting *zero* tests — so the deselection in `addopts` was a no-op and the
GPU path was validated only by a manual sbatch. `model`, `network` and
`dependency` have been removed from the declaration for exactly that reason.
Adding a new opt-in group means adding its tests in the same change; the
contract test also checks that every group named in the default `-m` expression
is declared, that `gpu`/`release` stay excluded by default, and that no test ever
carries both `cpu` and a resource marker.

The contract is decidable whenever the whole `tests/` tree is collected (a bare
`python -m pytest` or the `python -m pytest tests` CI split). A narrower run
skips it with that reason rather than failing on tests it never looked at.

---

## The GPU suite

`test_gpu_verifier.py` and `test_gpu_timing_protocol.py` are the only tests that
touch hardware, and they use no stubs: the driver really runs in a subprocess
through `KoreEnv._env` / `KoreEnv._exec`, the candidate really compiles through
Triton, and every verdict comes from measured output. This is the half of
`kore/env/kore_env.py` the CPU suite cannot reach — `tests/test_phase0_verifier_fixes.py`
monkeypatches `_exec`, `_env`, `_bench_multi` and `_bench_all` away, so it tests
the verifier's decision logic and never its execution.

```bash
python -m pytest -m gpu -q                       # ~70 s on an idle MI350X
KORE_TEST_GPU=7 python -m pytest -m gpu          # pin to another card
python -m pytest -m gpu -k timing_protocol       # one area
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `KORE_TEST_GPU` | `6` | physical GPU the suite runs on |
| `KORE_TEST_GPU_ALT` | `7` | second card, used only to prove the visible-device mapping selects the card it was asked for |

Pinning is deliberate rather than "whatever torch enumerates first": a shared
node hands the other cards to other jobs, and the timing tests need their card
to themselves (`KoreEnv._timing_lock` serializes the timed window per physical
GPU). Nothing runs on a card the suite was not given, and the parent pytest
process never initializes HIP — every probe runs in a child, so a CPU-only
collection stays clean.

**Skipping is honest, not vacuous.** When no card answers a child pinned to
`KORE_TEST_GPU`, the `gpu_harness` fixture skips the whole suite with that
reason. When AITER is not importable, the vendor-baseline tests skip and say so,
because a pass there would hide every `aiter_*` task silently measuring the torch
framework path instead of the production kernel.

### What the GPU suite covers

`test_gpu_verifier.py`

- the full `evaluate` path on a real card: compile, correctness on every
  requested shape, finite positive candidate and baseline timings, and the reward
  tier the observation lands in. A busy node may fail the CV/CI admission gates,
  which demotes the timing to `timing_grade="screening"`; the test asserts both
  outcomes precisely (publication ⇒ `correct_timed` and a clean independent
  recompute; screening ⇒ `correct_screening` with a stated measurement-noise
  reason) instead of accepting either loosely.
- a deliberately wrong kernel rejected by measured SNR, not by the static hack
  scanner and not by a mock — and never timed.
- the post-timing re-verification catching a **stateful** kernel that is correct
  when checked and wrong while timed (the invocation-count timing hack). A second
  test runs the same kernel with `do_bench=False` to show it *does* pass the
  correctness gate alone, so the anti-hack re-check is provably the only thing
  stopping it.
- the AITER vendor baseline: whether the runtime loads at all, and whether the
  `KORE_BASELINE_IMPL:` sentinel honestly reports `aiter_vendor` /
  `hipblaslt_vendor` rather than a silent `framework` fallback. Tasks declaring a
  vendor bar are benched with `--impl reference` and must emit a vendor sentinel;
  a task declaring the framework bar must emit `framework`. This is the runtime
  confirmation `kore.data.schemas.resolve_baseline_identity` explicitly does not
  provide.
- `HIP_VISIBLE_DEVICES` mapping and partial-allocation detection, checked against
  device UUIDs so the ordinal itself is not taken on trust, plus agreement
  between `_gpu_selection`'s recorded provenance and the child's real
  environment.

`test_gpu_timing_protocol.py`

- the versioned driver handshake (`kore-paired-v2`, protocol 2, every
  publication guarantee).
- the paired protocol on live output: exactly `repeat` `KORE_TIMING_PAIR` lines,
  ascending pair indices, alternating and balanced AB/BA ordering, `ratio` and
  `log_speedup` derivable from the raw times, and acceptance by the verifier's
  own `_parse_timing_pairs`.
- one verified block per shape for a multi-shape bench, each with its own
  post-timing verdict.
- `kore.reward.reward._publication_timing_error` agreeing with the driver:
  every ratio, log-speedup, CI and classification recomputed from the raw
  samples. If the node is too noisy for vendor-grade admission, the test proves
  the demotion is *justified* by the recomputed statistics before skipping.
- `KORE_BENCH_COLD` being observable in the measurement — cold-cache timing must
  be measurably slower than warm on a memory-bound shape, otherwise the
  documented L2 flush is not doing anything and both cold-cache provenance and
  the roofline speed-of-light floor are unfounded. Assertions are on
  relationships (best-of-N ratios), never on absolute latencies.

### CI

`.github/workflows/gpu.yml` runs this suite, but on `workflow_dispatch` only: no
GitHub-hosted runner has an AMD GPU, and a job pinned to a self-hosted label with
no matching runner would queue until it timed out — a red PR rather than a clean
skip. Dispatched against a GPU-less runner it still stays green: a preflight step
detects the missing `/dev/kfd`/`rocm-smi` and turns the remaining steps off, and
the suite itself would skip every test anyway. On a Slurm cluster,
`scripts/spur_gpu_smoke.sbatch` is the batch entry point.

---

## Coverage map

| Subsystem | Test files |
| --- | --- |
| Roofline / physics / P0 | `test_rooflines.py`, `test_roofline.py`, `test_p0_sol.py`, `test_reward_physics.py`, `test_whitebox_reward.py`, `test_profile_reward.py`, `test_dense_reward.py`, `test_pmc_counters.py` |
| Reward ladder / integrity | `test_reward_stats.py`, `test_timing_integrity.py`, `test_paradigm_credit.py` |
| Correctness oracle | `test_verify_equivalence.py`, `test_verifier_determinism.py`, `test_verify_rigor.py` |
| Verifier on hardware | `test_gpu_verifier.py`, `test_gpu_timing_protocol.py` (both `-m gpu`) |
| Tasks / ops | `test_genops.py`, `test_vendor_ops.py`, `test_augment.py`, `test_data_scale_audit.py`, `test_arch_normalize.py`, `test_coverage.py` |
| Data factory | `test_data.py`, `test_parallel_datagen.py`, `test_evolve.py`, `test_assemble.py`, `test_mixing.py`, `test_rejection.py`, `test_hard_negatives.py`, `test_onpolicy.py`, `test_gold_wins.py`, `test_gen_repair_quality.py`, `test_gen_wins_convergent.py`, `test_repair_dpo.py`, `test_reverify.py`, `test_corpus_quality.py`, `test_curate.py`, `test_hygiene.py`, `test_grounded_reasoning.py`, `test_synth_agentic.py`, `test_preference_quality.py` |
| Open-ended curriculum | `test_openended_proposer.py`, `test_openended_task_space.py`, `test_openended_archive.py`, `test_openended_coevolve.py`, `test_openended_controller.py`, `test_coevolve_distill.py` |
| Policy / RL | `test_rl_core.py`, `test_policy.py`, `test_grpo_fsdp.py`, `test_grpo_distill_hook.py`, `test_dynamic_steps.py`, `test_midtrain.py`, `test_distributed.py`, `test_frontier_ops_wiring.py`, `test_deep_cot_contract.py` |
| Value model | `test_value.py`, `test_value_replay_train.py` |
| Agent / transforms | `test_agent.py`, `test_agent_transform_discover.py` |
| Eval / gates | `test_eval.py`, `test_generalization.py`, `test_retention.py`, `test_champion.py`, `test_korebench.py`, `test_vs_opus.py` |
| Infra | `test_campaign_wiring.py`, `test_obs.py`, `test_contract.py`, `test_marker_contract.py`, `test_env_gpu_visibility.py`, `test_phase0_verifier_fixes.py` |

`test_campaign_wiring.py` and `test_distributed.py` are the fastest confidence
check that the orchestration and FSDP configuration are coherent.
`python -m pytest -m gpu` is the fastest confidence check that the verifier
actually works on hardware. Resource-specific tests belong to an explicit marker
group; the default remains CPU-only.
