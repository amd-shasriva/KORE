# What has actually been run on hardware, and what that changed

Three components were built against scripted fakes and had never touched a GPU:
the TRLOO advantage stage, the evolutionary loop, and the coverage reward. A
fourth problem, the 30B RL memory budget, was arithmetic rather than code. This
records what happened when each was put on real silicon, including the two cases
where the hardware disagreed with the design.

Host: `cv350-tnndh2-b05-1`, 8x gfx950 (`sramecc+:xnack-`), ROCm 7.2.3,
rocprofiler-sdk 1.1.0, torch 2.10.0+rocm7.0.

| Component | Status | What it cost to find out |
| --- | --- | --- |
| 30B RL memory | solved | ~61 GB/rank freed by dropping the KL reference replica |
| rocprofv3 / coverage | **validated, and a reward inversion found** | the reward would have paid for slower kernels |
| Evolutionary loop | ran on GPU; recovers a detuned kernel | **correct kernels were being recorded as incorrect** under contention |
| TRLOO | wiring verified end to end; config bug found | an unsafe `prs_min_coverage` was sitting in the production config |
| HIP task pool | 188/188 verified on gfx950 | 15 are correct but marginal on timing admission |

---

## 1. 30B RL memory

The sharded loop held a rollout replica and a frozen KL-anchor replica per rank
at ~61 GB each, ~122 GB before any training state, against 8x MI355X.

`ref_anchor_coef` is now `0.0`, which stops the reference model being loaded at
all. The KL term was contributing a fraction of a percent of the gradient, and
current GRPO practice (DAPO and successors) drops it outright. Static footprint
per rank halves. `tests/test_rl_memory_budget.py` fails if it is re-armed
without the memory being re-checked.

## 2. Coverage reward: validated, then found to be inverted

Full detail in [`coverage_denominator.md`](coverage_denominator.md). Summary:

**The profiler integration is correct.** The kernel-trace export is the
documented 22-column schema, there is no duration column, and our parser was
already computing `End_Timestamp - Start_Timestamp`. Decoy detection works
exactly as designed, twice: a synthetic kernel that is defined but never
launched, and a real `gen_add_bf16` candidate that returns `a + b` while still
declaring its `@triton.jit` kernel, both read coverage `0.000000` with
`never_ran=True` on healthy non-empty traces.

**The metric built on it was measuring the wrong denominator.** rocprofv3
profiles the whole process, and the driver re-ran full correctness verification
inside the profiled region, so random input generation, the ATen reference and
allclose reductions were in the denominator. The candidate was 5.6% of its own
trace. Because that harness cost is fixed, a *faster* kernel takes a *smaller*
share:

| variant | kernel time | coverage | Amdahl ceiling |
| --- | --- | --- | --- |
| correct seed | 471K ns | 0.057 | 1.06x |
| 11.9x slower | 5.6M ns | 0.420 | 1.72x |
| 46.5x slower | 21.9M ns | 0.739 | 3.83x |

Fed through `amdahl_end_to_end_speedup`, the 46x slower kernel earns the larger
reward. Arming `profiling_reward_weight` on the strength of "rocprofv3 works"
would have trained the policy to slow kernels down.

`KORE_TRACE_BENCH_ONLY=1` now suppresses the post-timing re-verification on the
measurement-only path (289 -> 14 dispatches, coverage 0.057 -> 0.452), but ~344K
ns of bench setup survives and the residual inversion with it. So:

* the zero-coverage decoy check stays on -- it is validated and sound;
* coverage is **not** Amdahl's `p` and must not be used as one. `p` is a
  property of the baseline workload; measuring it on the candidate is what
  inverted it;
* `profiling_reward_weight` stays `0.0` for a reason that has changed but not
  gone away.

Across ten *correct* seed kernels coverage spanned 0.036 to 0.587 purely from
harness overhead, so the old `prs_min_coverage = 0.1` rejected correct
`gen_relu_fp32` and `softmax_bf16` as "lazy optimisation". All six `hip_*` tasks
sampled produced no trace at all, so any coverage gate is Triton-only and would
silently exempt HIP.

## 3. Evolutionary loop on GPU

`kore/search/evolve_agent.py` was written so all GPU work is injected through
`env`, which made it fully testable against fakes and meant a real `KoreEnv` had
never driven it. Driven now by a deterministic, model-free launch-parameter
proposer, so a failure could not be blamed on a model writing bad kernels.

It runs. Real correctness verification each step (5 shapes, 616M elements,
random + determinism prongs), timing stable at `cv_pct` under 1.2%.

Two things only a real budget exposes:

* **On the shipped seed, nothing beat it and the loop said so.** Every
  launch-parameter variant measured ~0.92-0.94x. No champion was crowned, which
  is the correct answer, not a failure.
* **Archive membership and eliteship are different states.** A candidate enters
  the archive on one *screening* measurement but only becomes admissible -- and
  therefore a champion, parent or exemplar -- once *stabilised* (`measured and
  correct and stats.n > 0`). Under a 70-call budget the archive filled with four
  members across four distinct niches and `champion()` returned `None`. That
  reads as a broken search from the totals alone; it is a budget that ran out
  before stabilisation. `scripts/validate_evolve_gpu.py` now prints per-member
  admissibility so the two cannot be confused again.

Detuning the seed to `BLOCK_N=64, num_warps=1` gives the search a known gap:
the detuned control measures **0.44-0.55x** against the reference, and the
search reaches **~0.947x** -- a ~1.8x recovery. Speedup is measured against the
task's reference, not against the seed, so "did it work" is judged against that
measured control rather than against 1.0.

### A correct kernel with no timing was being recorded as incorrect

The run that exposed this landed on a contended GPU, where `cv_pct` rose from
the usual 0.5-1.2% to 9-42%. Every generation then reported `'incorrect': 8`
while the environment had logged `compiled=True correct=True` for all eight
kernels.

`StableEvaluator`'s sampler condemned a trial on
`not result.correct or result.speedup is None`, which folds together a real
finding and a non-finding:

* `not result.correct` -- a kernel that verified once and not again is genuinely
  unstable, and refusing to average it with the run that passed is right;
* `result.speedup is None` -- the timing harness declined to produce a number.
  That is the node being busy, not a wrong answer.

The second case set `trial.correct = False`, and since `Candidate.admissible`
requires `correct`, the design was excluded from the elite pool permanently. So
transient measurement noise silently deleted good kernels, and the run log
attributed it to the model. The giveaway was the verdict histogram: the archive
has an `unmeasured` bucket that read 0 while `incorrect` read 8.

This matters more in production than in this probe, because contention is the
normal case there -- every rank benches at once during RL.

Now separated: a missing timing skips the sample and leaves the trial correct,
so it stays in the archive but is not admissible (`stats.n == 0`) and cannot
become a champion on a measurement that never happened. Pinned by
`tests/test_evolve_agent.py::test_a_correct_kernel_with_no_timing_is_unmeasured_not_incorrect`.

This is the same defect class as the coverage finding above: treating "not
measured" as "measured and bad".

## 4. TRLOO

The estimator itself is a pure function with its own suite. What had never been
checked is the thing that decides whether an RL launch survives step one: TRLOO
needs each sample's `(trajectory_id, turn_id)` at tuple field 6, and
`sample_turn_keys` returns `None` -- deliberately refusing to fall back to the
biased pooled estimator -- if any sample lacks it, which makes
`_group_advantages_or_raise` raise.

Both rollout paths emit it correctly. The sharded path additionally maps its
local trajectory index back through `_rank_slice` (`my[ti]`) before using it as a
key, without which trajectory 0 of every rank would collide and contaminate the
others' baselines. This is correct as written.

The real defect was in the config, not the code:
`configs/grpo_coder30b_a3b_trloo.json` shipped `rejection_sampling: true` with
`prs_min_coverage: 0.1`. Only MRS is actually invoked -- `profiling_rejection_sample`
has no call site in `kore/` -- so PRS is inert today and nothing was trained on
it. But the value was a loaded gun for whoever wires PRS up next, since the
hardware numbers above show 0.1 rejects correct kernels. It is now `0.0`, which
leaves the sound decoy rejection on and the unsound lazy-optimisation rejection
off.

## 5. HIP task pool

`tests/test_hip_backend.py` ties the registry to
`data/hip_task_verification.json` so that adding a HIP task without measuring it
fails in CI rather than at datagen. The artifact was stale against a newer test
schema, so all 188 registry HIP tasks were re-run on gfx950, sharded across six
GPUs:

* **188/188 measured**, 282 rows across the shards;
* **188/188 compiled and verified correct**, zero infra errors, zero flagged
  hacks;
* **173/188 also cleared timing admission.** The remaining 15 are correct but
  marginal against the noise gate, which is why the artifact records correctness
  and timing admission separately per run instead of collapsing them into one
  boolean -- correctness is deterministic, admission is a noise gate and a
  marginal task can pass one sweep and fail the next.

---

## Reproducing

```bash
python scripts/validate_rocprofv3_coverage.py            # parser, synthetic workloads
python scripts/make_ktrace_receipt.py --tasks 16         # KoreEnv on registry tasks
python scripts/validate_evolve_gpu.py --detune --gpu 1 \
    --generations 4 --max-env-calls 240                  # evolutionary loop
python scripts/verify_hip_tasks_e2e.py --gpu 0 --json /tmp/h.json  # HIP pool
python -m pytest tests/test_coverage_denominator.py tests/test_evolve_agent.py
```

## What is still not proven

* Coverage as a *magnitude*. Needs the timed region delimited (roctx ranges
  around the bench loop, or an `--iters 0` setup trace to subtract) and a HIP
  kernel-name source. Until then it is a decoy detector, which is worth having
  on its own.
* TRLOO under a live 30B policy. The wiring is verified and the estimator is
  tested, but no RL step has run at that scale.
* The evolutionary loop with `HarnessProposer` -- a real model as the mutation
  operator -- rather than the deterministic proposer used here.
