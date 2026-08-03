# Coverage on real gfx950: what rocprofv3 actually measures

Hardware validation of the coverage reward path, run on gfx950
(`sramecc+:xnack-`), ROCm 7.2.3, rocprofiler-sdk 1.1.0, torch 2.10.0+rocm7.0.

Reproduce with:

```bash
python scripts/validate_rocprofv3_coverage.py   # parser, synthetic workloads
python scripts/make_ktrace_receipt.py --tasks 16  # KoreEnv, registry tasks
```

## Summary

The rocprofv3 integration works. The coverage *metric built on top of it* was
measuring the wrong thing, and arming `profiling_reward_weight` without this
check would have paid the policy to write slower kernels.

Three separable results:

1. The parser and the trace collection are correct on this hardware.
2. Decoy detection (`coverage == 0.0`, `never_ran`) is correct and is the one
   part of coverage that can be trusted today.
3. Coverage as a *magnitude* was measuring share of the test harness, not share
   of the workload, and therefore rose as the kernel got slower. Partially
   fixed; still not fit to be Amdahl's `p`.

## 1. The export schema matches the documentation

`rocprofv3 --kernel-trace --output-format csv` writes `*_kernel_trace.csv` with
22 columns:

```
Kind, Agent_Id, Queue_Id, Stream_Id, Thread_Id, Dispatch_Id, Kernel_Id,
Kernel_Name, Correlation_Id, Start_Timestamp, End_Timestamp, LDS_Block_Size,
Scratch_Size, VGPR_Count, Accum_VGPR_Count, SGPR_Count, Workgroup_Size_{X,Y,Z},
Grid_Size_{X,Y,Z}
```

There is **no duration column**; duration is `End_Timestamp - Start_Timestamp`.
`kore/verifier/parsers/rocprofv3.py` already computed it that way, so the
uncertainty recorded in `collect_kernel_trace`'s docstring is resolved: the
layout is the documented one.

## 2. Decoy detection works, standalone and through KoreEnv

The case coverage exists for is a candidate that reports a large speedup because
its kernel never ran. Two independent checks:

| probe | dispatches | coverage | `never_ran` |
| --- | --- | --- | --- |
| synthetic: kernel defined, never launched, matmuls running | 13 | `0.000000` | `True` |
| `gen_add_bf16`: `@triton.jit` kernel defined, `return a + b` | 29 | `0.000000` | `True` |

Both traces are healthy and non-empty, so zero coverage is a finding rather than
a missing measurement. This is the property worth keeping.

## 3. Coverage magnitude was measuring the test harness

`collect_kernel_trace` runs the driver with `--bench-mode --impl candidate`, but
`kore/tasks/_genops.py:driver_main` also ran `_run_correctness()` afterwards --
the anti-stateful-timing-hack re-verification. rocprofv3 profiles the whole
process, so random input generation, the ATen reference and the allclose
reductions landed in the same CSV as the benchmark.

On `gen_add_bf16`, 289 dispatches, of which the candidate was 10:

| share | kernel |
| --- | --- |
| 14.0% | `at::native::vectorized_elementwise_kernel` (reference) |
| 9.8% | `__amd_rocclr_copyBuffer` |
| 7.5% | `at::native::reduce_kernel<ReduceOp<bool>>` (allclose) |
| 7.5% | `distribution_elementwise` (random inputs) |
| **5.6%** | **`_add_kernel` (the candidate)** |

The denominator was a fixed ~8.2M ns of harness. Since it does not depend on the
candidate, a *faster* kernel takes a *smaller* share. Measured directly, holding
everything else constant and padding the kernel with unfoldable transcendentals:

| variant | kernel time | coverage | Amdahl ceiling |
| --- | --- | --- | --- |
| seed (correct, fast) | 471K ns | **0.0567** | 1.06x |
| 11.9x slower | 5.6M ns | 0.4195 | 1.72x |
| 46.5x slower | 21.9M ns | **0.7388** | 3.83x |

Coverage rose monotonically with slowness. Fed through
`amdahl_end_to_end_speedup` and `profiling_reward`, this is a reward inversion:
the 46x slower kernel earns the larger shaping term, and under
`prs_min_coverage = 0.1` the *correct fast* kernel is rejected as
`lazy_optimisation` while the slow one passes.

`profiling_reward_weight` was `0.0` and `rejection_sampling` was `False`, so
nothing was trained on this. The receipt gate is what kept it that way.

## 4. Partial fix: confine the trace to the benched region

`KORE_TRACE_BENCH_ONLY=1` (set by `collect_kernel_trace`, honoured by
`driver_main`) suppresses the post-timing correctness re-verification on the
measurement-only path. Verdicts are unaffected -- `collect_kernel_trace` issues
none, and the anti-hack check still runs on every path that does. It is an env
var rather than a CLI flag so that drivers predating it ignore it instead of
failing argparse.

Effect on `gen_add_bf16`: 289 -> 14 dispatches, coverage 0.057 -> 0.452.

The inversion is reduced but not removed. A constant ~344K ns of bench-input
setup survives in the denominator:

| variant | kernel ns | region ns | setup remainder | coverage |
| --- | --- | --- | --- | --- |
| seed | 283,519 | 629,953 | 346,434 | 0.4501 |
| 11.9x slower | 2,821,008 | 3,161,963 | 340,955 | 0.8922 |
| 46.5x slower | 10,997,126 | 11,341,202 | 344,076 | 0.9697 |

## 5. Coverage across correct seed kernels varies 16x

Every row below is a **correct** reference kernel, so any spread is harness
overhead rather than kernel quality:

| task | dispatches | coverage |
| --- | --- | --- |
| `gen_mul_bf16` | 14 | 0.5866 |
| `gen_relu_fp16` | 11 | 0.5494 |
| `gen_add_fp32` | 12 | 0.4618 |
| `gen_add_bf16` | 14 | 0.4520 |
| `gen_add_fp16` | 14 | 0.4487 |
| `gen_mul_fp16` | 14 | 0.2995 |
| `gen_relu_bf16` | 11 | 0.2908 |
| `gen_mul_fp32` | 12 | 0.1614 |
| `gen_relu_fp32` | 10 | **0.0952** |
| `softmax_bf16` | 230 | **0.0363** |

`gen_relu_fp32` and `softmax_bf16` fall below `prs_min_coverage = 0.1` while
being perfectly correct. `softmax_bf16` has its own hand-written driver that
does not honour `KORE_TRACE_BENCH_ONLY`, which is why it still shows 230
dispatches -- the fix only reaches `_genops`-generated drivers.

## 6. Coverage is Triton-only in practice

All six `hip_*` tasks sampled returned no trace at all (6/16 of the sample):

```
hip_add_bf16  hip_add_fp16  hip_add_fp32
hip_mul_bf16  hip_mul_fp16  hip_mul_fp32
```

`candidate_kernel_names` matches on `@triton.jit` definitions, so even a HIP
trace would yield no candidate names and unknowable coverage. Any coverage-based
gate would therefore apply to Triton tasks and silently exempt HIP ones, which
narrows training to a subset while looking like a quality filter.

## Conclusions

- **Keep** `never_ran` / `coverage == 0.0` decoy rejection. Validated twice.
- **Do not** use coverage as Amdahl's `p`. Amdahl's `p` is a property of the
  baseline workload; measuring it on the candidate makes the reward a function
  of the candidate's own speed, which is what inverted it. For single-op KORE
  tasks there is no surrounding application, so the honest `p` is 1.0 and the
  Amdahl term is inert by construction.
- **Do not** enable `lazy_optimisation` rejection against these numbers. At
  `prs_min_coverage = 0.1` it rejects correct kernels. The operational default is
  now `0.0`; `LAZY_COVERAGE_THRESHOLD` stays at the paper's `0.10` because that
  is the right threshold for a trustworthy coverage and it still drives advisory
  turn feedback. Nothing in `kore/` calls `profiling_rejection_sample` yet, so
  whatever wires PRS up must pass `GRPOConfig.prs_min_coverage` explicitly rather
  than inherit the module default.
- `profiling_reward_weight` stays `0.0`. The receipt at
  `data/ktrace_receipt.json` attests that the *trace path* works; it is not
  evidence that the *magnitude* means what the reward assumes.

Making coverage usable as `p` needs the timed region delimited explicitly
(roctx ranges around the bench loop, or a setup-only `--iters 0` trace to
subtract), plus a HIP kernel-name source. Until then coverage is a decoy
detector, and that is worth having on its own.
