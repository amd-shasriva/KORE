# `kore/eval` — kernel, retention, and external evaluation

KORE reports a kernel result only after the candidate passes the same
correctness gate used in training. `fast_p` uses the complete task denominator,
so failed and unattempted tasks cannot disappear from a headline number.

## Production evaluation sequence

1. Evaluate the SFT checkpoint against the exact instruct checkpoint it started
   from. This isolates SFT instead of crediting the vendor model.
2. Run AgentKernelArena on gfx950 with
   `scripts/run_agent_kernel_arena.py` and
   `scripts/spur_aka_1node.sbatch`. The runner uses copied workspaces because
   benchmark harnesses write artifacts; editing the checkout would change later
   tasks.
3. Run `scripts/run_kernelbench_amd.py` for KernelBench-AMD and report the
   torch-eager baseline separately from KORE's vendor lane.
4. Use `vs_opus.py` / `head_to_head.py` only for matched prompts, task sets,
   decode budgets, and execution scoring. A missing teacher credential produces
   a skipped frontier arm, not a fabricated comparison.
5. Run multi-turn RL evaluation and test-time scaling over the same verified
   task contract.

## AgentKernelArena

`agent_kernel_arena.py` discovers tasks whose `config.yaml` is active and
requires `gfx950`, then follows its declared compile, correctness, and
performance commands in that order. Its score is unchanged: 0 for compile
failure, 20 for compile-only, and `120 + 100 * speedup` for a correct result.

**416 tasks** are runnable on gfx950 under the discovery filter, measured by
`run_agent_kernel_arena.py discover` against AKA `b09f5eb`. Earlier figures of
402 and 413 appear in several places in this repo and are both stale; the count
moves when AKA adds tasks (`b09f5eb` added three `image_kernel` ones).

Published Opus speed bars are 6.89x (`torch2hip`), 6.69x (`hip2hip`), and 2.13x
(`triton2triton`). The runner places those values next to measured means; it
never treats them as KORE measurements.

### Measuring speedup requires two passes

Most of the arena reports an **absolute latency and no ratio**, because a single
run has nothing to compare against. Those tasks only yield a speedup when a
`baseline` pass has timed the same task first, into the **same `--out`**:

```
python scripts/run_agent_kernel_arena.py baseline --out runs/aka_v5 ...
python scripts/run_agent_kernel_arena.py run      --out runs/aka_v5 ...
```

This is not optional bookkeeping. On 2026-08-10 the `run` pass went first and
alone: 302 kernels were correct, 49 produced a speedup, and 246 of the remaining
253 had a valid optimized time next to a null denominator. The loss is
unrecoverable because workspaces are deleted after scoring, so the only way to
get those numbers is to run the whole sweep again. `run` now refuses to start
below `--min-baseline-coverage` (default 0.5) rather than warning about it.

Read `speedup_coverage` before quoting a `mean_speedup`. The mean is computed
only over tasks that produced a speedup, so sparse coverage flatters us —
correct-but-unmeasured tasks are dropped rather than counted as 1.0x.
`beats_opus` is therefore `None`, not `False`, below 80% coverage: the published
bars are full-category means and a mean over a handful of tasks is a different
quantity.

`scripts/verify_arena_speedup_e2e.py` proves the path on real GPUs by timing a
task, then submitting byte-identical source as the "optimization" and checking
the ratio lands near 1.0x. Verified on gfx950 across all three harness dialects:
`triton2triton/vllm` 0.993x, `instruction2triton/rocmbench` 1.051x, and
`hip2hip/gpumode` 1.000x.

### Provenance, and the contamination index after an AKA update

Every results file records `arena_provenance` — the AKA commit, its `git
describe`, and a count of locally modified files. AKA is a sibling clone rather
than a submodule, so nothing else captures which task set produced a number, and
that matters: the count moved 413 -> 416 on the `ea4c0ee..b09f5eb` pull, which
also rewrote `image_kernel` timing code. A dirty AKA checkout is reported
loudly, because an edited harness can change a timing method or a tolerance and
the result would still look ordinary.

`data/arena_contamination.json` is deliberately **frozen at the state the v5
training data was gated against** (built 2026-08-11, 355 arena tasks scored, 68
unscreened, 24 pool tasks blocked, 16 arena tasks implicated). It is not
regenerated when AKA moves, because the v5 mixture's provenance is a claim about
*that* table; rebuilding it would silently break the link between the data and
the screen that gated it.

The `ea4c0ee..b09f5eb` pull added three tasks —
`image_kernel/mi355x_vllm_hip_paged_attention_decode`,
`.../mi355x_vllm_triton_fused_moe_gemma4` and
`.../mi355x_vllm_triton_unified_attention_gemma4` — so the true unscreened count
is 71, not the 68 in the frozen metadata. Nothing else changes: all 18
pre-existing `image_kernel` tasks were already unscreenable (they ship no
parseable PyTorch reference module to compare against the pool), these three are
the same, and the blocked and implicated sets are untouched. Any claim on
`image_kernel` therefore carries no textual contamination evidence in either
direction, which was already true before the pull.

## Legacy checkpoint A/B

`checkpoint_ab.py` and `heldout_lm.py` are retained to make the 14B midtrain
failure reproducible. They are useful precisely because they showed the
instruction-following collapse described in
[`docs/EVAL_RESULTS.md`](../../docs/EVAL_RESULTS.md). They are not the
production evaluation route, because production has no CPT stage.

## Generalization and retention

The registry keeps whole-family holdouts out of training. Retention tests score
the model before and after a stage, but an offline smoke result is not comparable
to a full benchmark run; the source is recorded with the score. Champion
re-evaluation widens correctness and timing scrutiny rather than accepting a
training-time number unchanged.
