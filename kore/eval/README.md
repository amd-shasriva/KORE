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

The 412-task suite has 402 tasks runnable on gfx950 under the discovery filter.
Its published Opus speed bars are 6.89x (`torch2hip`), 6.69x (`hip2hip`), and
2.13x (`triton2triton`). The runner places those values next to measured means;
it never treats them as KORE measurements.

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
