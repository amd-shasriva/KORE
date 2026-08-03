# `kore/policy` — SFT and multi-turn RL

The production curriculum is:

```text
Qwen3-Coder-30B-A3B-Instruct → SFT → multi-turn RL → test-time scaling
```

The target is a code-specialized, instruct-only MoE. SFT uses
`configs/sft_coder30b_a3b.json`; it specifies full FSDP, BF16, explicit
`Qwen3MoeDecoderLayer` wrapping, one epoch, and a single retained checkpoint.

`sft.py` handles completion-only SFT. `grpo.py` provides the agentic RL loop:
the policy receives tool observations, proposes a revision, and is rewarded only
after KORE verifies and times it. Its distributed path uses `SHARD_GRAD_OP`
(ZeRO-2) with a full-weight generation replica. `FULL_SHARD` would repeatedly
re-gather parameters during autoregressive generation and is therefore not the
RL topology.

The reward remains correctness-gated. Roofline attainment can be used as a
potential-shaped auxiliary term, but only with a valid evidence artifact; the
current P0 result authorizes no family-specific empirical shaping. See
[`docs/P0_RESULTS.md`](../../docs/P0_RESULTS.md).

## Legacy modules

`midtrain.py`, `dpo.py`, `soup.py`, and `residual.py` remain to reproduce and
test the earlier 14B experiments. They are not the production chain.

The reason matters:

- continued pretraining needs a Base model; the 30B candidates are
  instruct-only;
- `residual.py` needs a compatible Base/Instruct pair;
- the 14B result showed that CPT on an instruct checkpoint destroyed
  instruction-following (`docs/EVAL_RESULTS.md`);
- DPO converts execution evidence into a weaker preference label, while the
  agentic RL loop can use compile, correctness, and timing feedback directly.

`residual.py` itself is conservative tensor arithmetic: it validates identical
keys, shapes, and tensor kinds before computing
`target + (instruct - base)` in FP32. That protects the legacy experiment from a
plausible-looking but mismatched model merge; it does not make a residual
available for the production model.
