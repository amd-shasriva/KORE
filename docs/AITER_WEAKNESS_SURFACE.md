# The AITER weakness surface on gfx950

How to read this document. Every claim below is tagged with how it was
obtained, because the three kinds are not interchangeable:

- **[source]** — read directly out of the AITER tree at a pinned commit. These
  are facts about code, verifiable by anyone with the same commit. They are not
  performance claims.
- **[paper]** — reported by HipKittens (MLSys 2026, arXiv 2511.08083) on their
  harness. Quotable only with attribution. **Never** as our number.
- **[measured]** — produced by our harness on our gfx950 hardware.

There are **no [measured] rows in this document yet.** The reason is recorded
in [Measurement status](#measurement-status) rather than papered over, and no
[source] or [paper] claim below should be promoted into a performance number
without one.

## Pinned commits

Two different AITER checkouts exist on this machine and they are **not** the
same commit. Conflating them would attach a measurement to a build that never
produced it.

| Role | Path | Commit | Date |
|---|---|---|---|
| **What we measure** (editable install, `import aiter` resolves here) | `/home/shasriva/aiter` | `7e0d1162642f1727e0c8d9bdff318daedecfe331` | 2026-07-06 |
| Corpus / upstream reference clone | `/home/shasriva/third_party/aiter` | `702aacd62c8a6e2fbbc260da338c510ab76f1b1c` | 2026-08-01 |

The installed tree is an editable install (`amd_aiter-0.1.1.dev1+g7e0d11626`),
its working tree is clean, and the upstream clone is **194 commits ahead** of
it. Any measurement must cite `7e0d11626`, because that is the code that runs.

The dispatch logic quoted below was diffed between the two commits and is
**byte-identical**, so the analysis holds for both. That will not stay true;
re-diff before reusing this.

License: AITER is **MIT**, © Advanced Micro Devices, Inc. Permissive for use,
modification and redistribution, and it requires the copyright notice to travel
with any substantial portion. Anything derived from AITER must carry that
attribution.

### Import-shadowing hazard

The corpus clone is at `third_party/aiter`, whose package directory is also
named `aiter`. Any Python process whose working directory is that clone (or
which puts it on `PYTHONPATH`) imports the **source tree** instead of the
installed build, and then tries to JIT-compile it from scratch. A measurement
taken that way would silently describe the wrong commit. Run benchmarks from a
neutral directory and assert on `aiter.__file__`.

## What AITER actually does for attention backward [source]

`aiter/ops/mha.py::_flash_attn_backward` chooses between two implementations:

- `fmha_v3_bwd` — the hand-written assembly backward (fast path), and
- `mha_bwd` — the Composable Kernel backward (fallback).

The asm path is taken only when
`can_impl_fmha_v3_bwd(...) | can_impl_fmha_v3_bwd_gfx950()` holds and
`seqlen_q > 16`. The gfx950-specific gate requires **all** of:

| Condition | Excluded when |
|---|---|
| `alibi_slopes is None` | ALiBi is used |
| `bias is None`, `dbias is None` | any attention bias |
| `dropout_p == 0.0` | dropout in training |
| `not deterministic or seqlen_k <= 256` | deterministic mode at real seqlen |
| `nhead_q % nhead_k == 0` | ragged GQA ratios |
| `64 < hdim_q <= 128`, or `hdim_q == 192 and hdim_v == 128` | **head_dim 64**, head_dim 256 |
| `not swa` | **sliding-window attention** |

The generic gate additionally requires `not deterministic` **unconditionally**,
plus `hdim_q == hdim_v` and `64 <= hdim_q <= 192`.

### The default-argument cliff

This is the sharpest finding, and it is mechanical:

- `flash_attn_func` (dense, `aiter/ops/mha.py`) defaults **`deterministic=True`**.
- `flash_attn_varlen_func` defaults **`deterministic=False`**.

The flag is stored on the autograd context in forward and read back in
backward, so it reaches both gates unchanged. Therefore a **default** call to
the dense `flash_attn_func` at any sequence length above 256 fails both gates
and lands on the CK `mha_bwd` fallback — the asm backward is never reached.
The varlen entry point, with the opposite default, can reach it.

Two consequences worth stating plainly:

1. This is a coherent mechanical explanation for why a benchmark would find
   AITER's dense GQA backward far off the achievable bar, and it predicts the
   effect is a *dispatch* problem rather than a *kernel quality* problem.
2. Our own baseline wrapper `kore/tasks/aiter_ref_attn.py`
   (`aiter_flash_attn_backward_prepare`) calls `flash_attn_func` **with
   defaults**, so if we ever pointed a backward task at it, we would be timing
   the CK fallback while labelling it the AITER production bar.

Claim (2) is the one that matters for our credibility and it is checkable
without a GPU: the wrapper passes only `causal` and `softmax_scale`.

### Asm kernels exist but are gated off

The asm backward binaries are shipped for gfx950 and are **not** the missing
piece:

| arch | hd64 | hd128 | hd192 |
|---|---|---|---|
| gfx950 | 40 | 38 | 46 |
| gfx942 | 44 | 76 | 36 |

So gfx950 ships 40 `hd64` backward objects that the gfx950 gate can never
select (it requires `hdim_q > 64`, strictly), reachable only through the
generic gate's `hdim_q == 64 and is_v3_atomic_fp32` branch — which still
demands `not deterministic`.

Note also that gfx950 has **half** the `hd128` backward variants of the older
gfx942 (38 vs 76). Variant count is not performance, and this should not be
quoted as one; it is a weak signal that backward tuning on the newer part is
less mature, and it is consistent with the [paper] result.

## What HipKittens reports [paper]

Attributed to arXiv 2511.08083, measured on **their** harness, not ours:

- AITER's Llama GQA backward reaches ~30% of SoTA on MI355X; PyTorch SDPA ~24%.
- HipKittens is 1.0–2.1x faster than AITER on attention generally.
- HipKittens is ~1.8x better on GQA **non-causal** backward, using an 8-wave
  ping-pong schedule.

### The baseline is genuinely moving [source]

HipKittens is not merely "being upstreamed into AITER" — at our installed
commit it is already a **build dependency**. `aiter/jit/core.py` defines
`HIP_KITTENS_DIR` and clones `https://github.com/HazyResearch/HipKittens.git`
on demand, and five gfx950 `opus_gemm` headers implement the "HipKittens XCD
swizzle (Algorithm 1)".

What has landed so far is in **GEMM** (and, at upstream HEAD, MLA), **not** in
the FMHA backward path. The attention-backward surface the paper flags is
therefore still open at `7e0d11626`. Between our commit and upstream HEAD,
these attention commits landed and would need re-checking before any claim:
`5bd9adf3d [GFX950] opus fmha d128 kernel optimization`,
`874840aef Add OPUS gfx950 bf16 fmha d192x128 kernel`,
`9bc48ade4 bf16 asm mha: enhance kernel to avoid corner issue`.

## What our task pool measures today [source]

Counted over the 1,334 tasks under `kore/tasks/`:

- **63** tasks (~4.7%) declare an `aiter*` comparison baseline. The rest are
  torch/hipBLASLt.
- **80** backward tasks exist. **Every one** of them uses a torch baseline.
- **7** are attention-backward. All are `head_dim=128`, all **causal**, all
  torch-baselined:
  `flash_attn_backward_bf16`, `genb_attn_bwd_gqa_causal_bf16` / `_fp16`,
  `genb_attn_bwd_mha_causal_bf16` / `_fp16`,
  `genb_attn_bwd_mqa_causal_bf16` / `_fp16`.
- **Zero** non-causal attention-backward tasks exist.

Two gaps follow directly. First, the surface the paper identifies as AITER's
weakest is the one place our pool never uses AITER as the bar. Second, GQA
**non-causal** backward — the paper's single largest reported gap — is not
represented in the pool at all.

`kore/tasks/flash_attn_backward_bf16/driver.py` justifies its torch baseline
with "NO AITER backward kernel exists for these ops". At `7e0d11626` that is
**incorrect**: `_flash_attn_backward` dispatches a real AITER backward, and
`kore/tasks/aiter_ref_attn.py` already wraps it. The comment should be fixed
whether or not the baseline changes.

## Measurement status

`scripts/aiter_bwd_dispatch_probe.py` (submitted via
`scripts/spur_aiter_bwd_probe.sbatch`) is written and committed to settle the
[source] claims above on hardware. It has **not produced results yet**: the
account's concurrent-node QoS is saturated by datagen, and the job sits in
`PENDING(QOSGrpNodeLimit)` behind ~30 queued jobs at flat priority 1000.
Shrinking the request from an exclusive 8-GPU node to a single GPU did not
help, because the cap counts nodes and the datagen jobs hold theirs
exclusively.

The probe is designed so the cheap half still works under contention:

- **Phase A** replaces `fmha_v3_bwd` and `mha_bwd` with sentinels that raise on
  entry. The real gate logic runs untouched, so the observation of *which*
  kernel AITER selects is faithful, but neither kernel is compiled or launched.
  This matters because **neither backward module is prebuilt** in this
  environment (only `module_fmha_v3_fwd` is), so a naive probe pays a long JIT
  build before reporting anything. Phase A is a logical observation and is
  immune to a noisy neighbour.
- **Phase B** times the real kernels with the same cold-cache CUDA-event median
  the task harness uses (`kore/tasks/_attn_common.py`: L2 flush between iters,
  median of sorted per-iter deltas), and records min/max so spread is visible.
  Because the job may land on a shared node, it logs its co-tenants; a Phase B
  row taken beside another job must be read with that in mind.

Until Phase A runs, the dispatch table above is a **prediction from source**,
not an observation.

## Recommended task priorities

Ranked by (vendor baseline plausibly weak) x (we can measure it honestly).

1. **A new GQA non-causal backward task, bf16, `head_dim=128`.** The paper's
   largest reported AITER gap, and absent from our pool. It is the only place a
   win would be both a headline and unclaimed.
2. **`genb_attn_bwd_gqa_causal_bf16`** (`B=2 H=32 HKV=8 SQ=2048 D=128`). Closest
   existing task to the target surface and already shaped like Qwen3-Coder
   attention.
3. **`flash_attn_backward_bf16`** (`D=128`, causal MHA). Fix the incorrect
   "no AITER backward" comment first; it is a docs-level error today.
4. **Sliding-window and `head_dim=64` backward.** The gate excludes both
   outright, so AITER is structurally on its fallback path there. Lower headline
   value than (1) but the cleanest place to demonstrate a real gap.

Before any of these are used as an RL target, the baseline they are scored
against must be measured, and the dispatch path recorded alongside the number.
A speedup over AITER's CK fallback is **not** a speedup over AITER, and must
never be reported as one.
