# Training-side BACKWARD DRAFT tasks: on-gfx950 verification checklist

STATUS: STAGED DRAFTS. Not live. Do NOT promote any task until the boxes below are
checked on a real gfx950 / MI350X (CDNA4) node with the installed torch(ROCm) /
AITER / AOTriton stack.

These tasks are integrity-critical AND they are the HARDEST class to get right
(gradients, where silent bugs hide). Everything here has been CPU-verified for
ORACLE + SEED-MATH correctness ONLY. NOTHING here has been GPU-verified: the Triton
seeds have not been compiled, the framework/vendor baselines have not been run, and
no timing has been measured. Do NOT claim GPU verification.

Drafted family (4 tasks, all bf16, gpu_target gfx950, snr_threshold 25 dB):

| task_id | grads returned | candidate entry |
|---------|----------------|-----------------|
| softmax_backward_bf16 | dx | `softmax_backward(y, dy)` |
| layernorm_backward_bf16 | dx, dgamma, dbeta | `layernorm_backward(x, gamma, dy)` |
| gemm_backward_bf16 | dx (dgrad), dw (wgrad) | `gemm_backward(x, w, dy)` |
| flash_attn_backward_bf16 | dq, dk, dv | `flash_attn_backward(q, k, v, o, do, lse, causal=True)` |

Deliberately NOT drafted (reserved HELDOUT_FAMILIES): MLA / latent-attention
backward, paged-KV backward. None present.

## 0. Why these are safe while staged

- They live at `kore/tasks/_drafts/training/<id>/task.yaml` (THREE dir levels under
  `kore/tasks/`). The registry discovers tasks with `TASKS_DIR.glob("*/task.yaml")`
  (ONE level, see `kore/tasks/registry.py::_discover`), so these are NOT
  auto-discovered and cannot enter any train/eval run or the running campaign.
  VERIFIED: `registry.task_ids()` == 251 (unchanged), contains none of the 4 draft
  ids, and the one-level glob matches nothing under `_drafts/`.
- Family classification (built directly, not via registry): `flash_attn_backward` ->
  `attention`, `layernorm_backward` -> `layernorm`, `softmax_backward` -> `softmax`,
  `gemm_backward` -> `gemm`. All are TRAINABLE families (none is `mla` /
  `paged_attention`), so if/when promoted they extend the training frontier, not the
  held-out eval. VERIFIED via `operator_family` / `is_heldout` (heldout=False).
- No draft id collides with a live task (the only pre-existing backward task is
  `rmsnorm_backward`; these four are new ids).
- Untouched: `data/`, `runs/`, live `kore/tasks/*/`, campaign/config files.

## 1. What has already been verified (CPU, oracle + seed-math correctness)

- `python kore/tasks/_drafts/training/_cpu_sanity_check.py` -> ALL PASS. For each
  task, on tiny float64 cases, each reference oracle is corroborated THREE ways:
  1. FORWARD: `reference.reference_forward` vs an INDEPENDENT forward
     (torch.einsum / manual softmax / manual LayerNorm). Agreement ~143 to 151 dB.
  2. FINITE DIFFERENCE (the mandated check): `reference.reference_grads` (torch
     AUTOGRAD oracle) vs a CENTRAL numerical gradient of `L = sum(upstream *
     F_independent(inputs))`. Because the FD differentiates the INDEPENDENT forward,
     agreement proves the oracle is the true gradient of the intended op, not merely
     self-consistent autograd. Agreement ~133 to 144 dB (limited by the oracle's
     fp32, not by the gradient) across dq/dk/dv, dx/dgamma/dbeta, dx, dgrad/wgrad.
  3. ANALYTIC: the oracle vs the closed-form backward the SEED implements (softmax
     JVP; LayerNorm dx/dgamma/dbeta; dgrad/wgrad; FlashAttention-2 dS/dQ/dK/dV).
     Agreement ~133 to 150 dB. This validates the SEED's MATH on CPU (the Triton
     kernel itself still needs gfx950).
  A wrong gradient (missing scale, wrong reduction axis, missing the softmax
  row-mean subtraction, wrong dS delta term) would score well below 30 dB, so the
  ~140 dB margin is decisive.
- `python -m py_compile` succeeds on every drafted `.py` (SYNTAX ONLY; NOT a Triton
  compile -- the kernels have not been built for gfx950).
- No U+2012..U+2015 dashes and no U+2212 minus anywhere under `_drafts/training/`
  (ASCII only), verified with `grep -nP`.

## 2. Global gfx950 verification (applies to EVERY task)

For each `<id>`, run from the repo root with the KORE venv active:

```
cp kore/tasks/_drafts/training/<id>/seed_triton.py kore/tasks/_drafts/training/<id>/kernel.py
python kore/tasks/_drafts/training/<id>/driver.py --shape "<primary shape>"                              # correctness
python kore/tasks/_drafts/training/<id>/driver.py --bench-mode --impl reference --shape "<primary>"      # baseline runs + times
python kore/tasks/_drafts/training/<id>/driver.py --bench-mode --impl candidate --shape "<primary>"      # seed times + anti-hack re-verify
rm kore/tasks/_drafts/training/<id>/kernel.py
```

Per task, confirm ALL of:

- [ ] SEED COMPILES: the Triton `seed_triton.py` compiles on gfx950 (no
      LDS-over-160KiB, no unsupported dtype, no non-pow2 `tl.arange`) and produces
      output for every `shapes` entry (minimal, primary, all validation).
- [ ] SEED CORRECT: correctness run prints `allclose: True` and `SNR` >= the task's
      `snr_threshold` (25 dB) as the WORST over all returned gradients, across all
      >=5 reseeded trials AND all `shapes`.
- [ ] BASELINE RUNS: `--impl reference` runs without error and prints `median_ms`
      (it is the framework/vendor backward -- see the per-task baseline note below).
- [ ] BENCH RUNS: both `--impl reference` and `--impl candidate` print `median_ms`,
      and the post-timing anti-hack re-verification prints `allclose: True`.
- [ ] NON-POW2 EDGE: the `*_backward` validation shape with a non-power-of-2 trailing
      dim (N tail / seqlen) is correct (exercises the masking in the seed).

## 3. Shared-driver promotion requirement

Every task's `driver.py` is a thin wrapper that imports `_training_common` from its
PARENT directory. When promoting a task from `kore/tasks/_drafts/training/<id>/` to
the live `kore/tasks/<id>/`:

- [ ] Copy `kore/tasks/_drafts/training/_training_common.py` to
      `kore/tasks/_training_common.py` (once), so the promoted `driver.py` (parent
      dir now `kore/tasks/`) still resolves it. (Alternatively inline it, as
      `rmsnorm_backward` does with its own single-file driver.)
- [ ] Re-run the task's driver from the promoted location to confirm imports resolve.
- [ ] Decide the promoted `comparison_baseline` label (see baseline note below).

## 4. Baseline honesty (READ THIS -- perf-only bars)

PROMINENT NOTE: AITER ships NO standalone backward python op for any of these four
ops (it exposes forward `rms_norm` / `layer_norm` / softmax-via-framework /
`gemm_a8w8` / `flash_attn_func`, but no `*_backward` symbol that
`kore/tasks/aiter_ref.py` wraps). Therefore the `--impl reference` baseline for
each task is a FRAMEWORK / vendor-library backward, used as the PERF-ONLY bar. The
correctness ground truth is ALWAYS the fp32 autograd oracle in `reference.py`.
NEVER weaken the oracle to match a baseline.

| task | `--impl reference` baseline | nature |
|------|-----------------------------|--------|
| softmax_backward_bf16 | autograd through `torch.softmax` (aten `_softmax_backward_data`) | framework fused kernel; perf-only bar |
| layernorm_backward_bf16 | autograd through `F.layer_norm` (aten `native_layer_norm_backward`) | framework fused kernel; perf-only bar |
| gemm_backward_bf16 | `dY @ W` and `dY^T @ X` via `torch.matmul` | REAL hipBLASLt GEMMs -- a strong vendor bar, NOT weak eager |
| flash_attn_backward_bf16 | autograd through `F.scaled_dot_product_attention` | fused flash backward on ROCm (AOTriton/CK) IF it dispatches there; else math backend (still a valid perf bar) |

- [ ] Confirm each baseline actually runs on gfx950 with this stack. If the SDPA
      backend does NOT pick the fused flash path for flash_attn_backward, the
      baseline still works (math backend) but is a weaker bar -- note it, do not fail.

## 5. Per-task status + FLAGS (read before promoting each)

Legend: CONFIDENCE = my confidence the SEED is correct-and-compiles on gfx950
(oracle confidence is HIGH for all four -- CPU triple-checked). FLAG = the specific
thing to confirm on gfx950 beyond the global checks.

### softmax_backward_bf16 -- CONFIDENCE: MEDIUM-HIGH
- Oracle: `dx = y*(dy - <dy,y>_row)`, autograd-verified (FD ~138-144 dB).
- Seed: one program/row, two column-streamed passes, fp32 reduction. Simple; low
  risk. Mirrors the verified `rmsnorm_backward` seed structure.
- FLAG: candidate entry is `softmax_backward(y, dy)` where `y` is the SAVED softmax
  OUTPUT (probabilities), NOT the pre-softmax logits. `candidate_grads` recomputes
  `y = softmax(x)` in torch and passes it in (a real training kernel saves `y`).
  Confirm the seed reads `y` as probabilities.

### layernorm_backward_bf16 -- CONFIDENCE: MEDIUM
- Oracle: standard LayerNorm backward, autograd-verified (FD ~136-142 dB). eps=1e-5
  (PyTorch LayerNorm default) is baked into `reference.EPS` AND the seed default;
  they MUST stay equal.
- Seed: per-row dx (three column-streamed passes) + `tl.atomic_add` of dgamma/dbeta
  into fp32 buffers, then cast to bf16.
- FLAGS: (a) confirm `tl.atomic_add` on an fp32 buffer performs and is correct at
  M up to 32768 (heavy contention -- a two-stage reduction is the intended
  optimization, not a correctness fix); (b) dgamma/dbeta reduce over up to 32k
  tokens, so bf16 output of a large sum is the error-prone part -- the seed keeps
  the accumulation in fp32; confirm the worst-gradient SNR (dgamma/dbeta) still
  clears 25 dB (their allclose bar is deliberately looser, `atol=2e-1`, matching
  `rmsnorm_backward` dw).

### gemm_backward_bf16 -- CONFIDENCE: MEDIUM
- Oracle: `dx = dY @ W`, `dw = dY^T @ X`, autograd-verified (FD ~140-143 dB).
- Seed: one generic tiled matmul kernel called twice; for `dw` the `dY^T` is
  expressed by SWAPPING dY's row/col strides (no physical transpose).
- FLAGS: (a) confirm the stride-swapped (effectively column-major) read of dY in the
  `dw` call compiles and is correct on gfx950 (it is the one non-standard access
  pattern; correctness was CPU-proven via the analytic check, but the Triton
  swapped-stride load is GPU-unverified); (b) confirm fp32 accumulation keeps `dw`
  (reduction over up to 32k tokens) above 25 dB; (c) BLOCK_K=32 / GROUP_M=8 are
  untuned starting points.

### flash_attn_backward_bf16 -- CONFIDENCE: LOW (FLAG HIGH)
This is the highest-risk draft: gradients AND the most complex seed. The ORACLE is
solid (autograd on fp32 causal SDPA; FD ~133-148 dB; the analytic FA2 dS/dQ/dK/dV
matches it ~133-148 dB), but the Triton SEED is intricate and entirely
GPU-unverified. Do NOT promote without careful on-gfx950 correctness AND a manual
read of the kernel.
- FLAGS (HIGH priority, confirm each on gfx950):
  1. SEED COMPILES + CORRECT: the 3-kernel FA2 backward (delta preprocess; dK/dV per
     KV-block; dQ per Q-block) compiles (watch LDS/register pressure: BLOCK=64 with
     D=128 makes [64,128] fp32 tiles; `tl.dot` on fp32 operands as the forward seed
     does; `tl.trans` then `.to(bf16)` for the reduction dots) and clears 25 dB
     worst-of-{dq,dk,dv} on ALL shapes including the S=1000 non-multiple-of-BLOCK edge.
  2. SAVED-ACTIVATION CONVENTION: the candidate is GIVEN `o` (forward output) and
     `lse` (row log-sum-exp, `lse_i = m_i + log(sum_j exp(scores_ij - m_i))`, fp32,
     shape [B,H,S]). Confirm a real flash forward on this node saves `lse` in exactly
     this definition/units; the seed recomputes `P_ij = exp(scale*q.k - lse_i)`.
  3. CAUSAL ALIGNMENT: oracle, seed, and the SDPA baseline all use TOP-LEFT causal
     (query i attends keys j <= i) with Sq == Skv. If any path disagrees, fix the
     mask -- keep oracle and baseline in agreement, never weaken the oracle.
  4. MHA ONLY: this draft is H == KV (no grouping). GQA/MQA backward (which must SUM
     dk/dv across the query heads sharing each KV head) is NOT drafted; do not feed
     grouped shapes until that reduction is added.
  5. ORACLE MEMORY: `reference_grads` materializes the [B,H,S,S] scores in fp32 (e.g.
     B=1,H=32,S=4096 -> a few GB across scores+probs+grads). Fine on MI350X (288 GB)
     but confirm it does not OOM alongside the candidate; reduce trial count via
     `KORE_CORRECTNESS_TRIALS` if needed.
  6. BASELINE: confirm `F.scaled_dot_product_attention` backward dispatches to a
     fused flash kernel here (else it is the math backend -- still valid, weaker bar).

## 6. If a FLAG fails

Reference correctness is paramount. If a baseline does not match the oracle on
gfx950, do NOT weaken the oracle to match the baseline. The oracle is the fp32
autograd ground truth (CPU-verified against BOTH finite differences and an
independent analytic backward). Instead: (a) fix the candidate/baseline CALL
(args / layout / saved-activation convention / causal alignment) so it computes the
same gradient, or (b) keep the oracle and set the perf baseline to the closest
framework/vendor backward (documented perf-only, as done here), or (c) hold the task
back. If a SEED is wrong or will not compile, fix the seed (the oracle stays); the
seed is only the starting point the policy optimizes, but it must compile and clear
25 dB before promotion. Update this checklist with the resolution before promoting.
