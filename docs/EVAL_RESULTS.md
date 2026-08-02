# Stage-0 mid-train, evaluated against its own base

**Verdict: the mid-train checkpoint is a large, unambiguous win on the objective it
was trained on, and a catastrophic regression at generating kernels. It learned the
content of Triton and lost the ability to emit syntactically valid code. It is not
usable as a generation policy as it stands.**

This is the first evaluation in this repository ever run against a trained KORE
checkpoint. Before it, `runs/midtrain_14b_frontier` was 16 hours of compute and
55 GiB of weights that nothing had asked a question of.

Three things were measured on the held-out generalization scope, all against the
exact weights the run started from (`Qwen/Qwen3-14B` @
`40c069824f4251a91eefaf281ebe4c544efd3e18`) at a matched budget:

| # | Measurement | Result | Power |
| --- | --- | --- | --- |
| 1 | Held-out LM loss on held-out Triton kernel source | **1.5787 -> 0.8181 bits/token**, paired effect **-0.837** [-0.923, -0.749], 34/34 documents improved, Wilcoxon p = 3.8e-7 | 38,088 paired tokens over 34 documents |
| 2 | Held-out LM loss on general-domain text | **2.8152 -> 3.0175 bits/token**, paired effect **+0.375** [+0.059, +0.674], 4/18 improved, Wilcoxon p = 0.037 | 578 paired tokens over 18 documents — thin |
| 3 | Single-shot kernel generation, verified + cold-cache benched on gfx950 | correct kernels **23/34 -> 0/34**; compiled **32/34 -> 4/34**; exact McNemar p = **2.4e-7** | 34 paired tasks, 1 bit each — and the effect is still overwhelming |

The honest summary: **(1) is what the training bought, (2) is part of what it cost,
and (3) is the part that matters and it is much worse than "no improvement".** The
checkpoint predicts held-out Triton source far better than the base while being
unable to *write* Triton at all — 29 of its 33 parseable responses are not valid
Python, against 0 of 32 for the base. Sections 1 and 3 are not in tension; together
they say the run taught the model the domain and broke its output surface.

**Consequence: Stage-1 SFT and Stage-2 GRPO both sample from this checkpoint.** SFT
supervises on gold assistant turns and may well repair the surface, but nothing here
supports launching GRPO (which needs the policy to emit valid kernels to get any
reward signal at all) from these weights.

- **Measured on:** `master` @ `aeada9b3`, working tree (nothing committed)
- **Hardware:** one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`, 288 GiB), Slurm
  job `27715` on `crsuse2-m2m-079`, exclusive node
- **Stack:** Python 3.12.3, torch 2.10.0+rocm7.0, transformers 4.57.6, triton 3.6.0
- **Artifacts:** `runs/eval_ab_27715/` on the cluster (see [Artifacts](#artifacts))
- **Harness:** `kore/eval/checkpoint_ab.py`, `kore/eval/heldout_lm.py`;
  regression tests `tests/test_checkpoint_ab.py` (39 CPU),
  `tests/test_heldout_lm.py` (21 CPU), `tests/test_gpu_checkpoint_ab.py` (7, `-m gpu`)

---

## What was compared, and why that scope

| | |
| --- | --- |
| Candidate | `runs/midtrain_14b_frontier` — 13 safetensors shards, 55.0 GiB fp32, 14,768,307,200 parameters |
| Reference | `Qwen/Qwen3-14B` at the pinned commit the corpus was built against, 14,768,307,200 parameters |
| Scope | 34 tasks — the held-out reservation (45) minus the 11 whose optimized source leaked into the mid-train corpus |
| Load dtype | `bfloat16` for **both** arms |
| Decoding | greedy (`temperature = 0.0`), `max_new_tokens = 4096`, template thinking OFF, for **both** arms |
| Prompt | one turn, byte-identical per task across arms, digest-verified |
| Budget | 1 bench per task, `mode = parallel` — matched |

Both arms are loaded at bf16 deliberately. The mid-train output is fp32 on disk
because FSDP `FULL_STATE_DICT` gathers the bf16-mixed-precision fp32 master copy
(`docs/SFT_READINESS.md` F1), while the Hub base is bf16; loading one at fp32 and
the other at bf16 would confound precision with training. Template thinking is
disabled for the same class of reason documented in `docs/E2E_SERVING_GATE.md`: with
Qwen3's thinking on, a bounded token budget is spent inside `<think>` and the answer
never arrives, which reads as a capability difference and is a budget artifact.

### The scope is clean — verified, not assumed

The 11 excluded tasks are excluded because `gen_curriculum.py`'s Tier-4 win glob
had no held-out filter. Since every number here is a held-out claim, that exclusion
was re-verified rather than trusted:

```
KORE's own detector (kore.data.decontam.decontaminate_corpus, 8-gram directional
containment, threshold 0.78) re-run over ALL 86,010 rows of
data/b05factory/midtrain/corpus.jsonl against the whole 45-task reservation:

  rows scanned      86,010
  rows dropped          15   (all directional_containment)
  held-out tasks hit     9   -- every one already on the exclusion list
  IN-SCOPE HITS          0   -> scope_is_clean = True
```

The recorded evidence in `kore/tasks/taxonomy.py` reports 17 hits over 11 tasks from
a scan of the 9,956 forwarded curriculum chunks; this full-corpus re-run finds 15
over 9. The exclusion list is therefore a **superset** of what the detector finds in
the shipped corpus, which is the safe direction, and **no task inside the scored
scope is flagged**.

### But "clean" does not mean "novel", and this matters for result (1)

All 1,052 `genb_*` tasks come from the same authoring engines, and the 1,289
**training** tasks are in the corpus. So the held-out seeds are template siblings of
text the model trained on. An independent line-level containment scan (the largest
fraction of a seed's >=20-character lines found inside one corpus document, the same
statistic scale as the recorded 0.795-0.943 contamination evidence, but **without**
the detector's generic-scaffolding suppression, so these numbers are an upper bound):

| | max containment |
| --- | --- |
| median over the 34 in-scope seeds | **0.56** |
| 28 of 34 seeds | >= 0.50 |
| highest (`genb_cv_conv2d_7x7_s1_d1_fp16`) | 0.805 |
| lowest (`mla_decode_bf16`, `paged_attn_decode_bf16` — the two whole-family holdouts) | 0.123, 0.203 |

This is the signature of a shared generator template, not of a leaked kernel — the
detector, which suppresses generic scaffolding, flags none of them. But it means
result (1) must not be described as loss on text the model has seen nothing like.
[The gain is quantified against this below](#does-the-gain-track-corpus-overlap-yes).

---

## 1. Held-out LM loss on held-out Triton kernel source

The mid-train stage is continued **pretraining**, so next-token loss on held-out
in-domain text is the measurement that speaks directly to its objective. It is also
the only comparison on this scope with real statistical power: single-shot
generation yields one bit per task (~34 bits total), while teacher-forced loss over
the same 34 kernels yields 38,088 paired per-token measurements.

Both arms score the **identical token sequence** — same tokenizer, same
`vocab_size` (151,936), verified per document by token-id digest;
`compare_documents` refuses to compare arms whose tokenization differs, so the
per-token delta is a true paired difference.

| arm | bits/token | perplexity |
| --- | --- | --- |
| base | 1.5787 | 2.987 |
| **mid-train** | **0.8181** | **1.763** |

- **paired per-document effect: -0.8368 bits/token**, 95% bootstrap CI
  [-0.9233, -0.7486]
- Wilcoxon p = 3.8e-7, exact sign p = 1.2e-10, bootstrap p = 1e-4
- **34 of 34 documents improved** — no task got worse
- corpus-level (token-weighted): -0.7606 bits/token, perplexity ratio **0.590x**

A 48% reduction in bits per token with every single document improving is not a
marginal effect. **The mid-train run did what continued pretraining is supposed to
do.**

The adjacent-domain document set — each held-out task's torch `reference.py` oracle,
5,292 tokens — moves further in the same direction: 3.1206 -> 1.3405 bits/token,
paired effect -3.2882 [-3.5350, -2.9796], 34/34 improved. These are thin, highly
templated shim files, which is precisely why the effect is larger, and is a second
reason to read the magnitudes as partly template acquisition.

### Device independence of the measurement

The same base-arm scoring was run twice, once on a 2x64-core EPYC 9575F CPU
allocation and once on the MI355X:

| device | base seed bits/token | tokens |
| --- | --- | --- |
| CPU (crsuse2-m2m-103) | 1.579139 | 38,088 |
| MI355X (crsuse2-m2m-079) | 1.578699 | 38,088 |

They agree to 4.4e-4 bits/token — three orders of magnitude below the effect being
claimed. The number is a property of the weights, not of the device.

### Does the gain track corpus overlap? Yes.

Per-task LM improvement against per-task line-level corpus containment:

| | Pearson | Spearman |
| --- | --- | --- |
| containment vs improvement | **-0.625** | **-0.505** |

Negative means *more* improvement where *more* of the task's own text is already in
the corpus. Splitting on containment:

| group | n | mean delta (bits/token) | 95% CI | sign p |
| --- | ---: | --- | --- | --- |
| containment >= 0.60 | 16 | **-0.941** | [-1.056, -0.820] | 3.1e-5 |
| containment < 0.35 | 4 | **-0.408** | [-0.508, -0.238] | 0.125 |

and the two structurally distinct whole-family holdouts individually:

| task | containment | base | mid-train | delta |
| --- | ---: | ---: | ---: | ---: |
| `mla_decode_bf16` | 0.123 | 1.247 | 1.094 | **-0.152** |
| `paged_attn_decode_bf16` | 0.203 | 1.593 | 1.131 | **-0.462** |

**The gain is real everywhere — all 34 documents improved, including the two least
overlapping — but it is roughly 2.3x larger where the surface form is
well-represented in the corpus, and smallest (-0.152 bits/token) on the single most
novel task.** Read the headline -0.837 as domain adaptation plus template
acquisition, in unknown proportion, with -0.15 to -0.46 as the defensible floor for
genuinely unfamiliar operator families.

---

## 2. What it cost: general-domain text

Specialization is only a problem if it costs general ability, so the same
teacher-forced measurement was run on out-of-domain text (English + generic Python,
from the bundled retention samples).

| arm | bits/token | perplexity |
| --- | --- | --- |
| base | 2.8152 | 7.038 |
| **mid-train** | **3.0175** | **8.097** |

- **paired per-document effect: +0.3754 bits/token** (worse), 95% CI
  [+0.0589, +0.6743]
- Wilcoxon p = 0.037, exact sign p = 0.031
- only **4 of 18 documents improved**
- corpus-level: +0.2023 bits/token, perplexity ratio **1.15x**

**This is a statistically significant general-domain regression**, and it happened
*despite* 21,486 general-replay rows (25% of the corpus) being present — the
`general_replay_frac: 0.30` in `configs/midtrain_14b_full.json` was honored in the
built corpus, contrary to the stale note in `data/DATASET_STATUS.md` that records the
replay slice as deferred.

**The honest caveats, which are large:**

- **578 tokens over 18 documents is thin.** The CI excludes zero, but only just, and
  a 578-token probe is not a benchmark. This detects "did general LM quality move",
  not "how much".
- **The probe is not decontaminated.** Canonical benchmark items may sit inside the
  general-replay slice, which would *flatter* the trained arm — so if anything this
  biases against finding the regression that was found.
- A bits/token regression on 578 tokens is not the same claim as a benchmark
  regression. The accuracy view below points the same way but cannot reach
  significance at its sample size.

### Retention smoke (accuracy, not loss)

`kore.eval.retention` on the bundled smoke sets, thinking disabled, greedy:

| bench | n | base | mid-train | discordant | exact McNemar p |
| --- | ---: | --- | --- | --- | --- |
| MMLU | 12 | 11/12 = 91.7% | **8/12 = 66.7%** | 3, all base-only | 0.25 |
| HumanEval | 3 | 3/3 | 3/3 | 0 | 1.0 |

All three MMLU flips are base-correct / mid-train-wrong, and all three are cases
where the base answered the gold letter and the mid-train arm answered a different
one — the same direction as the loss regression in section 2, from an independent
instrument. It is **not significant**: three discordant pairs cannot be, and a
one-item change moves this MMLU number by 8.3 points.

These are reported because they are cheap and they rule out catastrophic collapse.
They are **not** the no-regression gate. A real retention verdict needs the full
splits, which requires HuggingFace egress this node does not have
(`KORE_EVAL_FULL=1` falls back to smoke offline).

---

## 3. Kernel generation: the headline question

This was designed as the low-power measurement — one greedy attempt per task is one
bit of evidence per task, and 34 bits cannot resolve a subtle effect. It did not need
to: the effect is a near-total collapse and clears an exact test by five orders of
magnitude.

Each arm gets one greedy attempt per task, verified and cold-cache benched through the
real `KoreEnv` on the same MI355X, back to back in one process so a model load cannot
perturb the timings either arm is compared on.

The funnel is reported as five separate stages, because collapsing them hides why a
small number is small.

| stage | mid-train | base | delta | discordant (m/b) | exact McNemar p |
| --- | --- | --- | --- | --- | --- |
| emitted `FULL_KERNEL` | 33/34 = 97.1% [85.1, 99.5] | 32/34 = 94.1% [80.9, 98.4] | +2.9% | 3 (2/1) | 1.0 |
| **compiled** | **4/34 = 11.8%** [4.7, 26.6] | **32/34 = 94.1%** [80.9, 98.4] | **-82.4%** | 30 (1/29) | **5.8e-08** |
| correct (SNR gate) | **0/34 = 0.0%** [0.0, 10.2] | **23/34 = 67.6%** [50.8, 80.9] | **-67.6%** | 23 (0/23) | **2.4e-07** |
| correct AND publication-timed | 0/34 = 0.0% | 12/34 = 35.3% [21.5, 52.1] | -35.3% | 12 (0/12) | 4.9e-04 |
| faster than its baseline | 0/34 = 0.0% | 4/34 = 11.8% [4.7, 26.6] | -11.8% | 4 (0/4) | 0.125 |

Brackets are Wilson 95% intervals. One base-arm task recorded an infrastructure
fault (`genb_moe_fused_moe_silu_fp16`: correct at 65.6 dB, then an infra fault during
timing) and is counted as a failure above; mid-train had none.

`fast_p` over the whole scope (uncorrected denominator): base 35.3% / 17.6% / 11.8% /
5.9% / 2.9% at p = 0 / 0.5 / 1 / 1.5 / 2; mid-train **0.0% at every p**. Base's
geometric-mean speedup over its correct-and-timed kernels is 0.467x.

**The mid-train checkpoint did not write a single correct kernel, and the base model
wrote 23. That is a catastrophic regression on the task this project exists to do,
and it is significant by a wide margin (exact McNemar p = 2.4e-7, 23 discordant
pairs, all in the base's favour).**

### Why: the checkpoint emits character-level-corrupted Python

The failure is not truncation, not the response contract, and not the harness. The
mid-train arm honors the `FULL_KERNEL` contract *slightly more often* than the base
(33/34 vs 32/34) and then emits code that will not parse:

| | mid-train | base |
| --- | ---: | ---: |
| parsed kernels that are not valid Python (`ast.parse`) | **29 of 33** | **0 of 32** |
| responses showing degenerate repetition (>= 20 identical consecutive tokens) | 1 | 0 |
| responses that look truncated | 4 | 2 |

The `ast.parse` failures: 11 `unmatched ')'`, 7 `invalid syntax. Perhaps you forgot a
comma?`, 6 `invalid syntax`, 2 `'(' was never closed`, and one each of
`too many nested parentheses`, `unexpected indent`, `invalid decimal literal`. What
they look like, verbatim:

```python
acc = tl.zeros([ BLOCK_M, HEAD_DIM], dtype tl.float32)     # the '=' is gone
m_i = tl.full([ BLOCK_M], -float("inf"), dtype tl.float32)  # again
              BLOCK_M: tl.constexpr, ..., HEAD Dim: tl.constexpr,   # space inside HEAD_DIM
               IS_CAusal: tl.constexpr,                     # case corruption
acc = tl.zeros([ BLOCK_M, HEAD_dim], dtype tl.float32)      # and again
offs_m = start_m * BLOCK_M + tl.arange( offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
```

Dropped `=` signs, spaces inserted inside identifiers, case flips, and a duplicated
fragment. This is a **surface-form** failure: the checkpoint has demonstrably learned
the *content* of Triton kernels (section 1: a 48% reduction in bits/token on held-out
kernel source) while losing the ability to emit a long, exactly-formed token sequence
in response to a chat prompt. That combination is what continued pretraining on raw
code chunks, with no instruction-format supervision of its own outputs, does.

Seed fidelity makes the same point quantitatively. The task prompt contains the seed
kernel, so the cheapest correct answer is to return it:

| | mid-train | base |
| --- | ---: | ---: |
| parsed kernels byte-identical to the seed | **0 of 33** | **5 of 32** |
| token similarity to the seed, median | **0.745** | **0.995** |
| token similarity to the seed, 10th percentile | **0.250** | 0.795 |

### What this comparison is, and is not

**It is not "the base model is the better kernel engineer".** The base arm's 23
correct kernels have a median token similarity of 0.997 to the seed it was shown, and
two of its four faster-than-baseline results — `genb_fx_reglu_act_fp16` at 2.36x and
`genb_moe_sigmoid_topk_norenorm_fp16` at 1.65x — are **byte-identical to the seed**.
Those numbers are the *seed's*, not the model's. This is exactly the failure mode
`kore/eval/policies.py` documents ("the campaign reported the seed's `fast_p` as if
it were KORE's"), and it is why the base arm's absolute numbers must not be quoted as
a model capability.

Read the comparison as what it actually measures: **can the model return a
syntactically valid kernel at all when it is shown one?** The base can, almost always
by copying. The mid-train checkpoint cannot — it corrupts what it is copying. That is
a smaller claim than "worse at optimization" and a more actionable one: it says the
Stage-0 output is **not usable as a generation policy as it stands**, which matters
because Stage-1 SFT and Stage-2 GRPO both sample from it.

Two further limits on the kernel numbers specifically:

- **The baselines are almost all torch.** 33 of the 34 in-scope tasks declare a
  `torch_*` comparison baseline; only `paged_attn_decode_bf16` is vendor-graded
  (AITER). A >1x result on this scope means "beats PyTorch", never "beats AITER or
  hipBLASLt" (`kore/tasks/README.md`, Baselines). No `KORE_BASELINE_IMPL` sentinel was
  emitted by any driver in this run, so `baseline_impl` is `None` throughout and the
  provenance rests on the declared baseline alone.
- **Only 12 of the base's 23 correct kernels are scoreable for speed.** Nine were
  demoted to `screening` timing grade and two (`mla_decode_bf16`,
  `paged_attn_decode_bf16` — the whole-family holdouts) came back
  `performance_ineligible`, so `fast_p` sees 12 tasks' worth of timing out of 34.
  Those 11 are verifier PASSES with no defensible timing; see harness defect 1, which
  is what stopped them being filed as correctness failures.

---

## Harness defects found and fixed

Every defect encountered while building this, whether or not it changed a number.

| # | Defect | Severity | Status |
| --- | --- | --- | --- |
| 1 | `bakeoff.evaluate_policy`'s per-task `correct` silently means "correct AND carrying an integrity-gated speedup". A kernel the verifier ACCEPTS whose timing was demoted to screening grade has `rr.speedup is None`, so a **verifier pass is filed as a correctness failure**. This is right for `fast_p` (an unscoreable kernel must contribute 0) and wrong for any funnel built on the same field. | reporting-fatal for this eval | **Fixed** — `correct_gate` (the reward's own gate) and `timed` are now additive fields on the per-task record; `correct` keeps its `fast_p` meaning so no existing consumer changes. Regression: `test_a_correct_but_screening_timed_kernel_is_correct_and_unscoreable`. |
| 2 | No way to point any eval at a checkpoint-vs-base comparison. `bakeoff` compares *policies*, `vs_opus` compares against a teacher, `generalization` reads an offline measures JSON. Nothing prompted two checkpoints identically on the held-out scope at a matched budget. | missing capability | **Fixed** — `kore/eval/checkpoint_ab.py`. |
| 3 | `kore.eval.policies._task_prompt` was private, so an A/B could only re-render the prompt in a second copy — a silent path for the two arms to drift apart. | latent correctness | **Fixed** — promoted to `policies.task_prompt`, private alias retained. `test_first_turn_messages_match_the_live_model_policy` asserts the replayed prompt equals what `model_policy` sends live. |
| 4 | `kore.eval.e2e_sglang_vllm._openai_compatible_generate` takes a single prompt STRING, so it cannot carry the KORE policy contract (system prompt + task turn). Reusing it for an endpoint-backed eval would have silently changed what the model was asked. | would have invalidated an endpoint run | **Worked around** — `checkpoint_ab.endpoint_generate` sends a full message list. The serving-gate module is owned elsewhere and was not modified. |
| 5 | `kore.policy.serve.load_generate` has no per-token log-likelihood path and no batched generation, so an LM-loss measurement was impossible through it and a 34-prompt greedy sweep would have paid full per-request latency 34 times. | throughput / capability | **Worked around** — `checkpoint_ab.load_hf_batch_generate` returns `generate`, `generate_batch` and `nll` from ONE load. `serve.py` is owned elsewhere and was not modified. |
| 6 | `heldout_lm.assert_documents_uncontaminated` initially only asked `filter_generalization_scope` whether a task was *contaminated*. That function does not reject **training** tasks, so held-out loss over a trained task would have been accepted silently. | would have invalidated a held-out claim | **Fixed** — the guard now requires positive membership in `generalization_eval_ids()`. Regression: `test_a_training_task_document_is_refused_too`. |
| 7 | `scripts/operations_registry.json` must enumerate every file under `scripts/`, so adding the two sbatch entrypoints broke `tests/test_operations_registry.py`. | test contract | **Fixed** — both registered as `diagnostic`. (This file is outside the ownership boundary for this work; only the two new records were added.) |
| 8 | `kore.policy.format.parse_response` **leaves the opening ```` ```python ```` fence inside the extracted kernel when the closing fence is missing** — i.e. whenever a response is cut off mid-code-block. The result is a syntax error on line 1 of a kernel the model may otherwise have written correctly. Reproducer: `parse_response("FULL_KERNEL\n```python\nimport triton\n")` returns a kernel starting with ```` ```python ````; the same input with a closing fence parses correctly. | real, but changed nothing here | **Reported, not fixed.** It hit 3 of 34 mid-train tasks and **0 tasks were harness-attributable**: all 3 are independently invalid after stripping the fence (`Perhaps you forgot a comma?`, `too many nested parentheses`, `unindent does not match`). `kore/policy/format.py` is outside this work's ownership boundary, and patching around it inside `kore/eval` would have made the replayed source differ from what `model_policy` submits live — which is the property that makes this measurement credible. It belongs to that module's owner. |
| 9 | `checkpoint_ab.observation_summary` truncated `Observation.error_text` from the FRONT. `error_text` is a traceback, so the informative last line was discarded and every compile failure read as an anonymous frame of `_genops._run_correctness`. This actively obstructed diagnosing result 3. | self-inflicted, reporting | **Fixed** — the field is now `error_tail` and keeps the last 400 characters, marked with a leading `...` when clipped. Regressions: `test_error_text_is_truncated_from_the_END_not_the_front`, plus short-string and no-error cases. The shipped `measures_*.json` from job 27715 predates the fix and carries the old `error_head`; the syntax-error taxonomy in section 3 was recovered by re-parsing the archived completions on CPU, which needs no GPU. |

### Infrastructure faults, not code defects

| Fault | Evidence | Handling |
| --- | --- | --- |
| **Every GPU submission on `amd-spur` was instantly requeued into `JobHoldMaxRequeue`** for ~15 minutes, `StartTime=N/A`, `Priority=0`. Reproduced with a 2-CPU, 1-GPU, 5-minute no-op; CPU-only submissions ran normally on the same nodes throughout. Not a job-spec problem: `--gres=gpu:8`, `--gres=gpu:mi355x:1`, with and without `--exclusive`, all held. | jobs 27621, 27636, 27652, 27660, 27682, 27689-27691, 27695-27698, 27701, 27704, 27705 | Resubmit, do not release (releasing a held job does not clear it). **Job 27715 landed on the 6th attempt** and ran to completion. A CPU-only fallback (`scripts/spur_eval_ab_cpu.sbatch`) was built and run in parallel; its base-arm LM scores are the CPU/GPU cross-check in section 1. |
| **The mid-train checkpoint cannot be moved to the GPU box that has the container tooling.** The only route to the cluster is an SSH tunnel to `127.0.0.1:22023`; measured 0.9 MB/s single-stream and 1.2 MB/s over two parallel streams, i.e. ~13 hours for 55 GiB. The cluster's IP resolves but port 22 is unreachable directly. | `rsync --info=progress2`; `dd` over two concurrent channels | The whole evaluation runs on the cluster, which is gfx950 (MI355X) and therefore matches the tasks' declared `gpu_target` anyway. The SGLang container route in `docs/E2E_SERVING_GATE.md` is unavailable there (no `docker`/`podman`/`apptainer`), so the HF batched path was used instead. |

### Which serving route was used, and why

`docs/E2E_SERVING_GATE.md` documents a container-hosted SGLang endpoint as the sound
way to serve a 14B model on this hardware, and it is — on the box that has docker and
the image. It was **not** usable here: the checkpoint lives on cluster NFS, the
cluster has no container runtime, and the checkpoint cannot be transferred (above).

The route taken is `transformers` in-process, batched, via
`checkpoint_ab.load_hf_batch_generate`, which also gives the per-token likelihoods
section 1 needs and which an OpenAI-compatible endpoint cannot expose at all. Three
things it pins that a naive port would get wrong:

- **left padding** — a batched decoder-only `generate` with right padding continues
  from pad tokens and produces garbage for every sequence shorter than the longest
  in its batch;
- **`enable_thinking=False`** — matching `serve.load_generate`, so the token budget
  reaches the kernel;
- **one dtype for both arms** — see above.

Measured cost: 839 s for the base arm's 34 completions at `batch_size 12`
(~25 s/completion), which is why the batched path matters — 34 sequential requests
at 4,096 tokens each would have dominated the job.

---

## Reproduction

Everything below is exact. Phases are separately resumable; phase B can be re-run
without re-generating.

```bash
# --- one node, all three phases (what produced runs/eval_ab_27715/) ---------
export SPUR_CONTROLLER_ADDR=http://crs-m2m-cpu-spur-005:6817
sbatch scripts/spur_eval_ab_1node.sbatch
# GPU submissions may land in JobHoldMaxRequeue: scancel and RESUBMIT, do not
# release. Job 27715 landed on attempt 6.

# --- or by hand, phase at a time -------------------------------------------
export PYTHONPATH=. GPU_TARGET=gfx950 HF_HUB_OFFLINE=1
export KORE_BENCH_COLD=1 KORE_VERIFIED_CORRECTNESS=1
OUT=runs/eval_ab

# phase A: one model load per arm -> generations + LM loss + retention
HIP_VISIBLE_DEVICES=0 python -m kore.eval.checkpoint_ab run-arm \
  --arm base --outdir $OUT --backend hf-batch --model Qwen/Qwen3-14B \
  --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 --dtype bfloat16 \
  --max-tokens 4096 --temperature 0.0 --batch-size 12 --retention mmlu,humaneval

HIP_VISIBLE_DEVICES=0 python -m kore.eval.checkpoint_ab run-arm \
  --arm midtrain --outdir $OUT --backend hf-batch \
  --model runs/midtrain_14b_frontier --dtype bfloat16 \
  --max-tokens 4096 --temperature 0.0 --batch-size 12 --retention mmlu,humaneval

# phase B: both arms benched on the SAME idle device, back to back
for ARM in base midtrain; do
  HIP_VISIBLE_DEVICES=0 python -m kore.eval.checkpoint_ab measure \
    $OUT/generations_$ARM.jsonl --out $OUT/measures_$ARM.json \
    --arm $ARM --budget 1 --mode parallel
done

# phase C: pure, CPU, no GPU needed
python -m kore.eval.checkpoint_ab compare \
  --candidate $OUT/measures_midtrain.json --reference $OUT/measures_base.json \
  --out $OUT/report_kernel_ab
python -m kore.eval.heldout_lm \
  --candidate $OUT/lm_scores_midtrain.json --reference $OUT/lm_scores_base.json \
  --out $OUT/report_heldout_lm
```

Contamination re-verification (CPU, ~25 min over the 683 MB corpus):

```bash
python - <<'PY'
from kore.data.decontam import build_heldout_ngrams, decontaminate_corpus
from kore.tasks.registry import generalization_eval_ids
import json
rows = (json.loads(l) for l in open("data/b05factory/midtrain/corpus.jsonl") if l.strip())
_, stats = decontaminate_corpus(rows)
scope = set(generalization_eval_ids())
hits = [e for e in stats["evidence"]
        if str(e.get("reference_id","")).split(":")[1:2] and
           str(e["reference_id"]).split(":")[1] in scope]
print("dropped:", stats["n_dropped_contaminated"], "IN-SCOPE hits:", len(hits))
PY
```

Line-level containment (the overlap measure in
[the novelty caveat](#but-clean-does-not-mean-novel-and-this-matters-for-result-1),
CPU, ~1 min):

```bash
python - <<'PY'
import collections, json
from kore.tasks.registry import generalization_tasks
seeds, owners = {}, collections.defaultdict(set)
for t in generalization_tasks():
    seeds[t.task_id] = {l.strip() for l in t.seed_source.splitlines() if len(l.strip()) >= 20}
    for l in seeds[t.task_id]:
        owners[l].add(t.task_id)
best = collections.defaultdict(float)
for line in open("data/b05factory/midtrain/corpus.jsonl", encoding="utf-8"):
    if not line.strip():
        continue
    hits = collections.defaultdict(set)
    for l in (json.loads(line).get("text") or "").splitlines():
        for tid in owners.get(l.strip(), ()):
            hits[tid].add(l.strip())
    for tid, m in hits.items():
        best[tid] = max(best[tid], len(m) / len(seeds[tid]))
for tid, frac in sorted(best.items(), key=lambda kv: -kv[1]):
    print(f"{frac:.3f}  {tid}")
PY
```

Tests:

```bash
python -m pytest tests/test_checkpoint_ab.py tests/test_heldout_lm.py -q   # CPU, 60 tests
python -m pytest -m gpu tests/test_gpu_checkpoint_ab.py -q -o addopts=      # 3 pass on a GPU
KORE_AB_TEST_MODEL=<checkpoint-or-cached-hub-snapshot> \
  python -m pytest -m gpu tests/test_gpu_checkpoint_ab.py -q -o addopts=    # 5 pass
python -m pytest -q -p no:warnings                                         # whole suite
```

The `gpu` tests skip with a reason when there is no accelerator, no checkpoint, or
no endpoint. Verified locally: 3 passed / 4 skipped with a GPU and no model,
5 passed / 2 skipped with a GPU and real Qwen3-14B weights.

Whole suite after this work: **7,972 passed, 4 skipped, 58 deselected** in 142 s.

> **`pytest -q` prints no verdict, and that is a trap.** `pyproject.toml`'s `addopts`
> already contains `-q`, so the documented `python -m pytest -q -p no:warnings`
> resolves to `-qq`, which suppresses the `N passed` summary line entirely — the run
> ends on a bare `[100%]` and looks like it produced no result. The exit code is still
> correct. To see the counts, override `addopts`:
> `python -m pytest -p no:warnings --tb=no -q -o addopts="--strict-markers --import-mode=importlib -m 'not gpu and not release'"`.

---

## What remains unmeasurable, and why

- **Whether mid-train improves kernel *quality*.** The question is currently
  unaskable: a model that emits invalid Python has no quality to measure. It becomes
  askable again only after the output surface is repaired (Stage-1 SFT is the obvious
  candidate, and the same harness re-run on the SFT output is the natural next
  measurement). Everything here is also single-shot greedy at one bench per task,
  whereas KORE's protocol is serial refinement with verifier feedback; multi-turn
  needs feedback in the loop and cannot use the decoupled replay path (use
  `kore.eval.policies.model_policy` live).
- **Whether sampling, a larger token budget, or a different prompt recovers the
  surface.** All measurements here are greedy at `max_new_tokens = 4096` with template
  thinking off, matched across arms. A temperature sweep, a raw-completion prompt
  instead of the chat template, or few-shot formatting might each partially recover
  valid syntax; none was tried, so "the checkpoint cannot emit valid Triton" is
  established only under **this** decoding configuration. That configuration is the
  one the campaign's own bake-off uses, which is why it was the one measured.
- **A real general-capability verdict.** The retention smoke sets are 10-20 items
  and the general LM probe is 578 tokens. Both point the same way as of now (the
  loss probe significantly, the accuracy probe not at all), but the full splits need
  HuggingFace egress this node does not have. Until then the section-2 regression is
  a signal, not a measurement.
- **How much of result (1) is generalization versus template acquisition.**
  Quantified above (-0.94 bits/token at high overlap, -0.41 at low, -0.15 on the most
  novel task) but not separated. Separating it needs held-out kernels from a
  *different* generator, which this corpus does not contain.
- **Anything about vendor-baseline (AITER/hipBLASLt) competitiveness.** 33 of the 34
  in-scope tasks declare a `torch_*` baseline; only `paged_attn_decode_bf16` is
  vendor-graded. A speedup on this scope means "beats PyTorch", never "beats the
  state of practice" (`kore/tasks/README.md`, Baselines).
- **Speed on 22 of the 34 tasks, for either arm.** Nine of the base's correct kernels
  were demoted to `screening` timing grade and two came back
  `performance_ineligible`, so `fast_p` rests on 12 tasks' timing out of 34. Whatever
  is demoting those drivers is worth fixing before any speed claim is made on this
  scope; it is not investigated here.
- **Whether the general-domain regression matters downstream.** Stage-1 SFT is
  explicitly held and was not launched; no `runs/sft_14b_frontier` was written.

## Artifacts

All under `runs/eval_ab_27715/` on the cluster (job 27715, `crsuse2-m2m-079`):

| File | Contents |
| --- | --- |
| `backend_{base,midtrain}.json` | resolved model identity, dtype, param count, vocab size, device |
| `generations_{base,midtrain}.jsonl` | every completion, its parsed kernel, prompt digest, wall time |
| `lm_scores_{base,midtrain}.json` | per-document token counts, summed NLL, bits/token, token-id digest |
| `retention_{base,midtrain}.json` | MMLU / HumanEval smoke, per item |
| `measures_{base,midtrain}.json` | per-task verifier + bench outcome and the funnel summary |
| `report_kernel_ab.{json,md}` | the paired kernel comparison |
| `report_heldout_lm.{json,md}` | the paired LM comparison, per document kind |

`runs/eval_ab_cpu_27700/lm_scores_base.json` holds the CPU-side re-scoring used for
the device-independence check. The two corpus scans live at
`/tmp/authoritative_decontam.json` (detector re-run) and
`/tmp/corpus_containment.json` (line-level overlap) on the cluster login node.
