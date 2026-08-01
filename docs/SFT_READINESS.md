# Stage-1 SFT launch readiness

**Verdict: CONDITIONAL GO — one launch-fatal blocker, two durability blockers.**

The SFT stage trains. It was run end to end on this box against the real
`Qwen/Qwen3-14B` weights: identity resolution, template masking, dataset load,
FSDP construction, real optimizer steps, a 221 GB checkpoint write, and a resume
from that checkpoint all execute on the current `master`. Nothing in today's
changes broke the training path.

What is *not* ready is the launch itself. `configs/sft_14b_full.json` points at a
dataset path that no tool in this repo produces, and the failure surfaces only
after every rank has loaded 14B of weights. Two further defects mean that if the
run dies mid-flight it restarts from step 0 rather than from its last checkpoint.
All three have one-line-scale patches below.

- **Verified on:** `master` @ `3dfe1e8` (working tree dirty — several agents editing concurrently)
- **Hardware:** 2 × AMD Instinct MI350X (gfx950), HIP ordinals 6 and 7
- **Stack:** Python 3.10.14, torch 2.10.0+rocm7.0, transformers 4.57.6, trl 0.29.1, accelerate 1.14.0, peft 0.19.1, datasets 3.6.0. `flash_attn` is **absent** → SDPA.
- **Regression tests:** `tests/test_sft_launch_readiness.py` (30 pass, 4 xfail — the xfails *are* the blocker list, and turn green when the patches land)

| # | Item | Verdict |
|---|---|---|
| 1 | SFT data exists and is loadable | **PASS** (config path is wrong — Blocker 1) |
| 2 | Loss-masking template surgery | **PASS** — unqualified |
| 3 | Model identity resolution | **PASS** |
| 4 | SFT can actually start training | **PASS** — real 14B, real steps, real checkpoint, real resume |
| 5 | Handoff from midtrain | **PASS with defects** (Blockers 1 & 3) |
| 6 | Memory and time | **PASS with defects** (Blocker 2) |

---

## Blockers

### Blocker 1 — the shipped SFT config points at a dataset that nothing produces

**Severity: launch-fatal.** This is the one that stops the run.

```
configs/sft_14b_full.json : "dataset_path": "data/sft/multicap.jsonl"    <- does not exist
data/release/reassemble.sh: cat sft/multicap.jsonl.gz.part* | gunzip
                            > ../b05factory/sft/multicap.jsonl           <- what is actually written
DATASET_STATUS.md         : data/b05factory/sft/multicap.jsonl (630MB)   <- what is documented
```

`data/sft/` does not exist in this checkout and no script creates it. The
campaign happens to paper over this — `scripts/run_campaign.py:2130` overwrites
`dataset_path` with `<--data-root>/sft/multicap.jsonl`, and the campaign's
`--data-root` default is `data`, so the *campaign* is only correct when it is
invoked with `--data-root data/b05factory`. A direct
`scripts/launch_distributed.sh sft configs/sft_14b_full.json` — the documented
way to launch this stage, and the one the midtrain handoff will use — reads the
config verbatim and fails.

It fails *expensively*. In `train_sft` the dataset is not touched until line 338,
which is after the tokenizer load (282) and after
`AutoModelForCausalLM.from_pretrained` (317). Every rank pulls 27.5 GiB of
weights across the FSDP mesh, and only then does `load_sft_dataset` call
`Path(path).read_text()` and raise `FileNotFoundError`. On 8 ranks that is
several minutes of cluster time to learn that a path is misspelt.

**Proposed patch (two parts).**

`configs/sft_14b_full.json` — point at the path the repo actually produces:

```diff
-  "dataset_path": "data/sft/multicap.jsonl",
+  "dataset_path": "data/b05factory/sft/multicap.jsonl",
```

`kore/policy/sft.py` — fail in milliseconds instead of after a 14B load. Insert
directly after the preflight block that ends at line 266, before `import torch`:

```python
    # The dataset is not read until after the model load (~minutes x world_size),
    # so a bad path would otherwise cost a full 14B load on every rank to report.
    if not Path(dataset_path).is_file():
        raise FileNotFoundError(
            f"sft: training dataset not found at {dataset_path!r} (cwd={Path.cwd()}). "
            "Run `cd data/release && ./reassemble.sh` to materialize the packaged "
            "corpus, or point dataset_path at the built shard."
        )
```

Regression test: `test_shipped_config_dataset_path_is_what_reassemble_produces`.

---

### Blocker 2 — `save_total_limit` is hardcoded to 1 and cannot be configured

**Severity: durability.** `kore/policy/sft.py:391`:

```python
save_total_limit=1,   # a 14B full-FT ckpt is ~220GB w/ optimizer; cap to avoid disk-fill
```

The comment's size estimate is right — **measured 221 GB** for a real 14B
checkpoint on this box. The retention choice is the problem, and this repo has
already written down why. `kore/policy/configs.py:593-599`, on `MidTrainConfig`:

> Note that 1 is NOT crash-safe on its own: the Trainer rotates the previous
> checkpoint out around the new save, so a crash inside that window leaves
> nothing for `latest_checkpoint()` to find and the run silently restarts from
> step 0. **Every shipped launch config therefore sets >= 2 explicitly**, buying
> one resumable generation for ~220GB of disk.

`configs/midtrain_14b_full.json` sets `save_total_limit: 2` and carries a comment
justifying it. SFT cannot follow that rule, because the value is not read from
the config at all — and `SFTConfig` has no `save_total_limit` field, so putting
the key in an SFT launch JSON does not get ignored, it *crashes the parse*:

```
SFTConfig.__init__() got an unexpected keyword argument 'save_total_limit'
```

So the knob is simultaneously ignored and unrepresentable, and SFT is the one
14B full-FT stage that violates the repo's own stated retention policy.

**Proposed patch.** `kore/policy/configs.py`, in `SFTConfig` (next to `save_steps`):

```python
    # >= 2, matching MidTrainConfig: the Trainer rotates the previous checkpoint
    # out around each new save, so 1 leaves nothing resumable if the process dies
    # inside that window. Measured on gfx950: one 14B SFT checkpoint is 221 GB.
    save_total_limit: int = 2
```

`kore/policy/sft.py:391`:

```diff
-        save_total_limit=1,   # a 14B full-FT ckpt is ~220GB w/ optimizer; cap to avoid disk-fill
+        save_total_limit=getattr(config, "save_total_limit", 2),
```

and add `"save_total_limit": 2` to `configs/sft_14b_full.json`.

**Disk consequence, measured.** With the limit at 2, the transient peak during
rotation is three checkpoints at once, `3 x 221 GB ≈ 663 GB`, settling to
`2 x 221 GB + 55 GiB ≈ 497 GB` once the run finishes. That is real money, but the
current setting is worse than it looks: even at limit 1 the old and new
checkpoints coexist during the write (observed directly — `checkpoint-3` at
221 GB alongside a partially written `checkpoint-4` at 111 GB), so limit 1
already pays a 442 GB transient peak and buys *no* resumable spare for it.
**Provision ≥ 800 GB on the SFT output filesystem.**

Regression tests: `test_sft_launch_config_can_request_more_than_one_checkpoint`
(xfail), `test_sft_save_total_limit_is_currently_hardcoded`.

---

### Blocker 3 — `latest_checkpoint` gives up instead of falling back

**Severity: durability.** `kore/policy/configs.py:769-773`:

```python
    latest = max(ckpts, key=_step)
    # Only resume if the checkpoint actually has trainer state (not a half-written dir).
    if os.path.exists(os.path.join(latest, "trainer_state.json")):
        return latest
    return None
```

The guard is right; the control flow is not. It inspects only the
highest-numbered directory. `trainer_state.json` is written near the *end* of
`_save_checkpoint`, so a crash during a 221 GB save leaves `checkpoint-N` without
it — and this function returns `None`, restarting from step 0, even when a
complete `checkpoint-N-200` is sitting in the same directory.

This is precisely the failure mode the `MidTrainConfig` comment above warns
about, and it means Blocker 2's spare checkpoint would not actually help until
this is fixed too. **Blockers 2 and 3 must be fixed together** or neither buys
anything.

**Proposed patch:**

```diff
-    latest = max(ckpts, key=_step)
-    # Only resume if the checkpoint actually has trainer state (not a half-written dir).
-    if os.path.exists(os.path.join(latest, "trainer_state.json")):
-        return latest
-    return None
+    # Walk newest-first: a crash during a multi-hundred-GB save leaves the newest
+    # dir without trainer_state.json, and an older COMPLETE checkpoint is still a
+    # far better resume point than restarting from step 0.
+    for candidate in sorted(ckpts, key=_step, reverse=True):
+        if _step(candidate) < 0:
+            continue
+        if os.path.exists(os.path.join(candidate, "trainer_state.json")):
+            return candidate
+    return None
```

Regression test: `test_latest_checkpoint_falls_back_past_a_half_written_checkpoint` (xfail).

---

## Evidence, item by item

### 1. The SFT data exists and is loadable — PASS

Reassembled to `/tmp` (nothing under `data/b05factory/` was written):

```
cat data/release/sft/multicap.jsonl.gz.part{00,01} | gunzip > /tmp/.../multicap.jsonl
```

| | |
|---|---|
| Packaged parts | 94,371,840 + 53,304,569 B = 140.8 MiB |
| Reassembled | 630,488,937 B (601.3 MiB) |
| **Rows** | **56,493 — exactly the documented count** |
| Malformed / unparseable | 0 |
| Rows with no assistant turn | 0 (TRL raises on these; there are none) |
| Non-string or empty message content | 0 |
| Messages | 169,370 — user 60,304, assistant 66,967, system 32,515, tool 9,584 |
| Tokens (Qwen3-14B tokenizer) | 191,477,221 (DATASET_STATUS.md says 190.4M — 0.6% apart) |

Mixture by `_source`: `kernel_repair_opt` 19,630 · `kernel_qa` 9,965 ·
`general_chat` 7,984 · `agentic_tooluse` 6,917 · `general_code` 5,999 ·
`math_reasoning` 5,998. Provenance kinds: `repair` 15,083 · `win` 4,547 ·
untagged 36,863.

Two shape notes, neither harmful:

- 2,920 rows end on a `tool` turn rather than an assistant turn. They all contain
  earlier assistant turns, so the mask is non-empty and TRL is satisfied; the
  trailing tool output is simply masked context that trains nothing.
- `repair_loss_weight: 2.0` duplicates the 15,083 `_provenance.kind == "repair"`
  rows, so the trainer sees **71,576** rows, not 56,493.

### 2. The loss-masking template surgery still works — PASS

Run against the real tokenizer for `Qwen/Qwen3-14B` at revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`, `HF_HUB_OFFLINE=1`. This is the
strongest code in the SFT path and it is still fully intact.

`build_assistant_masked_template` turns the 4,168-char base template into a
4,143-char tagged one, is idempotent, and raises `ValueError` on a non-Qwen3
template. `_verify_assistant_masking` passes on all four conversation shapes.

Independent check on **real token ids**, decoding the surviving span:

| case | tokens | in loss | masked to −100 | what the model actually trains on |
|---|---|---|---|---|
| single-turn | 22 | 9 | 13 | `<think>\n\n</think>\n\nKERNEL_BODY_A<\|im_end\|>\n` |
| multi-turn + system | 39 | 12 | 27 | `RESP_ONE<\|im_end\|>\n<think>\n\n</think>\n\nRESP_TWO<\|im_end\|>\n` |
| `<think>` | 22 | 12 | 10 | `<think>\nreasoning_here\n</think>\n\nFINAL_ANS<\|im_end\|>\n` |
| tool loop | 41 | 16 | 25 | `TOOL_PREAMBLE<\|im_end\|>\n<think>\n\n</think>\n\nTOOL_FINAL<\|im_end\|>\n` |

In every case: the masked template renders **byte-identically** to the base; the
`<|im_start|>assistant` header is masked; `<|im_end|>` (151645) is *in* the loss;
and no system, user, or tool content appears in the unmasked span.

On one real row from each of the six sources, rendering is byte-identical, no
non-assistant content leaks, and every assistant turn is covered:

| `_source` | tokens | in loss |
|---|---|---|
| `agentic_tooluse` (4 tool round-trips) | 12,796 | 8,786 (68.7%) |
| `general_chat` | 591 | 539 (91.2%) |
| `general_code` | 301 | 127 (42.2%) |
| `kernel_qa` | 1,133 | 924 (81.6%) |
| `kernel_repair_opt` | 4,411 | 1,937 (43.9%) |
| `math_reasoning` | 16,805 | 16,718 (99.5%) |

### 3. Model identity resolution — PASS

`configs/sft_14b_full.json` carries
`"model_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18"` (added today) and
it resolves offline against
`~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069.../`,
reporting `parameter_count = 14,768,307,200` and
`load_kwargs = {"revision": "40c0698..."}`, which `sft.py` splats into both
`from_pretrained` calls.

Rejection behaviour is correct in both directions:

| configured revision | development | production |
|---|---|---|
| `main`, `v1.0`, `refs/pr/1`, `40c0698`, 41-hex | `FloatingRevisionError` | `FloatingRevisionError` |
| absent / `MEASURE` | unpinned + warning note | `UnpinnedModelError` |
| well-formed but uncached (`0`×40) | degrades to unpinned under `HF_HUB_OFFLINE=1` | `ModelSpecError` |

**Fingerprint cost.** Development (the default) uses the header-only metadata
tier: **2 ms**. Production escalates to the fingerprint tier, which SHA-256s all
8 safetensors shards plus config/tokenizer/generation files:

- `resolve` → **14.8 s** (27.51 GiB at 1.86 GiB/s), yielding
  `profile_hash = f883de54949469d7e2939c93bb3d5d17ea8cb2c3783f8f4c89a7250959501238`
- `validate_before_load()` re-hashes the same 27.51 GiB to close the TOCTOU
  window → **+14.7 s**
- **≈ 29.4 s per rank.** Every rank does this independently, so an 8-rank
  production launch reads ~220 GiB from one page cache / one filesystem before
  training starts. Warm cache makes it cheap; a cold NFS-backed cache will not.
  Development mode is unaffected (2 ms).

### 4. SFT can actually start training — PASS

Both runs used the real launcher (`scripts/launch_distributed.sh sft <config>`)
and the real entrypoint (`accelerate launch -m kore.policy.sft`), pinned with
`GPU_IDS=6,7` and `HF_HUB_OFFLINE=1`.

**(a) Tiny Qwen3 — full lifecycle.** A genuine 41.2M-param `Qwen3ForCausalLM`
built offline with the real Qwen3-14B tokenizer, so the same
`Qwen3DecoderLayer` FSDP wrap class and the same chat template are exercised.
30 real multicap rows (all six sources, 6 repair-tagged, 2 deliberately
over-length).

```
model identity resolved  ... parameter_count=41189120
sft: completion-only loss enabled  assistant_only_loss=True
sft: dataset loaded  ... assistant_only_loss=True dropped_overlong=2 repair_weight=2 n_rows=34
17 optimizer steps, loss ~11.9   (= ln(151669), exactly uniform for a random init)
checkpoint-17: model.safetensors optimizer.bin pytorch_model_fsdp.bin scheduler.pt
               rng_state_0.pth rng_state_1.pth trainer_state.json
```

Re-launched with `num_train_epochs: 2`:
`sft: resuming from checkpoint ckpt=.../checkpoint-17` → steps 18…34. Resume works.

The row accounting checks out end to end: 30 rows + 6 repair duplicates = 36,
minus the 2 over-length rows = 34.

**(b) Real Qwen3-14B, 2-way FSDP.** Same launcher, `max_seq_length: 4096`,
`per_device_train_batch_size: 1`.

```
model identity resolved  model_id=Qwen/Qwen3-14B revision=40c0698... revision_pinned_at_load=True
                         parameter_count=14768307200
resource preflight       persistent_state_bytes=236292915200   (220.1 GiB)
sft: completion-only loss enabled

step 1  loss 0.6841  grad_norm 7.688   peak 118.3 GB/GPU
step 2  loss 1.0667  grad_norm 10.41   mean_token_accuracy 0.781   peak 124.5 GB/GPU
step 3  loss 0.8266  grad_norm 5.949
step 4  loss 0.5378  grad_norm 2.769
```

Real pretrained-model loss and 78% token accuracy — the masking is producing a
sane objective on real data, not a degenerate one. Then:

- `checkpoint-3` written: **221 GB**, in **≈ 104 s** (2 ranks)
- final `trainer.save_model`: 13 shards, **55.0 GiB**
- resumed from `checkpoint-4` → steps 5–8, loss 0.365 / 0.847 / 0.609 / 0.533,
  token accuracy 0.865

Every stage of the requested path executed: dataset load → template masking →
trainer construction → real optimizer step → checkpoint write → resume.

### 5. Handoff from midtrain — PASS with defects

- **`latest_checkpoint` finds a written checkpoint.** Confirmed against real
  Trainer output (`checkpoint-34`, `checkpoint-4`) and against synthetic dirs
  with out-of-order step numbers. It has the fallback defect in Blocker 3.
- **A directory source is handled correctly.** Pointing `model_id` at a real
  `trainer.save_model` output while leaving the base model's `model_revision` in
  the config — exactly what `run_campaign.py:_launch_distributed` does, since
  `cfg.update(overrides)` replaces `model_id` but not `model_revision` — yields:

  ```
  revision=None  revision_pinned_at_load=False  load_kwargs={}
  local_path=/.../runs/midtrain_14b_full   parameter_count=...
  NOTE: '...' is a local checkpoint directory, so the configured revision
        40c069824f42... is IGNORED: a directory has no Hub commit ...
  ```

  This is right. The base model's commit is not laundered onto the midtrain
  output, no bogus `revision=` reaches `from_pretrained`, and the architecture is
  still verified from the safetensors headers. In production mode the fingerprint
  tier correctly degrades to metadata (there is no immutable id to bind a hash
  to) and a directory that is *not* a valid checkpoint raises.
- **The launcher dry-run is sane:**

  ```
  $ GPU_IDS=6,7 bash scripts/launch_distributed.sh sft configs/sft_14b_full.json --dry-run
  [launch_distributed] (dry-run) PYTHONPATH=/home/shasriva/Kore-RL/KORE accelerate launch \
    --config_file /home/shasriva/Kore-RL/KORE/configs/accelerate_fsdp.yaml \
    --gpu_ids 6,7 --num_processes 2 -m kore.policy.sft configs/sft_14b_full.json
  ```

  `--num_processes` is derived from `GPU_IDS`, the SFT path selects
  `accelerate_fsdp.yaml` (FULL_SHARD) rather than the GRPO variant, and
  `build_fsdp_kwargs` independently produces `transformer_layer_cls_to_wrap:
  ["Qwen3DecoderLayer"]` with `state_dict_type: FULL_STATE_DICT`, so the SFT
  output is a plain HF checkpoint that DPO can load.

### 6. Memory and time — PASS with defects

**Step count.** After the repair up-sampling and the over-length filter the
trainer sees 68,277 rows per epoch:

```
56,493 rows + 15,083 repair duplicates          =  71,576
       − 3,299 rows over max_seq_length 16384   =  68,277 rows/epoch (184.3M tokens)

effective batch = 2 x 8 x 8 ranks               =  128 sequences/step
optimizer steps per epoch                       =  533
TOTAL (3 epochs)                                =  1,599 steps
```

**Wall time.** Midtrain's measured 33.5 s/step at `4 x 4 x 8` micro-batches of
8192 tokens is 1,048,576 tokens/step, i.e. **31,301 tok/s across 8 GPUs**. SFT
trains 553.0M tokens over 3 epochs:

| assumption | wall time | implied s/step |
|---|---|---|
| token parity, no padding waste | 4.9 h | 11.0 |
| +15% padding (`group_by_length` is on) | **5.6 h** | 12.7 |
| +30% padding | 6.4 h | 14.4 |

Call it **5–7 hours**, plus ~15 min of checkpoint I/O (8 × ~110 s) and ~4 min of
single-threaded startup tokenization. Treat this as a token-throughput
extrapolation, not a measurement: midtrain packs uniform 8192-token chunks
whereas SFT pads to the longest row in each length-grouped batch, and its
16k-token rows carry a worse quadratic attention term.

**Memory.** Analytical persistent state for 14,768,307,200 params is
236,292,915,200 B (220.1 GiB) — bf16 weights + bf16 grads + fp32 master + two
fp32 Adam moments — which the preflight reports and which matches the code's
arithmetic exactly. Measured at 2-way sharding: **124.5 GB/GPU** at seq 4096,
micro-batch 1 (118.1 GB of that is sharded state, ~6.4 GB activations and
workspace). At 8-way the sharded state drops to 27.5 GiB/GPU; scaling the
activation term to seq 16384 × micro-batch 2 puts the worst-case batch near
90–100 GB/GPU against 252 GiB of HBM. **Wide margin, and 2-GPU 14B full-FT is
itself feasible** if the cluster is contended.

Host side: `load_sft_dataset` reads the whole 630 MB file with
`Path(path).read_text().splitlines()`, peaking at **8.1 GB RSS per rank** —
≈ 65 GB across 8 ranks, on top of the FULL_STATE_DICT gather. Fine on a 3 TB
host; worth knowing given that a previous 14B midtrain died of host-memory
exhaustion at step 492.

**`save_total_limit` — see Blocker 2.** With `save_steps` at its default 200 and
1,599 total steps, the run writes 7 periodic checkpoints plus one
end-of-training checkpoint, each 221 GB, each rotating the previous one out.

---

## Non-blocking findings

**F1 — the SFT output checkpoint is fp32, not bf16 (2× disk, 2× handoff cost).**
Measured on the real 14B: `model.safetensors.index.json` reports
`total_size: 59,073,228,800` (**55.0 GiB in 13 shards**, all `F32`) and
`config.json` records `"dtype": "float32"`. The base is 27.5 GiB of BF16 in 8
shards. This is inherent to the configuration rather than a bug: accelerate's
`mixed_precision: bf16` upcasts every trainable FSDP flat-parameter to an fp32
master, and `FULL_STATE_DICT` gathers that master. Numerically it is fine — every
downstream stage loads with an explicit `torch_dtype=torch.bfloat16` — but it
doubles the size of every inter-stage artifact and doubles rank-0 load time and
host RAM at each handoff. **The incoming midtrain checkpoint will be 55 GiB for
the same reason.** If that matters, cast to bf16 before `save_pretrained` in each
stage; if not, it should at least be written down, because "the 14B checkpoint is
28 GB" is wrong by 2×.

**F2 — the 16,384 limit deletes 53.6% of the math capability slice, by ~420
tokens.** `_filter_overlong` drops 3,299 of 71,576 rows (4.6%), and 3,214 of them
are `math_reasoning`. That is **53.58% of the entire math slice**, versus 0.43%
of `kernel_repair_opt` and 0% of every other source. The mixture is designed for
`frac_math_reasoning = 0.10`; after filtering, math contributes about half what
the recipe intends.

The distribution makes this cheap to fix. Bucketing all 5,998 math rows:

| length | rows | share |
|---|---|---|
| ≤ 16,384 (kept) | 2,784 | 46.4% |
| 16,385 – 17,408 | **3,214** | **53.6%** |
| 17,409 – 20,480 | 0 | 0% |
| > 20,480 | 0 | 0% |

Every dropped row lands in a single 1,024-token band immediately above the cut
(corpus p95 = 16,804 and p99 = 16,805 — the generator clearly capped its CoTs at
a ~16.8k budget that sits just above `max_seq_length`). Raising
`max_seq_length` to **17,408** recovers all 3,214 rows. The honest trade-off:
that adds ~54M tokens per epoch (+29%), taking the run from ~5.6 h to ~7.2 h, and
raises the worst-case sequence by 6% (negligible against the memory margin above).
Recovering half a capability slice for 1.6 h looks like the right trade, but it
is a recipe decision, not a bug fix — flagging it rather than proposing a patch.

**F3 — stale comment.** `kore/policy/sft.py:344` says the filter "drops ~8.7% of
rows"; the measured figure on this corpus is **5.84%** of base rows (4.6% of the
up-sampled set). Same claim, same line, also says "~0.6% of kernels" — measured
0.43%.

**F4 — ~4 minutes of single-threaded startup tokenization per rank.**
`_filter_overlong` renders all 71,576 rows through the chat template in a plain
Python loop with no `num_proc`, measured at ~311 rows/s → **≈ 230 s**, then
`_token_stats` re-renders 512 more. Every rank does the identical work
concurrently (which is what keeps the FSDP shards consistent), so wall-clock cost
is ~4 min, not 32. Well inside the 1800 s distributed timeout, but it is dead
time on 8 GPUs and it would be a straightforward `datasets.map(num_proc=...)`.

**F5 — `--dry-run` validates nothing.** `scripts/launch_distributed.sh:95` exits
before the `-f` checks at lines 100–107, so
`launch_distributed.sh sft /nonexistent/nope.json --dry-run` prints a command and
exits 0. The CI syntax check therefore cannot catch a mistyped config path.
Patch: move the `[ ! -f "$ACCEL_CONFIG" ]` and `[ ! -f "$CONFIG" ]` checks above
the dry-run block. Regression test:
`test_launcher_dry_run_rejects_a_missing_config` (xfail).

**F6 — `GPU_IDS` are HIP ordinals, and they do not match `rocm-smi` indices on
this host.** Verified by allocation probe and PCI BDF join:

| HIP / `CUDA_VISIBLE_DEVICES` | PCI bus | `rocm-smi` shows it as |
|---|---|---|
| 0 | 0x78 | GPU[3] |
| 1 | 0x08 | GPU[0] |
| 2 | 0x68 | GPU[2] |
| 3 | 0x18 | GPU[1] |
| 4 | 0xf9 | GPU[7] |
| 5 | 0x88 | GPU[4] |
| 6 | 0xe8 | GPU[6] |
| 7 | 0x98 | GPU[5] |

Only ordinals 2 and 6 coincide. Anyone who reads `rocm-smi`, sees GPUs 6 and 7
idle, and passes `GPU_IDS=6,7` gets HIP 6 and 7 — which are `rocm-smi` 6 and
**5**. This is exactly why `kore/policy/resources.py` refuses to infer ordinals
from discovery order and joins DRM to HIP by PCI BDF; the caution is justified.
(The runs in this report used HIP ordinals 6 and 7 as instructed — confirmed by
watching `rocm-smi` GPU[6] and GPU[5] carry the load.)

**F7 — the launcher invokes bare `accelerate`.** `exec accelerate ...` with no
interpreter path fails with `accelerate: not found` unless the venv is on
`PATH`. Activate the venv or export `PATH` before launching.

**F8 — the output directory keeps an extra checkpoint.** The Trainer writes an
end-of-training `checkpoint-<max_steps>` *in addition to* the final
`trainer.save_model()` output (observed in all four runs). The SFT output dir
therefore ends at 221 GB + 55 GiB ≈ 276 GB even after a clean finish.

**F9 — resource preflight reports `unresolved`, by design.** It cannot join DRM
cards to HIP ordinals without `KORE_HIP_INVENTORY_JSON`, so it logs
`status=unresolved ... visible_gpus=0` and warns. In the default `report` mode
this never raises and the run proceeds; it is only fatal under
`KORE_RESOURCE_PREFLIGHT=strict`. **Do not set `strict` for this launch** — no
measured peak profile exists for this workload, so strict mode would refuse to
start.

---

## Launch command

Once midtrain lands, from the repo root, with the venv active and
`HF_HUB_OFFLINE=1`:

```bash
cd /home/shasriva/Kore-RL/KORE
export PATH=/home/shasriva/kore-venv/bin:$PATH     # the launcher execs bare `accelerate` (F7)
export HF_HUB_OFFLINE=1

# 1. Materialize the packaged corpus once (writes data/b05factory/sft/multicap.jsonl).
(cd data/release && ./reassemble.sh)
wc -l data/b05factory/sft/multicap.jsonl          # must print 56493

# 2. Resolve the launch config: real dataset path + the midtrain checkpoint as the
#    SFT base. Drop the dataset_path override once Blocker 1's patch has landed;
#    model_id must always be overridden (the shipped config names the raw base).
python - <<'PY'
import json, pathlib
cfg = json.loads(pathlib.Path("configs/sft_14b_full.json").read_text())
cfg["dataset_path"] = "data/b05factory/sft/multicap.jsonl"   # Blocker 1
cfg["model_id"]     = "runs/midtrain_14b_full"               # Stage-0 output
cfg["output_dir"]   = "runs/sft_14b_full"
pathlib.Path("configs/sft_14b_full.resolved.json").write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg, indent=2))
PY

# 3. Sanity-check the handoff before spending an 8-rank load.
python -c "
from kore.policy.configs import latest_checkpoint
print('midtrain checkpoint:', latest_checkpoint('runs/midtrain_14b_full'))"
ls runs/midtrain_14b_full/config.json runs/midtrain_14b_full/*.safetensors >/dev/null

# 4. Dry-run, then launch on all 8 GPUs.
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/launch_distributed.sh sft \
    configs/sft_14b_full.resolved.json --dry-run

GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/launch_distributed.sh sft \
    configs/sft_14b_full.resolved.json 2>&1 | tee logs/sft_14b_full.log
```

Expect: `model identity resolved ... revision_pinned_at_load=False` (correct —
the base is now a directory), `sft: completion-only loss enabled`,
`dropped_overlong=3299`, then 1,599 steps over 5–7 hours.

Do **not** set `KORE_RESOURCE_PREFLIGHT=strict` (F9). Ensure the output
filesystem has **≥ 800 GB** free (Blocker 2). If the run dies, check that
`latest_checkpoint('runs/sft_14b_full')` returns a directory before relaunching —
until Blocker 3 is fixed, a `None` there means it will silently restart from step
0 even if a complete older checkpoint exists.

---

## Reproducing this report

```bash
pytest tests/test_sft_launch_readiness.py -q            # 30 pass, 4 xfail (the blockers)
pytest tests/test_sft_launch_readiness.py -q -m release # full 56,493-row corpus count
```

The GPU portions (items 4 and 6) are not in the test suite: they need two idle
MI350X and ~500 GB of scratch. The exact configs used are recorded above.
