# KORE production data specification

KORE trains an AMD gfx950 (MI355X, CDNA4) kernel improver, not a text-only code
generator. A record is useful only when its correctness claim comes from
execution against a task's oracle, not from a model's opinion of its own output.

## Scope

The product recipe is instruct → SFT → multi-turn RL for
`Qwen/Qwen3-Coder-30B-A3B-Instruct`. There is no production continued-pretraining
(CPT) or DPO stage: no candidate 30B-class Qwen ships a Base sibling, so neither
CPT nor a chat-vector residual merge can reach production, and the recorded 14B
CPT experiment on an instruct checkpoint destroyed instruction-following before
either question could matter. `docs/SOURCE_PROVENANCE.md` covers that 14B corpus
as a closed historical artifact.

This document specifies **v5**, the dataset the SFT config
(`configs/sft_coder30b_a3b.json`) actually points at. `DATAGEN_OVERVIEW.md`
explains how it is built; `docs/SOURCE_PROVENANCE.md` explains where its content
comes from and how it is screened. This document specifies what the shipped
files contain, the row contract, and the gates every row must clear.

## The v5 files

| File | Rows | Tokens | Role |
| --- | ---: | ---: | --- |
| `data/v5_sft.jsonl` | 206,000 | 490,174,073 | training |
| `data/v5_eval.jsonl` | 899 | n/a | held-out, per-capability retention check |

Mean row length is 2,379 tokens; the model's `max_seq_length` (17,408) and the
build-time token gate (16,896, leaving headroom for a later template change) are
both sized against that distribution, not against the mean.

Both files are gitignored (they are derived artifacts, not source) and shipped
as gzip parts under `data/release/sft/`: `v5_sft.jsonl.gz.partaa` through
`.partad`, and `v5_eval.jsonl.gz`. `data/release/reassemble.sh` concatenates and
decompresses them back to the two files above, byte-for-byte; `scripts/v5_verify_all.sh`
is the check that the shipped parts actually reproduce them and that both halves
still pass every gate below.

Of the 206,000 training rows, 165,047 carry a distinct assistant target across
11,793 distinct tasks; the remainder are the deliberate, capped repetitions of a
scarce shape or dialect described below, not accidental duplication.

## Row contract

Every row is `{"messages": [...], "_source": ..., "_task_id": ..., "_shape": ...,
"_dialect": ..., "_provenance": {...}, "_tokens": <int>}`. A kernel row's
`messages` end on exactly one assistant turn: the trainer puts full loss on
every assistant turn it sees with no per-turn opt-out, so a second assistant turn
would train on a rejected kernel revision. A general-replay row (chat, code,
math, instruction-following, tool-use) is exempt from that rule: multi-turn
ability is exactly the capability those rows exist to preserve.

`_provenance` is always an object, never a bare string: the trainer calls
`.get("kind")` on it whenever `repair_loss_weight >= 1.5`, and a string there
raises after the model has already loaded on every rank. `_shape` and
`_dialect` classify a kernel row (below); replay rows carry neither. `_tokens`
is the real chat-template token count measured with the pinned tokenizer
(`Qwen/Qwen3-Coder-30B-A3B-Instruct`, revision `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`),
not a character estimate: the v3-era build gated on characters/3.6, and 209 of
its rows still exceeded the true cap and were silently dropped at train time.

## Composition

**61.2% of rows are kernel, 38.8% are general replay.** Weighted by tokens
instead (the unit the optimizer actually sees, and the unit every mixture
result in the literature is stated in), replay is **12.0% of tokens**. That is
below the 25-30% plateau the forgetting-mitigation literature reports, and it is
the single largest known deviation from best practice in this build. The build
targets 14% of tokens (`scripts/v5_stage4_mixture.py --replay-target`); the
realized figure is lower because `scripts/v5_fix_truncated.py` removed 9.3M
tokens of truncated math chain-of-thought from the replay side after the target
was set (see "Truncated targets" below), and that loss was never made up. If
retained-capability loss climbs during training, the fix is more replay, not a
lower learning rate.

**Six task shapes, not v4's one.** v4 (the SFT mixture this cycle replaces)
was 69,851 rows and 288.4M tokens, all of it one shape: "here is a kernel,
improve or fix it." It scored 55.1% on AgentKernelArena against 55.9% for the
base model it was fine-tuned from, i.e. SFT made the model worse. The diagnosed
cause was shape monoculture, not data quality: the benchmark asks five distinct
questions and v4 had zero training examples of three of them. v5 answers six:

| Shape | Question | Source |
| --- | --- | --- |
| `optimize` | given a working kernel, make it faster | wins, gold wins, step-centric revisions |
| `repair` | given a broken kernel and its verifier error, fix it | repair records |
| `torch2kernel` (PyTorch-to-kernel) | read a PyTorch module, write the kernel | verified twins, re-posed |
| `port` (dialect port) | re-express a verified Triton kernel in another dialect | verified twins, re-posed |
| `instruction` (spec-to-kernel) | write a kernel from a written spec, no source given | verified twins, re-posed |
| `language` | the dialect itself: idioms, layouts, API surface | FlyDSL ecosystem anchor, kernel QA |

**Dialects, by kernel row:** Triton 61.2%, HIP 32.3%, FlyDSL 6.5%. FlyDSL is
deliberately over-represented relative to its natural yield: the public and
internal supply of FlyDSL code was expanded from 130 to 1,582 distinct kernels
across 21 repositories specifically to anchor this share (`docs/SOURCE_PROVENANCE.md`),
and the final mixture still caps FlyDSL repetition at 4x its distinct-row count
so the anchor does not become memorization.

## Task registry and pool

The `torch2kernel`, `port`, and `instruction` shapes are re-posed questions over
a task that was already verified for some other purpose (below); they do not
exist independently of the task registry and mined pool that back the rest of
the project. The registry holds 1,546 tasks, of which 1,052 are `genb_`
generated breadth tasks (elementwise/reduction ops mechanically generated across
dtypes) and the rest are hand-authored or curated. 110 of the 1,546 declare or
resolve to a production vendor baseline: 63 declare AITER, 4 declare hipBLASLt,
and runtime resolution adds 35 more `gemm_fusion` tasks against hipBLASLt and 8
gated-activation tasks against AITER; every other task, including all 1,052
generated breadth tasks, is graded against torch. A twin verified against a
torch baseline is still a correct kernel; it is not evidence of beating a
production library, and the shape tables above do not distinguish the two.

Beyond the registry, `kore/data/task_mining.py` admits PyTorch modules mined
from `GPUMODE/KernelBook` and a synthetic operator-composition pipeline through
five ordered gates (safety, classification, decontamination, execution,
dedup; see `docs/SOURCE_PROVENANCE.md`). 13,570 pool tasks currently clear that
pipeline and are eligible for translation and mining; this is also the pool
size the AgentKernelArena contamination index screens against (below).

## Held-out evaluation split

`data/v5_eval.jsonl` is 899 rows carved out of the built mixture by
`scripts/v5_split_eval.py`, stratified into eight capability groups so a
regression in one capability cannot hide inside an aggregate loss:

| Group | Rows | What it measures |
| --- | ---: | --- |
| `kernel_generate` | 239 | torch2kernel, port, and instruction shapes |
| `kernel_repair` | 80 | repair shape |
| `kernel_optimize` | 80 | wins, gold wins, step-centric |
| `instruction_following` | 120 | the run's top stated risk |
| `chat` | 120 | general conversational retention |
| `general_code` | 120 | general coding retention |
| `tool_use` | 80 | agentic / function-calling retention |
| `math` | 60 | math reasoning retention |

Two properties make the split trustworthy rather than cosmetic:

- **Held out by content, not by line.** The mixture upsamples scarce shapes and
  dialects, so a line-number split leaves duplicates behind: a first attempt at
  the split left 296 of 900 held-out rows with a surviving copy in training.
  Rows are hashed by their full message list and every matching hash is removed
  from both files, so the eval slice is never accidentally re-trained on.
  Verified: zero message-hash overlap between `data/v5_sft.jsonl` and
  `data/v5_eval.jsonl`.
- **Truncated eval rows are backfilled, not just dropped.** 32 of the math
  group's rows were themselves truncated (below); dropping them would leave that
  group at roughly half its intended size, too few for a stable per-group loss.
  They are replaced with clean rows drawn from the training pool, which are then
  removed from training so the two files stay disjoint.

## Truncated targets

The build's length gate asks whether a row is *under* the token cap, which a
truncated row passes by construction: truncation is how it got under the cap.
554 rows shipped with their assistant target cut off mid-generation (552 of them
in `math_reasoning`, e.g. `"...then g(x_6)= (g(x"`), because the upstream teacher
call hit its output-token ceiling and the row was written anyway. As SFT targets
these are worse than useless: they teach that a valid way to finish a hard
problem is to emit sixteen thousand tokens and stop mid-token with no end-of-turn
marker, and because they were the longest rows in the corpus they carried a
disproportionate share of the tokens assistant-only loss actually trains on.

`scripts/v5_fix_truncated.py` detects a row whose final assistant turn is long
enough to have plausibly hit the cap and does not end on sentence or code
punctuation, and drops it. The control that confirms this is a real signature
rather than a loose heuristic: 94% of capped `math_reasoning` rows lack terminal
punctuation, against 3% of short ones. Net effect on the training file: 586 rows
removed (554 dropped as truncated, plus 32 promoted into the eval math group to
replace its own truncated rows), accounting for the last step from 207,782 built
rows down to the shipped 206,000. `scripts/v5_verify.py` now runs a
`truncated_target` gate on every build so this class of defect is caught before
release rather than after.

## Correctness gates

`scripts/v5_verify.py` runs twelve correctness gates over the final file, plus a
`transformers` token-length pass. Eleven predate the truncation incident above;
the twelfth (`truncated_target`) was added because none of the original eleven
could catch a row that was under the token cap *because* it was cut off.

| Gate | What it catches |
| --- | --- |
| `heldout_probe_leak` | a near-generalization probe task reached training |
| `contaminated_task_leak` | a task already flagged contaminated reached training |
| `arena_contamination` | a pool task matches an AgentKernelArena source above threshold |
| `provenance_not_object` | `_provenance` is a bare string, not an object |
| `kernel_assistant_turns_not_1` | a kernel row has more than one assistant turn |
| `no_assistant_turn` | a row has no assistant turn at all |
| `bad_role` | a message role outside `system`/`user`/`assistant`/`tool` |
| `missing_content` | a message has no `content` key |
| `empty_content` | a message's content is blank |
| `last_not_assistant` | the row does not end on an assistant turn |
| `no_messages` | a row has an empty `messages` list |
| `truncated_target` | the assistant target looks cut off mid-generation |

Both `data/v5_sft.jsonl` and `data/v5_eval.jsonl` pass all twelve
(`VERDICT: PASS`); the eval half is checked with the same battery as training,
because a contaminated or degenerate eval row corrupts the retention signal the
run depends on exactly as surely as a bad training row corrupts the weights.
`scripts/v5_verify_all.sh` runs the round-trip, the disjointness check, and both
verification passes together as the pre-launch gate.

## What is deliberately absent

DPO pairs are not a production source for v5. Kernel quality is observed
directly by compile, oracle, and timing; converting that evidence to a static
preference loses the feedback that the multi-turn RL stage can use directly.
Nor is there a v5 midtrain corpus: no chosen 30B Qwen has the required Base
model, for the reason stated in Scope.

`kernel_qa` (natural-language GPU/ROCm reasoning that does not ask the model to
emit a kernel) is folded into the `language` shape rather than counted as
general replay. Counting it as replay in an earlier cycle made the general-half
share look larger than the genuinely general (chat/code/math/instruction/tool-use)
content actually was.
