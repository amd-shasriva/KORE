# Third-party material and attribution

KORE is proprietary and AMD-internal (see [`LICENSE`](LICENSE)). It nevertheless incorporates,
derives from, or measures against third-party material. This file records what that material is,
under what terms it was obtained, which shipped artifact it flows into, and — importantly —
**which items remain unresolved**.

Machine-readable provenance lives in `data/release/meta/source_metadata.json` (schema `1.0`,
28 source entries and 2 dataset entries). **That artifact is currently wrong in three places**
(§0.1, §0.2, §0.3); this document is authoritative until it is regenerated.

Every licence below was read from the upstream `LICENSE` file **at the exact pinned commit or
dataset revision that KORE consumed**, not from the project's current `main`. That distinction is
load-bearing: two of the findings in §0 exist only because the pinned revision differs from what a
casual reading of the project would suggest. Method and evidence are in §9.

---

# ⚠️ SECTION 0 — BLOCKING FINDINGS. DO NOT SHIP WITHOUT READING THIS.

Three sources are recorded with a licence they are not under. Two of them carry terms that
**directly constrain proprietary redistribution of a model trained on this data**; the third
(§0.3) is AMD's own material and is a labelling error rather than an exposure. Neither of the two
was visible to the build, because the dataset licences in `source_metadata.json` were hard-coded by
hand and `kore/data/midtrain_corpus.py::_license_from_root` only ever reads the licence file at a
repository *root* — never a per-directory licence or a per-file SPDX header.

## 0.1 — BLOCKER: KernelBook and kernelbot-data are NOT MIT. They carry a use-based restriction on training AI models.

`GPUMODE/KernelBook` and `GPUMODE/kernelbot-data` are recorded as **MIT** in
`source_metadata.json`, in every row of the shipped corpora, and in the previous version of this
file. **Both are actually under the "June 9 Researcher Reciprocity License, Version 1.0"** — a
bespoke licence adapting the Open RAIL-D pattern. It is not an OSI-approved open-source licence
and it is not MIT.

**This is not a case of pinning before a relicense.** The pinned KernelBook revision
`b76504d85f7f14ef4b1fad81f136f638f2ce625b` *is itself* the commit titled
`Add researcher reciprocity dataset license` (2026-06-09T20:06:03Z). The restrictive licence was in
force at the exact revision KORE consumed. `kernelbot-data` at `4159cf6b…` (2026-07-30) carries the
same licence.

**What the licence actually requires.** The grant is broad — reproduce, prepare derivative works,
sublicense, distribute, and explicitly "commercial analysis". It is *not* copyleft and it does not
forbid commercial use. But it defines "Training Use" to cover training, fine-tuning, distillation,
synthetic-data generation and embedding, defines any model so trained as a **"Covered Model"**
(expressly including model weights, checkpoints, adapters, APIs and hosted services), and then
imposes, at §4 and §5:

- **§5** — for Training Use, the Attachment A use restriction **"must be included as an enforceable
  provision in any legal agreement, terms of use, acceptable use policy, license, or other terms
  governing the use or Distribution of a Covered Model"**, and downstream users must be given
  notice that the model is subject to Attachment A.
- **Attachment A** — a covered provider may not impose terms that prevent GPU Mode, dataset
  contributors, or authorized researchers from generating outputs, evaluating the model,
  benchmarking it, publishing research, or exploring their own research ideas **on materially equal
  terms to ordinary users**, and may not retaliate against them for doing so.
- **§4.1–§4.4** — pass on the licence text or a link, retain attribution notices, credit GPU Mode
  and the dataset by name with a link, and mark modified files as changed.

**Affected artifacts — this is the largest single exposure in the corpus:**

| Shipped artifact | Rows | Share |
| --- | --- | --- |
| Midtrain `kernelbook` channel | 21,432 | 24.9% of midtrain |
| Midtrain `amd_kernels` channel (kernelbot-data) | 5,638 | 6.6% of midtrain |
| **Midtrain total** | **27,070** | **31.5% of midtrain** |
| SFT `kernel_qa` Tier-2 curriculum, generated OSS-Instruct-style over KernelBook kernels | 298 | of 9,965 curriculum rows |
| Any checkpoint trained on the above | — | the whole model is a "Covered Model" |

**Why it matters here specifically.** KORE's own [`LICENSE`](LICENSE) currently forbids external
"benchmarks" and "public claims" about results. That restriction applies to AMD personnel and not
to third parties, so there is no conflict today while the project is internal-only. But if a
Covered Model is ever offered externally, the terms of that offering must carry Attachment A and
must not restrict research access, benchmarking, or publication by the parties Attachment A
protects. Terms of service that forbid benchmarking or publishing evaluations — a common default in
commercial model ToS — would breach it.

**Remediation options, in increasing order of cost:**

1. **Accept and comply.** Propagate Attachment A into the model's terms of use, give the required
   notice, and attribute GPU Mode. Cheapest; requires legal sign-off that AMD is willing to bind
   its outbound terms this way, and it constrains all downstream ToS drafting.
2. **Rebuild without them.** Drop the `kernelbook` and `amd_kernels` channels and retrain. Removes
   31.5% of the midtrain corpus and the single largest breadth source; the Tier-2 curriculum rows
   would also need regenerating from a different seed set.
3. **Re-derive the same breadth from clean sources.** KernelBook's own card states its rows are
   derived from permissively licensed GitHub repositories (MIT, Apache-2.0, BSD, MPL, Unlicense,
   zlib) via The Stack v1, and **each row carries a per-row `licenses` field**. Re-ingesting from
   those upstream repositories directly, or re-running the torch-Inductor capture in-house, yields
   equivalent data without the reciprocity condition. `docs/DATASET_SPEC.md` §5 already specifies
   this local Inductor re-capture path and notes it was never executed.

**Also note:** KORE's loader (`_load_kernelbook_pairs`) reads only `python_code` and `triton_code`
and **discards KernelBook's per-row `licenses`, `repo_name`, `sha` and `repo_link` fields**. The
per-row upstream attribution for all 21,432 rows was available and was thrown away. Option 3 above
depends on recovering it, which is possible — it is still in the pinned upstream revision.

## 0.2 — BLOCKER: AGPL-3.0 source code is in the shipped midtrain corpus, stamped `Apache-2.0`.

`unslothai/unsloth` is recorded as **Apache-2.0**. At the pinned commit
`3b235895bdf08410e0a9032e663e82c0de60a6a4` (2026-07-14) the repository is **dual-licensed**: its
README states "Unsloth uses a dual-licensing model of Apache 2.0 and AGPL-3.0", `pyproject.toml`
declares `license = "Apache-2.0"`, and alongside the Apache-2.0 `LICENSE` the repository ships an
AGPL-3.0 `COPYING` plus three subtree licence files:

- `studio/LICENSE.AGPL-3.0`
- `unsloth/kernels/moe/LICENSE`
- `unsloth/kernels/moe/grouped_gemm/LICENSE`

The ingestion pipeline read only the root `LICENSE`, so the subtree relicensing was invisible to it.

**29 rows of the shipped midtrain corpus are AGPL-3.0-only content carrying `license: "Apache-2.0"`
in their own `source_metadata`.** They are self-identifying: the admitted text itself begins with
`# SPDX-License-Identifier: AGPL-3.0-only` or
`# SPDX-License-Identifier: GNU Affero General Public License v3.0`.

| Subtree | Rows | Channel | Nature |
| --- | --- | --- | --- |
| `studio/**` | 22 | 20 `triton`, 2 `docs` | Unsloth Studio backend: training/inference/export workers, provider registry, tests |
| `unsloth/kernels/moe/**` | 7 | 6 `triton`, 1 `docs` | MoE grouped-GEMM Triton kernels: `forward.py`, `backward.py`, `tuning.py`, `interface.py`, `autotune_cache.py`, `README.md` |
| **Total** | **29** | 26 `triton`, 3 `docs` | 0.034% of the 86,010-row midtrain corpus; the other 16 unsloth rows are genuinely Apache-2.0 |

Note the second group is precisely the kind of file KORE's `triton` channel is built to harvest —
MoE grouped GEMM is a P0/P1 coverage cell in `docs/DATASET_SPEC.md` §1.5. This was not a fluke of
the crawler; it is the crawler working as intended against a subtree it had no licence signal for.

**Why it matters.** AGPL-3.0 is strong copyleft with a network-use clause. Whether training weights
on AGPL source creates a derivative work of that source is legally unsettled and is not a question
this document can answer. What is certain is that AGPL-3.0 content is present, that it is
mislabelled, and that a proprietary product built on it without review carries avoidable risk.

**Remediation.** Cheap and unambiguous: 29 rows out of 86,010. Filter every row whose
`source_metadata.path` begins `studio/` or `unsloth/kernels/moe/` out of the corpus and rebuild the
Stage-0 artifact. There is no meaningful data loss. Do this before, not after, the next training
run. A corpus-side guard belongs in the ingestion path so the same class of subtree relicensing
cannot recur — the check must read per-directory licence files and per-file SPDX headers, not just
the repository root.

**No other copyleft or non-commercial source is present.** A full scan of all 86,010 midtrain rows
and both the SFT and DPO artifacts for AGPL / GPL / LGPL / MPL / CC-BY-NC / CC-BY-SA / research-only
markers returns unsloth and nothing else (§9).

## 0.3 — The catalog records this repository itself as MIT, contradicting `LICENSE`.

`source_metadata.json` records `amd-shasriva/KORE.git` with `"license": "MIT"`, and that value is
stamped into **2,323 midtrain rows** (1,784 `pytorch_triton_pairs` + 539 `kore_tasks`). KORE is
proprietary and AMD-internal. This is AMD's own material, so nothing is being infringed and there
is no third-party exposure — but it is a false licence statement inside a shipped artifact, it
would be read as an outbound MIT grant by anyone auditing the corpus, and it must be corrected to
`LicenseRef-AMD-Proprietary-Internal` when the catalog is regenerated.

---

## 1. Base model — RESOLVED

| Artifact | Licence (SPDX) | Upstream URL | Pinned revision | Used for / flows into |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-14B` | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-14B | 40c069824f4251a91eefaf281ebe4c544efd3e18 | Stage-0/1/2 base weights and tokenizer; every derived checkpoint |

Verified by reading `LICENSE` at that exact revision: it is the unmodified Apache License 2.0. The
model card at that revision declares `license: apache-2.0`. There is **no** `NOTICE` file, and Qwen3
carries **no** Qwen-specific community licence, naming clause, or user-count threshold — unlike
Llama-family models and unlike some earlier Qwen releases that used a bespoke research licence.

**This permits what KORE does**: training derivative models, and proprietary deployment of the
result, with no copyleft and no non-commercial restriction. The obligations are Apache-2.0 §4 and
they attach on *redistribution of the Work or a Derivative Work*: include the licence, retain
copyright/patent/attribution notices, and state that you changed the files. A checkpoint
fine-tuned from Qwen3-14B should therefore ship an Apache-2.0 notice attributing Qwen3-14B and
stating it was modified. Apache-2.0 §3 also grants a patent licence that terminates on patent
litigation against the Work — routine, but worth knowing.

The trainers still do not pass `revision=` to `from_pretrained`, so the pin is documentary rather
than enforced at load time; the local cache does hold exactly this revision.

## 2. Source repositories

27 external repositories plus this repository. All were read at the pinned commit. **All 14
previously-unresolved `SEE-REPO` entries are now resolved, and all 14 are permissive.** Row counts
are rows contributed to the shipped 86,010-row midtrain corpus.

### 2.1 Previously `SEE-REPO` — now resolved

| Source | Licence (SPDX) | Upstream URL | Pinned revision | Used for / flows into |
| --- | --- | --- | --- | --- |
| ROCmKernelWiki (jhinpan) | Apache-2.0 | https://github.com/jhinpan/ROCmKernelWiki | 9252153f81b4e2e861d412b85033e79c3256c37d | 6,350 midtrain rows; mostly the `docs` channel |
| pytorch (pytorch) | BSD-3-Clause | https://github.com/pytorch/pytorch | 0dc2beb6f35031bfadcb72a01ea2573e22f97d46 | 5,497 midtrain rows across `rocm_hip`, `triton`, `docs`, `amd_asm` |
| cutlass (NVIDIA) | BSD-3-Clause | https://github.com/NVIDIA/cutlass | f94ec46f4f63f96003d6cfdf2014731e7672c281 | 1,211 midtrain rows; device-code reference |
| MIOpen (ROCm) | MIT | https://github.com/ROCm/MIOpen | 06977176afd94476c18d5290f21cb40745bb73a9 | 1,000 midtrain rows; `rocm_hip` and `amd_asm` |
| rocThrust (ROCm) | Apache-2.0 | https://github.com/ROCm/rocThrust | 8c061ed4f0628254578a3de28df775bea765f89d | 743 midtrain rows; inherited from NVIDIA Thrust |
| rocSPARSE (ROCm) | MIT | https://github.com/ROCm/rocSPARSE | 57ab6fadeaa55fead8e69f9a67b489502f35ac17 | 658 midtrain rows |
| rccl (ROCm) | BSD-3-Clause | https://github.com/ROCm/rccl | 57e58688f44c77076ad536ef1f6b68741fc6e694 | 491 midtrain rows; inherited from NVIDIA NCCL |
| rocSOLVER (ROCm) | BSD-2-Clause | https://github.com/ROCm/rocSOLVER | fe28dc62bba3860872577f60a7c52b7d3d048367 | 320 midtrain rows |
| rocFFT (ROCm) | MIT | https://github.com/ROCm/rocFFT | ce6b8be358024b4d4246db8317895ff16388cf9d | 216 midtrain rows |
| Tensile (ROCm) | MIT | https://github.com/ROCm/Tensile | e8a8999e0e7374aaae546a6d7cb703d9e06b0ebf | 188 midtrain rows |
| HIP (ROCm) | MIT | https://github.com/ROCm/HIP | 1377114f8220724206f1f5a770501fda11d8d1e1 | 182 midtrain rows |
| rocWMMA (ROCm) | MIT | https://github.com/ROCm/rocWMMA | 97562d33167f5dda32fa319cc5b4f62815bdd9e3 | 179 midtrain rows |
| hipCUB (ROCm) | BSD-3-Clause | https://github.com/ROCm/hipCUB | 3d8584c373d96989bebf2b3305f311c1283aeb91 | 141 midtrain rows; inherited from Merrill/NVIDIA CUB |
| rocRAND (ROCm) | MIT | https://github.com/ROCm/rocRAND | 9a2aab8643f1e2390e202fc6a71e0e8ae181ac48 | 123 midtrain rows |

Compound-licence detail, recorded so a future audit does not have to re-derive it:

- **MIOpen** — MIT overall; `src/include/miopen/kernel_cache.hpp` and `src/kernel_cache.cpp` are
  additionally under Apache-2.0 (Vratis Ltd, 2015). SPDX expression: `MIT AND Apache-2.0`.
- **rocFFT** — MIT for AMD's own code; bundles CLI11 2.2 under BSD-3-Clause (University of
  Cincinnati). SPDX expression: `MIT AND BSD-3-Clause`.
- **rocSOLVER** — AMD's grant is two-clause (no endorsement clause), so BSD-2-Clause; the file also
  carries a bundled third-party BSD-3-Clause block. SPDX expression:
  `BSD-2-Clause AND BSD-3-Clause`.
- **rccl** — BSD-3-Clause from NVIDIA NCCL; Microsoft's contributions are MIT.
- **hipCUB** — BSD-3-Clause; copyright Duane Merrill and NVIDIA, with AMD modifications.
- **cutlass** — the file carries an explicit `SPDX-License-Identifier: BSD-3-Clause` tag.

GitHub's licence API reports `NOASSERTION` for MIOpen, cutlass, hipCUB, pytorch, rccl, rocFFT,
rocSOLVER, composable_kernel, unsloth and xformers, because each of those licence files is
non-verbatim or multi-block. `NOASSERTION` is why they were never auto-resolved; each was read in
full by hand.

### 2.2 Previously recorded, re-verified at the pinned commit

| Source | Licence (SPDX) | Upstream URL | Pinned revision | Used for / flows into |
| --- | --- | --- | --- | --- |
| composable_kernel (ROCm) | MIT | https://github.com/ROCm/composable_kernel | 8c5870f962db354cc175e9dd915be37615d79518 | 3,033 midtrain rows; CK device code |
| FlagGems (FlagOpen) | Apache-2.0 | https://github.com/FlagOpen/FlagGems | de8d0e9dd7886a4f8f2cde0044daadf58824505c | 1,182 midtrain rows; Triton operator library |
| aiter (ROCm) | MIT | https://github.com/ROCm/aiter | 028756633e4192785217838f4924dc16516f5780 | 733 midtrain rows; AMD serving-op reference |
| GEAK (AMD-AGI) | MIT | https://github.com/AMD-AGI/GEAK | 4965d5b2ccde927925c8c5501a25c1233daa52eb | 719 midtrain rows; mostly `docs` |
| vllm (vllm-project) | Apache-2.0 | https://github.com/vllm-project/vllm | 05d4f8bba3aac85814c3fc42cfc60bef21bb2bb4 | 626 midtrain rows; paged-attention and MoE kernels |
| triton (triton-lang) | MIT | https://github.com/triton-lang/triton | e7eab7dd36b76e085d1ec858c6cee3aa88208400 | 610 midtrain rows; Triton language and tutorials |
| TransformerEngine (NVIDIA) | Apache-2.0 | https://github.com/NVIDIA/TransformerEngine | aef96db0c0ee959f8197007e4ffad13bd4074003 | 361 midtrain rows; fp8 reference paths |
| rocPRIM (ROCm) | MIT | https://github.com/ROCm/rocPRIM | 14cd5e3c27a4b9ae7d510823a450723a03985ac0 | 283 midtrain rows |
| Liger-Kernel (linkedin) | BSD-2-Clause | https://github.com/linkedin/Liger-Kernel | e1eeb997701bd1196d532037cb628036875175fa | 119 midtrain rows; fused Triton kernels |
| flash-attention (Dao-AILab) | BSD-3-Clause | https://github.com/Dao-AILab/flash-attention | 2402cb0bed7a2185cb9ddbe88fb998656cf73066 | 116 midtrain rows; attention reference |
| xformers (facebookresearch) | BSD-3-Clause | https://github.com/facebookresearch/xformers | 42fc265f8831e7f900ffb89f331a1b43e0dfa13f | 38 midtrain rows; attention reference |
| kernels (triton-lang) | MIT | https://github.com/triton-lang/kernels | 4f3d31f009ef6a113b44803194ac3412360e882a | 11 midtrain rows |

### 2.3 Corrected at the pinned commit — see §0

| Source | Licence (SPDX) | Upstream URL | Pinned revision | Used for / flows into |
| --- | --- | --- | --- | --- |
| unsloth (unslothai) | Apache-2.0 AND AGPL-3.0-only | https://github.com/unslothai/unsloth | 3b235895bdf08410e0a9032e663e82c0de60a6a4 | 45 midtrain rows total; 29 of them AGPL-3.0-only. See §0.2 |
| KORE (this repository) | LicenseRef-AMD-Proprietary-Internal | https://github.com/amd-shasriva/KORE.git | 37b391bc71da8f8f0244f7d2a7ddd6806533954c | 2,323 midtrain rows: 1,784 `pytorch_triton_pairs` + 539 `kore_tasks`. Catalog wrongly says MIT. See §0.3 |

## 3. Datasets

### 3.1 Recorded in the catalog — both misattributed, see §0.1

| Dataset | Licence (SPDX) | Upstream URL | Pinned revision | Used for / flows into |
| --- | --- | --- | --- | --- |
| `GPUMODE/KernelBook` | LicenseRef-GPUMode-ResearcherReciprocity-1.0 | https://huggingface.co/datasets/GPUMODE/KernelBook | b76504d85f7f14ef4b1fad81f136f638f2ce625b | 21,432 midtrain rows; torch to Triton breadth pairs; seeds 298 Tier-2 curriculum rows |
| `GPUMODE/kernelbot-data` | LicenseRef-GPUMode-ResearcherReciprocity-1.0 | https://huggingface.co/datasets/GPUMODE/kernelbot-data | 4159cf6b2c6bab208be6dda885d6d87631cc16df | 5,638 midtrain rows via config `amd_successful_submissions`; real AMD competition kernels |

Catalog value for both is `MIT`. Actual value is the June 9 Researcher Reciprocity License v1.0,
in force at both pinned revisions.

### 3.2 General-replay datasets — previously undocumented, now identified (see §4)

| Dataset | Licence (SPDX) | Upstream URL | Pinned revision | Used for / flows into |
| --- | --- | --- | --- | --- |
| `allenai/tulu-3-sft-mixture` | ODC-By-1.0 | https://huggingface.co/datasets/allenai/tulu-3-sft-mixture | b14afda60f1bbebe55d5d2fa1e4df5042f97f8be | 8,135 midtrain rows (`chat` + `instruction_following`); 7,984 SFT `general_chat` rows |
| `open-thoughts/OpenThoughts3-1.2M` | Apache-2.0 | https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M | 61bcf9d4eb38b30295efc2021227a63cc5bb34c8 | 4,677 midtrain rows from 1,910 long-CoT traces; 5,998 SFT `math_reasoning` rows |
| `nvidia/OpenCodeInstruct` | CC-BY-4.0 | https://huggingface.co/datasets/nvidia/OpenCodeInstruct | 8f3ba5bafe4d6e8db46082cf7ae6741bc370604d | 4,677 midtrain rows; 5,999 SFT `general_code` rows |
| `Team-ACE/ToolACE` | Apache-2.0 | https://huggingface.co/datasets/Team-ACE/ToolACE | 6bda777c88d21e5a204703c1ee45597a8fa4f734 | 3,997 midtrain rows; 3,997 of the 6,917 SFT `agentic_tooluse` rows |

**These four revisions are not pinned in code.** `_load_from_hf` calls `load_dataset` without a
`revision=`, unlike `_load_pinned_replay`, so the revisions above are the ones `main` resolved to on
the build host and are recoverable only from that host's Hugging Face cache. They are reproducible
today and will silently stop being reproducible if the cache is lost or the datasets move. Pinning
them in `HF_SOURCES`, and adding them to `source_metadata.json` as `datasets` entries, is the fix.

Terms notes:

- **tulu-3-sft-mixture** is ODC-By-1.0 as a whole, but its card is explicit: *"different licenses
  apply to subsets of the data. Some portions of the dataset are non-commercial. We present the
  mixture as a research artifact."* **The subset identity therefore decides the terms, not the
  mixture licence.** All 19 rows confirmed by exact SHA-256 (9 `chat`, 10
  `instruction_following`) came from `ai2-adapt-dev/numinamath_tir_math_decontaminated`, which the
  card lists as **Apache-2.0** — with no exceptions and no non-commercial subset seen. The signature
  is uniform across all 8,135 rows (100% boxed-answer, 100% fenced Python, ~61% SymPy
  tool-integrated reasoning), consistent with the whole slice being that one subset. The subset id
  is not recorded per row, so this is very strong but not exhaustive — see §10.2. The mixture also
  contains outputs generated by third-party models, which the card notes are subject to their own
  terms.
- **OpenCodeInstruct** is CC-BY-4.0 and its card states it is "ready for commercial/non-commercial
  use". CC-BY-4.0 requires attribution to NVIDIA on redistribution of the dataset or adaptations.
- **OpenThoughts3-1.2M** and **ToolACE** are Apache-2.0; both are synthetic, so the terms of the
  generating models sit behind them.

## 4. General-replay slice — RESOLVED

The previous version of this file reported 21,486 midtrain rows (25.0% of the corpus) as
unrecoverable, stamped:

```
repository_url: development-bundled://kore/general-replay
commit:         development-bundled
license:        DEVELOPMENT-INTERNAL
```

**That stamp is a false negative, and all 21,486 rows are now attributed.** The stamp records
which *code path* produced the row, not where the content came from.
`build_midtrain_corpus` applies `_development_replay_metadata` to any replay row that arrives
without a `_source_metadata` key, and `kore/data/general_replay.py::_load_from_hf` never attaches
one — so real upstream data loaded through that path is stamped identically to the 75-row bundled
smoke set. The content is definitely not the bundled set: all 21,486 rows are distinct, and **zero**
match any of the 75 bundled sample texts.

Attribution by replay kind. Each `row_id` is `"{kind}:{index}:{digest}"`, and `kind` maps
one-to-one onto `HF_SOURCES` in `kore/data/general_replay.py`:

| Replay kind | Rows | Upstream dataset | Licence | How it was established |
| --- | --- | --- | --- | --- |
| `tool_use` | 3,997 | `Team-ACE/ToolACE` | Apache-2.0 | **Proven** — 6/6 sampled rows matched a specific upstream row by full-text SHA-256 |
| `chat` | 4,670 | `allenai/tulu-3-sft-mixture` | ODC-By-1.0 | **Proven** — 9 sampled rows matched by SHA-256, all in the Apache-2.0 NuminaMath-TIR subset |
| `instruction_following` | 3,465 | `allenai/tulu-3-sft-mixture` | ODC-By-1.0 | **Proven** — 10 sampled rows matched by SHA-256, all in the Apache-2.0 NuminaMath-TIR subset |
| `code` | 4,677 | `nvidia/OpenCodeInstruct` | CC-BY-4.0 | **Proven** — 1 sampled row matched by SHA-256; low sample yield is an index artifact, see below |
| `math` | 4,677 (1,910 traces) | `open-thoughts/OpenThoughts3-1.2M` | Apache-2.0 | **Strong content and cache evidence, not hash-proven** — see §10.2 |
| **Total** | **21,486** | | | |

A SHA-256 equality between a corpus row's `root_content_hash` and an upstream row re-rendered
through KORE's own formatter is conclusive for that row: it identifies the exact upstream record.

Notes on the two slices where the sampling method was weakest, and why:

- **`code`.** The Hugging Face dataset server indexes only 1,400,000 of OpenCodeInstruct's
  5,000,000 rows (`partial: true`), so ~72% of the dataset cannot be queried at all and most probes
  can never hit. One probe did land an exact SHA-256 match, which proves the slice is
  OpenCodeInstruct. Non-matching probes returned template siblings — OpenCodeInstruct is heavily
  templated, so many rows share a 100-character opening — not different content.
- **`math`.** Fully indexed, but only reachable through BM25 ranking over 1.2M rows; candidate
  pools ran 30k–90k rows deep and the top-100 window cannot surface a specific record, so this
  method cannot confirm regardless of the truth. The attribution rests instead on: the loader's
  `kind` mapping; the local Hugging Face cache holding `open-thoughts/OpenThoughts3-1.2M` at a
  pinned revision while holding **nothing** for the configured fallback `nvidia/OpenMathInstruct-2`;
  100% of the 1,910 distinct traces carrying `<think>` long-CoT blocks, which is the OpenThoughts3
  QwQ-32B signature and which OpenMathInstruct-2 does not have; and physics, chemistry and
  engineering content matching OpenThoughts3's documented 100k science subset, which
  OpenMathInstruct-2 (math-only) cannot contain.

**Independent corroboration for all five.** The local Hugging Face hub cache contains exactly the
four primary datasets above, each pinned to the revision recorded in §3.2, and **none** of the three
configured fallbacks (`ise-uiuc/Magicoder-Evol-Instruct-110K`, `nvidia/OpenMathInstruct-2`,
`Salesforce/xlam-function-calling-60k`). The dataset-cache lock files record the matching
`load_dataset` calls, including OpenCodeInstruct's non-default `train` config. This independently
rules out the fallback sources — which matters, because
`Salesforce/xlam-function-calling-60k` is **CC-BY-NC-4.0** and would have been a genuine
non-commercial blocker had `tool_use` fallen back to it. It did not.

**Categorisation of all 21,486 rows** against the three admissible provenance classes:

| Class | Rows | Detail |
| --- | --- | --- |
| (a) Upstream source with a known licence | 21,486 | 8,135 tulu-3 · 4,677 OpenCodeInstruct · 4,677 OpenThoughts3 · 3,997 ToolACE |
| (b) Generated by this repository's own code | 0 | — |
| (c) Model output carrying its own terms | 0 directly | but every one of these four datasets is itself synthetic or partly synthetic, so vendor terms sit one level upstream |
| Untraceable | **0** | |

**Residual risk is (c), not (a).** All four are synthetic instruction data generated by other
vendors' models: OpenThoughts3 from QwQ-32B, OpenCodeInstruct from NVIDIA's generation pipeline,
parts of tulu-3 from third-party models its card flags, ToolACE from its own. Each publisher grants
Apache-2.0 / CC-BY-4.0 / ODC-By over the artifact, so KORE's licence position rests on those grants.
Whether an upstream generating model's own terms reach through the publisher's grant to a
third-party trainer is a question for counsel, not for this file; it is recorded in §10 as an open
item rather than asserted either way.

**Fix the stamp, not just the record.** `_load_from_hf` should attach `_source_metadata` the way
`_load_pinned_replay` already does, and `_development_replay_metadata` should be reachable only in
`development_mode`. Until that changes, any future corpus built through the same path will again
launder real upstream provenance into `DEVELOPMENT-INTERNAL`.

## 5. Kernel-curriculum slice — provenance identified, stamping still absent

9,956 midtrain rows (11.6%) carry **no `source_metadata` at all** and no licence field. They were
appended post-hoc by `data/release/generators/augment_midtrain.py`, which writes bare
`{"text": ..., "source": "kernel_curriculum"}` records straight onto the finished corpus file,
bypassing the source contract — and therefore also bypassing decontamination, dedup and the
tokenizer admission check.

Their origin is nevertheless determinate: they are the non-distractor tiers of
`curriculum_all.jsonl` (9,965 rows), which is shipped under `data/release/curriculum/`. The
generator's `KEEP` set excludes the 9 `kernel_distractor` rows, and 9,965 − 9 = 9,956 exactly.

| Tier | Rows | Origin | Terms |
| --- | --- | --- | --- |
| `kernel_math` (Tier 1) | 8,873 | Claude prose over a deterministic in-house solver; every answer auto-verified against the computed number | Teacher output, §6 |
| `kernel_reasoning` (Tier 4) | 724 | Claude traces grounded in KORE's own measured gfx950 wins | Teacher output over AMD data, §6 |
| `kernel_qa` (Tier 2) | 298 | OSS-Instruct-style QA generated over **real KernelBook kernels** | Teacher output over §0.1 material |
| `kernel_concept` (Tier 3) | 41 | Claude concept prose | Teacher output, §6 |
| `kernel_evol` (Tier 5) | 20 | Claude evol-instruct | Teacher output, §6 |
| `kernel_distractor` (Tier 6) | 9 | Claude distractors | Excluded from midtrain; present in SFT |

So the content is AMD-originated or teacher-generated, with the single caveat that the 298 Tier-2
rows are derived from KernelBook and inherit §0.1. What is missing is the **stamping**: these rows
should carry `source_metadata` naming the teacher model, the generator, and the licence position,
and they should be re-admitted through `build_midtrain_corpus` rather than appended behind it.

## 6. Teacher-generated content

Portions of the SFT, DPO and curriculum data were generated by **Anthropic Claude**
(`kore/data/teacher.py` defaults to `claude-opus-4.8`, overridable via `KORE_TEACHER_MODEL`;
`DATASET_STATUS.md` records `claude-opus-5` for the curriculum run) accessed through AMD's internal
LLM gateway. External use of teacher-generated text is subject to the applicable Anthropic terms
and to AMD's gateway terms — including any restriction on using outputs to develop competing
models, which is the specific clause to check before a KORE checkpoint is offered externally.

Volumes in the shipped artifacts: 9,965 curriculum rows (all six tiers appear in SFT as
`kernel_qa`); 9,956 of them also in midtrain. The Tier-1 kernel-math subset (8,873 rows) is
additionally auto-verified against a deterministic solver, which raises its factual quality but
does not change its provenance.

SFT composition for reference (56,493 rows): `kernel_repair_opt` 19,630 · `kernel_qa` 9,965 ·
`general_chat` 7,984 · `agentic_tooluse` 6,917 · `general_code` 5,999 · `math_reasoning` 5,998.
The four `general_*` / `math_reasoning` / `agentic_tooluse` channels total 26,898 rows drawn from
the same four upstream datasets as §4 — 15,616 of them are byte-identical to midtrain replay rows —
and, like the midtrain slice, they carry **no provenance fields at all**. The DPO artifact
(96,675 pairs) carries no `_source` field on any row.

## 7. Evaluation benchmarks

The retention suite evaluates against MMLU, HumanEval, LiveCodeBench, IFEval, BFCL and MT-Bench.
Each carries its own licence and citation requirements, none of which is recorded here yet
(§10.1). The frozen decontamination artifact `data/release/meta/benchmark_artifact.json.gz`
contains hashed benchmark records used to exclude contamination; it is derived from those
benchmarks and inherits their terms. Because it stores hashes rather than benchmark text, its
redistribution exposure is lower than the benchmarks themselves, but it is not zero.

## 8. Runtime dependencies

Pinned in `requirements-conductor.txt` and `.github/constraints-ci.txt`. Principal components:
PyTorch (ROCm build), pytorch-triton-rocm, Transformers, TRL, PEFT, Accelerate, Datasets, NumPy,
scikit-learn, XGBoost, and the ROCm stack including AITER, hipBLASLt and rocprofv3. Each is used
under its own licence; none is vendored into this repository except as noted in §2. These are
build- and run-time dependencies that are not redistributed with the corpora or the checkpoint, so
their obligations are weaker than §2's — but a shipped container image would change that.

## 9. How this was verified

So that the next auditor can re-run rather than re-derive:

1. **Licences at the pin, not at `main`.** For each of the 28 catalog sources, both
   `https://api.github.com/repos/{owner}/{repo}/license?ref={commit}` and
   `https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{candidate}` over 17 conventional
   licence filenames. Every file found was classified by full text, ordered so copyleft and
   non-commercial families are matched before permissive ones. This is what surfaced unsloth's
   `COPYING`, which sits beside an Apache-2.0 `LICENSE` and would be missed by reading `LICENSE`
   alone.
2. **Subtree relicensing.** The unsloth tree at the pinned commit was enumerated via the GitHub
   trees API and filtered for licence-like filenames, which found three AGPL-3.0 subtree licences.
   Corpus rows were then matched by `source_metadata.path` prefix.
3. **Corpus-wide copyleft scan.** All four shipped artifacts — 86,010 midtrain rows, 56,493 SFT
   rows, 96,675 DPO pairs and 9,965 curriculum rows — were scanned for AGPL / GPL / LGPL / MPL /
   CC-BY-NC / CC-BY-SA / research-only / proprietary markers. The 29 unsloth rows in midtrain are
   the only matches anywhere; SFT, DPO and curriculum are clean.
4. **Replay attribution.** Corpus rows were re-rendered through KORE's own formatters
   (`_fmt_qa`, `_fmt_sharegpt`, `_fmt_messages_passthrough`, `_messages_to_text`) and compared by
   SHA-256 against upstream rows retrieved from the Hugging Face dataset server `/search` and
   `/filter` endpoints. Row-level integrity was checked first: for all 21,486 rows the `row_id`
   digest suffix matches `root_content_hash`, and the stored text hashes to `content_hash`.
5. **Cache corroboration.** `~/.cache/huggingface/hub` and `.../datasets` were inspected for which
   datasets and revisions were actually fetched on the build host, and for which configured
   fallbacks were not.
6. **Base model.** `LICENSE` and `README.md` read from `huggingface.co/Qwen/Qwen3-14B` at the pinned
   revision, cross-checked against the revision-scoped model API and the local snapshot.

Two `release`-marked tests in `tests/test_packaging_contract.py` keep this file honest:
`test_third_party_attribution_is_complete_and_structured` parses the tables above and fails if any
entry loses its licence, URL, pinned revision or usage, or if a placeholder like `SEE-REPO` returns;
`test_third_party_covers_every_catalog_source` fails if a source or dataset in
`source_metadata.json` is missing here, or is documented at a different revision than the build
actually used.

## 10. Unresolved items

Honest list. Everything not on it is resolved above.

**10.1 — Evaluation benchmark licences (§7).** MMLU, HumanEval, LiveCodeBench, IFEval, BFCL and
MT-Bench are used through `kore/eval/retention.py` and hashed into
`data/release/meta/benchmark_artifact.json.gz`. Their individual licences and citation requirements
have not been resolved. *Not resolved because* it was out of scope for this pass, which prioritised
material that flows into shipped training data; benchmarks are evaluation-only and are not
redistributed as text. Tractable: six datasets, same method as §3.

**10.2 — Sampling, not census, on two points (§4).** First: the `math` slice (4,677 rows from 1,910
traces) is attributed to OpenThoughts3-1.2M on content-signature and cache evidence, with no exact
hash match. Second: the tulu-3 slice's per-row subset identity is confirmed for 19 rows out of
8,135 — all Apache-2.0 NuminaMath-TIR — so a stray row from one of tulu-3's non-commercial subsets
cannot be excluded by census. *Not resolved because* the dataset server cannot answer
exact-membership queries against list-typed columns, its BM25 ranking cannot surface a specific
record from a 1.2M-row pool, and OpenCodeInstruct is only 28% indexed. Closeable offline: stream the
pinned parquet for each dataset, re-render each row through KORE's formatter, and intersect the
SHA-256 set with the corpus `root_content_hash` set — one pass settles both the `math` attribution
and every tulu-3 per-row subset id.

**10.3 — Whether upstream generating-model terms reach through §3.2's grants (§4).** All four replay
datasets are synthetic model output — OpenThoughts3 from QwQ-32B, the others from their publishers'
own generation pipelines, with tulu-3's card explicitly flagging third-party model output inside the
mixture. The publishers grant Apache-2.0 / CC-BY-4.0 / ODC-By over the result, but some generating
models' own terms restrict using their outputs to train a competing model. *Not resolved because*
this is a legal question about the effect of an intermediate publisher's grant, not a factual one
this audit can settle.

**10.4 — Anthropic and AMD-gateway terms for teacher-generated content (§6).** Which Anthropic terms
applied at generation time, and whether they restrict external release of a model trained on the
output. *Not resolved because* it needs AMD's gateway contract, which is not in this repository.

**10.5 — `source_metadata.json` still carries the wrong values (§0).** 14 `SEE-REPO` placeholders,
`MIT` for both GPUMODE datasets, `Apache-2.0` for unsloth, and `MIT` for KORE itself. *Not resolved
because* this pass does not own that artifact and rewriting it changes the content hash of a shipped
release file. The corrected values are all in §1–§3 and can be applied mechanically.

**10.6 — Selection of an outbound licence by an authorized AMD owner ([`LICENSE`](LICENSE)).**
Unchanged from before. Note that §0.1 constrains the choice: any outbound terms for a Covered Model
must carry Attachment A.

---

**Bottom line.** Nothing found here forbids AMD's *internal* research use of the trained model or the
shipped datasets. Two findings constrain *external* release: the GPU Mode reciprocity licence over
31.5% of the midtrain corpus (§0.1), which is compliable but binds AMD's outbound terms, and 29 rows
of AGPL-3.0 source (§0.2), which is cheaply removable and should be removed before the next training
run. Until §0 is closed and §10 is worked down, this repository, its datasets, and any model trained
from them remain internal-only.
