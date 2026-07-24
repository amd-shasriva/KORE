# Midtrain source provenance and decontamination

`kore.data.midtrain_corpus.build_midtrain_corpus` has two explicit modes.
Production is the default and fails closed. Development must be selected with
`development_mode=True` (or `KORE_MIDTRAIN_DEVELOPMENT=1`) and is labeled in the
report as well as locally inferred source metadata.

Non-midtrain dataset assembly can explicitly select bundled benchmark smoke
references with `KORE_DECONTAM_DEVELOPMENT=1`; without that label it also
requires the frozen artifact.

## Production inputs

A production build requires:

1. A full frozen benchmark-text artifact supplied through
   `benchmark_artifact=` or `KORE_DECONTAM_BENCHMARK_ARTIFACT`. Smoke benchmark
   files are not accepted.
2. The actual model tokenizer at an immutable revision. Pass a tokenizer object
   and `tokenizer_revision=`, or make the pinned tokenizer available locally and
   set `KORE_TOKENIZER_REVISION`. Loading uses `local_files_only=True`.
3. Verified source roots. Git URL, commit, and license can be inferred from a
   repository, or supplied in the source metadata artifact through
   `source_metadata=` / `KORE_SOURCE_METADATA`.
4. A verified immutable revision and license for every enabled external dataset.

The source metadata artifact is validated against
[`source_metadata.schema.json`](source_metadata.schema.json). A minimal catalog
looks like:

```json
{
  "schema_version": "1.0",
  "sources": [
    {
      "local_path": "/data/repos/triton",
      "repository_url": "https://github.com/triton-lang/triton.git",
      "commit": "<immutable commit>",
      "license": "MIT",
      "source_id": "triton",
      "lineage_id": "triton@<immutable commit>",
      "verified": true
    }
  ],
  "datasets": [
    {
      "dataset_id": "GPUMODE/KernelBook",
      "revision": "<immutable dataset commit>",
      "license": "Apache-2.0",
      "verified": true
    }
  ],
  "holdouts": {
    "families": ["mla", "paged_attention"],
    "source_ids": [],
    "lineage_ids": [],
    "training_cutoff": "2026-01-01T00:00:00Z"
  }
}
```

`_drafts` trees, metadata entries with `verified: false`, reserved source
lineages, post-cutoff sources, and held-out families are removed before source
pairs or chunks are derived.

## Frozen benchmark-text artifact

The artifact is generated outside the training build after fetching the complete
benchmark splits at pinned revisions. Tests never fetch it. The accepted shape
is:

```json
{
  "artifact_type": "kore.frozen-benchmark-texts",
  "schema_version": "1.0",
  "scope": "full",
  "benchmarks": {
    "mmlu": {
      "dataset": "cais/mmlu",
      "revision": "<immutable dataset commit>",
      "split": "test",
      "license": "MIT",
      "records": [
        {
          "row_id": "abstract_algebra:0",
          "text": "<complete admission text>",
          "content_hash": "sha256:<hash of text>"
        }
      ]
    }
  }
}
```

The artifact must contain all configured retention benchmarks. Every text hash
is recomputed before use. The canonical artifact hash and benchmark revisions
are copied into the corpus build report.

## Output row contract

Each JSONL row still has `text` and `source`, and now also has
`source_metadata`. The metadata preserves:

* repository URL and immutable commit, or dataset URL and revision;
* source-relative path, upstream row ID, and license;
* source and lineage IDs;
* SHA-256 of the admitted chunk and its source root;
* parent hashes and derivation steps for normalized, paired, and chunked rows;
* merged origins after dedup; and
* deterministic sampling weight and replica number when a channel is weighted.

Dedup is performed within a source channel. Identical content in a raw-code
channel and a weighted translation-pair channel is therefore not allowed to
erase the pair channel. Duplicate files inside one channel collapse and retain
all origins.

## Leakage decision order

Decontamination reports a stable reason and non-text evidence for each drop:

1. held-out family, task, source, lineage, or time partition;
2. declared ancestry from a held-out root;
3. exact SHA-256;
4. normalized Python AST;
5. normalized semantic graph;
6. MinHash near-duplicate; or
7. directional containment within one held-out lineage cluster.

Directional containment divides shared signal by the held-out reference size,
not the candidate size. A held-out kernel pasted into a long document cannot be
diluted. Common Triton imports, decorators, coordinate setup, loads, and stores
are removed from fuzzy evidence, so those idioms alone cannot contaminate a
legitimate training kernel.

## Rebuild requirement

Existing midtrain JSONL files do not satisfy this contract. They lack immutable
source provenance, frozen benchmark coverage, tokenizer identity, ancestry, and
reasoned decontamination evidence. They must be rebuilt from source roots; an
in-place metadata backfill cannot prove lineage or recover rows removed by the
old corpus-wide overlap and dedup behavior.
