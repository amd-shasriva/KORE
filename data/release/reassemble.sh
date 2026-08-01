#!/bin/bash
set -e
# Every target directory is created first: on a fresh checkout none of them exist
# yet, and under `set -e` the very first redirect below would abort the recovery.
mkdir -p ../b05factory/midtrain ../b05factory/sft ../b05factory/dpo ../../kore_offline
cat midtrain/corpus.jsonl.gz.part* | gunzip > ../b05factory/midtrain/corpus.jsonl
cat sft/multicap.jsonl.gz.part*   | gunzip > ../b05factory/sft/multicap.jsonl
cat dpo/pairs.jsonl.gz.part*      | gunzip > ../b05factory/dpo/pairs.jsonl
cat curriculum/curriculum_all.jsonl.gz.part* | gunzip > ../../kore_offline/curriculum_all.jsonl 2>/dev/null || true
cat provenance/datagen.tar.gz.part* | gunzip | tar -C ../b05factory -xf -
echo reassembled
