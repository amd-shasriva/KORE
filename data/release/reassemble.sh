#!/bin/bash
set -e
cat midtrain/corpus.jsonl.gz.part* | gunzip > ../b05factory/midtrain/corpus.jsonl
cat sft/multicap.jsonl.gz.part*   | gunzip > ../b05factory/sft/multicap.jsonl
cat dpo/pairs.jsonl.gz.part*      | gunzip > ../b05factory/dpo/pairs.jsonl
cat curriculum/curriculum_all.jsonl.gz.part* | gunzip > ../../kore_offline/curriculum_all.jsonl 2>/dev/null || true
cat provenance/datagen.tar.gz.part* | gunzip | tar -C ../b05factory -xf -
echo reassembled
