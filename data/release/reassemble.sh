#!/bin/bash
# Reassemble split+gzip per-stage datasets. Run from data/release/.
set -e
cat midtrain/corpus.jsonl.gz.part* | gunzip > ../b05factory/midtrain/corpus.jsonl
cat sft/multicap.jsonl.gz.part*   | gunzip > ../b05factory/sft/multicap.jsonl
cat dpo/pairs.jsonl.gz.part*      | gunzip > ../b05factory/dpo/pairs.jsonl
mkdir -p ../b05factory && cat provenance/datagen.tar.gz.part* | gunzip | tar -C ../b05factory -xf -
echo "reassembled corpus/sft/dpo + provenance"
