#!/bin/bash
set -e
# Every target directory is created first: on a fresh checkout none of them exist
# yet, and under `set -e` the very first redirect below would abort the recovery.
mkdir -p ../b05factory/midtrain ../b05factory/sft ../b05factory/dpo ../../kore_offline
cat midtrain/corpus.jsonl.gz.part* | gunzip > ../b05factory/midtrain/corpus.jsonl
cat sft/multicap.jsonl.gz.part*   | gunzip > ../b05factory/sft/multicap.jsonl
# multicap_v2 is the mixture SFT actually trains on: the base mix plus filtered
# multi-turn kernel-refinement trajectories. It is rebuilt here by concatenation
# rather than downloaded, so a fresh checkout reproduces it with no network.
cat sft/kernel_multiturn_refine.jsonl.gz.part* | gunzip > ../b05factory/sft/kernel_multiturn_refine.jsonl
cat ../b05factory/sft/multicap.jsonl ../b05factory/sft/kernel_multiturn_refine.jsonl \
    > ../b05factory/sft/multicap_v2.jsonl
cat dpo/pairs.jsonl.gz.part*      | gunzip > ../b05factory/dpo/pairs.jsonl
cat curriculum/curriculum_all.jsonl.gz.part* | gunzip > ../../kore_offline/curriculum_all.jsonl 2>/dev/null || true
cat provenance/datagen.tar.gz.part* | gunzip | tar -C ../b05factory -xf -
# v4: the mixture the 30B SFT config points at -- 244,732 rows, measured
# 65.9/21.5/12.3 kernel/chat/coding by tokens. Rebuilt by concatenation like the
# others so a fresh checkout reproduces it with no network and no hub account.
cat sft/multicap_v4.jsonl.gz.part* | gunzip > ../b05factory/sft/multicap_v4.jsonl
echo reassembled
