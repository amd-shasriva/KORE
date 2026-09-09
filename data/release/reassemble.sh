#!/bin/bash
set -e
# Every target directory is created first: on a fresh checkout none of them exist
# yet, and under `set -e` the very first redirect below would abort the recovery.
mkdir -p ../b05factory/midtrain ../b05factory/sft ../b05factory/dpo ../../kore_offline
cat midtrain/corpus.jsonl.gz.part* | gunzip > ../b05factory/midtrain/corpus.jsonl
cat sft/multicap.jsonl.gz.part*   | gunzip > ../b05factory/sft/multicap.jsonl
# multicap_v2 was the 14B-era mixture: the base mix plus filtered multi-turn
# kernel-refinement trajectories. configs/sft_14b_full.json and
# configs/sft_14b_pathb.json still point at it.
#
# Its kernel_multiturn_refine shards are NOT in this release, so v2 cannot be
# rebuilt from a fresh checkout. Guarded rather than deleted, because under
# `set -e` a bare `cat` on a glob that matches nothing failed HERE and aborted
# the script before the v5 rebuild at the bottom -- and v5 is the mixture the
# 30B config actually trains on. The recovery path for the current model was
# therefore dead while the message on screen was about a legacy artifact.
if compgen -G "sft/kernel_multiturn_refine.jsonl.gz.part*" > /dev/null; then
    cat sft/kernel_multiturn_refine.jsonl.gz.part* | gunzip > ../b05factory/sft/kernel_multiturn_refine.jsonl
    cat ../b05factory/sft/multicap.jsonl ../b05factory/sft/kernel_multiturn_refine.jsonl \
        > ../b05factory/sft/multicap_v2.jsonl
else
    echo "skip: multicap_v2 -- kernel_multiturn_refine shards are not in this release" >&2
fi
cat dpo/pairs.jsonl.gz.part*      | gunzip > ../b05factory/dpo/pairs.jsonl
cat curriculum/curriculum_all.jsonl.gz.part* | gunzip > ../../kore_offline/curriculum_all.jsonl 2>/dev/null || true
cat provenance/datagen.tar.gz.part* | gunzip | tar -C ../b05factory -xf -
# v4: the mixture the 30B SFT config points at -- 244,732 rows, measured
# 65.9/21.5/12.3 kernel/chat/coding by tokens. Rebuilt by concatenation like the
# others so a fresh checkout reproduces it with no network and no hub account.
cat sft/multicap_v4.jsonl.gz.part* | gunzip > ../b05factory/sft/multicap_v4.jsonl

# v5: what the 30B SFT config now points at. 206,586 rows after the eval split
# (207,782 built, 1,196 removed), 165,047 distinct targets over 11,793 tasks;
# 61.2% kernel / 38.8% replay by rows and 14% replay by tokens. Six task shapes
# rather than v4's one, every kernel target verified numerically on gfx950, and
# screened against the evaluation benchmark's own sources -- which v4 never was.
#
# Kept alongside v4 rather than replacing it. Deleting the v4 parts would not
# reclaim any space (the blobs stay in history regardless) and it would remove the
# only fallback if v5 evaluates worse, so there is cost and no benefit.
cat sft/v5_sft.jsonl.gz.part* | gunzip > ../b05factory/sft/v5_sft.jsonl

# The held-out eval slice, shipped rather than regenerated. scripts/v5_split_eval.py
# would rebuild it, but only from the PRE-split mixture -- rerunning it against the
# already-split file would carve a second slice out of what is left. Shipping both
# halves keeps the train/eval boundary reproducible byte-for-byte, which is the
# whole point of holding rows out.
gunzip -c sft/v5_eval.jsonl.gz > ../b05factory/sft/v5_eval.jsonl

# configs/sft_coder30b_a3b.json reads data/v5_sft.jsonl and data/v5_eval.jsonl,
# one directory above where every other reassembled artifact lands. Without
# these links the config's own comment -- "reassemble.sh reproduces exactly this
# file" -- is false, and kore/policy/sft.py fails at load telling you to run a
# script that has already run.
#
# Linked rather than copied: v5_sft.jsonl is 1.8 GB, and a second physical copy
# of the training corpus is the kind of thing that quietly fills a node.
ln -sfn b05factory/sft/v5_sft.jsonl  ../v5_sft.jsonl
ln -sfn b05factory/sft/v5_eval.jsonl ../v5_eval.jsonl
echo reassembled
