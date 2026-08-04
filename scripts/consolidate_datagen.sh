#!/usr/bin/env bash
# Gather every datagen wave into one directory the mixture builder can glob.
#
# build_sft_v3_mixture.py takes a single --agentic-dir and globs *.jsonl inside
# it, but the waves ran on two machines that share no filesystem and every one
# of them names its output shard_000.jsonl. Copying them into one place would
# silently overwrite three quarters of the corpus, so each file is prefixed with
# the wave it came from.
#
# Hard links, not copies: the shards total ~5.6GB and both the source and the
# destination live on /shared_nfs, so linking costs no space and no time. Falls
# back to copying if the source is on a different device.
#
# Telemetry shards are excluded explicitly. They match shard_*.jsonl too, and
# counting them once inflated a wave's apparent size by 2.34x.
set -uo pipefail

ROOT="${KORE_DATAGEN_ROOT:-/shared_nfs/shasriva/kore/datagen}"
DEST="${1:-$ROOT/all_v4}"
WAVES="${KORE_DATAGEN_WAVES:-agentic_v2 b501supp/out b501remain/out b501wave3/out}"

mkdir -p "$DEST"
find "$DEST" -maxdepth 1 -name '*.jsonl' -delete 2>/dev/null

n=0
for src in $WAVES; do
    [ -d "$ROOT/$src" ] || { echo "skip (missing): $src"; continue; }
    tag="${src//\//_}"
    c=0
    for f in "$ROOT/$src"/shard_*.jsonl; do
        [ -e "$f" ] || continue
        case "$f" in *telemetry*) continue;; esac
        ln -f "$f" "$DEST/${tag}_$(basename "$f")" 2>/dev/null \
            || cp "$f" "$DEST/${tag}_$(basename "$f")"
        c=$((c + 1)); n=$((n + 1))
    done
    echo "$src -> $c shards"
done

echo "consolidated $n shard files into $DEST"
echo "episodes: $(cat "$DEST"/*.jsonl 2>/dev/null | wc -l)"
