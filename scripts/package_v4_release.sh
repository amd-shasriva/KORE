#!/usr/bin/env bash
# Package the v4 SFT mixture into the repo's release layout.
#
# The point is that someone can clone the repo, run one script, and have the
# exact file the training config expects -- no network, no hub account, no
# regeneration from 82k raw episodes. That is why the RAW episodes are not
# shipped: they are provenance, ~5.6GB of them, and nobody training a model
# needs to re-derive the mixture we already derived.
#
# GitHub rejects any single file over 100MB on a normal push, so the mixture is
# gzipped and split into 95MB parts, matching the convention the existing
# midtrain/dpo/curriculum artifacts already use. reassemble.sh cats the parts and
# gunzips them back to one file.
set -uo pipefail

SRC="${1:-/shared_nfs/shasriva/kore/datagen/multicap_v4.jsonl}"
DEST="${2:-/home/shasriva/Kore-RL/KORE/data/release/sft}"
PART_MB="${PART_MB:-95}"

[ -s "$SRC" ] || { echo "missing or empty: $SRC" >&2; exit 1; }
mkdir -p "$DEST"

rows=$(wc -l < "$SRC")
raw=$(du -h "$SRC" | cut -f1)
echo "source : $SRC"
echo "rows   : $rows   raw: $raw"

rm -f "$DEST"/multicap_v4.jsonl.gz.part* 2>/dev/null
echo "compressing and splitting into ${PART_MB}MB parts..."
gzip -c "$SRC" | split -b "${PART_MB}m" - "$DEST/multicap_v4.jsonl.gz.part"

n=$(ls "$DEST"/multicap_v4.jsonl.gz.part* 2>/dev/null | wc -l)
tot=$(du -ch "$DEST"/multicap_v4.jsonl.gz.part* 2>/dev/null | tail -1 | cut -f1)
echo "wrote  : $n parts, $tot total"
ls -lh "$DEST"/multicap_v4.jsonl.gz.part* | awk '{print "  " $5, $9}'

# Checksum the REASSEMBLED stream, not the parts: that is what a consumer ends up
# with, and it catches a split/cat mismatch that per-part sums would not.
sum=$(cat "$DEST"/multicap_v4.jsonl.gz.part* | gunzip | sha256sum | cut -d' ' -f1)
cat > "$DEST/multicap_v4.manifest.json" <<EOF
{
  "artifact": "multicap_v4.jsonl",
  "rows": $rows,
  "parts": $n,
  "part_size_mb": $PART_MB,
  "sha256_reassembled": "$sum",
  "reassemble": "cat multicap_v4.jsonl.gz.part* | gunzip > multicap_v4.jsonl"
}
EOF
echo "sha256 (reassembled): $sum"
echo "manifest: $DEST/multicap_v4.manifest.json"
