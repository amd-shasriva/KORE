#!/usr/bin/env bash
# Fetch the KORE RL checkpoint-30 weights from the SPUR shared volume.
#
# This is the one thing that is NOT in this repository and cannot be: the
# checkpoint is 342 GB against GitHub's 100 MB per-file hard limit, and LICENSE
# separately forbids publishing a derived checkpoint to a public registry.
# Everything else -- code, the training corpus, the raw episodes, the run
# evidence -- is committed here and needs no fetching.
#
#   ./scripts/fetch_weights.sh /path/to/destination            # weights only, 114 GB
#   ./scripts/fetch_weights.sh /path/to/destination --resume   # + optimizer, 342 GB
#
# Run it from a SPUR login node, or from anywhere that can reach one over SSH
# (set KORE_SPUR_SSH, e.g. KORE_SPUR_SSH=shasriva@crs-spur.crusoe.amd.com).
#
# WHICH ONE DO YOU WANT
#   Default (114 GB) gives the model: evaluate it, serve it, or fine-tune from
#   it. That is what almost everyone needs.
#   --resume (342 GB) adds optimizer.pt, rng_state.pth and scheduler.pt, which
#   are only required to CONTINUE the interrupted RL run from step 30. The run
#   stopped at 31 of 1,000 configured steps, so resuming is a real option --
#   but read docs/evidence/RL_RUN_PROVENANCE.md first, because the run had a
#   masking bug worth fixing before spending more compute on it.
#
# Transfers resume safely: rsync --partial --append-verify means an interrupted
# run picks up where it stopped rather than starting the 114 GB again.

set -euo pipefail

SRC_HOST="${KORE_SPUR_SSH:-}"
SRC_PATH="${KORE_CHECKPOINT_PATH:-/shared_nfs/shasriva/rl_checkpoint-30}"
DEST="${1:-}"
MODE="${2:-weights}"

if [ -z "$DEST" ]; then
    sed -n '2,30p' "$0" | sed 's/^# \?//'
    exit 2
fi

case "$MODE" in
    weights|--weights) WANT_OPT=0 ;;
    --resume|resume)   WANT_OPT=1 ;;
    *) echo "unknown mode '$MODE' (expected --resume or nothing)" >&2; exit 2 ;;
esac

command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 1; }

# Exclusions are what separates the 114 GB model from the 342 GB training state.
EXCLUDES=()
if [ "$WANT_OPT" -eq 0 ]; then
    EXCLUDES=(--exclude 'optimizer.pt' --exclude 'rng_state.pth' --exclude 'scheduler.pt')
fi

if [ -n "$SRC_HOST" ]; then
    SRC="$SRC_HOST:$SRC_PATH/"
    PROBE=(ssh -o BatchMode=yes "$SRC_HOST" "test -d '$SRC_PATH'")
else
    SRC="$SRC_PATH/"
    PROBE=(test -d "$SRC_PATH")
fi

echo "source      : ${SRC_HOST:+$SRC_HOST:}$SRC_PATH"
echo "destination : $DEST"
echo "mode        : $([ "$WANT_OPT" -eq 1 ] && echo 'full (342 GB, resumable training state)' || echo 'weights only (114 GB)')"
echo

if ! "${PROBE[@]}" 2>/dev/null; then
    cat >&2 <<EOF
Cannot see $SRC_PATH.

Either you are not on a SPUR login node, or you cannot read that path. Two
things to check, in order:

  1. Set KORE_SPUR_SSH to a host that can reach it:
       KORE_SPUR_SSH=<user>@crs-spur.crusoe.amd.com $0 "$DEST"
  2. The path lives under a personal directory. If the owning account has been
     closed, ask cluster operations who owns /shared_nfs/shasriva now. See
     HANDOVER.md.
EOF
    exit 1
fi

mkdir -p "$DEST"

# --append-verify so an interrupted 114 GB transfer resumes instead of restarting;
# rsync checksums every block it reuses, so a resumed file is not taken on trust.
rsync -av --progress --partial --append-verify "${EXCLUDES[@]}" "$SRC" "$DEST/"

echo
echo "=== verifying ==="
SHARDS=$(find "$DEST" -maxdepth 1 -name 'model-*-of-00025.safetensors' | wc -l)
echo "safetensors shards: $SHARDS / 25"
[ "$SHARDS" -eq 25 ] || { echo "INCOMPLETE: re-run this script, it resumes." >&2; exit 1; }

for required in config.json model.safetensors.index.json tokenizer.json trainer_state.json; do
    [ -f "$DEST/$required" ] || { echo "MISSING: $required" >&2; exit 1; }
done

# The identity check that matters. Two checkpoint directories exist from
# different cycles, so confirm the step rather than trusting the folder name.
STEP=$(python3 -c "import json;print(json.load(open('$DEST/trainer_state.json')).get('global_step'))" 2>/dev/null || echo '?')
echo "global_step       : $STEP (expected 30)"
[ "$STEP" = "30" ] || echo "WARNING: this is not the checkpoint the published results describe." >&2

echo
echo "Done. Load it with:"
echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "  m = AutoModelForCausalLM.from_pretrained('$DEST', torch_dtype='bfloat16', device_map='auto')"
echo
echo "Results this checkpoint produced, and the caveats, are in"
echo "docs/evidence/RL_RUN_PROVENANCE.md."
