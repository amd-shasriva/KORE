#!/usr/bin/env bash
# Prove the checkpoint path can absorb a full-size write before the trainer needs it.
#
# Run 33992 died at step 400 with EDQUOT after writing 4 of 25 shards, and the
# only reason we knew the path was broken was that six hours of training were
# already gone. Checking beforehand costs minutes.
#
# Writes a file the size of one real checkpoint, in the actual output directory,
# in shard-sized pieces -- not one big dd -- because that is the shape of the
# failure: safetensors serialises 25 shards sequentially and died between two of
# them, which a single large write would not reproduce faithfully. Removes
# everything it wrote, including on interrupt.
set -uo pipefail

DIR="${1:-/shared_nfs/shasriva/kore/runs/sft_coder30b_a3b}"
# 30.5B params: ~61GB bf16 weights + ~427GB Adam fp32 moments and grads.
TOTAL_GB="${TOTAL_GB:-488}"
SHARD_GB="${SHARD_GB:-20}"

WORK="$DIR/.write_probe"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

mkdir -p "$WORK" || { echo "FAIL: cannot create $WORK"; exit 1; }

avail_g() { df -k "$1" | tail -1 | awk '{printf "%.0f", $4/1024/1024}'; }
echo "target dir : $DIR"
echo "free before: $(avail_g "$DIR")G"
echo "writing    : ${TOTAL_GB}G in ${SHARD_GB}G shards (one checkpoint's worth)"

written=0
n=0
while [ "$written" -lt "$TOTAL_GB" ]; do
    n=$((n + 1))
    if ! dd if=/dev/zero of="$WORK/shard_$n.bin" bs=1M count=$((SHARD_GB * 1024)) \
            status=none 2>/dev/null; then
        echo "FAIL: write died at ${written}G after $((n - 1)) shards"
        echo "      free now: $(avail_g "$DIR")G"
        exit 1
    fi
    written=$((written + SHARD_GB))
    [ $((n % 5)) -eq 0 ] && echo "  ${written}G ok"
done

# fsync everything, because a write that only reached page cache proves nothing
# about whether the server would have accepted it.
sync

echo "wrote      : ${written}G across $n shards, all fsynced"
echo "free after : $(avail_g "$DIR")G"
echo "PASS -- the checkpoint path can absorb a full ${TOTAL_GB}G checkpoint"
