#!/usr/bin/env bash
# Full pre-launch data verification: the shipped release parts must reproduce both
# halves of the mixture byte-for-byte, and both halves must pass the correctness
# gates. The eval half is checked with the same gates as training, because a
# contaminated or degenerate eval row corrupts the retention signal the run depends
# on just as surely as a bad training row corrupts the weights.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"

echo "=== round-trip: shipped parts vs working files ==="
cat data/release/sft/v5_sft.jsonl.gz.part* | gunzip > /tmp/rt_sft.jsonl
if cmp -s /tmp/rt_sft.jsonl data/v5_sft.jsonl; then
    echo "  PASS  train parts reproduce data/v5_sft.jsonl byte-for-byte"
else
    echo "  FAIL  train parts do not match"
fi
gunzip -c data/release/sft/v5_eval.jsonl.gz > /tmp/rt_eval.jsonl
if cmp -s /tmp/rt_eval.jsonl data/v5_eval.jsonl; then
    echo "  PASS  eval archive reproduces data/v5_eval.jsonl byte-for-byte"
else
    echo "  FAIL  eval archive does not match"
fi
rm -f /tmp/rt_sft.jsonl /tmp/rt_eval.jsonl

echo
echo "=== train/eval disjointness (by message hash, incl. upsampled duplicates) ==="
"$PY" - <<'PY'
import hashlib, json
def h(rec):
    return hashlib.sha1(json.dumps(rec.get("messages"), sort_keys=True).encode()).hexdigest()
ev = set()
for line in open("data/v5_eval.jsonl", errors="ignore"):
    if line.strip():
        ev.add(h(json.loads(line)))
n = overlap = 0
for line in open("data/v5_sft.jsonl", errors="ignore"):
    if not line.strip():
        continue
    n += 1
    if h(json.loads(line)) in ev:
        overlap += 1
print(f"  train {n:,}   eval {len(ev):,}   overlap {overlap}")
print("  PASS  eval is genuinely held out" if overlap == 0 else "  FAIL  leakage")
PY

echo
echo "=== correctness gates: TRAIN ==="
# Full output, not tail: the per-gate PASS/FAIL lines are the record, and
# truncating them left only the aggregate VERDICT visible.
"$PY" scripts/v5_verify.py 2>&1

echo
echo "=== correctness gates: EVAL ==="
"$PY" scripts/v5_verify.py --path data/v5_eval.jsonl 2>&1

echo
echo "=== DONE_ALL ==="
