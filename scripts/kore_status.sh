#!/bin/bash
# One page describing whether KORE is actually working right now.
#
# The failure this is aimed at is not a crash. It is a fleet that looks busy --
# jobs running, logs growing -- while producing nothing, which is what 96 empty
# teacher calls looked like from the queue. So every section here reports work
# produced, not work scheduled.
set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
PY="${KORE_PY:-/home/shasriva/kore-venv/bin/python}"
[ -z "${SPUR_CONTROLLER_ADDR:-}" ] && [ -r /etc/profile.d/spur.sh ] && . /etc/profile.d/spur.sh
cd "$REPO" || exit 1

echo "======== KORE status $(date -u '+%Y-%m-%d %H:%M:%S UTC') ========"

echo "-- queue"
if q=$(squeue -u "$USER" -o '%.7i %.26j %.9T %.16N %.9M' 2>/dev/null); then
    sed 's/^/   /' <<<"$q"
    r=$(squeue -u "$USER" -h -t R -o '%i' 2>/dev/null | grep -c .)
    p=$(squeue -u "$USER" -h -t PD -o '%i' 2>/dev/null | grep -c .)
    echo "   running=$r pending=$p cap=${GPU_JOB_CAP:-8}"
else
    echo "   CONTROLLER UNREACHABLE (jobs keep running; nothing can be submitted)"
fi

echo "-- daemons"
for d in frontier_pipeline node_scavenger flyseed_watch supervise; do
    n=$(pgrep -fc "$d" 2>/dev/null || echo 0)
    printf "   %-20s %s\n" "$d" "$([ "$n" -gt 0 ] && echo "up ($n proc)" || echo "DOWN")"
done

echo "-- data produced in the last 15 minutes"
for root in v5frontierhip v5frontier; do
    n=$(find "data/$root" -name '*.jsonl' -newermt '-15 minutes' 2>/dev/null | grep -c .)
    printf "   %-18s %s file(s) written\n" "$root" "$n"
done

echo "-- mined trainable tokens"
"$PY" - <<'PY'
import glob, json
PAD = 2000
for root in ("v5frontierhip", "v5frontier"):
    tok = 0.0
    tasks = set()
    for f in glob.glob(f"data/{root}/**/*.jsonl", recursive=True):
        for line in open(f, errors="ignore"):
            if len(line) < 2:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            t = r.get("type")
            if t == "ranked_group":
                srcs = [c.get("source") for c in (r.get("candidates") or [])
                        if isinstance(c, dict) and isinstance(c.get("source"), str)]
                if srcs:
                    tok += (sum(len(s) for s in srcs) + PAD * len(srcs)) / 4
                    tasks.add(r.get("task_id"))
            elif t in ("repair", "win"):
                msgs = r.get("messages")
                if isinstance(msgs, list):
                    n = sum(len(m.get("content", "")) for m in msgs if isinstance(m, dict))
                    if n:
                        tok += n / 4
                        tasks.add(r.get("task_id"))
    print(f"   {root:<18} {tok/1e6:7.2f}M tokens over {len(tasks)} tasks")
PY

echo "-- arena v4"
n=$(cat runs/aka_full_v4/*.partial.jsonl 2>/dev/null | grep -c . || echo 0)
echo "   $n/413 rows"
now=$(date +%s)
for f in runs/aka_full_v4/run_shard*.log; do
    [ -f "$f" ] || continue
    age=$(( (now - $(stat -c %Y "$f")) / 60 ))
    prog=$(grep -oE '^\[[0-9]+/[0-9]+\]' "$f" | tail -1)
    done_marker=$(grep -c 'finished its slice' "$f")
    state=$([ "$done_marker" -gt 0 ] && echo "done" || echo "${age}min idle")
    printf "   %-10s %-10s %s\n" "$(basename "$f" .log)" "${prog:-?}" "$state"
done

echo "-- verified FlyDSL seeding"
"$PY" - <<'PY'
import collections, json, os
p = "data/registry_flydsl_frontier/verified_seed_attempts.jsonl"
if not os.path.exists(p):
    print("   no ledger yet")
else:
    rows = []
    for line in open(p, errors="ignore"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    # The rows written before the thinking fix are all the same failure and
    # would otherwise dominate every count printed here for the rest of the run.
    new = [r for r in rows if "empty_replies" in r or r.get("status") == "pass"]
    old = len(rows) - len(new)
    n = len(new) or 1
    c = collections.Counter(r.get("status") for r in new)
    print(f"   {len(new)} judged since the fix: {dict(c)}  "
          f"pass rate {100*c['pass']/n:.1f}%   ({old} older rows ignored)")
    by = collections.Counter(r.get("attempt") for r in new if r.get("status") == "pass")
    if by:
        print(f"   passes by attempt: {dict(sorted(by.items()))}"
              f"   (>1 means the verifier feedback earned it)")
    empt = sum(r.get("empty_replies", 0) or 0 for r in new)
    if empt:
        print(f"   {empt} empty repl(ies) -- watch for the output budget again")
    errs = collections.Counter()
    for r in new:
        if r.get("status") == "failed":
            errs[(r.get("error") or "?")[:70]] += 1
    for e, k in errs.most_common(5):
        print(f"     {k:>3}x {e}")
PY
echo "-- passing flydsl twins on disk"
echo "   $(ls data/frontier_twins_ok/tasks 2>/dev/null | grep -c flydsl) promoted"
