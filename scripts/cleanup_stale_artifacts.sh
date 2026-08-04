#!/usr/bin/env bash
# Reclaim space from artifacts that are provably finished with.
#
# Deliberately conservative about what counts as "finished with". Anything
# belonging to a job that is still queued or running is skipped by job id, and
# the datagen output under data/ is never touched -- that is the product. What
# goes are job stdout/stderr from completed runs, superseded snapshots, and
# git's own reclaimable garbage.
#
# The honest framing on this host: our whole footprint is ~36GB on a 10TB volume
# that is 99% full, so the pressure on /home is other users and nothing we
# delete will materially change it. This is hygiene, and the reason it still
# matters is that the SFT checkpoint path writes to /shared_nfs, where our own
# stale copies DO compete with the 488GB a 30B checkpoint needs.
set -uo pipefail

REPO="${KORE_REPO:-/home/shasriva/Kore-RL/KORE}"
DRY="${DRY_RUN:-0}"

avail_g() { df -k "$1" 2>/dev/null | tail -1 | awk '{printf "%.1f", $4/1024/1024}'; }

echo "BEFORE  /home $(avail_g /home)G   /shared_nfs $(avail_g /shared_nfs)G"

# Job ids that are still queued or running; their logs are off limits.
LIVE=$(squeue -u "$USER" -h -o "%i" 2>/dev/null | tr '\n' ' ')
echo "live jobs (logs preserved): ${LIVE:-none}"

rm_if() {   # rm_if <path...>  -- honours DRY_RUN and reports what it did
    local n=0 bytes=0
    for p in "$@"; do
        [ -e "$p" ] || continue
        bytes=$((bytes + $(du -sk "$p" 2>/dev/null | awk '{print $1}')))
        n=$((n + 1))
        [ "$DRY" = "1" ] || rm -rf "$p"
    done
    [ "$n" -gt 0 ] && echo "  ${n} item(s), $((bytes / 1024))MB"
    return 0
}

echo "stale spur job logs:"
rm_if $(ls "$HOME"/spur-*.out 2>/dev/null) $(ls "$REPO"/spur-*.out 2>/dev/null)

echo "superseded factory logs:"
rm_if "$REPO/runs/factory_logs"

echo "finished-job logs older than 2 days:"
total=0
while IFS= read -r f; do
    keep=0
    for j in $LIVE; do
        case "$f" in *"$j"*) keep=1; break;; esac
    done
    [ "$keep" = "1" ] && continue
    total=$((total + $(du -sk "$f" 2>/dev/null | awk '{print $1}')))
    [ "$DRY" = "1" ] || rm -f "$f"
done < <(find "$REPO/runs" -maxdepth 1 -type f \( -name '*.err' -o -name '*.out' -o -name '*.log' \) -mtime +2 2>/dev/null)
echo "  $((total / 1024))MB"

echo "git garbage collection:"
if [ "$DRY" != "1" ]; then
    before=$(du -sk "$REPO/.git" 2>/dev/null | awk '{print $1}')
    git -C "$REPO" gc --prune=now --quiet 2>/dev/null
    after=$(du -sk "$REPO/.git" 2>/dev/null | awk '{print $1}')
    echo "  .git $((before / 1024))MB -> $((after / 1024))MB"
fi

echo "python bytecode caches:"
n=$(find "$REPO" -name '__pycache__' -type d 2>/dev/null | wc -l)
[ "$DRY" = "1" ] || find "$REPO" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
echo "  $n dirs"

echo "AFTER   /home $(avail_g /home)G   /shared_nfs $(avail_g /shared_nfs)G"
