#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib/ops_runtime.sh
source "$REPO/scripts/lib/ops_runtime.sh"
kore_deprecated_guard \
  "logs/reli.sh" \
  "use an isolated disposable work directory in a scheduler test job" \
  "bash logs/reli.sh [--dry-run]" \
  "$@"
kore_destructive_guard "logs/reli.sh"
PY="$(kore_resolve_python "$REPO")"
RUNTIME="$(kore_private_runtime)"
RUN_ID="${KORE_RUN_ID:-$(kore_new_run_id reli)}"
cd "$REPO"
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
kore_require_commands od stat
SUMMARY="logs/reli_summary_${RUN_ID}.txt"
for i in 1 2 3; do
  for target in "$REPO/runs_smoke/midtrain" "$REPO/data_smoke/launch" \
      "$REPO/data_smoke/campaign_manifest.json"; do
    case "$target" in
      "$REPO"/runs_smoke/*|"$REPO"/data_smoke/*) rm -rf -- "$target" ;;
      *) echo "ERROR: refusing out-of-scope deletion: $target" >&2; exit 65 ;;
    esac
  done
  LOG="logs/reli_${RUN_ID}_$i.log"
  set +e
  kore_owned_run "$PY" "$REPO" "$RUNTIME" "$RUN_ID" "reli-$i" "$LOG" \
    env HF_HUB_OFFLINE=1 KORE_LOG_COLOR=0 KORE_LOG_LEVEL=INFO \
    KORE_RUN_DIR=logs/smoke PYTHONPATH="$REPO" \
    "$PY" scripts/run_campaign.py --model Qwen/Qwen3-14B \
    --tasks rmsnorm_aiter,gemm_bf16 \
    --full-ft --stages midtrain --data-root data_smoke --midtrain-out runs_smoke/midtrain \
  rc=$?
  set -e
  if (( rc == 0 )); then
    kore_verify "$PY" "$REPO" campaign --repo "$REPO" \
      --data-root data_smoke --required-stages midtrain || rc=$?
  fi
  rc=$?
  shards="$(printf '%s\n' runs_smoke/midtrain/*.safetensors | grep -vc '\*' || true)"
  warns="$(grep -c "not set to 'flash_attention_2'" "$LOG" || true)"
  ckerr="$(grep -c "CheckpointError" "$LOG" || true)"
  echo "run $i: exit=$rc shards=$shards sdpa_warns=$warns checkpoint_errors=$ckerr" >> "$SUMMARY"
  (( rc == 0 )) || exit "$rc"
done
echo "VERIFIED COMPLETE" >> "$SUMMARY"
