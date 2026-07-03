#!/usr/bin/env bash
# KORE distributed full fine-tuning launcher (FSDP full-shard / ZeRO-3).
#
# Usage:
#   scripts/launch_distributed.sh <stage: midtrain|sft|dpo|grpo> <config.json> [--nproc N] [--dry-run]
#
# Examples:
#   scripts/launch_distributed.sh midtrain configs/midtrain_14b_full.json
#   scripts/launch_distributed.sh sft configs/sft_14b_full.json
#   scripts/launch_distributed.sh dpo configs/dpo_14b_full.json --nproc 8
#   scripts/launch_distributed.sh grpo configs/grpo_14b_full.json
#   scripts/launch_distributed.sh sft configs/sft_14b_full.json --dry-run   # print cmd only
#
# The campaign (scripts/run_campaign.py) shells out to this launcher UNDER THE
# HOOD when the user passes --full-ft, so a full-FT run stays ONE command.
#
# The <config.json> is a flat map of the stage's Config fields (see
# docs/DISTRIBUTED.md). It should have `use_lora: false` for real full-FT; each
# stage entrypoint defaults `distributed: true` so FSDP kicks in. LoRA runs do
# NOT need this launcher — the single-process path handles them.
#
# Each stage runs `python -m kore.policy.<stage> <config.json>`, which must read
# a JSON config positional. sft/dpo ship that entrypoint today; grpo/midtrain are
# owned by a sibling track and gain it when their `-m` JSON entry lands (until
# then the campaign runs those stages in-process with a LOUD warning — see
# docs/DISTRIBUTED.md#full-ft-per-stage-status). The launcher accepts all four so
# the plumbing is ready the moment those entrypoints ship.
#
# --dry-run (or DRY_RUN=1) prints the accelerate command WITHOUT executing it,
# which is what CI / the test-suite syntax check uses.
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") <stage: midtrain|sft|dpo|grpo> <config.json> [--nproc N] [--dry-run]" >&2
  exit 2
}

STAGE="${1:-}"
CONFIG="${2:-}"
[ -z "$STAGE" ] && usage
[ -z "$CONFIG" ] && usage
shift 2 || usage

case "$STAGE" in
  midtrain|sft|dpo|grpo) ;;
  *) echo "error: stage must be one of midtrain|sft|dpo|grpo (got '$STAGE')" >&2; usage ;;
esac

NPROC=""
DRY_RUN="${DRY_RUN:-0}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --nproc) NPROC="${2:-}"; shift 2 ;;
    --nproc=*) NPROC="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "error: unknown arg '$1'" >&2; usage ;;
  esac
done

# Repo root = parent of scripts/ (the package root that holds `kore/`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ACCEL_CONFIG="$REPO_ROOT/configs/accelerate_fsdp.yaml"

# Build the accelerate command. Passing PYTHONPATH keeps `-m kore.policy.<stage>`
# importable without an editable install.
ACCEL_ARGS=("launch" "--config_file" "$ACCEL_CONFIG")
if [ -n "$NPROC" ]; then
  ACCEL_ARGS+=("--num_processes" "$NPROC")
fi
ACCEL_ARGS+=("-m" "kore.policy.$STAGE" "$CONFIG")

CMD=(accelerate "${ACCEL_ARGS[@]}")

if [ "$DRY_RUN" = "1" ]; then
  echo "[launch_distributed] (dry-run) PYTHONPATH=$REPO_ROOT ${CMD[*]}"
  exit 0
fi

if [ ! -f "$ACCEL_CONFIG" ]; then
  echo "error: accelerate config not found at $ACCEL_CONFIG" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "error: training config not found at $CONFIG" >&2
  exit 1
fi

echo "[launch_distributed] stage=$STAGE config=$CONFIG accel=$ACCEL_CONFIG"
echo "[launch_distributed] PYTHONPATH=$REPO_ROOT ${CMD[*]}"
cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "${CMD[@]}"
