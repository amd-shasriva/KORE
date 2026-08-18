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
# NOT need this launcher - the single-process path handles them.
#
# Each stage runs `python -m kore.policy.<stage> <config.json>`, which must read
# a JSON config positional. sft/dpo/grpo ship that entrypoint (grpo via
# `grpo_config_from_dict` + `__main__`), so `--full-ft` shells each out here for
# real full-parameter sharded (ZeRO-3/FSDP) training. midtrain is owned by a
# sibling track and gains it when its `-m` JSON entry lands (until then the
# campaign runs midtrain in-process with a LOUD warning - see
# docs/DISTRIBUTED.md#full-ft-per-stage-status). The launcher accepts all four so
# the plumbing is ready the moment any remaining entrypoint ships.
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
# GRPO does in-loop generation and MUST use SHARD_GRAD_OP (ZeRO-2) so params stay
# replicated between forwards (local generate, no per-decode all_gather deadlock).
# SFT/DPO/midtrain have no generation, so FULL_SHARD (ZeRO-3) is fine + leaner.
if [ "$STAGE" = "grpo" ]; then
  ACCEL_CONFIG="$REPO_ROOT/configs/accelerate_fsdp_grpo.yaml"
else
  ACCEL_CONFIG="$REPO_ROOT/configs/accelerate_fsdp.yaml"
fi
# KORE_ACCEL_CONFIG replaces the stage default. The multi-node GRPO launcher
# needs configs/accelerate_fsdp_grpo_2node.yaml, which differs in the one place
# that cannot be expressed on the command line: fsdp_sharding_strategy. Under
# FSDP1 accelerate wraps with `sharding_strategy or reshard_after_forward` and
# ALWAYS exports FSDP_SHARDING_STRATEGY (argparse default "FULL_SHARD"), so the
# strategy is decided by the YAML, and a 2-node run that inherited the
# single-node YAML would silently run the wrong topology.
if [ -n "${KORE_ACCEL_CONFIG:-}" ]; then
  ACCEL_CONFIG="$KORE_ACCEL_CONFIG"
fi

# Build the accelerate command. Passing PYTHONPATH keeps `-m kore.policy.<stage>`
# importable without an editable install.
ACCEL_ARGS=("launch" "--config_file" "$ACCEL_CONFIG")
# GPU_IDS=1,2,4,5,6,7 pins the run to specific PHYSICAL GPUs (e.g. to dodge GPUs
# saturated by a neighbor's job on a shared node). Must go through accelerate's
# --gpu_ids: it is authoritative and sets the workers' CUDA_VISIBLE_DEVICES to
# exactly these ids, whereas a bare CUDA/ROCR_VISIBLE_DEVICES in the parent is
# overwritten by accelerate's own multi_gpu_launcher (workers -> 0..N-1).
if [ -n "${GPU_IDS:-}" ]; then
  ACCEL_ARGS+=("--gpu_ids" "$GPU_IDS")
  if [ -z "$NPROC" ]; then
    NPROC=$(printf '%s' "$GPU_IDS" | tr ',' '\n' | grep -c .)
  fi
fi
if [ -n "$NPROC" ]; then
  ACCEL_ARGS+=("--num_processes" "$NPROC")
fi

# ---- multi-node rendezvous (all four unset => single-node, unchanged) ---- #
# These come from the per-node srun task, not from the YAML, because three of
# them differ per node or per allocation. KORE_MAIN_IP must be a literal IP:
# compute nodes here cannot resolve each other by name (`hostname -f` returns
# localhost.localdomain and `scontrol show hostnames` is not implemented on this
# controller), so a hostname rendezvous fails with rank 0 exiting 1 and every
# other rank taking a SIGTERM.
if [ -n "${KORE_NUM_MACHINES:-}" ] && [ "${KORE_NUM_MACHINES}" != "1" ]; then
  : "${KORE_MACHINE_RANK:?KORE_MACHINE_RANK required when KORE_NUM_MACHINES>1}"
  : "${KORE_MAIN_IP:?KORE_MAIN_IP required when KORE_NUM_MACHINES>1 (literal IP, not a hostname)}"
  case "$KORE_MAIN_IP" in
    *[!0-9.]*) echo "error: KORE_MAIN_IP must be a literal IPv4 address, got '$KORE_MAIN_IP'" >&2; exit 1 ;;
  esac
  ACCEL_ARGS+=("--num_machines" "$KORE_NUM_MACHINES"
               "--machine_rank" "$KORE_MACHINE_RANK"
               "--main_process_ip" "$KORE_MAIN_IP"
               "--main_process_port" "${KORE_MAIN_PORT:-29577}")
fi
ACCEL_ARGS+=("-m" "kore.policy.$STAGE" "$CONFIG")

CMD=(accelerate "${ACCEL_ARGS[@]}")

# Validate BEFORE the dry-run exit, so --dry-run is a real preflight rather than
# an echo: a mistyped config path must fail here, not after an 8-rank 14B load.
if [ ! -f "$ACCEL_CONFIG" ]; then
  echo "error: accelerate config not found at $ACCEL_CONFIG" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "error: training config not found at $CONFIG" >&2
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "[launch_distributed] (dry-run) PYTHONPATH=$REPO_ROOT ${CMD[*]}"
  exit 0
fi

echo "[launch_distributed] stage=$STAGE config=$CONFIG accel=$ACCEL_CONFIG"
echo "[launch_distributed] PYTHONPATH=$REPO_ROOT ${CMD[*]}"
cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "${CMD[@]}"
