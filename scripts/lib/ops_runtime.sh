#!/usr/bin/env bash
# Shared, side-effect-free-until-called helpers for legacy operational scripts.

kore_deprecated_guard() {
  local script="$1" migration="$2" usage="$3"
  shift 3
  local arg
  for arg in "$@"; do
    case "$arg" in
      -h|--help)
        printf 'Usage: %s\n\nDEPRECATED: %s\nMigration: %s\n' "$usage" "$script" "$migration"
        printf 'Production execution is disabled. Development-only override: KORE_ALLOW_DEPRECATED_DEV=1.\n'
        exit 0
        ;;
      --dry-run)
        printf 'DRY-RUN: %s is deprecated; no command, process, state, or dependency check was run.\n' "$script"
        printf 'Migration: %s\nDevelopment-only override: KORE_ALLOW_DEPRECATED_DEV=1.\n' "$migration"
        exit 0
        ;;
    esac
  done
  if [[ "${KORE_ALLOW_DEPRECATED_DEV:-}" != "1" ]]; then
    printf 'ERROR: %s is deprecated and disabled for production.\n' "$script" >&2
    printf 'Migration: %s\n' "$migration" >&2
    printf 'Development-only override: KORE_ALLOW_DEPRECATED_DEV=1.\n' >&2
    exit 64
  fi
  printf 'WARNING: development override enabled for deprecated %s\n' "$script" >&2
}

kore_destructive_guard() {
  local script="$1"
  if [[ "${KORE_ALLOW_DESTRUCTIVE_DEV:-}" != "1" ]]; then
    printf 'ERROR: %s performs destructive filesystem operations.\n' "$script" >&2
    printf 'Set KORE_ALLOW_DESTRUCTIVE_DEV=1 in addition to the deprecated-script override.\n' >&2
    exit 65
  fi
}

kore_require_commands() {
  local missing=0 command_name
  for command_name in "$@"; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
      printf 'ERROR: missing required command: %s\n' "$command_name" >&2
      missing=1
    fi
  done
  (( missing == 0 )) || return 127
}

kore_resolve_python() {
  local repo="$1" candidate
  for candidate in \
    "${KORE_PY:-}" \
    "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
    "$repo/.venv/bin/python" \
    "$HOME/kore-venv/bin/python"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  candidate="$(command -v python3 2>/dev/null || true)"
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  printf 'ERROR: no usable Python interpreter; set KORE_PY or activate a venv.\n' >&2
  return 127
}

kore_private_runtime() {
  local base mode owner
  if [[ -n "${KORE_RUNTIME_DIR:-}" ]]; then
    base="$KORE_RUNTIME_DIR"
  elif [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
    base="$XDG_RUNTIME_DIR/kore-ops"
  else
    base="${TMPDIR:-/tmp}/kore-ops-$(id -u)"
  fi
  umask 077
  if [[ -L "$base" ]]; then
    printf 'ERROR: runtime directory must not be a symlink: %s\n' "$base" >&2
    return 74
  fi
  mkdir -m 0700 -p -- "$base" || return
  owner="$(stat -c '%u' -- "$base")" || return
  mode="$(stat -c '%a' -- "$base")" || return
  if [[ "$owner" != "$(id -u)" || $((8#$mode & 077)) -ne 0 ]]; then
    printf 'ERROR: runtime directory must be owned by this uid and private: %s (uid=%s mode=%s)\n' \
      "$base" "$owner" "$mode" >&2
    return 74
  fi
  chmod 0700 -- "$base"
  printf '%s\n' "$base"
}

kore_secure_source_env() {
  local env_file="$1" owner mode
  [[ -e "$env_file" ]] || return 0
  if [[ -L "$env_file" || ! -f "$env_file" ]]; then
    printf 'ERROR: refusing non-regular environment file: %s\n' "$env_file" >&2
    return 74
  fi
  owner="$(stat -c '%u' -- "$env_file")" || return
  mode="$(stat -c '%a' -- "$env_file")" || return
  if [[ "$owner" != "$(id -u)" || $((8#$mode & 022)) -ne 0 ]]; then
    printf 'ERROR: environment file must be owned by this uid and not writable by group/other: %s\n' \
      "$env_file" >&2
    return 74
  fi
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
}

kore_new_run_id() {
  local prefix="${1:-run}" suffix
  prefix="${prefix//[^a-zA-Z0-9-]/-}"
  suffix="$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
  printf '%s-%s-%s\n' "${prefix,,}" "$(date -u +%Y%m%dT%H%M%SZ)" "$suffix"
}

kore_export_rigor_env() {
  export KORE_VERIFIED_CORRECTNESS="${KORE_VERIFIED_CORRECTNESS:-1}"
  export KORE_COMPILE_BASELINE="${KORE_COMPILE_BASELINE:-1}"
  export KORE_SHAPE_AUGMENT="${KORE_SHAPE_AUGMENT:-1}"
  export KORE_BENCH_COLD="${KORE_BENCH_COLD:-1}"
}

kore_owned_run() {
  local python="$1" repo="$2" runtime="$3" run_id="$4" name="$5" log="$6"
  shift 6
  PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" \
    "$python" -m kore.ops run \
      --runtime-dir "$runtime" \
      --run-id "$run_id" \
      --name "$name" \
      --cwd "$repo" \
      --log "$log" \
      -- "$@"
}

kore_verify() {
  local python="$1" repo="$2"
  shift 2
  PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" "$python" -m kore.ops verify "$@"
}
