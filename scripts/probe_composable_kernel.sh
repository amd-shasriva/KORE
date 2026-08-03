#!/bin/bash
# Is Composable Kernel usable as an OUTPUT language on the nodes KORE runs on?
#
# The question is not whether CK is good. It is whether a kernel the model emits
# as CK C++ could be compiled by the verifier that scores it. If the headers are
# not installed on the episode-generating nodes, then training the model to emit
# CK teaches it to answer with code the harness cannot build -- the same
# negative transfer that (correctly) kept HipKittens out of the corpus as kernel
# bodies, recorded in docs/HIPKITTENS_INGEST.md.
#
# Emits JSON so the verdict is data rather than prose in a log.
#   bash scripts/probe_composable_kernel.sh [OUT.json]

set -uo pipefail
OUT="${1:-/dev/stdout}"

rocm_root="${ROCM_PATH:-/opt/rocm}"
hipcc_path="$(command -v hipcc 2>/dev/null || true)"
rocm_version="$(cat "$rocm_root/.info/version" 2>/dev/null | head -1 || true)"

ck_include=""
ck_tile_include=""
for cand in "$rocm_root/include/ck" /usr/include/ck /usr/local/include/ck; do
  [[ -d "$cand" ]] && { ck_include="$cand"; break; }
done
for cand in "$rocm_root/include/ck_tile" /usr/include/ck_tile /usr/local/include/ck_tile; do
  [[ -d "$cand" ]] && { ck_tile_include="$cand"; break; }
done

ck_headers=0
[[ -n "$ck_include" ]] && ck_headers=$(find "$ck_include" -name '*.hpp' 2>/dev/null | wc -l)
ck_tile_headers=0
[[ -n "$ck_tile_include" ]] && ck_tile_headers=$(find "$ck_tile_include" -name '*.hpp' 2>/dev/null | wc -l)

# CK also ships as a CMake package and/or a python wheel; check both, because
# "headers absent" and "CK entirely absent" are different findings.
ck_cmake="$(ls -d "$rocm_root"/lib/cmake/composable_kernel 2>/dev/null | head -1 || true)"
ck_pip="$(python -c 'import importlib.util as u; print(bool(u.find_spec("ck4inductor")))' 2>/dev/null || echo "unknown")"

# The decisive test: does a minimal CK translation unit actually compile here?
compile_ok=false
compile_err=""
if [[ -n "$hipcc_path" && -n "$ck_include" ]]; then
  tmp="$(mktemp -d)"
  cat > "$tmp/probe.cpp" <<'EOF'
#include <ck/ck.hpp>
#include <ck/utility/data_type.hpp>
int main() { return 0; }
EOF
  if hipcc --offload-arch=gfx950 -std=c++17 -I"$rocm_root/include" \
      -c "$tmp/probe.cpp" -o "$tmp/probe.o" 2> "$tmp/err"; then
    compile_ok=true
  else
    compile_err="$(head -c 400 "$tmp/err" | tr '\n' ' ' | sed 's/"/\\"/g')"
  fi
  rm -rf "$tmp"
else
  compile_err="skipped: hipcc=${hipcc_path:-absent} ck_include=${ck_include:-absent}"
fi

cat > "$OUT" <<EOF
{
  "host": "$(hostname)",
  "rocm_root": "$rocm_root",
  "rocm_version": "$rocm_version",
  "hipcc": "${hipcc_path:-absent}",
  "ck_include_dir": "${ck_include:-absent}",
  "ck_header_count": $ck_headers,
  "ck_tile_include_dir": "${ck_tile_include:-absent}",
  "ck_tile_header_count": $ck_tile_headers,
  "ck_cmake_package": "${ck_cmake:-absent}",
  "ck4inductor_importable": "$ck_pip",
  "minimal_ck_translation_unit_compiles": $compile_ok,
  "compile_error": "$compile_err"
}
EOF
echo "--- CK probe ---"
cat "$OUT"
