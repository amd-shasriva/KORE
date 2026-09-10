#!/usr/bin/env bash
# Restore the upstream datagen and evaluation artifacts from their release parts.
#
# reassemble.sh rebuilds the TRAINING CORPUS -- the thing you need to run SFT.
# This script rebuilds everything UPSTREAM of it: the raw agentic episodes the
# corpus was distilled from, the intermediate v5 build slices, the
# contamination quarantines, the 14B campaign, and the AgentKernelArena
# evaluation runs behind the published scores. You do not need any of it to
# train on the shipped corpus. You do need it to rebuild a corpus differently,
# or to audit where a training row came from.
#
# Each bundle is a tar.gz split into sub-100 MB parts, because GitHub rejects
# files at or above 100 MB. Every bundle carries a manifest with the SHA-256 of
# the reassembled stream, and this script checks it before extracting -- a
# truncated part is otherwise a silent corruption that only shows up much later
# as unparseable JSON.
#
# Usage:
#   ./restore_upstream.sh                 # restore everything, into ./upstream/
#   ./restore_upstream.sh DEST            # restore everything into DEST
#   ./restore_upstream.sh DEST episodes   # restore only bundles matching a prefix
#
# Roughly 941 MB of parts expand to about 10 GB. Check free space first.

set -euo pipefail
cd "$(dirname "$0")"

DEST="${1:-upstream}"
FILTER="${2:-}"

command -v sha256sum >/dev/null || { echo "need sha256sum" >&2; exit 1; }

mkdir -p "$DEST"
echo "restoring into: $DEST"
[ -n "$FILTER" ] && echo "filter: $FILTER"

restored=0
skipped=0

# Bundle manifests live one directory down, e.g. episodes/agentic_v2.manifest.json
for manifest in */*.manifest.json; do
    [ -e "$manifest" ] || continue
    dir="$(dirname "$manifest")"
    stem="$(basename "$manifest" .manifest.json)"
    bundle="$dir/$stem"

    # multicap_v4.manifest.json belongs to the corpus, not to a tar bundle.
    ls "$bundle".tar.gz.part* >/dev/null 2>&1 || continue

    if [ -n "$FILTER" ] && case "$bundle" in *"$FILTER"*) false;; *) true;; esac; then
        skipped=$((skipped + 1))
        continue
    fi

    expected="$(sed -n 's/.*"sha256_of_reassembled_tar_gz": "\([0-9a-f]*\)".*/\1/p' "$manifest")"
    if [ -z "$expected" ]; then
        echo "!! $bundle: manifest has no digest, refusing to extract" >&2
        exit 1
    fi

    echo "  $bundle"
    actual="$(cat "$bundle".tar.gz.part* | sha256sum | cut -d' ' -f1)"
    if [ "$actual" != "$expected" ]; then
        echo "!! $bundle: digest mismatch" >&2
        echo "   expected $expected" >&2
        echo "   actual   $actual" >&2
        echo "   A part is missing or truncated. Re-fetch before using this." >&2
        exit 1
    fi

    target="$DEST/$dir"
    mkdir -p "$target"
    cat "$bundle".tar.gz.part* | tar -C "$target" -xzf -
    restored=$((restored + 1))
done

echo
echo "restored $restored bundle(s)${FILTER:+, skipped $skipped by filter}"
echo "note: vendored aiter subtrees and compiled objects were excluded from the"
echo "arena workspaces as rebuildable; generated kernels and eval_result.yaml"
echo "are present."
