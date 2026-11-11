#!/bin/bash
# Static lint for uv invocations, in the style of check_sbatch_paths.sh.
#
# Rules (see README.md, section "Kolejnosc uruchamiania"):
#   - every `uv run` in experiments/hpc/**/*.sh has --no-sync; a plain
#     `uv run` re-syncs the venv to the default groups (dev + cpu) and swaps
#     the cu128 torch for the CPU wheel (job 94575),
#   - every `uv sync` in experiments/hpc/**/*.sh and in the Dockerfile has
#     --frozen, so uv.lock is installed as is and never re-resolved,
#   - the Dockerfile sets ENV UV_NO_SYNC=1.
#
# Comment lines and `echo` lines are ignored (they only mention the command).
#
# Usage: ./check_uv_flags.sh   (exits 1 and prints offending lines on failure)
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HPC_DIR/../.." && pwd)"
fail=0

check_file() {
    local f="$1" lineno line code
    while IFS=: read -r lineno line; do
        code="${line#"${line%%[![:space:]]*}"}"
        [[ "$code" == \#* || "$code" == echo* ]] && continue
        if [[ "$code" == *"uv run"* && "$code" != *"--no-sync"* ]]; then
            echo "${f#"$ROOT"/}:${lineno}: \`uv run\` without --no-sync"
            fail=1
        fi
        if [[ "$code" == *"uv sync"* && "$code" != *"--frozen"* ]]; then
            echo "${f#"$ROOT"/}:${lineno}: \`uv sync\` without --frozen"
            fail=1
        fi
    done < <(grep -n "uv run\|uv sync" "$f" || true)
}

while IFS= read -r -d '' f; do
    check_file "$f"
done < <(find "$HPC_DIR" -name "*.sh" ! -name "check_uv_flags.sh" -print0)

check_file "$ROOT/Dockerfile"
if ! grep -q "^ENV UV_NO_SYNC=1" "$ROOT/Dockerfile"; then
    echo "Dockerfile: missing ENV UV_NO_SYNC=1"
    fail=1
fi

if [ "$fail" -eq 1 ]; then
    echo "FAIL: uv invocation(s) above break the pinned-environment rules." >&2
    exit 1
fi
echo "OK: every uv run has --no-sync, every uv sync has --frozen, Dockerfile sets UV_NO_SYNC=1"
