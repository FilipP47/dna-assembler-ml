#!/bin/bash
# Static lint: catches the job-91700 bug class before it hits the queue.
#
# Under sbatch, SLURM copies the submitted script into a spool directory
# (/var/spool/slurmd/job<ID>/...) before running it, so a script that
# resolves its own location via `dirname "${BASH_SOURCE[0]}"` no longer
# finds the repo - "No such file or directory" when it then tries to
# source a sibling file (job 91700, smoke_test_train.sh:17 at the time).
#
# The fix is $SLURM_SUBMIT_DIR as the primary source, with BASH_SOURCE
# only as a fallback for direct (non-sbatch) execution, e.g.:
#   HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
#
# So this check does NOT ban `dirname "${BASH_SOURCE` outright (the
# fallback above still needs it) - it flags any line in a #SBATCH script
# that uses that pattern WITHOUT also mentioning SLURM_SUBMIT_DIR on the
# same line, which is exactly the unguarded form that broke job 91700.
#
# Usage: ./check_sbatch_paths.sh   (exits 1 and prints offending lines on
# failure; intended to run in CI / before committing a new *_array.sh).
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail=0

while IFS= read -r -d '' f; do
    grep -q "^#SBATCH" "$f" || continue
    while IFS=: read -r lineno line; do
        if [[ "$line" == *'dirname "${BASH_SOURCE'* ]] && [[ "$line" != *'SLURM_SUBMIT_DIR'* ]]; then
            echo "${f#"$HPC_DIR"/}:${lineno}: unguarded \`dirname \"\${BASH_SOURCE...}\"\` in an sbatch script - breaks under SLURM (script is copied to spool, see job 91700). Add a \$SLURM_SUBMIT_DIR fallback." >&2
            fail=1
        fi
    done < <(grep -n "" "$f")
done < <(find "$HPC_DIR" -name "*.sh" -print0)

if [ "$fail" -eq 1 ]; then
    echo "FAIL: unguarded BASH_SOURCE-based path lookup found in sbatch script(s) above." >&2
    exit 1
fi
echo "OK: no unguarded dirname/BASH_SOURCE pattern in any #SBATCH script under $HPC_DIR"
