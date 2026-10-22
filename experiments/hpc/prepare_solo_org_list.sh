#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$ROOT"

for grp in a b; do
    outdir="experiments/hpc/data/genomes_ecoli"
    [ "$grp" = "b" ] && outdir="experiments/hpc/data/genomes_enterobacteriaceae"

    cat "${outdir}/output/test_farthest_10.txt" "${outdir}/output/test_random_10.txt" \
        | sort -u > "${outdir}/solo_organisms.txt"

    n=$(wc -l < "${outdir}/solo_organisms.txt")
    echo "Grupa ${grp^^}: ${n} unikalnych organizmow held-out do solo baseline"
    cat "${outdir}/solo_organisms.txt"
    echo ""
done
