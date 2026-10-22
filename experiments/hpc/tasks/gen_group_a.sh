#!/bin/bash
# Task body: generate training-data CSV for one E. coli (group A) genome.
# Portable: called by groupA_full_gen_array.sh (SLURM) and
# run_local.sh gen_group_a 100. Requires the Rust binary already built
# (cargo build --release).
#
# Requires: ROOT set, SLURM_ARRAY_TASK_ID is a 1-based line number into
# data/genomes_ecoli/full_pool_100.txt.
set -euo pipefail

POOL_FILE="$ROOT/experiments/hpc/data/genomes_ecoli/full_pool_100.txt"
ACC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$POOL_FILE")
if [ -z "$ACC" ]; then
    echo "gen_group_a.sh: no line ${SLURM_ARRAY_TASK_ID} in $POOL_FILE" >&2
    exit 1
fi

echo "=== Task ${SLURM_ARRAY_TASK_ID}: $ACC ==="
time "$ROOT/target/release/dna-assembler" pipeline \
    --reference "$ROOT/experiments/hpc/data/genomes_ecoli/fasta/${ACC}.fna" \
    --k 21 \
    --coverage 100 \
    --gen-train-data \
    --train-data-dir "$ROOT/data/training" \
    --results-dir "$ROOT/results/group_a/${ACC}"
