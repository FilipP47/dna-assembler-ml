#!/bin/bash
#SBATCH --job-name=groupA_gen
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=1-100%10

set -euo pipefail
source $HOME/.cargo/env
cd /scratch/fpietrew/dna-assembler

ACC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" data/genomes_ecoli/full_pool_100.txt)

echo "=== Task ${SLURM_ARRAY_TASK_ID}: $ACC ==="
time ./target/release/dna-assembler pipeline \
    --reference "data/genomes_ecoli/fasta/${ACC}.fna" \
    --k 21 \
    --coverage 100 \
    --gen-train-data \
    --results-dir "results/group_a/${ACC}"
