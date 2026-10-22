#!/bin/bash
#SBATCH --job-name=groupA_pilot_gen
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
source "$HOME/.cargo/env"
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

cd "$ROOT"
PILOT_ACCESSIONS=$(head -n 3 experiments/hpc/data/genomes_ecoli/output/train_90_for_farthest_split.txt)

for acc in $PILOT_ACCESSIONS; do
    echo "=== [$(date +%T)] Generowanie danych dla $acc ==="
    time ./target/release/dna-assembler pipeline \
        --reference "experiments/hpc/data/genomes_ecoli/fasta/${acc}.fna" \
        --k 21 \
        --coverage 100 \
        --gen-train-data \
        --train-data-dir data/training \
        --results-dir "results/group_a_pilot/${acc}"
done

echo ""
echo "--- Co i gdzie zostalo zapisane (do zweryfikowania) ---"
echo "Zawartosc results/group_a_pilot/:"
find results/group_a_pilot -type f
echo ""
echo "Zawartosc data/training/ (sprawdz czy per-genom, czy nadpisywane):"
find data/training -type f -newer experiments/hpc/data/genomes_ecoli/assembly_summary.txt 2>/dev/null || echo "(brak - sprawdz sciezke recznie)"
