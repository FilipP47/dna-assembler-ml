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
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f "$HPC_DIR/config.sh" ]; then
    echo "ERROR: $HPC_DIR/config.sh nie istnieje. Pod SLURM-em ten skrypt jest kopiowany do spool, wiec BASH_SOURCE nie wskazuje repo - uruchom 'sbatch' z katalogu experiments/hpc/ (zob. README.md)." >&2
    exit 1
fi
source "$HPC_DIR/config.sh"

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
