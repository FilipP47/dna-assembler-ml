#!/bin/bash
#SBATCH --job-name=dna_gen_train_data
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f "$HPC_DIR/config.sh" ]; then
    echo "ERROR: $HPC_DIR/config.sh nie istnieje. Pod SLURM-em ten skrypt jest kopiowany do spool, wiec BASH_SOURCE nie wskazuje repo - uruchom 'sbatch' z katalogu experiments/hpc/ (zob. README.md)." >&2
    exit 1
fi
source "$HPC_DIR/config.sh"

cd "$ROOT"
cargo build --release

./target/release/dna-assembler pipeline \
  --reference genomes/cerevisiae.fna \
  --k 21 \
  --coverage 100 \
  --gen-train-data \
  --results-dir results/cerevisiae_k21
