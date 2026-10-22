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
source "$HOME/.cargo/env"
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

cd "$ROOT"
cargo build --release

./target/release/dna-assembler pipeline \
  --reference genomes/cerevisiae.fna \
  --k 21 \
  --coverage 100 \
  --gen-train-data \
  --results-dir results/cerevisiae_k21
