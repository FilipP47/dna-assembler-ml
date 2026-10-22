#!/bin/bash
#SBATCH --job-name=dna_build
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00

# Build the dna-assembler release binary. Needed before any of the
# gen_group_a/gen_group_b tasks (they call ./target/release/dna-assembler
# directly and don't build it themselves). Safe to run both under sbatch
# and directly (./build.sh) in a local/Docker shell.
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
