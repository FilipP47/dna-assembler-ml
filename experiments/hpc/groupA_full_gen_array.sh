#!/bin/bash
#SBATCH --job-name=groupA_gen
#SBATCH --output=gen_%A_%a.out
#SBATCH --error=gen_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=1-100%10

set -euo pipefail
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f "$HPC_DIR/config.sh" ]; then
    echo "ERROR: $HPC_DIR/config.sh nie istnieje. Pod SLURM-em ten skrypt jest kopiowany do spool, wiec BASH_SOURCE nie wskazuje repo - uruchom 'sbatch' z katalogu experiments/hpc/ (zob. README.md)." >&2
    exit 1
fi
source "$HPC_DIR/config.sh"

bash "$HPC_DIR/tasks/gen_group_a.sh"
