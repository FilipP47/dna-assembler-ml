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
source "$HOME/.cargo/env"
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

bash "$HPC_DIR/tasks/gen_group_a.sh"
