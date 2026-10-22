#!/bin/bash
#SBATCH --job-name=mv1_eval
#SBATCH --output=job-%A_%a.out
#SBATCH --error=job-%A_%a.err
#SBATCH --partition=gpu-m
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=1-4

set -euo pipefail
export PYTHONUNBUFFERED=1
module load uv/0.11.2
module load cuda/12.9

DEVICE=cuda
HPC_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f "$HPC_DIR/config.sh" ]; then
    echo "ERROR: $HPC_DIR/config.sh nie istnieje. Pod SLURM-em ten skrypt jest kopiowany do spool, wiec BASH_SOURCE nie wskazuje repo - uruchom 'sbatch' z katalogu experiments/hpc/ (zob. README.md)." >&2
    exit 1
fi
source "$HPC_DIR/config.sh"

export UV_PROJECT_ENVIRONMENT="$SLURM_SUBMIT_DIR/venv-eval-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
uv sync --project "$ROOT/ml" $UV_SYNC_EXTRA_ARGS

bash "$HPC_DIR/tasks/eval_zero_shot.sh"
