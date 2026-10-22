#!/bin/bash
#SBATCH --job-name=mv1_repeat
#SBATCH --output=job-%A_%a.out
#SBATCH --error=job-%A_%a.err
#SBATCH --partition=gpu-m
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=1-12%6

set -euo pipefail
export PYTHONUNBUFFERED=1
module load uv/0.11.2
module load cuda/12.9

DEVICE=cuda
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

export UV_PROJECT_ENVIRONMENT="$SLURM_SUBMIT_DIR/venv-repeat-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
uv sync --project "$ROOT/ml" $UV_SYNC_EXTRA_ARGS

bash "$HPC_DIR/tasks/train_repeat.sh"
