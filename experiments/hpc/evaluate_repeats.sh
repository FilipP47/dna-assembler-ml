#!/bin/bash
#SBATCH --job-name=mv1_eval_rep
#SBATCH --output=job-%A_%a.out
#SBATCH --error=job-%A_%a.err
#SBATCH --partition=gpu-m
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=1-12

set -euo pipefail
export PYTHONUNBUFFERED=1
module load uv/0.11.2
module load cuda/12.9

DEVICE=cuda
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

export UV_PROJECT_ENVIRONMENT="$SLURM_SUBMIT_DIR/venv-evalrep-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
uv sync --project "$ROOT/ml" $UV_SYNC_EXTRA_ARGS

bash "$HPC_DIR/tasks/eval_repeats.sh"
