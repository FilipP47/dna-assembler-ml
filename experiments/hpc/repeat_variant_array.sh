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

cd /scratch/fpietrew/dna-assembler/ml

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../repeat_variant_tasks.txt)
VARIANT=$(echo "$LINE" | cut -d, -f1)
DATA_PATH=$(echo "$LINE" | cut -d, -f2)
SEED=$(echo "$LINE" | cut -d, -f3)
RUN_NAME="${VARIANT}_seed${SEED}"

echo "[$(date +%T)] ${RUN_NAME} (data=${DATA_PATH}, seed=${SEED})"

VENV_DIR=$SLURM_SUBMIT_DIR/venv-repeat-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -r requirements.txt

python3 main.py train \
    --data_path "../${DATA_PATH}" \
    --context_len 150 \
    --no_cov \
    --augment \
    --epochs 50 \
    --early_stop 10 \
    --seed "$SEED" \
    --ckpt_dir ../results/checkpoints_repeats \
    --run_name "$RUN_NAME"

echo "[$(date +%T)] Koniec: ${RUN_NAME}"
