#!/bin/bash
#SBATCH --job-name=mv1_train
#SBATCH --output=job-%A_%a.out
#SBATCH --error=job-%A_%a.err
#SBATCH --partition=gpu-m
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=1-4

set -euo pipefail
export PYTHONUNBUFFERED=1
module load uv/0.11.2
module load cuda/12.9

cd /scratch/fpietrew/dna-assembler/ml

case $SLURM_ARRAY_TASK_ID in
    1) DATA_PATH="../data/training/group_a/farthest_train"; RUN_NAME="group_a_farthest" ;;
    2) DATA_PATH="../data/training/group_a/random_train";   RUN_NAME="group_a_random"   ;;
    3) DATA_PATH="../data/training/group_b/farthest_train"; RUN_NAME="group_b_farthest" ;;
    4) DATA_PATH="../data/training/group_b/random_train";   RUN_NAME="group_b_random"   ;;
esac

echo "[$(date +%T)] Task ${SLURM_ARRAY_TASK_ID}: ${RUN_NAME} (${DATA_PATH})"

VENV_DIR=$SLURM_SUBMIT_DIR/venv-train-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -r requirements.txt
echo "[$(date +%T)] Zaleznosci gotowe."

python3 main.py train \
    --data_path "$DATA_PATH" \
    --context_len 150 \
    --no_cov \
    --augment \
    --epochs 50 \
    --early_stop 10 \
    --ckpt_dir ../results/checkpoints \
    --run_name "$RUN_NAME"

echo "[$(date +%T)] Koniec: ${RUN_NAME}"
