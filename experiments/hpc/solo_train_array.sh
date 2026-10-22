#!/bin/bash
#SBATCH --job-name=mv1_solo
#SBATCH --output=job-%A_%a.out
#SBATCH --error=job-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=1-190%20

set -euo pipefail
export PYTHONUNBUFFERED=1
module load uv/0.11.2

cd /scratch/fpietrew/dna-assembler/ml

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../solo_tasks.txt)
GRP=$(echo "$LINE" | cut -d, -f1)
ACC=$(echo "$LINE" | cut -d, -f2)
CSV=$(echo "$LINE" | cut -d, -f3)
SEED=$(echo "$LINE" | cut -d, -f4)
RUN_NAME="solo_${GRP}_${ACC}_seed${SEED}"

echo "[$(date +%T)] ${RUN_NAME} (csv=${CSV})"

VENV_DIR=$SLURM_SUBMIT_DIR/venv-solo-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -r requirements.txt

python3 main.py train \
    --data_path "../${CSV}" \
    --context_len 150 \
    --no_cov \
    --augment \
    --epochs 50 \
    --early_stop 10 \
    --seed "$SEED" \
    --device cpu \
    --val_split 0.2 \
    --ckpt_dir ../results/checkpoints_solo \
    --run_name "$RUN_NAME"

echo "[$(date +%T)] Koniec: ${RUN_NAME}"
