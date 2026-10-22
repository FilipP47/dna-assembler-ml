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

cd /scratch/fpietrew/dna-assembler/ml

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../repeat_variant_tasks.txt)
VARIANT=$(echo "$LINE" | cut -d, -f1)
DATA_PATH=$(echo "$LINE" | cut -d, -f2)
SEED=$(echo "$LINE" | cut -d, -f3)

TEST_DIR="../$(echo "$DATA_PATH" | sed 's/_train$/_test/')"
CKPT="../results/checkpoints_repeats/mv1_siamese_cnn_${VARIANT}_seed${SEED}_best.pt"
RESULTS_DIR="../results/eval_repeats/${VARIANT}_seed${SEED}"

echo "[$(date +%T)] Ewaluacja ${VARIANT} seed=${SEED} na ${TEST_DIR}"
mkdir -p "$RESULTS_DIR"

VENV_DIR=$SLURM_SUBMIT_DIR/venv-evalrep-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -r requirements.txt

for csv_file in "$TEST_DIR"/*.csv; do
    acc=$(basename "$csv_file" _training.csv)
    python3 main.py evaluate \
        --model_path "$CKPT" \
        --test_csv "$csv_file" \
        --context_len 150 \
        --output_json "${RESULTS_DIR}/${acc}.json"
done

echo "[$(date +%T)] Koniec: ${VARIANT} seed=${SEED}"
