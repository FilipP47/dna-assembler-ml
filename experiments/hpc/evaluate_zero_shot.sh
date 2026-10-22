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

cd /scratch/fpietrew/dna-assembler/ml

case $SLURM_ARRAY_TASK_ID in
    1) TEST_DIR="../data/training/group_a/farthest_test"; CKPT="../results/checkpoints/mv1_siamese_cnn_group_a_farthest_best.pt"; OUT_PREFIX="group_a_farthest" ;;
    2) TEST_DIR="../data/training/group_a/random_test";   CKPT="../results/checkpoints/mv1_siamese_cnn_group_a_random_best.pt";   OUT_PREFIX="group_a_random"   ;;
    3) TEST_DIR="../data/training/group_b/farthest_test"; CKPT="../results/checkpoints/mv1_siamese_cnn_group_b_farthest_best.pt"; OUT_PREFIX="group_b_farthest" ;;
    4) TEST_DIR="../data/training/group_b/random_test";   CKPT="../results/checkpoints/mv1_siamese_cnn_group_b_random_best.pt";   OUT_PREFIX="group_b_random"   ;;
esac

VENV_DIR=$SLURM_SUBMIT_DIR/venv-eval-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -r requirements.txt

echo "[$(date +%T)] Ewaluacja ${OUT_PREFIX} na kazdym z 10 organizmow osobno"

RESULTS_DIR="../results/eval_${OUT_PREFIX}"
mkdir -p "$RESULTS_DIR"

for csv_file in "$TEST_DIR"/*.csv; do
    acc=$(basename "$csv_file" _training.csv)
    echo "[$(date +%T)] -- $acc --"
    python3 main.py evaluate \
        --model_path "$CKPT" \
        --test_csv "$csv_file" \
        --context_len 150 \
        --output_json "${RESULTS_DIR}/${acc}.json"
done

echo "[$(date +%T)] Koniec: ${OUT_PREFIX}"
