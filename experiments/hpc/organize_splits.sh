#!/bin/bash
# Odpalic PO zakonczeniu generacji (na wezle loginowym, lekkie - tylko symlinki)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$ROOT"

BASE=experiments/hpc/data/genomes_ecoli/output
TRAIN_DIR=data/training/group_a

for variant in farthest random; do
    mkdir -p "${TRAIN_DIR}/${variant}_train" "${TRAIN_DIR}/${variant}_test"

    while read -r acc; do
        src="../../${acc}_training.csv"
        ln -sf "$src" "${TRAIN_DIR}/${variant}_train/${acc}_training.csv"
    done < "${BASE}/train_90_for_${variant}_split.txt"

    while read -r acc; do
        src="../../${acc}_training.csv"
        ln -sf "$src" "${TRAIN_DIR}/${variant}_test/${acc}_training.csv"
    done < "${BASE}/test_${variant}_10.txt"

    echo "${variant}: train=$(ls ${TRAIN_DIR}/${variant}_train | wc -l), test=$(ls ${TRAIN_DIR}/${variant}_test | wc -l)"
done
