#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$ROOT"

# --- Lista 1: powtorzenia 4 glownych wariantow transferu (3 seedy kazdy) ---
> repeat_variant_tasks.txt
for variant in group_a_farthest group_a_random group_b_farthest group_b_random; do
    case $variant in
        group_a_farthest) data="data/training/group_a/farthest_train" ;;
        group_a_random)   data="data/training/group_a/random_train" ;;
        group_b_farthest) data="data/training/group_b/farthest_train" ;;
        group_b_random)   data="data/training/group_b/random_train" ;;
    esac
    for seed in "${SEEDS[@]}"; do
        echo "${variant},${data},${seed}" >> repeat_variant_tasks.txt
    done
done
echo "repeat_variant_tasks.txt: $(wc -l < repeat_variant_tasks.txt) zadan"

# --- Lista 2: solo baseline (5 seedow x kazdy unikalny organizm z A i B) ---
> solo_tasks.txt
for grp in a b; do
    outdir="experiments/hpc/data/genomes_ecoli"
    [ "$grp" = "b" ] && outdir="experiments/hpc/data/genomes_enterobacteriaceae"
    while read -r acc; do
        csv="data/training/${acc}_training.csv"
        for seed in 42 123 7 1 99; do
            echo "${grp},${acc},${csv},${seed}" >> solo_tasks.txt
        done
    done < "${outdir}/solo_organisms.txt"
done
echo "solo_tasks.txt: $(wc -l < solo_tasks.txt) zadan"
