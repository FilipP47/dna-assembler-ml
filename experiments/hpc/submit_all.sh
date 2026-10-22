#!/bin/bash
# Chain the whole group A/B pipeline on SLURM with sbatch --dependency=afterok,
# so each stage only starts once the previous one has actually succeeded.
# This only makes sense under SLURM; the local/Docker equivalent is running
# the matching run_local.sh calls by hand in the same order (see README.md).
#
# Order: download+mash -> split -> generate CSVs -> train -> eval zero-shot
#        -> repeat-variant training -> solo baseline -> eval repeats.
#
# Usage: ./submit_all.sh
#   Optional: SKIP_DOWNLOAD=1 to start from an existing genome pool (the
#   repo already ships output/*.txt, full_pool_100.txt, solo_organisms.txt
#   from a prior run - see config.sh and README.md "Reproducibility").
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$HPC_DIR"

sbatch_id() {
    # Print just the numeric job id from `sbatch`'s "Submitted batch job N".
    awk '{print $NF}'
}

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    echo "[1/8] download + mash (group A, group B)"
    JOB_B_FILTER=$(sbatch data/genomes_enterobacteriaceae/groupB_filter_and_sample.sh | sbatch_id)
    JOB_A_DL=$(sbatch data/genomes_ecoli/groupA_download_and_mash.sh | sbatch_id)
    JOB_B_DL=$(sbatch --dependency=afterok:"$JOB_B_FILTER" data/genomes_enterobacteriaceae/groupB_download_and_mash.sh | sbatch_id)
else
    echo "[1/8] download + mash: skipped (SKIP_DOWNLOAD=1, reusing checked-in genome lists)"
    JOB_A_DL=""
    JOB_B_DL=""
fi

# select_split.py and the login-node helper scripts run directly (not via
# sbatch - they're meant to be cheap enough for the login node). If the
# download jobs are running, sbatch --dependency on a *script*, not these,
# doesn't apply; in that case run step [2] by hand once [1] finishes and
# re-invoke this script with SKIP_DOWNLOAD=1 to continue from step [3].
if [ -n "$JOB_A_DL$JOB_B_DL" ]; then
    echo "NOTE: download jobs submitted (A=$JOB_A_DL, B=$JOB_B_DL)."
    echo "      Once both finish, run select_split.py for A and B, then"
    echo "      SKIP_DOWNLOAD=1 ./submit_all.sh to continue from step 3."
    exit 0
fi

echo "[2/8] split + organize (login node, no sbatch)"
bash organize_splits.sh
bash organize_splits_B.sh
bash prepare_solo_org_list.sh
bash generate_task_lists.sh

echo "[3/8] build + generate training CSVs (group A: 100 genomes, group B: 100 genomes)"
JOB_BUILD=$(sbatch build.sh | sbatch_id)
JOB_GEN_A=$(sbatch --dependency=afterok:"$JOB_BUILD" groupA_full_gen_array.sh | sbatch_id)
JOB_GEN_B=$(sbatch --dependency=afterok:"$JOB_BUILD" groupB_full_gen_array.sh | sbatch_id)

echo "[4/8] train (4 main variants)"
JOB_TRAIN=$(sbatch --dependency=afterok:"$JOB_GEN_A,$JOB_GEN_B" train_full_array.sh | sbatch_id)

echo "[5/8] evaluate zero-shot"
JOB_EVAL=$(sbatch --dependency=afterok:"$JOB_TRAIN" evaluate_zero_shot.sh | sbatch_id)

echo "[6/8] repeat-variant training (3 seeds x 4 variants)"
JOB_REPEAT=$(sbatch --dependency=afterok:"$JOB_EVAL" repeat_variant_array.sh | sbatch_id)

echo "[7/8] solo baseline (5 seeds x held-out organisms)"
# Solo baseline doesn't actually need the repeat-variant checkpoints; chained
# after JOB_REPEAT anyway to match the requested submission order and avoid
# saturating the queue with everything at once.
JOB_SOLO=$(sbatch --dependency=afterok:"$JOB_REPEAT" solo_train_array.sh | sbatch_id)

echo "[8/8] evaluate repeat-variant checkpoints"
JOB_EVAL_REPEAT=$(sbatch --dependency=afterok:"$JOB_SOLO" evaluate_repeats.sh | sbatch_id)

cat <<EOF

Submitted chain:
  gen_a=$JOB_GEN_A gen_b=$JOB_GEN_B
  train=$JOB_TRAIN eval_zero_shot=$JOB_EVAL
  repeat=$JOB_REPEAT solo=$JOB_SOLO eval_repeats=$JOB_EVAL_REPEAT

Track with: squeue -u \$USER
Collect results afterwards with collect_results.py / collect_repeats.py /
collect_solo.py (set ROOT=$ROOT if running them off the cluster).
EOF
