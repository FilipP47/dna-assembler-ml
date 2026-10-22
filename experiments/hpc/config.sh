#!/bin/bash
# Shared configuration for experiments/hpc/. Sourced by every *_array.sh
# (under SLURM), by run_local.sh (plain parallel loop), and by the
# standalone login-node scripts (generate_task_lists.sh, organize_splits*.sh,
# prepare_solo_org_list.sh). Nothing here is SLURM-specific; that lives in
# the scripts that source this file.
#
# Every value below can be overridden by exporting it before sourcing this
# file, e.g.:
#   ROOT=/scratch/fpietrew/dna-assembler CORES=16 ./run_local.sh train_full 4
set -euo pipefail

# Repo root. Default: walk up from this file's location, so it works
# whether you cd into experiments/hpc/ or source it from elsewhere. On the
# HPC cluster this used to be hardcoded to /scratch/fpietrew/dna-assembler;
# override with ROOT=/scratch/<user>/dna-assembler if the checkout lives on
# scratch rather than $HOME.
: "${ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# cpu or cuda. Passed straight through as --device to ml/main.py (which
# expects torch's own vocabulary, not "gpu"), and also picks which uv
# dependency group gets synced (cpu -> torch CPU wheel, cuda -> torch cu128
# wheel; see ml/pyproject.toml). SLURM gpu-partition scripts (gpu-m, with
# --gres=gpu:1) set DEVICE=cuda before sourcing this file; everything else
# (local runs, CPU-partition SLURM jobs) defaults to cpu.
: "${DEVICE:=cpu}"

# Seeds used for the 3-seed repeat-variant runs (repeat_variant_array.sh /
# tasks/train_repeat.sh) and available to any script that wants a
# reproducible seed list.
SEEDS=(42 123 7)

# Parallelism for run_local.sh (xargs -P). Defaults to the number of
# visible CPUs; SLURM array scripts use #SBATCH --cpus-per-task instead and
# don't read this.
: "${CORES:=$(nproc 2>/dev/null || echo 4)}"

# uv dependency group matching $DEVICE (see ml/pyproject.toml: "cpu" is the
# default group, "cu128" is the CUDA 12.8 group; they conflict so exactly
# one must be active).
# Plain string, not an array: kept portable across bash versions when
# word-split unquoted (no spaces in any of the flag values below).
if [ "$DEVICE" = "cuda" ]; then
    UV_SYNC_EXTRA_ARGS="--no-group cpu --group cu128"
else
    UV_SYNC_EXTRA_ARGS=""  # "cpu" is already in [tool.uv] default-groups
fi

# mash binary. On the cluster it's installed to $HOME/bin/mash (no module
# available); in the Docker image it's installed system-wide and just on
# PATH. Override with MASH_BIN=/some/other/path if needed.
if [ -x "$HOME/bin/mash" ]; then
    : "${MASH_BIN:=$HOME/bin/mash}"
else
    : "${MASH_BIN:=mash}"
fi

# Data roots (relative to $ROOT).
DATA_DIR="$ROOT/data"
TRAIN_DATA_DIR="$ROOT/data/training"
RESULTS_DIR="$ROOT/results"
HPC_DIR="$ROOT/experiments/hpc"

export ROOT DEVICE CORES MASH_BIN DATA_DIR TRAIN_DATA_DIR RESULTS_DIR HPC_DIR UV_SYNC_EXTRA_ARGS
