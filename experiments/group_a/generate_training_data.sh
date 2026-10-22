#!/usr/bin/env bash
# Generate training CSVs for all group A genomes (train + held-out) via the Rust pipeline.
#
# Usage (from repo root):
#   bash experiments/group_a/generate_training_data.sh [--jobs N]
#
# Without --jobs: uses `nproc` parallel pipeline runs (each genome is an
# independent process, safe to parallelize; on a shared/HPC node pass
# --jobs explicitly instead of relying on the local core count).
#
# Reads:
#   experiments/group_a/results/train_accessions.txt
#   experiments/group_a/results/heldout_accessions.txt
#   genomes/ecoli_group_a/<accession>.fna
# Writes:
#   data/training/ecoli_group_a/train/<accession>_training.csv
#   data/training/ecoli_group_a/heldout/<accession>_training.csv
#   experiments/group_a/results/benchmark/<accession>.json
#
# Pre-requisites: cargo build --release; experiments/group_a/select_heldout.py
# already run (needs train_accessions.txt / heldout_accessions.txt to exist).
# Resumable: skips any genome whose CSV already exists, safe to re-run after
# an interruption.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

JOBS="$(nproc)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs) JOBS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

GENOMES_DIR="genomes/ecoli_group_a"
RESULTS_ROOT="experiments/group_a/results"
TRAIN_LIST="${RESULTS_ROOT}/train_accessions.txt"
HELDOUT_LIST="${RESULTS_ROOT}/heldout_accessions.txt"
BENCHMARK_DIR="${RESULTS_ROOT}/benchmark"

READ_LENGTH=150
COVERAGE=100
KMER=21

if [ ! -f "target/release/dna-assembler" ]; then
    echo "[generate] Building Rust binary..."
    cargo build --release
fi

for f in "$TRAIN_LIST" "$HELDOUT_LIST"; do
    [ -f "$f" ] || { echo "ERROR: $f not found - run experiments/group_a/select_heldout.py first."; exit 1; }
done

mkdir -p "data/training/ecoli_group_a/train" "data/training/ecoli_group_a/heldout" "$BENCHMARK_DIR"

# One JSON file per accession (not one shared file) so concurrent jobs never
# race on a read-modify-write of the same file.
record_benchmark() {
    local accession="$1" split="$2" fasta="$3" time_s="$4" csv_out="$5"
    [ -f "$csv_out" ] || return 0
    local total_junctions size_bp
    total_junctions=$(( $(wc -l < "$csv_out") - 1 ))
    size_bp=$(awk '/^>/{next} {s+=length($0)} END{print s}' "$fasta")
    cat > "${BENCHMARK_DIR}/${accession}.json" <<EOF
{
  "accession": "${accession}",
  "split": "${split}",
  "genome_size_bp": ${size_bp},
  "pipeline_time_s": ${time_s},
  "total_junctions": ${total_junctions}
}
EOF
}

run_one() {
    local accession="$1" split="$2"
    local fasta="${GENOMES_DIR}/${accession}.fna"
    local out_dir="data/training/ecoli_group_a/${split}"
    local csv_out="${out_dir}/${accession}_training.csv"

    if [ -f "$csv_out" ]; then
        echo "[${accession}] CSV already exists - skipping."
        return 0
    fi
    if [ ! -f "$fasta" ]; then
        echo "[${accession}] WARNING: FASTA not found ($fasta) - skipping."
        return 0
    fi

    echo "[${accession}] (${split}) running pipeline..."
    local start end elapsed
    start=$(date +%s%3N)
    ./target/release/dna-assembler pipeline \
        --reference "$fasta" \
        --read-length "$READ_LENGTH" \
        --coverage "$COVERAGE" \
        --k "$KMER" \
        --results-dir "results/ecoli_group_a/${accession}" \
        --gen-train-data \
        --train-data-dir "$out_dir"
    end=$(date +%s%3N)
    elapsed=$(echo "scale=3; ($end - $start) / 1000" | bc)

    if [ -f "$csv_out" ]; then
        echo "[${accession}] done -> $csv_out (${elapsed}s)"
        record_benchmark "$accession" "$split" "$fasta" "$elapsed" "$csv_out"
    else
        echo "[${accession}] WARNING: $csv_out not created - check pipeline output above."
    fi
}

# Build the full (accession, split) job list, then run with bounded concurrency.
JOB_LIST=()
while IFS= read -r acc; do [ -n "$acc" ] && JOB_LIST+=("${acc} train"); done < "$TRAIN_LIST"
while IFS= read -r acc; do [ -n "$acc" ] && JOB_LIST+=("${acc} heldout"); done < "$HELDOUT_LIST"

echo "Generating training data for ${#JOB_LIST[@]} genomes with ${JOBS} parallel jobs..."

running=0
for job in "${JOB_LIST[@]}"; do
    read -r accession split <<< "$job"
    run_one "$accession" "$split" &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait

echo ""
echo "All genomes processed. Next step: experiments/group_a/run_group_a.py"
