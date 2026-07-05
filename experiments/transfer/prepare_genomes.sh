#!/usr/bin/env bash
# Download reference genomes from NCBI and generate training CSVs via the Rust pipeline.
#
# Usage (from repo root):
#   bash experiments/transfer/prepare_genomes.sh [organism_name ...]
#
# Without arguments: prepares all 5 organisms.
# Organisms whose FASTA is already in genomes/ are not downloaded again.
# .gz files are deleted after decompression.
#
# Pre-requisites:
#   - cargo build --release
#   - wget or curl

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

GENOMES_DIR="genomes"
mkdir -p "$GENOMES_DIR"

if [ ! -f "target/release/dna-assembler" ]; then
    echo "[prepare] Building Rust binary..."
    cargo build --release
fi

# ── Organism definitions ───────────────────────────────────────────────────────
# FTP_URL: empty string means FASTA is already local (no download needed)
declare -A FTP_URL=(
    [ecoli]="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
    [cerevisiae]="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/GCF_000146045.2_R64/GCF_000146045.2_R64_genomic.fna.gz"
    [bsubtilis]="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/009/045/GCF_000009045.1_ASM904v1/GCF_000009045.1_ASM904v1_genomic.fna.gz"
    [mtuberculosis]="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/195/955/GCF_000195955.2_ASM19595v2/GCF_000195955.2_ASM19595v2_genomic.fna.gz"
    [hpylori]="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/008/525/GCF_000008525.1_ASM852v1/GCF_000008525.1_ASM852v1_genomic.fna.gz"
)

READ_LENGTH=150
COVERAGE=100
KMER=21

BENCHMARK_JSON="experiments/transfer/results/benchmark/benchmark_results.json"
mkdir -p "$(dirname "$BENCHMARK_JSON")"
# Initialize JSON array if file doesn't exist
[ -f "$BENCHMARK_JSON" ] || echo "[]" > "$BENCHMARK_JSON"

# ── Which organisms to prepare ─────────────────────────────────────────────────
if [ $# -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=(ecoli cerevisiae bsubtilis mtuberculosis hpylori)
fi

# ── Helpers ────────────────────────────────────────────────────────────────────
download_file() {
    local url="$1" dest="$2"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$dest" "$url"
    else
        echo "ERROR: neither wget nor curl found."
        exit 1
    fi
}

# Find existing FASTA in genomes/ regardless of extension (.fna / .fasta / .fa)
find_fasta() {
    local name="$1"
    for ext in fna fasta fa; do
        local path="${GENOMES_DIR}/${name}.${ext}"
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

# ── Main loop ──────────────────────────────────────────────────────────────────
for NAME in "${TARGETS[@]}"; do
    if [ -z "${FTP_URL[$NAME]+x}" ]; then
        echo "ERROR: unknown organism '$NAME'. Known: ${!FTP_URL[*]}"
        exit 1
    fi

    CSV_OUT="data/training/${NAME}_training.csv"
    if [ -f "$CSV_OUT" ]; then
        echo "[${NAME}] CSV already exists - skipping."
        continue
    fi

    # Find or download FASTA
    if FASTA_FILE=$(find_fasta "$NAME"); then
        echo "[${NAME}] FASTA found: $FASTA_FILE"
    else
        URL="${FTP_URL[$NAME]}"
        if [ -z "$URL" ]; then
            echo "[${NAME}] ERROR: no FASTA in genomes/ and no download URL configured."
            echo "           Put the FASTA file in genomes/${NAME}.(fna|fasta) and retry."
            exit 1
        fi

        GZ_FILE="${GENOMES_DIR}/${NAME}.fna.gz"
        echo "[${NAME}] Downloading from NCBI..."
        download_file "$URL" "$GZ_FILE"
        echo "[${NAME}] Decompressing (and removing .gz)..."
        gunzip "$GZ_FILE"   # removes .gz, leaves .fna
        FASTA_FILE="${GENOMES_DIR}/${NAME}.fna"
    fi

    # Run Rust pipeline (with timing)
    echo "[${NAME}] Running Rust pipeline..."
    PIPELINE_START=$(date +%s%3N)
    ./target/release/dna-assembler pipeline \
        --reference "$FASTA_FILE" \
        --read-length "$READ_LENGTH" \
        --coverage  "$COVERAGE" \
        --k         "$KMER" \
        --results-dir "results/${NAME}" \
        --gen-train-data
    PIPELINE_END=$(date +%s%3N)
    PIPELINE_TIME_S=$(echo "scale=3; ($PIPELINE_END - $PIPELINE_START) / 1000" | bc)

    if [ -f "$CSV_OUT" ]; then
        echo "[${NAME}] Done -> $CSV_OUT"
        TOTAL_JUNCTIONS=$(( $(wc -l < "$CSV_OUT") - 1 ))

        # Append benchmark entry (remove existing entry for this organism first)
        GENOME_SIZE_BP=$(awk '/^>/{next} {s+=length($0)} END{print s}' "$FASTA_FILE")
        GENOME_SIZE_MBP=$(echo "scale=2; $GENOME_SIZE_BP / 1000000" | bc)

        python3 - <<PYEOF
import json, pathlib
path = pathlib.Path("$BENCHMARK_JSON")
data = json.loads(path.read_text())
data = [d for d in data if d.get("organism") != "$NAME"]
data.append({
    "organism": "$NAME",
    "genome_size_mbp": float("$GENOME_SIZE_MBP"),
    "pipeline_time_s": float("$PIPELINE_TIME_S"),
    "total_junctions": $TOTAL_JUNCTIONS,
})
path.write_text(json.dumps(data, indent=2))
PYEOF
    else
        echo "[${NAME}] WARNING: $CSV_OUT not created - check Rust output above."
    fi
done

echo ""
echo "All requested organisms prepared."
echo "Next step: python experiments/transfer/run_experiments.py"
