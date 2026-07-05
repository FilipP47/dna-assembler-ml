#!/usr/bin/env bash
# Download a FASTA file (plain or .gz) and save it to genomes/.
#
# Usage:
#   bash experiments/transfer/download_fasta.sh <url> <organism_name>
#
# Examples:
#   bash experiments/transfer/download_fasta.sh \
#       https://ftp.ncbi.nlm.nih.gov/.../genomic.fna.gz  hpylori
#
# The file is saved as genomes/<organism_name>.fna
# If the URL ends with .gz the archive is decompressed and .gz is deleted.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ $# -lt 2 ]; then
    echo "Usage: bash experiments/transfer/download_fasta.sh <url> <organism_name>"
    exit 1
fi

URL="$1"
NAME="$2"
GENOMES_DIR="genomes"
mkdir -p "$GENOMES_DIR"

DEST="${GENOMES_DIR}/${NAME}.fna"

if [ -f "$DEST" ]; then
    echo "[${NAME}] Already exists: $DEST"
    exit 0
fi

echo "[${NAME}] Downloading from: $URL"

if command -v wget &>/dev/null; then
    wget -q --show-progress -O "${DEST}.tmp" "$URL"
elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "${DEST}.tmp" "$URL"
else
    echo "ERROR: neither wget nor curl found."
    exit 1
fi

# Decompress if .gz
if [[ "$URL" == *.gz ]]; then
    echo "[${NAME}] Decompressing..."
    gunzip -c "${DEST}.tmp" > "$DEST"
    rm "${DEST}.tmp"
else
    mv "${DEST}.tmp" "$DEST"
fi

echo "[${NAME}] Saved to: $DEST"
