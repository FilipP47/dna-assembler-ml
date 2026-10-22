#!/bin/bash
#SBATCH --job-name=groupB_download_mash
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../config.sh"

WORKDIR="$ROOT/experiments/hpc/data/genomes_enterobacteriaceae"
cd "$WORKDIR"

MIN_SIZE=2000000
MAX_SIZE=8000000

mkdir -p fasta sketches

echo "[1/4] Pobieranie sekwencji FASTA dla sampled_pool.tsv..."
> download_manifest.tsv
> accession_to_genus.tsv
while IFS=$'\t' read -r line; do
    acc=$(echo "$line" | awk -F'\t' '{print $1}')
    genus=$(echo "$line" | awk -F'\t' '{print $8}' | awk '{print $1}')
    ftp_path=$(echo "$line" | awk -F'\t' '{print $20}')
    base=$(basename "$ftp_path")
    url="${ftp_path}/${base}_genomic.fna.gz"
    https_url=$(echo "$url" | sed 's|^ftp://|https://|')
    echo -e "${acc}\t${genus}" >> accession_to_genus.tsv
    if wget -q -O "fasta/${acc}.fna.gz" "$https_url"; then
        echo -e "${acc}\tOK" >> download_manifest.tsv
    else
        echo -e "${acc}\tFAILED" >> download_manifest.tsv
        echo "  UWAGA: nie udalo sie pobrac $acc" >&2
    fi
done < sampled_pool.tsv

gunzip -f fasta/*.fna.gz 2>/dev/null || true
n_downloaded=$(find fasta -name "*.fna" | wc -l)
echo "  Pobrano i rozpakowano: $n_downloaded"

echo "[2/4] Pomiar rozmiaru genomu i filtrowanie do okna ${MIN_SIZE}-${MAX_SIZE} bp..."
> genome_sizes.tsv
> filtered_accessions.txt
for f in fasta/*.fna; do
    acc=$(basename "$f" .fna)
    size=$(grep -v '^>' "$f" | tr -d '\n' | wc -c)
    echo -e "${acc}\t${size}" >> genome_sizes.tsv
    if [ "$size" -ge "$MIN_SIZE" ] && [ "$size" -le "$MAX_SIZE" ]; then
        echo "$acc" >> filtered_accessions.txt
    fi
done
n_filtered=$(wc -l < filtered_accessions.txt)
echo "  Po filtrze rozmiaru: $n_filtered genomow"

echo "[3/4] Sklad rodzajowy po filtrze rozmiaru:"
while read -r acc; do
    grep -P "^${acc}\t" accession_to_genus.tsv
done < filtered_accessions.txt | awk -F'\t' '{print $2}' | sort | uniq -c | sort -rn

echo "[4/4] Liczenie sketchy i macierzy odleglosci Mash..."
while read -r acc; do
    if [ ! -f "sketches/${acc}.msh" ]; then
        "$MASH_BIN" sketch -k 21 -s 10000 -o "sketches/${acc}" "fasta/${acc}.fna" 2>/dev/null
    fi
done < filtered_accessions.txt

find sketches -name "*.msh" | sort > sketch_list.txt
"$MASH_BIN" triangle -l sketch_list.txt > distance_matrix_triangle.txt

echo "Gotowe. Kolejny krok: python3 select_split.py (na wezle loginowym)"
