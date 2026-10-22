#!/bin/bash
#SBATCH --job-name=groupA_download_mash
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

WORKDIR="$ROOT/experiments/hpc/data/genomes_ecoli"
cd "$WORKDIR"

MIN_SIZE=4000000
MAX_SIZE=6000000
POOL_SIZE=200

mkdir -p fasta sketches

echo "[1/5] Filtrowanie Complete Genome z assembly_summary.txt..."
awk -F'\t' '$12=="Complete Genome"' assembly_summary.txt > complete_genomes.tsv
n_complete=$(wc -l < complete_genomes.tsv)
echo "  Complete Genome: $n_complete"

echo "[2/5] Deterministyczne losowanie puli $POOL_SIZE kandydatow..."
shuf -n "$POOL_SIZE" --random-source=<(yes 42) complete_genomes.tsv > sampled_pool.tsv
echo "  Wylosowano: $(wc -l < sampled_pool.tsv)"

echo "[3/5] Pobieranie sekwencji FASTA..."
> download_manifest.tsv
while IFS=$'\t' read -r line; do
    acc=$(echo "$line" | awk -F'\t' '{print $1}')
    ftp_path=$(echo "$line" | awk -F'\t' '{print $20}')
    base=$(basename "$ftp_path")
    url="${ftp_path}/${base}_genomic.fna.gz"
    https_url=$(echo "$url" | sed 's|^ftp://|https://|')
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

echo "[4/5] Pomiar rozmiaru genomu i filtrowanie do okna ${MIN_SIZE}-${MAX_SIZE} bp..."
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

echo "[5/5] Liczenie sketchy i macierzy odleglosci Mash dla przefiltrowanych genomow..."
while read -r acc; do
    if [ ! -f "sketches/${acc}.msh" ]; then
        "$MASH_BIN" sketch -k 21 -s 10000 -o "sketches/${acc}" "fasta/${acc}.fna" 2>/dev/null
    fi
done < filtered_accessions.txt

find sketches -name "*.msh" | sort > sketch_list.txt
"$MASH_BIN" triangle -l sketch_list.txt > distance_matrix_triangle.txt

echo "Gotowe. Kolejny krok: python3 select_split.py (na wezle loginowym, lekkie)"
