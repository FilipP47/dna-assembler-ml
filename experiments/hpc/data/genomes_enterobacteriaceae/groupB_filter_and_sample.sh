#!/bin/bash
#SBATCH --job-name=groupB_filter_sample
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

WORKDIR=/scratch/fpietrew/dna-assembler/data/genomes_enterobacteriaceae
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "[1/3] Pobieranie master assembly_summary.txt dla wszystkich bakterii RefSeq (moze byc duzy plik)..."
if [ ! -f assembly_summary_bacteria.txt ]; then
    wget -q -O assembly_summary_bacteria.txt \
        "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt"
fi
wc -l assembly_summary_bacteria.txt

echo "[2/3] Filtrowanie: Complete Genome + rodzaje z listy TARGET_GENERA..."
awk -F'\t' '$12=="Complete Genome"' assembly_summary_bacteria.txt > complete_all.tsv

TARGET_GENERA="Escherichia Salmonella Klebsiella Citrobacter Enterobacter Shigella Serratia Proteus Yersinia Cronobacter"

> enterobacteriaceae_candidates.tsv
for genus in $TARGET_GENERA; do
    awk -F'\t' -v g="$genus" '$8 ~ "^"g" "' complete_all.tsv >> enterobacteriaceae_candidates.tsv
done

echo "  Kandydatow po filtrze rodzajow: $(wc -l < enterobacteriaceae_candidates.tsv)"
echo "  Rozklad po rodzajach:"
awk -F'\t' '{print $8}' enterobacteriaceae_candidates.tsv | awk '{print $1}' | sort | uniq -c | sort -rn

echo "[3/3] Stratyfikowane losowanie: ~20 genomow z kazdego rodzaju (deterministycznie)..."
> sampled_pool.tsv
PER_GENUS=20
for genus in $TARGET_GENERA; do
    awk -F'\t' -v g="$genus" '$8 ~ "^"g" "' complete_all.tsv > "tmp_${genus}.tsv"
    n=$(wc -l < "tmp_${genus}.tsv")
    take=$PER_GENUS
    if [ "$n" -lt "$PER_GENUS" ]; then
        take=$n
        echo "  UWAGA: rodzaj $genus ma tylko $n kandydatow (mniej niz $PER_GENUS)"
    fi
    shuf -n "$take" --random-source=<(yes "42_${genus}") "tmp_${genus}.tsv" >> sampled_pool.tsv
    rm -f "tmp_${genus}.tsv"
done

echo "  Finalna pula: $(wc -l < sampled_pool.tsv) genomow"
echo "Gotowe. Kolejny krok: sbatch groupB_download_and_mash.sh"
