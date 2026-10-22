# Grupa A: zero-shot transfer na najdalszych szczepach E. coli

Grupa A z planu eksperymentow A/B/C (gradient wg dystansu genomowego).
Trening generalisty na ~90 genomach E. coli, ewaluacja zero-shot na 10
szczepach dobranych jako najdalsze filogenetycznie (nie losowo), zeby
wykluczyc zapamietywanie prawie-duplikatow. Kontekst i wyniki:
`RAPORT.md` (sufit ~58% na pojedynczym organizmie) oraz `RAPORT_TRANSFER.md`.

## Wymagania wstepne

- `cargo build --release` (binarka `target/release/dna-assembler` z flaga
  `--train-data-dir`, dodana specjalnie do tego eksperymentu).
- `mash` w PATH (`sudo apt install mash` - nowa zaleznosc CLI, potrzebna bo
  sama taksonomia nie rozroznia w obrebie jednego gatunku; patrz
  `select_heldout.py`).
- `ml/.venv` przygotowane wedlug reszty projektu (`cd ml && uv sync`) -
  potrzebne do `run_group_a.py` (polars, pyyaml) i do `ml/main.py` (torch).

## Kolejnosc uruchomienia

1. **`python3 experiments/group_a/fetch_genomes.py`**
   Pobiera `assembly_summary.txt` dla *E. coli* z NCBI FTP, filtruje
   (Complete Genome, nie wykluczone, 2-8 Mbp), losuje reprodukowalnie
   (seed=42) ~110 kandydatow i sciaga je do `genomes/ecoli_group_a/`.
   Zapisuje `results/manifest.csv`. Uzyj `--limit N`, zeby najpierw
   sprawdzic dzialanie na kilku genomach zamiast calej puli.

2. **`python3 experiments/group_a/select_heldout.py`**
   Liczy macierz dystansow Mash miedzy wszystkimi pobranymi genomami i
   wybiera 10 o najwiekszym dystansie do reszty puli jako held-out (reszta,
   do 90, jako train). Zapisuje `results/train_accessions.txt`,
   `results/heldout_accessions.txt`, `results/distances.csv` (przyda sie
   pozniej do wspolnego wykresu dystans-vs-delta z grupami B/C).

3. **`bash experiments/group_a/generate_training_data.sh [--jobs N]`**
   Dla kazdego genomu z obu list odpala pipeline Rust (k=21, coverage=100x,
   read-length=150 - te same parametry co reszta projektu) i zapisuje CSV
   bezposrednio do `data/training/ecoli_group_a/{train,heldout}/`. Domyslnie
   rownolegle na `nproc` procesow (kazdy genom to niezalezny proces) - na
   HPC podaj `--jobs` jawnie. Resumable: pomija juz wygenerowane CSV, wiec
   mozna bezpiecznie przerywac i wznawiac.

4. **`python3 experiments/group_a/run_group_a.py`**
   Trenuje model generalist na `data/training/ecoli_group_a/train/`
   (`ml/main.py train`), ewaluuje go zero-shot na kazdym z 10 held-out CSV
   osobno oraz na calym katalogu held-out zagregowane (`ml/main.py
   evaluate`), liczy baseline max-coverage bezposrednio z CSV (ta sama
   logika co `ml/experiments/cov_ratio_analysis.py`), i zapisuje
   `results/comparison.csv` + wypisuje tabele w formacie zgodnym z
   `RAPORT.md` (procenty z 1 miejscem po przecinku, delta w pp).

Wszystkie cztery kroki sa idempotentne/resumable (sentinel pliki, pomijanie
istniejacych CSV/checkpointow) - bezpiecznie przerywac i wznawiac w
dowolnym miejscu.

## Poza zakresem tego przebiegu

Solo-reference per held-out organizm (model trenowany i oceniany na tym
samym organizmie - gorna referencja "gdybysmy mieli dane z tego genomu") i
ablacja wielkosci zbioru treningowego (10/30/60/90 organizmow) NIE sa tu
zaimplementowane - to nastepny krok po zobaczeniu wyniku
generalist-vs-baseline z `run_group_a.py`.

## Wyniki

`results/comparison.csv` - glowna tabela (per organizm + zagregowane).
`results/distances.csv` - macierz dystansow Mash (do wykresu dystans-vs-delta
lacznie z grupami B/C, gdy powstana).
`results/benchmark/<accession>.json` - czas pipeline'u i liczba junctionow
per genom (do oszacowania kosztu skalowania do grup B/C).
