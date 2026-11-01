# experiments/hpc/

Skrypty, ktore wygenerowaly wyniki transferu miedzy genomami (grupa A/B,
patrz `RAPORT_TRANSFER.md`), teraz przenosne miedzy klastrem SLURM i
Dockerem. Logika kazdego zadania zyje w `tasks/*.sh` (brak `#SBATCH`, brak
`module load`); `*_array.sh` wola ja pod SLURM-em, `run_local.sh` wola ja
bez SLURM-a, rownolegle przez `xargs -P`. Wszystkie sciezki i parametry
(seed, liczba rdzeni, DEVICE=cpu/cuda) sa w `config.sh`.

## Kolejnosc uruchamiania

### SLURM (HPC)

**Wazne: kazdy `sbatch` trzeba wywolywac z katalogu `experiments/hpc/`**
(np. `cd experiments/hpc && sbatch train_full_array.sh`), nie z innego
miejsca i nie przez pelna sciezke z gdzie indziej. SLURM kopiuje skrypt do
`/var/spool/slurmd/job<ID>/...` przed wykonaniem, wiec `BASH_SOURCE` wewnatrz
skryptu wskazuje na ten katalog spool, nie na repo - kazdy `*_array.sh` i
inny skrypt z `#SBATCH` ustala wiec swoj katalog jako
`${SLURM_SUBMIT_DIR:-...}`, czyli katalog, z ktorego wywolano `sbatch`. Jesli
submit nie bedzie z `experiments/hpc/`, skrypt zakonczy sie czytelnym
"config.sh nie istnieje" zamiast cichym bledem (zobacz job 91700 -
`smoke_test_train.sh` kiedys to zalamalo, zanim `SLURM_SUBMIT_DIR` tu bylo).
`submit_all.sh` robi to poprawnie automatycznie (ma `cd "$HPC_DIR"` przed
kazdym `sbatch`). `experiments/hpc/check_sbatch_paths.sh` jest statycznym
testem, ktory wykrywa ten blad z powrotem, zanim trafi na klaster - uzyj go
przed commitem kazdego nowego `*_array.sh`.

`./submit_all.sh` wysyla caly lancuch przez `sbatch --dependency=afterok`:
pobranie+Mash -> podzial (login node) -> build+generacja CSV -> trening ->
ewaluacja zero-shot -> powtorzenia (3 seedy) -> solo baseline -> ewaluacja
powtorzen. Zobacz komentarze w `submit_all.sh` co dziala na login-node a co
przez `sbatch`, oraz flage `SKIP_DOWNLOAD=1` do wznowienia po pobraniu.

Repo juz ma gotowa pule genomow i podzialy (`data/genomes_*/full_pool_100.txt`,
`output/*.txt`, `solo_organisms.txt`) z faktycznego przebiegu - domyslnie
`submit_all.sh` z nich korzysta, zamiast losowac pule od zera z
`assembly_summary.txt`. Losowanie od zera to osobna, opcjonalna sciezka:
`data/genomes_ecoli/groupA_download_and_mash.sh` i
`data/genomes_enterobacteriaceae/groupB_filter_and_sample.sh` +
`groupB_download_and_mash.sh` (oba deterministyczne, seed=42).

### Docker (bez SLURM, bez konta na HPC)

```
docker build -t dna-assembler-ml .                              # CPU (domyslne)
docker build --build-arg DEVICE=cuda -t dna-assembler-ml:cuda .  # GPU

docker run --rm -v "$PWD/work:/work" dna-assembler-ml \
    experiments/hpc/smoke_test_docker.sh                         # test dymny

docker run --rm -v "$PWD/work:/work" dna-assembler-ml \
    experiments/hpc/run_local.sh gen_group_a 100                 # dowolne zadanie
```

`run_local.sh <task> <n>` odpala te sama logike co `*_array.sh` pod SLURM-em
(patrz `tasks/*.sh`), tyle ze lokalnie, rownolegle na `$CORES` procesach.
Kolejnosc dla pelnego pipelinu jest ta sama jak dla SLURM-a powyzej, tylko
kazdy krok wywolujesz recznie (login-node skrypty tak samo recznie, bez
`sbatch`):

```
bash experiments/hpc/build.sh
bash experiments/hpc/run_local.sh gen_group_a 100
bash experiments/hpc/run_local.sh gen_group_b 100
bash experiments/hpc/organize_splits.sh && bash experiments/hpc/organize_splits_B.sh
bash experiments/hpc/prepare_solo_org_list.sh && bash experiments/hpc/generate_task_lists.sh
bash experiments/hpc/run_local.sh train_full 4
bash experiments/hpc/run_local.sh eval_zero_shot 4
bash experiments/hpc/run_local.sh train_repeat 12
bash experiments/hpc/run_local.sh train_solo 190
bash experiments/hpc/run_local.sh eval_repeats 12
```

## Wymagane zasoby

- **Generacja CSV (Rust, CPU):** ok. 4 rdzenie, 8 GB RAM per genom (bakteria
  4-6 Mbp, k=21, coverage=100x). Pelna pula 100+100 genomow: liczy sie
  rownolegle, ale dysk rosnie - kazdy genom to kilka MB fasta + kilkaset KB
  CSV, plus kontigi/odczyty w `results/` jesli `--save-reads`.
- **Trening (GPU, main variants):** 1 GPU, 8 rdzeni, 32 GB RAM (`gpu-m` w
  SLURM-skryptach). Dziala i na CPU (`DEVICE=cpu`), wolniej.
- **Solo baseline:** CPU, 4 rdzenie, 8 GB RAM - male joby, 190 ich naraz w
  oryginalnym przebiegu (`--array=1-190%20`, 20 rownolegle).
- **Obraz Docker:** ok. 1-1.5 GB (torch CPU + reszta zaleznosci).

## Orientacyjne czasy

Zmierzone w teście dymnym (patrz nizej) na CPU, k=21, coverage=100x -
skaluje sie z dlugoscia genomu i liczba junctionow, nie liniowo z samym
rozmiarem:

| Genom | Dlugosc | Czas generacji CSV |
|---|---|---|
| H. pylori | 1.7 Mbp | ~50 s |
| B. subtilis | 4.2 Mbp | ~2 min 15 s |
| M. tuberculosis | 4.4 Mbp (3376 junctions, duzo wiecej niz u innych) | ~4 min 40 s |

Trening: 2 epoki na ~5900 wierszach (3 male genomy) - ok. 30 s na CPU.
Pelny trening (50 epok, early_stop=10, ~90 genomow E. coli) na GPU: budzet
w SLURM-skryptach to 6h dla glownych 4 wariantow, 3h dla wariantow z
powtorzeniami, 30 min dla solo/ewaluacji - to gorne limity czasu w kolejce
(`#SBATCH --time`), nie zmierzony rzeczywisty czas; rzeczywisty przebieg byl
krotszy, ale nie mamy dokladnych liczb zapisanych z tamtego przebiegu.

## Determinizm

Przy tym samym seedzie i tym samym sprzecie (CPU albo ten sam model GPU)
wyniki sa deterministyczne (`--seed`, ustawiane w `ml/main.py` przez
`torch.manual_seed`/`numpy.random.seed`). Miedzy CPU i GPU, i miedzy roznymi
modelami GPU, wyniki moga sie nieznacznie roznic - rzedu dziesiatych czesci
punktu procentowego w `junction_accuracy` - z powodu niedeterminizmu
niektorych operacji CUDA i roznic w zaokragleniach float. To nie jest blad;
nie oczekuj bitowej identycznosci miedzy sprzetem.

## Odtwarzalnosc: ml/ przed i po migracji na uv

Grupa A/B (`RAPORT_TRANSFER.md`) byla trenowana na klastrze gdy `ml/`
jeszcze uzywalo `requirements.txt` z nieprzypietymi zakresami wersji
(`torch>=2.0`, `polars>=0.20`, `numpy>=1.24`, `scikit-learn>=1.3`, ...) i
bez pliku lock - `uv pip install -r requirements.txt` resolvowal co popadlo
w dniu uruchomienia. Nie ma zapisu, jakie wersje faktycznie trafily do
treningu; nie da sie ich odtworzyc.

Commit `b3e2d48` (po grupie A/B) zmigrowal `ml/` na projekt `uv`
(`pyproject.toml` + `uv.lock`), co od tej pory przypina dokladne wersje.
Obraz Docker i wszystkie skrypty w tym katalogu uzywaja `uv sync --frozen`,
co dla grup `cpu`/`cu128` daje m.in.:

| Pakiet | Wersja (cpu) | Wersja (cu128) |
|---|---|---|
| torch | 2.14.0+cpu | 2.11.0+cu128 |
| polars | 1.44.1 | 1.44.1 |
| numpy | 2.4.6 | 2.5.3 |
| scikit-learn | 1.9.0 | 1.9.0 |
| matplotlib | 3.11.1 | 3.11.1 |
| seaborn | 0.13.2 | 0.13.2 |
| wandb | 0.29.0 | 0.29.0 |

Wynik: kazdy trening od `b3e2d48` wzwyz jest odtwarzalny co do wersji
pakietow; treningi grupy A/B nie sa, i to jest ograniczenie tamtego
raportu, nie tylko formalnosc - torch miedzy wersjami czasem zmienia
domyslne zachowanie niektorych operacji.

(Analogiczna, mniejsza nieprzypiecie: `Cargo.lock` jest w `.gitignore`, wiec
wersje crate'ow Rusta (`clap`, `rand`, `serde`, ...) takze plywaja miedzy
`cargo build` w roznym czasie; `Cargo.toml` ma tylko dolne ograniczenia.)

## Test dymny

`smoke_test_docker.sh` generuje CSV dla 3 malych lokalnych genomow
(`genomes/hpylori.fna`, `bsubtilis.fna`, `mtuberculosis.fna` - H. pylori,
B. subtilis, M. tuberculosis), trenuje 2 epoki, ewaluuje. Dziala w pelni bez
sieci (zweryfikowane z `docker run --network none`) - nie pobiera nic z
NCBI, bo uzywa genomow juz w repo. To jest zamierzone: test dymny ma
sprawdzic, ze obraz sam w sobie dziala od poczatku do konca, nie ze NCBI
jest dostepne.
