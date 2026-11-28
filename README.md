# Projekt

De novo asembler DNA na grafie de Bruijn z niewielkim modelem PyTorch,
ktory rozstrzyga wezly rozejscia. Klasyczny asembler przechodzi graf zachlannie
i tnie kontig na kazdym skrzyzowaniu (wezel gdzie out-degree >= 2), bo nie wie ktora
galaz kontynuuje genom. Zadaniem modelu jest wskazac wlasciwa galaz na podstawie
lokalnego kontekstu sekwencji, dzieki czemu takie kontigi mozna laczyc, a nie
przecinac.

Podzial odpowiedzialnosci miedzy jezykami:

- Rust (`src/`) buduje graf, symuluje odczyty, przechodzi graf i zapisuje po
  jednym wierszu treningowym na skrzyzowanie do `data/training/*.csv`.
- Python (`ml/`) czyta te CSV i odpowiada za architekture modelu, trening i
  analize.

## Dokumentacja

- `RAPORT.md` - metody i wyniki dla pojedynczego organizmu (cerevisiae). Wniosek:
  model nie bije baseline max-coverage, sufit ok. 58%.
- `docs/RAPORT_transfer_A_B.md` - transfer miedzy genomami. Trening na wielu szczepach
  E. coli bije baseline o ok. 15 pp na nowych szczepach tego samego gatunku
  (grupa A); efekt nie rozciaga sie na cala rodzine Enterobacteriaceae (grupa B).
- `docs/RAPORT_tuning_modelu.md` - strojenie MV1 na grupie A (harmonogram LR,
  glebokosc, szerokosc). Dluzszy trening z cosine i 4 bloki resztkowe daja
  ok. 79% zero-shot wobec ok. 71% przed strojeniem.
- `experiments/group_a/README.md` - jak odtworzyc eksperyment grupy A.

Glowny wynik pracy, transfer miedzy genomami w grupach A i B, opisuje
`docs/RAPORT_transfer_A_B.md`. Sekcja 7 raportu zbiera ograniczenia techniczne
znalezione przy audycie kodu oraz wynik odtworzenia na czystym klonie. Skrypty,
ktore wygenerowaly te wyniki (SLURM i Docker), sa w `experiments/hpc/`; jego
README opisuje uruchomienie i odtwarzalnosc srodowiska.

## Szybki start

Budowanie: `cargo build --release` (binarka w `target/release/dna-assembler`).

Generowanie danych treningowych (cerevisiae, k=21, pokrycie 100x):

```
./target/release/dna-assembler pipeline --reference genomes/cerevisiae.fna \
  --k 21 --coverage 100 --gen-train-data --results-dir results/cerevisiae_k21
```

Srodowisko Pythona (`ml/`): `cd ml && uv sync`. Na maszynie z GPU:
`cd ml && uv sync --no-group cpu --group cu128`.

Uruchomienie skryptu Pythona: `cd ml && uv run python3 <sciezka>`.
Testy: `cd ml && uv run pytest`.

## Trening i ewaluacja modelu

Trening uruchamia `cd ml && uv run python3 main.py train ...`; pelna lista flag
jest w `uv run python3 main.py train --help`. Flagi strojenia (`--lr_schedule`,
`--warmup_epochs`, `--width_mult`, `--n_blocks`, `--residual`, `--batchnorm`,
`--split_by`) maja wartosci domyslne odpowiadajace dotychczasowemu treningowi,
wiec bez nich wyniki sa takie same jak wczesniej. Pelna konfiguracja przebiegu
(wszystkie flagi), commit i najlepsza epoka trafiaja do
`<ckpt_dir>/<run_name>_metrics.json`.

**Walidacja przy `--split_by row` (domyslnie) jest zawyzona.** Podzial idzie po
wierszach CSV, a w `cerevisiae_training.csv` 71% oznaczonych wierszy dzieli
`node_seq` z innym wierszem (REQ022), wiec prawie identyczne przyklady trafiaja
jednoczesnie do train i val. Val accuracy, early stopping i wybor najlepszego
checkpointu sa przez to optymistyczne. `--split_by node_seq` trzyma wszystkie
wiersze z tym samym `node_seq` po jednej stronie, a `--split_by genome` (dla
katalogu z wieloma CSV) caly genom. Wyniki zero-shot na osobnych genomach
testowych nie sa tym dotkniete.

`main.py evaluate --output_json <plik>.json` zapisuje obok predykcje per
skrzyzowanie w `<plik>_predictions.csv` (wszystkie skrzyzowania, takze
Ambiguous i NotFound; format JSON bez zmian). Rozbicie dokladnosci modelu i
baseline max-coverage wedlug liczby galezi, `repeat_len`, pokrycia, typu
etykiety i organizmu daje
`uv run python3 analysis/error_analysis.py <katalog_z_predykcjami> --out_csv <plik>.csv`
(tabela CSV i podsumowanie `<plik>_summary.txt`).

## Wymagania i sledzenie

Wymagania sa w Doorstopie, w trzech dokumentach w `reqs/`: `CAT` (kategorie),
`REQ` (wymagania, kazde podlinkowane do kategorii) i `TST` (testy, kazdy
podlinkowany do wymagania, z `references` do pliku testu). Test, ktorego
jeszcze nie ma, ma tekst `TODO` i nie ma referencji. Wpisy nienormatywne
(`normative: false`) to znane ograniczenia i zalozenia; nie maja linkow, bo
Doorstop na to nie pozwala.

Doorstop jest zaleznoscia dev w `ml/`, wiec komendy uruchamia sie przez `uv` z
katalogu glownego repo:

```
uv run --project ml doorstop --error-all          # walidacja, ostrzezenia lamia wynik
uv run --project ml doorstop review all           # po zatwierdzeniu zmienionej tresci
uv run --project ml doorstop clear all            # po zatwierdzeniu zmienionych linkow
uv run --project ml doorstop publish all ./public # HTML i traceability.csv
```

`public/` jest w `.gitignore`.
