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
- `RAPORT_TRANSFER.md` - transfer miedzy genomami. Trening na wielu szczepach
  E. coli bije baseline o ok. 15 pp na nowych szczepach tego samego gatunku
  (grupa A); efekt nie rozciaga sie na cala rodzine Enterobacteriaceae (grupa B).
- `experiments/group_a/README.md` - jak odtworzyc eksperyment grupy A.

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
