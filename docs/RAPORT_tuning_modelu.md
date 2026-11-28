# Raport: strojenie modelu MV1 (etapy 1-3)

Kontynuacja `docs/RAPORT_transfer_A_B.md`. Tamten raport pokazal transfer miedzy genomami w grupie A (E. coli) przy treningu przez 50 epok. Tutaj sprawdzamy, ile da sie zyskac dzieki dluzszemu treningowi, harmonogramowi learning rate i zmianom architektury, bez zmiany danych i cech wejsciowych.

## Spis tresci

1. [Metoda](#1-metoda)
2. [Zmiany w kodzie](#2-zmiany-w-kodzie)
3. [Konfiguracje](#3-konfiguracje)
4. [Wyniki](#4-wyniki)
5. [Analiza bledow](#5-analiza-bledow)
6. [Wnioski](#6-wnioski)
7. [Ograniczenia](#7-ograniczenia)
8. [Etap 3 (w toku)](#8-etap-3-w-toku)

---

## 1. Metoda

- **Dane i metryka:** jak w raporcie transferu. Trening na `group_a/farthest_train` (90 genomow), test zero-shot na `group_a/farthest_test` (10 genomow, 6345 ocenianych junctionow). Metryka to `junction_accuracy` wazona `n_junctions`. Baseline max-coverage: 56,33%.
- **Stale ustawienia:** `--context_len 150`, `--no_cov`, `--augment`.
- **Wezel:** wszystkie przebiegi na glasser (RTX 4090). Wynik zalezy od modelu GPU (sekcja 7.8 raportu transferu), dlatego porownujemy tylko przebiegi z jednego wezla.
- **Seedy:** 42 dla wszystkich konfiguracji (etap 1), 123 i 7 dla trzech wybranych (etap 2). Trening jest deterministyczny dla danego seeda, wezla i kodu: base z seedem 42 daje 71,57%, tyle samo co odtworzenie 94577 z raportu transferu.
- **Kod:** wszystkie przebiegi etapow 1-2 uruchomiono z commita 4e63f03 (`git_dirty=False`). Wyniki sprawdzono niezaleznym skryptem: metryki z plikow JSON zgadzaja sie z predykcjami per junction co do 0,01 pp, kazdy przebieg ma 10 plikow (po jednym na genom testowy) i 6345 junctionow.
- **Skrypty:** `experiments/hpc/tune_array.sh` i `tasks/tune.sh` (konfiguracje x seedy w jednej tablicy zadan SLURM), agregacja `aggregate_tuning.py`. Skrypty sa na HPC i nie ma ich jeszcze w repozytorium. Wyniki per przebieg: `results/tuning/<konfiguracja>_s<seed>/`.

## 2. Zmiany w kodzie

Wszystkie nowe flagi maja wartosci domyslne odpowiadajace dotychczasowemu treningowi. Model z domyslnymi argumentami ma te same nazwy parametrow, kolejnosc inicjalizacji i liczbe parametrow co oryginalny MV1 (49 473 przy `--no_cov`), wiec zaseedowany model dostaje te same wagi, a stare checkpointy sie wczytuja. Pilnuja tego testy w `ml/tests/models/test_mv1.py`.

MV1 ma dwa osobne enkodery CNN (kontekst i galaz) o tej samej budowie; enkoder galezi jest wspoldzielony przez obie galezie. Blok enkodera to Conv1d -> ReLU -> BatchNorm1d; pierwszy blok ma jadro 5, kolejne jadro 3. Domyslnie sa trzy bloki o szerokosciach 32, 64 i 64 (ostatni blok zawsze ma `--embed_dim` kanalow), potem Global Average Pooling i dropout. Scorer to MLP 128 -> 64 -> 32 -> 1, wspolny dla obu galezi.

- `--lr_schedule {constant,cosine,plateau}` (domyslnie `plateau`) i `--warmup_epochs` (domyslnie 0): harmonogram learning rate. Rozgrzewka podnosi LR liniowo, `lr * epoka / warmup_epochs` w epokach 1..`warmup_epochs`. `cosine` zmniejsza potem LR po krzywej cosinusa od `--lr` w pierwszej epoce po rozgrzewce do prawie zera w epoce `--epochs` (dla cos150 ok. 1e-7, zero przypadloby na epoke 151). `plateau` (dotychczasowe zachowanie) to `ReduceLROnPlateau` na walidacyjnej junction accuracy z cierpliwoscia 5 epok i wspolczynnikiem 0,5. `constant` trzyma `--lr` przez caly trening.
- `--n_blocks` (domyslnie 3): liczba blokow konwolucyjnych w kazdym z enkoderow. Dodatkowe bloki maja 64 kanaly i jadro 3.
- `--residual`: polaczenie resztkowe w kazdym bloku obu enkoderow, wynik to `blok(x) + skip(x)`. Skip jest tozsamoscia, gdy liczba kanalow sie nie zmienia, a konwolucja 1x1, gdy sie zmienia. Przy `--n_blocks 4` (kanaly 4 -> 32 -> 64 -> 64 -> 64) konwolucje 1x1 maja dwa pierwsze bloki, a dwa ostatnie maja skip tozsamosciowy.
- `--width_mult` (domyslnie 1,0): mnoznik szerokosci ukrytych blokow konwolucyjnych (32 i 64). Ostatni blok zostaje przy `--embed_dim`, wiec rozmiar embeddingu i scorer sie nie zmieniaja. Przykladowo `--width_mult 4` daje szerokosci 128, 256 i 64.
- `--batchnorm`: BatchNorm1d po kazdej ukrytej warstwie Linear scorera (Linear -> BatchNorm -> ReLU -> Dropout). Enkodery maja BatchNorm zawsze.
- `--split_by {row,node_seq,genome}` (domyslnie `row`): jednostka podzialu train/val.
- `evaluate --output_json <plik>.json` zapisuje obok predykcje per junction (`<plik>_predictions.csv`), a `ml/analysis/error_analysis.py` rozbija na ich podstawie dokladnosc modelu i baseline.

## 3. Konfiguracje

Liczbe parametrow policzono, tworzac model z odpowiednimi flagami (bez treningu, `--no_cov`).

| Nazwa | Flagi (ponad stale ustawienia) | Parametry |
|---|---|---|
| base | `--epochs 50 --early_stop 10` (plateau) | 49 473 |
| cos150 | `--epochs 150 --early_stop 150 --lr_schedule cosine --warmup_epochs 3` | 49 473 |
| cos150_w2 | jak cos150 + `--width_mult 2` | 112 769 |
| cos150_w4 | jak cos150 + `--width_mult 4` | 313 089 |
| cos150_deep_res | jak cos150 + `--n_blocks 4 --residual` | 78 977 |

Przy `--early_stop 150` i `--epochs 150` wczesne zatrzymanie nie moze zadzialac, wiec konfiguracje cos150 zawsze ucza sie przez pelne 150 epok. Base uzywa domyslnych `--epochs 50` i `--early_stop 10`.

## 4. Wyniki

Zero-shot, grupa A farthest, glasser.

### 4.1 Etap 1 (seed 42)

| Konfiguracja | Zero-shot | Roznica vs base | Najlepsza epoka |
|---|---|---|---|
| cos150_deep_res | 79,02% | +7,45 pp | 142 |
| cos150 | 76,50% | +4,93 pp | 139 |
| cos150_w2 | 76,36% | +4,79 pp | 150 |
| cos150_w4 | 76,22% | +4,65 pp | 145 |
| base | 71,57% | 0 | 46 |

### 4.2 Etap 2 (seedy 42, 123, 7)

| Konfiguracja | Zero-shot (mean +/- std) | Min-max | Roznica vs base | Najlepsze epoki | Czas treningu |
|---|---|---|---|---|---|
| cos150_deep_res | 79,28% +/- 0,23pp | 79,02-79,46% | +7,95 pp | 142-149 | 3 h 15-17 min |
| cos150 | 76,52% +/- 0,10pp | 76,44-76,63% | +5,20 pp | 139-141 | 3 h 06-11 min |
| base | 71,33% +/- 0,89pp | 70,34-72,07% | 0 | 44-48 | 1 h 02 min |

Czas treningu dotyczy glasser. cos150_w4 (tylko etap 1) trenowal 3 h 23 min. Roznice w tabeli liczono z niezaokraglonych srednich, stad np. +5,20 pp przy srednich 76,52% i 71,33%.

Wzgledem baseline max-coverage (56,33%): cos150_deep_res +22,95 pp, cos150 +20,19 pp, base +15,00 pp.

Zrodlo: `aggregate_tuning.py` na HPC (2026-11-28), surowe pliki w `$HOME/dna-assembler-results/tuning-2026-11/`.

## 5. Analiza bledow

Checkpoint 108011 (h32, zero-shot 73,27%), czyli model sprzed strojenia. Skrot z `error_analysis.py`: 18 769 junctionow, z czego 6345 ocenianych (Known), reszta to Ambiguous bez etykiety.

| Podzial | Model | Baseline |
|---|---|---|
| ogolem | 73,27% | 56,33% |
| 2 galezie (6116) | 73,69% | 56,34% |
| 3 galezie (199) | 64,32% | 57,29% |
| 4 galezie (30) | 46,67% | 46,67% |
| repeat_len < 21 (6206) | 73,36% | 56,20% |
| repeat_len >= 21 (139) | 69,06% | 61,87% |
| cov_ratio_max 0,00-0,50 (100) | 57,00% | 41,00% |
| cov_ratio_max 0,50-0,60 (4356) | 73,26% | 49,82% |
| cov_ratio_max 0,60-0,75 (1136) | 70,16% | 65,58% |
| cov_ratio_max 0,75-1,00 (753) | 80,21% | 82,07% |

Prawie wszystkie oceniane junctiony maja dwie galezie. 69% z nich to niemal remis pokrycia, gdzie baseline jest rzutem moneta, a model korzysta z sekwencji. Jedyny przedzial, w ktorym baseline wygrywa, to `cov_ratio_max >= 0,75` (12% ocenianych).

Przedzialy `cov_ratio_max` roznia sie od przedzialow w `docs/RAPORT_transfer_A_B.md` (sekcje 5.2 i 5.4), wiec tych tabel nie mozna porownywac bin do binu. Cztery przedzialy powyzej obejmuja razem wszystkie 6345 ocenianych junctionow.

Wynik dla `repeat_len >= 21` dotyczy checkpointu 108011 (h32, 73,27%). W raporcie transferu (oryginalny checkpoint z wrzesnia) ten przedzial ma 51,80% przy tym samym n=139. Proba jest mala (blad standardowy ok. 4 pp), a wynik zalezy od genomu, wiec tak duza roznica miedzy checkpointami jest mozliwa.

## 6. Wnioski

1. Najwiekszy zysk daje dluzszy trening z harmonogramem cosine (+5,2 pp). Base dobiega do limitu 50 epok we wszystkich seedach (najlepsze epoki 44-48 z 50), wiec to limit epok, a nie wczesne zatrzymanie, konczy jego trening.
2. Cztery bloki z polaczeniami resztkowymi dokladaja +2,8 pp ponad cos150.
3. Szerszy model (`--width_mult` 2 i 4) nie poprawia wyniku (jeden seed, w granicach szumu).
4. Rozrzut miedzy seedami jest maly dla dluzszych konfiguracji (0,10-0,23 pp) i wiekszy dla base (0,89 pp), wiec roznice 2,8-5,2 pp nie sa szumem.
5. Najlepsze epoki 139-150 ze 150 moga oznaczac, ze trening jeszcze nie zbiegl, sprawdza to etap 3. Czesc tego efektu wynika z samego harmonogramu: przy cosine LR spada do prawie zera dopiero w ostatnich epokach, wiec najlepszy checkpoint pod koniec jest oczekiwany nawet przy zbieznym treningu.

## 7. Ograniczenia

- Konfiguracje wybieralismy na tych samych 10 genomach testowych, na ktorych je oceniamy, co daje optymistyczne obciazenie. Wynik wymaga potwierdzenia w pozostalych wariantach (A random, B farthest, B random).
- Wyniki pochodza tylko z glasser. Inny model GPU zmienial wynik o ok. 1,7 pp (raport transferu, sekcja 7.8).
- cos150_w2 i cos150_w4 maja tylko jeden seed (etap 1), a etap 2 obejmuje tylko trzy konfiguracje.
- Podzial train/val po wierszach CSV daje zawyzona walidacje (raport transferu, sekcja 7.3), a checkpoint jest wybierany po val.

## 8. Etap 3 (w toku)

Seed 42, konfiguracje:

| Nazwa | Flagi | Parametry |
|---|---|---|
| cos300_deep_res | jak cos150_deep_res, 300 epok | 78 977 |
| cos150_deep6_res | jak cos150 + `--n_blocks 6 --residual` | 128 897 |
| cos150_deep_res_w2 | jak cos150_deep_res + `--width_mult 2` | 245 633 |
| cos150_deep_res_bn | jak cos150_deep_res + `--batchnorm` | 79 169 |

Wyniki zostana dopisane.
