# Raport: Asemblacja de novo DNA z użyciem ML

## 1. Opis rozwiązania

Projekt implementuje asembler DNA oparty na **grafie de Bruijn** napisany w języku Rust. Asembler rekonstruuje oryginalny genom z nakładających się odczytów.

### Algorytm asemblacji

1. **Budowanie grafu de Bruijn**: każdy odczyt jest dzielony na k-mery długości *k*. Wierzchołki grafu to (*k*−1)-mery, krawędzie reprezentują k-mery. Waga krawędzi = liczba odczytów potwierdzających dane przejście (pokrycie).
2. **Traversal grafu**: asembler przechodzi graf zachłannie, wydłużając kontig dopóki wychodzi dokładnie jedna krawędź z węzła. W **węźle rozejścia** (out-degree ≥ 2) kontig jest kończony. ML ma tutaj pomóc łączyć te contigi, zamiast za każdym razem je ciąć.
3. **Wyjście**: kontigi zapisywane w formacie FASTA.

## 2. Porównanie z dnaasm (wkusmirek/dnaasm)

Eksperyment: te same **928 331 bezbłędnych odczytów** (długość 150 bp, pokrycie 30×, E. coli K-12 MG1655, 4 641 652 bp), parametr k = 31, tryb single-end.

| Metryka | **Nasz asembler** | **dnaasm** |
|---|---|---|
| Liczba kontigów (≥ 500 bp) | 801 | 687 |
| Sumaryczna długość | 4 419 383 bp | 9 010 347 bp |
| Największy kontig | 138 202 bp | 127 976 bp |
| **N50** | **12 307 bp** | 23 402 bp* |
| **NG50** | 11 089 bp | 41 229 bp* |
| Pokrycie genomu | 94,93% | 97,13%* |
| **Współczynnik duplikacji** | **1,003** | **1,999** |
| Błędne złożenia | **0** | **0** |
| Niezgodności / indels | **0** | **0** |

\*Pozornie lepsze wyniki dnaasm wynikają ze współczynnika duplikacji ~2,0: dnaasm przetwarza domyślnie obie nici DNA, produkując każdy kontig dwukrotnie. Po uwzględnieniu duplikacji oba asemblery rekonstruują porównywalną część genomu.

**Wnioski:** Nasz asembler poprawnie rekonstruuje bezbłędne odczyty (brak błędnych złożeń, niezgodności i duplikacji). Pokrycie genomu 94,9% wynika z nieciągłości w węzłach rozejścia (skrzyżowania), które są celowo pozostawione jako miejsca interwencji modelu ML.

## 3. Koncepcja rozwiązania z ML

### Problem
W węzłach rozejścia grafu de Bruijn asembler nie wie którą krawędź wybrać, więc tnie kontig i tworzy osobne ścieżki. Celem modelu jest **wybór poprawnej gałęzi** na podstawie kontekstu sekwencji.

### Architektura: sieć syjamska CNN (MV1)

Dla każdego skrzyżowania model otrzymuje trzy sekwencje DNA i wektor cech pokrycia:
- **context_seq** - sekwencja nukleotydów *przed* węzłem (historia tego co assembler zdążył złożyć)
- **branch_A_seq**, **branch_B_seq** - kandydaci na kontynuację (pierwsze nukleotydy każdej gałęzi)
- **cov_feat** - 6 liczb: pokrycie każdej gałęzi, stosunek max/suma, łączna liczba odczytów

#### Krok 1 - kodowanie sekwencji (one-hot)

Każda sekwencja DNA jest zamieniana na macierz 4 × L, gdzie 4 to kanały (A, C, G, T), a L to długość sekwencji. Każda pozycja to wektor `[1,0,0,0]` dla A, `[0,1,0,0]` dla C itd.

#### Krok 2 - enkodery CNN (dwa oddzielne, ale identyczne w architekturze)

Model posiada **dwa enkodery**: `context_encoder` i `branch_encoder`. Oba mają identyczną strukturę (3 warstwy Conv1D + BatchNorm + ReLU -> Global Average Pooling), ale uczą się niezależnie. Kontekst i gałąź mogą mieć inny charakter statystyczny, więc osobne wagi dają modelowi swobodę specjalizacji.

```
Conv1d(4 -> 32, kernel=5)  + ReLU + BatchNorm
Conv1d(32 -> 64, kernel=3) + ReLU + BatchNorm
Conv1d(64 -> 64, kernel=3) + ReLU + BatchNorm
GlobalAveragePooling (srednia po dlugosci)  -->  wektor 64d
```

Global Average Pooling (srednia po wszystkich pozycjach) sprawia, ze model akceptuje sekwencje dowolnej dlugosci - nie jest wymagana stala dlugosc wejscia.

`context_encoder` przetwarza `context_seq` -> `embedding_ctx` (64d).
`branch_encoder` przetwarza `branch_A_seq` -> `embedding_A` (64d), a nastepnie **te same wagi** uzywane sa dla `branch_B_seq` -> `embedding_B` (64d). Obie galezi sa oceniane identycznym enkoderem.

#### Krok 3 - scorer (jeden wspoldzielony MLP)

Dla kazdej galezi budowany jest wektor wejsciowy przez konkatenacje:

```
[embedding_ctx | embedding_A | cov_feat]  -->  scorer  -->  score_A  (1 liczba)
[embedding_ctx | embedding_B | cov_feat]  -->  scorer  -->  score_B  (1 liczba)
```

Scorer to MLP z trzema warstwami:
```
Linear(64 + 64 + 6 = 134  ->  64) + ReLU + Dropout(0.2)
Linear(64 -> 32) + ReLU
Linear(32 ->  1)                     -->  logit (surowy wynik)
```

Ten sam scorer jest wywolywany dwukrotnie - raz dla A, raz dla B - z tymi samymi wagami. Dzieki temu model ocenia kazda galaz w identyczny sposob; jedyna roznica to embedding gałezi.

#### Krok 4 - porownanie i decyzja

Dwa surowe wyniki (`score_A`, `score_B`) sa traktowane jako logity klasyfikatora binarnego:

- **Podczas treningu**: CrossEntropyLoss(`[score_A, score_B]`, label) uczy model, zeby poprawna galaz miala wyzszy wynik.
- **Podczas inferencji**: `argmax(score_A, score_B)` - wybierana jest galaz z wyzszym wynikiem.

Model nie mowi wprost "ta galaz jest poprawna" - mowi "ta galaz *bardziej pasuje do kontekstu* niz tamta". Jest to decyzja relatywna (porownawcza), nie absolutna.

```
WEJSCIE                        ENKODERY CNN                          KLASYFIKATOR
----------------------------------------------------------------------------------
context_seq                    +---------------------+
(sekwencja przed               |  context_encoder    |
 skrzyzowaniem,   --one-hot--> |  Conv1D x3          | --> emb_ctx (64d) --+
 np. 100 nt)                   |  GlobalAvgPool       |                    |
                               +---------------------+                    |
                                                                           |
branch_A_seq                   +---------------------+                    |  +-----------+
(kandydat A,      --one-hot--> |  branch_encoder     | --> emb_A  (64d) --+->|           |
 np. 100 nt)                   |  Conv1D x3          |                    |  |  scorer   | --> score_A
                               |  GlobalAvgPool       |                    |  |  MLP 3xFC |
                               +---------------------+                    |  |           | --> score_B
branch_B_seq                   +---------------------+  (te same wagi)    |  |  (te same |
(kandydat B,      --one-hot--> |  branch_encoder     | --> emb_B  (64d) --+->|   wagi)   |
 np. 100 nt)                   |  Conv1D x3          |                    |  +-----------+
                               |  GlobalAvgPool       |                    |
                               +---------------------+                    |
                                                                           |
cov_feat                                                  (6 liczb) ------+
(pokrycie galezi,
 stosunek max/sum,
 liczba odczytow)
----------------------------------------------------------------------------------
WYJSCIE: argmax(score_A, score_B)  ->  indeks wybranej galezi (0=A, 1=B)
```

### Dane treningowe i ground truth

Generator w Rust wyszukuje `context_seq` w genomie referencyjnym i sprawdza jaki nukleotyd następuje bezpośrednio po nim. Jeśli kontekst pojawia się w genomie jednoznacznie to ground truth znany. Jeśli w wielu miejscach z różnymi kontynuacjami (powtórzenia genomowe) to niestety skrzyżowanie oznaczone jako *ambiguous* i odrzucane z treningu.

#### Problem pozyskiwania danych treningowych

Głównym wyzwaniem jest wysoki odsetek skrzyżowań **ambiguous** - sięgający 78-83% w zależności od organizmu. Wynika to z dwóch przyczyn:

1. **Powtórzenia genomowe** - ten sam fragment sekwencji `context_seq` pojawia się w wielu miejscach genomu, po których następują różne nukleotydy. Algorytm ground truth słusznie odrzuca takie przypadki jako niejednoznaczne, bo właściwa kontynuacja zależy od miejsca w genomie, a nie od samej sekwencji kontekstu.

2. **Dobór referencji** - ground truth jest wyznaczany przez wyszukiwanie liniowe `context_seq` w pełnym genomie referencyjnym. Przy krótkim kontekście (100 nt) wiele fragmentów jest na tyle powtarzalnych, że metoda nie jest w stanie jednoznacznie wskazać miejsca. Zwiększenie długości kontekstu (`--context_len`) lub pokrycia odczytów (`--coverage`) pozwala pozyskać więcej labeled próbek, ale kosztem rozmiaru grafu i czasu asemblacji.

#### Statystyki zbioru treningowego

| Organizm | Rozmiar genomu | Skrzyżowań łącznie | **Labeled** | Ambiguous |
|---|---|---|---|---|
| *E. coli* K-12 MG1655 | 4,6 Mbp | 703 | **153 (21,8%)** | 550 (78,2%) |
| *S. cerevisiae* S288C | 12,1 Mbp | 3 742 | **636 (17,0%)** | 3 106 (83,0%) |
| **Łącznie** | - | **4 445** | **789 (17,8%)** | 3 656 (82,2%) |

Drożdże (*S. cerevisiae*) dostarczyły ponad czterokrotnie więcej skrzyżowań niż *E. coli*, co wynika z większego i bardziej złożonego genomu. Niższy odsetek labeled w drożdżach (~17% vs ~22%) potwierdza że większy genom z większą liczbą powtórzeń skutkuje trudniejszym wyznaczaniem ground truth.


### Augmentacja danych
Każda próbka jest rozszerzana na dwa niezależne sposoby:

1. **Zamiana pozycji A/B** — ta sama para `(context, correct, wrong)` pojawia się raz z `label=0` (correct na pozycji A) i raz z `label=1` (correct na pozycji B). Zapobiega to biasowi pozycji — model uczy się oceniać sekwencje relatywnie, a nie "zawsze wybieraj A".

2. **Reverse complement** — sekwencja DNA ma dwie nici; RC `(ctx, correct, wrong)` to taka sama informacja biologiczna, tylko czytana od drugiego końca.

Oba zabiegi są niezależne, więc każde skrzyżowanie daje 4 pary (1 oryginał × 2 pozycje × 2 nici), co **czterokrotnie** zwiększa rozmiar zbioru treningowego.

Po augmentacji łączny zbiór treningowy dla próbek Drożdże (S. cerevisiae) i E. coli wynosi ~**3 156 par** (789 skrzyżowań × 4 warianty).

### Wyniki (E. coli + S. cerevisiae, 789 skrzyżowań z etykietą)

| Metryka | Wartość |
|---|---|
| Baseline (max-coverage) | **67,6%** (533/789 skrzyżowań) |
| Najlepsza val. junction accuracy | **71,30%** (epoka 3) |
| Najlepsza val. pair accuracy | 73,04% |
| Najlepsza val. F1 | 0,8442 |

**Model przebija baseline** o ~3,7 pp. Wcześniej (sam *E. coli*, ~131 skrzyżowań) model osiągał wyniki zbliżone do heurystyki. Dodanie danych z drożdży (636 dodatkowych labeled skrzyżowań z innego organizmu) poprawiło generalizację - model nauczył się cech sekwencji niezależnych od konkretnego genomu.

**Early stopping** nastąpił w epoce 13 (brak poprawy przez 10 epok od epoki 3), co sugeruje że przy ~789 labeled przykładach model osiąga swoje optimum szybko. Dalszy wzrost jakości wymagałby zwiększenia zbioru treningowego (więcej genomów lub wyższe `--coverage`).




## Słownik pojęć

| Pojęcie | Znaczenie |
|---|---|
| **odczyt** (*read*) | Krótki fragment DNA odczytany przez sekwenator (tu: 150 bp). Sekwenator nie czyta całego genomu naraz — wytwarza miliony takich fragmentów. |
| **kontig** (*contig*) | Ciągły fragment genomu zrekonstruowany przez asembler przez składanie nakładających się odczytów. Im dłuższy i mniej liczny, tym lepsza asemblacja. |
| **k-mer** | Podciąg odczytu o długości *k*. Graf de Bruijn jest zbudowany z k-merów — każdy odczyt rozkłada się na wiele nakładających się k-merów. |
| **węzeł rozejścia** (*junction*) | Miejsce w grafie de Bruijn, gdzie contig mógłby iść w kilku kierunkach (out-degree ≥ 2). Asembler musi zdecydować którą gałąź wybrać, albo przerwać contig. |
| **ground truth** | Poprawna odpowiedź dla danego węzła rozejścia, wyznaczona automatycznie przez wyszukanie sekwencji kontekstu w genomie referencyjnym. Jeśli kontekst pojawia się w genomie tylko raz — wiadomo jaki nukleotyd następuje po nim, więc ground truth jest znany. |
| **ambiguous** | Węzeł rozejścia, dla którego **nie można wyznaczyć ground truth** — sekwencja kontekstu pojawia się w genomie wielokrotnie, po różnych nukleotydach. Takie skrzyżowania są odrzucane z treningu, bo nie wiadomo jaka jest prawidłowa odpowiedź. |
| **labeled** | Węzeł rozejścia z *poznanym* ground truth (przeciwieństwo *ambiguous*). Tylko te próbki trafiają do treningu. |
| **baseline** | Prosta heurystyka bez ML: wybieraj zawsze gałąź o wyższym pokryciu odczytów. Stanowi punkt odniesienia dla modelu. |
| **reverse complement (RC)** | Komplementarna nić DNA czytana od końca do początku. Ponieważ DNA jest dwuniciowe, każda sekwencja `ACGT...` ma swoje RC `...ACGT`, które koduje tę samą informację biologiczną z drugiej strony. |
| **N50 / NG50** | Statystyki jakości asemblacji. N50: połowa zsumowanej długości kontigów pochodzi z kontigów o długości ≥ N50. NG50: analogicznie, ale względem długości genomu referencyjnego. |