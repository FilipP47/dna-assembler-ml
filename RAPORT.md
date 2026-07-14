# Raport: Asemblacja de novo DNA z użyciem ML

## 1. Opis rozwiązania

Projekt implementuje asembler DNA oparty na **grafie de Bruijn** napisany w języku Rust. Asembler rekonstruuje oryginalny genom z nakładających się odczytów.

### Algorytm asemblacji

1. **Budowanie grafu de Bruijn**: każdy odczyt jest dzielony na k-mery długości *k*. Wierzchołki grafu to (*k*−1)-mery, krawędzie reprezentują k-mery.
2. **Traversal grafu**: asembler przechodzi graf zachłannie, wydłużając kontig dopóki wychodzi dokładnie jedna krawędź z węzła. W **węźle rozejścia** (out-degree ≥ 2) kontig jest kończony. ML ma tutaj pomóc łączyć te contigi, zamiast za każdym razem je ciąć.
3. **Wyjście**: kontigi zapisywane w formacie FASTA.

## 2. Porównanie z dnaasm (wkusmirek/dnaasm)

Eksperyment: te same **928 331 bezbłędnych odczytów** (długość 150 bp, pokrycie 30×, E. coli K-12 MG1655, 4 641 652 bp), parametr k = 31, tryb single-end.

| Metryka | **Nasz asembler** | **dnaasm** |
|---|---|---|
| Liczba kontigów (≥ 500 bp) | 798 | 687 |
| Sumaryczna długość | 4 419 383 bp | 9 010 347 bp |
| Największy kontig | 138 202 bp | 127 976 bp |
| **N50** | **12 307 bp** | 23 402 bp* |
| **NG50** | 11 089 bp | 41 229 bp* |
| Pokrycie genomu | 94,93% | 97,13%* |
| **Współczynnik duplikacji** | **1,003** | **1,999** |
| Błędne złożenia | **0** | **0** |
| Niezgodności / indels | **0** | **0** |

\*Pozornie lepsze wyniki dnaasm wynikają ze współczynnika duplikacji ~2,0: dnaasm przetwarza domyślnie obie nici DNA, produkując każdy kontig dwukrotnie. Po uwzględnieniu duplikacji oba asemblery rekonstruują porównywalną część genomu.

**Wnioski:** Nasz asembler poprawnie rekonstruuje bezbłędne odczyty (brak błędnych złożeń, niezgodności i duplikacji). Pokrycie genomu 95,0% wynika z nieciągłości w węzłach rozejścia, które są celowo pozostawione jako miejsca interwencji modelu ML.

### Analiza rozkładu długości kontigów

Wyższa liczba kontigów naszego asemblera (798 vs ~344 unikalne dnaasm) wynika z dużej liczby **bardzo krótkich kontigów** powstających między gęsto upakowanymi węzłami rozejścia:

| Przedział długości | Liczba kontigów |
|---|---|
| < 65 bp | 4 395 |
| 65 – 499 bp | 1 005 |
| 500 – 999 bp | 173 |
| 1 000 – 9 999 bp | 509 |
| ≥ 10 000 bp | 116 |
| **Razem** | **6 198** |

Mediana długości kontiga to zaledwie **39 bp**. Gdy dwa węzły rozejścia leżą w genomie blisko siebie (odległość < ~65 bp), ścieżka między nimi daje miniaturowy kontig. QUAST ignoruje alignmenty krótsze niż 65 bp przy obliczaniu pokrycia genomu - stąd brakujące 5%. Są to obszary gęstych powtórzeń genomowych, gdzie skrzyżowania następują jedno po drugim.

dnaasm osiąga 97% pokrycia, bo uruchamia korekcję błędów grafu, która usuwa słabe krawędzie i scala krótkie ścieżki z dłuższymi sąsiadami. Nasz asembler świadomie pomija ten etap, ponieważ krótkie kontigi między węzłami rozejścia są potencjalnymi miejscami interwencji modelu ML.

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

Scorer to MLP z trzema warstwami (dropout po obu warstwach ukrytych):
```
Linear(64 + 64 + 6 = 134  ->  64) + ReLU + Dropout(0.3)
Linear(64 -> 32)                   + ReLU + Dropout(0.3)
Linear(32 ->  1)                     -->  logit (surowy wynik)
```

Dropout(0.3) stosowany jest rowniez po Global Average Pooling w obu enkoderach CNN.

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

Aktualny zbior danych to *S. cerevisiae* S288C wygenerowany potokiem Rust z wyzszym pokryciem niz wczesniejsze eksperymenty:

| Organizm | Rozmiar genomu | Wiersze w CSV | **Labeled** | Ambiguous |
|---|---|---|---|---|
| *S. cerevisiae* S288C | 12,1 Mbp | 10 840 | **3 939 (36,3%)** | 5 496 (50,7%) |

Wiersze nieoznaczone (10 840 - 3 939 - 5 496 = 1 405) to skrzyzowania odfiltrowane przez minimalne wymagania dlugosci sekwencji (context >= 10 bp, branch >= 5 bp). Wyzszy odsetek labeled (36% vs ~17% w starszym przebiegu) wynika z wyzszego pokrycia odczytow: wiecej odczytow na kazda pozycje genomu oznacza ze nawet krotsze konteksty wystarczaja do jednoznacznego wskazania miejsca w genomie referencyjnym.

We wczesniejszym eksperymencie (raport ponizej) uzywano lacznie *E. coli* + *S. cerevisiae* z lacznie 789 labeled skrzyzowaniami.


### Augmentacja danych
Każda próbka jest rozszerzana na dwa niezależne sposoby:

1. **Zamiana pozycji A/B** - ta sama para `(context, correct, wrong)` pojawia się raz z `label=0` (correct na pozycji A) i raz z `label=1` (correct na pozycji B). Zapobiega to biasowi pozycji - model uczy się oceniać sekwencje relatywnie, a nie "zawsze wybieraj A".

2. **Reverse complement** - sekwencja DNA ma dwie nici; RC `(ctx, correct, wrong)` to taka sama informacja biologiczna, tylko czytana od drugiego końca.

Oba zabiegi są niezależne, więc każde skrzyżowanie daje 4 pary (1 oryginał × 2 pozycje × 2 nici), co **czterokrotnie** zwiększa rozmiar zbioru treningowego.

Po augmentacji łączny zbiór treningowy dla próbek Drożdże (S. cerevisiae) i E. coli wynosi ~**3 156 par** (789 skrzyżowań × 4 warianty).

### Czy model sie uczy i czy przebija baseline?

Krotka odpowiedz: **model uczy sie, ale nie bije baseline w sposob powtarzalny i statystycznie wiarygodny.**

Dowodem ze model uczy sie czegos jest zachowanie epok: z wlaczonymi cechami pokrycia `best_epoch=1` (model natychmiast kopiuje heurystyke max-coverage rowna baseline), bez pokrycia `best_epoch=6+` (CNN musi szukac sygnalu w sekwencji DNA i potrzebuje wiecej iteracji). Na dlugich kontekstach (>80 bp, 44% danych) model bez pokrycia nieznacznie bije baseline o +2,3 pp, co potwierdza ze CNN wyciaga z sekwencji jakis sygnał. Nie jest to jednak efekt na tyle silny zeby przelozyc sie na ogolny wynik.

Wyniki 5-fold CV przy dwoch wartosciach k (stratyfikowany podzial, seed=42):

| Konfiguracja | Labeled | Baseline | Model 5-fold CV | vs baseline |
|---|---|---|---|---|
| k=21, *S. cerevisiae* | 3 939 | 58,4% | **57,5% ± 0,9%** | -0,9 pp |
| k=31, *S. cerevisiae* | 590 | 67,6% | **64,9% ± 2,3%** | -2,7 pp |

k=31 ma wyzszy baseline (67,6%) bo przy dluzszym k-merze skrzyzowania sa rzadsze i bardziej specyficzne - pokrycie jest tam silniejszym sygnalem. k=21 jest lepszy do treningu: 6,7x wiecej probek, nizsza wariancja wynikow miedzy foldami.

Wczesniejszy wynik 71,3% (pojedynczy random split na danych k=31 z E. coli + S. cerevisiae lacznie) nie jest powtarzalny: ten sam zbior z 5-fold CV daje 64,9% przy baseline 67,6%. Roznica wynikala prawdopodobnie z losowego podzialu na val z nizszym baseline, a nie z faktycznej przewagi modelu.

### Szczegolowe wyniki: S. cerevisiae k=21, 5-fold CV

Obecny zbior danych: 3 939 labeled skrzyzowań z jednego organizmu (*S. cerevisiae*, k=21). Ewaluacja opiera sie na stratyfikowanym 5-fold CV (seed=42).

| Metryka | Wartość |
|---|---|
| Baseline (max-coverage, globalny) | **58,4%** (2 299/3 939) |
| Baseline (max-coverage, val fold) | ~58,9% (srednia po foldach) |
| Val. junction accuracy - 5-fold | **57,5% ± 0,9%** |
| Val. F1 - 5-fold | 0,744 ± 0,010 |
| Best epoch (typowy) | 1-3 |

#### Diagnoza: coverage jako shortcut

Test polegal na trenowaniu modelu na tym samym foldzie z i bez cech pokrycia:

| Konfiguracja | Junc acc | vs baseline (val fold) | Best epoch |
|---|---|---|---|
| Baseline (max-cov, val fold) | 58,9% | - | - |
| Model z pokryciem (use_cov=True) | 57,1% | -1,8 pp | 1 |
| Model bez pokrycia (use_cov=False) | 56,5% | -2,4 pp | 6 |

`best_epoch=1` z wlaczona cecha pokrycia wskazuje ze model w pierwszej epoce uczy sie heurystyki max-coverage (ta sama co baseline) i przestaje poprawiac wynik. Bez coverage model jest zmuszony szukac sygnalu w sekwencji - staje sie wyrazniejszy po epoce 6, ale nadal nie bije baseline.

Roznica miedzy konfiguracjami z i bez pokrycia wynosi tylko 0,6 pp, co oznacza ze CNN sekwencyjny uczy sie czegoś z sekwencji DNA, ale sygnał jest słabszy niz prosta heurystyka pokrycia.

#### Sweep dlugosci kontekstu

Test context_len ∈ {100, 150, 200, 250} przy embed_dim=64, use_cov=False, seed=42 (fold 1 z CV):

| context_len | Junc acc | vs baseline (val fold) | Best epoch |
|---|---|---|---|
| 100 bp | 56,5% | -2,4 pp | 6 |
| **150 bp** | **58,4%** | **-0,5 pp** | **8** |
| 200 bp | 57,2% | -1,6 pp | 3 |
| 250 bp | 58,0% | -0,9 pp | 6 |

Srednia dlugosc kontekstu w danych wynosi 44,7 bp, wiec wszystkie testowane wartosci to nadmiarowe okno (padding). Wyniki nie sa monotoniczne: 150 bp daje najlepszy wynik, a 200 bp spada - roznice rzedu 1 pp mieszcza sie czesciowo w szumie statystycznym (788 junctions w val foldzie, 1% ~ 8 junctions).

Kluczowe spostrzezenie: **context_len=150 prawie osiaga baseline (-0,5 pp)**, co oznacza ze CNN sekwencyjny potrafi wyciagnac sygnał z sekwencji DNA przy odpowiedniej dlugosci okna. Architektura nie osiagnela twardego sufitu - potrzebuje odpowiedniego rozmiaru kontekstu.

Dalszy wzrost dokladnosci moze zapewnic lepsza architektura lub trening na wielu organizmach jednoczesnie (transfer learning). Wiarygodny test transferu wymaga jednak osobnego zbioru walidacyjnego z innego organizmu i 5-fold CV - nie zostal przeprowadzony w ramach tego etapu projektu.

#### Analiza bledow: dokladnosc vs rzeczywista dlugosc kontekstu

Dodatkowy test na foldzie 1 (ctx=150, use_cov=False): model uczony i ewaluowany, a nastepnie junctiony val foldzie pogrupowane po rzeczywistej dlugosci `context_seq`:

| Bin (bp) | n | Model | Baseline | delta |
|---|---|---|---|---|
| 20-30 | 276 | 62,7% | 63,0% | -0,4 pp |
| 30-40 | 54 | 55,6% | 72,2% | -16,7 pp |
| 40-50 | 52 | 57,7% | 63,5% | -5,8 pp |
| 50-60 | 29 | 58,6% | 55,2% | +3,4 pp |
| 60-80 | 28 | 53,6% | 53,6% | +0,0 pp |
| >80 | 349 | 55,9% | 53,6% | +2,3 pp |

Srednia dlugosc kontekstu (44,7 bp) jest mylaaca, bo rozklad jest **bimodalny**: 35% junctionow ma kontekst 20-30 bp, a 44% ma kontekst powyzej 80 bp. Typowy junction to albo ten tuz po innym skrzyzowaniu (krotki kontekst), albo glęboko w dlugim kontigu (dlugi kontekst). Przedzialy 30-60 bp lacznie maja jedynie ~18% danych.

Na dlugich kontekstach (>80 bp, n=349) model nieznacznie bije baseline (+2,3 pp) - CNN ma wiecej sekwencji do analizy i wyciaga z niej sygnał. Na krotkich kontekstach (20-30 bp, n=276) model prawie wyrownuje baseline (-0,4 pp) - obu metody ogranicza brak informacji w krotkim fragmencie sekwencji. Wynik w przedziale 30-40 bp (-16,7 pp) jest najgorszy, ale przy n=54 jest statystycznie niegodny (1 pp ~ 0,5 junctionow).



## Słownik pojęć

| Pojęcie | Znaczenie |
|---|---|
| **odczyt** (*read*) | Krótki fragment DNA odczytany przez sekwenator (tu: 150 bp). Sekwenator nie czyta całego genomu naraz - wytwarza miliony takich fragmentów. |
| **kontig** (*contig*) | Ciągły fragment genomu zrekonstruowany przez asembler przez składanie nakładających się odczytów. Im dłuższy i mniej liczny, tym lepsza asemblacja. |
| **k-mer** | Podciąg odczytu o długości *k*. Graf de Bruijn jest zbudowany z k-merów - każdy odczyt rozkłada się na wiele nakładających się k-merów. |
| **węzeł rozejścia** (*junction*) | Miejsce w grafie de Bruijn, gdzie contig mógłby iść w kilku kierunkach (out-degree ≥ 2). Asembler musi zdecydować którą gałąź wybrać, albo przerwać contig. |
| **ground truth** | Poprawna odpowiedź dla danego węzła rozejścia, wyznaczona automatycznie przez wyszukanie sekwencji kontekstu w genomie referencyjnym. Jeśli kontekst pojawia się w genomie tylko raz - wiadomo jaki nukleotyd następuje po nim, więc ground truth jest znany. |
| **ambiguous** | Węzeł rozejścia, dla którego **nie można wyznaczyć ground truth** - sekwencja kontekstu pojawia się w genomie wielokrotnie, po różnych nukleotydach. Takie skrzyżowania są odrzucane z treningu, bo nie wiadomo jaka jest prawidłowa odpowiedź. |
| **labeled** | Węzeł rozejścia z *poznanym* ground truth (przeciwieństwo *ambiguous*). Tylko te próbki trafiają do treningu. |
| **baseline** | Prosta heurystyka bez ML: wybieraj zawsze gałąź o wyższym pokryciu odczytów. Stanowi punkt odniesienia dla modelu. |
| **reverse complement (RC)** | Komplementarna nić DNA czytana od końca do początku. Ponieważ DNA jest dwuniciowe, każda sekwencja `ACGT...` ma swoje RC `...ACGT`, które koduje tę samą informację biologiczną z drugiej strony. |
| **N50 / NG50** | Statystyki jakości asemblacji. N50: połowa zsumowanej długości kontigów pochodzi z kontigów o długości ≥ N50. NG50: analogicznie, ale względem długości genomu referencyjnego. |