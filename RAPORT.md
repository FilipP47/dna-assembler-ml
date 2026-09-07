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

Aktualny zbior danych to *S. cerevisiae* S288C wygenerowany potokiem Rust z pokryciem 100x (wyzszym niz wczesniejsze eksperymenty):

| Organizm | Rozmiar genomu | Wiersze w CSV | **Labeled** | Ambiguous |
|---|---|---|---|---|
| *S. cerevisiae* S288C | 12,1 Mbp | 10 840 | **3 965 (36,6%)** | 5 470 (50,5%) |

Wiersze nieoznaczone (10 840 - 3 965 - 5 470 = 1 405) to skrzyzowania odfiltrowane przez minimalne wymagania dlugosci sekwencji (context >= 10 bp, branch >= 5 bp). Wyzszy odsetek labeled (36% vs ~17% w starszym przebiegu) wynika z wyzszego pokrycia odczytow: wiecej odczytow na kazda pozycje genomu oznacza ze nawet krotsze konteksty wystarczaja do jednoznacznego wskazania miejsca w genomie referencyjnym.

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
| k=21, *S. cerevisiae* | 3 965 | 57,7% | **57,5% ± 0,9%** | -0,2 pp |
| k=31, *S. cerevisiae* | 590 | 67,6% | **64,9% ± 2,3%** | -2,7 pp |

k=31 ma wyzszy baseline (67,6%) bo przy dluzszym k-merze skrzyzowania sa rzadsze i bardziej specyficzne - pokrycie jest tam silniejszym sygnalem. k=21 jest lepszy do treningu: 6,7x wiecej probek, nizsza wariancja wynikow miedzy foldami.

Wczesniejszy wynik 71,3% (pojedynczy random split na danych k=31 z E. coli + S. cerevisiae lacznie) nie jest powtarzalny: ten sam zbior z 5-fold CV daje 64,9% przy baseline 67,6%. Roznica wynikala prawdopodobnie z losowego podzialu na val z nizszym baseline, a nie z faktycznej przewagi modelu.

### Szczegolowe wyniki: S. cerevisiae k=21, 5-fold CV

Obecny zbior danych: 3 965 labeled skrzyzowań z jednego organizmu (*S. cerevisiae*, k=21). Ewaluacja opiera sie na stratyfikowanym 5-fold CV (seed=42).

| Metryka | Wartość |
|---|---|
| Baseline (max-coverage, globalny) | **57,7%** (2 287/3 965) |
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

Test transferu miedzy genomami zostal przeprowadzony w nastepnym etapie i opisany w `RAPORT_TRANSFER.md`: trening na ~90 szczepach E. coli bije baseline o ~15 pp na nowych szczepach tego samego gatunku (grupa A), ale efekt nie rozciaga sie na cala rodzine Enterobacteriaceae (grupa B).

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

#### Kiedy wygrywa heurystyka, a kiedy model? Analiza wg pewnosci pokrycia

Model i heurystyka moga miec podobna srednia, a rozne rozklady bledow. Sprawdzono to predykcja out-of-fold: kazdy z 3 965 labeled junctionow dostal predykcje z modelu, ktory go nie widzial w treningu (5-fold CV, ctx=150, use_cov=False - czysto sekwencyjny, seed=42). Junctiony pogrupowano po `cov_ratio_max` (udzial dominujacej galezi w pokryciu; 0,50 to remis, 1,0 to jedna galaz bierze wszystko).

| cov_ratio_max | n | Model | Heurystyka | delta |
|---|---|---|---|---|
| <0,55 | 2 213 | 50,7% | 49,3% | +1,4 pp |
| 0,55-0,60 | 510 | 51,0% | 51,2% | -0,2 pp |
| 0,60-0,65 | 181 | 63,0% | 64,1% | -1,1 pp |
| 0,65-0,70 | 296 | 67,2% | 71,3% | -4,1 pp |
| 0,70-0,80 | 350 | 69,1% | 73,1% | -4,0 pp |
| >=0,80 | 415 | 83,4% | 84,8% | -1,4 pp |
| **Ogolem** | **3 965** | **57,6%** | **57,7%** | **-0,1 pp** |

Hipoteza brzmiala: w niejednoznacznych przypadkach (pokrycie 51 do 49) heurystyka jest rzutem moneta, ale moze model wyciagnie sygnał z sekwencji. Dane jej nie potwierdzaja. W binie `<0,55` model daje 50,7% - ten sam poziom losowy co heurystyka (49,3%). Pierwsze 100-150 bp kazdej galezi nie rozroznia tych skrzyzowan. We wszystkich pozostalych binach heurystyka jest nieznacznie lepsza. Poniewaz `cov_ratio_max` to udzial dominujacej galezi (max/suma), wartosc ponizej 0,55 oznacza faktyczny remis: 93% skrzyzowan w tym binie ma out-degree 2, a empiryczna dokladnosc heurystyki na poziomie 50% potwierdza ze sa to genuinnie niejednoznaczne przypadki.

Rozklad zgodnosci obu metod na wszystkich 3 965 junctionach:

| Przypadek | n | Udzial |
|---|---|---|
| Oba dobrze | 1 522 | 38,4% |
| Tylko model | 760 | 19,2% |
| Tylko heurystyka | 765 | 19,3% |
| Oba zle | 918 | 23,2% |

Metody nie zgadzaja sie na 38,5% skrzyzowan, ale rozklad jest niemal symetryczny: model ratuje 760 przypadkow, ktore heurystyka myli, a psuje 765, ktore heurystyka trafia. Gdyby dalo sie z gory wskazac ktora metoda ma racje, mozna by wybierac lepsza per skrzyzowanie i osiagnac gorna granice (oracle): 38,4% + 19,2% + 19,3% = **76,8%**.

#### Czy niezgodnosci sa przewidywalne? Test routera

Kluczowe pytanie: czy te 19% "tylko model" da sie odroznic z gory od przypadkow gdzie model sie myli? Jesli tak, hybryda dochodzi do 77%. Test: na przypadkach spornych (1 525 skrzyzowan, gdzie racje ma dokladnie jedna metoda) wytrenowano klasyfikator (regresja logistyczna, 5-fold CV) przewidujacy ktora metoda wygra, na cechach: `cov_ratio_max`, out-degree, dlugosc kontekstu, liczba odczytow oraz **pewnosc samego modelu** (margines `|score_A - score_B|`).

| Metryka | Wartość |
|---|---|
| AUC klasyfikatora na przypadkach spornych | **0,497** |
| Wyuczony router (5-fold) | 57,0% |
| Heurystyka (zawsze) | 57,7% |
| Oracle (idealny router) | 76,8% |

AUC 0,497 to praktycznie 0,5 - **niezgodnosci sa nieprzewidywalne**. Przypadki gdzie wygrywa model sa statystycznie nieodroznialne od tych, gdzie model przegrywa. Nawet wlasna pewnosc modelu (margines) nie koreluje z trafnoscia - model "nie wie, kiedy wie". Wyuczony router osiaga 57,0%, czyli **ponizej heurystyki**: skoro sygnalu nie ma, routowanie tylko dokłada szum. Dodanie `repeat_len` (dlugosci powtorzenia) jako kolejnej cechy routera tez nie pomaga (57,1%, nadal ponizej heurystyki).

Wniosek: oracle 76,8% pokazuje ze teoretycznie informacja na poprawe istnieje, ale jest niedostepna z obecnych cech i tej architektury. Sufit ~58% nie jest artefaktem zlego laczenia metod - to realna granica tego zbioru danych.

Prawdziwy sufit to sam zbior danych: **56% labeled junctionow ma `cov_ratio_max < 0,55`**, a tam ani pokrycie, ani sekwencja nie niosą sygnalu - obie metody sa na poziomie losowym. Podobne wyniki srednie nie wynikaja z tego, ze metody sa mocne w roznych miejscach, tylko z tego, ze na wiekszosci trudnych skrzyzowan obie zawodza jednakowo, w sposob ktorego nie da sie przewidziec.

Trafnosc modelu jest przewidywalna, ale tylko gruboziarnisto i tylko przez `cov_ratio_max`. Klasyfikator przewidujacy `model_ok` z wszystkich cech osiaga AUC 0,614, a z samej `cov_ratio_max` niemal tyle samo (0,615) - pozostale cechy, wlacznie z pewnoscia modelu, nie wnosza nic. Model dziala jak "kategoria latwosci": na latwych skrzyzowaniach (`cov_ratio_max >= 0,80`) trafia 83%, na trudnych (`< 0,55`) tylko 51%. Ale **wewnatrz** trudnego binu ktore konkretnie skrzyzowania trafi jest juz czysto losowe (AUC 0,497). Model nie ma wlasnej strefy kompetencji niezaleznej od heurystyki - jest dobry dokladnie tam, gdzie latwa jest tez heurystyka.

#### Natura sufitu: trudne skrzyzowania to krotkie powtorzenia

Ground truth wyznaczany jest przez wyszukanie `context_seq` (do 100 bp zlozonych przed skrzyzowaniem) w genomie referencyjnym: jesli wszystkie wystapienia kontekstu prowadza do tego samego nastepnego nukleotydu, etykieta jest `Known`, jesli do roznych - `Ambiguous` (odrzucone). Poniewaz odczyty sa bezbledne, galaz istnieje tylko gdy jest realna kontynuacja w genomie - out-degree 2 oznacza, ze **oba warianty naprawde wystepuja** (powtorzenie: 20-mer wezla pojawia sie w dwoch miejscach, raz po nim idzie A, raz B).

To wyjasnia dlaczego trudne skrzyzowania (`cov_ratio_max < 0,55`) sa trudne. Obie galezie sa realne i podobnie pokryte, wiec heurystyka max-coverage jest rzutem moneta. Etykieta `Known` istnieje tylko dlatego, ze 100 bp kontekstu wychodzi poza obszar powtorzenia i staje sie unikalne w referencji - sa to wiec **krotkie powtorzenia** (krotsze niz ~100 bp). Dluzsze powtorzenia daja `Ambiguous` i trafiaja do odrzuconych 56% zbioru.

Teze te potwierdza cecha `repeat_len` dodana do generatora: idac wstecz od skrzyzowania po grafie liczy sie kroki do najblizszego wezla, w ktorym schodza sie dwie drogi (in-degree >= 2), czyli do miejsca gdzie unikalne flanki obu kopii zlaczaja sie we wspolne powtorzenie. Rozklad `repeat_len` jest jednoznacznie rozdzielony:

| repeat_len | labeled | ambiguous |
|---|---|---|
| <10 | 3 849 | 3 492 |
| 10-20 | 46 | 689 |
| 20-40 | 43 | 569 |
| 40-60 | 19 | 301 |
| 60-100 | 8 | 196 |
| >=100 | 0 | 223 |

Labeled to niemal zawsze krotkie powtorzenia (mediana 0, p95=5, maks 75), i **zaden** labeled nie ma `repeat_len >= 100`. Ambiguous maja dlugi ogon (mediana 5, p95=85, az 223 skrzyzowania z powtorzeniem >= 100). To bezposrednio potwierdza mechanizm: skrzyzowanie jest rozwiazywalne tylko gdy powtorzenie jest krotsze niz okno kontekstu.

Wazny wniosek: dla labeled `repeat_len` wynosi zwykle 0, wiec baza rozrozniajaca lezy **tuz przy krawedzi** skrzyzowania i jest lokalnie dostepna dla modelu. Bariera nie jest wiec pozycja sygnalu (i nie usunie jej attention ani wieksza waga dla dalszych pozycji), tylko jego **nieuogolnialnosc**. Informacja rozstrzygajaca to "z ktorej z dwoch kopii genomu pochodzi ten dokladny kontekst" - fakt o ukladzie konkretnego genomu, a nie prawidlowosc sekwencji. Ground truth zna odpowiedz, bo robi dokladny lookup calej referencji (pamiec genomu); model probuje nauczyc sie uogolnialnej reguly, ale mapowanie "te poprzedzajace bazy -> galaz A" jest arbitralne i nie przenosi sie na powtorzenia walidacyjne, inne niz treningowe. Dlatego na krotkich powtorzeniach model zgaduje - potwierdza to AUC 0,497 wewnatrz trudnego binu.

Trafnosc w podziale na `repeat_len` (labeled, out-of-fold) pokazuje ze model nie ma przewagi w zadnym przedziale:

| repeat_len | n | Model | Heurystyka | delta |
|---|---|---|---|---|
| 0 | 3 365 | 57,2% | 56,9% | +0,3 pp |
| 1-2 | 279 | 59,9% | 59,1% | +0,7 pp |
| 3-5 | 129 | 62,0% | 62,0% | +0,0 pp |
| 6-10 | 81 | 59,3% | 65,4% | -6,2 pp |
| 11-20 | 43 | 62,8% | 72,1% | -9,3 pp |
| >=21 | 68 | 52,9% | 63,2% | -10,3 pp |

Przy krotkich powtorzeniach (0-5) model remisuje z heurystyka, a przy dluzszych (>=6) wyraznie przegrywa. (Przy zanizonym pokryciu 30x pojawia sie pozorna przewaga modelu przy `repeat_len` 1-10, ale znika przy prawidlowym pokryciu 100x - byla artefaktem szumu pokrycia, ktory oslabial heurystyke.)

Prawdziwe rozstrzygniecie tych przypadkow lezy nie w architekturze okna kontekstu, lecz w informacji na poziomie odczytow: odczyt 150 bp przekraczajacy cale powtorzenie (flank + R + kontynuacja) fizycznie swiadczy, ktora kontynuacja jest wlasciwa. To standardowy read threading, ktorego graf de Bruijn nie zachowuje, bo tnie odczyty na k-mery i gubi ich spojnosc. Kierunek poprawy to wiec przywrocenie informacji odczytowej (pary odczytow, read threading, dluzsze odczyty), a nie samo dostrajanie CNN na oknie przy skrzyzowaniu.



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