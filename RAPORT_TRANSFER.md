# Raport: transfer miedzy genomami (grupa A: E. coli, grupa B: Enterobacteriaceae)

Kontynuacja `RAPORT.md`. Tamten raport zakonczyl sie hipoteza: model trenowany i testowany na JEDNYM organizmie nie bije baseline (57,5% vs 57,7%), a diagnoza wskazala ze 56% labeled junctionow to krotkie powtorzenia, gdzie jedyna rozstrzygajaca informacja to "z ktorej kopii powtorzenia pochodzi ten kontekst" - fakt o KONKRETNYM genomie, niemozliwy do wyuczenia w obrebie jednego organizmu.

Hipoteza do przetestowania: jesli model zobaczy WIELE genomow z tej samej grupy taksonomicznej, moze nauczyc sie rozpoznawac rodziny elementow repetytywnych (IS, operon rRNA) jako cecha transferowalna, zamiast pamietac pozycje w jednym genomie. Test: trening na ~90 genomach z jednej grupy, ewaluacja zero-shot na ~10 innych, trzymanych poza treningiem.

Metodologia w sekcji 1 ponizej. Instrukcja odtworzenia grupy A: `experiments/group_a/README.md`.

---

## 1. Skrot metodologii

- **Grupa A (same-species):** 90 szczepow E. coli (trening) + 10 held-out, dobranych z puli 100 zdedupowanych genomow (RefSeq, Complete Genome, 4-6 Mbp). Dwa warianty testowego 10-zbioru: `farthest` (najbardziej odizolowane filogenetycznie wg macierzy odleglosci Mash) i `random`.
- **Grupa B (same-family):** analogicznie, ale pula 100 genomow to 10 rodzajow rodziny Enterobacteriaceae (Escherichia, Salmonella, Klebsiella, Citrobacter, Enterobacter, Shigella, Serratia, Proteus, Yersinia, Cronobacter), po ~10-20 genomow na rodzaj (stratyfikowany sampling, zeby pula nie byla zdominowana przez E. coli).
- **Dedup:** przed podzialem train/test, genomy o odleglosci Mash < 0,0005 (near-duplicate klony, np. z jednego ogniska epidemicznego) usuwane z puli - inaczej test "losowy" bylby sztucznie zawyzony przez niemal identyczne genomy w train i test.
- **Generowanie danych:** pipeline Rust bez zmian (k=21, coverage=100x, read-length=150), per-genom CSV z junctionami.
- **Model:** MV1 (siec syjamska CNN), `context_len=150` (najlepszy wynik ze sweepu w RAPORT.md), `--no_cov` (cechy pokrycia WYLACZONE - patrz uzasadnienie nizej), `--augment` (RC + swap A/B), `epochs=50`, `early_stop=10`.

### Dlaczego `--no_cov`?

RAPORT.md pokazal, ze wlaczone cechy pokrycia sprawiaja, ze model w epoce 1 uczy sie kopiowac heurystyke max-coverage i przestaje szukac sygnalu w sekwencji (`best_epoch=1` z coverage vs `best_epoch=6+` bez). Skoro ten eksperyment ma odpowiedziec na pytanie "czy model wyciaga z sekwencji cos transferowalnego miedzy genomami", trening bez cech pokrycia jest jedynym sposobem, zeby nie zamaskowac efektu, ktory badamy.

---

## 2. Wykryty i naprawiony blad: stronniczosc pozycyjna w budowie par treningowych

Podczas pierwszego pelnego treningu (przed napisaniem tego raportu) zaobserwowano niepokojacy wzorzec: `val_junction_accuracy` skakalo chaotycznie z epoki na epoke (53% -> 82% -> 90% -> 69%...), podczas gdy `train_pair_acc` rosl stabilnie i powoli. Taki obraz jest niespojny z normalnym uczeniem - gdyby model faktycznie przelamywal plateau, oczekiwalibysmy rownoleglej poprawy na treningu, nie tylko na walidacji.

### Na czym polegal blad

W `ml/data/dataset.py`, metoda `JunctionDataset._build_pairs` budowala pare `(branch_a, branch_b, label)` dla kazdego junctionu. Przy wlaczonej augmentacji (`augment=True`), kod tworzyl 4 warianty pary: oryginal, oryginal z zamieniona pozycja (A<->B), i obie wersje reverse-complement - co gwarantowalo, ze poprawna galaz ladowala rownie czesto na pozycji A co B (balans etykiet 50/50).

Problem: to losowanie pozycji istnialo TYLKO wewnatrz bloku `if self.augment:`. Gdy `augment=False`, kod bezwarunkowo budowal pare z poprawna galezia zawsze na pozycji A (`label=0`) - bez zadnego mechanizmu losowania.

Konsekwencja: **zbior walidacyjny w `train.py` jest budowany z `augment=False` NA SZTYWNO, niezaleznie od flagi `--augment` przekazanej do treningu** (`val_dataset = JunctionDataset(..., augment=False, ...)`). Skutek: **caly zbior walidacyjny mial 100% probek z `label=0`, zero probek z `label=1`.**

### Dlaczego to tlumaczylo obserwowany chaos

Siec syjamska ocenia obie galezie tymi samymi wagami (`branch_encoder` wywolywany dwukrotnie) - architektura z zalozenia ma byc symetryczna wzgledem pozycji A/B. Ale w trakcie treningu siec moze przypadkowo nabrac drobnego, nieistotnego odchylenia w strone "pozycja A zwykle wygrywa" (przez losowe niuanse inicjalizacji wag, kolejnosc aktualizacji itp.), kompletnie niezwiazanego z trescia sekwencji DNA.

Poniewaz caly zbior walidacyjny mial `label=0` (czyli "A zawsze poprawne"), taki przypadkowy dryf odchylenia pozycyjnego byl **calkowicie skorelowany** z etykieta walidacyjna - gdy siec akurat faworyzowala pozycje A, walidacja pokazywala kilkudziesieciopunktowy skok w gore; gdy dryf sie cofal, spadek byl rownie gwaltowny. Metryka mierzyla przypadkowe zachowanie pozycyjne sieci, nie jej rzeczywista zdolnosc rozrozniania sekwencji DNA.

To dotyczylo takze `training/evaluate.py` (ten sam kod `_build_pairs`), wiec **kazdy planowany test zero-shot na zbiorach testowych mialby dokladnie ta sama wade** - caly plan porownania model vs baseline bylby niewiarygodny.

### Poprawka

Przypisanie pozycji jest teraz losowane deterministycznie (hash z `pair_seed`, `junction_id` i indeksu galezi niepoprawnej -> ziarno do `random.Random`) **niezaleznie od flagi `augment`** - kazdy zbior (trening, walidacja, test) ma w oczekiwaniu zbalansowany rozklad etykiet 50/50, a wynik jest powtarzalny miedzy uruchomieniami (ten sam `pair_seed` daje ten sam przydzial pozycji). Dodano test regresyjny sprawdzajacy ten konkretny przypadek ("100% label=0" przy `augment=False`).

Po poprawce krzywe walidacji staly sie gladkie i monotoniczne, spojne z krzywa treningowa - potwierdzone w smoke-testach przed pelnym przebiegiem (patrz commit/sesja naprawy w historii repo).

**Wniosek metodologiczny:** wszystkie wyniki w tym raporcie pochodza z treningow wykonanych PO tej poprawce. Wczesniejsze przebiegi (przed poprawka) zostaly odrzucone i nie sa tu raportowane.

---

## 3. Rygor statystyczny: powtorzenia z roznymi seedami

Pojedynczy trening to pojedynczy stratyfikowany split 85/15 (`seed=42`) - nie 5-fold CV jak w oryginalnym eksperymencie cerevisiae z RAPORT.md. Zeby wyniki byly obronione statystycznie, kazdy z 4 glownych wariantow (grupa A/B x farthest/random) zostal powtorzony **3x z roznymi seedami** (42, 123, 7), a nastepnie kazdy z 3 checkpointow zostal osobno poddany ewaluacji zero-shot na odpowiadajacym mu zbiorze 10 organizmow testowych. Raportowane `mean +/- std` to statystyka po tych 3 niezaleznych powtorzeniach.

Dodatkowo policzono baseline (max-coverage, ta sama populacja junctionow co model - zweryfikowane identycznymi liczbami `n` w obu metodach) oraz model solo (trenowany i walidowany na pojedynczym, odlozonym organizmie osobno, 5x z roznymi seedami, 20% wewnetrznej walidacji) jako gorna referencja z planu.

---

## 4. Wyniki

### 4.1 Tabela glowna

| Wariant | Model zero-shot (mean +/- std, n=3 seedy) | Baseline (max-cov) | Solo (1 genom, mean +/- std, n=5 seedow) | Przewaga modelu nad baseline |
|---|---|---|---|---|
| Grupa A, farthest | **71,71% +/- 0,48pp** | 56,33% | 54,47% +/- 2,65pp | **+15,38 pp** |
| Grupa A, random   | **71,85% +/- 0,99pp** | 56,89% | 54,47% +/- 2,65pp | **+14,96 pp** |
| Grupa B, farthest | **54,90% +/- 0,73pp** | 55,44% | 54,52% +/- 4,00pp | -0,54 pp |
| Grupa B, random   | **53,58% +/- 0,02pp** | 54,98% | 54,52% +/- 4,00pp | -1,40 pp |

Kolumna baseline i "n_junctions" per organizm zweryfikowane jako identyczne miedzy metoda modelu i baseline (to samo filtrowanie: `min_context_len=10`, `min_branch_len=5`, tylko rzedy z jednoznacznym ground truth) - porownanie jest na dokladnie tej samej populacji junctionow.

### 4.2 Interpretacja

**Grupa A: silny, powtarzalny efekt transferu.** Przewaga nad baseline (~15 pp) jest ponad 15x wieksza niz jej odchylenie standardowe (0,48-0,99pp) w obu wariantach - to nie jest szum pojedynczego uruchomienia, tylko systematyczny, odtwarzalny efekt. Model trenowany na 90 szczepach E. coli generalizuje dobrze na calkowicie nowe, nigdy niewidziane szczepy tego samego gatunku.

**Farthest ~= random w grupie A.** Model radzi sobie rownie dobrze na filogenetycznie najdalszych szczepach testowych, jak na losowych. Brak spadku przy "trudniejszym" wariancie sugeruje, ze transfer dziala rownomiernie w calym gatunku, a nie tylko na szczepach blisko spokrewnionych z pula treningowa - i posrednio potwierdza, ze wynik na wariancie `random` nie jest zawyzony przez przypadkowy leakage bliskich klonow (dedup w doborze danych zadzialal).

**Grupa B: brak transferu, konsekwentnie w 3/3 seedach.** Model NIE bije baseline w zadnym z 3 powtorzen w zadnym z 2 wariantow (farthest, random) - roznica jest ujemna i mala (rzedu wielkosci wlasnego odchylenia std), co oznacza ze nie mozna nawet stwierdzic pewnej przewagi baseline'u, ale na pewno nie ma tam ukrytej przewagi modelu. Efekt transferu obserwowany w grupie A **nie rozciaga sie** na cala rodzine taksonomiczna Enterobacteriaceae.

**Gradient A >> B jest ostry, nie stopniowy.** Zamiast lagodnego oslabienia efektu ze wzrostem odleglosci taksonomicznej (czego oczekiwano w oryginalnej hipotezie: "A > B > C"), obserwujemy niemal binarny podzial: dziala w obrebie gatunku, calkowicie zanika na poziomie rodziny. Sugeruje to, ze mechanizm transferu (wspolne rodziny elementow IS + ich flanki) wymaga bardzo wysokiej identycznosci sekwencji (>98%, jak w obrebie E. coli), a nie samej obecnosci tych samych NAZW rodzin elementow w odleglejszych genomach - flanki otaczajace te elementy rozchodza sie ewolucyjnie zbyt szybko, zeby CNN zdazyl je rozpoznac jako wspolny wzorzec.

**Solo (1 genom) nie sluzy jako wiarygodna gorna referencja.** Solo ladowalo sie konsekwentnie w okolicach 54,5% w OBU grupach (statystycznie nierozroznialne: 54,47% vs 54,52%), ponizej baseline w obu przypadkach. Analiza krzywych treningowych pojedynczych przebiegow solo pokazala klasyczny obraz przeuczenia: `train_pair_acc` rosnie, `val_pair_acc` spada, najlepsza epoka to epoka 1 (przed jakimkolwiek realnym uczeniem). Przy 400-1400 labeled junctions na organizm CNN nie ma z czego nauczyc sie uogolnialnej reguly przed zapamietaniem zbioru treningowego. **Wniosek: solo mierzy "ile da sie nauczyc z ~500-900 przykladow", nie teoretyczny sufit zadania** - a to, ze model wielogenomowy (grupa A, 90 genomow) bije solo o ~17pp, dodatkowo potwierdza, ze cala wartosc dodana lezy w treningu na wielu genomach jednoczesnie, nie w samej architekturze.

### 4.3 Rozbicie per organizm (grupa A, farthest) - przyklad rozrzutu

| Organizm | Zero-shot accuracy (mean, n=3 seedy) |
|---|---|
| GCF_000618905.3 | 65,92% |
| GCF_017639185.1 | 68,97% |
| GCF_037094575.1 | 69,95% |
| GCF_053102695.1 | 70,12% |
| GCF_040500455.1 | 70,47% |
| GCF_002587005.1 | 70,93% |
| GCF_001886895.1 | 72,27% |
| GCF_022494785.1 | 73,24% |
| GCF_041316455.1 | 79,37% |
| GCF_041005995.1 | 80,60% |

Rozrzut ~15pp miedzy najslabszym a najlepszym szczepem. `GCF_041005995.1` i `GCF_041316455.1` konsekwentnie najlepsze w OBU wariantach (farthest i random) - sugeruje to systematyczna wlasciwosc tych konkretnych genomow (mozliwe hipotezy do sprawdzenia: bliskosc do puli treningowej mimo formalnego wykluczenia z "farthest", liczba/typ elementow IS, jakosc assembly), a nie przypadek.

---

## 5. Rozbicie wynikow po `repeat_len` i `cov_ratio_max` (grupa A, farthest, seed=42)

Analiza per-junction (nie tylko zagregowana celnosc) na 6345 junctionach z 10 organizmow testowych grupy A/farthest. Join predykcji modelu z metadanymi CSV zweryfikowany jako 100% kompletny (6345/6345).

### 5.1 Wedlug `repeat_len`

| repeat_len | n | Model (transfer, grupa A) | RAPORT.md (solo, cerevisiae) |
|---|---|---|---|
| 0     | 5157 | **73,24%** | 57,2% |
| 1-2   |  377 | 70,03% | 59,9% |
| 3-5   |  289 | 69,20% | 62,0% |
| 6-10  |  215 | 67,91% | 59,3% |
| 11-20 |  168 | 64,29% | 62,8% |
| >=21  |  139 | **51,80%** | 52,9% |

**Kluczowy wynik: monotoniczny spadek trafnosci z rosnacym `repeat_len`**, diametralnie inny obraz niz w oryginalnym eksperymencie solo (gdzie wartosci skakaly bez wyraznego trendu, 57-63%, bez systematycznej zaleznosci).

Interpretacja: przewaga modelu transferowego (+15pp nad baseline w zagregowanym wyniku) **nie jest rozlana rownomiernie** - koncentruje sie w krotkich powtorzeniach (`repeat_len=0`: 73,24%, daleko powyzej baseline), i **zanika niemal do poziomu losowego** przy dlugich powtorzeniach (`repeat_len>=21`: 51,80%, praktycznie identyczne z RAPORT.md). To jest zgodne z mechanizmem hipotezy: model uczy sie rozpoznawac krotkie, transferowalne wzorce sekwencyjne wspolne miedzy szczepami E. coli, ale przy dluzszych powtorzeniach dominuje informacja pozycyjna specyficzna per-genom (liczba kopii, ich ukladu), ktora - zgodnie z wnioskiem z RAPORT.md - nie jest mozliwa do wyuczenia niezaleznie od tego, ile genomow model widzial w treningu.

**Wniosek: nawet w obrebie tego samego gatunku, transfer ma swoj zasieg dzialania ograniczony do krotkich powtorzen.** To zawezenie hipotezy w stosunku do oryginalnego sformulowania z planu.

### 5.2 Wedlug `cov_ratio_max`

| cov_ratio_max | n | Model (transfer, grupa A) | RAPORT.md (solo, cerevisiae) |
|---|---|---|---|
| <0,55       | 3702 | **70,64%** | 50,7% (poziom losowy) |
| 0,55-0,60   |  754 | 75,73% | 51,0% |
| 0,60-0,65   |  280 | 70,36% | 63,0% |
| 0,65-0,70   |  554 | 68,41% | 67,2% |
| 0,70-0,80   |  579 | 70,98% | 69,1% |
| >=0,80      |  476 | **82,77%** | 83,4% |

**Najistotniejsza roznica jest w najtrudniejszym binie** (`<0,55`, gdzie pokrycie samo w sobie nie daje zadnej informacji - heurystyka to tam rzut moneta): RAPORT.md pokazal tam **50,7%** (dowod ze solo model rowniez zgaduje), podczas gdy model transferowy osiaga **70,64%** - a wiec **model znajduje realny sygnal sekwencyjny wlasnie tam, gdzie brak transferu calkowicie zawodzil**. To bezposredni dowod, ze mechanizm transferu faktycznie dziala tam, gdzie mial dzialac wedlug hipotezy (przypadki gdzie sama statystyka pokrycia jest bezuzyteczna).

W wyzszych binach (>=0,80, latwe przypadki gdzie i heurystyka radzi sobie dobrze) obie metody sa zbiezne (82,77% vs 83,4%) - spojne z oczekiwaniem, ze tam przewaga transferu nie ma juz duzo do dodania.

### 5.3 Rozszerzenie: wszystkie 4 warianty (repeat_len)

| repeat_len | group_a_farthest | group_a_random | group_b_farthest | group_b_random |
|---|---|---|---|---|
| 0     | 73,24% (n=5157) | 73,84% (n=4913) | 54,13% (n=3839) | 52,88% (n=5560) |
| 1-2   | 70,03% (n=377)  | 70,88% (n=340)  | 63,38% (n=355)  | 53,54% (n=536)  |
| 3-5   | 69,20% (n=289)  | 66,80% (n=256)  | 56,09% (n=230)  | 59,40% (n=266)  |
| 6-10  | 67,91% (n=215)  | 73,68% (n=209)  | 62,04% (n=137)  | 56,02% (n=166)  |
| 11-20 | 64,29% (n=168)  | 64,02% (n=164)  | 55,26% (n=76)   | 60,36% (n=111)  |
| >=21  | 51,80% (n=139)  | 56,72% (n=134)  | 58,33% (n=72)   | 67,44% (n=86)   |

**Grupa A: wzorzec spadkowy potwierdzony w obu wariantach** (farthest i random daja niemal identyczny ksztalt krzywej) - wniosek z 5.1 jest powtarzalny, nie zalezny od konkretnego doboru zbioru testowego.

**Grupa B: obraz bardziej zlozony niz "brak transferu wszedzie".** W dominujacym binie `repeat_len=0` (n=3839-5560, wiekszosc danych) wynik jest blisko losowego (52,88-54,13%), zgodnie z zagregowanym wynikiem calego wariantu. Ale w binach posrednich pojawiaja sie wartosci wyraznie powyzej 54% (np. `farthest, 1-2`: 63,38%; `random, >=21`: 67,44%). **Te biny maja bardzo mala liczebnosc (n=72-536)** - przy takich probkach, roznica kilkunastu punktow procentowych moze wynikac z szumu (pojedyncze dodatkowe trafienia), nie z prawdziwego czesciowego transferu. Odroznienie tych dwoch mozliwosci wymagaloby wiekszej puli testowej w grupie B (wiecej organizmow lub wiecej labeled junctionow na organizm) - **oznaczone jako otwarte pytanie**, nie rozstrzygniete w ramach tej sesji.

### 5.4 Rozszerzenie: wszystkie 4 warianty (cov_ratio_max)

| cov_ratio_max | group_a_farthest | group_a_random | group_b_farthest | group_b_random |
|---|---|---|---|---|
| [0,00-0,55) | 70,64% (n=3702) | 71,84% (n=3470) | **50,25%** (n=2943) | **49,93%** (n=4424) |
| [0,55-0,60) | 75,73% (n=754)  | 71,55% (n=791)  | 50,79% (n=699)  | 50,34% (n=1023) |
| [0,60-0,65) | 70,36% (n=280)  | 67,68% (n=297)  | 60,48% (n=167)  | 59,41% (n=271)  |
| [0,65-0,70) | 68,41% (n=554)  | 71,27% (n=456)  | 66,67% (n=327)  | 61,58% (n=367)  |
| [0,70-0,80) | 70,98% (n=579)  | 74,34% (n=491)  | 69,77% (n=258)  | 72,51% (n=342)  |
| [0,80-1,01) | 82,77% (n=476)  | 83,17% (n=511)  | 85,08% (n=315)  | 81,88% (n=298)  |

**Bin `<0,55` (najtrudniejsze przypadki, gdzie sama heurystyka to rzut moneta) daje najczystszy obraz.** Ma najwieksza liczebnosc (n=2943-4424 w grupie B) i jest binem, gdzie transfer sekwencyjny mialby najwiecej do zaoferowania (skoro pokrycie nic nie mowi). Wynik: grupa A **70,64-71,84%** (silny sygnal), grupa B **49,93-50,25%** (dokladnie poziom losowy, w obu wariantach). To jest najbardziej jednoznaczny dowod w calym raporcie na brak transferu sekwencyjnego w grupie B - w przeciwienstwie do rozbicia po `repeat_len`, tutaj liczebnosc jest duza we wszystkich wariantach, wiec wynik nie jest zagrozony szumem malej proby.

W binie `[0,80-1,01)` (latwe przypadki) grupa B wypada nawet wyzej niz A (85,08% vs 82,77%) - ale to nie jest dowod transferu, tylko efekt tego, ze przy tak dominujacym pokryciu kazda rozsadna metoda (model czy baseline) radzi sobie dobrze niezaleznie od sygnalu sekwencyjnego.

---

## 6. Co tlumaczy rozrzut miedzy organizmami w grupie A?

Zbadano 7 potencjalnych zmiennych wyjasniajacych rozrzut ~15pp miedzy najslabszym (`GCF_000618905.3`, 65,92%) a najlepszym (`GCF_041005995.1`, 80,60%) organizmem w `group_a_farthest`: odsetek junctionow z `repeat_len=0`, srednia dlugosc powtorzenia, srednie `cov_ratio_max`, odsetek trudnych przypadkow (`cov_ratio_max<0,55`), liczba junctionow testowych, srednia dlugosc kontekstu, oraz odleglosc Mash do najblizszego sasiada w zbiorze treningowym.

| Zmienna | Pearson r z accuracy |
|---|---|
| mean_cov_ratio_max | **0,542** |
| n_junctions | -0,482 |
| pct_repeat0 | 0,408 |
| mean_repeat_len | -0,396 |
| min_dist_train (Mash) | -0,314 |
| pct_cov_low | -0,194 |
| mean_ctx_len | 0,015 |

**Brak jednej dominujacej przyczyny.** Najsilniejsza korelacja (0,542, `mean_cov_ratio_max`) tlumaczy ~29% wariancji - zgodna kierunkowo z wynikami z sekcji 5 (latwiejsze junctiony = wyzsza pewnosc pokrycia), ale nie jest to odkrycie nowego mechanizmu, tylko powtorzenie tego samego efektu usrednione per organizm.

**Wazny wynik negatywny: `min_dist_train` NIE tlumaczy dobrze rozrzutu** (r=-0,314, slaba). `GCF_041005995.1` (najlepszy wynik) ma odleglosc Mash do treningu w SRODKU stawki (0,0119), nie najmniejsza - najblizszy treningowi jest `GCF_001886895.1` (0,0104), ktory ma wyraznie nizszy wynik (72,27%). **To obala prosta hipoteze, ze najlepsze wyniki wynikaja z ukrytego niedostatecznego odizolowania od zbioru treningowego** - gdyby tak bylo, oczekiwalibysmy znacznie silniejszej korelacji i zgodnosci rankingow.

**`n_junctions` (r=-0,482) sugeruje czesciowy udzial szumu statystycznego.** Najlepsze organizmy maja stosunkowo mala liczbe junctionow testowych (540 i 652), co oznacza wieksza wariancje samej metryki accuracy przy mniejszej probie - czesc obserwowanego rozrzutu miedzy organizmami moze byc efektem tej wariancji, nie prawdziwa wlasciwoscia biologiczna konkretnych genomow.

**Wniosek: rozrzut miedzy organizmami to prawdopodobnie kombinacja kilku slabych efektow** (subtelna latwosc/trudnosc danych, szum malej proby, mozliwy niewielki wplyw bliskosci filogenetycznej), a nie jedna wyrazna przyczyna. Nie znaleziono dowodu na przeciek danych ani na inny artefakt metodologiczny tlumaczacy skrajne wyniki.

---

## 7. Otwarte pytania / kolejne kroki

- [ ] Grupa B, biny posrednie `repeat_len` (1-20): odroznic prawdziwy czesciowy sygnal od szumu malej proby (n=72-536) - wymagaloby wiekszej puli testowej.
- [ ] Grupa C (kontrola, rozszerzenie istniejacego zbioru 5 organizmow z roznych rodzin) - z oryginalnego planu, jeszcze nie wykonana w ramach tej sesji.
- [ ] Ablacja wielkosci zbioru treningowego (10/30/60/90 organizmow) - krzywa uczenia w funkcji liczby genomow, kluczowy wykres z oryginalnego planu, jeszcze nie wykonany.