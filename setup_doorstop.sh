#!/bin/bash
set -e

echo "Tworzenie struktury katalogów..."
mkdir -p reqs

echo "Inicjalizacja dokumentów Doorstop..."
doorstop create CAT ./reqs/cat
doorstop create REQ ./reqs/req --parent CAT
doorstop create TST ./reqs/tst --parent REQ

# ── Helper ────────────────────────────────────────────────────────────────────
# Użycie: write_item <prefix> <id> <level> <link> <ref> <header> <text>
write_item() {
    local prefix="$1"
    local id="$2"
    local level="$3"
    local link="$4"
    local ref="$5"
    local header="$6"
    local text="$7"
    local dir="./reqs/${prefix,,}"
    local file="${dir}/${prefix}-${id}.yml"

    cat > "$file" << YAML
active: true
derived: false
header: |
  ${header}
level: ${level}
normative: true
ref: '${ref}'
YAML

    if [ -n "$link" ]; then
        echo "links:" >> "$file"
        echo "- ${link}:" >> "$file"
    fi

    echo "text: |" >> "$file"
    echo "  ${text}" >> "$file"
}

# ── CAT - Kategorie ───────────────────────────────────────────────────────────
echo "Generowanie kategorii (CAT)..."

write_item "CAT" "001" "1.0" "" "src/infra/config/" \
  "Configuration and Input" \
  "Zarządza parametrami CLI oraz parsowaniem plików wejściowych (odczyty, genom referencyjny)."

write_item "CAT" "002" "2.0" "" "src/core/graph/" \
  "Graph Construction and Assembly" \
  "Logika budowy grafu de Bruijna, traversal ścieżką Eulera oraz generowanie contigów."

write_item "CAT" "003" "3.0" "" "src/pipeline/analysis/" \
  "ML Data Preparation" \
  "Detekcja skrzyżowań i ekstrakcja cech sekwencji do uczenia maszynowego."

write_item "CAT" "004" "4.0" "" "src/pipeline/simulator/" \
  "Simulation" \
  "Niedeterministyczny podział genomu referencyjnego na syntetyczne odczyty sekwencyjne."

write_item "CAT" "005" "5.0" "" "src/pipeline/assembler/" \
  "Output and Reporting" \
  "Zapis wyników asemblacji do plików FASTA i CSV oraz statystyki contigów."

# ── REQ + TST - CAT-001: Configuration and Input ─────────────────────────────
echo "Generowanie REQ/TST dla CAT-001..."

write_item "REQ" "001" "1.1" "CAT-001" "src/main.rs" \
  "CLI k parameter" \
  "Program SHALL przyjmować argument CLI --k jako dodatnią liczbę całkowitą określającą długość k-meru."

write_item "TST" "001" "1.1" "REQ-001" "src/infra/config/assembler_config.rs" \
  "Test CLI k parameter parsing" \
  "Wywołanie AssemblerConfig::builder().input(\"dummy.fa\").k(31).build() SHALL zwrócić Ok z k=31."

write_item "REQ" "002" "1.2" "CAT-001" "src/infra/config/assembler_config.rs" \
  "k must be greater than 1" \
  "Program SHALL odrzucać wartość k <= 1 i kończyć działanie z komunikatem błędu wskazującym minimalną wymaganą wartość."

write_item "TST" "002" "1.2" "REQ-002" "src/infra/config/assembler_config.rs" \
  "Test k <= 1 rejection" \
  "Wywołanie AssemblerConfig::builder().input(\"dummy.fa\").k(1).build() SHALL zwrócić Err zawierający opis błędu o minimalnym k."

write_item "REQ" "003" "1.3" "CAT-001" "src/infra/config/assembler_config.rs" \
  "Even k warning" \
  "Program SHALL emitować ostrzeżenie na stderr gdy k jest parzyste, lecz kontynuować działanie. Nieparzyste k jest konwencją minimalizującą palindromy w grafie de Bruijna."

write_item "TST" "003" "1.3" "REQ-003" "src/infra/config/assembler_config.rs" \
  "Test even k warning" \
  "Wywołanie AssemblerConfig::builder().input(\"dummy.fa\").k(4).build() SHALL zakończyć się sukcesem i zapisać ostrzeżenie do stderr."

write_item "REQ" "004" "1.4" "CAT-001" "src/main.rs" \
  "Input file path via CLI" \
  "Program SHALL przyjmować ścieżkę do pliku wejściowego przez argument CLI --input. Nieistniejąca ścieżka SHALL skutkować błędem z czytelnym komunikatem."

write_item "TST" "004" "1.4" "REQ-004" "src/infra/io/factory.rs" \
  "Test missing input file error" \
  "Wywołanie reader_for_path na nieistniejącej ścieżce SHALL zwrócić Err zawierający nazwę pliku."

write_item "REQ" "005" "1.5" "CAT-001" "src/infra/io/fasta.rs" \
  "FASTA format parsing" \
  "Program SHALL parsować pliki FASTA z rozszerzeniem .fa .fasta .fna. Linie komentarzy (;) i puste linie SHALL być ignorowane. Wieloliniowe sekwencje SHALL być konkatenowane w jeden rekord."

write_item "TST" "005" "1.5" "REQ-005" "src/infra/io/fasta.rs" \
  "Test FASTA single record parsing" \
  "Parser wywołany na pliku z jednym rekordem SHALL zwrócić dokładnie jeden DnaRead z poprawnym header i sequence."

write_item "TST" "006" "1.6" "REQ-005" "src/infra/io/fasta.rs" \
  "Test FASTA multiline sequence" \
  "Parser wywołany na rekordzie z wieloliniową sekwencją SHALL zwrócić sequence będącą konkatenacją wszystkich linii sekwencji."

write_item "REQ" "006" "1.7" "CAT-001" "src/infra/config/simulator_config.rs" \
  "Simulator parameters via CLI" \
  "Symulator SHALL przyjmować przez CLI: --reference (ścieżka FASTA), --read-length (długość odczytu > 0), --coverage (pokrycie > 0), --seed (ziarno PRNG)."

write_item "TST" "007" "1.7" "REQ-006" "src/infra/config/simulator_config.rs" \
  "Test simulator config validation" \
  "SimulatorConfig::builder().read_length(0).build() SHALL zwrócić Err. build() z poprawnymi parametrami SHALL zwrócić Ok."

# ── REQ + TST - CAT-002: Graph Construction and Assembly ─────────────────────
echo "Generowanie REQ/TST dla CAT-002..."

write_item "REQ" "007" "2.1" "CAT-002" "src/core/graph/debruijn.rs" \
  "k-mer generation from reads" \
  "Program SHALL generować wszystkie k-mery z każdego odczytu przesuwając okno o rozmiarze k z krokiem 1. Odczyty krótsze niż k SHALL być pomijane bez błędu."

write_item "TST" "008" "2.1" "REQ-007" "src/core/graph/debruijn.rs" \
  "Test k-mer generation" \
  "Dla sekwencji ATCGAT i k=3 graf SHALL zawierać dokładnie 4 krawędzie: ATC TCG CGA GAT."

write_item "REQ" "008" "2.2" "CAT-002" "src/core/kmer/encoding.rs" \
  "De Bruijn graph construction" \
  "Graf de Bruijna SHALL reprezentować węzły jako (k-1)-mery zakodowane binarnie (A=00 C=01 G=10 T=11) i krawędzie jako k-mery. Węzeł SHALL być tworzony przy pierwszym napotkaniu danego (k-1)-meru."

write_item "TST" "009" "2.2" "REQ-008" "src/core/kmer/encoding.rs" \
  "Test DnaKey encoding roundtrip" \
  "DnaKey::encode(b\"ACGT\").decode() SHALL zwrócić \"ACGT\". size_of::<DnaKey>() SHALL wynosić 24 bajty."

write_item "REQ" "009" "2.3" "CAT-002" "src/core/graph/edge.rs" \
  "Coverage tracking per edge" \
  "Graf SHALL przechowywać dla każdego węzła 4 sloty wyjściowe indeksowane zasadą (A/C/G/T). Wielokrotne wystąpienie tego samego k-meru SHALL inkrementować licznik pokrycia krawędzi nie tworzyć duplikatu."

write_item "TST" "010" "2.3" "REQ-009" "src/core/graph/debruijn.rs" \
  "Test coverage increment on duplicate kmer" \
  "Wstawienie tego samego k-meru dwukrotnie SHALL skutkować coverage=2 na krawędzi i nadal jedną krawędzią w grafie."

write_item "REQ" "010" "2.4" "CAT-002" "src/core/graph/node.rs" \
  "Junction detection" \
  "Węzeł skrzyżowania SHALL być zdefiniowany jako węzeł którego out_degree > 1. Program SHALL identyfikować wszystkie takie węzły podczas traversalu."

write_item "TST" "011" "2.4" "REQ-010" "src/core/graph/node.rs" \
  "Test junction detection" \
  "DeBruijnNode z dwoma wypełnionymi slotami out_edges SHALL mieć out_degree=2 i is_junction()==true."

write_item "REQ" "011" "2.5" "CAT-002" "src/core/graph/debruijn.rs" \
  "Contig extraction via greedy traversal" \
  "Traversal SHALL budować contig zaczynając od węzłów z in_degree != 1. Contig SHALL być kończony gdy napotkany węzeł ma out_degree != 1 (skrzyżowanie lub ślepy zaułek)."

write_item "TST" "012" "2.5" "REQ-011" "src/core/graph/debruijn.rs" \
  "Test contig split at junction" \
  "Graf z jednym węzłem skrzyżowania (out_degree=2) SHALL produkować co najmniej 3 contigi: jeden prowadzący do skrzyżowania i po jednym dla każdej gałęzi."

write_item "REQ" "012" "2.6" "CAT-002" "src/core/graph/debruijn.rs" \
  "Cycle detection in traversal" \
  "Traversal SHALL wykrywać cykle: każda krawędź SHALL być odwiedzona co najwyżej raz. Napotkanie odwiedzonej krawędzi SHALL kończyć bieżący contig."

write_item "TST" "013" "2.6" "REQ-012" "src/core/graph/debruijn.rs" \
  "Test cycle handling" \
  "Graf zawierający cykl SHALL zostać w pełni przejrzany bez zapętlenia. Żadna krawędź nie SHALL pojawić się w dwóch contigach jednocześnie."

write_item "REQ" "013" "2.7" "CAT-002" "src/core/graph/debruijn.rs" \
  "Dead-end handling" \
  "Contig SHALL być kończony gdy bieżący węzeł ma out_degree=0 (ślepy zaułek). Zebrana sekwencja SHALL zostać zapisana jako contig nawet jeśli jest krótka."

write_item "TST" "014" "2.7" "REQ-013" "src/core/graph/debruijn.rs" \
  "Test dead-end termination" \
  "Graf z jedną liniową ścieżką (bez rozgałęzień) SHALL produkować dokładnie jeden contig kończący się przy węźle z out_degree=0."

# ── REQ + TST - CAT-003: ML Data Preparation ─────────────────────────────────
echo "Generowanie REQ/TST dla CAT-003..."

write_item "REQ" "015" "3.1" "CAT-003" "src/pipeline/analysis/collector.rs" \
  "Junction feature extraction" \
  "W momencie wykrycia skrzyżowania program SHALL zbierać: pokrycie każdej gałęzi wyjściowej (A/C/G/T), stopień wejściowy i wyjściowy węzła oraz głębokość w contigu."

write_item "TST" "015" "3.1" "REQ-015" "src/pipeline/analysis/collector.rs" \
  "Test junction feature collection" \
  "MlDataCollector::analyze() wywołane na JunctionContext z out_degree=2 SHALL zapisać rekord z poprawnymi wartościami out_degree i branch_coverage."

write_item "REQ" "016" "3.2" "CAT-003" "src/pipeline/analysis/collector.rs" \
  "Coverage ratio computation" \
  "Dla każdego skrzyżowania program SHALL obliczać coverage_ratio_max = max(branch_coverage) / sum(branch_coverage). Wartość SHALL być zapisywana w pliku CSV."

write_item "TST" "016" "3.2" "REQ-016" "src/pipeline/analysis/collector.rs" \
  "Test coverage ratio calculation" \
  "Dla branch_coverage=[Some(10), None, None, Some(5)] coverage_ratio_max SHALL wynosić 10.0/15.0 = 0.6667 (±0.001)."

write_item "REQ" "021" "3.3" "CAT-003" "src/pipeline/training_data/ground_truth.rs" \
  "Ground truth detection from reference" \
  "Generator danych SHALL wyszukiwać context_seq w genomie referencyjnym. Jeśli kontekst pojawia się jednoznacznie SHALL zapisać następujący nukleotyd jako ground_truth. Jeśli wielokrotnie z różnymi kontynuacjami SHALL oznaczyć jako ambiguous."

write_item "TST" "023" "3.3" "REQ-021" "src/pipeline/training_data/ground_truth.rs" \
  "Test unique ground truth detection" \
  "find_ground_truth z context_seq pojawiającym się jednokrotnie w referencji SHALL zwrócić GroundTruth::Known z poprawnym nukleotydem."

write_item "TST" "024" "3.4" "REQ-021" "src/pipeline/training_data/ground_truth.rs" \
  "Test ambiguous ground truth detection" \
  "find_ground_truth z context_seq pojawiającym się wielokrotnie z różnymi kontynuacjami SHALL zwrócić GroundTruth::Ambiguous."

# ── REQ + TST - CAT-004: Simulation ──────────────────────────────────────────
echo "Generowanie REQ/TST dla CAT-004..."

write_item "REQ" "018" "4.1" "CAT-004" "src/pipeline/simulator/random.rs" \
  "Random read simulator" \
  "Symulator SHALL generować odczyty jako dokładne podciągi genomu referencyjnego (zero błędów sekwencjonowania w Etapie 1). Pozycje startowe SHALL być losowane z rozkładu jednostajnego."

write_item "TST" "017" "4.1" "REQ-018" "src/pipeline/simulator/random.rs" \
  "Test reads are reference substrings" \
  "Każdy odczyt zwrócony przez simulate() SHALL być podciągiem genomu referencyjnego przekazanego jako wejście."

write_item "REQ" "019" "4.2" "CAT-004" "src/pipeline/simulator/random.rs" \
  "Reproducible simulation via seed" \
  "Dwa wywołania simulate() z identycznym seed SHALL zwrócić wektory odczytów o identycznej kolejności i zawartości. Różne seed SHALL produkować różne zestawy odczytów."

write_item "TST" "018" "4.2" "REQ-019" "src/pipeline/simulator/random.rs" \
  "Test simulation reproducibility" \
  "Dwa wywołania RandomReadSimulator z seed=42 SHALL zwrócić identyczne sekwencje. Wywołanie z seed=43 SHALL zwrócić różniący się zestaw."

write_item "REQ" "020" "4.3" "CAT-004" "src/pipeline/simulator/random.rs" \
  "Coverage-based read count calculation" \
  "Liczba odczytów SHALL być obliczana jako ceil(genome_length * coverage / read_length). Wynikowe pokrycie SHALL być zbliżone do zadanego parametru coverage."

write_item "TST" "019" "4.3" "REQ-020" "src/pipeline/simulator/random.rs" \
  "Test num_reads calculation" \
  "Dla genome_length=1000 coverage=10 read_length=100 num_reads SHALL wynosić dokładnie 100."

# ── REQ + TST - CAT-005: Output and Reporting ────────────────────────────────
echo "Generowanie REQ/TST dla CAT-005..."

write_item "REQ" "014" "5.1" "CAT-005" "src/pipeline/assembler/simple.rs" \
  "Contig FASTA output" \
  "Program SHALL zapisywać contigi do pliku contigs.fasta w katalogu wyjściowym. Nagłówek każdego contiga SHALL zawierać id długość avg_cov i liczbę krawędzi. Sekwencja SHALL być zawijana co 60 znaków."

write_item "TST" "020" "5.1" "REQ-014" "src/pipeline/assembler/simple.rs" \
  "Test FASTA contig output format" \
  "Contig::to_fasta_record() SHALL zaczynać się od '>' zawierać len= avg_cov= i sekwencję zawiniętą co 60 znaków."

write_item "REQ" "017" "5.2" "CAT-005" "src/pipeline/analysis/collector.rs" \
  "Junction CSV export" \
  "Zebrane cechy skrzyżowań SHALL być zapisywane do pliku junctions.csv z nagłówkiem CSV. Nieobecne cechy SHALL być reprezentowane jako NA. Plik SHALL być tworzony tylko gdy flaga --analyze jest aktywna."

write_item "TST" "021" "5.2" "REQ-017" "src/pipeline/analysis/collector.rs" \
  "Test CSV export format" \
  "MlDataCollector::export() SHALL tworzyć plik CSV z nagłówkiem zawierającym kolumny depth out_degree in_degree cov_A cov_C cov_G cov_T cov_ratio reads."

write_item "REQ" "022" "5.3" "CAT-005" "src/core/contig.rs" \
  "Assembly statistics reporting" \
  "Program SHALL obliczać i wypisywać po asemblacji: liczbę contigów całkowitą długość min/max długość N50 oraz średnie pokrycie."

write_item "TST" "022" "5.3" "REQ-022" "src/core/contig.rs" \
  "Test N50 computation" \
  "ContigStats::compute() dla zestawu contigów o długościach [10 20 30 40] SHALL obliczyć poprawne N50 total_length min_length i max_length."

echo "----------------------------------------------------"
echo "Sukces! Zbudowano drzewo Doorstop."
echo ""
echo "Weryfikacja:"
echo "  doorstop check"
echo ""
echo "Raport HTML:"
echo "  doorstop publish all ./docs/requirements/"