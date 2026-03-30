mod simple_assembler;

fn main() {
    let oryginal = "1_KOT_2_PIES_3_KOT_4_PIES_57_KOT_8_PIES_9_KOT_10_PIES_11_KOT_12_PIES_13_KOT_14_PIES_15_KOT_16_PIES_17_KOT_18_PIES_19_KOT_20_PIES";
    let dlugosc_odczytu = 10;
    let krok = 2;

    println!("CEL: Złożenie zdania\n");

    let reads = simple_assembler::generate_reads(oryginal, dlugosc_odczytu, krok);
    println!("Wygenerowano {} odczytów (długość: {}, krok: {}).\n", reads.len(), dlugosc_odczytu, krok);

    let k_values = 3..=dlugosc_odczytu;

    for k in k_values {
        println!("--------------------------------------------------");
        println!("Testuję asemblację dla k = {}", k);

        let graf = simple_assembler::build_de_bruijn_graph(&reads, k);
        if graf.is_empty() {
            println!("BŁĄD: Graf jest pusty. Przechodzę do następnego k.");
            continue;
        }

        let (in_deg, out_deg) = simple_assembler::get_degrees(&graf);
        let start_node = match simple_assembler::find_start_node(&graf, &in_deg, &out_deg) {
            Some(node) => node,
            None => {
                println!("BŁĄD: Nie znaleziono wierzchołka startowego. Przechodzę do następnego k.");
                continue;
            }
        };
        
        let sciezka = simple_assembler::find_eulerian_path(&graf, start_node);
        let wynik = simple_assembler::assemble(&sciezka);

        if wynik == oryginal {
            println!("WYNIK: SUKCES");
        } else {
            println!("WYNIK: PORAŻKA");
        }
        
        println!("ZŁOŻONO: {}", wynik);
    }
    
    println!("--------------------------------------------------");
    println!("Zakończono testowanie wszystkich wartości k.");
}