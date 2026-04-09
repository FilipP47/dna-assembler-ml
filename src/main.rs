mod simple_assembler;
mod reads;
mod fasta;

use fasta::{read_fasta, save_to_fasta};
use std::time::Instant;



fn main() {
    println!("--- SYMULATOR ASEMBLACJI DNA ---");

    let file_name = "ecoli.fasta";
    let read_length = 150;
    let coverage = 50;
    let k = 51;

    let oryginal = match read_fasta(file_name) {
        Ok(seq) => {
            println!("Pomyślnie załadowano plik referencyjny '{}' (długość: {} bp).", file_name, seq.len());
            seq
        }
        Err(_) => {
            println!("UWAGA: Nie znaleziono pliku '{}'.", file_name);
            println!("Używam krótkiej sekwencji testowej.");
            "ATGCGTACGTTAGCATGCGTACGTTAGC".to_string()
        }
    };
    


    println!("\nGenerowanie odczytów...");
    let start_time = Instant::now();
    let reads = reads::generate_circular_reads(&oryginal, coverage, read_length);
    println!(
        "Wygenerowano {} odczytów w {:?} (pokrycie: {}x, długość: {}).",
        reads.len(),
        start_time.elapsed(),
        coverage,
        read_length
    );

    println!("\nBudowanie grafu De Bruijna (k = {})...", k);
    let start_time = Instant::now();
    let graf = simple_assembler::build_de_bruijn_graph(&reads, k);
    if graf.is_empty() {
        println!("BŁĄD: Graf jest pusty. Zbyt mało odczytów lub błąd parametrów.");
        return;
    }
    println!("Graf zbudowany w {:?}. Węzły: {}", start_time.elapsed(), graf.len());

    let (in_deg, out_deg) = simple_assembler::get_degrees(&graf);
    let start_node = match simple_assembler::find_start_node(&graf, &in_deg, &out_deg) {
        Some(node) => node,
        None => {
            println!("BŁĄD: Nie znaleziono wierzchołka startowego.");
            return;
        }
    };
    
    println!("\nSzukanie ścieżki Eulera...");
    let start_time = Instant::now();
    let sciezka = simple_assembler::find_eulerian_path(&graf, start_node.clone());
    let wynik = simple_assembler::assemble(&sciezka);
    println!("Ścieżka znaleziona w {:?}", start_time.elapsed());

    let output_filename = format!(
        "zlozony_genom_odczyt{}_cov{}_k{}.fasta", 
        read_length, 
        coverage, 
        k
    );
    println!("\nZapisywanie wyniku do pliku: {}", output_filename);
    
    match save_to_fasta(&output_filename, "moj_zlozony_kontig", &wynik) {
        Ok(_) => {
            println!("SUKCES! Złożona sekwencja (długość: {} bp, realna długość: {} bp) została zapisana.", oryginal.len(), wynik.len());
        }
        Err(e) => {
            println!("BŁĄD przy zapisywaniu pliku: {}", e);
        }
    }
}