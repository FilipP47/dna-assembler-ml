mod simple_assembler;
mod reads;
mod fasta;

use fasta::{read_fasta, save_contigs_to_fasta};
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

    println!("\nSzukanie kontigów unitig...");
    let start_time = Instant::now();
    let sciezki_kontigow = simple_assembler::extract_contig_paths(&graf);
    let kontigi = simple_assembler::assemble_contigs(&sciezki_kontigow);
    println!("Znaleziono {} kontigów w {:?}", kontigi.len(), start_time.elapsed());

    for (index, kontig) in kontigi.iter().enumerate() {
        println!("  - kontig_{}: {} bp", index + 1, kontig.len());
    }

    let output_filename = format!(
        "zlozony_genom_odczyt{}_cov{}_k{}_kontigi.fasta", 
        read_length,
        coverage,
        k
    );
    println!("\nZapisywanie kontigów do pliku: {}", output_filename);
    
    match save_contigs_to_fasta(&output_filename, &kontigi) {
        Ok(_) => {
            let total_length: usize = kontigi.iter().map(|contig| contig.len()).sum();
            println!("SUKCES! Zapisano {} kontigów o łącznej długości {} bp.", kontigi.len(), total_length);
        }
        Err(e) => {
            println!("BŁĄD przy zapisywaniu pliku: {}", e);
        }
    }
}