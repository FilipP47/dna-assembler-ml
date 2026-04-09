use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::io::Write;


pub fn read_fasta<P: AsRef<Path>>(filename: P) -> io::Result<String> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut sequence = String::new();

    for line in reader.lines() {
        let line = line?;
        if !line.starts_with('>') {
            sequence.push_str(line.trim());
        }
    }
    Ok(sequence)
}


pub fn save_to_fasta(filename: &str, header: &str, sequence: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    writeln!(file, ">{}", header)?;
    writeln!(file, "{}", sequence)?;
    Ok(())
}
