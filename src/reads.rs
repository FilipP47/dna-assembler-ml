use rand::Rng;

pub fn generate_circular_reads(text: &str, coverage: usize, read_length: usize) -> Vec<String> {
    let genome_size = text.len();
    if genome_size == 0 {
        return vec![];
    }

    let reads_number = (coverage * genome_size).div_ceil(read_length);

    let mut rng = rand::thread_rng();
    let mut reads = Vec::with_capacity(reads_number);
    let text_bytes = text.as_bytes();

    for _ in 0..reads_number {
        let start = rng.gen_range(0..genome_size);
        let mut read = String::with_capacity(read_length);

        for i in 0..read_length {
            let idx = (start + i) % genome_size;
            read.push(text_bytes[idx] as char);
        }

        reads.push(read);
    }

    reads
}
