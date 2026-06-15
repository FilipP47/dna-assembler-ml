/// A single assembled contig produced by graph traversal.
#[derive(Debug, Clone)]
pub struct Contig {
    pub id:           usize,
    pub sequence:     String,
    pub length:       usize,
    pub avg_coverage: f64,
    pub edge_count:   usize,
}

impl Contig {
    /// Construct a contig, computing `length` from the sequence automatically.
    pub fn new(id: usize, sequence: String, avg_coverage: f64, edge_count: usize) -> Self {
        let length = sequence.len();
        Self { id, sequence, length, avg_coverage, edge_count }
    }

    /// Format this contig as a FASTA record with sequence lines wrapped at `line_width`.
    pub fn to_fasta_record(&self, line_width: usize) -> String {
        let mut s = format!(
            ">contig_{} len={} avg_cov={:.2} edges={}\n",
            self.id, self.length, self.avg_coverage, self.edge_count
        );
        for chunk in self.sequence.as_bytes().chunks(line_width) {
            s.push_str(std::str::from_utf8(chunk).unwrap());
            s.push('\n');
        }
        s
    }
}

/// Summary statistics over a collection of contigs.
#[derive(Debug)]
pub struct ContigStats {
    pub count:         usize,
    pub total_length:  usize,
    pub min_length:    usize,
    pub max_length:    usize,
    pub n50:           usize,
    pub mean_coverage: f64,
}

impl ContigStats {
    /// Compute summary statistics from a slice of contigs.
    pub fn compute(contigs: &[Contig]) -> Self {
        if contigs.is_empty() {
            return Self { count: 0, total_length: 0, min_length: 0,
                          max_length: 0, n50: 0, mean_coverage: 0.0 };
        }
        let count        = contigs.len();
        let total_length = contigs.iter().map(|c| c.length).sum();
        let min_length   = contigs.iter().map(|c| c.length).min().unwrap();
        let max_length   = contigs.iter().map(|c| c.length).max().unwrap();
        let mean_coverage = contigs.iter().map(|c| c.avg_coverage).sum::<f64>() / count as f64;

        let mut lengths: Vec<usize> = contigs.iter().map(|c| c.length).collect();
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        let half = total_length / 2;
        let mut cum = 0;
        let mut n50 = 0;
        for l in &lengths {
            cum += l;
            if cum >= half { n50 = *l; break; }
        }
        Self { count, total_length, min_length, max_length, n50, mean_coverage }
    }

    /// Print a human-readable summary to stdout.
    pub fn print(&self) {
        println!("Contig stats:");
        println!("  Count        = {}", self.count);
        println!("  Total length = {} bp", self.total_length);
        println!("  Min / Max    = {} / {} bp", self.min_length, self.max_length);
        println!("  N50          = {} bp", self.n50);
        println!("  Mean cov     = {:.2}x", self.mean_coverage);
    }
}
