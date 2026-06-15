use std::{fs, io::{BufWriter, Write}, path::Path};
use anyhow::{Context, Result};

use crate::core::contig::{Contig, ContigStats};
use crate::core::graph::DeBruijnGraph;
use crate::pipeline::analysis::JunctionAnalyzer;
use super::traits::Assembler;

/// A greedy assembler that cuts contigs at every junction (out-degree > 1).
pub struct SimpleAssembler;

impl Assembler for SimpleAssembler {
    fn name(&self) -> &'static str { "SimpleAssembler (cut-at-junction)" }

    fn assemble(
        &self,
        graph: &DeBruijnGraph,
        analyzer: &mut Option<Box<dyn JunctionAnalyzer>>,
    ) -> Result<Vec<Contig>> {
        graph.print_stats();
        Ok(graph.traverse(analyzer))
    }
}

impl SimpleAssembler {
    /// Assemble contigs, write them to `output_dir/contigs.fasta`, and optionally
    /// export junction data to `output_dir/junctions.csv`.
    pub fn run(
        &self,
        graph: &DeBruijnGraph,
        output_dir: &Path,
        analyzer: &mut Option<Box<dyn JunctionAnalyzer>>,
    ) -> Result<Vec<Contig>> {
        let contigs = self.assemble(graph, analyzer)?;

        fs::create_dir_all(output_dir)
            .with_context(|| format!("Cannot create: {}", output_dir.display()))?;

        let fasta_path = output_dir.join("contigs.fasta");
        let file = fs::File::create(&fasta_path)?;
        let mut w = BufWriter::new(file);
        for c in &contigs {
            write!(w, "{}", c.to_fasta_record(60))?;
        }
        println!("Contigs written → {}", fasta_path.display());

        if let Some(ref a) = analyzer {
            if a.junction_count() > 0 {
                let csv = output_dir.join("junctions.csv");
                a.export(&csv)?;
                println!("Junctions ({}) → {}", a.junction_count(), csv.display());
            }
        }

        ContigStats::compute(&contigs).print();
        Ok(contigs)
    }
}
