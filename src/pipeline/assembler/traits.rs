use anyhow::Result;
use crate::core::contig::Contig;
use crate::core::graph::DeBruijnGraph;
use crate::pipeline::analysis::JunctionAnalyzer;

/// Strategy interface for a DNA assembler.
pub trait Assembler {
    /// Traverse `graph` and return the assembled contigs.
    ///
    /// `analyzer` receives a callback at every junction if `Some`.
    fn assemble(
        &self,
        graph: &DeBruijnGraph,
        analyzer: &mut Option<Box<dyn JunctionAnalyzer>>,
    ) -> Result<Vec<Contig>>;

    /// Human-readable name of this assembler strategy.
    fn name(&self) -> &'static str;
}
