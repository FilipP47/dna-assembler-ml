//! Fine-grained feature flags for the junction analysis module.
//!
//! Each flag controls whether a specific feature category is collected
//! when the assembler encounters a branching node.
//!
//! Flags are independent: you can collect coverage without local graph, etc.

/// Feature-flag set controlling what the junction analyzer records.
#[derive(Debug, Clone)]
pub struct AnalysisOptions {
    /// Per-branch edge coverage at the junction.
    pub coverage: bool,

    /// (k-1)-nucleotide sequence context before and after the junction.
    pub sequence_context: bool,

    /// Number of reads that pass through the junction node.
    pub read_count: bool,

    /// Local subgraph around the junction (up to `local_graph_depth` hops).
    pub local_graph: bool,

    /// How many hops to explore when `local_graph` is true.
    pub local_graph_depth: usize,
}

impl AnalysisOptions {
    /// Enable every feature with default depth.
    pub fn all() -> Self {
        Self {
            coverage: true,
            sequence_context: true,
            read_count: true,
            local_graph: true,
            local_graph_depth: 3,
        }
    }

    /// Disable everything — useful as a starting point for selective enabling.
    pub fn none() -> Self {
        Self {
            coverage: false,
            sequence_context: false,
            read_count: false,
            local_graph: false,
            local_graph_depth: 0,
        }
    }
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self::all()
    }
}
