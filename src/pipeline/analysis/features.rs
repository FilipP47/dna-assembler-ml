//! Data structures describing a single junction in the de Bruijn graph.
//!
//! `JunctionContext`  — everything the assembler knows at the moment it reaches
//!                      a branching node.  Passed to `JunctionAnalyzer::analyze`.
//!
//! `JunctionFeatures` — the computed feature vector ready for ML export.
//!                      Fields are `Option<…>` so the collector can populate
//!                      only those enabled by `AnalysisOptions`.

use crate::core::kmer::NodeId;

/// Snapshot of the assembler state at a junction node.
///
/// The analyzer receives this and extracts whatever features it needs.
#[derive(Debug, Clone)]
pub struct JunctionContext {
    /// Encoded ID of the junction node ((k-1)-mer).
    pub node_id: NodeId,

    /// Number of edges (kmers) traversed so far in the current contig.
    pub depth_in_contig: usize,

    /// Number of outgoing edges from this node (always ≥ 2 at a junction).
    pub out_degree: u8,

    /// Number of incoming edges into this node.
    pub in_degree: u8,

    /// Per-branch coverage: `branch_coverage[i]` is the coverage of
    /// the edge at `out_edges[i]`, or `None` if that slot is empty.
    pub branch_coverage: [Option<u32>; 4],

    pub context_seq: String,
    pub contig_id: usize,
}

impl JunctionContext {
    /// Construct a junction context from all assembler-side fields.
    pub fn new(
        node_id: NodeId,
        depth_in_contig: usize,
        out_degree: u8,
        in_degree: u8,
        branch_coverage: [Option<u32>; 4],
        context_seq: String,
        contig_id: usize,
    ) -> Self {
        Self {
            node_id,
            depth_in_contig,
            out_degree,
            in_degree,
            branch_coverage,
            context_seq,
            contig_id,
        }
    }
}

/// Feature vector for one junction — the ML model's input at inference time.
///
/// Fields are `Option<>` so absent features are explicit, not zero-filled.
#[derive(Debug, Clone, Default)]
pub struct JunctionFeatures {
    /// Coverage of each outgoing branch (indices 0-3 → A/C/G/T).
    pub branch_coverage: Option<[Option<u32>; 4]>,

    /// Ratio of max-branch to total coverage at this junction.
    pub coverage_ratio_max: Option<f64>,

    /// The (k-1)-mer sequence at the junction node (decoded to String).
    pub node_sequence: Option<String>,

    /// Total reads passing through this node (sum of in-edge coverage).
    pub total_read_count: Option<u32>,

    /// Number of nodes reachable within `local_graph_depth` hops.
    pub local_node_count: Option<usize>,

    /// Number of additional junctions within `local_graph_depth` hops.
    pub local_junction_count: Option<usize>,

    /// How many bases into the current contig this junction was reached.
    pub depth_in_contig: usize,

    /// Out-degree of the junction node.
    pub out_degree: u8,

    /// In-degree of the junction node.
    pub in_degree: u8,
}
