//! A directed edge in the de Bruijn graph, representing a k-mer.
//!
//! The `KmerKey` itself is the edge identifier (used as the HashMap key).
//! This struct stores the two endpoint `NodeId`s and the coverage counter.

use crate::core::kmer::NodeId;

/// A directed edge in the de Bruijn graph, corresponding to one distinct k-mer.
#[derive(Debug, Clone)]
pub struct DeBruijnEdge {
    /// Source node: the prefix (k-1)-mer of this k-mer.
    pub from: NodeId,

    /// Destination node: the suffix (k-1)-mer of this k-mer.
    pub to: NodeId,

    /// How many times this exact k-mer appeared across all reads.
    /// Incremented on every duplicate; never decremented.
    pub coverage: u32,
}

impl DeBruijnEdge {
    /// Create a new edge with initial coverage of 1.
    pub fn new(from: NodeId, to: NodeId) -> Self {
        Self { from, to, coverage: 1 }
    }
}
