//! A node in the de Bruijn graph, representing a (k-1)-mer.
//!
//! Each node stores up to 4 outgoing and 4 incoming edges indexed by the
//! nucleotide base (A=0, C=1, G=2, T=3).  Because the DNA alphabet has
//! exactly 4 symbols, the maximum degree is 4 in either direction.
//!
//! Using `[Option<KmerKey>; 4]` instead of a `HashSet` gives:
//!   • zero heap allocation for the adjacency lists
//!   • O(1) lookup and insertion by base index
//!   • the entire node fits in a single cache line for small k

use crate::core::kmer::KmerKey;

/// A node in the de Bruijn graph, representing one distinct (k-1)-mer.
#[derive(Debug, Clone)]
pub struct DeBruijnNode {
    /// Outgoing edge for each possible next base.
    /// `out_edges[i]` is the kmer whose last base has index `i`.
    /// `None` means no edge exists for that base.
    pub out_edges: [Option<KmerKey>; 4],

    /// Incoming edge for each possible preceding base.
    /// `in_edges[i]` is the kmer whose first base has index `i`.
    pub in_edges: [Option<KmerKey>; 4],

    /// Cached count of non-None slots in `out_edges`.
    /// Avoids iterating the array every time we test for junctions.
    pub out_degree: u8,

    /// Cached count of non-None slots in `in_edges`.
    pub in_degree: u8,
}

impl DeBruijnNode {
    /// Construct an isolated node with no edges.
    pub fn new() -> Self {
        Self {
            out_edges: [None, None, None, None],
            in_edges: [None, None, None, None],
            out_degree: 0,
            in_degree: 0,
        }
    }

    /// Register an outgoing edge.
    ///
    /// `base_idx` is the 0-3 index of the last nucleotide of `edge`.
    /// Idempotent: calling twice with the same base index is a no-op
    /// (coverage is tracked on the edge, not the node).
    pub fn add_out_edge(&mut self, base_idx: usize, edge: KmerKey) {
        debug_assert!(base_idx < 4, "base_idx must be 0-3");
        if self.out_edges[base_idx].is_none() {
            self.out_edges[base_idx] = Some(edge);
            self.out_degree += 1;
        }
    }

    /// Register an incoming edge.
    ///
    /// `base_idx` is the 0-3 index of the first nucleotide of `edge`.
    pub fn add_in_edge(&mut self, base_idx: usize, edge: KmerKey) {
        debug_assert!(base_idx < 4, "base_idx must be 0-3");
        if self.in_edges[base_idx].is_none() {
            self.in_edges[base_idx] = Some(edge);
            self.in_degree += 1;
        }
    }

    /// Returns `true` if this node has more than one outgoing edge (contig split point).
    #[inline]
    pub fn is_junction(&self) -> bool {
        self.out_degree > 1
    }

    /// Returns `true` if this node has no outgoing edges (end of a contig).
    #[inline]
    pub fn is_dead_end(&self) -> bool {
        self.out_degree == 0
    }

    /// Returns `true` if this node has no incoming edges (start of a contig).
    #[inline]
    pub fn is_source(&self) -> bool {
        self.in_degree == 0
    }

    /// Returns the single outgoing edge key, or `None` if degree ≠ 1.
    ///
    /// Avoids an iterator allocation for the common unambiguous-continuation case.
    pub fn single_out_edge(&self) -> Option<&KmerKey> {
        if self.out_degree != 1 {
            return None;
        }
        self.out_edges.iter().find_map(|e| e.as_ref())
    }
}

impl Default for DeBruijnNode {
    fn default() -> Self {
        Self::new()
    }
}
