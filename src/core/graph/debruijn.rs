//! De Bruijn graph construction from reads and greedy Eulerian traversal.
//!
//! ## Building
//! For every read, slide a window of width k and insert each k-mer:
//!   • New k-mer → create edge + update source/destination nodes.
//!   • Seen k-mer → increment coverage on the existing edge (no duplicates).
//!
//! ## Traversal
//! Greedy contig extraction (two passes):
//!   Pass 1 — Start from every node where `in_degree ≠ 1` (sources,
//!             convergence points, and the split side of a junction).
//!             For each unvisited outgoing edge, walk forward appending
//!             the last base of each k-mer until:
//!               - dead end   (out_degree == 0) → end of contig
//!               - junction   (out_degree  > 1) → cut contig here
//!               - cycle      (edge already visited)
//!   Pass 2 — Pick up any edges still unvisited (pure cycles).

use std::collections::{HashMap, HashSet};
use anyhow::Result;

use crate::{
    infra::io::DnaRead,
    core::kmer::{KmerKey, NodeId, INDEX_TO_BASE},
    pipeline::analysis::{
        features::{JunctionContext},
        traits::JunctionAnalyzer,
    },
};
use super::{edge::DeBruijnEdge, node::DeBruijnNode};
use crate::core::contig::Contig;

/// The de Bruijn graph: a directed multigraph where nodes are (k-1)-mers
/// and edges are k-mers, weighted by sequencing coverage.
pub struct DeBruijnGraph {
    /// All nodes indexed by their (k-1)-mer NodeId.
    pub nodes: HashMap<NodeId, DeBruijnNode>,

    /// All edges indexed by their k-mer KmerKey.
    pub edges: HashMap<KmerKey, DeBruijnEdge>,

    /// The k value used during construction.
    pub k: usize,
}

impl DeBruijnGraph {
    /// Build the de Bruijn graph from a slice of DNA reads.
    ///
    /// Reads shorter than `k` are silently skipped.
    /// 'N' bases in reads will cause a panic in `KmerKey::encode`;
    /// filter them out beforehand if your reads may contain them.
    pub fn build(reads: &[DnaRead], k: usize) -> Result<Self> {
        anyhow::ensure!(k >= 2, "k must be at least 2");

        let mut graph = DeBruijnGraph {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            k,
        };

        for read in reads {
            let seq = read.sequence.as_bytes();
            if seq.len() < k {
                continue;
            }

            for i in 0..=(seq.len() - k) {
                graph.insert_kmer(&seq[i..i + k]);
            }
        }

        Ok(graph)
    }

    /// Insert a single k-mer into the graph.
    ///
    /// If the edge already exists, increment its coverage.
    /// Otherwise, create the edge and update both endpoint nodes.
    fn insert_kmer(&mut self, kmer_seq: &[u8]) {
        let edge_key = KmerKey::encode(kmer_seq);

        if let Some(edge) = self.edges.get_mut(&edge_key) {
            edge.coverage += 1;
            return;
        }

        let prefix = edge_key.prefix_node();
        let suffix = edge_key.suffix_node();
        let last_idx  = edge_key.last_base_idx();
        let first_idx = edge_key.first_base_idx();

        {
            let src = self.nodes.entry(prefix.clone()).or_default();
            src.add_out_edge(last_idx, edge_key.clone());
        }

        {
            let dst = self.nodes.entry(suffix.clone()).or_default();
            dst.add_in_edge(first_idx, edge_key.clone());
        }

        self.edges.insert(edge_key, DeBruijnEdge::new(prefix, suffix));
    }

    /// Total number of nodes (distinct (k-1)-mers).
    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// Total number of edges (distinct k-mers).
    pub fn edge_count(&self) -> usize { self.edges.len() }

    /// Number of nodes with out-degree > 1 (branching points).
    pub fn junction_count(&self) -> usize {
        self.nodes.values().filter(|n| n.is_junction()).count()
    }

    /// Print a brief graph summary to stdout.
    pub fn print_stats(&self) {
        println!("Graph stats:");
        println!("  k            = {}", self.k);
        println!("  Nodes        = {}", self.node_count());
        println!("  Edges        = {}", self.edge_count());
        println!("  Junctions    = {}", self.junction_count());
    }

    /// Extract all contigs via greedy Eulerian traversal.
    ///
    /// `analyzer` is called at every junction when `Some`; pass `None` to skip
    /// analysis entirely (no overhead).
    pub fn traverse(
        &self,
        analyzer: &mut Option<Box<dyn JunctionAnalyzer>>,
    ) -> Vec<Contig> {
        let mut visited: HashSet<KmerKey> = HashSet::new();
        let mut contigs: Vec<Contig> = Vec::new();
        let mut contig_id = 0usize;

        // Pass 1: start from all "contig start" nodes.
        // A node starts one or more new contigs when in_degree ≠ 1.
        // This covers:
        //   • true sources         (in_degree == 0)
        //   • convergence points   (in_degree  > 1)
        //   • the source of a split after a junction is itself reached again
        let start_ids: Vec<NodeId> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.in_degree != 1)
            .map(|(id, _)| id.clone())
            .collect();

        for node_id in &start_ids {
            let node = match self.nodes.get(node_id) {
                Some(n) => n,
                None => continue,
            };
            for slot in &node.out_edges {
                if let Some(edge_key) = slot {
                    if !visited.contains(edge_key) {
                        if let Some(c) = self.extract_contig(
                            node_id,
                            edge_key.clone(),
                            &mut visited,
                            analyzer,
                            contig_id,
                        ) {
                            contig_id += 1;
                            contigs.push(c);
                        }
                    }
                }
            }
        }

        // Pass 2: pick up any remaining edges (pure cycles).
        let remaining: Vec<(KmerKey, NodeId)> = self
            .edges
            .iter()
            .filter(|(k, _)| !visited.contains(k))
            .map(|(k, e)| (k.clone(), e.from.clone()))
            .collect();

        for (edge_key, from_id) in remaining {
            if !visited.contains(&edge_key) {
                if let Some(c) = self.extract_contig(
                    &from_id,
                    edge_key,
                    &mut visited,
                    analyzer,
                    contig_id,
                ) {
                    contig_id += 1;
                    contigs.push(c);
                }
            }
        }

        contigs
    }

    /// Walk an unambiguous path starting from `start_node_id` via `first_edge`
    /// until a dead end, a junction, or a revisited edge is reached.
    fn extract_contig(
        &self,
        start_node_id: &NodeId,
        first_edge: KmerKey,
        visited: &mut HashSet<KmerKey>,
        analyzer: &mut Option<Box<dyn JunctionAnalyzer>>,
        id: usize,
    ) -> Option<Contig> {
        let _node_len = self.k - 1;

        let mut sequence = start_node_id.decode();
        let mut total_coverage: u64 = 0;
        let mut edge_count: usize = 0;

        let mut current_edge = first_edge;

        loop {
            // Cycle guard: stop if this edge was already visited in a previous contig.
            if visited.contains(&current_edge) {
                break;
            }

            let (to_node_id, coverage) = {
                let edge = self.edges.get(&current_edge)?;
                (edge.to.clone(), edge.coverage)
            };

            let last_char = INDEX_TO_BASE[current_edge.last_base_idx()];
            sequence.push(last_char);
            total_coverage += u64::from(coverage);
            edge_count += 1;

            visited.insert(current_edge);

            let (out_degree, in_degree, branch_coverage, next_edge) = {
                let next_node = match self.nodes.get(&to_node_id) {
                    Some(n) => n,
                    None => break,
                };

                let branch_cov: [Option<u32>; 4] =
                    std::array::from_fn(|i| {
                        next_node.out_edges[i].as_ref().and_then(|ek| {
                            self.edges.get(ek).map(|e| e.coverage)
                        })
                    });

                let next = if next_node.out_degree == 1 {
                    next_node.single_out_edge().cloned()
                } else {
                    None
                };

                (next_node.out_degree, next_node.in_degree, branch_cov, next)
            };

            match out_degree {
                0 => break,

                1 => {
                    match next_edge {
                        Some(key) => current_edge = key,
                        None => break,
                    }
                }

                _ => {
                    if let Some(ref mut a) = analyzer {
                        let ctx = JunctionContext::new(
                            to_node_id.clone(),
                            edge_count,
                            out_degree,
                            in_degree,
                            branch_coverage,
                            sequence.clone(),
                            id,
                        );
                        a.analyze(&ctx);
                    }
                    break;
                }
            }
        }

        if edge_count == 0 {
            return None;
        }

        let avg_coverage = total_coverage as f64 / edge_count as f64;
        Some(Contig::new(id, sequence, avg_coverage, edge_count))
    }
}
