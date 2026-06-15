//! Training-data generator: writes a CSV during graph traversal.
//!
//! For every junction the assembler encounters, one row is emitted containing:
//!   - Context (the last `context_len` nucleotides of the contig before the junction)
//!   - Sequences and coverages of all outgoing branches
//!   - Ground truth (the correct branch according to the reference genome)
//!   - Aggregate coverage statistics (cov_ratio, total_reads)

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};
use anyhow::{Context, Result};

use crate::{
    core::kmer::{NodeId, INDEX_TO_BASE},
    core::graph::DeBruijnGraph,
};
use super::ground_truth::{find_ground_truth, GroundTruth};

/// Parameters controlling what context and branch data are captured per junction.
#[derive(Debug, Clone)]
pub struct TrainingDataConfig {
    /// Number of nucleotides of context captured before the junction.
    pub context_len: usize,
    /// Number of nucleotides sampled along each branch after the junction.
    pub branch_len: usize,
    /// Minimum branch coverage — branches below this threshold are skipped as likely noise.
    pub min_branch_coverage: u32,
}

impl Default for TrainingDataConfig {
    fn default() -> Self {
        Self {
            context_len: 100,
            branch_len: 100,
            min_branch_coverage: 2,
        }
    }
}


/// Streams training data to a CSV file during graph traversal.
///
/// Implements `JunctionAnalyzer` so it plugs directly into the assembler pipeline.
pub struct TrainingDataGenerator {
    cfg:       TrainingDataConfig,
    reference: String,
    writer:    BufWriter<File>,
    stats:     GeneratorStats,
    graph_ptr: *const DeBruijnGraph,
}

/// Counters accumulated while generating training data.
#[derive(Debug, Default)]
pub struct GeneratorStats {
    pub total_junctions:   usize,
    pub labeled:           usize,
    pub ambiguous:         usize,
    pub not_found:         usize,
    pub skipped_low_cov:   usize,
}

impl GeneratorStats {
    /// Print a human-readable summary to stdout.
    pub fn print(&self) {
        println!("Training data stats:");
        println!("  Total junctions:   {}", self.total_junctions);
        println!("  Labeled (usable):  {}", self.labeled);
        println!("  Ambiguous:         {}", self.ambiguous);
        println!("  Not in reference:  {}", self.not_found);
        println!("  Low cov (skipped): {}", self.skipped_low_cov);
        if self.total_junctions > 0 {
            println!(
                "  Label rate:        {:.1}%",
                self.labeled as f64 / self.total_junctions as f64 * 100.0
            );
        }
    }
}

impl TrainingDataGenerator {
    /// Create a new generator, writing the CSV header to `output_path`.
    ///
    /// # Arguments
    /// - `cfg` — feature-extraction parameters
    /// - `reference` — full reference genome string (used for ground-truth lookup)
    /// - `output_path` — CSV destination; parent directories are created if needed
    /// - `graph` — the de Bruijn graph (borrowed for the lifetime of traversal)
    pub fn new(
        cfg: TrainingDataConfig,
        reference: String,
        output_path: &Path,
        graph: &DeBruijnGraph,
    ) -> Result<Self> {
        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let file = File::create(output_path)
            .with_context(|| format!("Cannot create training CSV: {}", output_path.display()))?;
        let mut writer = BufWriter::new(file);

        writeln!(
            writer,
            "junction_id,contig_id,depth_in_contig,out_degree,in_degree,\
             node_seq,context_seq,\
             branch_A_seq,branch_A_cov,\
             branch_C_seq,branch_C_cov,\
             branch_G_seq,branch_G_cov,\
             branch_T_seq,branch_T_cov,\
             ground_truth,gt_ambiguous,\
             cov_ratio_max,total_reads"
        )?;

        Ok(Self {
            cfg,
            reference,
            writer,
            stats: GeneratorStats::default(),
            graph_ptr: graph as *const DeBruijnGraph,
        })
    }

    /// Record one junction encountered during traversal and write its CSV row.
    ///
    /// # Arguments
    /// - `graph` — the de Bruijn graph (for branch-sequence extraction)
    /// - `node_id` — junction node identifier
    /// - `context_seq` — contig sequence assembled so far
    /// - `contig_id` — index of the current contig
    /// - `depth` — number of edges traversed in this contig before the junction
    /// - `out_degree` — number of outgoing edges at the junction
    /// - `in_degree` — number of incoming edges at the junction
    pub fn record_junction(
        &mut self,
        graph: &DeBruijnGraph,
        node_id: &NodeId,
        context_seq: &str,
        contig_id: usize,
        depth: usize,
        out_degree: u8,
        in_degree: u8,
    ) -> Result<()> {
        self.stats.total_junctions += 1;

        let node = match graph.nodes.get(node_id) {
            Some(n) => n,
            None => return Ok(()),
        };

        let node_seq = node_id.decode();
        let k = graph.k;

        let mut branch_seq: [Option<String>; 4] = [None, None, None, None];
        let mut branch_cov: [Option<u32>;   4] = [None, None, None, None];
        let mut available_branches: Vec<u8>     = Vec::new();

        for i in 0..4 {
            let edge_key = match &node.out_edges[i] {
                Some(k) => k,
                None => continue,
            };

            let edge = match graph.edges.get(edge_key) {
                Some(e) => e,
                None => continue,
            };

            if edge.coverage < self.cfg.min_branch_coverage {
                self.stats.skipped_low_cov += 1;
                continue;
            }

            let seq = self.extract_branch_sequence(graph, edge_key, k);
            let base_char = INDEX_TO_BASE[i] as u8;

            branch_seq[i] = Some(seq);
            branch_cov[i] = Some(edge.coverage);
            available_branches.push(base_char);
        }

        let active_branches = branch_seq.iter().filter(|s| s.is_some()).count();
        if active_branches < 2 {
            return Ok(());
        }

        let ctx = if context_seq.len() > self.cfg.context_len {
            &context_seq[context_seq.len() - self.cfg.context_len..]
        } else {
            context_seq
        };

        let ground_truth = find_ground_truth(ctx, &self.reference, &available_branches);

        match &ground_truth {
            GroundTruth::Known { .. } => self.stats.labeled += 1,
            GroundTruth::Ambiguous    => self.stats.ambiguous += 1,
            GroundTruth::NotFound     => self.stats.not_found += 1,
        }

        let covs: Vec<u32> = branch_cov.iter().flatten().copied().collect();
        let total_reads: u32 = covs.iter().sum();
        let max_cov = covs.iter().max().copied().unwrap_or(0);
        let cov_ratio_max = if total_reads > 0 {
            max_cov as f64 / total_reads as f64
        } else {
            0.0
        };

        let branch_fields: String = (0..4)
            .flat_map(|i| {
                let seq = branch_seq[i].as_deref().unwrap_or("NA").to_string();
                let cov = branch_cov[i].map(|v| v.to_string()).unwrap_or_else(|| "NA".into());
                [seq, cov]
            })
            .collect::<Vec<_>>()
            .join(",");

        let (gt, gt_amb) = ground_truth.to_csv_value();

        write!(
            self.writer,
            "{},{},{},{},{},{},{},{},{},{},{:.4},{}\n",
            self.stats.total_junctions - 1,
            contig_id, depth, out_degree, in_degree,
            node_seq, ctx,
            branch_fields,
            gt, gt_amb,
            cov_ratio_max, total_reads,
        )?;
        Ok(())
    }

    /// Walk along `first_edge` for up to `branch_len` steps, collecting one base per step.
    ///
    /// Stops early at a dead end or a junction (so the sampled sequence stays unambiguous).
    fn extract_branch_sequence(
        &self,
        graph: &DeBruijnGraph,
        first_edge: &crate::core::kmer::KmerKey,
        _k: usize,
    ) -> String {
        let mut seq = String::with_capacity(self.cfg.branch_len);
        let mut current = first_edge.clone();
        let target_len = self.cfg.branch_len;

        loop {
            if seq.len() >= target_len {
                break;
            }

            let edge = match graph.edges.get(&current) {
                Some(e) => e,
                None => break,
            };

            seq.push(INDEX_TO_BASE[current.last_base_idx()]);

            let next_node = match graph.nodes.get(&edge.to) {
                Some(n) => n,
                None => break,
            };

            // Stop at junctions so the extracted sequence stays unambiguous.
            if next_node.out_degree != 1 {
                break;
            }

            current = match next_node.single_out_edge() {
                Some(k) => k.clone(),
                None => break,
            };
        }

        seq
    }

    /// Flush the writer and return the accumulated statistics.
    pub fn finish(mut self) -> Result<GeneratorStats> {
        self.writer.flush()?;
        Ok(self.stats)
    }
}

impl crate::pipeline::analysis::JunctionAnalyzer for TrainingDataGenerator {
    fn analyze(
        &mut self,
        ctx: &crate::pipeline::analysis::JunctionContext
    ) -> crate::pipeline::analysis::JunctionFeatures {
        // SAFETY: graph_ptr was set from a live reference in `new` and the graph
        // outlives this generator — the caller holds both for the same traversal.
        let graph = unsafe { &*self.graph_ptr };

        let _ = self.record_junction(
            graph,
            &ctx.node_id,
            &ctx.context_seq,
            ctx.contig_id,
            ctx.depth_in_contig,
            ctx.out_degree,
            ctx.in_degree,
        );

        // Return an empty feature object; the generator writes directly to CSV.
        crate::pipeline::analysis::JunctionFeatures::default()
    }

    fn export(&self, _path: &std::path::Path) -> Result<()> {
        // The BufWriter is flushed incrementally; nothing extra to do on export.
        Ok(())
    }

    fn junction_count(&self) -> usize {
        self.stats.total_junctions
    }
}
