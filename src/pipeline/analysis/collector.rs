use std::path::Path;
use anyhow::Result;

use super::{
    features::{JunctionContext, JunctionFeatures},
    options::AnalysisOptions,
    traits::JunctionAnalyzer,
};

/// Junction analyzer that accumulates feature records for ML export.
pub struct MlDataCollector {
    opts:    AnalysisOptions,
    records: Vec<JunctionFeatures>,
}

impl MlDataCollector {
    /// Create a new collector with the given feature flags.
    pub fn new(opts: AnalysisOptions) -> Self {
        Self { opts, records: Vec::new() }
    }

    /// Return all collected junction feature records.
    pub fn records(&self) -> &[JunctionFeatures] { &self.records }
}

impl JunctionAnalyzer for MlDataCollector {
    fn analyze(&mut self, ctx: &JunctionContext) -> JunctionFeatures {
        let mut feat = JunctionFeatures {
            depth_in_contig: ctx.depth_in_contig,
            out_degree:      ctx.out_degree,
            in_degree:       ctx.in_degree,
            ..Default::default()
        };

        if self.opts.coverage {
            feat.branch_coverage = Some(ctx.branch_coverage);
            let vals: Vec<u32> = ctx.branch_coverage.iter().flatten().copied().collect();
            if !vals.is_empty() {
                let total: u32 = vals.iter().sum();
                let max = *vals.iter().max().unwrap();
                if total > 0 {
                    feat.coverage_ratio_max = Some(max as f64 / total as f64);
                }
            }
        }

        if self.opts.read_count {
            feat.total_read_count = Some(ctx.branch_coverage.iter().flatten().sum());
        }

        // TODO(stage-2): sequence_context, local_graph

        self.records.push(feat.clone());
        feat
    }

    fn junction_count(&self) -> usize { self.records.len() }

    fn export(&self, path: &Path) -> Result<()> {
        use std::{fs::File, io::{BufWriter, Write}};
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "depth,out_degree,in_degree,cov_A,cov_C,cov_G,cov_T,cov_ratio,reads")?;
        for r in &self.records {
            let cov = r.branch_coverage.unwrap_or([None; 4])
                .map(|v| v.map_or("NA".into(), |n: u32| n.to_string()));
            writeln!(w, "{},{},{},{},{},{},{},{},{}",
                r.depth_in_contig, r.out_degree, r.in_degree,
                cov[0], cov[1], cov[2], cov[3],
                r.coverage_ratio_max.map_or("NA".into(), |v| format!("{:.4}", v)),
                r.total_read_count.map_or("NA".into(), |v| v.to_string()),
            )?;
        }
        Ok(())
    }
}
