//! Configuration for the de Bruijn graph assembler.
//!
//! Built via `AssemblerConfig::builder()`.

use std::path::PathBuf;
use anyhow::{bail, Result};
use crate::pipeline::analysis::options::AnalysisOptions;

/// Runtime configuration for one assembler run.
#[derive(Debug, Clone)]
pub struct AssemblerConfig {
    /// Path to the reads file (FASTA / future formats via factory).
    pub input_path: PathBuf,

    /// k-mer length.  Nodes in the de Bruijn graph are (k-1)-mers.
    pub k: usize,

    /// Directory where output files are written.
    pub output_dir: PathBuf,

    /// When `Some`, the junction analysis module is active.
    /// When `None`, no analysis data is collected (zero overhead).
    pub analysis: Option<AnalysisOptions>,
}

impl AssemblerConfig {
    /// Return a new builder with no fields set.
    pub fn builder() -> AssemblerConfigBuilder {
        AssemblerConfigBuilder::default()
    }

    /// Whether junction analysis is enabled.
    pub fn analysis_enabled(&self) -> bool {
        self.analysis.is_some()
    }
}

/// Builder for `AssemblerConfig`.
#[derive(Debug, Default)]
pub struct AssemblerConfigBuilder {
    input_path: Option<PathBuf>,
    k: Option<usize>,
    output_dir: Option<PathBuf>,
    analysis: Option<AnalysisOptions>,
}

impl AssemblerConfigBuilder {
    /// Set the path to the input reads file.
    pub fn input(mut self, path: impl Into<PathBuf>) -> Self {
        self.input_path = Some(path.into());
        self
    }

    /// Set the k-mer length (default: 31).
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Set the output directory (default: `"output"`).
    pub fn output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = Some(dir.into());
        self
    }

    /// Enable junction analysis with the given options.
    pub fn analysis(mut self, opts: AnalysisOptions) -> Self {
        self.analysis = Some(opts);
        self
    }

    /// Validate all fields and build the final `AssemblerConfig`.
    ///
    /// # Errors
    /// Returns an error if `input_path` was not set or `k < 2`.
    pub fn build(self) -> Result<AssemblerConfig> {
        let input_path = self
            .input_path
            .ok_or_else(|| anyhow::anyhow!("input path is required"))?;

        let k = self.k.unwrap_or(31);
        let output_dir = self.output_dir.unwrap_or_else(|| PathBuf::from("output"));

        if k < 2 {
            bail!("k must be at least 2 (got {})", k);
        }
        // k should be odd to avoid palindrome ambiguity (convention, not hard requirement)
        if k % 2 == 0 {
            eprintln!("Warning: even k={} is unusual; odd values are conventional.", k);
        }

        Ok(AssemblerConfig {
            input_path,
            k,
            output_dir,
            analysis: self.analysis,
        })
    }
}
