//! Configuration for the read simulator.
//!
//! Built via `SimulatorConfig::builder()` — all fields have sensible defaults
//! so callers only need to set what they care about.

use std::path::PathBuf;
use anyhow::{bail, Result};

/// Runtime configuration for one simulator run.
#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    /// Path to the reference genome FASTA file.
    pub reference_path: PathBuf,

    /// Desired length of each simulated read (in bases).
    pub read_length: usize,

    /// Target sequencing depth (X-fold coverage).
    /// `num_reads = (genome_length × coverage) / read_length`
    pub coverage: usize,

    /// PRNG seed — fixes the random start positions for reproducibility.
    pub seed: u64,

    /// Where to write the simulated reads (FASTA format).
    pub output_path: PathBuf,
}

impl SimulatorConfig {
    /// Start building a new `SimulatorConfig`.
    pub fn builder() -> SimulatorConfigBuilder {
        SimulatorConfigBuilder::default()
    }
}

/// Builder for `SimulatorConfig`.
#[derive(Debug, Default)]
pub struct SimulatorConfigBuilder {
    reference_path: Option<PathBuf>,
    read_length: Option<usize>,
    coverage: Option<usize>,
    seed: Option<u64>,
    output_path: Option<PathBuf>,
}

impl SimulatorConfigBuilder {
    /// Set the path to the reference genome FASTA file.
    pub fn reference(mut self, path: impl Into<PathBuf>) -> Self {
        self.reference_path = Some(path.into());
        self
    }

    /// Set the length of each simulated read (default: 150 bp).
    pub fn read_length(mut self, len: usize) -> Self {
        self.read_length = Some(len);
        self
    }

    /// Set the target sequencing coverage (default: 30×).
    pub fn coverage(mut self, cov: usize) -> Self {
        self.coverage = Some(cov);
        self
    }

    /// Set the PRNG seed for reproducibility (default: 42).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the output path for simulated reads (default: `"reads.fasta"`).
    pub fn output(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_path = Some(path.into());
        self
    }

    /// Validate and construct the final `SimulatorConfig`.
    ///
    /// # Errors
    /// Returns an error if `reference_path` was not set, `read_length < 2`, or `coverage == 0`.
    pub fn build(self) -> Result<SimulatorConfig> {
        let reference_path = self
            .reference_path
            .ok_or_else(|| anyhow::anyhow!("reference path is required"))?;

        let read_length = self.read_length.unwrap_or(150);
        let coverage = self.coverage.unwrap_or(30);
        let seed = self.seed.unwrap_or(42);
        let output_path = self.output_path.unwrap_or_else(|| PathBuf::from("reads.fasta"));

        if read_length < 2 {
            bail!("read_length must be at least 2 (got {})", read_length);
        }
        if coverage == 0 {
            bail!("coverage must be > 0");
        }

        Ok(SimulatorConfig {
            reference_path,
            read_length,
            coverage,
            seed,
            output_path,
        })
    }
}
