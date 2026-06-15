//! `ReadWriter` — the write-side counterpart of `ReadParser`.
//!
//! Implemented by format-specific writers (FASTA, FASTQ, …).

use std::path::Path;
use anyhow::Result;
use super::reader::DnaRead;

/// Strategy interface for writing DNA reads to a file.
pub trait ReadWriter {
    /// Write all `reads` to the file at `path`, creating it if necessary.
    fn write(&self, reads: &[DnaRead], path: &Path) -> Result<()>;

    /// Human-readable name of the output format (e.g. `"FASTA"`).
    fn format_name(&self) -> &'static str;
}
