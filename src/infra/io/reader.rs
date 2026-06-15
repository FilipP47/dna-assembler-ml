//! Core data type for a single DNA read and the `ReadParser` trait that all
//! format-specific readers must implement.
//!
//! New formats (FASTQ, plain-text, …) are added by implementing `ReadParser`
//! without touching any other module — Open/Closed Principle.

use std::path::Path;
use anyhow::Result;

/// A single DNA read as returned by any `ReadParser` implementation.
#[derive(Debug, Clone)]
pub struct DnaRead {
    /// Original header line from the file (without the leading `>` or `@`).
    pub header: String,
    /// Raw nucleotide sequence (only A/C/G/T characters, upper-case).
    pub sequence: String,
}

impl DnaRead {
    /// Construct a read from a header and sequence.
    pub fn new(header: impl Into<String>, sequence: impl Into<String>) -> Self {
        Self {
            header: header.into(),
            sequence: sequence.into(),
        }
    }

    /// Length of this read in bases.
    #[inline]
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Returns `true` if the sequence is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }
}

/// Strategy interface for reading DNA reads from a file.
///
/// Implement this trait to add support for a new file format.
/// The assembler pipeline depends only on this trait, never on a concrete reader.
pub trait ReadParser {
    /// Parse all reads from the file at `path`.
    ///
    /// # Errors
    /// Returns an error for IO failures or malformed records.
    fn parse(&self, path: &Path) -> Result<Vec<DnaRead>>;

    /// Human-readable name of the format this parser handles (e.g. `"FASTA"`).
    fn format_name(&self) -> &'static str;
}
