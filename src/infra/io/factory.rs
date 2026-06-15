//! Factory that selects the correct `ReadParser` implementation based on the
//! file extension.
//!
//! Adding a new format means adding one `match` arm here and providing the
//! concrete reader — nothing else changes.

use std::path::Path;
use anyhow::{bail, Result};

use super::{fasta::FastaReader, reader::ReadParser};

/// Returns a boxed `ReadParser` appropriate for the given file path.
///
/// Supported extensions (case-insensitive):
///   `.fa`, `.fasta`, `.fna`  →  `FastaReader`
///
/// # Errors
/// Returns an error for unrecognised extensions so the caller can surface a
/// helpful message instead of panicking.
pub fn reader_for_path(path: &Path) -> Result<Box<dyn ReadParser>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "fa" | "fasta" | "fna" => Ok(Box::new(FastaReader)),
        other => bail!(
            "Unsupported read file extension '.{}'. \
             Supported: .fa, .fasta, .fna",
            other
        ),
    }
}
